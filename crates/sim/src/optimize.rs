//! Finds the set of chips that does best what is asked of it for a model: the fastest, the
//! smallest, the one with least silicon for each token per second or the one that spends
//! least energy on each token, among those that meet every limit given.
//!
//! Every design it recommends is measured on its own generated circuit, as `sizing` does:
//! the cycles of its microprograms, its clock and its silicon. The energy and power come
//! from the architecture estimator, scaled to the measured speed, and say so.
//!
//! The space is too large to measure whole (a full frontier of a 27 B model takes most of
//! an hour), so the search goes in steps:
//! 1. the estimator rules out numbers of chips whose dies cannot hold the weights;
//! 2. each way of sharing that is left gets its best number of cells at a middle shape;
//! 3. the most promising ones climb through the number and width of their units.
//!
//! Each measurement is kept on disk, so asking again with another goal or other limits
//! measures only what is new.

use crate::sizing::{measure, sharing_name, sharings, Candidate, Costs, Process, Sharing, Target};
use crate::Weights;
use compiler::Program;
use model_ir::ModelGraph;
use netlist::DieConfig;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Mutex;

/// What to make best.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Goal {
    /// The most tokens per second.
    #[default]
    Speed,
    /// The least silicon in all.
    Area,
    /// The least silicon for each token per second.
    Efficiency,
    /// The least energy for each token.
    Energy,
}

/// How much of the space to measure.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Effort {
    /// The widest lanes only and the three most promising ways of sharing.
    #[default]
    Quick,
    /// Every width of lane and the eight most promising ways of sharing.
    Thorough,
}

/// What every design must meet. A zero leaves a limit out.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Limits {
    pub min_tok_s: f64,
    /// Milliseconds to the first token after a prompt of `Request::prompt` tokens.
    pub max_first_token_ms: f64,
    pub max_chips: u32,
    pub max_area_mm2: f64,
    pub max_power_w: f64,
}

/// What may be tried. The sizes are log2.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Space {
    pub chips: Vec<u32>,
    /// Ways of sharing, as `sizing::parse_sharings` reads them: `mixed:auto` tries every
    /// group size that divides a number of chips.
    pub sharing: String,
    pub cluster_bits: (u32, u32),
    pub lane_bits: (u32, u32),
    pub cell_bits: (u32, u32),
}

impl Default for Space {
    fn default() -> Self {
        Self { chips: vec![1, 2, 4, 6, 8, 12, 16, 24, 32], sharing: "layers,rows,mixed:auto".into(), cluster_bits: (5, 11), lane_bits: (6, 7), cell_bits: (13, 24) }
    }
}

/// What the person asks for.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Request {
    pub goal: Goal,
    pub limits: Limits,
    pub space: Space,
    pub effort: Effort,
    /// Tokens of context every design must hold; the speed is that of the last one.
    pub context: u32,
    /// Tokens of the prompt the first-token limit is for.
    pub prompt: u32,
    /// The highest clock wanted: each die runs at the lower of this and its own fastest.
    pub max_clock_ghz: f64,
}

impl Default for Request {
    fn default() -> Self {
        Self { goal: Goal::Speed, limits: Limits::default(), space: Space::default(), effort: Effort::Quick, context: 8192, prompt: 1024, max_clock_ghz: 5.0 }
    }
}

/// A design that was measured, with what the estimator adds to it.
#[derive(Clone, Debug, Serialize)]
pub struct Design {
    #[serde(flatten)]
    pub candidate: Candidate,
    /// `layers`, `rows` or `mixed:N`.
    pub sharing_name: String,
    /// Estimated from the architecture model at the measured speed, when it can be.
    pub energy_mj: Option<f64>,
    pub power_w: Option<f64>,
    pub area_per_tok_s: f64,
    /// The designer's parameters for this design.
    pub params: Option<serde_json::Value>,
}

/// A design worth seeing next to the one recommended, and why.
#[derive(Clone, Debug, Serialize)]
pub struct Alternative {
    /// `fastest`, `smallest`, `efficient` or `frugal`.
    pub label: &'static str,
    pub design: Design,
}

/// What the search found.
#[derive(Clone, Debug, Serialize)]
pub struct Report {
    pub goal: Goal,
    pub recommended: Option<Design>,
    pub alternatives: Vec<Alternative>,
    /// The measured designs that fit the reticle and that no other beats on both speed and
    /// silicon, from the smallest to the fastest.
    pub frontier: Vec<Design>,
    /// What stopped the recommended design from going further: `reticle`, `max_area`,
    /// `max_power`, `min_tok_s` or `first_token`.
    pub limiting: Vec<&'static str>,
    /// Numbers of chips the estimator ruled out because their dies cannot hold the model.
    pub too_few_chips: Vec<u32>,
    pub measured: usize,
    pub from_cache: usize,
    /// Why a design could not be built, when one could not: with no design found, why.
    pub failure: Option<String>,
}

/// How far the search has gone.
#[derive(Clone, Debug, Serialize)]
pub struct Progress {
    /// `pruning`, `screening`, `climbing` or `done`.
    pub phase: &'static str,
    pub done: usize,
    pub planned: usize,
    pub best: Option<Design>,
}

/// Measurements kept between searches, for one model and one build of the generator.
#[derive(Default, Serialize, Deserialize)]
struct Store {
    measured: HashMap<String, Option<Candidate>>,
}

/// What the search needs of the model.
pub struct Model<'a> {
    pub program: &'a Program,
    pub weights: &'a Weights,
    /// For the energy and for ruling out numbers of chips; without it, neither.
    pub graph: Option<&'a ModelGraph>,
    /// Where measurements are kept: `None` keeps them only while the search runs.
    pub cache: Option<PathBuf>,
}

/// The silicon and the dies to measure on.
pub struct Silicon {
    pub base: DieConfig,
    pub costs: Costs,
    pub process: Process,
}

struct Search<'a> {
    model: &'a Model<'a>,
    silicon: &'a Silicon,
    request: &'a Request,
    target: Target,
    params: Option<arch::ChipParams>,
    store: Mutex<Store>,
    measured: AtomicUsize,
    cached: AtomicUsize,
    planned: AtomicUsize,
    phase: Mutex<&'static str>,
    best: Mutex<Option<Design>>,
    /// Designs measured by an earlier search and not yet used by this one.
    earlier: Mutex<std::collections::HashSet<String>>,
    /// Why the first design that could not be built could not.
    failure: Mutex<Option<String>>,
    cancel: &'a AtomicBool,
    progress: &'a (dyn Fn(Progress) + Sync),
}

/// Searches for the design that does best what `request` asks, telling `progress` how it
/// goes, and stops early when `cancel` is set.
pub fn optimize(model: &Model, silicon: &Silicon, request: &Request, cancel: &AtomicBool, progress: &(dyn Fn(Progress) + Sync)) -> Result<Report, String> {
    let chips: Vec<u32> = request.space.chips.iter().copied().filter(|&n| n >= 1 && (request.limits.max_chips == 0 || n <= request.limits.max_chips)).collect();
    if chips.is_empty() {
        return Err("no number of chips to try within the limit".into());
    }
    let sharing = crate::sizing::parse_sharings(&request.space.sharing, &chips)?;
    let target = Target {
        tokens_per_second: 0.0,
        prompt_seconds: f64::INFINITY,
        prompt: request.prompt.max(1),
        chips,
        clock_hz: request.max_clock_ghz.max(0.01) * 1e9,
        process: silicon.process,
        context: request.context.max(2),
        lane_bits: { let (a, b) = low_high(request.space.lane_bits); a..=b },
        cluster_bits: { let (a, b) = low_high(request.space.cluster_bits); a..=b },
        cell_bits: { let (a, b) = low_high(request.space.cell_bits); a..=b },
        sharing,
    };
    let params = model.graph.and_then(|graph| {
        let start = arch::ChipParams { target_tok_s: 0.0, context_tokens: target.context as u64, ..Default::default() };
        arch::feasibility::fit_to_model(graph, &start).ok()
    });
    let store: Store = model.cache.as_ref().and_then(|path| std::fs::read(path).ok()).and_then(|bytes| serde_json::from_slice(&bytes).ok()).unwrap_or_default();
    let earlier = Mutex::new(store.measured.keys().cloned().collect());
    let search = Search { model, silicon, request, target, params, earlier, store: Mutex::new(store), measured: AtomicUsize::new(0), cached: AtomicUsize::new(0), planned: AtomicUsize::new(0), phase: Mutex::new("pruning"), best: Mutex::new(None), failure: Mutex::new(None), cancel, progress };
    let report = search.run();
    search.save();
    report
}

fn low_high((a, b): (u32, u32)) -> (u32, u32) {
    (a.min(b), a.max(b))
}

impl Search<'_> {
    fn run(&self) -> Result<Report, String> {
        // 1. Numbers of chips whose dies cannot hold the weights, by the estimator.
        let too_few: Vec<u32> = self.target.chips.iter().copied().filter(|&n| self.cannot_hold(n)).collect();
        let families: Vec<(u32, Sharing)> = sharings(&self.target).into_iter().filter(|(n, _)| !too_few.contains(n)).collect();
        if families.is_empty() {
            return Ok(self.report(Vec::new(), too_few));
        }
        // 2. Each family's best number of cells at a middle shape, the most promising first.
        let middle_units = 7u32.clamp(*self.target.cluster_bits.start(), *self.target.cluster_bits.end());
        let widest = *self.target.lane_bits.end();
        let keep = match self.request.effort {
            Effort::Quick => 3,
            Effort::Thorough => 8,
        };
        let lanes: Vec<u32> = match self.request.effort {
            Effort::Quick => vec![widest],
            Effort::Thorough => self.target.lane_bits.clone().rev().collect(),
        };
        let cells_span = (self.target.cell_bits.end() - self.target.cell_bits.start() + 1) as usize;
        let per_shape = 2 * (usize::BITS - cells_span.leading_zeros()) as usize + 1;
        self.planned.store(families.len() * per_shape + keep * lanes.len() * 2 * per_shape, Ordering::Relaxed);
        self.set_phase("screening");
        let mut screened: Vec<((u32, Sharing), Option<Design>)> = families.par_iter().map(|&family| (family, self.best_cells(family, middle_units, widest))).collect();
        screened.sort_by(|a, b| self.order(a.1.as_ref(), b.1.as_ref()));
        // 3. The most promising climb through the number of units, and the width of lane.
        self.set_phase("climbing");
        let leaders: Vec<(u32, Sharing)> = screened.iter().filter(|(_, d)| d.is_some()).take(keep).map(|(f, _)| *f).collect();
        let climbed: Vec<Option<Design>> = leaders.par_iter().flat_map(|&family| lanes.par_iter().map(move |&l| self.climb(family, middle_units, l))).collect();
        let mut found: Vec<Design> = screened.into_iter().filter_map(|(_, d)| d).chain(climbed.into_iter().flatten()).collect();
        found.sort_by(|a, b| self.order(Some(a), Some(b)));
        self.set_phase("done");
        Ok(self.report(found, too_few))
    }

    /// The best design of a family with `units` units of `lanes` lanes, at the number of
    /// cells that suits the goal; `None` if no number of cells meets every limit.
    fn best_cells(&self, (chips, sharing): (u32, Sharing), units: u32, lanes: u32) -> Option<Design> {
        let lowest = (*self.target.cell_bits.start()).max(units + lanes);
        let highest = *self.target.cell_bits.end();
        if lowest > highest || self.stopped() {
            return None;
        }
        let at = |cells: u32| self.measure(chips, sharing, units, lanes, cells);
        // More cells make a die larger and faster: the most that fit the limits above, and
        // the fewest that reach those below.
        let fits = |cells: u32| at(cells).is_some_and(|d| self.under(&d).is_empty());
        if !fits(lowest) {
            return None;
        }
        let (mut low, mut high) = (lowest, highest);
        while low < high {
            let middle = (low + high).div_ceil(2);
            if fits(middle) { low = middle } else { high = middle - 1 }
        }
        let most = low;
        let reaches = |cells: u32| at(cells).is_some_and(|d| self.over(&d).is_empty());
        if !reaches(most) {
            return None;
        }
        let (mut low, mut high) = (lowest, most);
        while low < high {
            let middle = (low + high) / 2;
            if reaches(middle) { high = middle } else { low = middle + 1 }
        }
        let fewest = low;
        let chosen = match self.request.goal {
            Goal::Speed => most,
            Goal::Area => fewest,
            // Silicon for each token per second, or energy for each token, can be least
            // anywhere between: every number of cells in between is tried.
            Goal::Efficiency | Goal::Energy => (fewest..=most).min_by(|a, b| self.order(at(*a).as_ref(), at(*b).as_ref())).unwrap_or(most),
        };
        let design = at(chosen)?;
        self.offer(&design);
        Some(design)
    }

    /// From `start` units, more or fewer while the goal gets better.
    fn climb(&self, family: (u32, Sharing), start: u32, lanes: u32) -> Option<Design> {
        let (lowest, highest) = (*self.target.cluster_bits.start(), *self.target.cluster_bits.end());
        let mut best = (start, self.best_cells(family, start, lanes));
        for step in [1i32, -1] {
            let mut units = start as i32 + step;
            while units >= lowest as i32 && units <= highest as i32 && !self.stopped() {
                let here = self.best_cells(family, units as u32, lanes);
                if self.order(here.as_ref(), best.1.as_ref()) != std::cmp::Ordering::Less {
                    break;
                }
                best = (units as u32, here);
                units += step;
            }
            // Having moved one way, the other way only goes back.
            if best.0 != start {
                break;
            }
        }
        best.1
    }

    /// Measures one design, or takes it from what was measured before.
    fn measure(&self, chips: u32, sharing: Sharing, units: u32, lanes: u32, cells: u32) -> Option<Design> {
        let key = self.key(chips, sharing, units, lanes, cells);
        let known = self.store.lock().unwrap().measured.get(&key).copied();
        let candidate = match known {
            Some(c) => {
                if self.earlier.lock().unwrap().remove(&key) {
                    self.cached.fetch_add(1, Ordering::Relaxed);
                }
                c
            }
            None if self.stopped() => return None,
            None => {
                let s = self.silicon;
                let c = match measure(self.model.program, self.model.weights, s.base, &s.costs, &self.target, sharing, (chips, units, lanes, cells)) {
                    Ok(c) => Some(c),
                    Err(e) => {
                        self.failure.lock().unwrap().get_or_insert(e);
                        None
                    }
                };
                self.store.lock().unwrap().measured.insert(key, c);
                self.measured.fetch_add(1, Ordering::Relaxed);
                self.tell();
                c
            }
        };
        candidate.map(|c| self.design(c))
    }

    /// What a measurement is kept under: the design, and everything else it depends on.
    fn key(&self, chips: u32, sharing: Sharing, units: u32, lanes: u32, cells: u32) -> String {
        format!("{chips}/{}/{units}/{lanes}/{cells}/{}", sharing_name(sharing), self.conditions())
    }

    /// The part of a key that this search shares with every measurement it may use: the
    /// context, the prompt, the clock and the cycles a cell takes to apply its factors.
    fn conditions(&self) -> String {
        format!("{}/{}/{}/{}", self.target.context, self.target.prompt, self.target.clock_hz as u64, self.silicon.base.features.factor_steps)
    }

    /// Whether a kept measurement answers this search: the same conditions, a way of sharing
    /// it tries and a shape within its ranges. Earlier searches may have asked otherwise.
    fn in_scope(&self, key: &str, families: &[(u32, Sharing)]) -> bool {
        let parts: Vec<&str> = key.splitn(6, '/').collect();
        let [chips, sharing, units, lanes, cells, conditions] = parts[..] else { return false };
        let number = |s: &str| s.parse::<u32>().ok();
        let (Some(chips), Some(units), Some(lanes), Some(cells)) = (number(chips), number(units), number(lanes), number(cells)) else { return false };
        conditions == self.conditions()
            && families.iter().any(|(n, s)| *n == chips && sharing_name(*s) == sharing)
            && self.target.cluster_bits.contains(&units)
            && self.target.lane_bits.contains(&lanes)
            && self.target.cell_bits.contains(&cells)
    }

    /// A measured candidate with its energy, power and the designer's parameters.
    fn design(&self, c: Candidate) -> Design {
        let params = self.params.as_ref().map(|base| {
            let (split, group) = match c.sharing {
                Sharing::Layers => (arch::ChipSplit::Layers, base.tensor_group),
                Sharing::Rows => (arch::ChipSplit::Tensor, c.chips),
                Sharing::Mixed(g) => (arch::ChipSplit::Mixed, g),
            };
            arch::ChipParams {
                dies: c.chips,
                chip_split: split,
                tensor_group: group,
                macs_per_die: 1u64 << c.cell_bits,
                vector_units_per_die: 1u64 << c.cluster_bits,
                vector_lanes_per_die: 1u64 << (c.cluster_bits + c.lane_bits),
                freq_ghz: c.clock_hz / 1e9,
                context_tokens: self.target.context as u64,
                target_tok_s: 0.0,
                ..base.clone()
            }
        });
        // The estimator's energy for each token, and its leakage, at the measured speed.
        let energy = self.model.graph.zip(params.as_ref()).and_then(|(graph, p)| arch::estimate(graph, p).ok()).and_then(|r| {
            let (energy, power, rate) = (r.metric("energy_token_mj")?, r.metric("power_w")?, r.metric("tok_s_base")?);
            let leakage = (power - energy * 1e-3 * rate).max(0.0);
            let measured = c.tokens_per_second.max(1e-9);
            Some((energy + leakage / measured * 1e3, energy * 1e-3 * measured + leakage))
        });
        Design {
            candidate: c,
            sharing_name: sharing_name(c.sharing),
            energy_mj: energy.map(|e| e.0),
            power_w: energy.map(|e| e.1),
            area_per_tok_s: c.area_mm2 / c.tokens_per_second.max(1e-9),
            params: params.and_then(|p| serde_json::to_value(p).ok()),
        }
    }

    /// The limits above that a design breaks: what more cells would only make worse.
    fn under(&self, d: &Design) -> Vec<&'static str> {
        let l = &self.request.limits;
        let mut broken = Vec::new();
        if d.candidate.die_mm2 > self.target.process.reticle_mm2 {
            broken.push("reticle");
        }
        if l.max_area_mm2 > 0.0 && d.candidate.area_mm2 > l.max_area_mm2 {
            broken.push("max_area");
        }
        if l.max_power_w > 0.0 && d.power_w.is_some_and(|w| w > l.max_power_w) {
            broken.push("max_power");
        }
        broken
    }

    /// The limits below that a design misses: what more cells would help.
    fn over(&self, d: &Design) -> Vec<&'static str> {
        let l = &self.request.limits;
        let mut missed = Vec::new();
        if d.candidate.tokens_per_second < l.min_tok_s {
            missed.push("min_tok_s");
        }
        if l.max_first_token_ms > 0.0 && d.candidate.prompt_seconds * 1e3 > l.max_first_token_ms {
            missed.push("first_token");
        }
        missed
    }

    /// What the goal ranks: the lower the better.
    fn score(&self, d: &Design) -> f64 {
        match self.request.goal {
            Goal::Speed => -d.candidate.tokens_per_second,
            Goal::Area => d.candidate.area_mm2,
            Goal::Efficiency => d.area_per_tok_s,
            Goal::Energy => d.energy_mj.unwrap_or(d.area_per_tok_s * 1e3),
        }
    }

    /// Designs that meet every limit first, then by the goal; nothing last.
    fn order(&self, a: Option<&Design>, b: Option<&Design>) -> std::cmp::Ordering {
        let key = |d: Option<&Design>| d.map(|d| (!(self.under(d).is_empty() && self.over(d).is_empty()), self.score(d)));
        match (key(a), key(b)) {
            (Some(x), Some(y)) => x.0.cmp(&y.0).then(x.1.total_cmp(&y.1)),
            (Some(_), None) => std::cmp::Ordering::Less,
            (None, Some(_)) => std::cmp::Ordering::Greater,
            (None, None) => std::cmp::Ordering::Equal,
        }
    }

    /// Keeps `d` as the best so far if it is.
    fn offer(&self, d: &Design) {
        let mut best = self.best.lock().unwrap();
        if self.under(d).is_empty() && self.over(d).is_empty() && self.order(Some(d), best.as_ref()) == std::cmp::Ordering::Less {
            *best = Some(d.clone());
            drop(best);
            self.tell();
        }
    }

    /// Whether `chips` dies of the reticle cannot hold the model's weights at all, by the
    /// estimator with the fewest cells allowed. Without a model graph, none is ruled out.
    fn cannot_hold(&self, chips: u32) -> bool {
        let (Some(graph), Some(base)) = (self.model.graph, self.params.as_ref()) else { return false };
        let p = arch::ChipParams { dies: chips, macs_per_die: 1u64 << *self.target.cell_bits.start(), target_tok_s: 0.0, ..base.clone() };
        // The estimator only advises here: a margin keeps it from ruling out a die the
        // circuit would fit.
        arch::estimate(graph, &p).ok().and_then(|r| r.metric("die_area_mm2")).is_some_and(|mm2| mm2 > 1.25 * self.target.process.reticle_mm2)
    }

    fn report(&self, found: Vec<Design>, too_few: Vec<u32>) -> Report {
        // Everything measured for this search's numbers of chips, the steps on the way
        // included: the best of them by the goal is the one recommended.
        let families = sharings(&self.target);
        let mut everything: Vec<Design> = {
            let store = self.store.lock().unwrap();
            store.measured.iter().filter(|(key, _)| self.in_scope(key, &families)).filter_map(|(_, c)| *c).map(|c| self.design(c)).collect()
        };
        everything.extend(found);
        everything.sort_by(|a, b| self.order(Some(a), Some(b)));
        let ok = |d: &&Design| self.under(d).is_empty() && self.over(d).is_empty();
        let recommended = everything.iter().find(ok).cloned();
        let best_by = |key: &dyn Fn(&Design) -> f64| everything.iter().filter(ok).min_by(|a, b| key(a).total_cmp(&key(b))).cloned();
        let mut alternatives: Vec<Alternative> = Vec::new();
        for (label, design) in [
            ("fastest", best_by(&|d| -d.candidate.tokens_per_second)),
            ("smallest", best_by(&|d| d.candidate.area_mm2)),
            ("efficient", best_by(&|d| d.area_per_tok_s)),
            ("frugal", best_by(&|d| d.energy_mj.unwrap_or(f64::INFINITY))),
        ] {
            let Some(design) = design else { continue };
            let same = |other: &Design| other.candidate == design.candidate;
            if recommended.as_ref().is_some_and(same) || alternatives.iter().any(|a| same(&a.design)) {
                continue;
            }
            alternatives.push(Alternative { label, design });
        }
        // Every measured design that fits the reticle, to draw the frontier with.
        let mut all: Vec<Design> = everything.iter().filter(|d| d.candidate.die_mm2 <= self.target.process.reticle_mm2).cloned().collect();
        all.dedup_by(|a, b| a.candidate == b.candidate);
        all.sort_by(|a, b| a.candidate.area_mm2.total_cmp(&b.candidate.area_mm2));
        let mut fastest = 0.0;
        all.retain(|d| {
            let better = d.candidate.tokens_per_second > fastest;
            fastest = d.candidate.tokens_per_second.max(fastest);
            better
        });
        // What kept the recommended design from more cells: the limits the next one breaks.
        let limiting = recommended.as_ref().map_or_else(Vec::new, |d| {
            let c = d.candidate;
            match self.request.goal {
                Goal::Speed if c.cell_bits < *self.target.cell_bits.end() => self.measure(c.chips, c.sharing, c.cluster_bits, c.lane_bits, c.cell_bits + 1).map_or_else(Vec::new, |next| self.under(&next)),
                Goal::Area if c.cell_bits > *self.target.cell_bits.start() => self.measure(c.chips, c.sharing, c.cluster_bits, c.lane_bits, c.cell_bits - 1).map_or_else(Vec::new, |fewer| self.over(&fewer)),
                _ => Vec::new(),
            }
        });
        Report { goal: self.request.goal, recommended, alternatives, frontier: all, limiting, too_few_chips: too_few, measured: self.measured.load(Ordering::Relaxed), from_cache: self.cached.load(Ordering::Relaxed), failure: self.failure.lock().unwrap().clone() }
    }

    fn stopped(&self) -> bool {
        self.cancel.load(Ordering::Relaxed)
    }

    fn set_phase(&self, phase: &'static str) {
        *self.phase.lock().unwrap() = phase;
        self.tell();
    }

    fn tell(&self) {
        let done = self.measured.load(Ordering::Relaxed);
        let planned = self.planned.load(Ordering::Relaxed).max(done);
        (self.progress)(Progress { phase: *self.phase.lock().unwrap(), done, planned, best: self.best.lock().unwrap().clone() });
    }

    /// Keeps what was measured for the next search.
    fn save(&self) {
        let Some(path) = &self.model.cache else { return };
        if let Some(dir) = path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        if let Ok(bytes) = serde_json::to_vec(&*self.store.lock().unwrap()) {
            let _ = std::fs::write(path, bytes);
        }
    }
}

/// Where the measurements of a model are kept: under `dir`, named after the model's
/// configuration and this build of the generator, whose circuits they measured.
pub fn cache_path(dir: &std::path::Path, config: &str) -> PathBuf {
    let mut hash = crc32fast::Hasher::new();
    hash.update(config.as_bytes());
    if let Ok(exe) = std::env::current_exe().and_then(std::fs::read) {
        hash.update(&exe);
    }
    dir.join(format!("sizing-{:08x}.json", hash.finalize()))
}
