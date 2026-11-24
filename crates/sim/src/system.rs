//! Systems of stages: several programs, each on dies of its own, one after another. A
//! speech recogniser, a language model and a speech synthesiser make one; so would a small
//! model drafting tokens for a large one, or an image encoder on dies of its own.
//!
//! What goes from one stage to the next is text, as bytes: each chip already reads its
//! input and writes its output as UTF-8 bytes through its host port, with its own
//! tokenizer, so stages with different vocabularies join without translating token ids.
//! Joining two chips is joining one's output port to the other's input port, or the host
//! passing the bytes on, which is what the simulation does.
//!
//! A system's silicon is the sum of its stages'. Its latency, from the end of the input to
//! the first output of the last stage, is the sum along the chain of how long each stage
//! takes to give the next what it needs to start: the whole utterance for a recogniser,
//! the first sentence for a synthesiser.

use crate::sizing::Candidate;
use serde::{Deserialize, Serialize};

/// A program on dies of its own, and what it costs.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Stage {
    pub name: String,
    pub dies: u32,
    pub area_mm2: f64,
    pub clock_hz: f64,
    /// Cycles from a complete input to the first output: a prompt's prefill, or a clip's
    /// encoding and the first token.
    pub first_cycles: u64,
    /// Cycles for each further output.
    pub output_cycles: u64,
}

impl Stage {
    /// A language model's stage from its sizing: the prompt's prefill to its first token,
    /// then a token every `cycles`.
    pub fn from_candidate(name: &str, candidate: &Candidate) -> Self {
        Self {
            name: name.into(),
            dies: candidate.chips,
            area_mm2: candidate.area_mm2,
            clock_hz: candidate.clock_hz,
            first_cycles: (candidate.prompt_seconds * candidate.clock_hz).round() as u64,
            output_cycles: candidate.cycles,
        }
    }

    /// Seconds until it has given `outputs` outputs, from a complete input.
    pub fn seconds_to(&self, outputs: u32) -> f64 {
        (self.first_cycles + outputs.saturating_sub(1) as u64 * self.output_cycles) as f64 / self.clock_hz
    }

    /// Outputs a second once it is going.
    pub fn outputs_per_second(&self) -> f64 {
        self.clock_hz / self.output_cycles.max(1) as f64
    }
}

/// Stages in a chain, each fed by the one before.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct System {
    pub stages: Vec<Stage>,
}

impl System {
    pub fn new(stages: Vec<Stage>) -> Self {
        Self { stages }
    }

    pub fn dies(&self) -> u32 {
        self.stages.iter().map(|s| s.dies).sum()
    }

    pub fn area_mm2(&self) -> f64 {
        self.stages.iter().map(|s| s.area_mm2).sum()
    }

    /// Seconds from the end of the input to the last stage's first output, each stage
    /// starting once the one before has given it `handoff[i]` outputs. `handoff` has one
    /// entry per stage but the last.
    pub fn first_output_seconds(&self, handoff: &[u32]) -> Result<f64, String> {
        if handoff.len() + 1 != self.stages.len() {
            return Err(format!("{} stages need {} handoffs", self.stages.len(), self.stages.len().saturating_sub(1)));
        }
        Ok(self.stages.iter().zip(handoff.iter().copied().chain([1])).map(|(stage, outputs)| stage.seconds_to(outputs)).sum())
    }

    /// Whether every stage keeps up with what the next one consumes: `rates[i]` outputs a
    /// second of stage `i` are needed for the chain not to wait.
    pub fn keeps_up(&self, rates: &[f64]) -> bool {
        self.stages.iter().zip(rates).all(|(stage, rate)| stage.outputs_per_second() >= *rate)
    }
}

/// What a stage costs on a die of one shape: what its silicon goes to (logic, the cells'
/// memories, the vector memories and the microcode, in mm²), how many such dies it takes,
/// and its timing there.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Fit {
    /// `(log2 units, log2 lanes, log2 cells)`, and how many dies of it the stage takes.
    pub shape: (u32, u32, u32),
    pub dies: u32,
    pub parts_mm2: [f64; 4],
    pub clock_hz: f64,
    pub first_cycles: u64,
    pub output_cycles: u64,
    /// Outputs of the stage before it needs to start (ignored for the first).
    pub needs: u32,
    /// What else sets this way of running it apart, such as the windows a decoder takes.
    pub variant: String,
}

/// A stage as the planner sees it: what it costs on each shape and way of running it
/// tried, and how many outputs a second it must keep up with.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StagePlan {
    pub name: String,
    pub fits: Vec<Fit>,
    /// Outputs a second the stage must give once it is going.
    pub rate: f64,
}

/// What a plan must meet.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Limits {
    /// The largest die the process exposes, mm².
    pub reticle_mm2: f64,
    /// The longest wait from the end of the input to the last stage's first output.
    pub first_output_seconds: f64,
    /// The most silicon in all, mm².
    pub area_mm2: f64,
}

/// Stages on one die, at one shape.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Die {
    pub stages: Vec<usize>,
    pub shape: (u32, u32, u32),
    /// Dies of this kind: a stage that takes several cannot share them.
    pub count: u32,
    pub parts_mm2: [f64; 4],
    pub clock_hz: f64,
}

impl Die {
    pub fn mm2(&self) -> f64 {
        self.parts_mm2.iter().sum()
    }
}

/// A system's dies and how it does.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Plan {
    pub dies: Vec<Die>,
    pub area_mm2: f64,
    pub first_output_seconds: f64,
    /// Each stage's timing on its die, and the way it runs, in the stages' order.
    pub stages: Vec<Stage>,
    pub variants: Vec<String>,
}

/// A grouping that was weighed, and why it was or was not chosen.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Weighed {
    pub groups: Vec<Vec<usize>>,
    pub best: Option<Plan>,
    pub reason: Option<String>,
}

/// Every way of putting `n` items in groups.
fn partitions(n: usize) -> Vec<Vec<Vec<usize>>> {
    let mut all: Vec<Vec<Vec<usize>>> = vec![Vec::new()];
    for item in 0..n {
        let mut next = Vec::new();
        for groups in &all {
            for g in 0..groups.len() {
                let mut with = groups.clone();
                with[g].push(item);
                next.push(with);
            }
            let mut alone = groups.clone();
            alone.push(vec![item]);
            next.push(alone);
        }
        all = next;
    }
    all
}

/// A die of a group of stages, and each stage's timing on it and the way it runs there.
type DieOption = (Die, Vec<(Stage, Fit)>);

/// Stages `group` on one die of `shape`, each in every way it was measured there: the
/// logic once, the cells and microcode of all of them, the vector memory of the one that
/// needs most (they never run at once), the slowest clock. A stage that takes several dies
/// shares them with no other.
fn merged(stages: &[StagePlan], group: &[usize], shape: (u32, u32, u32)) -> Vec<DieOption> {
    let mut combos: Vec<Vec<&Fit>> = vec![Vec::new()];
    for s in group {
        let here: Vec<&Fit> = stages[*s].fits.iter().filter(|f| f.shape == shape).collect();
        combos = combos.into_iter().flat_map(|c| here.iter().map(move |f| c.iter().copied().chain([*f]).collect::<Vec<_>>())).collect();
    }
    combos
        .into_iter()
        .filter(|fits| group.len() == 1 || fits.iter().all(|f| f.dies == 1))
        .map(|fits| {
            let max = |i: usize| fits.iter().map(|f| f.parts_mm2[i]).fold(0.0, f64::max);
            let sum = |i: usize| fits.iter().map(|f| f.parts_mm2[i]).sum::<f64>();
            let clock = fits.iter().map(|f| f.clock_hz).fold(f64::INFINITY, f64::min);
            let die = Die { stages: group.to_vec(), shape, count: fits[0].dies, parts_mm2: [max(0), sum(1), max(2), sum(3)], clock_hz: clock };
            let timing = group.iter().zip(&fits).map(|(s, f)| (Stage { name: stages[*s].name.clone(), dies: f.dies, area_mm2: 0.0, clock_hz: clock, first_cycles: f.first_cycles, output_cycles: f.output_cycles }, (*f).clone())).collect();
            (die, timing)
        })
        .collect()
}

/// The least silicon that meets `limits`: every grouping of the stages on dies, stages that
/// run at once (`overlaps`) never on the same die, each die at the best of the shapes its
/// stages were measured on. Gives the plan and every grouping weighed, with why the others
/// were not chosen.
pub fn plan(stages: &[StagePlan], overlaps: &[(usize, usize)], limits: Limits) -> (Option<Plan>, Vec<Weighed>) {
    let together = |a: usize, b: usize| overlaps.iter().any(|(x, y)| (*x, *y) == (a, b) || (*x, *y) == (b, a));
    let mut weighed = Vec::new();
    let mut best: Option<Plan> = None;
    let better = |a: &Plan, b: &Plan| (a.area_mm2, a.dies.iter().map(|d| d.count).sum::<u32>(), a.first_output_seconds).partial_cmp(&(b.area_mm2, b.dies.iter().map(|d| d.count).sum::<u32>(), b.first_output_seconds)) == Some(std::cmp::Ordering::Less);
    for groups in partitions(stages.len()) {
        if let Some((a, b)) = groups.iter().find_map(|g| g.iter().flat_map(|a| g.iter().map(move |b| (*a, *b))).find(|(a, b)| a < b && together(*a, *b))) {
            weighed.push(Weighed { groups, best: None, reason: Some(format!("{} and {} run at once", stages[a].name, stages[b].name)) });
            continue;
        }
        // Each group's shapes: those every one of its stages was measured on.
        let options: Vec<Vec<DieOption>> = groups
            .iter()
            .map(|g| {
                let mut shapes: Vec<(u32, u32, u32)> = stages[g[0]].fits.iter().map(|f| f.shape).collect();
                shapes.sort_unstable();
                shapes.dedup();
                shapes.retain(|shape| g.iter().all(|s| stages[*s].fits.iter().any(|f| f.shape == *shape)));
                shapes.into_iter().flat_map(|shape| merged(stages, g, shape)).collect()
            })
            .collect();
        if options.iter().any(Vec::is_empty) {
            weighed.push(Weighed { groups, best: None, reason: Some("no shape was measured for all the stages of a die".into()) });
            continue;
        }
        let mut found: Option<Plan> = None;
        let mut why = String::new();
        let mut choice = vec![0usize; options.len()];
        loop {
            let dies: Vec<&DieOption> = choice.iter().zip(&options).map(|(c, o)| &o[*c]).collect();
            let mut timing: Vec<Option<(Stage, Fit)>> = vec![None; stages.len()];
            for (die, stage_timing) in &dies {
                for (s, t) in die.stages.iter().zip(stage_timing) {
                    timing[*s] = Some(t.clone());
                }
            }
            let (timing, fits): (Vec<Stage>, Vec<Fit>) = timing.into_iter().map(Option::unwrap).unzip();
            let system = System::new(timing.clone());
            // Each stage starts once the one before has given it what it needs.
            let handoff: Vec<u32> = fits[1..].iter().map(|f| f.needs).collect();
            let first = system.first_output_seconds(&handoff).unwrap_or(f64::INFINITY);
            let area: f64 = dies.iter().map(|(d, _)| d.mm2() * d.count as f64).sum();
            let too_big = dies.iter().find(|(d, _)| d.mm2() > limits.reticle_mm2);
            let slow = stages.iter().zip(&timing).find(|(s, t)| t.outputs_per_second() < s.rate);
            if let Some((d, _)) = too_big {
                why = format!("a die of {:.0} mm² is larger than the reticle", d.mm2());
            } else if let Some((s, t)) = slow {
                why = format!("{} gives {:.0} a second and must give {:.0}", s.name, t.outputs_per_second(), s.rate);
            } else if first > limits.first_output_seconds {
                why = format!("the first output comes after {:.0} ms", first * 1e3);
            } else if area > limits.area_mm2 {
                why = format!("{area:.0} mm² is more than allowed");
            } else {
                let plan = Plan { dies: dies.iter().map(|(d, _)| d.clone()).collect(), area_mm2: area, first_output_seconds: first, stages: timing, variants: fits.iter().map(|f| f.variant.clone()).collect() };
                if found.as_ref().is_none_or(|f| better(&plan, f)) {
                    found = Some(plan);
                }
            }
            // The next combination of shapes.
            let mut i = 0;
            while i < choice.len() {
                choice[i] += 1;
                if choice[i] < options[i].len() {
                    break;
                }
                choice[i] = 0;
                i += 1;
            }
            if i == choice.len() {
                break;
            }
        }
        if let Some(plan) = &found {
            if best.as_ref().is_none_or(|b| better(plan, b)) {
                best = Some(plan.clone());
            }
        }
        weighed.push(Weighed { groups, reason: found.is_none().then_some(why), best: found });
    }
    (best, weighed)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stage(name: &str, dies: u32, first: u64, each: u64) -> Stage {
        Stage { name: name.into(), dies, area_mm2: dies as f64 * 100.0, clock_hz: 1e9, first_cycles: first, output_cycles: each }
    }

    fn fit(shape: (u32, u32, u32), parts: [f64; 4], first: u64, each: u64) -> Fit {
        Fit { shape, dies: 1, parts_mm2: parts, clock_hz: 1e9, first_cycles: first, output_cycles: each, needs: 5, variant: String::new() }
    }

    fn limits() -> Limits {
        Limits { reticle_mm2: 858.0, first_output_seconds: 1.0, area_mm2: f64::INFINITY }
    }

    /// Two small stages that never run at once, measured on the same shape.
    fn small_pair() -> Vec<StagePlan> {
        vec![
            StagePlan { name: "listen".into(), fits: vec![fit((0, 6, 11), [10.0, 20.0, 30.0, 0.5], 1_000_000, 10_000)], rate: 0.0 },
            StagePlan { name: "speak".into(), fits: vec![fit((0, 6, 11), [10.0, 60.0, 40.0, 0.5], 2_000_000, 10_000)], rate: 50.0 },
        ]
    }

    #[test]
    fn two_small_stages_that_never_run_at_once_share_a_die() {
        let (chosen, weighed) = plan(&small_pair(), &[], limits());
        let chosen = chosen.unwrap();
        // One die: the logic once, both cells and microcodes, the larger vector memory.
        assert_eq!(chosen.dies.len(), 1);
        assert!((chosen.area_mm2 - (10.0 + 80.0 + 40.0 + 1.0)).abs() < 1e-9);
        assert_eq!(weighed.len(), 2);
    }

    #[test]
    fn stages_that_run_at_once_never_share_a_die() {
        let (chosen, weighed) = plan(&small_pair(), &[(0, 1)], limits());
        assert_eq!(chosen.unwrap().dies.len(), 2);
        assert!(weighed.iter().any(|w| w.reason.as_deref().is_some_and(|r| r.contains("at once"))));
    }

    #[test]
    fn stages_stay_apart_when_together_they_would_not_fit_the_reticle() {
        let mut stages = small_pair();
        stages[0].fits[0].parts_mm2[1] = 500.0;
        stages[1].fits[0].parts_mm2[1] = 500.0;
        let (chosen, weighed) = plan(&stages, &[], limits());
        assert_eq!(chosen.unwrap().dies.len(), 2);
        assert!(weighed.iter().any(|w| w.reason.as_deref().is_some_and(|r| r.contains("reticle"))));
    }

    #[test]
    fn a_wider_shape_is_chosen_only_when_the_narrow_one_is_too_slow() {
        let mut stages = small_pair();
        // On a wider die the second stage is three times as fast, and larger.
        stages[1].fits.push(fit((4, 6, 11), [40.0, 60.0, 200.0, 0.5], 700_000, 3_000));
        let (chosen, _) = plan(&stages, &[(0, 1)], limits());
        assert_eq!(chosen.unwrap().dies[1].shape, (0, 6, 11));
        stages[1].rate = 200_000.0;
        let (chosen, _) = plan(&stages, &[(0, 1)], limits());
        assert_eq!(chosen.unwrap().dies[1].shape, (4, 6, 11));
    }

    #[test]
    fn a_chain_adds_its_dies_and_waits_for_each_handoff() {
        let system = System::new(vec![stage("listen", 1, 2_000_000, 50_000), stage("think", 12, 30_000_000, 300_000), stage("speak", 1, 1_000_000, 100_000)]);
        assert_eq!(system.dies(), 14);
        assert_eq!(system.area_mm2(), 1400.0);
        // The recogniser's 10 tokens, the model's first 12, the synthesiser's first code.
        let seconds = system.first_output_seconds(&[10, 12]).unwrap();
        let expected = (2_000_000 + 9 * 50_000) as f64 / 1e9 + (30_000_000 + 11 * 300_000) as f64 / 1e9 + 1_000_000_f64 / 1e9;
        assert!((seconds - expected).abs() < 1e-12);
        assert!(system.first_output_seconds(&[10]).is_err());
        // The synthesiser must write 150 codes a second; at 100,000 cycles it writes 10,000.
        assert!(system.keeps_up(&[20.0, 30.0, 150.0]));
        assert!(!system.keeps_up(&[20.0, 30.0, 20_000.0]));
    }
}
