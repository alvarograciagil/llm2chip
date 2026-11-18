//! Chooses the die for a model: the one with the least logic whose own microprogram,
//! followed to the last position of the context, gives the tokens per second asked for.
//!
//! The cycles are those of the circuit generated for each candidate. The logic is counted
//! on small dies built from the same generator and grown to the candidate's size: a unit's
//! share grows with its lanes, each level of the network that joins the units costs every
//! unit once more, and each cell costs the same wherever it is.

pub use crate::chip::Sharing;
use crate::lower::{Layout, ENTRY_LOGITS, ENTRY_TOKEN};
use crate::Weights;
use compiler::Program;
use netlist::{die_design, word_bits, DieConfig, DieFeatures, Microprogram};
use rayon::prelude::*;
use serde::Serialize;

/// One die the search tried.
#[derive(Clone, Copy, Debug, Serialize, serde::Deserialize, PartialEq)]
pub struct Candidate {
    /// Chips that share the layers, each one this die.
    pub chips: u32,
    pub cluster_bits: u32,
    pub lane_bits: u32,
    pub cell_bits: u32,
    pub cycles: u64,
    /// The die's clock: the one asked for, or its fastest if that is lower.
    pub clock_hz: f64,
    pub tokens_per_second: f64,
    /// Seconds to the first token after a prompt of `Target::prompt` tokens.
    pub prompt_seconds: f64,
    /// Logic of all the chips in NAND2 equivalents, memories left out.
    pub gates: f64,
    /// Silicon of all the chips: the logic, the cells' memories and the vector memories.
    pub area_mm2: f64,
    /// Silicon of each chip, which a reticle limits.
    pub die_mm2: f64,
    /// What each chip's silicon goes to: logic, the cells' memories, the vector units'
    /// memories and the microcode.
    pub parts_mm2: [f64; 4],
    /// How the chips share the model.
    pub sharing: Sharing,
}

/// Gate equivalents of the parts that change from one die to another.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct Costs {
    /// The sequencer and everything there is one of.
    pub fixed: f64,
    /// A vector unit, with no lanes and with each lane.
    pub unit: f64,
    pub lane: f64,
    /// One level of the network between units, for each unit, with no lanes and each lane.
    pub level: f64,
    pub level_lane: f64,
    pub cell: f64,
    /// The slowest path between registers, in full-adder delays: with one unit of four
    /// lanes, and how much each doubling of the lanes and of the units adds.
    pub path: f64,
    pub path_lane_bit: f64,
    /// With four units, where the network between them is already there, and what each
    /// further doubling of the units adds.
    pub path_four_units: f64,
    pub path_unit_bit: f64,
}

/// Delays of an AND, OR, XOR, NOT and MUX in full-adder delays.
const GATE_DELAYS: [f64; 5] = [0.25, 0.25, 0.5, 0.12, 0.4];

impl Costs {
    /// Measured on dies of one, two and four units with four and eight lanes.
    pub fn measure(base: DieConfig) -> Self {
        let gates = |cluster_bits: u32, lane_bits: u32, cells_per_lane: u32| {
            let lanes = 1u32 << lane_bits;
            let c = DieConfig { cells: (cells_per_lane * lanes) << cluster_bits, features: base.features.with_lane_bits(lane_bits).with_cluster_bits(cluster_bits), ..base };
            die_design(c).gate_equivalents()
        };
        // One unit with twice the cells gives a cell; then units and levels, at two widths.
        let cell = (gates(0, 2, 2) - gates(0, 2, 1)) / 4.0;
        let at = |lane_bits: u32| {
            let lanes = (1u32 << lane_bits) as f64;
            let (one, two, four) = (gates(0, lane_bits, 1) - lanes * cell, gates(1, lane_bits, 1) - 2.0 * lanes * cell, gates(2, lane_bits, 1) - 4.0 * lanes * cell);
            // one = f + u, two = f + 2u + 2l, four = f + 4u + 8l.
            let level = ((four - two) - 2.0 * (two - one)) / 2.0;
            let unit = (two - one) - 2.0 * level;
            (one - unit, unit, level)
        };
        let ((f4, u4, l4), (f8, u8, l8)) = (at(2), at(3));
        let lane = (u8 - u4) / 4.0;
        let level_lane = (l8 - l4) / 4.0;
        // The clock: grows with the logarithm of the lanes and of the units, so three
        // small dies give its line.
        // The simulator holds every memory, so they are made small: only the index adders
        // would be a little deeper with the real widths.
        let path = |cluster_bits: u32, lane_bits: u32| {
            let small = DieConfig { reg_bits: 8, index_bits: 12, microcode_bits: 12, cell: netlist::CellConfig { addr_bits: 8, ..base.cell }, ..base };
            let c = DieConfig { cells: 1 << (cluster_bits + lane_bits), features: base.features.with_lane_bits(lane_bits).with_cluster_bits(cluster_bits), ..small };
            netlist::Simulator::new(&die_design(c)).map_or(f64::INFINITY, |sim| sim.critical_path(GATE_DELAYS))
        };
        let (start, wider, four, many) = (path(0, 2), path(0, 4), path(2, 2), path(5, 2));
        Costs { fixed: (f4 + f8) / 2.0, unit: u4 - 4.0 * lane, lane, level: l4 - 4.0 * level_lane, level_lane, cell, path: start, path_lane_bit: (wider - start) / 2.0, path_four_units: four, path_unit_bit: ((many - four) / 3.0).max(0.0) }
    }

    /// The fastest clock of a die, given the delay of a full adder in its process.
    pub fn clock_hz(&self, cluster_bits: u32, lane_bits: u32, full_adder_ns: f64) -> f64 {
        let units = if cluster_bits >= 2 {
            self.path_four_units + self.path_unit_bit * (cluster_bits - 2) as f64
        } else {
            self.path + (self.path_four_units - self.path) * cluster_bits as f64 / 2.0
        };
        let path = units + self.path_lane_bit * (lane_bits as f64 - 2.0).max(0.0);
        1e9 / (path * full_adder_ns)
    }

    pub fn gates(&self, cluster_bits: u32, lane_bits: u32, cell_bits: u32) -> f64 {
        let (units, lanes) = ((1u64 << cluster_bits) as f64, (1u64 << lane_bits) as f64);
        let per_unit = self.unit + lanes * self.lane + cluster_bits as f64 * (self.level + lanes * self.level_lane);
        self.fixed + units * per_unit + (1u64 << cell_bits) as f64 * self.cell
    }
}

/// What the search is asked for and what it may try.
pub struct Target {
    pub tokens_per_second: f64,
    /// The longest wait for the first token, after a prompt of `prompt` tokens.
    pub prompt_seconds: f64,
    pub prompt: u32,
    pub chips: Vec<u32>,
    /// The highest clock wanted; each die runs at the lower of this and its own fastest.
    pub clock_hz: f64,
    pub process: Process,
    pub context: u32,
    pub lane_bits: std::ops::RangeInclusive<u32>,
    pub cluster_bits: std::ops::RangeInclusive<u32>,
    pub cell_bits: std::ops::RangeInclusive<u32>,
    /// The ways of sharing the model to try, each with every number of chips it suits.
    pub sharing: Vec<Sharing>,
}

/// Each chip's microprogram when the model is shared among `chips` of them.
pub fn chip_programs(program: &Program, weights: &Weights, die: DieConfig, context: u32, chips: u32, sharing: Sharing) -> Result<Vec<Microprogram>, String> {
    Ok(built(program, weights, die, context, chips, sharing)?.programs)
}

/// What the chips of a set run and the memory each needs.
pub struct Built {
    pub programs: Vec<Microprogram>,
    /// Bits of each cell's memory on the chip that needs most: every chip is the same silicon.
    pub cell_bits: u64,
    /// Words of vector memory of each unit on the chip that needs most.
    pub vector_words: u64,
    /// What those words hold, on that chip.
    pub vector_kinds: Vec<(&'static str, u64)>,
    /// Bits of microcode memory on the chip that needs most.
    pub microcode_bits: u64,
}

/// Bits of microcode memory a set of chips needs: on each, the fewer of its words one
/// after another and a dictionary of its distinct words with an index into it. Every chip
/// is the same silicon, so the set needs the most any of them does.
pub fn microcode_memory(die: DieConfig, codes: &[Microprogram]) -> u64 {
    let bits_for = |count: u64| (u64::BITS - count.max(2).saturating_sub(1).leading_zeros()).max(1);
    let features = codes.iter().map(DieFeatures::of).reduce(DieFeatures::union).unwrap_or_default();
    let features = features.with_lane_bits(die.features.lane_bits).with_cluster_bits(die.features.cluster_bits).with_key_shift(die.features.key_shift).with_factor_steps(die.features.factor_steps).with_wide_link(die.features.wide_link);
    let longest = codes.iter().map(|code| code.words.len() as u64).max().unwrap_or(1);
    let config = DieConfig { features, microcode_bits: bits_for(longest), ..die };
    let width = word_bits(&config) as u64;
    codes
        .iter()
        .map(|code| {
            let words = code.words.len() as u64;
            let plain = words * width;
            if config.microcode_bits > netlist::die::ENTRY_INDEX_SHIFT {
                return plain;
            }
            let distinct = netlist::die::code_dictionary(code, &config).0.len() as u64;
            plain.min(distinct * width + words * 2 * bits_for(distinct) as u64)
        })
        .max()
        .unwrap_or(0)
}

/// Each chip's microprogram and the memory it needs. By layers, they are cut among the
/// chips so the one with most memory has as little as can be; by rows, every chip holds
/// an equal slice of each matrix.
pub fn built(program: &Program, weights: &Weights, die: DieConfig, context: u32, chips: u32, sharing: Sharing) -> Result<Built, String> {
    let share = crate::chip::ShareOut::new(program, weights, chips as usize, context as usize, die.cell.activation_bits, sharing)?;
    let (mut cell_bits, mut vector, mut kinds) = (0, 0, Vec::new());
    let codes = (0..chips as usize)
        .map(|chip| {
            // The embedding table stays in the image, as on a large model's chip: the
            // firmware brings each token's row, so the vector memory holds one.
            // A model with more than one table keeps them all inside.
            let mut layout = match Layout::for_image_table_outside(program, weights, die, context as usize) {
                Ok(layout) => layout,
                Err(_) => Layout::for_image(program, weights, die, context as usize)?,
            };
            let code = share.microprogram(&mut layout, program, chip)?;
            let (depth, groups) = layout.cell_depths();
            // The depth counts words, each as wide as the cell's weights.
            cell_bits = cell_bits.max(depth * die.cell.weight_bits as u64 + groups * 2 * die.scale_bits as u64);
            // The words its own routines reach, as the compact vector memory holds them.
            let words = layout.vector_words_used();
            if words > vector {
                (vector, kinds) = (words, layout.vector_words_by_kind());
            }
            Ok(code)
        })
        .collect::<Result<Vec<_>, String>>()?;
    let microcode_bits = microcode_memory(die, &codes);
    Ok(Built { programs: codes, cell_bits, vector_words: vector, vector_kinds: kinds, microcode_bits })
}

/// TSMC N6 as `tech/n6.json` estimates it: an AND2 is 1.33 NAND2 equivalents, and wiring
/// adds a fifth. The reticle is 26 by 33 mm.
pub const N6: Process = Process { full_adder_ns: 0.04, gate_um2: 0.0875 / 1.33 * 1.2, sram_cell_um2: 0.027, sram_periphery_k: 5.0, reticle_mm2: 858.0 };

/// The die the searches size: 8-bit weights against 16-bit activations into 36-bit sums,
/// with `factor_steps` cycles for each cell to apply a group's scale and bias. Its cells,
/// units and lanes are what `die_of` sets.
pub fn reference_die(factor_steps: u32) -> DieConfig {
    let cell = netlist::CellConfig { weight_bits: 8, activation_bits: 16, accumulator_bits: 36, mac_stages: 1, adder_stages: 1, addr_bits: 24, coding: Default::default(), exponent_bits: 0, multiplier: netlist::multiplier::MultiplierKind::Booth, fixed_weights: false, ecc: false, skip_zeros: false };
    DieConfig { cells: 1, cell, scale_bits: 21, reg_bits: 16, index_bits: 24, microcode_bits: 24, broadcast_stages: 1, function_bits: 12, function_select_bits: 3, features: DieFeatures::default().with_factor_steps(factor_steps) }
}

/// Silicon of a process, for the area of what is generated.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct Process {
    /// A full adder's delay, which sets each die's fastest clock.
    pub full_adder_ns: f64,
    /// Square micrometres of a NAND2 equivalent of logic, wiring included.
    pub gate_um2: f64,
    /// A memory bit, and how much periphery a memory block of a kilobit adds.
    pub sram_cell_um2: f64,
    pub sram_periphery_k: f64,
    /// The largest die the process can expose at once.
    pub reticle_mm2: f64,
}

impl Process {
    /// Square millimetres of `cells` memories of `bits` each: a small one wastes more on
    /// its periphery.
    pub fn memory_mm2(&self, cells: u64, bits: u64) -> f64 {
        let kbit = (bits as f64 / 1e3).max(0.25);
        let efficiency = 1.0 / (1.0 + self.sram_periphery_k / kbit.sqrt());
        cells as f64 * bits as f64 * self.sram_cell_um2 / efficiency / 1e6
    }
}

/// The speed and the wait of a set of chips, from their own microprograms.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct Timing {
    /// One token at the last position of the context, through every chip, logits included.
    pub token_cycles: u64,
    /// From the first token of a prompt to the logits of its last: the chips work on
    /// consecutive tokens at once, so after the first each token waits for the slowest.
    pub prompt_cycles: u64,
}

pub fn timing(codes: &[Microprogram], die: DieConfig, context: u32, prompt: u32, stride: u32, sharing: Sharing) -> Timing {
    // Groups that share each product: the members work at once, each group waits for its
    // slowest, and the groups follow one another as single chips would.
    let group = match sharing {
        Sharing::Mixed(g) if g > 1 && codes.len() > g as usize && codes.len().is_multiple_of(g as usize) => Some(g as usize),
        Sharing::Mixed(g) if g > 1 && codes.len() > 1 && codes.len() <= g as usize => return timing(codes, die, context, prompt, stride, Sharing::Rows),
        _ => None,
    };
    if let Some(g) = group {
        let stage = |position: u32| -> Vec<u64> { codes.chunks(g).map(|members| members.iter().map(|code| code.cycles(&die, ENTRY_TOKEN, 1, position)).max().unwrap_or(0)).collect() };
        let logits = |position: u32| codes.iter().map(|code| code.cycles(&die, ENTRY_LOGITS, 1, position)).max().unwrap_or(0);
        let token_cycles = stage(context - 1).iter().sum::<u64>() + logits(context - 1);
        let slowest = |position: u32| stage(position).into_iter().max().unwrap_or(0) as f64;
        let (mut total, mut from) = (stage(0).iter().sum::<u64>() as f64, 1);
        while from < prompt {
            let to = (from + stride.max(1)).min(prompt);
            total += (slowest(from) + slowest(to - 1)) / 2.0 * (to - from) as f64;
            from = to;
        }
        return Timing { token_cycles, prompt_cycles: total.round() as u64 + logits(prompt.max(1) - 1) };
    }
    let token = |position: u32| -> Vec<u64> { codes.iter().map(|code| code.cycles(&die, ENTRY_TOKEN, 1, position)).collect() };
    if sharing == Sharing::Rows {
        // The chips all work on the same token at once, waiting for one another at each
        // exchange: a token takes what the slowest does.
        let both = |position: u32| codes.iter().map(|code| code.cycles(&die, ENTRY_TOKEN, 1, position) + code.cycles(&die, ENTRY_LOGITS, 1, position)).max().unwrap_or(0);
        let slowest = |position: u32| token(position).into_iter().max().unwrap_or(0) as f64;
        let (mut total, mut from) = (0.0, 0);
        while from + 1 < prompt {
            let to = (from + stride.max(1)).min(prompt - 1);
            total += (slowest(from) + slowest(to - 1)) / 2.0 * (to - from) as f64;
            from = to;
        }
        return Timing { token_cycles: both(context - 1), prompt_cycles: total.round() as u64 + both(prompt.max(1) - 1) };
    }
    // The chips make the logits together: the last sends the first what it needs.
    let logits = |position: u32| codes.iter().map(|code| code.cycles(&die, ENTRY_LOGITS, 1, position)).max().unwrap_or(0);
    let token_cycles = token(context - 1).iter().sum::<u64>() + logits(context - 1);
    let slowest = |position: u32| token(position).into_iter().max().unwrap_or(0) as f64;
    // Positions every `stride`, the ones between counted along the line joining them.
    let (mut total, mut from) = (token(0).iter().sum::<u64>() as f64, 1);
    while from < prompt {
        let to = (from + stride.max(1)).min(prompt);
        total += (slowest(from) + slowest(to - 1)) / 2.0 * (to - from) as f64;
        from = to;
    }
    Timing { token_cycles, prompt_cycles: total.round() as u64 + logits(prompt.max(1) - 1) }
}

/// The die a candidate stands for. Its link to the next chip carries a block of lanes a
/// cycle, so a boundary between chips costs a few dozen cycles rather than thousands.
pub fn die_of(base: DieConfig, context: u32, (cluster_bits, lane_bits, cell_bits): (u32, u32, u32)) -> DieConfig {
    DieConfig { cells: 1 << cell_bits, features: base.features.with_lane_bits(lane_bits).with_cluster_bits(cluster_bits).with_key_shift(context.next_power_of_two().trailing_zeros()).with_wide_link(true), ..base }
}

/// One set of chips, measured: its own microprograms, its clock and its silicon.
pub fn evaluate(program: &Program, weights: &Weights, base: DieConfig, costs: &Costs, target: &Target, sharing: Sharing, shape: (u32, u32, u32, u32)) -> Option<Candidate> {
    measure(program, weights, base, costs, target, sharing, shape).ok()
}

/// Like `evaluate`, saying why a set of chips cannot be built.
pub fn measure(program: &Program, weights: &Weights, base: DieConfig, costs: &Costs, target: &Target, sharing: Sharing, (chips, cluster_bits, lane_bits, cell_bits): (u32, u32, u32, u32)) -> Result<Candidate, String> {
    let die = die_of(base, target.context, (cluster_bits, lane_bits, cell_bits));
    let Built { programs: codes, cell_bits: bits, vector_words, microcode_bits, .. } = built(program, weights, die, target.context, chips, sharing)?;
    let timing = timing(&codes, die, target.context, target.prompt, 256, sharing);
    let clock = target.clock_hz.min(costs.clock_hz(cluster_bits, lane_bits, target.process.full_adder_ns));
    let cycles = timing.token_cycles;
    let gates = costs.gates(cluster_bits, lane_bits, cell_bits);
    // Every unit holds the words its routines reach, a share in each lane, twice: once for
    // each read port.
    let banks = 1u64 << (cluster_bits + lane_bits);
    let vectors = 2.0 * target.process.memory_mm2(banks, (vector_words * die.cell.activation_bits as u64) >> lane_bits);
    // The sequencer keeps the microprogram in a memory of its own.
    let parts_mm2 = [gates * target.process.gate_um2 / 1e6, target.process.memory_mm2(1 << cell_bits, bits), vectors, target.process.memory_mm2(1, microcode_bits)];
    let die_mm2 = parts_mm2.iter().sum::<f64>();
    Ok(Candidate { chips, cluster_bits, lane_bits, cell_bits, cycles, clock_hz: clock, tokens_per_second: clock / cycles as f64, prompt_seconds: timing.prompt_cycles as f64 / clock, gates: chips as f64 * gates, area_mm2: chips as f64 * die_mm2, die_mm2, parts_mm2, sharing })
}

/// For each number of chips, width of unit and number of units, the fewest cells that
/// reach the target, least silicon first. Those that cannot reach it are left out.
/// Each number of chips with each way of sharing that comes to something different there.
pub fn sharings(target: &Target) -> Vec<(u32, Sharing)> {
    let mut pairs: Vec<(u32, Sharing)> = Vec::new();
    for &chips in &target.chips {
        for sharing in &target.sharing {
            let pair = (chips, sharing.for_chips(chips as usize));
            if !pairs.contains(&pair) {
                pairs.push(pair);
            }
        }
    }
    pairs
}

/// The ways of sharing a list names, `layers`, `rows` or `mixed:N` separated by commas:
/// `mixed:auto` stands for groups of every size that divides one of `chips` and leaves more
/// than one group.
pub fn parse_sharings(text: &str, chips: &[u32]) -> Result<Vec<Sharing>, String> {
    let mut all: Vec<Sharing> = Vec::new();
    for part in text.split(',').map(str::trim).filter(|p| !p.is_empty()) {
        let found: Vec<Sharing> = match part.split_once(':') {
            Some(("mixed", "auto")) => chips.iter().flat_map(|&n| (2..n).filter(move |g| n.is_multiple_of(*g))).map(Sharing::Mixed).collect(),
            Some(("mixed", n)) => vec![Sharing::Mixed(n.parse().map_err(|_| format!("the size of the groups in {part}"))?)],
            None if part == "rows" => vec![Sharing::Rows],
            None if part == "layers" => vec![Sharing::Layers],
            _ => return Err(format!("{part} is not layers, rows or mixed:N")),
        };
        for s in found {
            if !all.contains(&s) {
                all.push(s);
            }
        }
    }
    if all.is_empty() {
        all.push(Sharing::Layers);
    }
    Ok(all)
}

/// How a way of sharing is written: `layers`, `rows` or `mixed:N`.
pub fn sharing_name(sharing: Sharing) -> String {
    match sharing {
        Sharing::Layers => "layers".into(),
        Sharing::Rows => "rows".into(),
        Sharing::Mixed(n) => format!("mixed:{n}"),
    }
}

pub fn search(program: &Program, weights: &Weights, base: DieConfig, costs: &Costs, target: &Target) -> Vec<Candidate> {
    let measure = |sharing: Sharing, shape: (u32, u32, u32, u32)| evaluate(program, weights, base, costs, target, sharing, shape);
    // More cells only make a die faster and larger: the fewest that are fast enough, if
    // they fit in the reticle.
    let fits = |c: &Candidate| c.tokens_per_second >= target.tokens_per_second && c.prompt_seconds <= target.prompt_seconds;
    let shapes: Vec<(u32, Sharing, u32, u32)> = sharings(target).into_iter().flat_map(|(n, s)| target.lane_bits.clone().flat_map(move |l| target.cluster_bits.clone().map(move |u| (n, s, u, l)))).collect();
    let mut found: Vec<Candidate> = shapes
        .par_iter()
        .filter_map(|&(chips, sharing, cluster_bits, lane_bits)| {
            let measure = |shape| measure(sharing, shape);
            // Each unit needs a cell for each lane; more cells only shorten the products.
            let lowest = (*target.cell_bits.start()).max(cluster_bits + lane_bits);
            let (mut low, mut high) = (lowest, *target.cell_bits.end());
            let mut best = measure((chips, cluster_bits, lane_bits, high)).filter(fits)?;
            while low < high {
                let middle = (low + high) / 2;
                match measure((chips, cluster_bits, lane_bits, middle)) {
                    Some(c) if fits(&c) => {
                        best = c;
                        high = middle;
                    }
                    _ => low = middle + 1,
                }
            }
            Some(best).filter(|c| c.die_mm2 <= target.process.reticle_mm2)
        })
        .collect();
    found.sort_by(|a, b| a.area_mm2.total_cmp(&b.area_mm2));
    found
}

/// Every shape in the target's ranges, measured, keeping those no other beats on both
/// speed and logic: from the smallest to the fastest.
pub fn frontier(program: &Program, weights: &Weights, base: DieConfig, costs: &Costs, target: &Target) -> Vec<Candidate> {
    let shapes: Vec<(Sharing, u32, u32, u32, u32)> = sharings(target)
        .into_iter()
        .flat_map(|(n, s)| target.lane_bits.clone().flat_map(move |l| target.cluster_bits.clone().flat_map(move |u| target.cell_bits.clone().filter(move |c| *c >= u + l).map(move |c| (s, n, u, l, c)))))
        .collect();
    let mut all: Vec<Candidate> = shapes
        .par_iter()
        .filter_map(|&(sharing, chips, cluster_bits, lane_bits, cell_bits)| {
            evaluate(program, weights, base, costs, target, sharing, (chips, cluster_bits, lane_bits, cell_bits)).filter(|c| c.die_mm2 <= target.process.reticle_mm2)
        })
        .collect();
    all.sort_by(|a, b| a.area_mm2.total_cmp(&b.area_mm2));
    let mut best = 0.0;
    all.retain(|c| {
        let better = c.tokens_per_second > best;
        best = best.max(c.tokens_per_second);
        better
    });
    all
}

/// The point of the frontier past which each token per second costs much more silicon than
/// those before it: the last whose next step costs at most `ratio` times the area per
/// token per second that the point itself has.
pub fn balanced(frontier: &[Candidate], ratio: f64) -> Option<Candidate> {
    let mut chosen = *frontier.first()?;
    for pair in frontier.windows(2) {
        let (here, next) = (pair[0], pair[1]);
        let marginal = (next.area_mm2 - here.area_mm2) / (next.tokens_per_second - here.tokens_per_second);
        let average = here.area_mm2 / here.tokens_per_second;
        if marginal > ratio * average {
            break;
        }
        chosen = next;
    }
    Some(chosen)
}

/// The point of the frontier that gives the most speed for its silicon: the least area for
/// each token per second.
pub fn efficient(frontier: &[Candidate]) -> Option<Candidate> {
    frontier.iter().copied().min_by(|a, b| (a.area_mm2 / a.tokens_per_second).total_cmp(&(b.area_mm2 / b.tokens_per_second)))
}
