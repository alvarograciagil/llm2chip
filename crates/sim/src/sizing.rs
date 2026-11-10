//! Chooses the die for a model: the one with the least logic whose own microprogram,
//! followed to the last position of the context, gives the tokens per second asked for.
//!
//! The cycles are those of the circuit generated for each candidate. The logic is counted
//! on small dies built from the same generator and grown to the candidate's size: a unit's
//! share grows with its lanes, each level of the network that joins the units costs every
//! unit once more, and each cell costs the same wherever it is.

use crate::lower::{Layout, ENTRY_LOGITS, ENTRY_TOKEN};
use crate::Weights;
use compiler::Program;
use netlist::{die_design, DieConfig};
use rayon::prelude::*;
use serde::Serialize;

/// One die the search tried.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct Candidate {
    pub cluster_bits: u32,
    pub lane_bits: u32,
    pub cell_bits: u32,
    pub cycles: u64,
    pub tokens_per_second: f64,
    /// Logic in NAND2 equivalents, memories left out.
    pub gates: f64,
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
}

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
        Costs { fixed: (f4 + f8) / 2.0, unit: u4 - 4.0 * lane, lane, level: l4 - 4.0 * level_lane, level_lane, cell }
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
    pub clock_hz: f64,
    pub context: u32,
    pub lane_bits: std::ops::RangeInclusive<u32>,
    pub cluster_bits: std::ops::RangeInclusive<u32>,
    pub cell_bits: std::ops::RangeInclusive<u32>,
}

/// Cycles of one token at the last position of the context, logits included.
pub fn cycles(program: &Program, weights: &Weights, die: DieConfig, context: u32) -> Result<u64, String> {
    let layout = Layout::for_image(program, weights, die, context as usize)?;
    let code = layout.microprogram(program)?;
    Ok(code.cycles(&die, ENTRY_TOKEN, 1, context - 1) + code.cycles(&die, ENTRY_LOGITS, 1, context - 1))
}

/// Cycles from the first token of a prompt of `prompt` tokens to the logits of the last:
/// each token at its own position, which only the attention's loops depend on. Positions
/// are taken every `stride` and the ones between them counted along the line joining them.
pub fn prompt_cycles(program: &Program, weights: &Weights, die: DieConfig, context: u32, prompt: u32, stride: u32) -> Result<u64, String> {
    let layout = Layout::for_image(program, weights, die, context as usize)?;
    let code = layout.microprogram(program)?;
    let at = |position: u32| code.cycles(&die, ENTRY_TOKEN, 1, position) as f64;
    let (mut total, mut from) = (0.0, 0);
    while from < prompt {
        let to = (from + stride.max(1)).min(prompt);
        let (first, last) = (at(from), at(to - 1));
        total += (first + last) / 2.0 * (to - from) as f64;
        from = to;
    }
    Ok(total.round() as u64 + code.cycles(&die, ENTRY_LOGITS, 1, prompt - 1))
}

/// The die a candidate stands for.
pub fn die_of(base: DieConfig, context: u32, (cluster_bits, lane_bits, cell_bits): (u32, u32, u32)) -> DieConfig {
    DieConfig { cells: 1 << cell_bits, features: base.features.with_lane_bits(lane_bits).with_cluster_bits(cluster_bits).with_key_shift(context.next_power_of_two().trailing_zeros()), ..base }
}

/// For each width of unit and each number of units, the fewest cells that reach the target,
/// best first. Those that cannot reach it with the most cells allowed are left out.
pub fn search(program: &Program, weights: &Weights, base: DieConfig, costs: &Costs, target: &Target) -> Vec<Candidate> {
    let evaluate = |shape: (u32, u32, u32)| -> Option<Candidate> {
        let die = die_of(base, target.context, shape);
        let cycles = cycles(program, weights, die, target.context).ok()?;
        let (cluster_bits, lane_bits, cell_bits) = shape;
        Some(Candidate { cluster_bits, lane_bits, cell_bits, cycles, tokens_per_second: target.clock_hz / cycles as f64, gates: costs.gates(cluster_bits, lane_bits, cell_bits) })
    };
    let shapes: Vec<(u32, u32)> = target.lane_bits.clone().flat_map(|l| target.cluster_bits.clone().map(move |u| (u, l))).collect();
    let mut found: Vec<Candidate> = shapes
        .par_iter()
        .filter_map(|&(cluster_bits, lane_bits)| {
            // Each unit needs a cell for each lane; more cells only shorten the products.
            let lowest = (*target.cell_bits.start()).max(cluster_bits + lane_bits);
            let (mut low, mut high) = (lowest, *target.cell_bits.end());
            let mut best = evaluate((cluster_bits, lane_bits, high)).filter(|c| c.tokens_per_second >= target.tokens_per_second)?;
            while low < high {
                let middle = (low + high) / 2;
                match evaluate((cluster_bits, lane_bits, middle)) {
                    Some(c) if c.tokens_per_second >= target.tokens_per_second => {
                        best = c;
                        high = middle;
                    }
                    _ => low = middle + 1,
                }
            }
            Some(best)
        })
        .collect();
    found.sort_by(|a, b| a.gates.total_cmp(&b.gates));
    found
}
