//! Dies beside the language model's: listening and speaking (`voice`) and drawing
//! (`drawing`). The language model's dies come from the estimator; these hold models of
//! their own, each die at the size measured for its stages, and the floorplan draws them in
//! a row below the language model's.

use crate::tech::node;
use serde::{Deserialize, Serialize};

/// What a die is for. Text is the language model; a die may hold several parts.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Part {
    Text,
    Vision,
    Listen,
    Speak,
    Draw,
}

/// A kind of die beside the language model's: the stages on it and its shape.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SideDie {
    /// The stages, by their planners' names (`speech to text`, `speech writer`,
    /// `sound decoder`, `image drawer`).
    pub stages: Vec<String>,
    /// Dies of this kind.
    pub count: u32,
    pub units: u32,
    pub lanes: u32,
    pub cells: u64,
    /// Area of one die by part: logic, cell memory, vector memory and microcode, mm².
    pub parts_mm2: [f64; 4],
    pub clock_mhz: f64,
}

/// A die beside the language model's as the estimate counts it.
#[derive(Clone, Debug, Serialize)]
pub struct CountedDie {
    pub die: SideDie,
    pub parts: Vec<Part>,
    /// Taken from a measured plan of other models, not planned for this project's.
    pub reference: bool,
}

impl CountedDie {
    pub fn mm2(&self) -> f64 {
        self.die.parts_mm2.iter().sum()
    }
}

pub(crate) fn die(stages: &[&str], units: u32, cells: u64, parts_mm2: [f64; 4], clock_mhz: f64) -> SideDie {
    SideDie { stages: stages.iter().map(|s| s.to_string()).collect(), count: 1, units, lanes: 64, cells, parts_mm2, clock_mhz }
}

/// Watts a die beside the language model's leaks: its logic, and its memories at the
/// density of small macros. Their planners size them in N6.
pub fn leakage_w(d: &CountedDie) -> f64 {
    let n6 = node("n6").expect("the N6 process");
    let [logic, cells, vectors, microcode] = d.die.parts_mm2;
    let bits = (cells + vectors + microcode) * n6.sram_density(64.0) * 1e6;
    d.die.count as f64 * (logic * n6.logic_leak_w_mm2 + bits * n6.sram_leak_pw_bit * 1e-12)
}
