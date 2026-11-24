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
use serde::Serialize;

/// A program on dies of its own, and what it costs.
#[derive(Clone, Debug, Serialize)]
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
#[derive(Clone, Debug, Default, Serialize)]
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

#[cfg(test)]
mod tests {
    use super::*;

    fn stage(name: &str, dies: u32, first: u64, each: u64) -> Stage {
        Stage { name: name.into(), dies, area_mm2: dies as f64 * 100.0, clock_hz: 1e9, first_cycles: first, output_cycles: each }
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
