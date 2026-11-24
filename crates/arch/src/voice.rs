//! The dies of listening and speaking. The language model's dies come from the estimator;
//! these come from the voice planner (the Listen and Speak views), which measures each stage
//! in gates on several shapes of die and groups the stages that never run at once. Until a
//! project is planned with its own models, reference dies from measured plans stand in.

use crate::side::{die, CountedDie, Part, SideDie};
use serde::{Deserialize, Serialize};

/// The voice planner's names for its stages, and the part each belongs to.
pub const LISTEN_STAGES: [&str; 1] = ["speech to text"];
pub const SPEAK_STAGES: [&str; 2] = ["speech writer", "sound decoder"];

pub fn part_of_stage(stage: &str) -> Option<Part> {
    if LISTEN_STAGES.contains(&stage) {
        Some(Part::Listen)
    } else if SPEAK_STAGES.contains(&stage) {
        Some(Part::Speak)
    } else {
        None
    }
}

/// The voice planner's dies for a project, and the parts the project had when it planned
/// them: they count only while it still has the same ones.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VoicePlanned {
    pub listens: bool,
    pub speaks: bool,
    pub dies: Vec<SideDie>,
}

/// Dies from the plans measured with Moonshine-tiny, Qwen3-0.6B, OuteTTS and DAC
/// (docs/voice.md): the recogniser shares a die with the decoder when the chip does both.
pub fn reference_dies(listens: bool, speaks: bool) -> Vec<SideDie> {
    match (listens, speaks) {
        (true, true) => vec![
            die(&["speech to text", "sound decoder"], 1, 2048, [3.71, 92.3, 27.22, 0.11], 936.0),
            die(&["speech writer"], 16, 8192, [10.9, 168.9, 65.1, 0.1], 1000.0),
        ],
        (true, false) => vec![die(&["speech to text"], 1, 2048, [3.0, 25.9, 16.0, 0.1], 1000.0)],
        (false, true) => vec![
            die(&["speech writer"], 16, 2048, [3.84, 154.65, 65.2, 0.13], 1000.0),
            die(&["sound decoder"], 1, 2048, [3.7, 66.0, 27.2, 0.1], 936.0),
        ],
        (false, false) => Vec::new(),
    }
}

/// The voice dies a project with these parts counts: its own plan when it was made for the
/// same parts and holds only voice stages, the reference dies otherwise.
pub fn counted(listens: bool, speaks: bool, planned: Option<&VoicePlanned>) -> Vec<CountedDie> {
    let own = planned.filter(|p| {
        p.listens == listens
            && p.speaks == speaks
            && !p.dies.is_empty()
            && p.dies.iter().all(|d| d.count > 0 && !d.stages.is_empty() && d.stages.iter().all(|s| part_of_stage(s).is_some()))
    });
    let (dies, reference) = match own {
        Some(p) => (p.dies.clone(), false),
        None => (reference_dies(listens, speaks), true),
    };
    dies.into_iter()
        .map(|die| {
            let mut parts: Vec<Part> = die.stages.iter().filter_map(|s| part_of_stage(s)).collect();
            parts.sort();
            parts.dedup();
            CountedDie { die, parts, reference }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_set_of_parts_has_its_reference_dies() {
        assert!(counted(false, false, None).is_empty());
        let listen = counted(true, false, None);
        assert_eq!(listen.len(), 1);
        assert_eq!(listen[0].parts, [Part::Listen]);
        let speak = counted(false, true, None);
        assert!(speak.iter().all(|d| d.parts == [Part::Speak] && d.reference));
        let both = counted(true, true, None);
        assert_eq!(both[0].parts, [Part::Listen, Part::Speak]);
        let area: f64 = both.iter().map(|d| d.mm2() * d.die.count as f64).sum();
        assert!((area - 368.3).abs() < 1.0, "{area}");
    }

    #[test]
    fn a_plan_counts_only_for_the_parts_it_was_made_for() {
        let plan = VoicePlanned { listens: true, speaks: true, dies: vec![die(&["speech to text", "speech writer", "sound decoder"], 4, 4096, [5.0, 50.0, 20.0, 1.0], 900.0)] };
        let own = counted(true, true, Some(&plan));
        assert!(!own[0].reference && own[0].mm2() == 76.0);
        // The chip stopped speaking: the plan's dies would still hold the speaking stages.
        assert!(counted(true, false, Some(&plan))[0].reference);
        // A plan that puts a stage on the language model's die is not the project's.
        let shared = VoicePlanned { dies: vec![die(&["speech to text", "language model"], 1, 2048, [1.0; 4], 1000.0)], ..plan };
        assert!(counted(true, true, Some(&shared))[0].reference);
    }

    #[test]
    fn voice_dies_leak_a_little() {
        let leak: f64 = counted(true, true, None).iter().map(crate::side::leakage_w).sum();
        assert!(leak > 0.1 && leak < 20.0, "{leak}");
    }
}
