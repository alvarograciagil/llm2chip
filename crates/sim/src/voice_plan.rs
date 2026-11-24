//! A chip that listens, answers and speaks, planned as a system: each of its four stages
//! (Moonshine, a language model, OuteTTS and DAC) measured on a few shapes of die from its
//! own microprogram, with the optimiser's counts in N6, and `system::plan` choosing which
//! share a die and at which shape, for the least silicon that answers in time.
//!
//! Within a spoken turn the recogniser runs alone, before the rest; the language model, the
//! speech writer and the decoder run at once, each taking the one before's first sentence
//! or window while it goes on. So only the recogniser may share a die, with any one of them.

use crate::dac::DacConfig;
use crate::dac_chip::DacChip;
use crate::moonshine::MoonshineConfig;
use crate::moonshine_chip::{Lengths, MoonshineChip};
use crate::sizing::{die_of, measure, reference_die, routines_die, Costs, Sharing, Target, N6};
use crate::system::{plan, Fit, Limits, Plan, StagePlan, Weighed};
use crate::Weights;
use netlist::format::{WeightCoding, WeightFormat};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// The four models' folders.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VoiceModels {
    pub listen: PathBuf,
    pub think: PathBuf,
    pub speak: PathBuf,
    pub sound: PathBuf,
}

/// What the system is planned for.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VoiceAsk {
    /// The longest question, in seconds.
    pub seconds: f64,
    /// Tokens of the answer's first sentence.
    pub words: u32,
    /// The longest wait from the end of the question to the first sound, in seconds.
    pub first_sound_seconds: f64,
    /// The most silicon in all, mm².
    pub area_mm2: f64,
    /// The die shapes tried, `(log2 units, log2 lanes, log2 cells)`.
    pub shapes: Vec<(u32, u32, u32)>,
    /// The frames the decoder may keep from each window.
    pub chunks: Vec<usize>,
}

impl Default for VoiceAsk {
    fn default() -> Self {
        Self { seconds: 3.0, words: 12, first_sound_seconds: 0.5, area_mm2: f64::INFINITY, shapes: vec![(0, 6, 11), (0, 6, 13), (4, 6, 11), (4, 6, 13), (4, 6, 16)], chunks: vec![16, 32, 64] }
    }
}

/// The stages that run at once within a turn: the language model, the speech writer and
/// the decoder, by their places (1, 2 and 3).
pub const AT_ONCE: [(usize, usize); 3] = [(1, 2), (2, 3), (1, 3)];

const CLOCK: f64 = 1e9;
/// Tokens OuteTTS writes for each frame of speech: its two codes and its words' own tokens.
pub const TOKENS_PER_FRAME: f64 = 2.4;
/// Frames of speech a second, and tokens a question's text takes a second of speech.
const FRAMES_PER_SECOND: f64 = 75.0;
const TOKENS_PER_SECOND_SPOKEN: f64 = 3.0;

/// 8-bit codes with a scale and a bias per group of `group` columns: 64 is the chip's
/// usual; Moonshine's widths (288) need 32, and DAC's codebooks, 8 values wide, need 8.
fn weights(dir: &Path, group: u32) -> Result<Weights, String> {
    let format = WeightFormat { coding: WeightCoding::Affine, bits: 8, exponent_bits: 0, bias: true, group_size: group, fine_scale: false, keep_source: false };
    Ok(Weights::open(dir)?.with_format(Some(format)))
}

fn language_model(dir: &Path, costs: &Costs, shape: (u32, u32, u32), prompt: u32, needs: u32) -> Result<Fit, String> {
    let program = compiler::compile_model(&std::fs::read_to_string(dir.join("config.json")).map_err(|e| e.to_string())?)?;
    let target = Target { tokens_per_second: 0.0, prompt_seconds: f64::INFINITY, prompt, chips: vec![1], clock_hz: CLOCK, process: N6, context: 1024, lane_bits: shape.1..=shape.1, cluster_bits: shape.0..=shape.0, cell_bits: shape.2..=shape.2, sharing: vec![Sharing::Layers] };
    let c = measure(&program, &weights(dir, 64)?, reference_die(8), costs, &target, Sharing::Layers, (1, shape.0, shape.1, shape.2))?;
    Ok(Fit { shape, dies: c.chips, parts_mm2: c.parts_mm2, clock_hz: c.clock_hz, first_cycles: (c.prompt_seconds * c.clock_hz).round() as u64, output_cycles: c.cycles, needs, variant: String::new() })
}

/// Each stage measured on every shape of `ask` it can be built on, saying what it is doing.
pub fn measure_stages(models: &VoiceModels, ask: &VoiceAsk, progress: &mut dyn FnMut(String)) -> Result<Vec<StagePlan>, String> {
    let costs = Costs::measure(reference_die(8));
    let base = reference_die(8);
    // The decoder needs 20-bit vectors: see docs/voice.md.
    let wide = netlist::DieConfig { cell: netlist::CellConfig { activation_bits: 20, accumulator_bits: 2 * 20 + 16, ..base.cell }, ..base };
    let wide_costs = Costs::measure(wide);
    let samples = (ask.seconds * 16_000.0) as usize;
    let lengths = Lengths::of(samples)?;
    let listen_config = MoonshineConfig::read(&models.listen)?;
    let context = lengths.sample_rows.max(listen_config.max_tokens).next_power_of_two();
    let sound_config = DacConfig::read(&models.sound)?;
    let question_tokens = (ask.seconds * TOKENS_PER_SECOND_SPOKEN).ceil() as u32;
    let mut listen = Vec::new();
    let mut think = Vec::new();
    let mut speak = Vec::new();
    let mut sound = Vec::new();
    for &shape in &ask.shapes {
        let name = format!("{} units × {} lanes, {} cells", 1u32 << shape.0, 1u32 << shape.1, 1u64 << shape.2);
        progress(format!("Moonshine on {name}"));
        let chip = MoonshineChip::from_parts(listen_config.clone(), weights(&models.listen, 32)?, samples, die_of(base, context as u32, shape), Some(context));
        if let Ok(chip) = chip {
            let calls: Vec<_> = (0..lengths.sample_rows).map(|r| chip.store_samples_call(r)).chain(chip.encoder_calls()).chain(chip.token_calls(listen_config.start, 0)).collect();
            if let Ok((_, code)) = chip.microprogram(&calls) {
                let (parts, clock) = routines_die(chip.parts().1, &code, &costs, &N6, shape, CLOCK);
                let stage = chip.stage(1, parts.iter().sum(), clock)?;
                listen.push(Fit { shape, dies: 1, parts_mm2: parts, clock_hz: clock, first_cycles: stage.first_cycles, output_cycles: stage.output_cycles, needs: 0, variant: String::new() });
            }
        }
        progress(format!("the language model on {name}"));
        if let Ok(fit) = language_model(&models.think, &costs, shape, 40, question_tokens) {
            think.push(fit);
        }
        progress(format!("OuteTTS on {name}"));
        if let Ok(fit) = language_model(&models.speak, &costs, shape, 40, ask.words) {
            speak.push(fit);
        }
        for &chunk in &ask.chunks {
            let frames = chunk + 2 * sound_config.reach_frames();
            progress(format!("DAC on {name}, windows of {frames} frames"));
            let Ok(chip) = DacChip::from_parts(sound_config.clone(), weights(&models.sound, 8)?, frames, die_of(wide, 32_768, shape), true) else { continue };
            let Ok(code) = chip.microprogram() else { continue };
            let calls = chip.calls(&vec![vec![0; frames]; sound_config.codebooks])?;
            let (parts, clock) = routines_die(chip.parts().1, &code, &wide_costs, &N6, shape, CLOCK);
            let cycles = chip.cycles(&calls)?;
            let needs = (frames as f64 * TOKENS_PER_FRAME).ceil() as u32;
            sound.push(Fit { shape, dies: 1, parts_mm2: parts, clock_hz: clock, first_cycles: cycles, output_cycles: cycles / chunk as u64, needs, variant: format!("windows of {frames} frames, {chunk} kept") });
        }
    }
    let stages = vec![
        StagePlan { name: "speech to text".into(), fits: listen, rate: 0.0 },
        StagePlan { name: "language model".into(), fits: think, rate: 0.0 },
        StagePlan { name: "speech writer".into(), fits: speak, rate: FRAMES_PER_SECOND * TOKENS_PER_FRAME },
        StagePlan { name: "sound decoder".into(), fits: sound, rate: FRAMES_PER_SECOND },
    ];
    if let Some(empty) = stages.iter().find(|s| s.fits.is_empty()) {
        return Err(format!("the {} could not be built on any of the shapes tried", empty.name));
    }
    Ok(stages)
}

/// The plan chosen, if any meets what was asked, every grouping weighed, and the stages
/// as measured.
pub type Planned = (Option<Plan>, Vec<Weighed>, Vec<StagePlan>);

/// The system planned for `ask`: the plan chosen, if any meets it, every grouping weighed,
/// and the stages as measured.
pub fn plan_voice(models: &VoiceModels, ask: &VoiceAsk, progress: &mut dyn FnMut(String)) -> Result<Planned, String> {
    let stages = measure_stages(models, ask, progress)?;
    let limits = Limits { reticle_mm2: N6.reticle_mm2, first_output_seconds: ask.first_sound_seconds, area_mm2: ask.area_mm2 };
    let (best, weighed) = plan(&stages, &AT_ONCE, limits);
    Ok((best, weighed, stages))
}
