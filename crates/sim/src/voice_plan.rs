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
    /// The speech writer reads its speaker's profile before each sentence.
    #[serde(default)]
    pub speaker: bool,
    /// Sets of dies the language model is measured on when it fits no single die of
    /// `shapes`: `(chips, sharing, shape)`, sharing as the frontier names it (`mixed:4`).
    #[serde(default = "large_sets")]
    pub large: Vec<(u32, String, (u32, u32, u32))>,
}

/// Sets of dies from the frontier for Qwen3.8-27B: 12 dies of 64 × 128 units and 32,768
/// cells in groups of four (the least area there), and the fastest ones, 12 of 128 × 128
/// in groups of four and 16 in groups of eight.
fn large_sets() -> Vec<(u32, String, (u32, u32, u32))> {
    vec![(12, "mixed:4".into(), (6, 7, 15)), (12, "mixed:4".into(), (7, 7, 16)), (16, "mixed:8".into(), (7, 7, 16))]
}

impl Default for VoiceAsk {
    fn default() -> Self {
        Self { seconds: 3.0, words: 12, first_sound_seconds: 0.5, area_mm2: f64::INFINITY, shapes: vec![(0, 6, 11), (0, 6, 13), (4, 6, 11), (4, 6, 13), (4, 6, 16)], chunks: vec![16, 32, 64], speaker: false, large: large_sets() }
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

/// A product's energy in N6, picojoules: an 8-bit weight by an `activation_bits`-bit value
/// in the cell, as the estimator simulates it, and the weight's read from its memory.
fn pj_per_mac(activation_bits: u32) -> f64 {
    let tech = arch::node("n6").expect("N6");
    let cell = netlist::CellConfig { activation_bits, accumulator_bits: 2 * activation_bits + 16, ..reference_die(8).cell };
    arch::mac_energy_pj(tech, cell) + 8.0 * tech.e_sram_fj_bit / 1e3
}

/// 8-bit codes with a scale and a bias per group of `group` columns: 64 is the chip's
/// usual; Moonshine's widths (288) need 32, and DAC's codebooks, 8 values wide, need 8.
/// A checkpoint already quantized keeps its own codes, as the frontier keeps them.
fn weights(dir: &Path, group: u32) -> Result<Weights, String> {
    let weights = Weights::open(dir)?;
    if quantized(dir) {
        return Ok(weights);
    }
    let format = WeightFormat { coding: WeightCoding::Affine, bits: 8, exponent_bits: 0, bias: true, group_size: group, fine_scale: false, keep_source: false };
    Ok(weights.with_format(Some(format)))
}

fn quantized(dir: &Path) -> bool {
    std::fs::read_to_string(dir.join("config.json")).ok().and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok()).is_some_and(|c| c.get("quantization").is_some() || c.get("quantization_config").is_some())
}

/// Whether a model's weights could fit one die within the reticle at all: at four bits
/// each, one weight for each multiplication of a token, in the process's densest memory.
fn could_fit_one_die(dir: &Path) -> Result<bool, String> {
    let program = compiler::compile_model(&std::fs::read_to_string(dir.join("config.json")).map_err(|e| e.to_string())?)?;
    let weights = crate::system::macs(&program.ops, &program.buffers) as f64;
    let tech = arch::node("n6").ok_or("no N6")?;
    Ok(weights * 4.0 < N6.reticle_mm2 * 1e6 / tech.sram_cell_um2)
}

/// A language model on `chips` dies of `shape` shared as `sharing`, after a prompt of
/// `prompt` tokens, with a context long enough for it and its answer.
fn language_model(dir: &Path, costs: &Costs, (chips, sharing): (u32, Sharing), shape: (u32, u32, u32), prompt: u32, needs: u32) -> Result<Fit, String> {
    let context = (prompt + 512).next_power_of_two().max(1024);
    let program = compiler::compile_model(&std::fs::read_to_string(dir.join("config.json")).map_err(|e| e.to_string())?)?;
    let per_token = crate::system::macs(&program.ops, &program.buffers);
    let target = Target { tokens_per_second: 0.0, prompt_seconds: f64::INFINITY, prompt, chips: vec![chips], clock_hz: CLOCK, process: N6, context, lane_bits: shape.1..=shape.1, cluster_bits: shape.0..=shape.0, cell_bits: shape.2..=shape.2, sharing: vec![sharing] };
    let c = measure(&program, &weights(dir, 64)?, reference_die(8), costs, &target, sharing, (chips, shape.0, shape.1, shape.2))?;
    Ok(Fit { shape, dies: c.chips, parts_mm2: c.parts_mm2, clock_hz: c.clock_hz, first_cycles: (c.prompt_seconds * c.clock_hz).round() as u64, output_cycles: c.cycles, needs, variant: String::new(), first_macs: per_token * prompt as u64, output_macs: per_token, pj_per_mac: pj_per_mac(16) })
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
    let frames = listen_config.frames_of(samples);
    let context = match &listen_config.streaming {
        // A streaming recogniser's stores of the front end, and its text.
        Some(streaming) => (4 + 4 * frames).max(frames * streaming.samples_per_frame() * 6 / 16_000 + 2).next_power_of_two(),
        None => lengths.sample_rows.max(listen_config.max_tokens).next_power_of_two(),
    };
    let sound_config = DacConfig::read(&models.sound)?;
    let question_tokens = (ask.seconds * TOKENS_PER_SECOND_SPOKEN).ceil() as u32;
    // The writer's prompt: a sentence's, and the profile's sentence and codes before it.
    let writer_prompt = match crate::voice::speaker_of(&models.speak)?.filter(|_| ask.speaker) {
        Some(speaker) => {
            let vocabulary = crate::speak::SpeechVocabulary::read(&models.speak)?;
            let mut tokenizer = crate::tokenizer::TokenizerTables::from_file(&models.speak.join("tokenizer.json"), 4096)?;
            tokenizer.set_add_special(false);
            vocabulary.prompt_with(&speaker, "A sentence of about twelve words, as the answer's first.", &|t| tokenizer.encode(t))?.len() as u32
        }
        None => 40,
    };
    let mut listen = Vec::new();
    let mut think = Vec::new();
    let think_fits_one = could_fit_one_die(&models.think)?;
    let mut speak = Vec::new();
    let mut sound = Vec::new();
    for &shape in &ask.shapes {
        let name = format!("{} units × {} lanes, {} cells", 1u32 << shape.0, 1u32 << shape.1, 1u64 << shape.2);
        progress(format!("Moonshine on {name}"));
        if listen_config.streaming.is_some() {
            // Its widths, the front end's 80 among them, take groups of 16.
            let chip = crate::moonshine_stream_chip::StreamingChip::from_parts(listen_config.clone(), weights(&models.listen, 16)?, frames, die_of(base, context as u32, shape), Some(context));
            if let Some((chip, code)) = chip.ok().and_then(|chip| chip.routines().ok().map(|(_, code)| (chip, code))) {
                let (parts, clock) = routines_die(chip.parts().1, &code, &costs, &N6, shape, CLOCK);
                let t = chip.timing(frames, &code)?;
                // It listens while the question is spoken: each 20 ms in less than 20 ms.
                if t.step as f64 / clock <= 0.02 {
                    listen.push(Fit { shape, dies: 1, parts_mm2: parts, clock_hz: clock, first_cycles: t.after, output_cycles: t.token, needs: 0, variant: "listening as the question is spoken".into(), first_macs: t.clip_macs, output_macs: t.token_macs, pj_per_mac: pj_per_mac(16) });
                }
            }
        }
        let chip = if listen_config.streaming.is_some() { Err(String::new()) } else { MoonshineChip::from_parts(listen_config.clone(), weights(&models.listen, 32)?, samples, die_of(base, context as u32, shape), Some(context)) };
        if let Ok(chip) = chip {
            let calls: Vec<_> = (0..lengths.sample_rows).map(|r| chip.store_samples_call(r)).chain(chip.encoder_calls()).chain(chip.token_calls(listen_config.start, 0)).collect();
            if let Ok((_, code)) = chip.microprogram(&calls) {
                let (parts, clock) = routines_die(chip.parts().1, &code, &costs, &N6, shape, CLOCK);
                let stage = chip.stage(1, parts.iter().sum(), clock)?;
                let ops = &chip.parts().0.rows().expect("the routines").ops;
                let count = |calls: &[crate::moonshine_chip::Call]| calls.iter().map(|c| crate::system::macs(&ops[c.routine.clone()], &chip.parts().0.buffers)).sum::<u64>();
                let token = chip.token_calls(listen_config.start, 0);
                // The firmware orders every routine: its cycles come on top of the die's.
                let (first, each) = chip.firmware_cycles();
                listen.push(Fit { shape, dies: 1, parts_mm2: parts, clock_hz: clock, first_cycles: stage.first_cycles + first, output_cycles: stage.output_cycles + each, needs: 0, variant: String::new(), first_macs: count(&calls), output_macs: count(&token), pj_per_mac: pj_per_mac(16) });
            }
        }
        if think_fits_one {
            progress(format!("the language model on {name}"));
            if let Ok(fit) = language_model(&models.think, &costs, (1, Sharing::Layers), shape, 40, question_tokens) {
                think.push(fit);
            }
        }
        progress(format!("OuteTTS on {name}"));
        if let Ok(fit) = language_model(&models.speak, &costs, (1, Sharing::Layers), shape, writer_prompt, ask.words) {
            speak.push(fit);
        }
        for &chunk in &ask.chunks {
            let frames = chunk + 2 * sound_config.reach_frames();
            progress(format!("DAC on {name}, windows of {frames} frames"));
            let Ok(chip) = DacChip::from_parts(sound_config.clone(), weights(&models.sound, 8)?, frames, die_of(wide, 32_768, shape), true) else { continue };
            let Ok(code) = chip.microprogram() else { continue };
            let calls = chip.calls(&vec![vec![0; frames]; sound_config.codebooks])?;
            let (parts, clock) = routines_die(chip.parts().1, &code, &wide_costs, &N6, shape, CLOCK);
            // The die's cycles, and the firmware's to order its routines.
            let cycles = chip.cycles(&calls)? + chip.firmware_cycles(chunk * sound_config.hop());
            let needs = (frames as f64 * TOKENS_PER_FRAME).ceil() as u32;
            let window_macs: u64 = calls.iter().map(|c| crate::system::macs(&chip.routines()[c.routine], &chip.parts().0.buffers)).sum();
            sound.push(Fit { shape, dies: 1, parts_mm2: parts, clock_hz: clock, first_cycles: cycles, output_cycles: cycles / chunk as u64, needs, variant: format!("windows of {frames} frames, {chunk} kept"), first_macs: window_macs, output_macs: window_macs / chunk as u64, pj_per_mac: pj_per_mac(20) });
        }
    }
    // A language model that fits no single die within the reticle: on the larger sets.
    if !think.iter().any(|f| f.parts_mm2.iter().sum::<f64>() <= N6.reticle_mm2) {
        for (chips, sharing, shape) in &ask.large {
            progress(format!("the language model on {chips} dies, {sharing}"));
            let sharing = crate::sizing::parse_sharings(sharing, &[*chips])?.into_iter().next().ok_or("no sharing")?;
            match language_model(&models.think, &costs, (*chips, sharing), *shape, 40, question_tokens) {
                Ok(fit) => think.push(fit),
                Err(e) => progress(format!("the language model on {chips} dies: {e}")),
            }
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

/// Changes when the way stages are measured changes, so plans kept from before are not used.
const PLAN_VERSION: u32 = 2;

/// The system planned for `ask`: the plan chosen, if any meets it, every grouping weighed,
/// and the stages as measured.
pub fn plan_voice(models: &VoiceModels, ask: &VoiceAsk, progress: &mut dyn FnMut(String)) -> Result<Planned, String> {
    let stages = measure_stages(models, ask, progress)?;
    let limits = Limits { reticle_mm2: N6.reticle_mm2, first_output_seconds: ask.first_sound_seconds, area_mm2: ask.area_mm2 };
    let (best, weighed) = plan(&stages, &AT_ONCE, limits);
    Ok((best, weighed, stages))
}

/// `plan_voice`, kept in `dir` for the same models, question and engine: measuring every
/// stage on every shape takes a minute or two.
pub fn plan_voice_cached(dir: &Path, models: &VoiceModels, ask: &VoiceAsk, progress: &mut dyn FnMut(String)) -> Result<Planned, String> {
    let key = serde_json::to_string(&(PLAN_VERSION, models, ask)).map_err(|e| e.to_string())?;
    let path = crate::optimize::cache_path(dir, &format!("voice plan {key}"));
    if let Some(planned) = std::fs::read_to_string(&path).ok().and_then(|text| serde_json::from_str::<Planned>(&text).ok()) {
        return Ok(planned);
    }
    let planned = plan_voice(models, ask, progress)?;
    if std::fs::create_dir_all(dir).is_ok() {
        if let Ok(text) = serde_json::to_string(&planned) {
            let _ = std::fs::write(&path, text);
        }
    }
    Ok(planned)
}

/// What a spoken turn costs in energy on a planned system: each stage's work, and every
/// die's leakage while the turn lasts.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TurnEnergy {
    /// Joules each stage spends working, in the stages' order.
    pub stages_j: Vec<f64>,
    /// Joules every die leaks while the turn lasts.
    pub leakage_j: f64,
    /// The turn: the question, the wait and the answer said, seconds.
    pub seconds: f64,
    pub total_j: f64,
}

/// The energy of a turn on `plan`: a question of `ask.seconds`, an answer of `ask.words`
/// tokens, said at about three tokens a second. Leakage is counted on every die for the
/// whole turn, none powered off, with every memory bit leaking as N6's do; the work is the
/// products of weights by values and the weights' reads, as the estimator counts a token.
pub fn turn_energy(plan: &Plan, ask: &VoiceAsk) -> TurnEnergy {
    let tech = arch::node("n6").expect("N6");
    let question_tokens = (ask.seconds * TOKENS_PER_SECOND_SPOKEN).ceil() as u64;
    let answer_seconds = ask.words as f64 / TOKENS_PER_SECOND_SPOKEN;
    let frames = (answer_seconds * FRAMES_PER_SECOND).ceil() as u64;
    let outputs = [question_tokens, ask.words as u64, (frames as f64 * TOKENS_PER_FRAME) as u64, frames];
    let stages_j: Vec<f64> = plan.fits.iter().zip(outputs).map(|(f, n)| (f.first_macs + n.saturating_sub(1) * f.output_macs) as f64 * f.pj_per_mac * 1e-12).collect();
    let seconds = ask.seconds + plan.first_output_seconds + answer_seconds;
    let leak_w: f64 = plan.dies.iter().map(|d| {
        let memory_mm2 = d.parts_mm2[1] + d.parts_mm2[2] + d.parts_mm2[3];
        (memory_mm2 * 1e6 / tech.sram_cell_um2 * tech.sram_leak_pw_bit * 1e-12 + d.parts_mm2[0] * tech.logic_leak_w_mm2) * d.count as f64
    }).sum();
    let leakage_j = leak_w * seconds;
    TurnEnergy { total_j: stages_j.iter().sum::<f64>() + leakage_j, stages_j, leakage_j, seconds }
}
