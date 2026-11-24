//! A chip that listens, speaks or both, planned as a system: each stage the project has (a
//! recogniser, the language model, a speech writer and a sound decoder) measured on a few
//! shapes of die from its own microprogram, with the optimiser's counts in N6, and
//! `system::plan` choosing which share a die and at which shape, for the least silicon that
//! answers in time.
//!
//! Listening and speaking are separate: a chip that only listens is the recogniser and the
//! language model, and answers in text; one that only speaks is the language model, the
//! writer and the decoder, and takes its question as text. Within a turn the recogniser runs
//! alone, before the rest; the language model, the writer and the decoder run at once, each
//! taking the one before's first sentence or window while it goes on.

#[cfg(feature = "listen")]
use crate::listen::moonshine::MoonshineConfig;
#[cfg(feature = "listen")]
use crate::listen::moonshine_chip::{Lengths, MoonshineChip};
#[cfg(feature = "speak")]
use crate::speak::dac::DacConfig;
#[cfg(feature = "speak")]
use crate::speak::dac_chip::DacChip;
use crate::sizing::{die_of, measure, reference_die, routines_die, Costs, Sharing, Target, N6};
use crate::system::{plan, Fit, Limits, Plan, StagePlan, Weighed};
use crate::Weights;
use netlist::format::{WeightCoding, WeightFormat};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// The models' folders: the language model always, the recogniser when the chip listens,
/// the speech writer and the sound decoder when it speaks.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VoiceModels {
    #[serde(default)]
    pub listen: Option<PathBuf>,
    pub think: PathBuf,
    #[serde(default)]
    pub speak: Option<PathBuf>,
    #[serde(default)]
    pub sound: Option<PathBuf>,
}

impl VoiceModels {
    /// The stages these models make, in their order in a turn.
    pub fn roles(&self) -> Result<Vec<Role>, String> {
        if self.speak.is_some() != self.sound.is_some() {
            return Err("speaking needs both the speech writer and the sound decoder".into());
        }
        let mut roles = Vec::new();
        if self.listen.is_some() {
            roles.push(Role::Listen);
        }
        roles.push(Role::Think);
        if self.speak.is_some() {
            roles.extend([Role::Speak, Role::Sound]);
        }
        Ok(roles)
    }
}

/// A stage of a turn.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Role {
    Listen,
    Think,
    Speak,
    Sound,
}

impl Role {
    pub fn name(self) -> &'static str {
        match self {
            Role::Listen => "speech to text",
            Role::Think => "language model",
            Role::Speak => "speech writer",
            Role::Sound => "sound decoder",
        }
    }
}

/// The stages that run at once within a turn, by their places in `roles`: the language
/// model, the writer and the decoder, each with the others. The recogniser runs alone.
pub fn at_once(roles: &[Role]) -> Vec<(usize, usize)> {
    let together: Vec<usize> = roles.iter().enumerate().filter(|(_, r)| **r != Role::Listen).map(|(i, _)| i).collect();
    together.iter().enumerate().flat_map(|(k, a)| together[k + 1..].iter().map(move |b| (*a, *b))).collect()
}

/// What the system is planned for.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VoiceAsk {
    /// The longest question, in seconds of speech (or the text of one that long).
    pub seconds: f64,
    /// Tokens of the answer's first sentence.
    pub words: u32,
    /// The longest wait from the end of the question to the first output: the first sound
    /// when the chip speaks, the answer's first token when it only listens. Seconds.
    #[serde(alias = "first_sound_seconds")]
    pub first_output_seconds: f64,
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
    /// The recogniser never shares the language model's die: a project's language model is
    /// on dies of its own, the estimator's, so its voice dies hold only voice stages.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub text_apart: bool,
    /// Tokens of the fixed system prompt: the chip reads them when a conversation starts,
    /// while it waits for the first question, so they take context but no time before the
    /// first output.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub system_tokens: u32,
}

fn is_zero(n: &u32) -> bool {
    *n == 0
}

/// Sets of dies from the frontier for Qwen3.8-27B: 12 dies of 64 × 128 units and 32,768
/// cells in groups of four (the least area there), and the fastest ones, 12 of 128 × 128
/// in groups of four and 16 in groups of eight.
fn large_sets() -> Vec<(u32, String, (u32, u32, u32))> {
    vec![(12, "mixed:4".into(), (6, 7, 15)), (12, "mixed:4".into(), (7, 7, 16)), (16, "mixed:8".into(), (7, 7, 16))]
}

impl Default for VoiceAsk {
    fn default() -> Self {
        Self { seconds: 3.0, words: 12, first_output_seconds: 0.5, area_mm2: f64::INFINITY, shapes: vec![(0, 6, 11), (0, 6, 13), (4, 6, 11), (4, 6, 13), (4, 6, 16)], chunks: vec![16, 32, 64], speaker: false, large: large_sets(), text_apart: false, system_tokens: 0 }
    }
}

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
/// `prompt` tokens that follows `before` already in its cache (a system prompt read while it
/// waited), with a context long enough for them and its answer.
#[allow(clippy::too_many_arguments)]
fn language_model(dir: &Path, costs: &Costs, (chips, sharing): (u32, Sharing), shape: (u32, u32, u32), before: u32, prompt: u32, needs: u32) -> Result<Fit, String> {
    let context = (before + prompt + 512).next_power_of_two().max(1024);
    let program = compiler::compile_model(&std::fs::read_to_string(dir.join("config.json")).map_err(|e| e.to_string())?)?;
    let per_token = crate::system::macs(&program.ops, &program.buffers);
    let target = Target { tokens_per_second: 0.0, prompt_seconds: f64::INFINITY, prompt, chips: vec![chips], clock_hz: CLOCK, process: N6, context, lane_bits: shape.1..=shape.1, cluster_bits: shape.0..=shape.0, cell_bits: shape.2..=shape.2, sharing: vec![sharing] };
    let c = measure(&program, &weights(dir, 64)?, reference_die(8), costs, &target, sharing, (chips, shape.0, shape.1, shape.2))?;
    Ok(Fit { shape, dies: c.chips, parts_mm2: c.parts_mm2, clock_hz: c.clock_hz, first_cycles: (c.prompt_seconds * c.clock_hz).round() as u64, output_cycles: c.cycles, needs, variant: String::new(), first_macs: per_token * prompt as u64, output_macs: per_token, pj_per_mac: pj_per_mac(16) })
}

/// A die whose programs are routines the firmware orders, with a command queue of 16 orders
/// before it: the CPU orders the next routine while the die runs this one.
fn queued(die: netlist::DieConfig) -> netlist::DieConfig {
    netlist::DieConfig { features: die.features.with_queue_bits(4), ..die }
}

/// Each stage of `models` measured on every shape of `ask` it can be built on, saying what
/// it is doing.
pub fn measure_stages(models: &VoiceModels, ask: &VoiceAsk, progress: &mut dyn FnMut(String)) -> Result<Vec<StagePlan>, String> {
    let roles = models.roles()?;
    let costs = Costs::measure(reference_die(8));
    let question_tokens = (ask.seconds * TOKENS_PER_SECOND_SPOKEN).ceil() as u32;
    let mut stages = Vec::new();
    for role in roles {
        let fits = match role {
            Role::Listen => measure_listen(models.listen.as_deref().ok_or("no recogniser")?, ask, &costs, progress)?,
            Role::Think => measure_think(&models.think, ask, &costs, question_tokens, progress)?,
            Role::Speak => measure_speak(models.speak.as_deref().ok_or("no speech writer")?, ask, &costs, progress)?,
            Role::Sound => measure_sound(models.sound.as_deref().ok_or("no sound decoder")?, ask, progress)?,
        };
        let rate = match role {
            Role::Speak => FRAMES_PER_SECOND * TOKENS_PER_FRAME,
            Role::Sound => FRAMES_PER_SECOND,
            _ => 0.0,
        };
        if fits.is_empty() {
            return Err(format!("the {} could not be built on any of the shapes tried", role.name()));
        }
        stages.push(StagePlan { name: role.name().into(), fits, rate });
    }
    Ok(stages)
}

/// The language model, on single dies when its weights could fit one, else on the larger
/// sets.
fn measure_think(dir: &Path, ask: &VoiceAsk, costs: &Costs, question_tokens: u32, progress: &mut dyn FnMut(String)) -> Result<Vec<Fit>, String> {
    let mut think = Vec::new();
    if could_fit_one_die(dir)? {
        for &shape in &ask.shapes {
            progress(format!("the language model on {}", shape_name(shape)));
            if let Ok(fit) = language_model(dir, costs, (1, Sharing::Layers), shape, ask.system_tokens, 40, question_tokens) {
                think.push(fit);
            }
        }
    }
    // A language model that fits no single die within the reticle: on the larger sets.
    if !think.iter().any(|f| f.parts_mm2.iter().sum::<f64>() <= N6.reticle_mm2) {
        for (chips, sharing, shape) in &ask.large {
            progress(format!("the language model on {chips} dies, {sharing}"));
            let sharing = crate::sizing::parse_sharings(sharing, &[*chips])?.into_iter().next().ok_or("no sharing")?;
            match language_model(dir, costs, (*chips, sharing), *shape, ask.system_tokens, 40, question_tokens) {
                Ok(fit) => think.push(fit),
                Err(e) => progress(format!("the language model on {chips} dies: {e}")),
            }
        }
    }
    Ok(think)
}

fn shape_name(shape: (u32, u32, u32)) -> String {
    format!("{} units × {} lanes, {} cells", 1u32 << shape.0, 1u32 << shape.1, 1u64 << shape.2)
}

#[cfg(not(feature = "listen"))]
fn measure_listen(_: &Path, _: &VoiceAsk, _: &Costs, _: &mut dyn FnMut(String)) -> Result<Vec<Fit>, String> {
    Err("this engine was built without listening (the `listen` feature)".into())
}

/// The recogniser on every shape: Moonshine after the question, or streaming Moonshine
/// while it is spoken.
#[cfg(feature = "listen")]
fn measure_listen(dir: &Path, ask: &VoiceAsk, costs: &Costs, progress: &mut dyn FnMut(String)) -> Result<Vec<Fit>, String> {
    let base = reference_die(8);
    let samples = (ask.seconds * 16_000.0) as usize;
    let lengths = Lengths::of(samples)?;
    let config = MoonshineConfig::read(dir)?;
    let frames = config.frames_of(samples);
    let context = match &config.streaming {
        // A streaming recogniser's stores of the front end, and its text.
        Some(streaming) => (4 + 4 * frames).max(frames * streaming.samples_per_frame() * 6 / 16_000 + 2).next_power_of_two(),
        None => lengths.sample_rows.max(config.max_tokens).next_power_of_two(),
    };
    let mut listen = Vec::new();
    for &shape in &ask.shapes {
        progress(format!("Moonshine on {}", shape_name(shape)));
        let die = queued(die_of(base, context as u32, shape));
        if config.streaming.is_some() {
            // Its widths, the front end's 80 among them, take groups of 16.
            let chip = crate::listen::moonshine_stream_chip::StreamingChip::from_parts(config.clone(), weights(dir, 16)?, frames, die, Some(context));
            if let Some((chip, code)) = chip.ok().and_then(|chip| chip.routines().ok().map(|(_, code)| (chip, code))) {
                let (parts, clock) = routines_die(chip.parts().1, &code, costs, &N6, shape, CLOCK);
                let t = chip.timing(frames, &code)?;
                // It listens while the question is spoken: each 20 ms in less than 20 ms.
                if t.step as f64 / clock <= 0.02 {
                    listen.push(Fit { shape, dies: 1, parts_mm2: parts, clock_hz: clock, first_cycles: t.after, output_cycles: t.token, needs: 0, variant: "listening as the question is spoken".into(), first_macs: t.clip_macs, output_macs: t.token_macs, pj_per_mac: pj_per_mac(16) });
                }
            }
            continue;
        }
        let Ok(chip) = MoonshineChip::from_parts(config.clone(), weights(dir, 32)?, samples, die, Some(context)) else { continue };
        let calls: Vec<_> = (0..lengths.sample_rows).map(|r| chip.store_samples_call(r)).chain(chip.encoder_calls()).chain(chip.token_calls(config.start, 0)).collect();
        let Ok((_, code)) = chip.microprogram(&calls) else { continue };
        let (parts, clock) = routines_die(chip.parts().1, &code, costs, &N6, shape, CLOCK);
        let stage = chip.stage(1, parts.iter().sum(), clock)?;
        let ops = &chip.parts().0.rows().expect("the routines").ops;
        let count = |calls: &[crate::listen::moonshine_chip::Call]| calls.iter().map(|c| crate::system::macs(&ops[c.routine.clone()], &chip.parts().0.buffers)).sum::<u64>();
        let token = chip.token_calls(config.start, 0);
        // The firmware orders every routine: its cycles come on top of the die's.
        let (first, each) = chip.firmware_cycles()?;
        listen.push(Fit { shape, dies: 1, parts_mm2: parts, clock_hz: clock, first_cycles: stage.first_cycles + first, output_cycles: stage.output_cycles + each, needs: 0, variant: String::new(), first_macs: count(&calls), output_macs: count(&token), pj_per_mac: pj_per_mac(16) });
    }
    Ok(listen)
}

#[cfg(not(feature = "speak"))]
fn measure_speak(_: &Path, _: &VoiceAsk, _: &Costs, _: &mut dyn FnMut(String)) -> Result<Vec<Fit>, String> {
    Err("this engine was built without speaking (the `speak` feature)".into())
}

/// OuteTTS on every shape, its prompt a sentence's, with the profile's sentence and codes
/// before it when the writer reads one.
#[cfg(feature = "speak")]
fn measure_speak(dir: &Path, ask: &VoiceAsk, costs: &Costs, progress: &mut dyn FnMut(String)) -> Result<Vec<Fit>, String> {
    let prompt = match crate::voice::speaker_of(dir)?.filter(|_| ask.speaker) {
        Some(speaker) => {
            let vocabulary = crate::speak::outetts::SpeechVocabulary::read(dir)?;
            let mut tokenizer = crate::tokenizer::TokenizerTables::from_file(&dir.join("tokenizer.json"), 4096)?;
            tokenizer.set_add_special(false);
            vocabulary.prompt_with(&speaker, "A sentence of about twelve words, as the answer's first.", &|t| tokenizer.encode(t))?.len() as u32
        }
        None => 40,
    };
    let mut speak = Vec::new();
    for &shape in &ask.shapes {
        progress(format!("OuteTTS on {}", shape_name(shape)));
        if let Ok(fit) = language_model(dir, costs, (1, Sharing::Layers), shape, 0, prompt, ask.words) {
            speak.push(fit);
        }
    }
    Ok(speak)
}

#[cfg(not(feature = "speak"))]
fn measure_sound(_: &Path, _: &VoiceAsk, _: &mut dyn FnMut(String)) -> Result<Vec<Fit>, String> {
    Err("this engine was built without speaking (the `speak` feature)".into())
}

/// DAC on every shape and window, with 20-bit vectors (see docs/voice.md).
#[cfg(feature = "speak")]
fn measure_sound(dir: &Path, ask: &VoiceAsk, progress: &mut dyn FnMut(String)) -> Result<Vec<Fit>, String> {
    let base = reference_die(8);
    let wide = netlist::DieConfig { cell: netlist::CellConfig { activation_bits: 20, accumulator_bits: 2 * 20 + 16, ..base.cell }, ..base };
    let wide_costs = Costs::measure(wide);
    let config = DacConfig::read(dir)?;
    let mut sound = Vec::new();
    for &shape in &ask.shapes {
        for &chunk in &ask.chunks {
            let frames = chunk + 2 * config.reach_frames();
            progress(format!("DAC on {}, windows of {frames} frames", shape_name(shape)));
            let Ok(chip) = DacChip::from_parts(config.clone(), weights(dir, 8)?, frames, queued(die_of(wide, 32_768, shape)), true) else { continue };
            let Ok(code) = chip.microprogram() else { continue };
            let calls = chip.calls(&vec![vec![0; frames]; config.codebooks])?;
            let (parts, clock) = routines_die(chip.parts().1, &code, &wide_costs, &N6, shape, CLOCK);
            // The die's cycles, and the firmware's to order its routines.
            let cycles = chip.cycles(&calls)? + chip.firmware_cycles(chunk * config.hop(), &code);
            let needs = (frames as f64 * TOKENS_PER_FRAME).ceil() as u32;
            let window_macs: u64 = calls.iter().map(|c| crate::system::macs(&chip.routines()[c.routine], &chip.parts().0.buffers)).sum();
            sound.push(Fit { shape, dies: 1, parts_mm2: parts, clock_hz: clock, first_cycles: cycles, output_cycles: cycles / chunk as u64, needs, variant: format!("windows of {frames} frames, {chunk} kept"), first_macs: window_macs, output_macs: window_macs / chunk as u64, pj_per_mac: pj_per_mac(20) });
        }
    }
    Ok(sound)
}

/// The plan chosen, if any meets what was asked, every grouping weighed, and the stages
/// as measured.
pub type Planned = (Option<Plan>, Vec<Weighed>, Vec<StagePlan>);

/// Changes when the way stages are measured changes, so plans kept from before are not used.
const PLAN_VERSION: u32 = 5;

/// The system planned for `ask`: the plan chosen, if any meets it, every grouping weighed,
/// and the stages as measured.
pub fn plan_voice(models: &VoiceModels, ask: &VoiceAsk, progress: &mut dyn FnMut(String)) -> Result<Planned, String> {
    let stages = measure_stages(models, ask, progress)?;
    let limits = Limits { reticle_mm2: N6.reticle_mm2, first_output_seconds: ask.first_output_seconds, area_mm2: ask.area_mm2 };
    let roles = models.roles()?;
    let mut apart = at_once(&roles);
    let listen = roles.iter().position(|r| *r == Role::Listen);
    let think = roles.iter().position(|r| *r == Role::Think);
    if let (true, Some(listen), Some(think)) = (ask.text_apart, listen, think) {
        apart.push((listen, think));
    }
    let (best, mut weighed) = plan(&stages, &apart, limits);
    if let (true, Some(listen), Some(think)) = (ask.text_apart, listen, think) {
        // They never run at once: they are kept apart because the language model's dies are the project's.
        let said = format!("{} and {} run at once", stages[listen].name, stages[think].name);
        for w in weighed.iter_mut().filter(|w| w.reason.as_deref() == Some(said.as_str())) {
            w.reason = Some("the language model has dies of its own".into());
        }
    }
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

/// The energy of a turn on `plan`, whose stages play `roles`: a question of `ask.seconds`
/// (spoken, when the chip listens), an answer of `ask.words` tokens (said at about three
/// tokens a second, when it speaks). Leakage is counted on every die for the whole turn,
/// none powered off, with every memory bit leaking as N6's do; the work is the products of
/// weights by values and the weights' reads, as the estimator counts a token.
pub fn turn_energy(plan: &Plan, roles: &[Role], ask: &VoiceAsk) -> TurnEnergy {
    let tech = arch::node("n6").expect("N6");
    let question_tokens = (ask.seconds * TOKENS_PER_SECOND_SPOKEN).ceil() as u64;
    let speaks = roles.contains(&Role::Speak);
    let answer_seconds = if speaks { ask.words as f64 / TOKENS_PER_SECOND_SPOKEN } else { 0.0 };
    let frames = (answer_seconds * FRAMES_PER_SECOND).ceil() as u64;
    let outputs = roles.iter().map(|role| match role {
        Role::Listen => question_tokens,
        Role::Think => ask.words as u64,
        Role::Speak => (frames as f64 * TOKENS_PER_FRAME) as u64,
        Role::Sound => frames,
    });
    let stages_j: Vec<f64> = plan.fits.iter().zip(outputs).map(|(f, n)| (f.first_macs + n.saturating_sub(1) * f.output_macs) as f64 * f.pj_per_mac * 1e-12).collect();
    let listening = if roles.contains(&Role::Listen) { ask.seconds } else { 0.0 };
    let seconds = listening + plan.first_output_seconds + answer_seconds;
    let leak_w: f64 = plan.dies.iter().map(|d| {
        let memory_mm2 = d.parts_mm2[1] + d.parts_mm2[2] + d.parts_mm2[3];
        (memory_mm2 * 1e6 / tech.sram_cell_um2 * tech.sram_leak_pw_bit * 1e-12 + d.parts_mm2[0] * tech.logic_leak_w_mm2) * d.count as f64
    }).sum();
    let leakage_j = leak_w * seconds;
    TurnEnergy { total_j: stages_j.iter().sum::<f64>() + leakage_j, stages_j, leakage_j, seconds }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_stages_a_project_has_and_which_run_at_once() {
        let models = |listen: bool, speak: bool| VoiceModels { listen: listen.then(|| "m".into()), think: "q".into(), speak: speak.then(|| "o".into()), sound: speak.then(|| "d".into()) };
        let all = models(true, true).roles().unwrap();
        assert_eq!(all, [Role::Listen, Role::Think, Role::Speak, Role::Sound]);
        assert_eq!(at_once(&all), [(1, 2), (1, 3), (2, 3)]);
        let listening = models(true, false).roles().unwrap();
        assert_eq!(listening, [Role::Listen, Role::Think]);
        assert!(at_once(&listening).is_empty());
        let speaking = models(false, true).roles().unwrap();
        assert_eq!(speaking, [Role::Think, Role::Speak, Role::Sound]);
        assert_eq!(at_once(&speaking), [(0, 1), (0, 2), (1, 2)]);
        let half = VoiceModels { sound: None, ..models(false, true) };
        assert!(half.roles().is_err(), "a writer without a decoder");
    }
}
