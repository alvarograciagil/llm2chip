//! A spoken turn in the engine, for a chip that listens, speaks or both: a question heard by
//! Moonshine from a clip (or given as text), answered by a language model (or not, to read a
//! text aloud), and said by OuteTTS and DAC (or left as text). Each stage runs in floating
//! point or in the die's arithmetic. Clips come and go as WAV files in base64. Listening is
//! built with the `listen` feature and speaking with `speak`.

use serde::Deserialize;
use serde_json::{json, Value};
use sim::voice::{Arithmetic, Conversation};
use std::path::Path;
use std::sync::Mutex;
use std::time::Instant;
use tokenizers::Tokenizer;

/// A turn: the question as a clip (`wav`, with the recogniser `listen`) or as `text`; the
/// language model `think`, if it answers; the speech writer `speak` and the decoder `sound`,
/// if the answer is said.
#[derive(Deserialize)]
pub struct VoiceStart {
    /// The question, a WAV file in base64.
    #[serde(default)]
    pub wav: Option<String>,
    /// Or the question as text.
    #[serde(default)]
    pub text: Option<String>,
    #[serde(default)]
    pub listen: Option<String>,
    #[serde(default)]
    pub think: Option<String>,
    #[serde(default)]
    pub speak: Option<String>,
    #[serde(default)]
    pub sound: Option<String>,
    /// The die's vectors' bits, or floating point.
    pub bits: Option<u32>,
    /// Start a new conversation instead of going on with the last one.
    #[serde(default)]
    pub fresh: bool,
    /// Speak in the voice of the profile kept with OuteTTS, if it keeps one.
    #[serde(default)]
    pub speaker: bool,
    /// The project's fixed system prompt, which the language model reads before the first
    /// question, as the chip runs it.
    #[serde(default)]
    pub system: String,
}

impl VoiceStart {
    /// Why this turn cannot be, if it cannot: what each part needs, and what this engine has.
    pub fn refusal(&self) -> Option<String> {
        if self.wav.is_some() == self.text.is_some() {
            return Some("a turn takes its question as a clip or as text, one of them".into());
        }
        if self.wav.is_some() && self.listen.is_none() {
            return Some("a spoken question needs a recogniser to listen".into());
        }
        if self.speak.is_some() != self.sound.is_some() {
            return Some("speaking needs both the speech writer and the sound decoder".into());
        }
        if self.text.is_some() && self.speak.is_none() {
            return Some("a written question with nothing to say is not a spoken turn".into());
        }
        if self.wav.is_some() && !cfg!(feature = "listen") {
            return Some("this engine was built without listening".into());
        }
        if self.speak.is_some() && !cfg!(feature = "speak") {
            return Some("this engine was built without speaking".into());
        }
        None
    }

    /// The models it names, each with the part it plays: `listens`, `speaks`, `sounds`, or
    /// none for the language model.
    pub fn models(&self) -> Vec<(&str, Option<&'static str>)> {
        [(&self.listen, Some("listens")), (&self.think, None), (&self.speak, Some("speaks")), (&self.sound, Some("sounds"))].into_iter().filter_map(|(path, role)| path.as_deref().map(|p| (p, role))).collect()
    }
}

#[derive(Default)]
pub struct VoiceJob {
    /// What it is doing: `listening`, `thinking` or `speaking`, and what each stage gave.
    pub phase: Mutex<String>,
    pub partial: Mutex<Value>,
    pub outcome: Mutex<Option<Result<Value, String>>>,
}

/// Base64 to bytes, ignoring what is not a digit of it (a data URL's head included).
#[cfg(any(feature = "listen", test))]
pub fn unbase64(text: &str) -> Vec<u8> {
    let text = text.rsplit(',').next().unwrap_or(text);
    let digit = |c: u8| match c {
        b'A'..=b'Z' => Some(c - b'A'),
        b'a'..=b'z' => Some(c - b'a' + 26),
        b'0'..=b'9' => Some(c - b'0' + 52),
        b'+' | b'-' => Some(62),
        b'/' | b'_' => Some(63),
        _ => None,
    };
    let (mut out, mut bits, mut held) = (Vec::with_capacity(text.len() * 3 / 4), 0u32, 0u32);
    for value in text.bytes().filter_map(digit) {
        held = held << 6 | value as u32;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((held >> bits) as u8);
        }
    }
    out
}

pub fn run(start: &VoiceStart, job: &VoiceJob, kept: &Mutex<Option<Conversation>>) -> Result<Value, String> {
    if let Some(why) = start.refusal() {
        return Err(why);
    }
    let arithmetic = start.bits.map_or(Arithmetic::Float, Arithmetic::Die);
    let step = |phase: &str, key: &str, value: Value| {
        *job.phase.lock().unwrap() = phase.into();
        if let Value::Object(map) = &mut *job.partial.lock().unwrap() {
            map.insert(key.into(), value);
        }
    };
    *job.partial.lock().unwrap() = json!({});

    // The question: heard, or given.
    let question = match (&start.wav, &start.text) {
        (Some(wav), _) => {
            *job.phase.lock().unwrap() = "listening".into();
            let started = Instant::now();
            let heard = hear(wav, start.listen.as_deref().unwrap_or_default(), arithmetic)?;
            step("thinking", "heard", json!({ "text": heard, "seconds": started.elapsed().as_secs_f64() }));
            if heard.is_empty() {
                return Err("nothing was heard in the clip".into());
            }
            heard
        }
        (None, Some(text)) => text.trim().to_string(),
        (None, None) => unreachable!("refused above"),
    };

    // The answer, or the question itself when no language model answers.
    let answer = match &start.think {
        Some(think) => {
            *job.phase.lock().unwrap() = "thinking".into();
            let started = Instant::now();
            let think = Path::new(think);
            let tokenizer = Tokenizer::from_file(think.join("tokenizer.json")).map_err(|e| e.to_string())?;
            let encode = |text: &str, add: bool| tokenizer.encode(text, add).map(|e| e.get_ids().to_vec()).map_err(|e| e.to_string());
            let decode = |tokens: &[u32]| tokenizer.decode(tokens, true).map_err(|e| e.to_string());
            // The conversation goes on with the same model in the same arithmetic, its cache kept.
            let mut slot = kept.lock().unwrap();
            let same = slot.as_ref().is_some_and(|c| c.dir == think && c.arithmetic == arithmetic && c.system == start.system.trim());
            if start.fresh || !same {
                *slot = Some(Conversation::new(think, arithmetic, &start.system, &|t| tokenizer.token_to_id(t))?);
            }
            let conversation = slot.as_mut().unwrap();
            let (said, _) = conversation.ask(&question, 60, &encode, &decode)?;
            let turn = conversation.turns();
            drop(slot);
            step("speaking", "answer", json!({ "text": said, "seconds": started.elapsed().as_secs_f64(), "turn": turn }));
            said
        }
        None => question,
    };

    // The answer said, when the chip speaks.
    let mut report = job.partial.lock().unwrap().clone();
    if let (Some(writer), Some(decoder)) = (&start.speak, &start.sound) {
        *job.phase.lock().unwrap() = "speaking".into();
        report["spoken"] = say(&answer, writer, decoder, arithmetic, start.speaker)?;
    }
    report["bits"] = json!(start.bits);
    Ok(report)
}

/// The text the recogniser in `dir` hears in a WAV clip in base64.
#[cfg(feature = "listen")]
fn hear(wav: &str, dir: &str, arithmetic: Arithmetic) -> Result<String, String> {
    let clip = sim::audio::Clip::parse(&unbase64(wav))?;
    if clip.samples.len() < clip.rate as usize / 10 {
        return Err("the clip is too short".into());
    }
    if clip.samples.len() > clip.rate as usize * 30 {
        return Err("the clip is longer than 30 seconds".into());
    }
    sim::voice::listen(Path::new(dir), &clip, arithmetic)
}

#[cfg(not(feature = "listen"))]
fn hear(_: &str, _: &str, _: Arithmetic) -> Result<String, String> {
    Err("this engine was built without listening".into())
}

/// `text` said by the writer in `writer` and the decoder in `decoder`: the sound as a WAV in
/// base64, with how long it took and how many tokens it was.
#[cfg(feature = "speak")]
fn say(text: &str, writer: &str, decoder: &str, arithmetic: Arithmetic, speaker: bool) -> Result<Value, String> {
    let started = Instant::now();
    // OuteTTS's own sampling, a new draw each time.
    let seed = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(1, |d| d.as_nanos() as u64);
    let speaker = if speaker { sim::voice::speaker_of(Path::new(writer))? } else { None };
    let (sound, tokens) = sim::voice::speak(Path::new(writer), Path::new(decoder), text, arithmetic, sim::speak::outetts::Sampling::outetts(seed), speaker.as_ref())?;
    let audio = format!("data:audio/wav;base64,{}", crate::base64(&sound.to_wav()));
    Ok(json!({ "tokens": tokens, "seconds": started.elapsed().as_secs_f64(), "duration": sound.samples.len() as f64 / sound.rate as f64, "audio": audio }))
}

#[cfg(not(feature = "speak"))]
fn say(_: &str, _: &str, _: &str, _: Arithmetic, _: bool) -> Result<Value, String> {
    Err("this engine was built without speaking".into())
}

/// What the system is planned for: the models of the parts the chip has (the language model
/// always; the recogniser when it listens; the writer and the decoder when it speaks) and
/// the limits, as the views ask.
#[derive(Deserialize)]
pub struct PlanStart {
    #[serde(default)]
    pub listen: Option<String>,
    pub think: String,
    #[serde(default)]
    pub speak: Option<String>,
    #[serde(default)]
    pub sound: Option<String>,
    /// The longest wait from the end of the question to the first output (the first sound,
    /// or the answer's first token when the chip only listens), ms.
    #[serde(alias = "first_sound_ms")]
    pub first_output_ms: f64,
    /// The longest question, s.
    pub seconds: f64,
    /// The most silicon in all, mm², if limited.
    pub area_mm2: Option<f64>,
    /// The speech writer reads a speaker's profile before each sentence.
    #[serde(default)]
    pub speaker: bool,
    /// The project's fixed system prompt, which the language model reads before every
    /// question.
    #[serde(default)]
    pub system: String,
}

/// Tokens the system prompt takes framed for the language model in `think`.
fn system_tokens(think: &Path, system: &str) -> Result<u32, String> {
    let system = system.trim();
    if system.is_empty() {
        return Ok(0);
    }
    let tokenizer = Tokenizer::from_file(think.join("tokenizer.json")).map_err(|e| e.to_string())?;
    let family = compiler::compile_model(&std::fs::read_to_string(think.join("config.json")).map_err(|e| e.to_string())?)?.family;
    let style = crate::prompt::read(think, &family, &tokenizer);
    let (framed, special) = style.render_with_system(system, "?", false)?;
    let encode = |text: &str, special: bool| tokenizer.encode(text, special).map(|e| e.get_ids().to_vec()).map_err(|e| e.to_string());
    Ok(style.split_system(system, &framed, false, special, encode)?.0.len() as u32)
}

#[derive(Default)]
pub struct PlanJob {
    pub doing: Mutex<String>,
    pub outcome: Mutex<Option<Result<Value, String>>>,
}

/// The system planned, as the view shows it: the dies chosen with their stages' names, each
/// stage's timing, and every grouping weighed.
pub fn run_plan(start: &PlanStart, job: &PlanJob, home: &Path) -> Result<Value, String> {
    use sim::voice_plan::{plan_voice_cached, turn_energy, VoiceAsk, VoiceModels};
    let models = VoiceModels { listen: start.listen.as_ref().map(Into::into), think: start.think.clone().into(), speak: start.speak.as_ref().map(Into::into), sound: start.sound.as_ref().map(Into::into) };
    let roles = models.roles()?;
    if !(0.5..=30.0).contains(&start.seconds) {
        return Err("the question must last between half a second and 30 seconds".into());
    }
    let ask = VoiceAsk { first_output_seconds: start.first_output_ms / 1e3, seconds: start.seconds, area_mm2: start.area_mm2.unwrap_or(f64::INFINITY), speaker: start.speaker, text_apart: true, system_tokens: system_tokens(Path::new(&start.think), &start.system)?, ..VoiceAsk::default() };
    let (best, weighed, stages) = plan_voice_cached(&home.join(".llm2chip/cache"), &models, &ask, &mut |what| *job.doing.lock().unwrap() = what)?;
    let names: Vec<&str> = stages.iter().map(|s| s.name.as_str()).collect();
    let named = |group: &[usize]| group.iter().map(|s| names[*s]).collect::<Vec<_>>();
    let plan = best.map(|p| {
        let energy = turn_energy(&p, &roles, &ask);
        json!({
            "energy": { "total_j": energy.total_j, "leakage_j": energy.leakage_j, "seconds": energy.seconds, "stages_j": energy.stages_j },
            "area_mm2": p.area_mm2,
            "first_output_ms": p.first_output_seconds * 1e3,
            "dies": p.dies.iter().map(|d| json!({ "stages": named(&d.stages), "count": d.count, "units": 1u32 << d.shape.0, "lanes": 1u32 << d.shape.1, "cells": 1u64 << d.shape.2, "mm2": d.mm2(), "parts_mm2": d.parts_mm2, "clock_mhz": d.clock_hz / 1e6 })).collect::<Vec<_>>(),
            "stages": p.stages.iter().zip(&p.variants).map(|(s, v)| json!({ "name": s.name, "first_ms": s.seconds_to(1) * 1e3, "per_second": s.outputs_per_second(), "variant": v })).collect::<Vec<_>>(),
        })
    });
    let weighed: Vec<Value> = weighed.iter().map(|w| json!({ "groups": w.groups.iter().map(|g| named(g)).collect::<Vec<_>>(), "area_mm2": w.best.as_ref().map(|p| p.area_mm2), "first_output_ms": w.best.as_ref().map(|p| p.first_output_seconds * 1e3), "reason": w.reason })).collect();
    Ok(json!({ "plan": plan, "weighed": weighed, "listens": models.listen.is_some(), "speaks": models.speak.is_some() }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_comes_back_as_the_bytes_it_was() {
        let bytes: Vec<u8> = (0..=255).collect();
        assert_eq!(unbase64(&format!("data:audio/wav;base64,{}", crate::base64(&bytes))), bytes);
    }
}
