//! A spoken turn in the engine: a clip heard by Moonshine, answered by a language model and
//! said by OuteTTS and DAC, one at a time, each stage in floating point or in the die's
//! arithmetic. The clip comes and the answer goes as WAV files in base64.

use serde::Deserialize;
use serde_json::{json, Value};
use sim::audio::Clip;
use sim::voice::{listen, speak, Arithmetic, Conversation};
use std::path::Path;
use std::sync::Mutex;
use std::time::Instant;
use tokenizers::Tokenizer;

#[derive(Deserialize)]
pub struct VoiceStart {
    /// The question, a WAV file in base64.
    pub wav: String,
    /// The folders of the recogniser, the language model, the speech writer and the decoder.
    pub listen: String,
    pub think: String,
    pub speak: String,
    pub sound: String,
    /// The die's vectors' bits, or floating point.
    pub bits: Option<u32>,
    /// Start a new conversation instead of going on with the last one.
    #[serde(default)]
    pub fresh: bool,
    /// Speak in the voice of the profile kept with OuteTTS, if it keeps one.
    #[serde(default)]
    pub speaker: bool,
}

#[derive(Default)]
pub struct VoiceJob {
    /// What it is doing: `listening`, `thinking` or `speaking`, and what each stage gave.
    pub phase: Mutex<String>,
    pub partial: Mutex<Value>,
    pub outcome: Mutex<Option<Result<Value, String>>>,
}

/// Base64 to bytes, ignoring what is not a digit of it (a data URL's head included).
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
    let arithmetic = start.bits.map_or(Arithmetic::Float, Arithmetic::Die);
    let clip = Clip::parse(&unbase64(&start.wav))?;
    if clip.samples.len() < clip.rate as usize / 10 {
        return Err("the clip is too short".into());
    }
    if clip.samples.len() > clip.rate as usize * 30 {
        return Err("the clip is longer than 30 seconds".into());
    }
    let step = |phase: &str, key: &str, value: Value| {
        *job.phase.lock().unwrap() = phase.into();
        if let Value::Object(map) = &mut *job.partial.lock().unwrap() {
            map.insert(key.into(), value);
        }
    };
    *job.partial.lock().unwrap() = json!({});

    *job.phase.lock().unwrap() = "listening".into();
    let started = Instant::now();
    let heard = listen(Path::new(&start.listen), &clip, arithmetic)?;
    step("thinking", "heard", json!({ "text": heard, "seconds": started.elapsed().as_secs_f64() }));
    if heard.is_empty() {
        return Err("nothing was heard in the clip".into());
    }

    let started = Instant::now();
    let think = Path::new(&start.think);
    let tokenizer = Tokenizer::from_file(think.join("tokenizer.json")).map_err(|e| e.to_string())?;
    let encode = |text: &str, add: bool| tokenizer.encode(text, add).map(|e| e.get_ids().to_vec()).map_err(|e| e.to_string());
    let decode = |tokens: &[u32]| tokenizer.decode(tokens, true).map_err(|e| e.to_string());
    // The conversation goes on with the same model in the same arithmetic, its cache kept.
    let mut slot = kept.lock().unwrap();
    let same = slot.as_ref().is_some_and(|c| c.dir == think && c.arithmetic == arithmetic);
    if start.fresh || !same {
        *slot = Some(Conversation::new(think, arithmetic, &|t| tokenizer.token_to_id(t))?);
    }
    let conversation = slot.as_mut().unwrap();
    let (said, _) = conversation.ask(&heard, 60, &encode, &decode)?;
    let turn = conversation.turns();
    drop(slot);
    step("speaking", "answer", json!({ "text": said, "seconds": started.elapsed().as_secs_f64(), "turn": turn }));

    let started = Instant::now();
    // OuteTTS's own sampling, a new draw each time.
    let seed = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(1, |d| d.as_nanos() as u64);
    let speaker = if start.speaker { sim::voice::speaker_of(Path::new(&start.speak))? } else { None };
    let (sound, tokens) = speak(Path::new(&start.speak), Path::new(&start.sound), &said, arithmetic, sim::speak::outetts::Sampling::outetts(seed), speaker.as_ref())?;
    let audio = format!("data:audio/wav;base64,{}", crate::base64(&sound.to_wav()));
    let spoken = json!({ "tokens": tokens, "seconds": started.elapsed().as_secs_f64(), "duration": sound.samples.len() as f64 / sound.rate as f64, "audio": audio });
    let mut report = job.partial.lock().unwrap().clone();
    report["spoken"] = spoken;
    report["bits"] = json!(start.bits);
    Ok(report)
}

/// What the system is planned for: the four models and the limits, as the view asks.
#[derive(Deserialize)]
pub struct PlanStart {
    pub listen: String,
    pub think: String,
    pub speak: String,
    pub sound: String,
    /// The longest wait from the end of the question to the first sound, ms.
    pub first_sound_ms: f64,
    /// The longest question, s.
    pub seconds: f64,
    /// The most silicon in all, mm², if limited.
    pub area_mm2: Option<f64>,
    /// The speech writer reads a speaker's profile before each sentence.
    #[serde(default)]
    pub speaker: bool,
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
    let models = VoiceModels { listen: start.listen.clone().into(), think: start.think.clone().into(), speak: start.speak.clone().into(), sound: start.sound.clone().into() };
    if !(0.5..=30.0).contains(&start.seconds) {
        return Err("the question must last between half a second and 30 seconds".into());
    }
    let ask = VoiceAsk { first_sound_seconds: start.first_sound_ms / 1e3, seconds: start.seconds, area_mm2: start.area_mm2.unwrap_or(f64::INFINITY), speaker: start.speaker, ..VoiceAsk::default() };
    let (best, weighed, stages) = plan_voice_cached(&home.join(".llm2chip/cache"), &models, &ask, &mut |what| *job.doing.lock().unwrap() = what)?;
    let names: Vec<&str> = stages.iter().map(|s| s.name.as_str()).collect();
    let named = |group: &[usize]| group.iter().map(|s| names[*s]).collect::<Vec<_>>();
    let plan = best.map(|p| {
        let energy = turn_energy(&p, &ask);
        json!({
            "energy": { "total_j": energy.total_j, "leakage_j": energy.leakage_j, "seconds": energy.seconds, "stages_j": energy.stages_j },
            "area_mm2": p.area_mm2,
            "first_sound_ms": p.first_output_seconds * 1e3,
            "dies": p.dies.iter().map(|d| json!({ "stages": named(&d.stages), "count": d.count, "units": 1u32 << d.shape.0, "lanes": 1u32 << d.shape.1, "cells": 1u64 << d.shape.2, "mm2": d.mm2(), "parts_mm2": d.parts_mm2, "clock_mhz": d.clock_hz / 1e6 })).collect::<Vec<_>>(),
            "stages": p.stages.iter().zip(&p.variants).map(|(s, v)| json!({ "name": s.name, "first_ms": s.seconds_to(1) * 1e3, "per_second": s.outputs_per_second(), "variant": v })).collect::<Vec<_>>(),
        })
    });
    let weighed: Vec<Value> = weighed.iter().map(|w| json!({ "groups": w.groups.iter().map(|g| named(g)).collect::<Vec<_>>(), "area_mm2": w.best.as_ref().map(|p| p.area_mm2), "first_sound_ms": w.best.as_ref().map(|p| p.first_output_seconds * 1e3), "reason": w.reason })).collect();
    Ok(json!({ "plan": plan, "weighed": weighed }))
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
