//! A spoken turn in the engine: a clip heard by Moonshine, answered by a language model and
//! said by OuteTTS and DAC, one at a time, each stage in floating point or in the die's
//! arithmetic. The clip comes and the answer goes as WAV files in base64.

use serde::Deserialize;
use serde_json::{json, Value};
use sim::audio::Clip;
use sim::voice::{answer, listen, speak, Arithmetic};
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

pub fn run(start: &VoiceStart, job: &VoiceJob) -> Result<Value, String> {
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
    let said = answer(think, &heard, arithmetic, 60, &encode, &decode, &|t| tokenizer.token_to_id(t))?;
    step("speaking", "answer", json!({ "text": said, "seconds": started.elapsed().as_secs_f64() }));

    let started = Instant::now();
    let (sound, tokens) = speak(Path::new(&start.speak), Path::new(&start.sound), &said, arithmetic)?;
    let audio = format!("data:audio/wav;base64,{}", crate::base64(&sound.to_wav()));
    let spoken = json!({ "tokens": tokens, "seconds": started.elapsed().as_secs_f64(), "duration": sound.samples.len() as f64 / sound.rate as f64, "audio": audio });
    let mut report = job.partial.lock().unwrap().clone();
    report["spoken"] = spoken;
    report["bits"] = json!(start.bits);
    Ok(report)
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
