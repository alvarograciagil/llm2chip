//! A spoken turn's three stages, as the talk example and the application run them: hear a
//! clip with Moonshine, answer with a language model, and say the answer with OuteTTS and
//! DAC. Each runs in floating point or in the die's arithmetic; only text goes from one to
//! the next.

use crate::audio::Clip;
use crate::dac::{Dac, DacConfig};
use crate::moonshine::Moonshine;
use crate::moonshine_chip::MoonshineChip;
use crate::prompt::PromptStyle;
use crate::speak::{write, SpeechVocabulary};
use crate::tokenizer::TokenizerTables;
use crate::{Engine, ExactEngine, Numeric, Weights};
use std::path::Path;

/// Cuts a text into tokens, adding the special ones when told to.
pub type Encode<'a> = dyn Fn(&str, bool) -> Result<Vec<u32>, String> + 'a;
/// Joins tokens back into text.
pub type Decode<'a> = dyn Fn(&[u32]) -> Result<String, String> + 'a;

/// Where a stage runs: floating point, or the die's arithmetic with vectors of so many bits.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Arithmetic {
    Float,
    Die(u32),
}

/// A language model in either arithmetic: a token in, the next one's scores out.
pub enum Model {
    Float(Box<Engine>),
    Die(Box<ExactEngine>),
}

impl Model {
    pub fn open(dir: &Path, arithmetic: Arithmetic) -> Result<Self, String> {
        let program = compiler::compile_model(&std::fs::read_to_string(dir.join("config.json")).map_err(|e| e.to_string())?)?;
        Ok(match arithmetic {
            Arithmetic::Float => Self::Float(Box::new(Engine::new(program, Weights::open(dir)?, Numeric::Float)?)),
            Arithmetic::Die(bits) => Self::Die(Box::new(ExactEngine::new(program, Weights::open(dir)?, bits)?)),
        })
    }

    /// Back to an empty cache, for a new text.
    pub fn reset(&mut self) {
        match self {
            Self::Float(engine) => engine.reset(),
            Self::Die(engine) => engine.reset(),
        }
    }

    pub fn step(&mut self, token: u32) -> Result<Vec<f32>, String> {
        match self {
            Self::Float(engine) => Ok(engine.step(token, true)?.0.ok_or("no scores")?.to_vec()),
            Self::Die(engine) => engine.step(token, true)?.ok_or_else(|| "no scores".into()),
        }
    }
}

/// The text Moonshine, in `dir`, hears in `clip`.
pub fn listen(dir: &Path, clip: &Clip, arithmetic: Arithmetic) -> Result<String, String> {
    let samples = clip.resampled(16_000).samples;
    let tokens = match arithmetic {
        Arithmetic::Float => Moonshine::open(dir)?.transcribe(&samples)?,
        Arithmetic::Die(bits) => MoonshineChip::exact(dir, samples.len(), bits)?.transcribe(&samples)?,
    };
    let text = TokenizerTables::from_file(&dir.join("tokenizer.json"), 256)?.decode(&tokens);
    Ok(String::from_utf8_lossy(&text).trim().to_string())
}

/// The language model in `dir`'s answer to `question`, in one short sentence and without
/// reasoning, greedily, at most `limit` tokens. `encode` cuts a text into tokens (and adds
/// the special ones when told to), `decode` joins them back and `token_id` names one.
pub fn answer(dir: &Path, question: &str, arithmetic: Arithmetic, limit: usize, encode: &Encode, decode: &Decode, token_id: &dyn Fn(&str) -> Option<u32>) -> Result<String, String> {
    let family = compiler::compile_model(&std::fs::read_to_string(dir.join("config.json")).map_err(|e| e.to_string())?)?.family;
    let style = PromptStyle::read(&family, |name| std::fs::read_to_string(dir.join(name)).ok(), token_id);
    let (framed, add) = style.render(&format!("{question}\nAnswer in one short sentence."), false)?;
    let mut model = Model::open(dir, arithmetic)?;
    let mut logits = Vec::new();
    for token in encode(&framed, add)? {
        logits = model.step(token)?;
    }
    let mut reply = Vec::new();
    for _ in 0..limit {
        let next = crate::moonshine::argmax(&logits);
        if style.stop.contains(&next) {
            break;
        }
        reply.push(next);
        logits = model.step(next)?;
    }
    Ok(decode(&reply)?.trim().to_string())
}

/// The sentences of `text`, each with its closing mark: what the speech writer takes one at
/// a time, so the first can sound while the next is written.
pub fn sentences(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::new();
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        current.push(c);
        if matches!(c, '.' | '!' | '?') && chars.peek().is_none_or(|n| n.is_whitespace()) {
            out.push(current.trim().to_string());
            current.clear();
        }
    }
    if !current.trim().is_empty() {
        out.push(current.trim().to_string());
    }
    out.retain(|s| !s.is_empty());
    out
}

/// Frames the sound decoder keeps from each window it decodes, on the die: a window is
/// this plus a frame's reach on both sides (see `DacConfig::windows`).
pub const CHUNK_FRAMES: usize = 16;

/// `text` said by OuteTTS, in `tts`, and DAC, in `dac`, a sentence at a time; on the die,
/// the decoder takes a window of frames at a time. The sound, and how many tokens OuteTTS
/// wrote for it.
pub fn speak(tts: &Path, dac: &Path, text: &str, arithmetic: Arithmetic) -> Result<(Clip, usize), String> {
    let vocabulary = SpeechVocabulary::read(tts)?;
    let mut tokenizer = TokenizerTables::from_file(&tts.join("tokenizer.json"), 1024)?;
    tokenizer.set_add_special(false);
    let mut writer = Model::open(tts, arithmetic)?;
    let rate = DacConfig::read(dac)?.sampling_rate;
    let floating = match arithmetic {
        Arithmetic::Float => Some(Dac::open(dac)?),
        Arithmetic::Die(_) => None,
    };
    let (mut samples, mut tokens) = (Vec::new(), 0);
    for sentence in sentences(text) {
        writer.reset();
        let prompt = vocabulary.prompt(&tokenizer.encode(&sentence)?);
        let written = write(&vocabulary, &prompt, 1500, &mut |token| writer.step(token))?;
        tokens += written.len();
        let codes = vocabulary.codes(&written);
        if codes[0].is_empty() {
            continue;
        }
        let sound = match (&floating, arithmetic) {
            (Some(dac), _) => dac.decode(&codes)?,
            (None, Arithmetic::Die(bits)) => crate::dac_chip::decode_chunked(dac, &codes, CHUNK_FRAMES, bits)?.0,
            (None, Arithmetic::Float) => unreachable!(),
        };
        samples.extend(sound);
    }
    if samples.is_empty() {
        return Err("the speech model wrote no sound".into());
    }
    Ok((Clip { rate, samples }, tokens))
}

#[cfg(test)]
mod tests {
    #[test]
    fn a_text_splits_into_its_sentences() {
        assert_eq!(super::sentences("Hello there. It is 3.5 degrees! Is it? yes"), vec!["Hello there.", "It is 3.5 degrees!", "Is it?", "yes"]);
        assert!(super::sentences("  ").is_empty());
    }
}
