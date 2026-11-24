//! A spoken turn's three stages, as the talk example and the application run them: hear a
//! clip with Moonshine, answer with a language model, and say the answer with OuteTTS and
//! DAC. Each runs in floating point or in the die's arithmetic; only text goes from one to
//! the next.

use crate::audio::Clip;
use crate::dac::{Dac, DacConfig};
use crate::moonshine::Moonshine;
use crate::moonshine_chip::MoonshineChip;
use crate::prompt::PromptStyle;
use crate::speak::{write_sampled, Sampling, Speaker, SpeechVocabulary};
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

/// A spoken conversation with a language model: its cache stays between questions, so
/// each new one costs only its own tokens. What is fed on top of the cache is what the chat
/// template writes after the last answer's text: its end and the new turn. A template that
/// does not write the last answer as it was makes it start again from the whole conversation.
pub struct Conversation {
    model: Model,
    style: PromptStyle,
    messages: Vec<(String, String)>,
    /// Every token the cache holds.
    tokens: Vec<u32>,
    pub arithmetic: Arithmetic,
    pub dir: std::path::PathBuf,
    /// Times the conversation had to start again from the whole text.
    pub replayed: usize,
}

impl Conversation {
    pub fn new(dir: &Path, arithmetic: Arithmetic, token_id: &dyn Fn(&str) -> Option<u32>) -> Result<Self, String> {
        let family = compiler::compile_model(&std::fs::read_to_string(dir.join("config.json")).map_err(|e| e.to_string())?)?.family;
        let style = PromptStyle::read(&family, |name| std::fs::read_to_string(dir.join(name)).ok(), token_id);
        Ok(Self { model: Model::open(dir, arithmetic)?, style, messages: Vec::new(), tokens: Vec::new(), arithmetic, dir: dir.to_path_buf(), replayed: 0 })
    }

    /// Questions asked so far.
    pub fn turns(&self) -> usize {
        self.messages.len() / 2
    }

    /// Every token the cache holds, in order.
    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    /// The answer to `question` in one short sentence, without reasoning, greedily, at most
    /// `limit` tokens, after every earlier question and answer; with the next token's scores.
    pub fn ask(&mut self, question: &str, limit: usize, encode: &Encode, decode: &Decode) -> Result<(String, Vec<f32>), String> {
        let earlier: Vec<(&str, &str)> = self.messages.iter().map(|(r, c)| (r.as_str(), c.as_str())).collect();
        let asked = format!("{question}\nAnswer in one short sentence.");
        let mut messages = earlier.clone();
        messages.push(("user", &asked));
        let (framed, add) = self.style.render_conversation(&messages, false, true)?;
        // On top of the cache: what the template writes after the last answer's text, before
        // the new question (the answer's end), and the new turn itself. Templates may write
        // earlier turns otherwise than they wrote them last time (Qwen3 drops an answer's
        // empty reasoning once a question follows), so only the new framing is read.
        let on_top = match earlier.last() {
            Some((_, last)) if !self.tokens.is_empty() && !last.is_empty() => framed.rfind(asked.as_str()).and_then(|question| framed[..question].rfind(last)).map(|at| framed[at + last.len()..].to_string()),
            _ => None,
        };
        let new = match on_top {
            Some(rest) => encode(&rest, false)?,
            None => {
                self.replayed += usize::from(!self.tokens.is_empty());
                self.model.reset();
                self.tokens.clear();
                encode(&framed, add)?
            }
        };
        let mut logits = Vec::new();
        for token in &new {
            logits = self.model.step(*token)?;
        }
        self.tokens.extend(&new);
        let mut reply = Vec::new();
        for _ in 0..limit {
            let next = crate::moonshine::argmax(&logits);
            if self.style.stop.contains(&next) {
                break;
            }
            reply.push(next);
            logits = self.model.step(next)?;
        }
        self.tokens.extend(&reply);
        let said = decode(&reply)?.trim().to_string();
        self.messages.push(("user".into(), asked));
        self.messages.push(("assistant".into(), said.clone()));
        Ok((said, logits))
    }
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

/// Words of a speaker profile the speech writer reads before each sentence: OuteTTS's own
/// profiles have about 26, some 1,800 tokens; the first three keep the voice in about 200.
pub const PROFILE_WORDS: usize = 3;

/// The voice OuteTTS imitates, if its folder keeps one: the first profile in `speakers/`,
/// its first `PROFILE_WORDS` words.
pub fn speaker_of(tts: &Path) -> Result<Option<Speaker>, String> {
    let Ok(entries) = std::fs::read_dir(tts.join("speakers")) else { return Ok(None) };
    let mut profiles: Vec<_> = entries.filter_map(Result::ok).map(|e| e.path()).filter(|p| p.extension().is_some_and(|e| e == "json")).collect();
    profiles.sort();
    Ok(profiles.first().map(|p| Speaker::read(p)).transpose()?.map(|s| s.first_words(PROFILE_WORDS)))
}

/// `text` said by OuteTTS, in `tts`, and DAC, in `dac`, a sentence at a time, each token
/// chosen by `sampling`, in `speaker`'s voice if given (its sentence and codes go before
/// each sentence's: about 1,800 tokens more to read); on the die, the decoder takes a window
/// of frames at a time. The sound, and how many tokens OuteTTS wrote for it.
pub fn speak(tts: &Path, dac: &Path, text: &str, arithmetic: Arithmetic, sampling: Sampling, speaker: Option<&Speaker>) -> Result<(Clip, usize), String> {
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
        let prompt = match speaker {
            Some(speaker) => vocabulary.prompt_with(speaker, &sentence, &|t| tokenizer.encode(t))?,
            None => vocabulary.prompt(&tokenizer.encode(&sentence)?),
        };
        let written = write_sampled(&vocabulary, &prompt, 1500, sampling, &mut |token| writer.step(token))?;
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
