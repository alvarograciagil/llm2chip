//! A spoken turn's stages, as the talk example and the application run them: hear a clip
//! with Moonshine (`listen`, with the `listen` feature), answer with a language model
//! (`answer`, `Conversation`), and say the answer with OuteTTS and DAC (`speak`, with the
//! `speak` feature). Each runs in floating point or in the die's arithmetic; only text goes
//! from one to the next. A chip may listen, speak or both: each stage stands alone.

#[cfg(any(feature = "listen", feature = "speak"))]
use crate::audio::Clip;
#[cfg(feature = "listen")]
use crate::listen::moonshine::Moonshine;
#[cfg(feature = "listen")]
use crate::listen::moonshine_chip::MoonshineChip;
use crate::prompt::PromptStyle;
#[cfg(feature = "speak")]
use crate::speak::dac::{Dac, DacConfig};
#[cfg(feature = "speak")]
use crate::speak::outetts::{write_sampled, Sampling, Speaker, SpeechVocabulary};
#[cfg(any(feature = "listen", feature = "speak"))]
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

#[cfg(feature = "listen")]
/// The text Moonshine, in `dir`, hears in `clip`.
pub fn listen(dir: &Path, clip: &Clip, arithmetic: Arithmetic) -> Result<String, String> {
    let mut samples = clip.resampled(16_000).samples;
    let config = crate::listen::moonshine::MoonshineConfig::read(dir)?;
    // A streaming recogniser takes whole 20 ms.
    if let Some(streaming) = &config.streaming {
        samples.truncate(samples.len() / streaming.samples_per_frame() * streaming.samples_per_frame());
    }
    let tokens = match arithmetic {
        Arithmetic::Float => Moonshine::open(dir)?.transcribe(&samples)?,
        Arithmetic::Die(bits) if config.streaming.is_some() => crate::listen::moonshine_stream_chip::StreamingChip::exact(dir, config.frames_of(samples.len()), bits)?.transcribe(&samples)?.0,
        Arithmetic::Die(bits) => MoonshineChip::exact(dir, samples.len(), bits)?.transcribe(&samples)?,
    };
    let text = TokenizerTables::from_file(&dir.join("tokenizer.json"), 256)?.decode(&tokens);
    Ok(String::from_utf8_lossy(&text).trim().to_string())
}

/// The language model in `dir`'s answer to `question` after the fixed `system` prompt (none
/// when empty; a voice project's asks for one short sentence), without reasoning, greedily,
/// at most `limit` tokens. `encode` cuts a text into tokens (and adds the special ones when
/// told to), `decode` joins them back and `token_id` names one.
#[allow(clippy::too_many_arguments)]
pub fn answer(dir: &Path, system: &str, question: &str, arithmetic: Arithmetic, limit: usize, encode: &Encode, decode: &Decode, token_id: &dyn Fn(&str) -> Option<u32>) -> Result<String, String> {
    let family = compiler::compile_model(&std::fs::read_to_string(dir.join("config.json")).map_err(|e| e.to_string())?)?.family;
    let style = PromptStyle::read(&family, |name| std::fs::read_to_string(dir.join(name)).ok(), token_id);
    let (framed, add) = style.render_with_system(system.trim(), question, false)?;
    let mut model = Model::open(dir, arithmetic)?;
    let mut logits = Vec::new();
    for token in encode(&framed, add)? {
        logits = model.step(token)?;
    }
    let mut reply = Vec::new();
    for _ in 0..limit {
        let next = argmax(&logits);
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
/// template writes after the last answer's text: its end and the new turn
/// (`PromptStyle::next_turn`), as a chip that holds a conversation is sent it.
pub struct Conversation {
    model: Model,
    style: PromptStyle,
    messages: Vec<(String, String)>,
    /// Every token the cache holds.
    tokens: Vec<u32>,
    pub arithmetic: Arithmetic,
    pub dir: std::path::PathBuf,
    /// The fixed system prompt it starts with, empty for none.
    pub system: String,
    /// The most tokens the cache holds, as the chip's context; zero for no limit. A question
    /// that does not fit is refused, and an answer ends when the cache is full.
    pub context: usize,
}

impl Conversation {
    /// A conversation with the model in `dir` that starts with the fixed `system` prompt, as
    /// a chip that carries one runs it before the first question (none when it is empty).
    pub fn new(dir: &Path, arithmetic: Arithmetic, system: &str, token_id: &dyn Fn(&str) -> Option<u32>) -> Result<Self, String> {
        let family = compiler::compile_model(&std::fs::read_to_string(dir.join("config.json")).map_err(|e| e.to_string())?)?.family;
        let style = PromptStyle::read(&family, |name| std::fs::read_to_string(dir.join(name)).ok(), token_id);
        let system = system.trim().to_string();
        let messages = if system.is_empty() { Vec::new() } else { vec![("system".to_string(), system.clone())] };
        Ok(Self { model: Model::open(dir, arithmetic)?, style, messages, tokens: Vec::new(), arithmetic, dir: dir.to_path_buf(), system, context: 0 })
    }

    /// Questions asked so far.
    pub fn turns(&self) -> usize {
        (self.messages.len() - usize::from(!self.system.is_empty())) / 2
    }

    /// Every token the cache holds, in order.
    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    /// The answer to `question`, without reasoning, greedily, at most `limit` tokens, after the
    /// system prompt and every earlier question and answer; with the next token's scores. How
    /// long it is, the system prompt says.
    pub fn ask(&mut self, question: &str, limit: usize, encode: &Encode, decode: &Decode) -> Result<(String, Vec<f32>), String> {
        // On top of the cache, as a host sends a chip that holds the conversation: what the
        // template writes after the last answer's text (its end) and the new turn.
        let new = if self.tokens.is_empty() {
            let (framed, add) = self.style.render_with_system(&self.system, question, false)?;
            encode(&framed, add)?
        } else {
            encode(&self.style.next_turn(&self.system, question, false)?, false)?
        };
        if self.context > 0 && self.tokens.len() + new.len() >= self.context {
            return Err(if self.tokens.is_empty() {
                format!("the prompt takes {} tokens and the chip remembers {}: raise the design's context", new.len(), self.context)
            } else {
                format!("the conversation takes {} tokens and the question {} more, and the chip remembers {}: start a new conversation", self.tokens.len(), new.len(), self.context)
            });
        }
        let mut logits = Vec::new();
        for token in &new {
            logits = self.model.step(*token)?;
        }
        self.tokens.extend(&new);
        let mut reply = Vec::new();
        for _ in 0..limit {
            let next = argmax(&logits);
            if self.style.stop.contains(&next) || (self.context > 0 && self.tokens.len() + reply.len() >= self.context) {
                break;
            }
            reply.push(next);
            logits = self.model.step(next)?;
        }
        self.tokens.extend(&reply);
        let said = decode(&reply)?.trim().to_string();
        self.messages.push(("user".into(), question.to_string()));
        self.messages.push(("assistant".into(), said.clone()));
        Ok((said, logits))
    }
}

/// The highest score's token, the first of equals: a greedy pick.
fn argmax(values: &[f32]) -> u32 {
    values.iter().enumerate().fold((0, f32::NEG_INFINITY), |best, (i, v)| if *v > best.1 { (i, *v) } else { best }).0 as u32
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

#[cfg(feature = "speak")]
/// Frames the sound decoder keeps from each window it decodes, on the die: a window is
/// this plus a frame's reach on both sides (see `DacConfig::windows`).
pub const CHUNK_FRAMES: usize = 16;

#[cfg(feature = "speak")]
/// Words of a speaker profile the speech writer reads before each sentence: OuteTTS's own
/// profiles have about 26, some 1,800 tokens; the first three keep the voice in about 200.
pub const PROFILE_WORDS: usize = 3;

#[cfg(feature = "speak")]
/// The voice OuteTTS imitates, if its folder keeps one: the first profile in `speakers/`,
/// its first `PROFILE_WORDS` words.
pub fn speaker_of(tts: &Path) -> Result<Option<Speaker>, String> {
    let Ok(entries) = std::fs::read_dir(tts.join("speakers")) else { return Ok(None) };
    let mut profiles: Vec<_> = entries.filter_map(Result::ok).map(|e| e.path()).filter(|p| p.extension().is_some_and(|e| e == "json")).collect();
    profiles.sort();
    Ok(profiles.first().map(|p| Speaker::read(p)).transpose()?.map(|s| s.first_words(PROFILE_WORDS)))
}

#[cfg(feature = "speak")]
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
            (None, Arithmetic::Die(bits)) => crate::speak::dac_chip::decode_chunked(dac, &codes, CHUNK_FRAMES, bits)?.0,
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
