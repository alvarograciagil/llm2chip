//! A spoken conversation turn through three programs, as a system of three stages would
//! run them on dies of their own: speech to text (Moonshine), an answer (any language
//! model of the catalogue), and text to speech (OuteTTS and DAC). Only text passes from
//! one stage to the next.
//!
//!     cargo run --release -p sim --example talk -- MOONSHINE LLM OUTETTS DAC question.wav answer.wav [die BITS]
//!
//! With `die BITS` every stage runs in the die's arithmetic with vectors of BITS bits (20
//! or more for the speech decoder: see docs/voice.md); without it, in floating point.

use sim::audio::Clip;
use sim::moonshine::Moonshine;
use sim::moonshine_chip::MoonshineChip;
use sim::prompt::PromptStyle;
use sim::speak::{write, SpeechVocabulary};
use sim::tokenizer::TokenizerTables;
use sim::{Engine, ExactEngine, Numeric, Weights};
use std::path::Path;
use std::time::Instant;
use tokenizers::Tokenizer;

/// A language model in floating point or in the die's arithmetic: a token in, the next
/// one's scores out.
enum Model {
    Float(Box<Engine>),
    Die(Box<ExactEngine>),
}

impl Model {
    fn open(dir: &Path, die: Option<u32>) -> Result<Self, String> {
        let program = compiler::compile_model(&std::fs::read_to_string(dir.join("config.json")).map_err(|e| e.to_string())?)?;
        Ok(match die {
            None => Self::Float(Box::new(Engine::new(program, Weights::open(dir)?, Numeric::Float)?)),
            Some(bits) => Self::Die(Box::new(ExactEngine::new(program, Weights::open(dir)?, bits)?)),
        })
    }

    fn step(&mut self, token: u32) -> Result<Vec<f32>, String> {
        match self {
            Self::Float(engine) => Ok(engine.step(token, true)?.0.ok_or("no scores")?.to_vec()),
            Self::Die(engine) => engine.step(token, true)?.ok_or_else(|| "no scores".into()),
        }
    }
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [listen, think, tts, dac, question, answer, rest @ ..] = args.as_slice() else {
        return Err("usage: talk MOONSHINE LLM OUTETTS DAC question.wav answer.wav [die BITS]".into());
    };
    let die: Option<u32> = rest.iter().position(|r| r == "die").and_then(|i| rest.get(i + 1)).and_then(|b| b.parse().ok());
    let (listen, think, tts, dac) = (Path::new(listen), Path::new(think), Path::new(tts), Path::new(dac));
    let arithmetic = die.map_or("floating point".to_string(), |b| format!("the die's arithmetic, {b}-bit vectors"));
    println!("every stage in {arithmetic}");

    // Speech to text.
    let started = Instant::now();
    let clip = Clip::read(Path::new(question))?.resampled(16_000);
    let tokens = match die {
        None => Moonshine::open(listen)?.transcribe(&clip.samples)?,
        Some(bits) => MoonshineChip::exact(listen, clip.samples.len(), bits)?.transcribe(&clip.samples)?,
    };
    let heard = String::from_utf8_lossy(&TokenizerTables::from_file(&listen.join("tokenizer.json"), 256)?.decode(&tokens)).trim().to_string();
    println!("1. heard ({:.1} s of audio, {:.1} s): “{heard}”", clip.samples.len() as f32 / 16_000.0, started.elapsed().as_secs_f32());

    // The answer.
    let started = Instant::now();
    let tokenizer = Tokenizer::from_file(think.join("tokenizer.json")).map_err(|e| e.to_string())?;
    let family = compiler::compile_model(&std::fs::read_to_string(think.join("config.json")).map_err(|e| e.to_string())?)?.family;
    let style = PromptStyle::read(&family, |name| std::fs::read_to_string(think.join(name)).ok(), |t| tokenizer.token_to_id(t));
    let (framed, add) = style.render(&format!("{heard}\nAnswer in one short sentence."), false)?;
    let prompt = tokenizer.encode(framed, add).map_err(|e| e.to_string())?.get_ids().to_vec();
    let mut model = Model::open(think, die)?;
    let mut logits = Vec::new();
    for token in &prompt {
        logits = model.step(*token)?;
    }
    let mut reply = Vec::new();
    for _ in 0..60 {
        let next = sim::moonshine::argmax(&logits);
        if style.stop.contains(&next) {
            break;
        }
        reply.push(next);
        logits = model.step(next)?;
    }
    let said = tokenizer.decode(&reply, true).map_err(|e| e.to_string())?.trim().to_string();
    println!("2. answered ({} tokens, {:.1} s): “{said}”", reply.len(), started.elapsed().as_secs_f32());

    // Text to speech.
    let started = Instant::now();
    let vocabulary = SpeechVocabulary::read(tts)?;
    let mut text = TokenizerTables::from_file(&tts.join("tokenizer.json"), 1024)?;
    text.set_add_special(false);
    let prompt = vocabulary.prompt(&text.encode(&said)?);
    let mut writer = Model::open(tts, die)?;
    let written = write(&vocabulary, &prompt, 1500, &mut |token| writer.step(token))?;
    let codes = vocabulary.codes(&written);
    let sound = match die {
        None => sim::dac::Dac::open(dac)?.decode(&codes)?,
        Some(bits) => sim::dac_chip::DacChip::exact(dac, codes[0].len(), bits)?.decode(&codes)?.0,
    };
    let rate = sim::dac::DacConfig::read(dac)?.sampling_rate;
    Clip { rate, samples: sound }.write(Path::new(answer))?;
    println!("3. spoke ({} tokens, {:.2} s of sound, {:.1} s) to {answer}", written.len(), codes[0].len() as f32 / 75.0, started.elapsed().as_secs_f32());
    Ok(())
}
