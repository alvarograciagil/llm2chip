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
use sim::voice::{answer, listen, speak, Arithmetic};
use std::path::Path;
use std::time::Instant;
use tokenizers::Tokenizer;

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [hear, think, tts, dac, question, out, rest @ ..] = args.as_slice() else {
        return Err("usage: talk MOONSHINE LLM OUTETTS DAC question.wav answer.wav [die BITS]".into());
    };
    let arithmetic = match rest.iter().position(|r| r == "die").and_then(|i| rest.get(i + 1)).and_then(|b| b.parse().ok()) {
        Some(bits) => Arithmetic::Die(bits),
        None => Arithmetic::Float,
    };
    let (hear, think, tts, dac) = (Path::new(hear), Path::new(think), Path::new(tts), Path::new(dac));
    println!("every stage in {}", match arithmetic {
        Arithmetic::Float => "floating point".to_string(),
        Arithmetic::Die(bits) => format!("the die's arithmetic, {bits}-bit vectors"),
    });

    let started = Instant::now();
    let clip = Clip::read(Path::new(question))?;
    let heard = listen(hear, &clip, arithmetic)?;
    println!("1. heard ({:.1} s of audio, {:.1} s): “{heard}”", clip.samples.len() as f32 / clip.rate as f32, started.elapsed().as_secs_f32());

    let started = Instant::now();
    let tokenizer = Tokenizer::from_file(think.join("tokenizer.json")).map_err(|e| e.to_string())?;
    let encode = |text: &str, add: bool| tokenizer.encode(text, add).map(|e| e.get_ids().to_vec()).map_err(|e| e.to_string());
    let decode = |tokens: &[u32]| tokenizer.decode(tokens, true).map_err(|e| e.to_string());
    let said = answer(think, &heard, arithmetic, 60, &encode, &decode, &|t| tokenizer.token_to_id(t))?;
    println!("2. answered ({:.1} s): “{said}”", started.elapsed().as_secs_f32());

    let started = Instant::now();
    let (sound, tokens) = speak(tts, dac, &said, arithmetic)?;
    sound.write(Path::new(out))?;
    println!("3. spoke ({tokens} tokens, {:.2} s of sound, {:.1} s) to {out}", sound.samples.len() as f32 / sound.rate as f32, started.elapsed().as_secs_f32());
    Ok(())
}
