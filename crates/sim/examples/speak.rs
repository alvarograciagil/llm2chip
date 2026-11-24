//! Text to speech with OuteTTS 1.0 and DAC: in floating point, or with both in the die's
//! arithmetic, optionally checked against what `tools/golden/speak.py` wrote.
//!
//!     cargo run --release -p sim --example speak -- OUTETTS_DIR DAC_DIR "Text." out.wav [golden.json] [die BITS]
//!
//! `die BITS` runs the language model and the decoder in the die's arithmetic with vectors
//! of BITS bits (20 or more for the decoder: see docs/voice.md).

use sim::audio::Clip;
use sim::dac::Dac;
use sim::dac_chip::DacChip;
use sim::speak::{write, SpeechVocabulary};
use sim::tokenizer::TokenizerTables;
use sim::{Engine, ExactEngine, Numeric, Weights};
use std::path::Path;

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [tts, dac_dir, text, out, rest @ ..] = args.as_slice() else {
        return Err("usage: speak OUTETTS_DIR DAC_DIR \"Text.\" out.wav [golden.json] [die BITS]".into());
    };
    let (tts, dac_dir) = (Path::new(tts), Path::new(dac_dir));
    let golden: Option<serde_json::Value> = rest.iter().find(|r| r.ends_with(".json")).map(|p| std::fs::read_to_string(p).map_err(|e| e.to_string()).and_then(|t| serde_json::from_str(&t).map_err(|e| e.to_string()))).transpose()?;
    let die: Option<u32> = rest.iter().position(|r| r == "die").and_then(|i| rest.get(i + 1)).and_then(|b| b.parse().ok());

    let vocabulary = SpeechVocabulary::read(tts)?;
    let mut tokenizer = TokenizerTables::from_file(&tts.join("tokenizer.json"), 1024)?;
    tokenizer.set_add_special(false);
    let prompt = vocabulary.prompt(&tokenizer.encode(text)?);
    if let Some(golden) = &golden {
        let expected: Vec<u32> = serde_json::from_value(golden["prompt"].clone()).map_err(|e| e.to_string())?;
        println!("prompt: {} tokens, {}", prompt.len(), if prompt == expected { "as Transformers'" } else { "NOT as Transformers'" });
    }

    let config = std::fs::read_to_string(tts.join("config.json")).map_err(|e| e.to_string())?;
    let program = compiler::compile_model(&config)?;
    let started = std::time::Instant::now();
    let mut first = Vec::new();
    let tokens = match die {
        None => {
            let mut engine = Engine::new(program, Weights::open(tts)?, Numeric::Float)?;
            write(&vocabulary, &prompt, 600, &mut |token| {
                let logits = engine.step(token, true)?.0.ok_or("no scores")?.to_vec();
                first.push(logits.clone());
                Ok(logits)
            })?
        }
        Some(bits) => {
            let mut engine = ExactEngine::new(program, Weights::open(tts)?, bits)?;
            write(&vocabulary, &prompt, 600, &mut |token| {
                let logits = engine.step(token, true)?.ok_or("no scores")?;
                first.push(logits.clone());
                Ok(logits)
            })?
        }
    };
    let codes = vocabulary.codes(&tokens);
    let where_ = die.map_or("floating point".to_string(), |b| format!("the die's arithmetic, {b}-bit vectors"));
    println!("{where_}: {} tokens, {} frames ({:.2} s of sound) in {:.1} s", tokens.len(), codes[0].len(), codes[0].len() as f32 / 75.0, started.elapsed().as_secs_f32());
    if let Some(golden) = &golden {
        let expected: Vec<u32> = serde_json::from_value(golden["generated"].clone()).map_err(|e| e.to_string())?;
        let scores: Vec<Vec<f32>> = serde_json::from_value(golden["scores"].clone()).map_err(|e| e.to_string())?;
        let same = tokens.iter().zip(&expected).take_while(|(a, b)| a == b).count();
        println!("  the first {same} of {} tokens as Transformers'", expected.len());
        for (step, reference) in scores.iter().enumerate() {
            let got = &first[prompt.len() - 1 + step];
            let largest = reference.iter().fold(0f32, |m, v| m.max(v.abs()));
            let worst = got.iter().zip(reference).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
            println!("  step {step}: scores within {:.4} %", worst / largest * 100.0);
        }
    }

    let started = std::time::Instant::now();
    let sound = match die {
        None => Dac::open(dac_dir)?.decode(&codes)?,
        Some(bits) => DacChip::exact(dac_dir, codes[0].len(), bits)?.decode(&codes)?.0,
    };
    let rate = sim::dac::DacConfig::read(dac_dir)?.sampling_rate;
    Clip { rate, samples: sound }.write(Path::new(out))?;
    println!("decoded in {:.1} s to {out}", started.elapsed().as_secs_f32());
    Ok(())
}
