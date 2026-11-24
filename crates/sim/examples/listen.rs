//! Speech to text with Moonshine: in floating point and in the die's arithmetic, optionally
//! checked against what Transformers makes of the same clip.
//!
//!     cargo run --release -p sim --example listen -- MODEL_DIR clip.wav [golden.json] [bits]
//!
//! `golden.json` is what `tools/golden/moonshine.py` writes; `bits` the die's vectors (16).

use sim::audio::Clip;
use sim::moonshine::Moonshine;
use sim::moonshine_chip::MoonshineChip;
use sim::tokenizer::TokenizerTables;
use std::path::Path;

fn relative(a: &[f32], b: &[f32]) -> f32 {
    let largest = b.iter().fold(0f32, |m, v| m.max(v.abs()));
    a.iter().zip(b).fold(0f32, |m, (x, y)| m.max((x - y).abs())) / largest
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [dir, wav, rest @ ..] = args.as_slice() else {
        return Err("usage: listen MODEL_DIR clip.wav [golden.json] [bits]".into());
    };
    let dir = Path::new(dir);
    let clip = Clip::read(Path::new(wav))?.resampled(16_000);
    let golden: Option<serde_json::Value> = rest.first().filter(|r| r.ends_with(".json")).map(|p| std::fs::read_to_string(p).map_err(|e| e.to_string()).and_then(|t| serde_json::from_str(&t).map_err(|e| e.to_string()))).transpose()?;
    let bits: u32 = rest.iter().find_map(|r| r.parse().ok()).unwrap_or(16);
    let tokenizer = TokenizerTables::from_file(&dir.join("tokenizer.json"), 256)?;
    let text = |tokens: &[u32]| String::from_utf8_lossy(&tokenizer.decode(tokens)).into_owned();
    println!("{:.2} s of audio, {} frames", clip.samples.len() as f32 / 16_000.0, sim::moonshine::MoonshineConfig::frames(clip.samples.len()));

    let model = Moonshine::open(dir)?;
    let started = std::time::Instant::now();
    let tokens = model.transcribe(&clip.samples)?;
    println!("floating point ({:.1} s): {tokens:?} “{}”", started.elapsed().as_secs_f32(), text(&tokens));

    if let Some(golden) = &golden {
        let frames: Vec<Vec<f32>> = serde_json::from_value(golden["frames"].clone()).map_err(|e| e.to_string())?;
        let expected: Vec<u32> = serde_json::from_value(golden["tokens"].clone()).map_err(|e| e.to_string())?;
        let scores: Vec<Vec<f32>> = serde_json::from_value(golden["scores"].clone()).map_err(|e| e.to_string())?;
        let ours = model.encode(&clip.samples)?;
        let worst = ours.iter().zip(&frames).map(|(a, b)| relative(a, b)).fold(0f32, f32::max);
        println!("Transformers: {expected:?} “{}”; encoder frames within {:.4} %", text(&expected), worst * 100.0);
        let mut decoder = model.decoder(&ours)?;
        let mut token = model.config.start;
        for (step, reference) in scores.iter().enumerate() {
            let got = decoder.step(token)?;
            println!("  step {step}: scores within {:.4} %", relative(&got, reference) * 100.0);
            token = expected.get(step).copied().unwrap_or(model.config.end);
        }
        if tokens != expected {
            return Err("the reference does not write what Transformers writes".into());
        }
    }

    let started = std::time::Instant::now();
    let mut chip = MoonshineChip::exact(dir, clip.samples.len(), bits)?;
    let on_chip = chip.transcribe(&clip.samples)?;
    println!("die's arithmetic, {bits}-bit vectors ({:.1} s): {on_chip:?} “{}”", started.elapsed().as_secs_f32(), text(&on_chip));
    if on_chip != tokens {
        println!("  differs from floating point");
    }
    Ok(())
}
