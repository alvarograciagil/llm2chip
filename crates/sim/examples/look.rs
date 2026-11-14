//! Asks Janus about a picture: its SigLIP encoder reads it, the language model answers.
//!
//! `cargo run --release -p sim --example look -- <folder> image.bmp "<question>" [exact]`

use sim::janus::JanusEye;
use sim::{Engine, ExactEngine, Numeric, Runner, Weights};
use std::path::Path;
use tokenizers::Tokenizer;

/// A 24-bit BMP as RGB rows from the top.
fn read_bmp(path: &Path) -> (Vec<u8>, usize, usize) {
    let data = std::fs::read(path).unwrap();
    let at = |o: usize| u32::from_le_bytes(data[o..o + 4].try_into().unwrap());
    let (offset, width, height) = (at(10) as usize, at(18) as usize, at(22) as i32);
    let rows = height.unsigned_abs() as usize;
    let stride = (width * 3).next_multiple_of(4);
    let mut rgb = Vec::with_capacity(width * rows * 3);
    for r in 0..rows {
        let y = if height < 0 { r } else { rows - 1 - r };
        for x in 0..width {
            let p = offset + y * stride + x * 3;
            rgb.extend([data[p + 2], data[p + 1], data[p]]);
        }
    }
    (rgb, width, rows)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let dir = Path::new(&args[1]);
    let eye = JanusEye::open(dir)?.ok_or("not a Janus")?;
    let started = std::time::Instant::now();
    // With "-" for the image, only text: to tell the image's path from the model's.
    let vectors = if args[2] == "-" { Vec::new() } else {
        let (rgb, width, height) = read_bmp(Path::new(&args[2]));
        eye.encode(&rgb, width, height)?
    };
    eprintln!("{} image vectors in {:.0} s", vectors.len(), started.elapsed().as_secs_f64());
    if let Some(v) = vectors.first() {
        let top = v.iter().fold(0f32, |m, x| m.max(x.abs()));
        let mean = v.iter().map(|x| x.abs()).sum::<f32>() / v.len() as f32;
        let tops: Vec<f32> = vectors.iter().map(|v| v.iter().fold(0f32, |m, x| m.max(x.abs()))).collect();
        eprintln!("first vector: maximum {top:.3}, mean {mean:.4}; maxima of all between {:.2} and {:.2}", tops.iter().cloned().fold(f32::MAX, f32::min), tops.iter().cloned().fold(0f32, f32::max));
    }
    let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| e.to_string())?;
    let picture = if vectors.is_empty() { "" } else { "<begin_of_image><image_placeholder><end_of_image>\n" };
    let text = format!("<｜begin▁of▁sentence｜>You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.\n\n<|User|>: {picture}{}\n\n<|Assistant|>:", args[3]);
    let ids = tokenizer.encode(text.as_str(), false).map_err(|e| e.to_string())?.get_ids().to_vec();
    let config: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json"))?)?;
    let program = compiler::compile_model(&config["text_config"].to_string())?;
    let weights = Weights::open(dir)?;
    // With INYECTAR, text tokens too go in as vectors, through the same path as the image's.
    let inject = std::env::var("INJECT").is_ok();
    let table = program.ops.iter().find_map(|i| match &i.op { compiler::Op::EmbRead { table, .. } => Some(table.clone()), _ => None }).unwrap();
    let rows = Weights::open(dir)?;
    let hidden = program.hidden;
    // "exact" runs the chip's integers at 16 bits, "exact24" at 24.
    let mode = args.get(4).map(String::as_str).unwrap_or("");
    let mut engine: Box<dyn Runner> = match mode.strip_prefix("exact") {
        Some(bits) => Box::new(ExactEngine::new(program, weights, bits.parse().unwrap_or(16))?),
        None => Box::new(Engine::new(program, weights, Numeric::Float)?),
    };
    let mut position = 0u64;
    let mut logits = Vec::new();
    for (i, id) in ids.iter().enumerate() {
        let last = i + 1 == ids.len();
        if *id == eye.placeholder {
            for v in &vectors {
                engine.step_embedding(v, [position; 3], false)?;
                position += 1;
            }
            continue;
        }
        logits = if inject {
            let row = rows.matrix(&table)?.row(*id as usize, hidden)?;
            engine.step_embedding(&row, [position; 3], last)?.0.map(<[f32]>::to_vec).unwrap_or_default()
        } else {
            engine.step(*id, last)?.0.map(<[f32]>::to_vec).unwrap_or_default()
        };
        position += 1;
    }
    eprintln!("prompt read in {:.0} s", started.elapsed().as_secs_f64());
    let mut out = Vec::new();
    for _ in 0..24 {
        let next = logits.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| i as u32).unwrap();
        if next == 100_001 {
            break;
        }
        out.push(next);
        logits = engine.step(next, true)?.0.map(<[f32]>::to_vec).unwrap_or_default();
    }
    println!("{}", tokenizer.decode(&out, true).map_err(|e| e.to_string())?);
    Ok(())
}
