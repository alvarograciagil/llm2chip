//! Continues a text with any supported model, in floating point and choosing the most
//! likely token each time.
//!
//! `cargo run --release -p sim --example complete -- <folder> "<text>" [tokens]`

use sim::{Engine, Numeric, Weights};
use std::path::Path;
use tokenizers::Tokenizer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let dir = Path::new(&args[1]);
    let count: usize = args.get(3).map_or(48, |n| n.parse().unwrap());
    let program = compiler::compile_model(&std::fs::read_to_string(dir.join("config.json"))?)?;
    let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| e.to_string())?;
    let ids = tokenizer.encode(args[2].as_str(), true).map_err(|e| e.to_string())?.get_ids().to_vec();
    let mut engine = Engine::new(program, Weights::open(dir)?, Numeric::Float)?;
    let mut next = 0;
    for (i, id) in ids.iter().enumerate() {
        if let (Some(logits), _) = engine.step(*id, i + 1 == ids.len())? {
            next = argmax(logits);
        }
    }
    let mut output = Vec::new();
    for _ in 0..count {
        output.push(next);
        next = argmax(engine.step(next, true)?.0.unwrap());
    }
    println!("{}{}", args[2], tokenizer.decode(&output, true).map_err(|e| e.to_string())?);
    Ok(())
}

fn argmax(values: &[f32]) -> u32 {
    values.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map_or(0, |(i, _)| i as u32)
}
