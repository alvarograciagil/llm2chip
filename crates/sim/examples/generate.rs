//! `cargo run --release -p sim --example generate -- <folder> "<text>" [tokens]`

use compiler::compile_qwen3_5;
use sim::{Engine, Numeric, Weights};
use std::path::Path;
use tokenizers::Tokenizer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let dir = Path::new(&args[1]);
    let count: usize = args.get(3).map_or(16, |n| n.parse().unwrap());
    let config = std::fs::read_to_string(dir.join("config.json"))?;
    let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| e.to_string())?;
    let prompt = format!("<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", args[2]);
    let ids = tokenizer.encode(prompt, false).map_err(|e| e.to_string())?.get_ids().to_vec();

    let program = compile_qwen3_5(&config)?;
    let mut engine = Engine::new(program, Weights::open(dir)?, Numeric::Float)?;
    println!("Prompt: {} tokens", ids.len());

    let mut last = 0;
    let mut output = Vec::new();
    for (i, id) in ids.iter().enumerate() {
        let (logits, stats) = engine.step(*id, i + 1 == ids.len())?;
        eprintln!("prefill {i}: {:.2} s", stats.seconds);
        if let Some(l) = logits {
            last = argmax(l);
        }
    }
    for _ in 0..count {
        output.push(last);
        if last == 248046 || last == 248044 {
            break;
        }
        let (logits, stats) = engine.step(last, true)?;
        last = argmax(logits.unwrap());
        eprintln!("token: {:.2} s", stats.seconds);
    }
    println!("ids: {output:?}");
    println!("{}", tokenizer.decode(&output, false).map_err(|e| e.to_string())?);
    Ok(())
}

fn argmax(values: &[f32]) -> u32 {
    values.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map_or(0, |(i, _)| i as u32)
}
