//! Compares the die's exact integer arithmetic with floating point, on real weights.
//! `cargo run --release -p sim --example exact -- <carpeta> [tokens]`

use compiler::compile_qwen3_5;
use sim::{Engine, ExactEngine, Numeric, Weights};
use std::path::Path;
use std::time::Instant;
use tokenizers::Tokenizer;

const PROMPT: &str = "Explain step by step how a CPU cache works, why it speeds up programs, and give a short example in C.";

fn best(logits: &[f32]) -> u32 {
    logits.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0 as u32
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let dir = Path::new(&args[1]);
    let count: usize = args.get(2).map_or(20, |n| n.parse().unwrap());
    let config = std::fs::read_to_string(dir.join("config.json"))?;
    let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| e.to_string())?;
    let text = format!("<|im_start|>user\n{PROMPT}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
    let ids = tokenizer.encode(text, false).map_err(|e| e.to_string())?.get_ids().to_vec();

    let mut float = Engine::new(compile_qwen3_5(&config)?, Weights::open(dir)?, Numeric::Float)?;
    let mut exact = ExactEngine::new(compile_qwen3_5(&config)?, Weights::open(dir)?)?;
    let mut sequence = ids.clone();
    let (mut matches, mut compared) = (0, 0);
    let (mut float_time, mut exact_time) = (0.0, 0.0);
    for i in 0..ids.len() + count - 1 {
        let token = sequence[i];
        let want = i + 1 >= ids.len();
        let started = Instant::now();
        let reference = float.step(token, want)?.0.map(best);
        float_time += started.elapsed().as_secs_f64();
        let started = Instant::now();
        let got = exact.step(token, want)?.map(|l| best(&l));
        exact_time += started.elapsed().as_secs_f64();
        if let (Some(r), Some(g)) = (reference, got) {
            compared += 1;
            matches += (r == g) as usize;
            if sequence.len() == i + 1 {
                sequence.push(r);
            }
        }
    }
    let tokens = sequence.len() - 1;
    println!("coincidencia top-1: {matches}/{compared}");
    println!("float: {:.2} s/token, enteros: {:.2} s/token", float_time / tokens as f64, exact_time / tokens as f64);
    println!("texto: {}", tokenizer.decode(&sequence[ids.len()..], true).map_err(|e| e.to_string())?);
    Ok(())
}
