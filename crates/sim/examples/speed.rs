//! Compares simulation speed on the CPU and the GPU, on real weights.
//! `cargo run --release -p sim --example speed -- <folder> [tokens]`, with `BITS` for the exact mode.

use compiler::compile_qwen3_5;
use sim::{Engine, ExactEngine, Numeric, Weights};
use std::path::Path;
use std::time::Instant;
use tokenizers::Tokenizer;

const PROMPT: &str = "Explain step by step how a CPU cache works, why it speeds up programs, and give a short example in C.";

fn best(logits: &[f32]) -> u32 {
    logits.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0 as u32
}

fn open(dir: &Path, gpu: bool) -> Result<Weights, String> {
    let weights = Weights::open(dir)?;
    Ok(if gpu { weights.with_gpu() } else { weights })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let dir = Path::new(&args[1]);
    let count: usize = args.get(2).map_or(8, |n| n.parse().unwrap());
    let bits: u32 = std::env::var("BITS").map_or(16, |v| v.parse().unwrap());
    let config = std::fs::read_to_string(dir.join("config.json"))?;
    let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| e.to_string())?;
    let text = format!("<|im_start|>user\n{PROMPT}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
    let ids = tokenizer.encode(text, false).map_err(|e| e.to_string())?.get_ids().to_vec();
    if let Some(name) = open(dir, true)?.accelerator_name() {
        println!("GPU: {name}");
    }

    println!("| mode | s/token | matches the CPU |\n|---|---|---|");
    let mut float_cpu: Vec<Vec<f32>> = Vec::new();
    let mut sequence = ids.clone();
    for gpu in [false, true] {
        let mut engine = Engine::new(compile_qwen3_5(&config)?, open(dir, gpu)?, Numeric::Float)?;
        let started = Instant::now();
        let (mut outputs, mut matches) = (0, 0);
        for i in 0..ids.len() + count - 1 {
            let token = sequence[i];
            if let Some(l) = engine.step(token, i + 1 >= ids.len())?.0 {
                if !gpu {
                    float_cpu.push(l.to_vec());
                    if sequence.len() == i + 1 {
                        sequence.push(best(l));
                    }
                } else {
                    matches += (best(l) == best(&float_cpu[outputs])) as usize;
                }
                outputs += 1;
            }
        }
        let time = started.elapsed().as_secs_f64() / (ids.len() + count - 1) as f64;
        println!("| floating point, {} | {time:.3} | {} |", if gpu { "GPU" } else { "CPU" }, if gpu { format!("{matches} of {outputs} in top-1") } else { "referencia".into() });
    }

    let mut exact_cpu: Vec<Vec<f32>> = Vec::new();
    for gpu in [false, true] {
        let mut engine = ExactEngine::new(compile_qwen3_5(&config)?, open(dir, gpu)?, bits)?;
        let started = Instant::now();
        let (mut outputs, mut matches) = (0, 0);
        for (i, token) in sequence.iter().enumerate().take(ids.len() + count - 1) {
            if let Some(l) = engine.step(*token, i + 1 >= ids.len())? {
                if gpu {
                    matches += (l == exact_cpu[outputs]) as usize;
                } else {
                    exact_cpu.push(l.clone());
                }
                outputs += 1;
            }
        }
        let time = started.elapsed().as_secs_f64() / (ids.len() + count - 1) as f64;
        println!("| exacto a {bits} bits, {} | {time:.3} | {} |", if gpu { "GPU" } else { "CPU" }, if gpu { format!("{matches} of {outputs} with identical logits") } else { "referencia".into() });
    }
    Ok(())
}
