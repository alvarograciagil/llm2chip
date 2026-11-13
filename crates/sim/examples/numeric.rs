//! Measures how much fixed-point precision the model tolerates, with teacher forcing.
//! `cargo run --release -p sim --example numeric -- <folder> [tokens] [bits,state,tables ...]`

use compiler::compile_qwen3_5;
use sim::{Engine, Numeric, Weights};
use std::path::Path;
use tokenizers::Tokenizer;

const PROMPT: &str = "Explain step by step how a CPU cache works, why it speeds up programs, and give a short example in C that shows the effect of cache misses.";

fn softmax(logits: &[f32]) -> Vec<f64> {
    let top = logits.iter().copied().fold(f32::MIN, f32::max) as f64;
    let raw: Vec<f64> = logits.iter().map(|v| (*v as f64 - top).exp()).collect();
    let sum: f64 = raw.iter().sum();
    raw.iter().map(|v| v / sum).collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let dir = Path::new(&args[1]);
    let count: usize = args.get(2).map_or(40, |n| n.parse().unwrap());
    let config = std::fs::read_to_string(dir.join("config.json"))?;
    let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| e.to_string())?;
    let text = format!("<|im_start|>user\n{PROMPT}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
    let ids = tokenizer.encode(text, false).map_err(|e| e.to_string())?.get_ids().to_vec();

    let mut engine = Engine::new(compile_qwen3_5(&config)?, Weights::open(dir)?, Numeric::Float)?;
    let mut sequence = ids.clone();
    let mut reference: Vec<Vec<f64>> = Vec::new();
    for i in 0..ids.len() + count - 1 {
        let token = sequence[i];
        let (logits, _) = engine.step(token, i + 1 >= ids.len())?;
        if let Some(l) = logits {
            let p = softmax(l);
            let next = p.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0 as u32;
            reference.push(p);
            if sequence.len() == i + 1 {
                sequence.push(next);
            }
        }
    }
    println!("activation state | top1_agreement mean_KL max_KL saturations");

    let mut configs = vec![(16, 16, 16), (16, 16, 12), (16, 16, 10), (16, 16, 8), (16, 16, 6), (14, 16, 10), (14, 14, 10), (14, 12, 8)];
    // Extra arguments such as `20,20,12` replace the list of configurations.
    if args.len() > 3 {
        configs = args[3..].iter().map(|a| { let v: Vec<u32> = a.split(',').map(|n| n.parse().unwrap()).collect(); (v[0], v[1], v[2]) }).collect();
    }
    for (ab, sb, tb) in configs {
        let numeric = Numeric::Block { activation_bits: ab, state_bits: sb, table_bits: Some(tb) };
        let mut engine = Engine::new(compile_qwen3_5(&config)?, Weights::open(dir)?, numeric)?;
        let (mut agree, mut kl_sum, mut kl_max, mut saturated, mut seen) = (0, 0f64, 0f64, 0u64, 0usize);
        for (i, token) in sequence.iter().enumerate().take(ids.len() + count - 1) {
            let (logits, stats) = engine.step(*token, i + 1 >= ids.len())?;
            saturated += stats.saturated;
            if let Some(l) = logits {
                let p = softmax(l);
                let reference = &reference[seen];
                seen += 1;
                let top = |d: &[f64]| d.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
                if top(&p) == top(reference) {
                    agree += 1;
                }
                let kl: f64 = reference.iter().zip(&p).filter(|(r, _)| **r > 1e-12).map(|(r, q)| r * (r / q.max(1e-30)).ln()).sum();
                kl_sum += kl;
                kl_max = kl_max.max(kl);
            }
        }
        println!("block {ab} bits, state {sb} bits, tables {tb} bits | {:.1}% {:.4} {:.3} {saturated}", 100.0 * agree as f64 / seen as f64, kl_sum / seen as f64, kl_max);
    }
    Ok(())
}
