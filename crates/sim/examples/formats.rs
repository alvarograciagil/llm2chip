//! Measures how much each weight format costs in output quality, on real weights.
//! `cargo run --release -p sim --example formats -- <carpeta> <tokens> q8 q4_group fp16 ...`

use compiler::compile_qwen3_5;
use netlist::format::{WeightCoding, WeightFormat};
use sim::{Engine, Numeric, Weights};
use std::path::Path;
use std::time::Instant;
use tokenizers::Tokenizer;

const PROMPT: &str = "Explain step by step how a CPU cache works, why it speeds up programs, and give a short example in C that shows the effect of cache lines.";

fn parse(name: &str) -> Option<WeightFormat> {
    let digits = |text: &str| text.parse::<u32>().ok();
    let plain = |coding, bits, exponent_bits| Some(WeightFormat { coding, bits, exponent_bits, bias: true, group_size: 0, keep_source: false });
    let grouped = |coding, bits, exponent_bits, bias, group_size| Some(WeightFormat { coding, bits, exponent_bits, bias, group_size, keep_source: false });
    match name {
        "fp8" => plain(WeightCoding::Float, 8, 4),
        "fp16" => plain(WeightCoding::Float, 16, 5),
        "bf16" => plain(WeightCoding::Float, 16, 8),
        "fp24" => plain(WeightCoding::Float, 24, 7),
        "fp32" => plain(WeightCoding::Float, 32, 8),
        "mxfp4" => grouped(WeightCoding::Float, 4, 2, false, 32),
        "mxfp6" => grouped(WeightCoding::Float, 6, 2, false, 32),
        "mxfp8" => grouped(WeightCoding::Float, 8, 4, false, 32),
        "posit8" => plain(WeightCoding::Posit, 8, 1),
        "posit16" => plain(WeightCoding::Posit, 16, 1),
        "log8" => plain(WeightCoding::Log, 8, 4),
        "log16" => plain(WeightCoding::Log, 16, 8),
        "book4" => plain(WeightCoding::Codebook, 4, 0),
        "book8" => plain(WeightCoding::Codebook, 8, 0),
        _ => {
            if let Some(bits) = name.strip_prefix("q").and_then(digits) {
                return plain(WeightCoding::Fixed, bits, 0);
            }
            if let Some(bits) = name.strip_prefix("gguf").and_then(digits) {
                return grouped(WeightCoding::Affine, bits, 0, false, 32);
            }
            if let Some(bits) = name.strip_prefix("mlx").and_then(digits) {
                return grouped(WeightCoding::Affine, bits, 0, true, 32);
            }
            None
        }
    }
}

fn softmax(logits: &[f32]) -> Vec<f64> {
    let top = logits.iter().copied().fold(f32::MIN, f32::max) as f64;
    let raw: Vec<f64> = logits.iter().map(|v| (*v as f64 - top).exp()).collect();
    let sum: f64 = raw.iter().sum();
    raw.iter().map(|v| v / sum).collect()
}

fn best(p: &[f64]) -> u32 {
    p.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0 as u32
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let dir = Path::new(&args[1]);
    let count: usize = args[2].parse()?;
    let config = std::fs::read_to_string(dir.join("config.json"))?;
    let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| e.to_string())?;
    let text = format!("<|im_start|>user\n{PROMPT}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
    let ids = tokenizer.encode(text, false).map_err(|e| e.to_string())?.get_ids().to_vec();

    let mut engine = Engine::new(compile_qwen3_5(&config)?, Weights::open(dir)?, Numeric::Float)?;
    let mut sequence = ids.clone();
    let mut reference = Vec::new();
    for i in 0..ids.len() + count - 1 {
        let token = sequence[i];
        if let Some(l) = engine.step(token, i + 1 >= ids.len())?.0 {
            let p = softmax(l);
            if sequence.len() == i + 1 {
                sequence.push(best(&p));
            }
            reference.push(p);
        }
    }

    println!("| formato | top-1 | KL media | s/token |\n|---|---|---|---|");
    for name in &args[3..] {
        let format = parse(name).ok_or_else(|| format!("formato desconocido: {name}"))?;
        let weights = Weights::open(dir)?.with_format(Some(format));
        let mut engine = Engine::new(compile_qwen3_5(&config)?, weights, Numeric::Float)?;
        let started = Instant::now();
        let (mut matches, mut kl, mut index) = (0, 0.0, 0);
        for (i, token) in sequence.iter().enumerate().take(ids.len() + count - 1) {
            if let Some(l) = engine.step(*token, i + 1 >= ids.len())?.0 {
                let p = softmax(l);
                matches += (best(&p) == best(&reference[index])) as usize;
                kl += reference[index].iter().zip(&p).filter(|(a, _)| **a > 1e-12).map(|(a, b)| a * (a / b.max(1e-30)).ln()).sum::<f64>();
                index += 1;
            }
        }
        println!("| {name} | {:.1} % | {:.4} | {:.1} |", 100.0 * matches as f64 / count as f64, kl / count as f64, started.elapsed().as_secs_f64() / (sequence.len() - 1) as f64);
    }
    Ok(())
}
