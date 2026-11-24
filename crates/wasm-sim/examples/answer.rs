//! The browser's simulator run natively, to compare with its WebAssembly build:
//! `cargo run --release -p llm2chip-wasm-sim --example answer -- <folder> "<text>" [tokens]`
//! prints the answer's token ids and the median seconds per token.

use llm2chip_wasm_sim::{Generator, Request, Setup};
use std::collections::HashMap;
use std::path::Path;

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let dir = Path::new(&args[1]);
    let tokens: usize = args.get(3).map_or(16, |n| n.parse().unwrap());
    let texts: HashMap<String, String> = ["config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json", "chat_template.jinja"].into_iter().filter_map(|n| Some((n.to_string(), std::fs::read_to_string(dir.join(n)).ok()?))).collect();
    let mut names: Vec<_> = std::fs::read_dir(dir).map_err(|e| e.to_string())?.filter_map(Result::ok).map(|e| e.path()).filter(|p| p.extension().is_some_and(|e| e == "safetensors")).collect();
    names.sort();
    let weights = names.iter().map(|p| std::fs::read(p).map(Vec::into_boxed_slice).map_err(|e| e.to_string())).collect::<Result<_, _>>()?;
    let mut generator = Generator::new(&texts, weights, &Setup { activation_bits: 16, format: None, context_tokens: 512 })?;
    generator.start(Request { prompt: args[2].clone(), max_tokens: tokens, temperature: 0.0, top_k: 1, system: String::new(), thinking: false, think_budget: None, seed: 7 })?;
    let (mut ids, mut seconds) = (Vec::new(), Vec::new());
    loop {
        let event = generator.advance()?;
        match event["type"].as_str() {
            Some("token") => {
                ids.push(event["id"].as_u64().unwrap());
                seconds.push(event["seconds"].as_f64().unwrap());
            }
            Some("done") => break,
            _ => {}
        }
    }
    seconds.sort_by(f64::total_cmp);
    println!("ids {ids:?}");
    println!("median {:.3} s/token", seconds.get(seconds.len() / 2).copied().unwrap_or(0.0));
    Ok(())
}
