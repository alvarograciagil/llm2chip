//! Measures how often the MTP head guesses the following tokens correctly.
//! `cargo run --release -p sim --example mtp -- <folder> [tokens]`

use compiler::compile_qwen3_5;
use sim::{Engine, Numeric, Weights};
use std::path::Path;
use tokenizers::Tokenizer;

const PROMPT: &str = "Write a short Python function that reverses a string, then explain how it works in two sentences.";
const DEPTH: usize = 3;

fn argmax(values: &[f32]) -> u32 {
    values.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map_or(0, |(i, _)| i as u32)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let dir = Path::new(&args[1]);
    let count: usize = args.get(2).map_or(60, |n| n.parse().unwrap());
    let config = std::fs::read_to_string(dir.join("config.json"))?;
    let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| e.to_string())?;
    let text = format!("<|im_start|>user\n{PROMPT}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
    let ids = tokenizer.encode(text, false).map_err(|e| e.to_string())?.get_ids().to_vec();

    let mut engine = Engine::new(compile_qwen3_5(&config)?, Weights::open(dir)?, Numeric::Float)?;
    assert!(engine.has_draft_head(), "the model has no MTP head");

    let mut sequence = ids.clone();
    let mut hits = [0usize; DEPTH];
    let mut tries = [0usize; DEPTH];
    let mut drafts: Vec<Vec<u32>> = Vec::new();

    for i in 0..ids.len() + count {
        let token = sequence[i];
        let logits = engine.step(token, true)?.0.unwrap().to_vec();
        let next = argmax(&logits);
        if sequence.len() == i + 1 {
            sequence.push(next);
        }
        let actual_next = sequence[i + 1];

        // Speculate from the main model's hidden state, then undo the speculation.
        if i + 1 >= ids.len() {
            let mark = engine.draft_tokens();
            let mut guess = actual_next;
            let mut predicted = Vec::new();
            engine.feed_draft_from_main();
            for _ in 0..DEPTH {
                let logits = engine.draft(guess, true)?.unwrap().to_vec();
                guess = argmax(&logits);
                predicted.push(guess);
                engine.feed_draft_from_draft();
            }
            engine.truncate_draft(mark);
            drafts.push(predicted);
        }

        // Keep the draft head's cache in step with the accepted sequence.
        engine.feed_draft_from_main();
        engine.draft(actual_next, false)?;
    }

    for (start, predicted) in drafts.iter().enumerate() {
        let base = ids.len() + start + 1;
        for (depth, guess) in predicted.iter().enumerate() {
            if let Some(actual) = sequence.get(base + depth) {
                tries[depth] += 1;
                if actual == guess {
                    hits[depth] += 1;
                }
            }
        }
    }
    for d in 0..DEPTH {
        println!("depth {}: {:.1} % ({} of {})", d + 1, 100.0 * hits[d] as f64 / tries[d].max(1) as f64, hits[d], tries[d]);
    }
    println!("{}", tokenizer.decode(&sequence[ids.len()..], true).map_err(|e| e.to_string())?);
    Ok(())
}
