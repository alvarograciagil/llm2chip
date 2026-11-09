//! Runs tiny Llama-shaped checkpoints and compares with the logits mlx-lm computed.

use compiler::compile_model;
use serde_json::Value;
use sim::{Engine, ExactEngine, Numeric, Runner, Weights};
use std::path::PathBuf;

const FAMILIES: [&str; 3] = ["llama", "mistral", "qwen3"];

fn case(family: &str) -> (PathBuf, String, Vec<u32>, Vec<Vec<f32>>) {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden/plain").join(family);
    let config = std::fs::read_to_string(dir.join("config.json")).unwrap();
    let reference: Value = serde_json::from_str(&std::fs::read_to_string(dir.join("reference.json")).unwrap()).unwrap();
    let tokens = reference["tokens"].as_array().unwrap().iter().map(|t| t.as_u64().unwrap() as u32).collect();
    let logits = reference["logits"].as_array().unwrap().iter().map(|row| row.as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect()).collect();
    (dir, config, tokens, logits)
}

fn best(values: &[f32]) -> usize {
    values.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0
}

#[test]
fn every_family_matches_the_reference_token_by_token() {
    for family in FAMILIES {
        let (dir, config, tokens, expected) = case(family);
        let mut engine = Engine::new(compile_model(&config).unwrap(), Weights::open(&dir).unwrap(), Numeric::Float).unwrap();
        for (position, (token, reference)) in tokens.iter().zip(&expected).enumerate() {
            let (logits, _) = Runner::step(&mut engine, *token, true).unwrap();
            for (index, (got, want)) in logits.unwrap().iter().zip(reference).enumerate() {
                assert!((got - want).abs() < 2e-3, "{family}, token {position}, logit {index}: {got} frente a {want}");
            }
        }
    }
}

#[test]
fn the_die_arithmetic_keeps_the_same_choice() {
    for family in FAMILIES {
        let (dir, config, tokens, expected) = case(family);
        let mut engine = ExactEngine::new(compile_model(&config).unwrap(), Weights::open(&dir).unwrap(), 16).unwrap();
        for (position, (token, reference)) in tokens.iter().zip(&expected).enumerate() {
            let (logits, _) = Runner::step(&mut engine, *token, true).unwrap();
            let logits = logits.unwrap();
            assert_eq!(best(logits), best(reference), "{family}, token {position}");
        }
    }
}
