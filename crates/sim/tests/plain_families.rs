//! Runs tiny Llama-shaped checkpoints and compares with the logits mlx-lm computed.

use compiler::compile_model;
use serde_json::Value;
use sim::{Engine, ExactEngine, Numeric, Runner, Weights};
use std::path::PathBuf;

const FAMILIES: [&str; 9] = ["deepseek_v3", "llama", "llama3", "mistral", "qwen2", "qwen3", "gemma", "gemma2", "gemma3_text"];

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

#[test]
fn llama_3_depends_on_its_scaled_waves() {
    let (dir, config, tokens, expected) = case("llama3");
    let mut program = compile_model(&config).unwrap();
    assert!(program.rope_scaling.take().is_some());
    let mut engine = Engine::new(program, Weights::open(&dir).unwrap(), Numeric::Float).unwrap();
    let mut largest = 0f32;
    for (token, reference) in tokens.iter().zip(&expected) {
        let (logits, _) = Runner::step(&mut engine, *token, true).unwrap();
        largest = logits.unwrap().iter().zip(reference).fold(largest, |most, (got, want)| most.max((got - want).abs()));
    }
    assert!(largest > 0.05, "sin escalar se aleja solo {largest}");
}

mod common;

/// The error function, to ten decimals, for a GELU like the image heads'.
fn erf(x: f32) -> f32 {
    let t = 1.0 / (1.0 + 0.3275911 * x.abs() as f64);
    let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * (-(x as f64) * x as f64).exp();
    (y as f32).copysign(x)
}

#[test]
fn an_image_heads_scores_and_inputs_come_from_the_chip_as_in_float() {
    let (plain, text, tokens, _) = case("llama");
    let dir = std::env::temp_dir().join(format!("llm2chip-janus-heads-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::copy(plain.join("model.safetensors"), dir.join("model.safetensors")).unwrap();
    let text: Value = serde_json::from_str(&text).unwrap();
    let hidden = text["hidden_size"].as_u64().unwrap() as usize;
    let (codes, code_dim, width) = (16usize, 4usize, 2 * hidden);
    // Random heads in floating point, in a file of their own.
    let mut rng = common::Rng(0x5EED_1234);
    let mut writer = common::Writer::default();
    let tensor = |writer: &mut common::Writer, name: &str, shape: &[usize], rng: &mut common::Rng| -> Vec<f32> {
        let values: Vec<f32> = (0..shape.iter().product::<usize>()).map(|_| rng.unit() * 0.5).collect();
        let rounded: Vec<f32> = values.iter().map(|v| half::f16::from_f32(*v).to_f32()).collect();
        writer.tensor(name, "F16", shape, values.iter().flat_map(|v| half::f16::from_f32(*v).to_le_bytes()).collect());
        rounded
    };
    let proj = (tensor(&mut writer, "model.generation_head.proj_out.weight", &[hidden, hidden], &mut rng), tensor(&mut writer, "model.generation_head.proj_out.bias", &[hidden], &mut rng));
    let head = (tensor(&mut writer, "model.generation_head.vision_head.weight", &[codes, hidden], &mut rng), tensor(&mut writer, "model.generation_head.vision_head.bias", &[codes], &mut rng));
    let book = tensor(&mut writer, "model.generation_embeddings.weight", &[codes, code_dim], &mut rng);
    let fc1 = (tensor(&mut writer, "model.generation_aligner.fc1.weight", &[width, code_dim], &mut rng), tensor(&mut writer, "model.generation_aligner.fc1.bias", &[width], &mut rng));
    let fc2 = (tensor(&mut writer, "model.generation_aligner.hidden_layers.0.weight", &[hidden, width], &mut rng), tensor(&mut writer, "model.generation_aligner.hidden_layers.0.bias", &[hidden], &mut rng));
    writer.write(&dir.join("generation.safetensors"));
    let config = serde_json::json!({ "model_type": "janus", "text_config": text, "vq_config": { "num_embeddings": codes, "embed_dim": code_dim, "image_token_embed_dim": width } }).to_string();

    let linear = |x: &[f32], (w, b): &(Vec<f32>, Vec<f32>)| -> Vec<f32> { b.iter().enumerate().map(|(o, bias)| w[o * x.len()..(o + 1) * x.len()].iter().zip(x).map(|(a, v)| a * v).sum::<f32>() + bias).collect() };
    let gelu = |v: f32| 0.5 * v * (1.0 + erf(v / std::f32::consts::SQRT_2));
    let close = |got: &[f32], want: &[f32], what: &str| {
        let scale = want.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let error = got.iter().zip(want).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(error < 0.02 * scale, "{what}: {error} sobre {scale}: {got:?} {want:?}");
    };

    let mut float = Engine::new(compile_model(&config).unwrap(), Weights::open(&dir).unwrap(), Numeric::Float).unwrap();
    let mut exact = ExactEngine::new(compile_model(&config).unwrap(), Weights::open(&dir).unwrap(), 16).unwrap();
    for (step, token) in tokens.iter().take(3).enumerate() {
        let state = float.step_hidden(*token, None).unwrap();
        exact.step_hidden(*token, None).unwrap();
        // The scores of each code, from the state the token left.
        let want = linear(&linear(&state, &proj).into_iter().map(gelu).collect::<Vec<_>>(), &head);
        close(&exact.image_logits().unwrap(), &want, &format!("puntuaciones del paso {step}"));
        // The vector a drawn code goes in as.
        let code = (step * 5 + 3) % codes;
        let want = linear(&linear(&book[code * code_dim..(code + 1) * code_dim], &fc1).into_iter().map(gelu).collect::<Vec<_>>(), &fc2);
        close(&exact.image_input(code as u32).unwrap(), &want, &format!("entrada del código {code}"));
    }
    std::fs::remove_dir_all(dir).ok();
}
