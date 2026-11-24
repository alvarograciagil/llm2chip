//! The browser's simulator against the local engine's path, on the small models of
//! ~/.llm2chip/models when they are there: the chip's tokenizer must cut the text as Hugging
//! Face's does, and a checkpoint held in memory must answer token for token as one mapped
//! from disk, framed, sampled and decoded by the engine's rules.

use llm2chip_wasm_sim::{Generator, Request, Setup};
use sim::prompt::PromptStyle;
use sim::tokenizer::TokenizerTables;
use sim::{sample, ExactEngine, Runner, Weights};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;

const MODELS: [&str; 2] = ["stories260K", "HuggingFaceTB--SmolLM2-135M"];
const TEXTS: [&str; 4] = ["Once upon a time", "The capital of France is", "¿Qué tal? Hola, señor, ¿un café? 😀", "  spaces\tand\nlines  "];
const TOKENS: usize = 12;

fn folder(name: &str) -> Option<PathBuf> {
    let dir = Path::new(&std::env::var("HOME").ok()?).join(".llm2chip/models").join(name);
    dir.join("config.json").exists().then_some(dir)
}

fn texts(dir: &Path) -> HashMap<String, String> {
    ["config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json", "chat_template.jinja"]
        .into_iter()
        .filter_map(|name| Some((name.to_string(), std::fs::read_to_string(dir.join(name)).ok()?)))
        .collect()
}

fn weights(dir: &Path) -> Vec<Box<[u8]>> {
    let mut names: Vec<_> = std::fs::read_dir(dir).unwrap().filter_map(Result::ok).map(|e| e.path()).filter(|p| p.extension().is_some_and(|e| e == "safetensors")).collect();
    names.sort();
    names.iter().map(|p| std::fs::read(p).unwrap().into_boxed_slice()).collect()
}

const SETUP: Setup = Setup { activation_bits: 16, format: None, context_tokens: 512 };

#[test]
fn the_chips_tokenizer_cuts_text_as_hugging_faces_does() {
    for model in MODELS {
        let Some(dir) = folder(model) else { continue };
        let reference = Tokenizer::from_file(dir.join("tokenizer.json")).unwrap();
        let mut chip = TokenizerTables::from_json(&std::fs::read_to_string(dir.join("tokenizer.json")).unwrap(), 4096, 1 << 16).unwrap();
        for add in [true, false] {
            chip.set_add_special(add);
            for text in TEXTS {
                let expected = reference.encode(text, add).unwrap().get_ids().to_vec();
                assert_eq!(chip.encode(text).unwrap(), expected, "{model}, special tokens {add}: {text:?}");
                assert_eq!(String::from_utf8_lossy(&chip.decode(&expected)), reference.decode(&expected, true).unwrap(), "{model}: decoding {text:?}");
            }
        }
    }
}

#[test]
fn a_checkpoint_in_memory_answers_as_one_on_disk() {
    for model in MODELS {
        let Some(dir) = folder(model) else { continue };
        let config = std::fs::read_to_string(dir.join("config.json")).unwrap();
        let program = compiler::compile_model(&config).unwrap();
        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json")).unwrap();
        let style = PromptStyle::read(&program.family, |name| std::fs::read_to_string(dir.join(name)).ok(), |t| tokenizer.token_to_id(t));
        let mut engine = ExactEngine::new(program, Weights::open(&dir).unwrap(), SETUP.activation_bits).unwrap();
        let mut generator = Generator::new(&texts(&dir), weights(&dir), &SETUP).unwrap();

        for text in TEXTS {
            // The local engine's way: framed, cut by Hugging Face, greedy, its stops.
            let (framed, add) = style.render(text, false).unwrap();
            let ids = tokenizer.encode(framed, add).unwrap().get_ids().to_vec();
            engine.reset();
            let mut next = 0;
            for (i, id) in ids.iter().enumerate() {
                let (logits, _) = Runner::step(&mut engine, *id, i + 1 == ids.len()).unwrap();
                if let Some(l) = logits {
                    next = sample(l, 0.0, 1, &mut 1);
                }
            }
            let mut expected = Vec::new();
            while expected.len() < TOKENS && !style.stop.contains(&next) {
                expected.push(next);
                let (logits, _) = Runner::step(&mut engine, next, true).unwrap();
                next = sample(logits.unwrap(), 0.0, 1, &mut 1);
            }

            let start = generator.start(Request { prompt: text.into(), max_tokens: TOKENS, temperature: 0.0, top_k: 1, thinking: false, think_budget: None, seed: 7, system: String::new() }).unwrap();
            assert_eq!(start["prompt_tokens"], ids.len(), "{model}: {text:?}");
            let (mut got, mut said) = (Vec::new(), String::new());
            loop {
                let event = generator.advance().unwrap();
                match event["type"].as_str().unwrap() {
                    "token" => {
                        got.push(event["id"].as_u64().unwrap() as u32);
                        said.push_str(event["text"].as_str().unwrap());
                    }
                    "done" => break,
                    _ => {}
                }
            }
            // A loop the chip notices may end the answer sooner; what it said must agree.
            assert!(!got.is_empty() || expected.is_empty(), "{model}: {text:?} said nothing");
            assert_eq!(got[..], expected[..got.len()], "{model}: {text:?}");
            assert_eq!(said, tokenizer.decode(&got, true).unwrap(), "{model}: the text of {text:?}");
        }
    }
}
