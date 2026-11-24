//! A spoken conversation keeps the language model's cache between questions: the second
//! question is fed on top of the first and its answer, and the scores after it are those of
//! the whole conversation fed from the start, system prompt first. With a real model, when
//! it is on disk.
#![cfg(any(feature = "listen", feature = "speak"))]

use sim::voice::{Arithmetic, Conversation, Model};
use std::path::PathBuf;
use tokenizers::Tokenizer;

#[test]
#[ignore = "slow: a real model, Qwen3-0.6B, if it is in ~/.llm2chip/models"]
fn a_conversation_goes_on_from_its_cache() {
    let dir = PathBuf::from(std::env::var_os("HOME").unwrap_or_default()).join(".llm2chip/models/Qwen--Qwen3-0.6B");
    if !dir.join("config.json").exists() {
        eprintln!("skipped: Qwen3-0.6B is not in {}", dir.display());
        return;
    }
    let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json")).unwrap();
    let encode = |text: &str, add: bool| tokenizer.encode(text, add).map(|e| e.get_ids().to_vec()).map_err(|e| e.to_string());
    let decode = |tokens: &[u32]| tokenizer.decode(tokens, true).map_err(|e| e.to_string());
    let system = "You are a geography tutor. Answer in one short sentence.";
    let mut conversation = Conversation::new(&dir, Arithmetic::Float, system, &|t| tokenizer.token_to_id(t)).unwrap();
    let (first, _) = conversation.ask("What is the capital of France?", 30, &encode, &decode).unwrap();
    let (second, logits) = conversation.ask("And of Italy?", 30, &encode, &decode).unwrap();
    eprintln!("{first:?} / {second:?}, {} tokens in the cache", conversation.tokens().len());
    assert_eq!(conversation.replayed, 0, "the second question went on from the cache");
    assert_eq!(conversation.turns(), 2);
    // The system prompt is the first thing in the cache, as the chip runs it.
    let framed = encode(&format!("<|im_start|>system\n{system}<|im_end|>\n"), false).unwrap();
    assert_eq!(conversation.tokens()[..framed.len()], framed[..]);
    assert!(second.contains("Rome"), "the second answer knows the first question: {second:?}");
    // The whole conversation fed from the start gives the same scores.
    let mut fresh = Model::open(&dir, Arithmetic::Float).unwrap();
    let mut replayed = Vec::new();
    for token in conversation.tokens() {
        replayed = fresh.step(*token).unwrap();
    }
    let worst = replayed.iter().zip(&logits).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
    assert!(worst < 1e-3, "{worst}");
}
