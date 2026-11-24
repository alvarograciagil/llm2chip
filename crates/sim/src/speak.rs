//! OuteTTS 1.0: a Qwen3 that writes speech as tokens of its vocabulary, `<|c1_N|>` and
//! `<|c2_N|>` for DAC's two codebooks, a pair per frame of 75 a second. The language model
//! is an ordinary one for the compiler; what is particular to speech is here: its prompt,
//! how it samples, and which tokens are codes.
//!
//! The prompt is `OuteTTS`'s without a speaker profile:
//! `<|im_start|>\n<|text_start|>TEXT<|text_end|>\n<|audio_start|>\n`. It writes words, each
//! with a few tokens of its own (duration, energy, pitch) and its codes, until
//! `<|audio_end|>`. It samples with a repetition penalty of 1.1 over the last 64 tokens;
//! here greedily, so that the same text gives the same speech.

use serde_json::Value;
use std::collections::HashMap;
use std::path::Path;

pub const PENALTY: f32 = 1.1;
pub const WINDOW: usize = 64;

/// The tokens of OuteTTS's vocabulary that speech needs.
#[derive(Clone, Debug)]
pub struct SpeechVocabulary {
    /// The token of each code, by codebook: `<|c1_N|>` is `codes[0][N]`.
    codes: [HashMap<u32, u32>; 2],
    pub start: u32,
    pub end: u32,
    pub text_start: u32,
    pub text_end: u32,
    pub audio_start: u32,
    pub audio_end: u32,
    /// The newline's token, between the prompt's parts.
    pub newline: u32,
}

impl SpeechVocabulary {
    /// From the checkpoint's `tokenizer.json`.
    pub fn read(dir: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(dir.join("tokenizer.json")).map_err(|e| e.to_string())?;
        let json: Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
        let mut ids = HashMap::new();
        for token in json["added_tokens"].as_array().ok_or("tokenizer.json has no added tokens")? {
            if let (Some(content), Some(id)) = (token["content"].as_str(), token["id"].as_u64()) {
                ids.insert(content.to_string(), id as u32);
            }
        }
        let vocabulary = json["model"]["vocab"].as_object().ok_or("tokenizer.json has no vocabulary")?;
        let id = |name: &str| ids.get(name).copied().ok_or(format!("{name} is not in the vocabulary"));
        let mut codes = [HashMap::new(), HashMap::new()];
        for (name, token) in &ids {
            for (book, prefix) in ["<|c1_", "<|c2_"].into_iter().enumerate() {
                if let Some(number) = name.strip_prefix(prefix).and_then(|n| n.strip_suffix("|>")).and_then(|n| n.parse().ok()) {
                    codes[book].insert(*token, number);
                }
            }
        }
        if codes.iter().any(HashMap::is_empty) {
            return Err("the vocabulary has no codes: not an OuteTTS 1.0 checkpoint".into());
        }
        // Byte-level BPE writes a newline as Ċ.
        let newline = vocabulary.get("Ċ").and_then(Value::as_u64).ok_or("the vocabulary has no newline")? as u32;
        Ok(Self { codes, start: id("<|im_start|>")?, end: id("<|im_end|>")?, text_start: id("<|text_start|>")?, text_end: id("<|text_end|>")?, audio_start: id("<|audio_start|>")?, audio_end: id("<|audio_end|>")?, newline })
    }

    /// The prompt around the text's own tokens.
    pub fn prompt(&self, text: &[u32]) -> Vec<u32> {
        let mut prompt = vec![self.start, self.newline, self.text_start];
        prompt.extend_from_slice(text);
        prompt.extend([self.text_end, self.newline, self.audio_start, self.newline]);
        prompt
    }

    /// The code of `token` in codebook `book`, if it is one.
    pub fn code(&self, book: usize, token: u32) -> Option<u32> {
        self.codes[book].get(&token).copied()
    }

    /// The two codebooks' codes among `tokens`, each in order, as long as the shorter.
    pub fn codes(&self, tokens: &[u32]) -> Vec<Vec<u32>> {
        let mut books: Vec<Vec<u32>> = (0..2).map(|b| tokens.iter().filter_map(|t| self.code(b, *t)).collect()).collect();
        let frames = books[0].len().min(books[1].len());
        books.iter_mut().for_each(|b| b.truncate(frames));
        books
    }

    pub fn ends(&self, token: u32) -> bool {
        token == self.audio_end || token == self.end
    }
}

/// The next token from `logits`, greedily, after the repetition penalty over the last
/// `WINDOW` of `tokens`.
pub fn choose(logits: &[f32], tokens: &[u32]) -> u32 {
    let recent: std::collections::HashSet<u32> = tokens[tokens.len().saturating_sub(WINDOW)..].iter().copied().collect();
    let mut best = (0u32, f32::NEG_INFINITY);
    for (i, value) in logits.iter().enumerate() {
        let value = if recent.contains(&(i as u32)) {
            if *value > 0.0 { value / PENALTY } else { value * PENALTY }
        } else {
            *value
        };
        if value > best.1 {
            best = (i as u32, value);
        }
    }
    best.0
}

/// Writes speech tokens after `prompt` with `step`, which takes a token and gives the
/// scores of the next, until the end of the audio or `limit` tokens: gives them all.
pub fn write(vocabulary: &SpeechVocabulary, prompt: &[u32], limit: usize, step: &mut dyn FnMut(u32) -> Result<Vec<f32>, String>) -> Result<Vec<u32>, String> {
    let mut tokens = prompt.to_vec();
    let mut logits = Vec::new();
    for token in prompt {
        logits = step(*token)?;
    }
    for _ in 0..limit {
        let token = choose(&logits, &tokens);
        tokens.push(token);
        if vocabulary.ends(token) {
            break;
        }
        logits = step(token)?;
    }
    Ok(tokens[prompt.len()..].to_vec())
}
