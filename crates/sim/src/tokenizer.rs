//! The tokenizer's tables, as the chip's firmware reads them from data memory.

use serde_json::Value;
use std::collections::HashMap;
use std::path::Path;

/// Fields at the start of the tables; the firmware has the same list.
const HEADER_WORDS: usize = 13;

/// Bytes of memory set aside for the incoming text.
const TEXT_CAPACITY_BYTES: u32 = 1 << 20;

/// What the firmware needs to turn text into tokens and tokens back into bytes.
#[derive(Clone, Debug, Default)]
pub struct TokenizerTables {
    pub byte_symbols: Vec<u32>,
    /// `(first, second, result)`, in rank order.
    pub merges: Vec<(u32, u32, u32)>,
    pub letters: Vec<(u32, u32)>,
    pub numbers: Vec<(u32, u32)>,
    /// The bytes each token writes, by token number.
    pub token_bytes: Vec<Vec<u8>>,
    /// `(token, text)` matched literally before anything else.
    pub added: Vec<(u32, Vec<u8>)>,
    pub text_capacity: u32,
}

fn hash_slot(first: u32, second: u32) -> u32 {
    first.wrapping_mul(0x9E37_79B1) ^ second.wrapping_mul(0x85EB_CA77)
}

/// GPT-2's reversible map from bytes to printable characters.
fn byte_characters() -> Vec<char> {
    let printable = |b: u32| (0x21..=0x7E).contains(&b) || (0xA1..=0xAC).contains(&b) || (0xAE..=0xFF).contains(&b);
    let mut next = 0;
    (0..256u32)
        .map(|b| {
            if printable(b) {
                char::from_u32(b).expect("carácter")
            } else {
                next += 1;
                char::from_u32(255 + next).expect("carácter")
            }
        })
        .collect()
}

/// Sorted, inclusive ranges of every character the pattern accepts.
fn ranges_of(pattern: &str) -> Result<Vec<(u32, u32)>, String> {
    let class = regex::Regex::new(pattern).map_err(|e| e.to_string())?;
    let mut ranges: Vec<(u32, u32)> = Vec::new();
    let mut buffer = [0u8; 4];
    for code in 0..=0x10FFFFu32 {
        let Some(c) = char::from_u32(code) else { continue };
        if class.is_match(c.encode_utf8(&mut buffer)) {
            match ranges.last_mut() {
                Some((_, end)) if *end + 1 == code => *end = code,
                _ => ranges.push((code, code)),
            }
        }
    }
    Ok(ranges)
}

impl TokenizerTables {
    /// Reads the tables out of a `tokenizer.json` with a byte-level BPE model.
    pub fn from_file(path: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
        let json: Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
        let model = &json["model"];
        let vocabulary = model["vocab"].as_object().ok_or("tokenizer.json sin vocabulario")?;
        let ids: HashMap<&str, u32> = vocabulary.iter().map(|(k, v)| (k.as_str(), v.as_u64().unwrap_or(0) as u32)).collect();

        let characters = byte_characters();
        let byte_of: HashMap<char, u8> = characters.iter().enumerate().map(|(b, c)| (*c, b as u8)).collect();
        let byte_symbols = characters.iter().map(|c| ids.get(c.to_string().as_str()).copied().ok_or("falta un símbolo de byte")).collect::<Result<Vec<u32>, _>>()?;

        let mut merges = Vec::new();
        for pair in model["merges"].as_array().ok_or("tokenizer.json sin fusiones")? {
            let (first, second) = (pair[0].as_str().unwrap_or(""), pair[1].as_str().unwrap_or(""));
            let joined = format!("{first}{second}");
            if let (Some(a), Some(b), Some(result)) = (ids.get(first), ids.get(second), ids.get(joined.as_str())) {
                merges.push((*a, *b, *result));
            }
        }

        let added_list = json["added_tokens"].as_array().ok_or("tokenizer.json sin tokens añadidos")?;
        let last_added = added_list.iter().filter_map(|t| t["id"].as_u64()).max().unwrap_or(0) as usize;
        let largest = vocabulary.len().max(last_added + 1);
        let mut token_bytes = vec![Vec::new(); largest];
        for (text, id) in &ids {
            token_bytes[*id as usize] = text.chars().filter_map(|c| byte_of.get(&c).copied()).collect();
        }
        let mut added = Vec::new();
        for token in added_list {
            let (id, content) = (token["id"].as_u64().unwrap_or(0) as u32, token["content"].as_str().unwrap_or(""));
            added.push((id, content.as_bytes().to_vec()));
            // Special tokens write nothing, as when decoding with them skipped.
            token_bytes[id as usize] = if token["special"].as_bool().unwrap_or(false) { Vec::new() } else { content.as_bytes().to_vec() };
        }
        Ok(Self { byte_symbols, merges, letters: ranges_of(r"^\p{L}$")?, numbers: ranges_of(r"^\p{N}$")?, token_bytes, added, text_capacity: TEXT_CAPACITY_BYTES })
    }

    /// The tables as the words of the firmware's memory.
    pub fn to_words(&self) -> Vec<u32> {
        let slots = (self.merges.len().max(1) * 2).next_power_of_two();
        let mut hash = vec![0u32; slots * 4];
        for (rank, (first, second, result)) in self.merges.iter().enumerate() {
            let mut slot = hash_slot(*first, *second) as usize & (slots - 1);
            while hash[slot * 4] != 0 {
                slot = (slot + 1) & (slots - 1);
            }
            hash[slot * 4..slot * 4 + 4].copy_from_slice(&[first + 1, *second, rank as u32, *result]);
        }

        let mut blob: Vec<u8> = Vec::new();
        let mut offsets = Vec::with_capacity(self.token_bytes.len() + 1);
        for bytes in &self.token_bytes {
            offsets.push(blob.len() as u32);
            blob.extend_from_slice(bytes);
        }
        offsets.push(blob.len() as u32);
        let mut added = Vec::with_capacity(self.added.len() * 3);
        for (id, text) in &self.added {
            added.extend([*id, blob.len() as u32, text.len() as u32]);
            blob.extend_from_slice(text);
        }
        let blob_words: Vec<u32> = blob.chunks(4).map(|c| c.iter().rev().fold(0, |word, byte| word << 8 | *byte as u32)).collect();

        let mut words = vec![0u32; HEADER_WORDS];
        let place = |words: &mut Vec<u32>, section: &[u32]| {
            let at = words.len() as u32;
            words.extend_from_slice(section);
            at
        };
        let flat = |ranges: &[(u32, u32)]| ranges.iter().flat_map(|(a, b)| [*a, *b]).collect::<Vec<u32>>();
        words[0] = slots as u32;
        words[1] = place(&mut words, &hash);
        words[2] = self.letters.len() as u32;
        words[3] = place(&mut words, &flat(&self.letters));
        words[4] = self.numbers.len() as u32;
        words[5] = place(&mut words, &flat(&self.numbers));
        words[6] = place(&mut words, &self.byte_symbols);
        words[7] = place(&mut words, &offsets);
        words[8] = place(&mut words, &blob_words);
        words[9] = self.added.len() as u32;
        words[10] = place(&mut words, &added);
        words[11] = words.len() as u32;
        words[12] = self.text_capacity;
        let end = words.len() + self.text_capacity.div_ceil(4) as usize;
        words.resize(end, 0);
        words
    }
}
