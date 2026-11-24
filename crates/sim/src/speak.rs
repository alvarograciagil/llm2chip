//! OuteTTS 1.0: a Qwen3 that writes speech as tokens of its vocabulary, `<|c1_N|>` and
//! `<|c2_N|>` for DAC's two codebooks, a pair per frame of 75 a second. The language model
//! is an ordinary one for the compiler; what is particular to speech is here: its prompt,
//! how it samples, and which tokens are codes.
//!
//! The prompt is `OuteTTS`'s without a speaker profile:
//! `<|im_start|>\n<|text_start|>TEXT<|text_end|>\n<|audio_start|>\n`. It writes words, each
//! with a few tokens of its own (duration, energy, pitch) and its codes, until
//! `<|audio_end|>`. It samples as OuteTTS recommends (`Sampling::outetts`): temperature 0.4,
//! top-k 40, top-p 0.9, min-p 0.05, after a repetition penalty of 1.1 over the last 64
//! tokens, from a seed; or greedily after the same penalty, which is what the comparison
//! with Transformers uses.

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
    /// Every added token by its text, for a speaker's words: `<|word_start|>`, `<|t_0.20|>`…
    added: HashMap<String, u32>,
}

/// A voice to imitate, as OuteTTS keeps one: a sentence it said and, word by word, its
/// duration, its loudness, brightness and pitch, and its codes.
#[derive(Clone, Debug, serde::Deserialize)]
pub struct Speaker {
    pub text: String,
    pub words: Vec<SpeakerWord>,
}

#[derive(Clone, Debug, serde::Deserialize)]
pub struct SpeakerWord {
    pub word: String,
    pub duration: f64,
    pub c1: Vec<u32>,
    pub c2: Vec<u32>,
    pub features: SpeakerFeatures,
}

#[derive(Clone, Debug, serde::Deserialize)]
pub struct SpeakerFeatures {
    pub energy: i64,
    pub spectral_centroid: i64,
    pub pitch: i64,
}

impl Speaker {
    pub fn read(path: &Path) -> Result<Self, String> {
        serde_json::from_str(&std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?).map_err(|e| e.to_string())
    }
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
        Ok(Self { codes, start: id("<|im_start|>")?, end: id("<|im_end|>")?, text_start: id("<|text_start|>")?, text_end: id("<|text_end|>")?, audio_start: id("<|audio_start|>")?, audio_end: id("<|audio_end|>")?, newline, added: ids })
    }

    /// The prompt around the text's own tokens.
    pub fn prompt(&self, text: &[u32]) -> Vec<u32> {
        let mut prompt = vec![self.start, self.newline, self.text_start];
        prompt.extend_from_slice(text);
        prompt.extend([self.text_end, self.newline, self.audio_start, self.newline]);
        prompt
    }

    fn added(&self, name: &str) -> Result<u32, String> {
        self.added.get(name).copied().ok_or(format!("{name} is not in the vocabulary"))
    }

    /// The prompt in `speaker`'s voice, as OuteTTS writes it: the speaker's sentence and
    /// `text` as one text, then the speaker's words with their features and codes, and a
    /// word begun for the text's first. `encode` cuts plain text into tokens.
    pub fn prompt_with(&self, speaker: &Speaker, text: &str, encode: &dyn Fn(&str) -> Result<Vec<u32>, String>) -> Result<Vec<u32>, String> {
        let spoken = speaker.text.trim();
        let joined = if spoken.ends_with(['.', '?', '!']) { format!("{spoken} {}", text.trim()) } else { format!("{spoken}. {}", text.trim()) };
        let mut prompt = vec![self.start, self.newline, self.text_start];
        prompt.extend(encode(&joined)?);
        prompt.extend([self.text_end, self.newline, self.audio_start, self.newline]);
        for (i, w) in speaker.words.iter().enumerate() {
            if i > 0 {
                prompt.push(self.newline);
            }
            prompt.push(self.added("<|word_start|>")?);
            prompt.extend(encode(&w.word)?);
            prompt.push(self.added("<|features|>")?);
            prompt.push(self.added(&format!("<|t_{:.2}|>", w.duration))?);
            let f = &w.features;
            for name in [format!("<|energy_{}|>", f.energy), format!("<|spectral_centroid_{}|>", f.spectral_centroid), format!("<|pitch_{}|>", f.pitch)] {
                prompt.push(self.added(&name)?);
            }
            prompt.push(self.added("<|code|>")?);
            for (a, b) in w.c1.iter().zip(&w.c2) {
                prompt.push(self.added(&format!("<|c1_{a}|>"))?);
                prompt.push(self.added(&format!("<|c2_{b}|>"))?);
            }
            prompt.push(self.added("<|word_end|>")?);
        }
        prompt.extend([self.newline, self.added("<|word_start|>")?]);
        Ok(prompt)
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

/// How the next token is chosen.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Sampling {
    /// The most likely, after the repetition penalty.
    Greedy,
    /// Drawn at `temperature` among the `top_k` most likely, within `top_p` of the mass and
    /// at least `min_p` times the likeliest's chance, after the penalty, from `seed`.
    Drawn { temperature: f32, top_k: usize, top_p: f32, min_p: f32, seed: u64 },
}

impl Sampling {
    /// OuteTTS's own settings.
    pub fn outetts(seed: u64) -> Self {
        Self::Drawn { temperature: 0.4, top_k: 40, top_p: 0.9, min_p: 0.05, seed }
    }
}

/// `logits` after the repetition penalty over the last `WINDOW` of `tokens`.
fn penalised(logits: &[f32], tokens: &[u32]) -> Vec<f32> {
    let mut out = logits.to_vec();
    for t in tokens[tokens.len().saturating_sub(WINDOW)..].iter().collect::<std::collections::HashSet<_>>() {
        if let Some(v) = out.get_mut(*t as usize) {
            *v = if *v > 0.0 { *v / PENALTY } else { *v * PENALTY };
        }
    }
    out
}

/// The next token from `logits` by `sampling`; `state` carries the draws' generator.
pub fn sample(logits: &[f32], tokens: &[u32], sampling: Sampling, state: &mut u64) -> u32 {
    let Sampling::Drawn { temperature, top_k, top_p, min_p, .. } = sampling else {
        return choose(logits, tokens);
    };
    let scores = penalised(logits, tokens);
    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_unstable_by(|a, b| scores[*b].total_cmp(&scores[*a]));
    order.truncate(top_k.max(1));
    let top = scores[order[0]];
    let mut chances: Vec<(usize, f64)> = order.iter().map(|i| (*i, (((scores[*i] - top) / temperature) as f64).exp())).collect();
    let total: f64 = chances.iter().map(|(_, p)| p).sum();
    chances.iter_mut().for_each(|(_, p)| *p /= total);
    // min-p against the likeliest, then the smallest set reaching top-p of the mass.
    let floor = chances[0].1 * min_p as f64;
    chances.retain(|(_, p)| *p >= floor);
    let mut mass = 0.0;
    let mut kept = 0;
    for (_, p) in &chances {
        kept += 1;
        mass += p;
        if mass >= top_p as f64 {
            break;
        }
    }
    chances.truncate(kept.max(1));
    let total: f64 = chances.iter().map(|(_, p)| p).sum();
    // xorshift64*, as the rest of the project's draws.
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    let draw = (state.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 / (1u64 << 53) as f64 * total;
    let mut sum = 0.0;
    for (i, p) in &chances {
        sum += p;
        if draw < sum {
            return *i as u32;
        }
    }
    chances.last().map_or(0, |(i, _)| *i as u32)
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
/// scores of the next, until the end of the audio or `limit` tokens, greedily: gives them all.
pub fn write(vocabulary: &SpeechVocabulary, prompt: &[u32], limit: usize, step: &mut dyn FnMut(u32) -> Result<Vec<f32>, String>) -> Result<Vec<u32>, String> {
    write_sampled(vocabulary, prompt, limit, Sampling::Greedy, step)
}

/// `write`, choosing each token by `sampling`.
pub fn write_sampled(vocabulary: &SpeechVocabulary, prompt: &[u32], limit: usize, sampling: Sampling, step: &mut dyn FnMut(u32) -> Result<Vec<f32>, String>) -> Result<Vec<u32>, String> {
    let mut tokens = prompt.to_vec();
    let mut logits = Vec::new();
    let mut state = match sampling {
        Sampling::Drawn { seed, .. } => seed.max(1),
        Sampling::Greedy => 1,
    };
    for token in prompt {
        logits = step(*token)?;
    }
    for _ in 0..limit {
        let token = sample(&logits, &tokens, sampling, &mut state);
        tokens.push(token);
        if vocabulary.ends(token) {
            break;
        }
        logits = step(token)?;
    }
    Ok(tokens[prompt.len()..].to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drawing_stays_among_the_likely_tokens_and_repeats_with_its_seed() {
        let logits: Vec<f32> = (0..100).map(|i| if i < 3 { 10.0 - 0.1 * i as f32 } else { -5.0 }).collect();
        let draws = |seed: u64| {
            let mut state = seed;
            (0..50).map(|_| sample(&logits, &[], Sampling::outetts(seed), &mut state)).collect::<Vec<_>>()
        };
        let first = draws(7);
        assert!(first.iter().all(|t| *t < 3), "min-p and top-p keep the three likely ones");
        assert_eq!(first, draws(7));
        assert!(first.iter().any(|t| *t != first[0]), "it does draw");
        assert_eq!(sample(&logits, &[], Sampling::Greedy, &mut 1), 0);
    }
}
