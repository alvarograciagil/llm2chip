//! Moonshine, a speech recogniser, in floating point: what the chip has to reproduce to
//! turn speech into text. It follows `modeling_moonshine.py` in Transformers.
//!
//! - **Encoder:** the 16 kHz waveform goes through three convolutions (kernel 127 stride
//!   64 with tanh and a GroupNorm of one group, kernel 7 stride 3 with GELU, kernel 3
//!   stride 2 with GELU), one frame every 384 samples, then through blocks of
//!   bidirectional attention with partial RoPE and an MLP with GELU, and a final LayerNorm.
//! - **Decoder:** a causal language model whose layers attend to their own tokens (with
//!   partial RoPE), then to the encoder's frames (cross-attention, without RoPE), then go
//!   through an MLP gated with SiLU. Its head is the embedding table.
//!
//! RoPE rotates adjacent pairs of each head (`x[2i]`, `x[2i + 1]`), not its two halves.
//! Every LayerNorm has a weight and no bias; the convolutions and the MLPs have biases.

use crate::Weights;
use serde_json::Value;
use std::collections::HashMap;
use std::path::Path;

const EPSILON: f32 = 1e-5;

/// What a Moonshine checkpoint looks like, from its `config.json`.
#[derive(Clone, Debug)]
pub struct MoonshineConfig {
    pub hidden: usize,
    pub intermediate: usize,
    pub encoder_layers: usize,
    pub decoder_layers: usize,
    pub heads: usize,
    pub head_dim: usize,
    /// `int(head_dim · partial_rotary_factor)`: the frequencies' denominator. RoPE rotates
    /// the first `rotary` values of each head, this rounded up to even.
    pub rope_dim: usize,
    pub rotary: usize,
    pub rope_theta: f32,
    pub vocab: usize,
    /// The token the decoder starts from and the one that ends the text.
    pub start: u32,
    pub end: u32,
    /// The most tokens the decoder writes.
    pub max_tokens: usize,
}

impl MoonshineConfig {
    pub fn read(dir: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(dir.join("config.json")).map_err(|e| e.to_string())?;
        Self::parse(&text)
    }

    pub fn parse(text: &str) -> Result<Self, String> {
        let config: Value = serde_json::from_str(text).map_err(|e| e.to_string())?;
        if config["model_type"].as_str() != Some("moonshine") {
            return Err("not a Moonshine checkpoint".into());
        }
        let number = |key: &str| config[key].as_u64().map(|v| v as usize).ok_or(format!("“{key}” is missing from the config"));
        let hidden = number("hidden_size")?;
        let heads = number("encoder_num_attention_heads")?;
        if number("decoder_num_attention_heads")? != heads || number("encoder_num_key_value_heads").unwrap_or(heads) != heads || number("decoder_num_key_value_heads").unwrap_or(heads) != heads {
            return Err("only encoders and decoders with the same heads, each with its own keys, are read".into());
        }
        if config["attention_bias"].as_bool() == Some(true) {
            return Err("attention with biases is not read".into());
        }
        let head_dim = hidden / heads;
        let rope = config.get("rope_parameters");
        let factor = config["partial_rotary_factor"].as_f64().or_else(|| rope.and_then(|r| r["partial_rotary_factor"].as_f64())).unwrap_or(0.9);
        let rope_theta = config["rope_theta"].as_f64().or_else(|| rope.and_then(|r| r["rope_theta"].as_f64())).unwrap_or(10_000.0) as f32;
        let rope_dim = (head_dim as f64 * factor) as usize;
        let rotary = rope_dim.div_ceil(2) * 2;
        Ok(Self {
            hidden,
            intermediate: number("intermediate_size")?,
            encoder_layers: number("encoder_num_hidden_layers")?,
            decoder_layers: number("decoder_num_hidden_layers")?,
            heads,
            head_dim,
            rope_dim,
            rotary,
            rope_theta,
            vocab: number("vocab_size")?,
            start: config["decoder_start_token_id"].as_u64().unwrap_or(1) as u32,
            end: config["eos_token_id"].as_u64().unwrap_or(2) as u32,
            max_tokens: config["max_position_embeddings"].as_u64().unwrap_or(194) as usize,
        })
    }

    /// The encoder's frames for `samples` samples of audio: none below 1,151 samples.
    pub fn frames(samples: usize) -> usize {
        let conv1 = if samples < 127 { 0 } else { (samples - 127) / 64 + 1 };
        let conv2 = if conv1 < 7 { 0 } else { (conv1 - 7) / 3 + 1 };
        if conv2 < 3 {
            0
        } else {
            (conv2 - 3) / 2 + 1
        }
    }
}

/// A Moonshine checkpoint, its weights read as floats.
pub struct Moonshine {
    pub config: MoonshineConfig,
    tensors: HashMap<String, Vec<f32>>,
}

/// Cosine and sine of each rotated pair at `position`.
fn rotation(config: &MoonshineConfig, position: usize) -> Vec<(f32, f32)> {
    (0..config.rotary / 2)
        .map(|i| {
            let frequency = 1.0 / (config.rope_theta as f64).powf(2.0 * i as f64 / config.rope_dim as f64);
            let angle = position as f64 * frequency;
            (angle.cos() as f32, angle.sin() as f32)
        })
        .collect()
}

/// Rotates the first `rotary` values of each head of `x`, by adjacent pairs.
fn rotate(x: &mut [f32], head_dim: usize, turns: &[(f32, f32)]) {
    for head in x.chunks_mut(head_dim) {
        for (i, (cos, sin)) in turns.iter().enumerate() {
            let (a, b) = (head[2 * i], head[2 * i + 1]);
            head[2 * i] = a * cos - b * sin;
            head[2 * i + 1] = b * cos + a * sin;
        }
    }
}

fn layer_norm(x: &[f32], weight: &[f32]) -> Vec<f32> {
    let n = x.len() as f32;
    let mean = x.iter().sum::<f32>() / n;
    let variance = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
    let scale = 1.0 / (variance + EPSILON).sqrt();
    x.iter().zip(weight).map(|(v, w)| (v - mean) * scale * w).collect()
}

fn gelu(v: f32) -> f32 {
    0.5 * v * (1.0 + crate::vision::erf(v / std::f32::consts::SQRT_2))
}

fn silu(v: f32) -> f32 {
    v / (1.0 + (-v).exp())
}

fn add(a: &mut [f32], b: &[f32]) {
    a.iter_mut().zip(b).for_each(|(a, b)| *a += b);
}

/// Each head's attention over `keys` and `values`, one row per token.
fn attend(query: &[f32], keys: &[Vec<f32>], values: &[Vec<f32>], head_dim: usize) -> Vec<f32> {
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut out = vec![0f32; query.len()];
    for (h, q) in query.chunks(head_dim).enumerate() {
        let span = h * head_dim..(h + 1) * head_dim;
        let scores: Vec<f32> = keys.iter().map(|k| q.iter().zip(&k[span.clone()]).map(|(a, b)| a * b).sum::<f32>() * scale).collect();
        let top = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let weights: Vec<f32> = scores.iter().map(|s| (s - top).exp()).collect();
        let total: f32 = weights.iter().sum();
        for (w, v) in weights.iter().zip(values) {
            for (o, x) in out[span.clone()].iter_mut().zip(&v[span.clone()]) {
                *o += w / total * x;
            }
        }
    }
    out
}

impl Moonshine {
    /// The checkpoint in `dir`: its config and its weights.
    pub fn open(dir: &Path) -> Result<Self, String> {
        Self::new(MoonshineConfig::read(dir)?, &Weights::open(dir)?)
    }

    pub fn new(config: MoonshineConfig, weights: &Weights) -> Result<Self, String> {
        let mut tensors = HashMap::new();
        for (name, _, _) in weights.raw_tensors() {
            tensors.insert(name.to_string(), weights.floats(name)?);
        }
        Ok(Self { config, tensors })
    }

    pub fn tensor(&self, name: &str) -> Result<&[f32], String> {
        self.tensors.get(name).map(Vec::as_slice).ok_or_else(|| format!("{name} is missing"))
    }

    /// `W · x`, plus the bias if the checkpoint has one.
    fn linear(&self, name: &str, x: &[f32]) -> Result<Vec<f32>, String> {
        let weight = self.tensor(&format!("{name}.weight"))?;
        let rows = weight.len() / x.len();
        let mut y: Vec<f32> = weight.chunks(x.len()).map(|row| row.iter().zip(x).map(|(a, b)| a * b).sum()).collect();
        if let Ok(bias) = self.tensor(&format!("{name}.bias")) {
            add(&mut y[..rows], bias);
        }
        Ok(y)
    }

    /// A convolution without padding over `x`, one row per step, `kernel` steps wide and
    /// `stride` apart, with the weights laid `[out, in, kernel]`.
    fn conv(&self, name: &str, x: &[Vec<f32>], kernel: usize, stride: usize) -> Result<Vec<Vec<f32>>, String> {
        let weight = self.tensor(&format!("{name}.weight"))?;
        let bias = self.tensor(&format!("{name}.bias")).ok();
        let inputs = x.first().map_or(0, Vec::len);
        let outputs = weight.len() / (inputs * kernel);
        let steps = if x.len() < kernel { 0 } else { (x.len() - kernel) / stride + 1 };
        Ok((0..steps)
            .map(|t| {
                (0..outputs)
                    .map(|o| {
                        let mut sum = bias.map_or(0.0, |b| b[o]);
                        for c in 0..inputs {
                            for k in 0..kernel {
                                sum += weight[(o * inputs + c) * kernel + k] * x[t * stride + k][c];
                            }
                        }
                        sum
                    })
                    .collect()
            })
            .collect())
    }

    /// The front end: samples to frames, before the encoder's blocks.
    pub fn front(&self, samples: &[f32]) -> Result<Vec<Vec<f32>>, String> {
        let p = |n: &str| format!("model.encoder.{n}");
        let wave: Vec<Vec<f32>> = samples.iter().map(|s| vec![*s]).collect();
        let mut x = self.conv(&p("conv1"), &wave, 127, 64)?;
        x.iter_mut().flatten().for_each(|v| *v = v.tanh());
        // GroupNorm of one group: over every frame and channel at once.
        let count = x.iter().map(Vec::len).sum::<usize>() as f64;
        let mean = x.iter().flatten().map(|v| *v as f64).sum::<f64>() / count;
        let variance = x.iter().flatten().map(|v| (*v as f64 - mean).powi(2)).sum::<f64>() / count;
        let scale = 1.0 / (variance + EPSILON as f64).sqrt();
        let (w, b) = (self.tensor(&p("groupnorm.weight"))?, self.tensor(&p("groupnorm.bias"))?);
        for row in &mut x {
            for (c, v) in row.iter_mut().enumerate() {
                *v = ((*v as f64 - mean) * scale) as f32 * w[c] + b[c];
            }
        }
        let mut x = self.conv(&p("conv2"), &x, 7, 3)?;
        x.iter_mut().flatten().for_each(|v| *v = gelu(*v));
        let mut x = self.conv(&p("conv3"), &x, 3, 2)?;
        x.iter_mut().flatten().for_each(|v| *v = gelu(*v));
        Ok(x)
    }

    /// The encoder's frames for `samples`, 16 kHz mono in [−1, 1].
    pub fn encode(&self, samples: &[f32]) -> Result<Vec<Vec<f32>>, String> {
        let c = &self.config;
        let mut x = self.front(samples)?;
        let turns: Vec<Vec<(f32, f32)>> = (0..x.len()).map(|t| rotation(c, t)).collect();
        for layer in 0..c.encoder_layers {
            let p = |n: &str| format!("model.encoder.layers.{layer}.{n}");
            let normed: Vec<Vec<f32>> = x.iter().map(|row| layer_norm(row, self.tensor(&p("input_layernorm.weight")).unwrap())).collect();
            let mut queries = Vec::new();
            let (mut keys, mut values) = (Vec::new(), Vec::new());
            for (t, row) in normed.iter().enumerate() {
                let mut q = self.linear(&p("self_attn.q_proj"), row)?;
                let mut k = self.linear(&p("self_attn.k_proj"), row)?;
                rotate(&mut q, c.head_dim, &turns[t]);
                rotate(&mut k, c.head_dim, &turns[t]);
                queries.push(q);
                keys.push(k);
                values.push(self.linear(&p("self_attn.v_proj"), row)?);
            }
            for (t, q) in queries.iter().enumerate() {
                let attended = attend(q, &keys, &values, c.head_dim);
                add(&mut x[t], &self.linear(&p("self_attn.o_proj"), &attended)?);
                let normed = layer_norm(&x[t], self.tensor(&p("post_attention_layernorm.weight"))?);
                let mut inner = self.linear(&p("mlp.fc1"), &normed)?;
                inner.iter_mut().for_each(|v| *v = gelu(*v));
                let out = self.linear(&p("mlp.fc2"), &inner)?;
                add(&mut x[t], &out);
            }
        }
        let weight = self.tensor("model.encoder.layer_norm.weight")?;
        Ok(x.iter().map(|row| layer_norm(row, weight)).collect())
    }

    /// A decoder that reads `frames`, the encoder's output: each layer's keys and values
    /// of every frame are made once, here.
    pub fn decoder(&self, frames: &[Vec<f32>]) -> Result<Decoder<'_>, String> {
        let mut cross = Vec::new();
        for layer in 0..self.config.decoder_layers {
            let p = |n: &str| format!("model.decoder.layers.{layer}.encoder_attn.{n}");
            let keys = frames.iter().map(|f| self.linear(&p("k_proj"), f)).collect::<Result<Vec<_>, _>>()?;
            let values = frames.iter().map(|f| self.linear(&p("v_proj"), f)).collect::<Result<Vec<_>, _>>()?;
            cross.push((keys, values));
        }
        let own = vec![(Vec::new(), Vec::new()); self.config.decoder_layers];
        Ok(Decoder { model: self, cross, own, position: 0 })
    }

    /// The text of `samples`, greedily: the tokens after the start token, up to the end one.
    pub fn transcribe(&self, samples: &[f32]) -> Result<Vec<u32>, String> {
        let frames = self.encode(samples)?;
        let mut decoder = self.decoder(&frames)?;
        let mut token = self.config.start;
        let mut text = Vec::new();
        // A limit of this reference's own, well above speech: six tokens a second of audio.
        let limit = (samples.len() * 6 / 16_000 + 1).min(self.config.max_tokens - 1);
        for _ in 0..limit {
            token = argmax(&decoder.step(token)?);
            if token == self.config.end {
                break;
            }
            text.push(token);
        }
        Ok(text)
    }
}

pub fn argmax(values: &[f32]) -> u32 {
    values.iter().enumerate().fold((0, f32::NEG_INFINITY), |best, (i, v)| if *v > best.1 { (i, *v) } else { best }).0 as u32
}

type Rows = Vec<Vec<f32>>;

/// The decoder of one utterance: the frames' keys and values for cross-attention, and the
/// tokens' own for self-attention.
pub struct Decoder<'a> {
    model: &'a Moonshine,
    cross: Vec<(Rows, Rows)>,
    own: Vec<(Rows, Rows)>,
    position: usize,
}

impl Decoder<'_> {
    /// The scores of the next token after `token`, at the next position.
    pub fn step(&mut self, token: u32) -> Result<Vec<f32>, String> {
        let m = self.model;
        let c = &m.config;
        let embedding = m.tensor("model.decoder.embed_tokens.weight")?;
        let mut x = embedding[token as usize * c.hidden..(token as usize + 1) * c.hidden].to_vec();
        let turns = rotation(c, self.position);
        for layer in 0..c.decoder_layers {
            let p = |n: &str| format!("model.decoder.layers.{layer}.{n}");
            let normed = layer_norm(&x, m.tensor(&p("input_layernorm.weight"))?);
            let mut q = m.linear(&p("self_attn.q_proj"), &normed)?;
            let mut k = m.linear(&p("self_attn.k_proj"), &normed)?;
            rotate(&mut q, c.head_dim, &turns);
            rotate(&mut k, c.head_dim, &turns);
            let (keys, values) = &mut self.own[layer];
            keys.push(k);
            values.push(m.linear(&p("self_attn.v_proj"), &normed)?);
            let attended = attend(&q, keys, values, c.head_dim);
            add(&mut x, &m.linear(&p("self_attn.o_proj"), &attended)?);

            let normed = layer_norm(&x, m.tensor(&p("post_attention_layernorm.weight"))?);
            let q = m.linear(&p("encoder_attn.q_proj"), &normed)?;
            let (keys, values) = &self.cross[layer];
            let attended = attend(&q, keys, values, c.head_dim);
            add(&mut x, &m.linear(&p("encoder_attn.o_proj"), &attended)?);

            let normed = layer_norm(&x, m.tensor(&p("final_layernorm.weight"))?);
            let both = m.linear(&p("mlp.fc1"), &normed)?;
            let (value, gate) = both.split_at(c.intermediate);
            let gated: Vec<f32> = value.iter().zip(gate).map(|(v, g)| silu(*g) * v).collect();
            add(&mut x, &m.linear(&p("mlp.fc2"), &gated)?);
        }
        self.position += 1;
        let normed = layer_norm(&x, m.tensor("model.decoder.norm.weight")?);
        let head = m.tensor("proj_out.weight").unwrap_or(embedding);
        Ok(head.chunks(c.hidden).map(|row| row.iter().zip(&normed).map(|(a, b)| a * b).sum()).collect())
    }
}
