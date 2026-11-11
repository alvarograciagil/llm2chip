//! Text to image with Janus, in floating point. The language model writes an image as
//! 576 tokens of a visual codebook, one at a time and guided by the text without the
//! classifier; a VQ decoder of convolutions turns their 24×24 grid into 384×384 pixels.
//!
//! The language model is a plain Llama the chip already runs; the generation head, the
//! embedding of each drawn token and the decoder are what this module adds.

use crate::{Engine, Numeric, Weights};
use rayon::prelude::*;
use serde_json::Value;
use std::path::Path;

/// What a Janus checkpoint needs to draw.
#[derive(Clone, Debug)]
pub struct JanusConfig {
    pub tokens: usize,
    pub grid: usize,
    pub codebook: usize,
    pub code_dim: usize,
    pub base_channels: usize,
    pub multipliers: Vec<usize>,
    pub blocks: usize,
    pub latent_channels: usize,
    pub bos: u32,
    pub pad: u32,
    pub begin_image: u32,
    pub guidance: f32,
}

impl JanusConfig {
    /// The generator described in a checkpoint's folder, if it has one.
    pub fn read(dir: &Path) -> Result<Option<Self>, String> {
        let config: Value = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
        let vq = &config["vq_config"];
        if config["model_type"].as_str() != Some("janus") || !vq.is_object() {
            return Ok(None);
        }
        let generation: Value = std::fs::read_to_string(dir.join("generation_config.json")).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or(Value::Null);
        let number = |v: &Value, d: usize| v.as_u64().map_or(d, |n| n as usize);
        let grid = number(&vq["num_patches"], 24);
        Ok(Some(Self {
            tokens: grid * grid,
            grid,
            codebook: number(&vq["num_embeddings"], 16384),
            code_dim: number(&vq["embed_dim"], 8),
            base_channels: number(&vq["base_channels"], 128),
            multipliers: vq["channel_multiplier"].as_array().map(|a| a.iter().filter_map(Value::as_u64).map(|m| m as usize).collect()).unwrap_or_else(|| vec![1, 1, 2, 2, 4]),
            blocks: number(&vq["num_res_blocks"], 2),
            latent_channels: number(&vq["latent_channels"], 256),
            bos: number(&config["text_config"]["bos_token_id"], 100_000) as u32,
            pad: number(&config["text_config"]["pad_token_id"], 100_002) as u32,
            begin_image: number(&generation["generation_kwargs"]["boi_token_id"], 100_003) as u32,
            guidance: generation["guidance_scale"].as_f64().unwrap_or(5.0) as f32,
        }))
    }

    /// Side, in pixels, of the images it draws: the grid doubled once per level but the last.
    pub fn side(&self) -> usize {
        self.grid << (self.multipliers.len() - 1)
    }
}

fn gelu(v: f32) -> f32 {
    0.5 * v * (1.0 + crate::vision::erf(v / std::f32::consts::SQRT_2))
}

fn linear(x: &[f32], weight: &[f32], bias: &[f32]) -> Vec<f32> {
    let inputs = x.len();
    bias.par_iter().enumerate().map(|(o, b)| weight[o * inputs..(o + 1) * inputs].iter().zip(x).map(|(w, v)| w * v).sum::<f32>() + b).collect()
}

/// A picture as channels of rows of pixels.
struct Planes {
    channels: usize,
    side: usize,
    values: Vec<f32>,
}

impl Planes {
    fn plane(&self, c: usize) -> &[f32] {
        &self.values[c * self.side * self.side..(c + 1) * self.side * self.side]
    }
}

/// A convolution of `kernel × kernel` (1 or 3, the latter padded by one) with bias.
fn conv(input: &Planes, weight: &[f32], bias: &[f32], kernel: usize) -> Planes {
    let (inputs, side) = (input.channels, input.side);
    let outputs = bias.len();
    let mut values = vec![0f32; outputs * side * side];
    values.par_chunks_mut(side * side).enumerate().for_each(|(o, out)| {
        out.fill(bias[o]);
        for c in 0..inputs {
            let plane = input.plane(c);
            let taps = &weight[(o * inputs + c) * kernel * kernel..(o * inputs + c + 1) * kernel * kernel];
            if kernel == 1 {
                let w = taps[0];
                out.iter_mut().zip(plane).for_each(|(y, x)| *y += w * x);
                continue;
            }
            for ky in 0..3 {
                for kx in 0..3 {
                    let w = taps[ky * 3 + kx];
                    // Each output row adds a shifted row of the input; the border stays zero.
                    for y in 0..side {
                        let sy = y as isize + ky as isize - 1;
                        if sy < 0 || sy >= side as isize {
                            continue;
                        }
                        let row = &plane[sy as usize * side..(sy as usize + 1) * side];
                        let target = &mut out[y * side..(y + 1) * side];
                        let (from, to) = match kx {
                            0 => (0, 1),
                            1 => (0, 0),
                            _ => (1, 0),
                        };
                        let len = side - from - to;
                        for (t, x) in target[to..to + len].iter_mut().zip(&row[from..from + len]) {
                            *t += w * x;
                        }
                    }
                }
            }
        }
    });
    Planes { channels: outputs, side, values }
}

/// Group normalisation over 32 groups of channels, with the swish after it when asked.
fn group_norm(input: &mut Planes, weight: &[f32], bias: &[f32], swish: bool) {
    let per_group = input.channels / 32;
    let area = input.side * input.side;
    input.values.par_chunks_mut(per_group * area).enumerate().for_each(|(g, values)| {
        let n = values.len() as f64;
        let mean = values.iter().map(|v| *v as f64).sum::<f64>() / n;
        let variance = values.iter().map(|v| (*v as f64 - mean).powi(2)).sum::<f64>() / n;
        let inverse = 1.0 / (variance + 1e-6).sqrt();
        for (i, chunk) in values.chunks_mut(area).enumerate() {
            let c = g * per_group + i;
            for v in chunk.iter_mut() {
                let y = ((*v as f64 - mean) * inverse) as f32 * weight[c] + bias[c];
                *v = if swish { y / (1.0 + (-y).exp()) } else { y };
            }
        }
    });
}

/// The generator: the language model's checkpoint holds every part.
pub struct Janus {
    pub config: JanusConfig,
    dir: std::path::PathBuf,
    weights: Weights,
}

impl Janus {
    pub fn open(dir: &Path) -> Result<Option<Self>, String> {
        let Some(config) = JanusConfig::read(dir)? else { return Ok(None) };
        Ok(Some(Self { config, dir: dir.to_path_buf(), weights: Weights::open(dir)? }))
    }

    fn tensor(&self, name: &str) -> Result<Vec<f32>, String> {
        self.weights.floats(&format!("model.{name}"))
    }

    /// The language model alone, ready to run.
    fn language_model(&self) -> Result<Engine, String> {
        let config: Value = serde_json::from_str(&std::fs::read_to_string(self.dir.join("config.json")).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
        let program = compiler::compile_model(&config["text_config"].to_string())?;
        Engine::new(program, Weights::open(&self.dir)?.with_gpu(), Numeric::Float)
    }

    /// Draws `prompt_ids` (the text already framed by the chat template, ending in the
    /// token that opens an image): 576 codebook tokens, guided `guidance` times away from
    /// what the model would draw without the text, sampled at `temperature`.
    pub fn draw(&self, prompt_ids: &[u32], temperature: f32, seed: u64, mut progress: impl FnMut(usize, usize)) -> Result<Vec<u32>, String> {
        let c = &self.config;
        let (mut with_text, mut without) = (self.language_model()?, self.language_model()?);
        // Without the text: only the start and the image's opening token remain.
        let blank: Vec<u32> = prompt_ids.iter().map(|t| if *t == c.bos || *t == c.begin_image { *t } else { c.pad }).collect();
        let (mut hidden, mut empty) = (Vec::new(), Vec::new());
        for (t, b) in prompt_ids.iter().zip(&blank) {
            hidden = with_text.step_hidden(*t, None)?;
            empty = without.step_hidden(*b, None)?;
        }
        let (proj_w, proj_b) = (self.tensor("generation_head.proj_out.weight")?, self.tensor("generation_head.proj_out.bias")?);
        let (head_w, head_b) = (self.tensor("generation_head.vision_head.weight")?, self.tensor("generation_head.vision_head.bias")?);
        let embeddings = self.tensor("generation_embeddings.weight")?;
        let (fc1_w, fc1_b) = (self.tensor("generation_aligner.fc1.weight")?, self.tensor("generation_aligner.fc1.bias")?);
        let (fc2_w, fc2_b) = (self.tensor("generation_aligner.hidden_layers.0.weight")?, self.tensor("generation_aligner.hidden_layers.0.bias")?);
        let scores = |h: &[f32]| {
            let projected: Vec<f32> = linear(h, &proj_w, &proj_b).into_iter().map(gelu).collect();
            linear(&projected, &head_w, &head_b)
        };
        let mut state = seed | 1;
        let mut random = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        let mut drawn = Vec::with_capacity(c.tokens);
        for i in 0..c.tokens {
            let (guided, plain) = (scores(&hidden), scores(&empty));
            let logits: Vec<f32> = guided.iter().zip(&plain).map(|(g, p)| p + c.guidance * (g - p)).collect();
            let token = if temperature <= 0.0 {
                logits.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map_or(0, |(i, _)| i)
            } else {
                let top = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let weights: Vec<f64> = logits.iter().map(|l| (((l - top) / temperature) as f64).exp()).collect();
                let mut target = random() * weights.iter().sum::<f64>();
                weights.iter().position(|w| {
                    target -= w;
                    target <= 0.0
                }).unwrap_or(weights.len() - 1)
            };
            drawn.push(token as u32);
            progress(i + 1, c.tokens);
            // The drawn token goes back in as the aligned embedding of its code.
            let code = &embeddings[token * c.code_dim..(token + 1) * c.code_dim];
            let up: Vec<f32> = linear(code, &fc1_w, &fc1_b).into_iter().map(gelu).collect();
            let input = linear(&up, &fc2_w, &fc2_b);
            if i + 1 < c.tokens {
                hidden = with_text.step_hidden(0, Some(&input))?;
                empty = without.step_hidden(0, Some(&input))?;
            }
        }
        Ok(drawn)
    }

    /// Turns the drawn tokens into RGB pixels, row by row.
    pub fn decode(&self, tokens: &[u32]) -> Result<Vec<u8>, String> {
        let c = &self.config;
        let book = self.tensor("vqmodel.quantize.embedding.weight")?;
        // Each code, normalised, as `code_dim` planes of the grid.
        let area = c.grid * c.grid;
        let mut values = vec![0f32; c.code_dim * area];
        for (at, token) in tokens.iter().enumerate() {
            let code = &book[*token as usize * c.code_dim..(*token as usize + 1) * c.code_dim];
            let norm = code.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
            for (d, v) in code.iter().enumerate() {
                values[d * area + at] = v / norm;
            }
        }
        let mut x = Planes { channels: c.code_dim, side: c.grid, values };
        let w = |name: &str| self.tensor(&format!("vqmodel.{name}"));
        x = conv(&x, &w("post_quant_conv.weight")?, &w("post_quant_conv.bias")?, 1);
        let d = |name: &str| w(&format!("decoder.{name}"));
        x = conv(&x, &d("conv_in.weight")?, &d("conv_in.bias")?, 3);
        let resnet = |x: Planes, name: &str| -> Result<Planes, String> {
            let p = |part: &str| d(&format!("{name}.{part}"));
            let mut h = Planes { channels: x.channels, side: x.side, values: x.values.clone() };
            group_norm(&mut h, &p("norm1.weight")?, &p("norm1.bias")?, true);
            let mut h = conv(&h, &p("conv1.weight")?, &p("conv1.bias")?, 3);
            group_norm(&mut h, &p("norm2.weight")?, &p("norm2.bias")?, true);
            let h = conv(&h, &p("conv2.weight")?, &p("conv2.bias")?, 3);
            let shortcut = if x.channels != h.channels { conv(&x, &p("nin_shortcut.weight")?, &p("nin_shortcut.bias")?, 1) } else { x };
            Ok(Planes { channels: h.channels, side: h.side, values: shortcut.values.iter().zip(&h.values).map(|(a, b)| a + b).collect() })
        };
        let attention = |x: Planes, name: &str| -> Result<Planes, String> {
            let p = |part: &str| d(&format!("{name}.{part}"));
            let mut h = Planes { channels: x.channels, side: x.side, values: x.values.clone() };
            group_norm(&mut h, &p("norm.weight")?, &p("norm.bias")?, false);
            let (q, k, v) = (conv(&h, &p("q.weight")?, &p("q.bias")?, 1), conv(&h, &p("k.weight")?, &p("k.bias")?, 1), conv(&h, &p("v.weight")?, &p("v.bias")?, 1));
            let (channels, n) = (x.channels, x.side * x.side);
            let scale = 1.0 / (channels as f32).sqrt();
            let mut out = vec![0f32; channels * n];
            // Every position attends to every other, over all the channels at once.
            let rows: Vec<Vec<f32>> = (0..n).into_par_iter().map(|i| {
                let scores: Vec<f32> = (0..n).map(|j| (0..channels).map(|ch| q.values[ch * n + i] * k.values[ch * n + j]).sum::<f32>() * scale).collect();
                let top = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let weights: Vec<f32> = scores.iter().map(|s| (s - top).exp()).collect();
                let total: f32 = weights.iter().sum();
                (0..channels).map(|ch| weights.iter().enumerate().map(|(j, w)| w * v.values[ch * n + j]).sum::<f32>() / total).collect()
            }).collect();
            for (i, row) in rows.iter().enumerate() {
                for (ch, value) in row.iter().enumerate() {
                    out[ch * n + i] = *value;
                }
            }
            let attended = conv(&Planes { channels, side: x.side, values: out }, &p("proj_out.weight")?, &p("proj_out.bias")?, 1);
            Ok(Planes { channels, side: x.side, values: x.values.iter().zip(&attended.values).map(|(a, b)| a + b).collect() })
        };
        x = resnet(x, "mid.block_1")?;
        x = attention(x, "mid.attn_1")?;
        x = resnet(x, "mid.block_2")?;
        let levels = c.multipliers.len();
        for level in 0..levels {
            for block in 0..=c.blocks {
                x = resnet(x, &format!("up.{level}.block.{block}"))?;
                // Only the coarsest level, which comes first, attends.
                if level == 0 {
                    x = attention(x, &format!("up.{level}.attn.{block}"))?;
                }
            }
            if level + 1 < levels {
                // Nearest-neighbour doubling, then a convolution.
                let side = x.side * 2;
                let mut values = vec![0f32; x.channels * side * side];
                for ch in 0..x.channels {
                    for y in 0..side {
                        for xx in 0..side {
                            values[(ch * side + y) * side + xx] = x.values[(ch * x.side + y / 2) * x.side + xx / 2];
                        }
                    }
                }
                let doubled = Planes { channels: x.channels, side, values };
                x = conv(&doubled, &d(&format!("up.{level}.upsample.conv.weight"))?, &d(&format!("up.{level}.upsample.conv.bias"))?, 3);
            }
        }
        group_norm(&mut x, &d("norm_out.weight")?, &d("norm_out.bias")?, true);
        let rgb = conv(&x, &d("conv_out.weight")?, &d("conv_out.bias")?, 3);
        let n = rgb.side * rgb.side;
        Ok((0..n).flat_map(|i| (0..3).map(move |ch| (ch, i))).map(|(ch, i)| (((rgb.values[ch * n + i].clamp(-1.0, 1.0) + 1.0) * 127.5).round()) as u8).collect())
    }
}
