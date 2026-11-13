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
#[derive(Clone)]
pub(crate) struct Planes {
    pub(crate) channels: usize,
    pub(crate) side: usize,
    pub(crate) values: Vec<f32>,
}

impl Planes {
    pub(crate) fn plane(&self, c: usize) -> &[f32] {
        &self.values[c * self.side * self.side..(c + 1) * self.side * self.side]
    }
}

/// A convolution of `kernel × kernel` (1 or 3, the latter padded by one) with bias.
pub(crate) fn conv(input: &Planes, weight: &[f32], bias: &[f32], kernel: usize) -> Planes {
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
pub(crate) fn group_norm(input: &mut Planes, weight: &[f32], bias: &[f32], swish: bool) {
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

/// A language model that gives its last hidden state instead of logits.
pub trait HiddenSteps {
    fn step_hidden(&mut self, token: u32, embedding: Option<&[f32]>) -> Result<Vec<f32>, String>;
    /// The image head's scores of each code, when the engine runs the head itself.
    fn image_logits(&mut self) -> Option<Result<Vec<f32>, String>> {
        None
    }
    /// The vector for a drawn code, when the engine runs the aligner itself.
    fn image_input(&mut self, _code: u32) -> Option<Result<Vec<f32>, String>> {
        None
    }
}

impl HiddenSteps for Engine {
    fn step_hidden(&mut self, token: u32, embedding: Option<&[f32]>) -> Result<Vec<f32>, String> {
        Engine::step_hidden(self, token, embedding)
    }
}

impl HiddenSteps for crate::ExactEngine {
    fn step_hidden(&mut self, token: u32, embedding: Option<&[f32]>) -> Result<Vec<f32>, String> {
        crate::ExactEngine::step_hidden(self, token, embedding)
    }
    fn image_logits(&mut self) -> Option<Result<Vec<f32>, String>> {
        Some(crate::ExactEngine::image_logits(self))
    }
    fn image_input(&mut self, code: u32) -> Option<Result<Vec<f32>, String>> {
        Some(crate::ExactEngine::image_input(self, code))
    }
}

impl Janus {
    pub fn open(dir: &Path) -> Result<Option<Self>, String> {
        let Some(config) = JanusConfig::read(dir)? else { return Ok(None) };
        Ok(Some(Self { config, dir: dir.to_path_buf(), weights: Weights::open(dir)? }))
    }

    fn tensor(&self, name: &str) -> Result<Vec<f32>, String> {
        self.weights.floats(&format!("model.{name}"))
    }

    /// The language model alone, ready to run: in floating point, or with `exact` the die's
    /// own integer arithmetic at that many bits.
    fn language_model(&self, exact: Option<u32>) -> Result<Box<dyn HiddenSteps>, String> {
        let config: Value = serde_json::from_str(&std::fs::read_to_string(self.dir.join("config.json")).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
        // The whole checkpoint's configuration, so the program carries the image heads.
        let program = compiler::compile_model(&config.to_string())?;
        Ok(match exact {
            Some(bits) => Box::new(crate::ExactEngine::new(program, Weights::open(&self.dir)?, bits)?),
            None => Box::new(Engine::new(program, Weights::open(&self.dir)?.with_gpu(), Numeric::Float)?),
        })
    }

    /// Draws `prompt_ids` (the text already framed by the chat template, ending in the
    /// token that opens an image): 576 codebook tokens, guided `guidance` times away from
    /// what the model would draw without the text, sampled at `temperature`.
    /// With `exact`, the language model and its two image heads, the generation head and
    /// the aligner, run in the die's arithmetic; the guidance's mixing, the sampling and the
    /// pixel decoder stay on the host.
    pub fn draw(&self, prompt_ids: &[u32], temperature: f32, seed: u64, exact: Option<u32>, mut progress: impl FnMut(usize, usize)) -> Result<Vec<u32>, String> {
        let c = &self.config;
        let (mut with_text, mut without) = (self.language_model(exact)?, self.language_model(exact)?);
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
            let (guided, plain) = match (with_text.image_logits(), without.image_logits()) {
                (Some(guided), Some(plain)) => (guided?, plain?),
                _ => (scores(&hidden), scores(&empty)),
            };
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
            // The same for both contexts, so it is made once.
            let input = match with_text.image_input(token as u32) {
                Some(input) => input?,
                None => {
                    let code = &embeddings[token * c.code_dim..(token + 1) * c.code_dim];
                    let up: Vec<f32> = linear(code, &fc1_w, &fc1_b).into_iter().map(gelu).collect();
                    linear(&up, &fc2_w, &fc2_b)
                }
            };
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

/// What Janus sees in a picture: a SigLIP encoder of 16×16 patches of a 384×384 image and
/// a small network that turns each patch into a vector the language model reads in place
/// of one `<image_placeholder>`.
pub struct JanusEye {
    weights: Weights,
    side: usize,
    patch: usize,
    hidden: usize,
    heads: usize,
    layers: usize,
    epsilon: f32,
    background: [f32; 3],
    /// The tokens around the image in the prompt: before it, in place of each patch, after it.
    pub begin: u32,
    pub placeholder: u32,
    pub end: u32,
}

/// Each row of `xs` times the matrix, plus the bias: one row per patch, the patches at once.
fn rows_times(xs: &[Vec<f32>], weight: &[f32], bias: &[f32]) -> Vec<Vec<f32>> {
    let inputs = xs.first().map_or(0, Vec::len);
    xs.par_iter().map(|x| bias.iter().enumerate().map(|(o, b)| weight[o * inputs..(o + 1) * inputs].iter().zip(x).map(|(w, v)| w * v).sum::<f32>() + b).collect()).collect()
}

fn layer_norm(x: &[f32], weight: &[f32], bias: &[f32], epsilon: f32) -> Vec<f32> {
    let n = x.len() as f32;
    let mean = x.iter().sum::<f32>() / n;
    let variance = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
    let scale = 1.0 / (variance + epsilon).sqrt();
    x.iter().zip(weight).zip(bias).map(|((v, w), b)| (v - mean) * scale * w + b).collect()
}

impl JanusEye {
    pub fn open(dir: &Path) -> Result<Option<Self>, String> {
        let config: Value = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
        let vision = &config["vision_config"];
        if config["model_type"].as_str() != Some("janus") || !vision.is_object() {
            return Ok(None);
        }
        let processor: Value = std::fs::read_to_string(dir.join("preprocessor_config.json")).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or(Value::Null);
        let number = |v: &Value, d: usize| v.as_u64().map_or(d, |n| n as usize);
        let tokenizer: Value = serde_json::from_str(&std::fs::read_to_string(dir.join("tokenizer.json")).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
        let token = |text: &str, default: u32| tokenizer["added_tokens"].as_array().and_then(|a| a.iter().find(|t| t["content"] == text)).and_then(|t| t["id"].as_u64()).map_or(default, |v| v as u32);
        let gray = |i: usize| processor["background_color"][i].as_f64().unwrap_or(127.0) as f32;
        Ok(Some(Self {
            weights: Weights::open(dir)?,
            side: number(&vision["image_size"], 384),
            patch: number(&vision["patch_size"], 16),
            hidden: number(&vision["hidden_size"], 1024),
            heads: number(&vision["num_attention_heads"], 16),
            layers: number(&vision["num_hidden_layers"], 24),
            epsilon: vision["layer_norm_eps"].as_f64().unwrap_or(1e-6) as f32,
            background: [gray(0), gray(1), gray(2)],
            begin: token("<begin_of_image>", 100_003),
            placeholder: token("<image_placeholder>", 100_581),
            end: token("<end_of_image>", 100_580),
        }))
    }

    /// How many vectors a picture becomes: one per patch.
    pub fn tokens(&self) -> usize {
        (self.side / self.patch).pow(2)
    }

    fn tensor(&self, name: &str) -> Result<Vec<f32>, String> {
        self.weights.floats(&format!("model.{name}"))
    }

    /// The picture in the encoder's frame: its longer side brought to the encoder's, the rest
    /// filled with the processor's gray, and each value taken from [0, 255] to [-1, 1].
    fn frame(&self, rgb: &[u8], width: usize, height: usize) -> Result<Vec<f32>, String> {
        if rgb.len() != width * height * 3 || width == 0 || height == 0 {
            return Err(format!("the {width}×{height} image should have {} RGB bytes", width * height * 3));
        }
        let side = self.side;
        let scale = side as f32 / width.max(height) as f32;
        let (w, h) = (((width as f32 * scale).round() as usize).clamp(1, side), ((height as f32 * scale).round() as usize).clamp(1, side));
        let (left, top) = ((side - w) / 2, (side - h) / 2);
        let mut planes = vec![0.0f32; 3 * side * side];
        for y in 0..side {
            for x in 0..side {
                for c in 0..3 {
                    let value = if (left..left + w).contains(&x) && (top..top + h).contains(&y) {
                        // Bilinear, from the centre of the output pixel.
                        let sx = ((x - left) as f32 + 0.5) / scale - 0.5;
                        let sy = ((y - top) as f32 + 0.5) / scale - 0.5;
                        let (x0, y0) = (sx.floor().max(0.0) as usize, sy.floor().max(0.0) as usize);
                        let (x1, y1) = ((x0 + 1).min(width - 1), (y0 + 1).min(height - 1));
                        let (fx, fy) = ((sx - x0 as f32).clamp(0.0, 1.0), (sy - y0 as f32).clamp(0.0, 1.0));
                        let at = |xx: usize, yy: usize| rgb[(yy * width + xx) * 3 + c] as f32;
                        (at(x0, y0) * (1.0 - fx) + at(x1, y0) * fx) * (1.0 - fy) + (at(x0, y1) * (1.0 - fx) + at(x1, y1) * fx) * fy
                    } else {
                        self.background[c]
                    };
                    planes[(c * side + y) * side + x] = (value / 255.0 - 0.5) / 0.5;
                }
            }
        }
        Ok(planes)
    }

    /// The vectors the language model reads for a picture, one per patch, row by row.
    pub fn encode(&self, rgb: &[u8], width: usize, height: usize) -> Result<Vec<Vec<f32>>, String> {
        let planes = self.frame(rgb, width, height)?;
        let (side, patch, hidden) = (self.side, self.patch, self.hidden);
        let grid = side / patch;
        // Patches through the convolution that has a patch for kernel and stride.
        let kernel = self.tensor("vision_model.embeddings.patch_embedding.weight")?;
        let kernel_bias = self.tensor("vision_model.embeddings.patch_embedding.bias")?;
        let place = self.tensor("vision_model.embeddings.position_embedding.weight")?;
        let mut x: Vec<Vec<f32>> = (0..grid * grid)
            .into_par_iter()
            .map(|p| {
                let (row, col) = (p / grid, p % grid);
                let pixels: Vec<f32> = (0..3).flat_map(|c| (0..patch).flat_map(move |dy| (0..patch).map(move |dx| (c, dy, dx)))).map(|(c, dy, dx)| planes[(c * side + row * patch + dy) * side + col * patch + dx]).collect();
                (0..hidden).map(|o| kernel[o * pixels.len()..(o + 1) * pixels.len()].iter().zip(&pixels).map(|(w, v)| w * v).sum::<f32>() + kernel_bias[o] + place[p * hidden + o]).collect()
            })
            .collect();
        let heads = self.heads;
        let head = hidden / heads;
        for layer in 0..self.layers {
            let p = |name: &str| self.tensor(&format!("vision_model.encoder.layers.{layer}.{name}"));
            let (w1, b1) = (p("layer_norm1.weight")?, p("layer_norm1.bias")?);
            let normed: Vec<Vec<f32>> = x.iter().map(|v| layer_norm(v, &w1, &b1, self.epsilon)).collect();
            let (q, k, v) = (rows_times(&normed, &p("self_attn.q_proj.weight")?, &p("self_attn.q_proj.bias")?), rows_times(&normed, &p("self_attn.k_proj.weight")?, &p("self_attn.k_proj.bias")?), rows_times(&normed, &p("self_attn.v_proj.weight")?, &p("self_attn.v_proj.bias")?));
            let scale = 1.0 / (head as f32).sqrt();
            // Every patch looks at every other, in each head.
            let attended: Vec<Vec<f32>> = (0..x.len())
                .into_par_iter()
                .map(|i| {
                    let mut out = vec![0.0f32; hidden];
                    for h in 0..heads {
                        let range = h * head..(h + 1) * head;
                        let scores: Vec<f32> = k.iter().map(|kj| q[i][range.clone()].iter().zip(&kj[range.clone()]).map(|(a, b)| a * b).sum::<f32>() * scale).collect();
                        let top = scores.iter().copied().fold(f32::MIN, f32::max);
                        let weights: Vec<f32> = scores.iter().map(|s| (s - top).exp()).collect();
                        let total: f32 = weights.iter().sum();
                        for (w, vj) in weights.iter().zip(&v) {
                            for (o, value) in out[range.clone()].iter_mut().zip(&vj[range.clone()]) {
                                *o += w / total * value;
                            }
                        }
                    }
                    out
                })
                .collect();
            let projected = rows_times(&attended, &p("self_attn.projection_layer.weight")?, &p("self_attn.projection_layer.bias")?);
            for (row, add) in x.iter_mut().zip(&projected) {
                row.iter_mut().zip(add).for_each(|(a, b)| *a += b);
            }
            let (w2, b2) = (p("layer_norm2.weight")?, p("layer_norm2.bias")?);
            let normed: Vec<Vec<f32>> = x.iter().map(|v| layer_norm(v, &w2, &b2, self.epsilon)).collect();
            let mut up = rows_times(&normed, &p("mlp.fc1.weight")?, &p("mlp.fc1.bias")?);
            up.par_iter_mut().for_each(|row| row.iter_mut().for_each(|v| *v = gelu(*v)));
            let down = rows_times(&up, &p("mlp.fc2.weight")?, &p("mlp.fc2.bias")?);
            for (row, add) in x.iter_mut().zip(&down) {
                row.iter_mut().zip(add).for_each(|(a, b)| *a += b);
            }
        }
        let (w, b) = (self.tensor("vision_model.post_layernorm.weight")?, self.tensor("vision_model.post_layernorm.bias")?);
        let x: Vec<Vec<f32>> = x.iter().map(|v| layer_norm(v, &w, &b, self.epsilon)).collect();
        // The aligner: up to the language model's width, a GELU, and one more layer.
        let mut aligned = rows_times(&x, &self.tensor("aligner.fc1.weight")?, &self.tensor("aligner.fc1.bias")?);
        aligned.par_iter_mut().for_each(|row| row.iter_mut().for_each(|v| *v = gelu(*v)));
        Ok(rows_times(&aligned, &self.tensor("aligner.hidden_layers.0.weight")?, &self.tensor("aligner.hidden_layers.0.bias")?))
    }
}
