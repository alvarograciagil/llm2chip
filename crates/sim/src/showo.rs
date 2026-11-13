//! Show-o: one transformer that reads text and fills in images. Its images are grids of
//! codes of MAGVIT-v2, a convolutional tokenizer; to retouch a picture, the codes of the
//! region to change are masked and the transformer fills them in a few steps, each keeping
//! the codes it is surest of, guided by a text and away from what it would draw without it.
//!
//! This is the floating-point reference, as `showlab/Show-o` runs it, step by step.

use crate::janus::{conv, group_norm, Planes};
use crate::weights::Weights;
use rayon::prelude::*;
use serde_json::Value;
use std::path::Path;

/// Phi-1.5's end of text, which also starts a text.
const END_OF_TEXT: u32 = 50256;
/// Tokens Show-o adds after the text's: padding, then the markers of images, videos and tasks.
const SPECIAL_TOKENS: u32 = 10;
const LAYER_NORM_EPSILON: f32 = 1e-5;

#[derive(Clone, Debug)]
pub struct ShowOConfig {
    /// The text tokenizer's size: padding and the special tokens follow, then the codes.
    pub text_vocabulary: u32,
    pub codebook: u32,
    pub mask_token: u32,
    /// Codes of an image, a square grid.
    pub image_tokens: usize,
    /// Longest text, in tokens.
    pub max_text: usize,
    /// Attention heads of Phi-1.5, and how many elements of each head rotate with the
    /// position: 32 and 32 of 64, unless `phi/config.json` says otherwise.
    pub heads: usize,
    pub rotary_fraction: f64,
}

impl ShowOConfig {
    pub fn read(dir: &Path) -> Result<Option<Self>, String> {
        let Ok(text) = std::fs::read_to_string(dir.join("config.json")) else { return Ok(None) };
        let config: Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
        if config["_class_name"] != "Showo" {
            return Ok(None);
        }
        let number = |key: &str| config[key].as_u64().ok_or_else(|| format!("falta «{key}» en la configuración de Show-o"));
        let phi: Value = std::fs::read_to_string(dir.join("phi/config.json")).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or(Value::Null);
        let heads = phi["num_attention_heads"].as_u64().unwrap_or(32) as usize;
        let rotary_fraction = phi["partial_rotary_factor"].as_f64().unwrap_or(0.5);
        let max_text = config["max_text"].as_u64().unwrap_or(128) as usize;
        Ok(Some(Self { text_vocabulary: number("llm_vocab_size")? as u32, codebook: number("codebook_size")? as u32, mask_token: number("mask_token_id")? as u32, image_tokens: number("num_vq_tokens")? as usize, max_text, heads, rotary_fraction }))
    }

    pub fn start_of_image(&self) -> u32 {
        self.text_vocabulary + 1
    }

    pub fn end_of_image(&self) -> u32 {
        self.text_vocabulary + 2
    }

    fn text_to_image(&self) -> u32 {
        self.text_vocabulary + 5
    }

    /// The token of code 0; the others follow.
    pub fn first_code(&self) -> u32 {
        self.text_vocabulary + SPECIAL_TOKENS
    }

    /// The text before an image, left-padded to its full length as the model was trained:
    /// the task, the text between its start and end, and nothing but padding before them.
    /// Gives its tokens and their positions, the padding left out: nothing reads it.
    pub fn text_prefix(&self, text: &[u32]) -> (Vec<u32>, Vec<u64>) {
        let mut ids = vec![self.text_to_image()];
        if text.first() != Some(&END_OF_TEXT) {
            ids.push(END_OF_TEXT);
        }
        ids.extend(text);
        ids.truncate(self.max_text);
        ids.push(END_OF_TEXT);
        let length = self.max_text + 1;
        let positions = (length - ids.len()..length).map(|p| p as u64).collect();
        (ids, positions)
    }

    /// The codes of a picture, those of the `region` (one flag a pixel) masked: the picture
    /// resized and cropped to the model's square.
    pub fn masked_codes(&self, magvit: &Magvit, rgb: &[u8], width: usize, height: usize, region: &[bool]) -> Result<Vec<Option<u32>>, String> {
        let side = self.side();
        let picture = square(&rgb.iter().map(|v| *v as f32 / 127.5 - 1.0).collect::<Vec<_>>(), 3, width, height, side, -0.5);
        let codes = magvit.encode(&picture)?;
        // The region on the grid of codes: bicubic samples at their centres, then halfway.
        let mask = square(&region.iter().map(|r| *r as u8 as f32).collect::<Vec<_>>(), 1, width, height, side, -0.5);
        let grid = self.grid();
        let on_grid = bicubic(&mask, 1, side, side, grid, grid, -0.75);
        Ok(codes.iter().zip(&on_grid).map(|(code, m)| (*m < 0.5).then_some(*code)).collect())
    }

    /// Codes per side of the grid.
    pub fn grid(&self) -> usize {
        (self.image_tokens as f64).sqrt().round() as usize
    }

    /// Pixels per side of an image: each code covers 16 × 16.
    pub fn side(&self) -> usize {
        16 * self.grid()
    }
}

struct Linear {
    weight: Vec<f32>,
    bias: Vec<f32>,
}

impl Linear {
    fn outputs(&self) -> usize {
        self.bias.len()
    }
}

struct Norm {
    weight: Vec<f32>,
    bias: Vec<f32>,
}

struct Layer {
    norm: Norm,
    q: Linear,
    k: Linear,
    v: Linear,
    q_norm: Norm,
    k_norm: Norm,
    dense: Linear,
    up: Linear,
    down: Linear,
}

/// Each layer's keys and values of the rows that went before, after their norms and
/// rotations, as the rows after them read them.
#[derive(Clone, Default)]
struct Cache {
    keys: Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
    rows: usize,
}

pub struct ShowO {
    pub config: ShowOConfig,
    hidden: usize,
    embed: Vec<f32>,
    layers: Vec<Layer>,
    final_norm: Norm,
    /// The rows of the output that score the codes, and their biases.
    code_head: Linear,
    /// MAGVIT-v2, when its folder is there: only pictures need it.
    tokenizer: Option<Magvit>,
}

/// Eight sums at once, so the products vectorise.
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut sums = [0f32; 8];
    let ((a_chunks, a_rest), (b_chunks, b_rest)) = (a.as_chunks::<8>(), b.as_chunks::<8>());
    let tail: f32 = a_rest.iter().zip(b_rest).map(|(x, y)| x * y).sum();
    for (x, y) in a_chunks.iter().zip(b_chunks) {
        for i in 0..8 {
            sums[i] += x[i] * y[i];
        }
    }
    sums.iter().sum::<f32>() + tail
}

/// Rows of `x`, one after another, times the matrix, plus the bias.
fn times(x: &[f32], layer: &Linear) -> Vec<f32> {
    let outputs = layer.outputs();
    let inputs = layer.weight.len() / outputs;
    let rows = x.len() / inputs;
    // Each output's weights are read once for every row.
    let columns: Vec<Vec<f32>> = (0..outputs)
        .into_par_iter()
        .with_min_len(8)
        .map(|o| {
            let w = &layer.weight[o * inputs..(o + 1) * inputs];
            (0..rows).map(|r| dot(w, &x[r * inputs..(r + 1) * inputs]) + layer.bias[o]).collect()
        })
        .collect();
    let mut out = vec![0f32; rows * outputs];
    for (o, column) in columns.iter().enumerate() {
        for (r, v) in column.iter().enumerate() {
            out[r * outputs + o] = *v;
        }
    }
    out
}

fn layer_norm(x: &mut [f32], width: usize, norm: &Norm) {
    x.par_chunks_mut(width).for_each(|row| {
        let n = width as f32;
        let mean = row.iter().sum::<f32>() / n;
        let variance = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
        let scale = 1.0 / (variance + LAYER_NORM_EPSILON).sqrt();
        for ((v, w), b) in row.iter_mut().zip(&norm.weight).zip(&norm.bias) {
            *v = (*v - mean) * scale * w + b;
        }
    });
}

/// GELU with the tanh approximation, Phi's `gelu_new`.
fn gelu_tanh(x: f32) -> f32 {
    0.5 * x * (1.0 + ((2.0 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x * x * x)).tanh())
}

impl ShowO {
    /// Show-o from its folder, with MAGVIT-v2 in `magvitv2` inside it, if it is one.
    pub fn open(dir: &Path) -> Result<Option<Self>, String> {
        let Some(config) = ShowOConfig::read(dir)? else { return Ok(None) };
        let weights = Weights::open(dir)?;
        let tensor = |name: &str| weights.floats(&format!("showo.{name}"));
        let linear = |name: &str| -> Result<Linear, String> { Ok(Linear { weight: tensor(&format!("{name}.weight"))?, bias: tensor(&format!("{name}.bias"))? }) };
        let norm = |name: &str| -> Result<Norm, String> { Ok(Norm { weight: tensor(&format!("{name}.weight"))?, bias: tensor(&format!("{name}.bias"))? }) };
        let final_norm = norm("model.final_layernorm")?;
        let hidden = final_norm.weight.len();
        let mut layers = Vec::new();
        while let Ok(layer_norm) = norm(&format!("model.layers.{}.input_layernorm", layers.len())) {
            let p = |name: &str| format!("model.layers.{}.{name}", layers.len());
            layers.push(Layer {
                norm: layer_norm,
                q: linear(&p("self_attn.q_proj"))?,
                k: linear(&p("self_attn.k_proj"))?,
                v: linear(&p("self_attn.v_proj"))?,
                q_norm: norm(&p("self_attn.q_layernorm"))?,
                k_norm: norm(&p("self_attn.k_layernorm"))?,
                dense: linear(&p("self_attn.dense"))?,
                up: linear(&p("mlp.fc1"))?,
                down: linear(&p("mlp.fc2"))?,
            });
        }
        let head = linear("lm_head")?;
        let first = config.first_code() as usize;
        let codes = config.codebook as usize;
        let code_head = Linear { weight: head.weight[first * hidden..(first + codes) * hidden].to_vec(), bias: head.bias[first..first + codes].to_vec() };
        let embed = tensor("model.embed_tokens.weight")?;
        let tokenizer = if dir.join("magvitv2").is_dir() { Some(Magvit::open(&dir.join("magvitv2"))?) } else { None };
        Ok(Some(Self { config, hidden, embed, layers, final_norm, code_head, tokenizer }))
    }

    /// The transformer over `ids` at `positions`, after the rows `before` holds: each row
    /// sees every row before, and those of `ids` only the earlier ones when `causal`, or all
    /// of them. Gives the final hidden states, normalised, and the cache with these rows.
    fn forward(&self, ids: &[u32], positions: &[u64], before: &Cache, causal: bool) -> (Vec<f32>, Cache) {
        let (hidden, rows) = (self.hidden, ids.len());
        let heads = self.config.heads;
        let head = hidden / heads;
        let rotary = (head as f64 * self.config.rotary_fraction).round() as usize;
        let mut x: Vec<f32> = ids.iter().flat_map(|id| self.embed[*id as usize * hidden..(*id as usize + 1) * hidden].iter().copied()).collect();
        let mut cache = Cache { rows: before.rows + rows, ..Default::default() };
        let frequencies: Vec<f32> = (0..rotary / 2).map(|i| 1.0 / 10000f32.powf((2 * i) as f32 / rotary as f32)).collect();
        for (index, layer) in self.layers.iter().enumerate() {
            let mut h = x.clone();
            layer_norm(&mut h, hidden, &layer.norm);
            let (mut q, mut k, v) = (times(&h, &layer.q), times(&h, &layer.k), times(&h, &layer.v));
            // Each head's queries and keys: normalised, then the first half rotated.
            for (data, norm) in [(&mut q, &layer.q_norm), (&mut k, &layer.k_norm)] {
                layer_norm(data, head, norm);
                data.par_chunks_mut(hidden).zip(positions).for_each(|(row, position)| {
                    for part in row.chunks_mut(head) {
                        let original = part[..rotary].to_vec();
                        let half = rotary / 2;
                        for i in 0..half {
                            let angle = *position as f32 * frequencies[i];
                            let (cos, sin) = (angle.cos(), angle.sin());
                            part[i] = original[i] * cos - original[half + i] * sin;
                            part[half + i] = original[half + i] * cos + original[i] * sin;
                        }
                    }
                });
            }
            let keys: Vec<f32> = before.keys.get(index).map_or(&[][..], Vec::as_slice).iter().chain(&k).copied().collect();
            let values: Vec<f32> = before.values.get(index).map_or(&[][..], Vec::as_slice).iter().chain(&v).copied().collect();
            let scale = 1.0 / (head as f32).sqrt();
            let mut attended = vec![0f32; rows * hidden];
            attended.par_chunks_mut(hidden).enumerate().for_each(|(r, out)| {
                let seen = before.rows + if causal { r + 1 } else { rows };
                for h in 0..heads {
                    let query = &q[r * hidden + h * head..][..head];
                    let scores: Vec<f32> = (0..seen).map(|j| dot(query, &keys[j * hidden + h * head..][..head]) * scale).collect();
                    let top = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let weights: Vec<f32> = scores.iter().map(|s| (s - top).exp()).collect();
                    let total: f32 = weights.iter().sum();
                    let slot = &mut out[h * head..(h + 1) * head];
                    for (j, w) in weights.iter().enumerate() {
                        let value = &values[j * hidden + h * head..][..head];
                        for (o, v) in slot.iter_mut().zip(value) {
                            *o += w / total * v;
                        }
                    }
                }
            });
            let attention = times(&attended, &layer.dense);
            let mut up = times(&h, &layer.up);
            up.par_iter_mut().for_each(|v| *v = gelu_tanh(*v));
            let mlp = times(&up, &layer.down);
            // Attention and the MLP read the same normalised input, side by side.
            x.par_iter_mut().zip(&attention).zip(&mlp).for_each(|((v, a), m)| *v += a + m);
            cache.keys.push(keys);
            cache.values.push(values);
        }
        layer_norm(&mut x, hidden, &self.final_norm);
        (x, cache)
    }

    /// MAGVIT-v2, which turns pictures into codes and back.
    pub fn tokenizer(&self) -> Result<&Magvit, String> {
        self.tokenizer.as_ref().ok_or_else(|| "falta MAGVIT-v2 en la carpeta «magvitv2»".to_string())
    }

    /// One pass of the model: each code's score at every place of the image, after `text`.
    /// The image is given as the model's tokens, a code's (`first_code() + code`) or the mask.
    pub fn scores(&self, text: &[u32], image: &[u32]) -> Vec<f32> {
        let (ids, positions) = self.text_prefix(text);
        let cache = self.forward(&ids, &positions, &Cache::default(), true).1;
        self.code_scores(&cache, image)
    }

    pub fn text_prefix(&self, text: &[u32]) -> (Vec<u32>, Vec<u64>) {
        self.config.text_prefix(text)
    }

    /// Each code's scores at every place of the image, from the image's tokens (a code, or
    /// the mask) after the text's cache.
    fn code_scores(&self, text: &Cache, image: &[u32]) -> Vec<f32> {
        let start = (self.config.max_text + 1) as u64;
        let ids: Vec<u32> = std::iter::once(self.config.start_of_image()).chain(image.iter().copied()).chain([self.config.end_of_image()]).collect();
        let positions: Vec<u64> = (start..start + ids.len() as u64).collect();
        let (hidden, _) = self.forward(&ids, &positions, text, false);
        // The image's places only, without its start and end markers.
        times(&hidden[self.hidden..(1 + image.len()) * self.hidden], &self.code_head)
    }

    /// Fills the masked places of `codes` (`None`), guided by `text`, in `steps` steps.
    /// `guidance` pushes the model's scores away from those without the text. `progress` hears
    /// of each step before it runs, and an error from it stops the filling with that error.
    pub fn fill(&self, codes: &[Option<u32>], text: &[u32], steps: usize, guidance: f32, seed: u64, progress: impl FnMut(usize, usize) -> Result<(), String>) -> Result<Vec<u32>, String> {
        let cache = |text: &[u32]| {
            let (ids, positions) = self.text_prefix(text);
            self.forward(&ids, &positions, &Cache::default(), true).1
        };
        let with_text = cache(text);
        let without_text = (guidance > 0.0).then(|| cache(&[]));
        let scores = |tokens: &[u32], with: bool| Ok(self.code_scores(if with { &with_text } else { without_text.as_ref().expect("sin guía no hay pasada sin texto") }, tokens));
        fill_with(&self.config, codes, steps, guidance, seed, progress, scores)
    }

    /// The codes of a picture, those of the `region` (one flag a pixel) masked: the picture
    /// resized and cropped to the model's square, as `inpaint` takes it.
    pub fn masked_codes(&self, rgb: &[u8], width: usize, height: usize, region: &[bool]) -> Result<Vec<Option<u32>>, String> {
        self.config.masked_codes(self.tokenizer()?, rgb, width, height, region)
    }
}

/// MaskGIT's filling of the masked places of `codes` (`None`), in `steps` steps, with the
/// scores `scores` gives for the image's tokens, with the text (`true`) or without it.
pub fn fill_with(
    c: &ShowOConfig,
    codes: &[Option<u32>],
    steps: usize,
    guidance: f32,
    seed: u64,
    mut progress: impl FnMut(usize, usize) -> Result<(), String>,
    mut model: impl FnMut(&[u32], bool) -> Result<Vec<f32>, String>,
) -> Result<Vec<u32>, String> {
    {
        if codes.len() != c.image_tokens {
            return Err(format!("una imagen de Show-o tiene {} códigos", c.image_tokens));
        }
        let mut rng = Rng(seed | 1);
        let n = codes.len();
        let codebook = c.codebook as usize;
        let mut current: Vec<Option<u32>> = codes.to_vec();
        let mut sampled: Vec<u32> = codes.iter().map(|c| c.unwrap_or(0)).collect();
        let mut temperature = 1f32;
        for step in 0..steps {
            progress(step, steps)?;
            let tokens: Vec<u32> = current.iter().map(|code| code.map_or(c.mask_token, |code| c.first_code() + code)).collect();
            let mut scores = model(&tokens, true)?;
            if guidance > 0.0 {
                let plain = model(&tokens, false)?;
                scores.iter_mut().zip(&plain).for_each(|(s, p)| *s = (1.0 + guidance) * *s - guidance * p);
            }
            // Each place's probabilities; a code is drawn from them where it is unknown.
            let probabilities: Vec<Vec<f32>> = scores
                .chunks(codebook)
                .map(|row| {
                    let top = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let exp: Vec<f32> = row.iter().map(|s| (s - top).exp()).collect();
                    let total: f32 = exp.iter().sum();
                    exp.iter().map(|e| e / total).collect()
                })
                .collect();
            let mut confidence = vec![0f32; n];
            for (place, row) in probabilities.iter().enumerate() {
                match current[place] {
                    Some(code) => {
                        sampled[place] = code;
                        confidence[place] = f32::MAX;
                    }
                    None => {
                        let draw = rng.uniform();
                        let mut total = 0f32;
                        let code = row.iter().position(|p| {
                            total += p;
                            total > draw
                        });
                        sampled[place] = code.unwrap_or(codebook - 1) as u32;
                        confidence[place] = row[sampled[place] as usize];
                    }
                }
            }
            let ratio = (step + 1) as f32 / steps as f32;
            let unknown = current.iter().filter(|c| c.is_none()).count();
            // How many stay masked after this step: fewer each time, along a cosine.
            let remaining = (n as f32 * (ratio * std::f32::consts::FRAC_PI_2).cos()).floor() as i64;
            let masked = remaining.min(unknown as i64 - 1).max(1) as usize;
            temperature *= 1.0 - ratio;
            // The least sure, with some noise that dies down, stay masked.
            let mut ranked: Vec<(f32, usize)> = confidence.iter().enumerate().map(|(place, p)| (p.max(1e-20).ln() + temperature * rng.gumbel(), place)).collect();
            ranked.sort_by(|a, b| a.0.total_cmp(&b.0));
            let cut = ranked[masked.min(n - 1)].0;
            for (place, code) in current.iter_mut().enumerate() {
                *code = Some(sampled[place]);
            }
            for (value, place) in &ranked {
                if *value < cut {
                    current[*place] = None;
                }
            }
        }
        progress(steps, steps)?;
        Ok(sampled)
    }
}

impl ShowO {
    /// Retouches the `region` of a picture (one flag a pixel, true to change) as `text`
    /// describes: the picture is resized and cropped to the model's square, and the result is
    /// that square, RGB, a row after another.
    #[allow(clippy::too_many_arguments)]
    pub fn inpaint(&self, rgb: &[u8], width: usize, height: usize, region: &[bool], text: &[u32], steps: usize, guidance: f32, seed: u64, progress: impl FnMut(usize, usize) -> Result<(), String>) -> Result<Vec<u8>, String> {
        let known = self.masked_codes(rgb, width, height, region)?;
        let filled = self.fill(&known, text, steps, guidance, seed, progress)?;
        self.tokenizer()?.decode(&filled)
    }

    /// A new picture from `text` alone: every code masked.
    pub fn draw(&self, text: &[u32], steps: usize, guidance: f32, seed: u64, progress: impl FnMut(usize, usize) -> Result<(), String>) -> Result<Vec<u8>, String> {
        let filled = self.fill(&vec![None; self.config.image_tokens], text, steps, guidance, seed, progress)?;
        self.tokenizer()?.decode(&filled)
    }
}

/// Bicubic resampling of `channels` planes of `width × height` to `to_width × to_height`,
/// sample points at pixel centres, edges repeated, with the kernel's `a`: PyTorch's
/// interpolation uses -0.75 and PIL's -0.5.
fn bicubic(values: &[f32], channels: usize, width: usize, height: usize, to_width: usize, to_height: usize, a: f32) -> Vec<f32> {
    let kernel = |x: f32| {
        let x = x.abs();
        if x <= 1.0 {
            ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0
        } else if x < 2.0 {
            ((a * x - 5.0 * a) * x + 8.0 * a) * x - 4.0 * a
        } else {
            0.0
        }
    };
    let taps = |out: usize, size: usize, to: usize| -> Vec<(usize, f32)> {
        let x = (out as f32 + 0.5) * size as f32 / to as f32 - 0.5;
        let base = x.floor();
        (-1..=2).map(|k| ((base as isize + k).clamp(0, size as isize - 1) as usize, kernel(x - (base + k as f32)))).collect()
    };
    let rows: Vec<Vec<(usize, f32)>> = (0..to_height).map(|y| taps(y, height, to_height)).collect();
    let cols: Vec<Vec<(usize, f32)>> = (0..to_width).map(|x| taps(x, width, to_width)).collect();
    let mut out = vec![0f32; channels * to_width * to_height];
    for c in 0..channels {
        let plane = &values[c * width * height..(c + 1) * width * height];
        for (y, row_taps) in rows.iter().enumerate() {
            for (x, col_taps) in cols.iter().enumerate() {
                let mut total = 0f32;
                for (sy, wy) in row_taps {
                    for (sx, wx) in col_taps {
                        total += wy * wx * plane[sy * width + sx];
                    }
                }
                out[(c * to_height + y) * to_width + x] = total;
            }
        }
    }
    out
}

/// Interleaved channels of a picture as planes, resized so its shorter side is `side`, and
/// the centre square cut out.
fn square(interleaved: &[f32], channels: usize, width: usize, height: usize, side: usize, a: f32) -> Vec<f32> {
    let planes: Vec<f32> = (0..channels).flat_map(|c| (0..width * height).map(move |i| interleaved[i * channels + c])).collect();
    let scale = side as f32 / width.min(height) as f32;
    let (to_width, to_height) = (((width as f32 * scale).round() as usize).max(side), ((height as f32 * scale).round() as usize).max(side));
    let resized = bicubic(&planes, channels, width, height, to_width, to_height, a);
    let (left, top) = ((to_width - side) / 2, (to_height - side) / 2);
    (0..channels).flat_map(|c| (0..side * side).map(move |i| (c, i))).map(|(c, i)| resized[(c * to_height + top + i / side) * to_width + left + i % side]).collect()
}

/// A small generator of the noise MaskGIT needs.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// In [0, 1).
    fn uniform(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }

    fn gumbel(&mut self) -> f32 {
        -(-(self.uniform().max(1e-20)).ln()).max(1e-20).ln()
    }
}

/// MAGVIT-v2: an image of 3 × 16n × 16n pixels in [-1, 1] to n × n codes of 13 bits, and back.
pub struct Magvit {
    weights: Weights,
}

impl Magvit {
    pub fn open(dir: &Path) -> Result<Self, String> {
        Ok(Self { weights: Weights::open(dir)? })
    }

    fn tensor(&self, name: &str) -> Result<Vec<f32>, String> {
        self.weights.floats(name)
    }

    fn has(&self, name: &str) -> bool {
        self.weights.floats(name).is_ok()
    }

    fn conv(&self, x: &Planes, name: &str, kernel: usize) -> Result<Planes, String> {
        Ok(conv(x, &self.tensor(&format!("{name}.weight"))?, &self.tensor(&format!("{name}.bias"))?, kernel))
    }

    fn norm(&self, x: &mut Planes, name: &str, swish: bool) -> Result<(), String> {
        group_norm(x, &self.tensor(&format!("{name}.weight"))?, &self.tensor(&format!("{name}.bias"))?, swish);
        Ok(())
    }

    fn resnet(&self, x: Planes, name: &str) -> Result<Planes, String> {
        let mut h = x.clone();
        self.norm(&mut h, &format!("{name}.norm1"), true)?;
        let mut h = self.conv(&h, &format!("{name}.conv1"), 3)?;
        self.norm(&mut h, &format!("{name}.norm2"), true)?;
        let h = self.conv(&h, &format!("{name}.conv2"), 3)?;
        let shortcut = if self.has(&format!("{name}.nin_shortcut.weight")) { self.conv(&x, &format!("{name}.nin_shortcut"), 1)? } else { x };
        Ok(Planes { values: shortcut.values.iter().zip(&h.values).map(|(a, b)| a + b).collect(), ..h })
    }

    /// One head of attention over every pixel, with its residual.
    fn attention(&self, x: Planes, name: &str) -> Result<Planes, String> {
        let mut h = x.clone();
        self.norm(&mut h, &format!("{name}.norm"), false)?;
        let (q, k, v) = (self.conv(&h, &format!("{name}.q"), 1)?, self.conv(&h, &format!("{name}.k"), 1)?, self.conv(&h, &format!("{name}.v"), 1)?);
        let (channels, area) = (x.channels, x.side * x.side);
        let scale = 1.0 / (channels as f32).sqrt();
        let mut out = vec![0f32; channels * area];
        let columns: Vec<Vec<f32>> = (0..area)
            .into_par_iter()
            .map(|i| {
                let scores: Vec<f32> = (0..area).map(|j| (0..channels).map(|c| q.values[c * area + i] * k.values[c * area + j]).sum::<f32>() * scale).collect();
                let top = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let weights: Vec<f32> = scores.iter().map(|s| (s - top).exp()).collect();
                let total: f32 = weights.iter().sum();
                (0..channels).map(|c| (0..area).map(|j| v.values[c * area + j] * weights[j]).sum::<f32>() / total).collect()
            })
            .collect();
        for (i, column) in columns.iter().enumerate() {
            for (c, value) in column.iter().enumerate() {
                out[c * area + i] = *value;
            }
        }
        let projected = self.conv(&Planes { channels, side: x.side, values: out }, &format!("{name}.proj_out"), 1)?;
        Ok(Planes { values: x.values.iter().zip(&projected.values).map(|(a, b)| a + b).collect(), ..projected })
    }

    /// Half the side: padded by one on the right and bottom, a 3 × 3 convolution every
    /// other pixel.
    fn downsample(&self, x: &Planes, name: &str) -> Result<Planes, String> {
        let (weight, bias) = (self.tensor(&format!("{name}.weight"))?, self.tensor(&format!("{name}.bias"))?);
        let (inputs, side) = (x.channels, x.side);
        let half = side / 2;
        let mut values = vec![0f32; bias.len() * half * half];
        values.par_chunks_mut(half * half).enumerate().for_each(|(o, out)| {
            out.fill(bias[o]);
            for c in 0..inputs {
                let plane = x.plane(c);
                for ky in 0..3 {
                    for kx in 0..3 {
                        let w = weight[((o * inputs + c) * 3 + ky) * 3 + kx];
                        for y in 0..half {
                            let sy = 2 * y + ky;
                            if sy >= side {
                                continue;
                            }
                            for xo in 0..half {
                                let sx = 2 * xo + kx;
                                if sx < side {
                                    out[y * half + xo] += w * plane[sy * side + sx];
                                }
                            }
                        }
                    }
                }
            }
        });
        Ok(Planes { channels: bias.len(), side: half, values })
    }

    /// Twice the side, each pixel repeated, then a 3 × 3 convolution.
    fn upsample(&self, x: &Planes, name: &str) -> Result<Planes, String> {
        let side = 2 * x.side;
        let values = (0..x.channels).flat_map(|c| (0..side * side).map(move |i| (c, i))).map(|(c, i)| x.values[c * x.side * x.side + (i / side / 2) * x.side + (i % side) / 2]).collect();
        self.conv(&Planes { channels: x.channels, side, values }, name, 3)
    }

    /// The codes of a picture of 3 planes of `side × side` in [-1, 1], row after row.
    pub fn encode(&self, planes: &[f32]) -> Result<Vec<u32>, String> {
        let side = ((planes.len() / 3) as f64).sqrt() as usize;
        let mut x = self.conv(&Planes { channels: 3, side, values: planes.to_vec() }, "encoder.conv_in", 3)?;
        let mut level = 0;
        while self.has(&format!("encoder.down.{level}.block.0.conv1.weight")) {
            let mut block = 0;
            while self.has(&format!("encoder.down.{level}.block.{block}.conv1.weight")) {
                x = self.resnet(x, &format!("encoder.down.{level}.block.{block}"))?;
                block += 1;
            }
            if self.has(&format!("encoder.down.{level}.downsample.conv.weight")) {
                x = self.downsample(&x, &format!("encoder.down.{level}.downsample.conv"))?;
            }
            level += 1;
        }
        x = self.resnet(x, "encoder.mid.block_1")?;
        x = self.attention(x, "encoder.mid.attn_1")?;
        x = self.resnet(x, "encoder.mid.block_2")?;
        self.norm(&mut x, "encoder.norm_out", true)?;
        let x = self.conv(&x, "encoder.conv_out", 3)?;
        let z = self.conv(&x, "encoder.quant_conv", 1)?;
        // Each pixel's 13 values become the bits of its code, the first the highest.
        let area = z.side * z.side;
        Ok((0..area).map(|i| (0..z.channels).fold(0u32, |code, c| (code << 1) | (z.values[c * area + i] > 0.0) as u32)).collect())
    }

    /// The picture of `codes`, a square grid, as 8-bit RGB a row after another.
    pub fn decode(&self, codes: &[u32]) -> Result<Vec<u8>, String> {
        let grid = (codes.len() as f64).sqrt() as usize;
        let bits = 13;
        let values = (0..bits).flat_map(|c| codes.iter().map(move |code| if code >> (bits - 1 - c) & 1 == 1 { 1.0 } else { -1.0 })).collect();
        let z = Planes { channels: bits, side: grid, values };
        let z = self.conv(&z, "decoder.post_quant_conv", 1)?;
        let mut x = self.conv(&z, "decoder.conv_in", 3)?;
        x = self.resnet(x, "decoder.mid.block_1")?;
        x = self.attention(x, "decoder.mid.attn_1")?;
        x = self.resnet(x, "decoder.mid.block_2")?;
        let levels = (0..).take_while(|l| self.has(&format!("decoder.up.{l}.block.0.conv1.weight"))).count();
        for level in (0..levels).rev() {
            let mut block = 0;
            while self.has(&format!("decoder.up.{level}.block.{block}.conv1.weight")) {
                x = self.resnet(x, &format!("decoder.up.{level}.block.{block}"))?;
                block += 1;
            }
            if self.has(&format!("decoder.up.{level}.upsample.conv.weight")) {
                x = self.upsample(&x, &format!("decoder.up.{level}.upsample.conv"))?;
            }
        }
        self.norm(&mut x, "decoder.norm_out", true)?;
        let x = self.conv(&x, "decoder.conv_out", 3)?;
        let area = x.side * x.side;
        Ok((0..area).flat_map(|i| (0..3).map(move |c| (c, i))).map(|(c, i)| (((x.values[c * area + i] + 1.0) / 2.0).clamp(0.0, 1.0) * 255.0) as u8).collect())
    }
}
