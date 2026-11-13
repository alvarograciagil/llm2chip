//! The image encoder of Qwen3.5-style models, in floating point: what the chip has to
//! reproduce to take images. An image becomes patches of 16×16 pixels, a ViT turns them
//! into vectors, and every 2×2 of them merges into one token the language model reads in
//! place of an `<|image_pad|>`, with three positions (time, row, column) for its rotary
//! embedding.

use crate::Weights;
use rayon::prelude::*;
use serde_json::Value;
use std::path::Path;

/// What the encoder of a checkpoint looks like.
#[derive(Clone, Debug)]
pub struct VisionConfig {
    pub depth: usize,
    pub hidden: usize,
    pub heads: usize,
    pub intermediate: usize,
    pub patch: usize,
    pub temporal: usize,
    pub merge: usize,
    pub out_hidden: usize,
    /// Side of the square grid of learned position embeddings.
    pub position_side: usize,
    pub rope_theta: f32,
    /// The image may hold this many pixels at least and at most once resized.
    pub min_pixels: usize,
    pub max_pixels: usize,
    pub image_token: u32,
    pub vision_start: u32,
    pub vision_end: u32,
    /// How many of the language model's rotary frequencies follow each position: time,
    /// row and column, interleaved.
    pub mrope_section: [usize; 3],
}

impl VisionConfig {
    /// The encoder described in a checkpoint's folder, if it has one.
    pub fn read(dir: &Path) -> Result<Option<Self>, String> {
        let text = std::fs::read_to_string(dir.join("config.json")).map_err(|e| e.to_string())?;
        let config: Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
        let vision = &config["vision_config"];
        if !vision.is_object() || !has_encoder(dir) {
            return Ok(None);
        }
        let number = |value: &Value, default: usize| value.as_u64().map_or(default, |v| v as usize);
        let preprocessor: Value = std::fs::read_to_string(dir.join("preprocessor_config.json")).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or(Value::Null);
        let rope = &config["text_config"]["rope_parameters"];
        let section: Vec<usize> = rope["mrope_section"].as_array().map(|a| a.iter().filter_map(Value::as_u64).map(|v| v as usize).collect()).unwrap_or_default();
        Ok(Some(Self {
            depth: number(&vision["depth"], 27),
            hidden: number(&vision["hidden_size"], 1152),
            heads: number(&vision["num_heads"], 16),
            intermediate: number(&vision["intermediate_size"], 4304),
            patch: number(&vision["patch_size"], 16),
            temporal: number(&vision["temporal_patch_size"], 2),
            merge: number(&vision["spatial_merge_size"], 2),
            out_hidden: number(&vision["out_hidden_size"], 5120),
            position_side: (number(&vision["num_position_embeddings"], 2304) as f64).sqrt() as usize,
            rope_theta: vision["rope_parameters"]["rope_theta"].as_f64().unwrap_or(10_000.0) as f32,
            min_pixels: number(&preprocessor["size"]["shortest_edge"], 65_536),
            max_pixels: number(&preprocessor["size"]["longest_edge"], 16_777_216),
            image_token: number(&config["image_token_id"], 248_056) as u32,
            vision_start: number(&config["vision_start_token_id"], 248_053) as u32,
            vision_end: number(&config["vision_end_token_id"], 248_054) as u32,
            mrope_section: if section.len() == 3 { [section[0], section[1], section[2]] } else { [11, 11, 10] },
        }))
    }

    /// Pixels per side of each merged token.
    pub fn factor(&self) -> usize {
        self.patch * self.merge
    }

    /// The same encoder for images of at most `patches` patches, as a chip that carries it
    /// takes them: larger ones are resized down to fit.
    pub fn within(&self, patches: usize) -> Self {
        let most = patches * self.patch * self.patch;
        Self { max_pixels: self.max_pixels.min(most), min_pixels: self.min_pixels.min(most), ..self.clone() }
    }

    /// The size an image is resized to: sides multiple of the factor, its area within
    /// bounds, its shape kept as well as that allows.
    pub fn fitted_size(&self, width: usize, height: usize) -> (usize, usize) {
        let factor = self.factor() as f64;
        let (w, h) = (width as f64, height as f64);
        let round = |v: f64| ((v / factor).round() * factor).max(factor);
        let shrunk = || {
            let beta = (w * h / self.max_pixels as f64).sqrt();
            (((w / beta / factor).floor() * factor).max(factor), ((h / beta / factor).floor() * factor).max(factor))
        };
        let (mut fw, mut fh) = (round(w), round(h));
        if fw * fh > self.max_pixels as f64 {
            (fw, fh) = shrunk();
        } else if fw * fh < self.min_pixels as f64 {
            let beta = (self.min_pixels as f64 / (w * h)).sqrt();
            fw = (w * beta / factor).ceil() * factor;
            fh = (h * beta / factor).ceil() * factor;
            // Rounding up may overshoot the most, which wins.
            if fw * fh > self.max_pixels as f64 {
                (fw, fh) = shrunk();
            }
        }
        (fw as usize, fh as usize)
    }
}

/// An image ready for the encoder: its patches in merge-block order, each as channel,
/// time, row and column of pixels, and its grid of patches.
pub struct Patches {
    pub values: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
}

impl Patches {
    /// Tokens the language model reads for this image.
    pub fn tokens(&self, merge: usize) -> usize {
        self.rows * self.cols / (merge * merge)
    }
}

/// Bilinear resampling of 8-bit RGB pixels, with the sample points at pixel centres.
fn resize(rgb: &[u8], width: usize, height: usize, to_width: usize, to_height: usize) -> Vec<f32> {
    let mut out = vec![0f32; to_width * to_height * 3];
    let sample = |x: f32, size: usize| {
        let x = x.clamp(0.0, (size - 1) as f32);
        let low = x.floor() as usize;
        (low, (low + 1).min(size - 1), x - low as f32)
    };
    for y in 0..to_height {
        let (y0, y1, fy) = sample((y as f32 + 0.5) * height as f32 / to_height as f32 - 0.5, height);
        for x in 0..to_width {
            let (x0, x1, fx) = sample((x as f32 + 0.5) * width as f32 / to_width as f32 - 0.5, width);
            for c in 0..3 {
                let at = |yy: usize, xx: usize| rgb[(yy * width + xx) * 3 + c] as f32;
                let top = at(y0, x0) * (1.0 - fx) + at(y0, x1) * fx;
                let bottom = at(y1, x0) * (1.0 - fx) + at(y1, x1) * fx;
                out[(y * to_width + x) * 3 + c] = top * (1.0 - fy) + bottom * fy;
            }
        }
    }
    out
}

/// Turns RGB pixels, row by row, into the encoder's patches. An image already of a size
/// the encoder takes is used as it is; any other is resized bilinearly.
pub fn patches(config: &VisionConfig, rgb: &[u8], width: usize, height: usize) -> Result<Patches, String> {
    if rgb.len() != width * height * 3 {
        return Err(format!("the {width}×{height} image should have {} RGB bytes", width * height * 3));
    }
    let (to_width, to_height) = config.fitted_size(width, height);
    let pixels = if (to_width, to_height) == (width, height) { rgb.iter().map(|v| *v as f32).collect() } else { resize(rgb, width, height, to_width, to_height) };
    let (p, m, t) = (config.patch, config.merge, config.temporal);
    let (rows, cols) = (to_height / p, to_width / p);
    let size = 3 * t * p * p;
    let mut values = Vec::with_capacity(rows * cols * size);
    for block_row in 0..rows / m {
        for block_col in 0..cols / m {
            for in_row in 0..m {
                for in_col in 0..m {
                    let (row, col) = (block_row * m + in_row, block_col * m + in_col);
                    for c in 0..3 {
                        // A still image is the same frame at every time step.
                        for _ in 0..t {
                            for py in 0..p {
                                for px in 0..p {
                                    let value = pixels[((row * p + py) * to_width + col * p + px) * 3 + c];
                                    // Scaled to [0, 1], then normalised with mean and deviation 0.5.
                                    values.push((value / 255.0 - 0.5) / 0.5);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(Patches { values, rows, cols })
}

/// `y[n][o] = Σ_i x[n][i] · w[o][i] + bias[o]` for every row of `x`.
fn linear(x: &[f32], inputs: usize, weight: &[f32], bias: &[f32]) -> Vec<f32> {
    let (count, outputs) = (x.len() / inputs, bias.len());
    let mut y = vec![0f32; count * outputs];
    y.par_chunks_mut(outputs).zip(x.par_chunks(inputs)).for_each(|(out, row)| {
        for (o, value) in out.iter_mut().enumerate() {
            let w = &weight[o * inputs..(o + 1) * inputs];
            *value = w.iter().zip(row).map(|(a, b)| a * b).sum::<f32>() + bias[o];
        }
    });
    y
}

fn layer_norm(x: &mut [f32], width: usize, weight: &[f32], bias: &[f32]) {
    x.par_chunks_mut(width).for_each(|row| {
        let mean = row.iter().sum::<f32>() / width as f32;
        let variance = row.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / width as f32;
        let inverse = 1.0 / (variance + 1e-6).sqrt();
        for (i, v) in row.iter_mut().enumerate() {
            *v = (*v - mean) * inverse * weight[i] + bias[i];
        }
    });
}

fn gelu_tanh(v: f32) -> f32 {
    0.5 * v * (1.0 + ((2.0 / std::f32::consts::PI).sqrt() * (v + 0.044715 * v * v * v)).tanh())
}

fn gelu(v: f32) -> f32 {
    0.5 * v * (1.0 + erf(v / std::f32::consts::SQRT_2))
}

/// The error function, to float precision (Abramowitz and Stegun 7.1.26 is not enough
/// near zero, so this sums its series there).
pub(crate) fn erf(x: f32) -> f32 {
    let x = x as f64;
    let result = if x.abs() < 2.0 {
        let (mut term, mut sum, mut n) = (x, x, 0.0);
        while term.abs() > 1e-12 * sum.abs().max(1e-30) {
            n += 1.0;
            term *= -x * x / n;
            sum += term / (2.0 * n + 1.0);
        }
        sum * 2.0 / std::f64::consts::PI.sqrt()
    } else {
        // Continued fraction for the tail.
        let z = x.abs();
        let mut f = 0.0;
        for k in (1..60).rev() {
            f = k as f64 / 2.0 / (z + f);
        }
        let tail = (-z * z).exp() / std::f64::consts::PI.sqrt() / (z + f);
        (1.0 - tail).copysign(x)
    };
    result as f32
}

/// The image encoder with its weights.
pub struct VisionTower {
    pub config: VisionConfig,
    weights: Weights,
}

/// Whether the checkpoint carries the image encoder: in a file of its own, or among the
/// language model's tensors, as the Qwen3.6 mixtures of experts keep it.
fn has_encoder(dir: &Path) -> bool {
    if dir.join("model-vision.safetensors").exists() {
        return true;
    }
    let index = std::fs::read_to_string(dir.join("model.safetensors.index.json")).ok().and_then(|t| serde_json::from_str::<Value>(&t).ok());
    if let Some(map) = index.as_ref().and_then(|i| i["weight_map"].as_object()) {
        return map.keys().any(|k| k.starts_with("vision_tower."));
    }
    Weights::open(dir).is_ok_and(|w| w.raw_tensors().iter().any(|(name, ..)| name.starts_with("vision_tower.")))
}

impl VisionTower {
    pub fn open(dir: &Path) -> Result<Option<Self>, String> {
        let Some(config) = VisionConfig::read(dir)? else { return Ok(None) };
        let weights = if dir.join("model-vision.safetensors").exists() { Weights::open_vision(dir)? } else { Weights::open(dir)? };
        Ok(Some(Self { config, weights }))
    }

    fn tensor(&self, name: &str) -> Result<Vec<f32>, String> {
        self.weights.floats(&format!("vision_tower.{name}"))
    }

    /// Each patch's learned position, resampled bilinearly (corners aligned) to the image's
    /// grid, in the encoder's order: what the host adds to each patch.
    pub fn positions(&self, rows: usize, cols: usize) -> Result<Vec<Vec<f32>>, String> {
        let c = &self.config;
        let (table, side, hidden) = (self.tensor("pos_embed.weight")?, c.position_side, c.hidden);
        let axis = |index: usize, size: usize| {
            let source = if size > 1 { index as f32 * (side - 1) as f32 / (size - 1) as f32 } else { 0.0 };
            let low = source.floor() as usize;
            (low.min(side - 1), (low + 1).min(side - 1), source - low as f32)
        };
        Ok(merge_order(rows, cols, c.merge)
            .iter()
            .map(|(row, col)| {
                let (r0, r1, fr) = axis(*row, rows);
                let (c0, c1, fc) = axis(*col, cols);
                let mut out = vec![0f32; hidden];
                for (r, col_tap, weight) in [(r0, c0, (1.0 - fr) * (1.0 - fc)), (r0, c1, (1.0 - fr) * fc), (r1, c0, fr * (1.0 - fc)), (r1, c1, fr * fc)] {
                    for (v, e) in out.iter_mut().zip(&table[(r * side + col_tap) * hidden..(r * side + col_tap + 1) * hidden]) {
                        *v += weight * e;
                    }
                }
                out
            })
            .collect())
    }

    /// The tokens the language model reads for an image: one vector of `out_hidden`
    /// values for each 2×2 of patches, in the order of `positions`.
    pub fn encode(&self, image: &Patches) -> Result<Vec<Vec<f32>>, String> {
        let c = &self.config;
        let (count, hidden) = (image.rows * image.cols, c.hidden);
        let patch_size = 3 * c.temporal * c.patch * c.patch;
        let mut x = linear(&image.values, patch_size, &self.tensor("patch_embed.proj.weight")?, &self.tensor("patch_embed.proj.bias")?);

        for (patch, row) in self.positions(image.rows, image.cols)?.iter().enumerate() {
            for (v, e) in x[patch * hidden..(patch + 1) * hidden].iter_mut().zip(row) {
                *v += e;
            }
        }
        let head_dim = hidden / c.heads;
        let angles = patch_angles(c, image.rows, image.cols);

        for block in 0..c.depth {
            let p = |name: &str| self.tensor(&format!("blocks.{block}.{name}"));
            let mut normed = x.clone();
            layer_norm(&mut normed, hidden, &p("norm1.weight")?, &p("norm1.bias")?);
            let mut qkv = linear(&normed, hidden, &p("attn.qkv.weight")?, &p("attn.qkv.bias")?);
            let half = head_dim / 2;
            qkv.par_chunks_mut(3 * hidden).zip(&angles).for_each(|(row, angle)| {
                // Queries and keys rotate; values do not.
                for part in 0..2 {
                    for head in row[part * hidden..(part + 1) * hidden].chunks_exact_mut(head_dim) {
                        let original = head.to_vec();
                        for i in 0..half {
                            let (cos, sin) = (angle[i].cos(), angle[i].sin());
                            head[i] = original[i] * cos - original[half + i] * sin;
                            head[half + i] = original[half + i] * cos + original[i] * sin;
                        }
                    }
                }
            });
            // Every patch attends to every other, in both directions.
            let scale = 1.0 / (head_dim as f32).sqrt();
            let mut attended = vec![0f32; count * hidden];
            attended.par_chunks_mut(hidden).enumerate().for_each(|(query, out)| {
                for head in 0..c.heads {
                    let q = &qkv[query * 3 * hidden + head * head_dim..][..head_dim];
                    let scores: Vec<f32> = (0..count).map(|key| qkv[key * 3 * hidden + hidden + head * head_dim..][..head_dim].iter().zip(q).map(|(a, b)| a * b).sum::<f32>() * scale).collect();
                    let top = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let weights: Vec<f32> = scores.iter().map(|s| (s - top).exp()).collect();
                    let total: f32 = weights.iter().sum();
                    let slot = &mut out[head * head_dim..(head + 1) * head_dim];
                    for (key, w) in weights.iter().enumerate() {
                        let v = &qkv[key * 3 * hidden + 2 * hidden + head * head_dim..][..head_dim];
                        for (o, value) in slot.iter_mut().zip(v) {
                            *o += w / total * value;
                        }
                    }
                }
            });
            let projected = linear(&attended, hidden, &p("attn.proj.weight")?, &p("attn.proj.bias")?);
            x.iter_mut().zip(&projected).for_each(|(v, a)| *v += a);

            let mut normed = x.clone();
            layer_norm(&mut normed, hidden, &p("norm2.weight")?, &p("norm2.bias")?);
            let mut up = linear(&normed, hidden, &p("mlp.linear_fc1.weight")?, &p("mlp.linear_fc1.bias")?);
            up.par_iter_mut().for_each(|v| *v = gelu_tanh(*v));
            let down = linear(&up, c.intermediate, &p("mlp.linear_fc2.weight")?, &p("mlp.linear_fc2.bias")?);
            x.iter_mut().zip(&down).for_each(|(v, d)| *v += d);
        }

        // Each 2×2 of patches, already next to each other, becomes one token.
        layer_norm(&mut x, hidden, &self.tensor("merger.norm.weight")?, &self.tensor("merger.norm.bias")?);
        let merged = hidden * c.merge * c.merge;
        let mut up = linear(&x, merged, &self.tensor("merger.linear_fc1.weight")?, &self.tensor("merger.linear_fc1.bias")?);
        up.par_iter_mut().for_each(|v| *v = gelu(*v));
        let out = linear(&up, merged, &self.tensor("merger.linear_fc2.weight")?, &self.tensor("merger.linear_fc2.bias")?);
        Ok(out.chunks(c.out_hidden).map(<[f32]>::to_vec).collect())
    }
}

/// Each patch's rotation angles, in the encoder's order: the first half of each head's
/// frequencies follow its row, the second its column.
pub fn patch_angles(c: &VisionConfig, rows: usize, cols: usize) -> Vec<Vec<f32>> {
    let head_dim = c.hidden / c.heads;
    let quarter = head_dim / 4;
    let frequency: Vec<f32> = (0..quarter).map(|i| 1.0 / c.rope_theta.powf((2 * i) as f32 / (head_dim / 2) as f32)).collect();
    merge_order(rows, cols, c.merge).iter().map(|(row, col)| frequency.iter().map(|f| *row as f32 * f).chain(frequency.iter().map(|f| *col as f32 * f)).collect()).collect()
}

/// Row and column of each patch, in the order the encoder takes them: block by block of
/// `merge × merge`.
pub fn merge_order(rows: usize, cols: usize, merge: usize) -> Vec<(usize, usize)> {
    let mut order = Vec::with_capacity(rows * cols);
    for block_row in 0..rows / merge {
        for block_col in 0..cols / merge {
            for in_row in 0..merge {
                for in_col in 0..merge {
                    order.push((block_row * merge + in_row, block_col * merge + in_col));
                }
            }
        }
    }
    order
}

/// The time, row and column positions of an image's tokens when the image starts at
/// position `start`, and the position the text after it continues from.
pub fn image_positions(start: u64, rows: usize, cols: usize, merge: usize) -> (Vec<[u64; 3]>, u64) {
    let (h, w) = (rows / merge, cols / merge);
    let positions = (0..h).flat_map(|r| (0..w).map(move |c| [start, start + r as u64, start + c as u64])).collect();
    (positions, start + h.max(w) as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_error_function_matches_known_values() {
        for (x, expected) in [(0.0f32, 0.0f32), (0.5, 0.520_499_9), (1.0, 0.842_700_8), (2.5, 0.999_593), (-1.0, -0.842_700_8)] {
            assert!((erf(x) - expected).abs() < 1e-6, "{x}: {}", erf(x));
        }
    }

    #[test]
    fn positions_follow_the_grid_and_the_text_goes_on_after_the_longest_side() {
        let (positions, next) = image_positions(5, 4, 8, 2);
        assert_eq!(positions.len(), 8);
        assert_eq!(positions[0], [5, 5, 5]);
        assert_eq!(positions[5], [5, 6, 6]);
        assert_eq!(next, 9);
        assert_eq!(merge_order(4, 4, 2)[..5], [(0, 0), (0, 1), (1, 0), (1, 1), (0, 2)]);
    }
}
