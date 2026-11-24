//! The Descript Audio Codec's decoder in floating point: codes to sound, what the chip has
//! to reproduce to speak. It reads the folder `tools/convert/dac.py` writes.
//!
//! Each frame's codes, one per codebook, pick a row of 8 values of their codebook; a
//! 1 × 1 convolution takes each to the latent's width, and they add up. The decoder is a
//! convolution, then blocks that each lengthen the sound by their rate with a transposed
//! convolution and refine it with three residual units (dilations 1, 3 and 9), then a
//! convolution down to one channel and tanh. Its activation is Snake, `x + sin²(αx) / α`
//! with an α per channel. Sound is kept a row per time step, all its channels in the row.

use crate::Weights;
use rayon::prelude::*;
use serde_json::Value;
use std::collections::HashMap;
use std::path::Path;

/// What a converted DAC checkpoint looks like.
#[derive(Clone, Debug)]
pub struct DacConfig {
    pub sampling_rate: u32,
    pub latent_dim: usize,
    pub decoder_dim: usize,
    pub rates: Vec<usize>,
    pub codebooks: usize,
    pub codebook_size: usize,
    pub codebook_dim: usize,
}

impl DacConfig {
    pub fn read(dir: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(dir.join("config.json")).map_err(|e| e.to_string())?;
        let c: Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
        if c["model_type"].as_str() != Some("dac") {
            return Err("not a converted DAC checkpoint (tools/convert/dac.py)".into());
        }
        let number = |key: &str| c[key].as_u64().map(|v| v as usize).ok_or(format!("“{key}” is missing from the config"));
        Ok(Self {
            sampling_rate: number("sampling_rate")? as u32,
            latent_dim: number("latent_dim")?,
            decoder_dim: number("decoder_dim")?,
            rates: c["decoder_rates"].as_array().ok_or("decoder_rates is missing")?.iter().filter_map(Value::as_u64).map(|v| v as usize).collect(),
            codebooks: number("n_codebooks")?,
            codebook_size: number("codebook_size")?,
            codebook_dim: number("codebook_dim")?,
        })
    }

    /// How many frames on each side one frame's sound depends on: the first convolution's
    /// three, then at each block a row of its input (the transposed convolution) and 39 rows
    /// of its output (three residual units of kernel 7, dilations 1, 3 and 9), and the last
    /// convolution's three samples, each in frames at its level's rate, rounded up.
    pub fn reach_frames(&self) -> usize {
        let mut rows_per_frame = 1.0;
        let mut reach = 3.0;
        for s in &self.rates {
            reach += 1.0 / rows_per_frame;
            rows_per_frame *= *s as f64;
            reach += 39.0 / rows_per_frame;
        }
        (reach + 3.0 / rows_per_frame).ceil() as usize
    }

    /// Samples a frame makes in the middle of the sound: the product of the rates.
    pub fn hop(&self) -> usize {
        self.rates.iter().product()
    }

    /// The windows a sound of `frames` frames is decoded in, `chunk` frames kept from each:
    /// each window is `chunk + 2 · reach_frames()` long (or the whole sound, if shorter), and
    /// keeps the frames it owns, at least `reach_frames()` from either of its ends unless
    /// that end is the sound's. Gives `(first frame of the window, frames it owns)`.
    pub fn windows(&self, frames: usize, chunk: usize) -> Vec<(usize, std::ops::Range<usize>)> {
        let (reach, chunk) = (self.reach_frames(), chunk.max(1));
        let width = (chunk + 2 * reach).min(frames);
        (0..frames.div_ceil(chunk)).map(|i| {
            let owned = i * chunk..((i + 1) * chunk).min(frames);
            (owned.start.saturating_sub(reach).min(frames - width), owned)
        }).collect()
    }

    /// Frames in each window of `windows`.
    pub fn window_frames(&self, frames: usize, chunk: usize) -> usize {
        (chunk.max(1) + 2 * self.reach_frames()).min(frames)
    }

    /// Samples for `frames` frames: each transposed convolution of rate `s` takes `L` rows
    /// to `s·L + s − 2·⌈s/2⌉`.
    pub fn samples(&self, frames: usize) -> usize {
        self.rates.iter().fold(frames, |length, s| s * length + s - 2 * s.div_ceil(2))
    }
}

/// A converted DAC checkpoint, its weights read as floats.
pub struct Dac {
    pub config: DacConfig,
    tensors: HashMap<String, Vec<f32>>,
}

/// Snake: `x + sin²(αx) / α`, with the codec's guard against a zero α.
pub fn snake(x: f32, alpha: f32) -> f32 {
    let s = (alpha * x).sin();
    x + s * s / (alpha + 1e-9)
}

type Rows = Vec<Vec<f32>>;

impl Dac {
    pub fn open(dir: &Path) -> Result<Self, String> {
        let weights = Weights::open(dir)?;
        let mut tensors = HashMap::new();
        for (name, _, _) in weights.raw_tensors() {
            tensors.insert(name.to_string(), weights.floats(name)?);
        }
        Ok(Self { config: DacConfig::read(dir)?, tensors })
    }

    pub fn tensor(&self, name: &str) -> Result<&[f32], String> {
        self.tensors.get(name).map(Vec::as_slice).ok_or_else(|| format!("{name} is missing"))
    }

    /// The latent of each frame: `codes[book][frame]`.
    pub fn latent(&self, codes: &[Vec<u32>]) -> Result<Rows, String> {
        let c = &self.config;
        if codes.len() != c.codebooks || codes.iter().any(|b| b.len() != codes[0].len()) {
            return Err(format!("{} codebooks of codes of equal length are needed", c.codebooks));
        }
        let mut rows = vec![vec![0f32; c.latent_dim]; codes[0].len()];
        for (book, list) in codes.iter().enumerate() {
            let p = |n: &str| format!("quantizer.quantizers.{book}.{n}");
            let (table, weight, bias) = (self.tensor(&p("codebook.weight"))?, self.tensor(&p("out_proj.weight"))?, self.tensor(&p("out_proj.bias"))?);
            for (row, code) in rows.iter_mut().zip(list) {
                let code = *code as usize;
                if code >= c.codebook_size {
                    return Err(format!("code {code} is outside the codebook"));
                }
                let vector = &table[code * c.codebook_dim..(code + 1) * c.codebook_dim];
                for (o, out) in row.iter_mut().enumerate() {
                    *out += bias[o] + weight[o * c.codebook_dim..(o + 1) * c.codebook_dim].iter().zip(vector).map(|(a, b)| a * b).sum::<f32>();
                }
            }
        }
        Ok(rows)
    }

    /// A convolution with `padding` zeros each side and `dilation` between taps, its
    /// weights laid `[out, in, kernel]`.
    fn conv(&self, name: &str, x: &Rows, kernel: usize, dilation: usize, padding: usize) -> Result<Rows, String> {
        let (weight, bias) = (self.tensor(&format!("{name}.weight"))?, self.tensor(&format!("{name}.bias"))?);
        let inputs = x[0].len();
        let length = x.len() + 2 * padding - dilation * (kernel - 1);
        Ok((0..length)
            .into_par_iter()
            .map(|t| {
                let mut out = bias.to_vec();
                for k in 0..kernel {
                    let Some(source) = (t + k * dilation).checked_sub(padding).filter(|i| *i < x.len()) else { continue };
                    let row = &x[source];
                    for (o, value) in out.iter_mut().enumerate() {
                        let w = &weight[o * inputs * kernel..(o + 1) * inputs * kernel];
                        *value += row.iter().enumerate().map(|(c, v)| w[c * kernel + k] * v).sum::<f32>();
                    }
                }
                out
            })
            .collect())
    }

    /// A transposed convolution of rate `s`, kernel `2s` and padding `⌈s/2⌉`, its weights
    /// laid `[in, out, kernel]`.
    fn transposed(&self, name: &str, x: &Rows, s: usize) -> Result<Rows, String> {
        let (weight, bias) = (self.tensor(&format!("{name}.weight"))?, self.tensor(&format!("{name}.bias"))?);
        let (outputs, kernel, padding) = (bias.len(), 2 * s, s.div_ceil(2));
        let length = s * x.len() + s - 2 * padding;
        Ok((0..length)
            .into_par_iter()
            .map(|n| {
                let mut out = bias.to_vec();
                // The inputs that reach output n: i·s + k = n + padding, k < 2s.
                let reach = n + padding;
                let first = reach.saturating_sub(kernel - 1).div_ceil(s);
                for (i, row) in x.iter().enumerate().take(reach / s + 1).skip(first) {
                    let k = reach - i * s;
                    for (c, v) in row.iter().enumerate() {
                        let w = &weight[(c * outputs) * kernel..(c + 1) * outputs * kernel];
                        for (o, value) in out.iter_mut().enumerate() {
                            *value += v * w[o * kernel + k];
                        }
                    }
                }
                out
            })
            .collect())
    }

    fn snake_rows(&self, name: &str, x: &Rows) -> Result<Rows, String> {
        let alpha = self.tensor(&format!("{name}.alpha"))?;
        Ok(x.par_iter().map(|row| row.iter().zip(alpha).map(|(v, a)| snake(*v, *a)).collect()).collect())
    }

    /// Every Snake's largest |αx| for these codes: how far the sine's argument goes.
    pub fn snake_reach(&self, codes: &[Vec<u32>]) -> Result<f32, String> {
        let mut reach = 0f32;
        self.decode_watching(codes, &mut |name, x| {
            let alpha = self.tensor(&format!("{name}.alpha")).unwrap();
            for row in x {
                for (v, a) in row.iter().zip(alpha) {
                    reach = reach.max((v * a).abs());
                }
            }
        })?;
        Ok(reach)
    }

    /// The sound of `codes[book][frame]`, in [−1, 1] at the codec's rate.
    pub fn decode(&self, codes: &[Vec<u32>]) -> Result<Vec<f32>, String> {
        self.decode_watching(codes, &mut |_, _| {})
    }

    /// The same sound decoded a window at a time (`DacConfig::windows`), as a chip that holds
    /// only a window does: each keeps the samples of the frames it owns.
    pub fn decode_chunked(&self, codes: &[Vec<u32>], chunk: usize) -> Result<Vec<f32>, String> {
        let frames = codes.first().map_or(0, Vec::len);
        stitch(&self.config, frames, chunk, &mut |start, width| self.decode(&window(codes, start, width)))
    }

    fn decode_watching(&self, codes: &[Vec<u32>], watch: &mut dyn FnMut(&str, &Rows)) -> Result<Vec<f32>, String> {
        let mut x = self.conv("decoder.model.0", &self.latent(codes)?, 7, 1, 3)?;
        for (b, s) in self.config.rates.iter().enumerate() {
            let p = |n: &str| format!("decoder.model.{}.block.{n}", b + 1);
            watch(&p("0"), &x);
            x = self.transposed(&p("1"), &self.snake_rows(&p("0"), &x)?, *s)?;
            for (u, dilation) in [1, 3, 9].into_iter().enumerate() {
                let q = |n: &str| p(&format!("{}.block.{n}", u + 2));
                watch(&q("0"), &x);
                let y = self.conv(&q("1"), &self.snake_rows(&q("0"), &x)?, 7, dilation, 3 * dilation)?;
                watch(&q("2"), &y);
                let y = self.conv(&q("3"), &self.snake_rows(&q("2"), &y)?, 1, 1, 0)?;
                x.iter_mut().zip(&y).for_each(|(a, b)| a.iter_mut().zip(b).for_each(|(a, b)| *a += b));
            }
        }
        let last = self.config.rates.len() + 1;
        watch(&format!("decoder.model.{last}"), &x);
        let x = self.snake_rows(&format!("decoder.model.{last}"), &x)?;
        let x = self.conv(&format!("decoder.model.{}", last + 1), &x, 7, 1, 3)?;
        Ok(x.iter().map(|row| row[0].tanh()).collect())
    }
}

/// `codes` from frame `start`, `width` frames.
pub fn window(codes: &[Vec<u32>], start: usize, width: usize) -> Vec<Vec<u32>> {
    codes.iter().map(|book| book[start..start + width].to_vec()).collect()
}

/// A sound of `frames` frames put together from its windows, each decoded by `decode` from
/// its first frame and its width: from each, the samples of the frames it owns; the last
/// keeps the sound's end too.
pub fn stitch(config: &DacConfig, frames: usize, chunk: usize, decode: &mut dyn FnMut(usize, usize) -> Result<Vec<f32>, String>) -> Result<Vec<f32>, String> {
    let (hop, width, total) = (config.hop(), config.window_frames(frames, chunk), config.samples(frames));
    let mut sound = Vec::with_capacity(total);
    for (start, owned) in config.windows(frames, chunk) {
        let samples = decode(start, width)?;
        let from = (owned.start - start) * hop;
        let to = if owned.end == frames { samples.len() } else { (owned.end - start) * hop };
        sound.extend_from_slice(&samples[from..to]);
    }
    debug_assert_eq!(sound.len(), total);
    Ok(sound)
}
