//! Reads MLX-quantised safetensors checkpoints without loading them into memory.

use half::f16;
use memmap2::Mmap;
use rayon::prelude::*;
use serde_json::Value;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

struct TensorRef {
    file: usize,
    start: usize,
    end: usize,
    dtype: String,
    shape: Vec<usize>,
}

pub struct Weights {
    files: Vec<Mmap>,
    tensors: HashMap<String, TensorRef>,
}

#[derive(Clone, Copy, Debug)]
pub struct SamplerConfig {
    pub temperature: f32,
    pub top_k: usize,
}

/// Maps both MLX (`language_model.model.…`) and Hugging Face (`model.language_model.…`)
/// tensor names onto the program's `model.layers.…` names.
fn canonical_name(name: &str) -> String {
    if let Some(rest) = name.strip_prefix("language_model.") {
        return rest.to_string();
    }
    if let Some(rest) = name.strip_prefix("model.language_model.") {
        return format!("model.{rest}");
    }
    name.to_string()
}

impl Weights {
    pub fn open(dir: &Path) -> Result<Self, String> {
        let mut paths: Vec<_> = std::fs::read_dir(dir)
            .map_err(|e| e.to_string())?
            .filter_map(Result::ok)
            .map(|e| e.path())
            .filter(|p| {
                let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
                name.ends_with(".safetensors") && !name.starts_with("model-vision")
            })
            .collect();
        paths.sort();
        if paths.is_empty() {
            return Err("la carpeta no contiene pesos safetensors".into());
        }

        let mut files = Vec::new();
        let mut tensors = HashMap::new();
        for (index, path) in paths.iter().enumerate() {
            let file = File::open(path).map_err(|e| e.to_string())?;
            // SAFETY: checkpoints are treated as read-only for the life of the engine.
            let map = unsafe { Mmap::map(&file) }.map_err(|e| e.to_string())?;
            if map.len() < 8 {
                return Err("archivo safetensors truncado".into());
            }
            let size = u64::from_le_bytes(map[..8].try_into().unwrap()) as usize;
            let header: Value = serde_json::from_slice(&map[8..8 + size]).map_err(|e| e.to_string())?;
            for (name, info) in header.as_object().ok_or("cabecera inválida")? {
                if name == "__metadata__" {
                    continue;
                }
                let offsets = info["data_offsets"].as_array().ok_or("tensor sin offsets")?;
                let start = 8 + size + offsets[0].as_u64().unwrap_or(0) as usize;
                let end = 8 + size + offsets[1].as_u64().unwrap_or(0) as usize;
                let shape = info["shape"].as_array().ok_or("tensor sin forma")?.iter().map(|v| v.as_u64().unwrap_or(0) as usize).collect();
                tensors.insert(canonical_name(name), TensorRef { file: index, start, end, dtype: info["dtype"].as_str().unwrap_or("").to_string(), shape });
            }
            files.push(map);
        }
        Ok(Self { files, tensors })
    }

    fn bytes(&self, name: &str) -> Result<(&[u8], &TensorRef), String> {
        let t = self.tensors.get(name).ok_or_else(|| format!("falta el tensor {name}"))?;
        Ok((&self.files[t.file][t.start..t.end], t))
    }

    /// A small float tensor such as a norm weight, converted to `f32`.
    pub fn floats(&self, name: &str) -> Result<Vec<f32>, String> {
        let (bytes, t) = self.bytes(name)?;
        match t.dtype.as_str() {
            "F16" => Ok(bytes.as_chunks::<2>().0.iter().map(|c| f16::from_le_bytes(*c).to_f32()).collect()),
            "BF16" => Ok(bytes.as_chunks::<2>().0.iter().map(|c| half::bf16::from_le_bytes(*c).to_f32()).collect()),
            "F32" => Ok(bytes.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect()),
            "F64" => Ok(bytes.as_chunks::<8>().0.iter().map(|c| f64::from_le_bytes(*c) as f32).collect()),
            other => Err(format!("{name}: tipo {other} no soportado")),
        }
    }

    pub fn matrix(&self, name: &str) -> Result<Matrix<'_>, String> {
        let weight = format!("{name}.weight");
        let (bytes, t) = self.bytes(&weight)?;
        let (rows_shape, last) = t.shape.split_at(t.shape.len() - 1);
        let rows: usize = rows_shape.iter().product();
        if t.dtype == "U32" {
            let (scales, ts) = self.bytes(&format!("{name}.scales"))?;
            let (biases, _) = self.bytes(&format!("{name}.biases"))?;
            return Ok(Matrix { rows, source: Source::Packed { words_per_row: last[0], groups: *ts.shape.last().unwrap_or(&1), words: bytes, scales, biases } });
        }
        let format = Plain::parse(&t.dtype).ok_or_else(|| format!("{name}: tipo {} no soportado", t.dtype))?;
        // Integer checkpoints may carry the factor that turns codes into real values.
        let factor = match self.floats(&format!("{name}.scale")) {
            Ok(values) if !values.is_empty() => values[0],
            _ => format.default_factor(),
        };
        Ok(Matrix { rows, source: Source::Plain { format, bytes, factor } })
    }
}

/// How the elements of an unpacked matrix are stored.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Plain {
    F32,
    F16,
    Bf16,
    I8,
    U8,
    I16,
    U16,
}

impl Plain {
    fn parse(dtype: &str) -> Option<Self> {
        Some(match dtype {
            "F32" => Plain::F32,
            "F16" => Plain::F16,
            "BF16" => Plain::Bf16,
            "I8" => Plain::I8,
            "U8" => Plain::U8,
            "I16" => Plain::I16,
            "U16" => Plain::U16,
            _ => return None,
        })
    }

    fn size(self) -> usize {
        match self {
            Plain::F32 => 4,
            Plain::I8 | Plain::U8 => 1,
            _ => 2,
        }
    }

    /// Without an explicit scale, integer codes span the unit interval.
    fn default_factor(self) -> f32 {
        match self {
            Plain::I8 => 1.0 / 128.0,
            Plain::U8 => 1.0 / 255.0,
            Plain::I16 => 1.0 / 32768.0,
            Plain::U16 => 1.0 / 65535.0,
            _ => 1.0,
        }
    }

    #[inline]
    fn decode(self, bytes: &[u8], factor: f32) -> f32 {
        match self {
            Plain::F32 => f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
            Plain::F16 => f16::from_le_bytes([bytes[0], bytes[1]]).to_f32(),
            Plain::Bf16 => half::bf16::from_le_bytes([bytes[0], bytes[1]]).to_f32(),
            Plain::I8 => bytes[0] as i8 as f32 * factor,
            Plain::U8 => bytes[0] as f32 * factor,
            Plain::I16 => i16::from_le_bytes([bytes[0], bytes[1]]) as f32 * factor,
            Plain::U16 => u16::from_le_bytes([bytes[0], bytes[1]]) as f32 * factor,
        }
    }
}

enum Source<'a> {
    /// MLX affine quantisation: every weight is `q · scale + bias`, per group of columns.
    Packed { words_per_row: usize, groups: usize, words: &'a [u8], scales: &'a [u8], biases: &'a [u8] },
    Plain { format: Plain, bytes: &'a [u8], factor: f32 },
}

pub struct Matrix<'a> {
    rows: usize,
    source: Source<'a>,
}

#[inline]
fn half_at(bytes: &[u8], index: usize) -> f32 {
    f16::from_le_bytes([bytes[2 * index], bytes[2 * index + 1]]).to_f32()
}

const MANTISSA_FRACTION: i32 = 15;
const AFFINE_EXTRA: i32 = 4;
const MANTISSA_LIMIT: i64 = 32767;

fn exponent_of(largest: f32) -> i64 {
    if largest > 0.0 {
        largest.log2().floor() as i64 + 1
    } else {
        0
    }
}

impl Matrix<'_> {
    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn row(&self, index: usize, out: usize) -> Result<Vec<f32>, String> {
        if index >= self.rows {
            return Err("fila fuera de rango".into());
        }
        let cols = out;
        let mut y = vec![0f32; cols];
        match &self.source {
            Source::Packed { words_per_row, groups, words, scales, biases } => {
                let group_size = cols / groups;
                let bits = words_per_row * 32 / cols;
                let row = &words[index * words_per_row * 4..(index + 1) * words_per_row * 4];
                for g in 0..*groups {
                    let scale = half_at(scales, index * groups + g);
                    let bias = half_at(biases, index * groups + g);
                    for j in 0..group_size {
                        let at = g * group_size + j;
                        let q = if bits == 8 { row[at] } else { (row[at / 2] >> (4 * (at & 1))) & 0xF };
                        y[at] = q as f32 * scale + bias;
                    }
                }
            }
            Source::Plain { format, bytes, factor, .. } => {
                let size = format.size();
                let row = &bytes[index * cols * size..(index + 1) * cols * size];
                for (out, element) in y.iter_mut().zip(row.chunks_exact(size)) {
                    *out = format.decode(element, *factor);
                }
            }
        }
        Ok(y)
    }

    /// `y = M · x`, in parallel across rows. `x.len()` is the column count.
    pub fn apply(&self, x: &[f32], y: &mut [f32]) {
        self.apply_rows(0, x, y);
    }

    /// Like `apply`, over the rows starting at `first`; `y.len()` sets how many.
    pub fn apply_rows(&self, first: usize, x: &[f32], y: &mut [f32]) {
        let cols = x.len();
        match &self.source {
            Source::Packed { words_per_row, groups, words, scales, biases } => {
                let group_size = cols / groups;
                let bits = words_per_row * 32 / cols;
                let sums: Vec<f32> = x.chunks_exact(group_size).map(|c| c.iter().sum()).collect();
                let stride = words_per_row * 4;
                y.par_iter_mut().with_min_len(32).enumerate().for_each(|(index, out)| {
                    let r = first + index;
                    let row = &words[r * stride..(r + 1) * stride];
                    let mut total = 0f32;
                    for (g, xs) in x.chunks_exact(group_size).enumerate() {
                        let dot = if bits == 4 {
                            dot_nibbles(&row[g * group_size / 2..(g + 1) * group_size / 2], xs)
                        } else {
                            dot_bytes(&row[g * group_size..(g + 1) * group_size], xs)
                        };
                        total += dot * half_at(scales, r * groups + g) + half_at(biases, r * groups + g) * sums[g];
                    }
                    *out = total;
                });
            }
            Source::Plain { format, bytes, factor, .. } => {
                let size = format.size();
                y.par_iter_mut().with_min_len(32).enumerate().for_each(|(index, out)| {
                    let r = first + index;
                    let row = &bytes[r * cols * size..(r + 1) * cols * size];
                    *out = row.chunks_exact(size).zip(x).map(|(e, v)| format.decode(e, *factor) * v).sum();
                });
            }
        }
    }

    /// The exponent shared by the matrix's 16-bit mantissas.
    pub fn exponent(&self, cols: usize) -> i64 {
        match &self.source {
            Source::Packed { words_per_row, groups, scales, biases, .. } => {
                let top = ((1u32 << (words_per_row * 32 / cols)) - 1) as f32;
                let largest = (0..self.rows * groups)
                    .into_par_iter()
                    .with_min_len(4096)
                    .map(|i| {
                        let (scale, bias) = (half_at(scales, i), half_at(biases, i));
                        bias.abs().max((top * scale + bias).abs())
                    })
                    .reduce(|| 0.0, f32::max);
                exponent_of(largest)
            }
            Source::Plain { format, bytes, factor, .. } => {
                let largest = bytes.par_chunks_exact(format.size()).with_min_len(4096).map(|e| format.decode(e, *factor).abs()).reduce(|| 0.0, f32::max);
                exponent_of(largest)
            }
        }
    }

    /// One row as the 16-bit mantissas the hardware stores.
    pub fn fixed_row(&self, index: usize, cols: usize, exponent: i64) -> Result<Vec<i64>, String> {
        match &self.source {
            Source::Packed { words_per_row, groups, words, scales, biases } => {
                let group_size = cols / groups;
                let bits = words_per_row * 32 / cols;
                let row = &words[index * words_per_row * 4..(index + 1) * words_per_row * 4];
                let factor = fixed_factor(exponent);
                let mut out = Vec::with_capacity(cols);
                for g in 0..*groups {
                    let (scale, bias) = affine_integers(half_at(scales, index * groups + g), half_at(biases, index * groups + g), factor);
                    for j in 0..group_size {
                        let at = g * group_size + j;
                        let q = if bits == 8 { row[at] } else { (row[at / 2] >> (4 * (at & 1))) & 0xF };
                        out.push(affine_weight(q, scale, bias));
                    }
                }
                Ok(out)
            }
            Source::Plain { .. } => {
                let scale = 2f64.powi(MANTISSA_FRACTION - exponent as i32);
                Ok(self.row(index, cols)?.iter().map(|v| ((*v as f64 * scale).round() as i64).clamp(-MANTISSA_LIMIT, MANTISSA_LIMIT)).collect())
            }
        }
    }

    /// Full-precision `Σ w · x` for each row, using the mantissas of `fixed_row`.
    pub fn exact_sums(&self, first: usize, rows: usize, x: &[i32], exponent: i64) -> Vec<i64> {
        let cols = x.len();
        let mut sums = vec![0i64; rows];
        match &self.source {
            Source::Packed { words_per_row, groups, words, scales, biases } => {
                let group_size = cols / groups;
                let bits = words_per_row * 32 / cols;
                let stride = words_per_row * 4;
                let factor = fixed_factor(exponent);
                sums.par_iter_mut().with_min_len(16).enumerate().for_each(|(index, out)| {
                    let r = first + index;
                    let row = &words[r * stride..(r + 1) * stride];
                    let mut code = vec![0u8; group_size];
                    let mut total = 0i64;
                    for (g, xs) in x.chunks_exact(group_size).enumerate() {
                        if bits == 4 {
                            for (j, c) in code.iter_mut().enumerate() {
                                let at = g * group_size + j;
                                *c = (row[at / 2] >> (4 * (at & 1))) & 0xF;
                            }
                        } else {
                            code.copy_from_slice(&row[g * group_size..(g + 1) * group_size]);
                        }
                        let (scale, bias) = affine_integers(half_at(scales, r * groups + g), half_at(biases, r * groups + g), factor);
                        total += dot_affine(&code, xs, scale, bias);
                    }
                    *out = total;
                });
            }
            Source::Plain { format, bytes, factor, .. } => {
                let scale = 2f64.powi(MANTISSA_FRACTION - exponent as i32);
                let size = format.size();
                sums.par_iter_mut().with_min_len(16).enumerate().for_each(|(index, out)| {
                    let r = first + index;
                    let row = &bytes[r * cols * size..(r + 1) * cols * size];
                    *out = row
                        .chunks_exact(size)
                        .zip(x)
                        .map(|(e, v)| ((format.decode(e, *factor) as f64 * scale).round() as i64).clamp(-MANTISSA_LIMIT, MANTISSA_LIMIT) * *v as i64)
                        .sum();
                });
            }
        }
        sums
    }
}

fn fixed_factor(exponent: i64) -> f64 {
    2f64.powi(MANTISSA_FRACTION + AFFINE_EXTRA - exponent as i32)
}

#[inline]
fn affine_integers(scale: f32, bias: f32, factor: f64) -> (i32, i32) {
    ((scale as f64 * factor).round() as i32, (bias as f64 * factor).round() as i32)
}

#[inline]
fn affine_weight(q: u8, scale: i32, bias: i32) -> i64 {
    (((q as i64 * scale as i64 + bias as i64) >> AFFINE_EXTRA)).clamp(-MANTISSA_LIMIT, MANTISSA_LIMIT)
}

/// `Σ ((q·scale + bias) >> 4) · x`, written so the compiler can vectorise it.
#[inline]
fn dot_affine(codes: &[u8], x: &[i32], scale: i32, bias: i32) -> i64 {
    let limit = MANTISSA_LIMIT as i32;
    let mut total = 0i64;
    for (cs, xs) in codes.chunks(8).zip(x.chunks(8)) {
        let mut sum = [0i64; 8];
        for i in 0..cs.len() {
            let weight = ((cs[i] as i32 * scale + bias) >> AFFINE_EXTRA).clamp(-limit, limit);
            sum[i] = weight as i64 * xs[i] as i64;
        }
        total += sum.iter().sum::<i64>();
    }
    total
}

#[inline]
fn dot_bytes(q: &[u8], x: &[f32]) -> f32 {
    let mut sum = [0f32; 8];
    for (qs, xs) in q.as_chunks::<8>().0.iter().zip(x.as_chunks::<8>().0) {
        for i in 0..8 {
            sum[i] += qs[i] as f32 * xs[i];
        }
    }
    let tail = q.len() - q.len() % 8;
    sum.iter().sum::<f32>() + q[tail..].iter().zip(&x[tail..]).map(|(a, b)| *a as f32 * b).sum::<f32>()
}

#[inline]
fn dot_nibbles(packed: &[u8], x: &[f32]) -> f32 {
    let mut low = [0f32; 8];
    let mut high = [0f32; 8];
    for (bytes, xs) in packed.as_chunks::<8>().0.iter().zip(x.as_chunks::<16>().0) {
        for i in 0..8 {
            low[i] += (bytes[i] & 0xF) as f32 * xs[2 * i];
            high[i] += (bytes[i] >> 4) as f32 * xs[2 * i + 1];
        }
    }
    // Groups smaller than the vector width only occur in small test models.
    let tail = packed.len() - packed.len() % 8;
    let rest: f32 = packed[tail..]
        .iter()
        .enumerate()
        .map(|(i, byte)| (byte & 0xF) as f32 * x[2 * (tail + i)] + (byte >> 4) as f32 * x[2 * (tail + i) + 1])
        .sum();
    low.iter().sum::<f32>() + high.iter().sum::<f32>() + rest
}
