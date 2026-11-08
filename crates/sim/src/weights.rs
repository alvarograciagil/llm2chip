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

const PREFIX: &str = "language_model.";

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
                let short = name.strip_prefix(PREFIX).unwrap_or(name).to_string();
                tensors.insert(short, TensorRef { file: index, start, end, dtype: info["dtype"].as_str().unwrap_or("").to_string(), shape });
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
            other => Err(format!("{name}: tipo {other} no soportado")),
        }
    }

    pub fn matrix(&self, name: &str) -> Result<Matrix<'_>, String> {
        let weight = format!("{name}.weight");
        let (w, t) = self.bytes(&weight)?;
        let (scales, ts) = self.bytes(&format!("{name}.scales"))?;
        let (biases, _) = self.bytes(&format!("{name}.biases"))?;
        if t.dtype != "U32" {
            return Err(format!("{name}: se esperaba cuantización empaquetada"));
        }
        let (rows_shape, last) = t.shape.split_at(t.shape.len() - 1);
        Ok(Matrix { rows: rows_shape.iter().product(), words_per_row: last[0], groups: *ts.shape.last().unwrap_or(&1), words: w, scales, biases })
    }
}

/// One affine-quantised matrix: every weight is `q · scale + bias`, per group of columns.
pub struct Matrix<'a> {
    rows: usize,
    words_per_row: usize,
    groups: usize,
    words: &'a [u8],
    scales: &'a [u8],
    biases: &'a [u8],
}

#[inline]
fn half_at(bytes: &[u8], index: usize) -> f32 {
    f16::from_le_bytes([bytes[2 * index], bytes[2 * index + 1]]).to_f32()
}

impl Matrix<'_> {
    pub fn row(&self, index: usize, out: usize) -> Result<Vec<f32>, String> {
        let _ = out;
        if index >= self.rows {
            return Err("fila fuera de rango".into());
        }
        let cols = out;
        let group_size = cols / self.groups;
        let bits = self.words_per_row * 32 / cols;
        let row = &self.words[index * self.words_per_row * 4..(index + 1) * self.words_per_row * 4];
        let mut y = vec![0f32; cols];
        for g in 0..self.groups {
            let scale = half_at(self.scales, index * self.groups + g);
            let bias = half_at(self.biases, index * self.groups + g);
            for j in 0..group_size {
                let at = g * group_size + j;
                let q = if bits == 8 { row[at] } else { (row[at / 2] >> (4 * (at & 1))) & 0xF };
                y[at] = q as f32 * scale + bias;
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
        let group_size = cols / self.groups;
        let bits = self.words_per_row * 32 / cols;
        let sums: Vec<f32> = x.chunks_exact(group_size).map(|c| c.iter().sum()).collect();
        let stride = self.words_per_row * 4;
        y.par_iter_mut().with_min_len(32).enumerate().for_each(|(index, out)| {
            let r = first + index;
            let row = &self.words[r * stride..(r + 1) * stride];
            let mut total = 0f32;
            for (g, xs) in x.chunks_exact(group_size).enumerate() {
                let dot = if bits == 4 {
                    dot_nibbles(&row[g * group_size / 2..(g + 1) * group_size / 2], xs)
                } else {
                    dot_bytes(&row[g * group_size..(g + 1) * group_size], xs)
                };
                total += dot * half_at(self.scales, r * self.groups + g) + half_at(self.biases, r * self.groups + g) * sums[g];
            }
            *out = total;
        });
    }
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
