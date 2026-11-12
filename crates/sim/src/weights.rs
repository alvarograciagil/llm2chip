//! Reads MLX-quantised safetensors checkpoints without loading them into memory.

use crate::accel::{self, Accelerator, ExactScale, PackedTensor};
use crate::requant::{quantize, recode, train_table, Recoded, Reference};
use half::f16;
use memmap2::Mmap;
use netlist::format::{WeightCoding, WeightFormat};
use rayon::prelude::*;
use serde_json::Value;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

struct TensorRef {
    file: usize,
    start: usize,
    end: usize,
    dtype: String,
    shape: Vec<usize>,
}

pub struct Weights {
    // The GPU's buffers point into `files`, so it must be dropped first.
    accelerator: Option<Box<dyn Accelerator>>,
    files: Vec<Mmap>,
    tensors: HashMap<String, TensorRef>,
    /// The chip's format, when weights must be rewritten to match it.
    format: Option<WeightFormat>,
    references: Mutex<HashMap<String, Arc<Reference>>>,
    recoded: Mutex<HashMap<String, Option<Arc<Recoded>>>>,
    /// Bytes of memory that rewritten tensors may still take.
    recode_budget: AtomicUsize,
    gpu_products: AtomicUsize,
    largest: Mutex<HashMap<String, f32>>,
}

/// Smaller products finish sooner on the CPU than a GPU round trip takes.
const GPU_MIN_WEIGHTS: usize = 1 << 20;

/// Rewritten weights may take half of the machine's memory.
fn default_recode_budget() -> usize {
    let system = sysinfo::System::new_with_specifics(sysinfo::RefreshKind::nothing().with_memory(sysinfo::MemoryRefreshKind::nothing().with_ram()));
    (system.total_memory() / 2) as usize
}

/// Tensors that real mixed checkpoints keep at 8 bits.
fn keeps_eight_bits(name: &str) -> bool {
    name.contains("embed_tokens") || name.contains("lm_head") || name.contains("out_proj")
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
    /// The language model's weights: every safetensors file but the image encoder's.
    pub fn open(dir: &Path) -> Result<Self, String> {
        Self::open_where(dir, |name| !name.starts_with("model-vision"))
    }

    /// The image encoder's weights, which some checkpoints keep in a file of their own.
    pub fn open_vision(dir: &Path) -> Result<Self, String> {
        Self::open_where(dir, |name| name.starts_with("model-vision"))
    }

    /// The language model's weights and its image encoder's, for a chip that carries both.
    pub fn open_with_vision(dir: &Path) -> Result<Self, String> {
        Self::open_where(dir, |_| true)
    }

    fn open_where(dir: &Path, keep: impl Fn(&str) -> bool) -> Result<Self, String> {
        let mut paths: Vec<_> = std::fs::read_dir(dir)
            .map_err(|e| e.to_string())?
            .filter_map(Result::ok)
            .map(|e| e.path())
            .filter(|p| {
                let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
                name.ends_with(".safetensors") && keep(name)
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
        Ok(Self { accelerator: None, files, tensors, format: None, references: Mutex::new(HashMap::new()), recoded: Mutex::new(HashMap::new()), recode_budget: AtomicUsize::new(default_recode_budget()), gpu_products: AtomicUsize::new(0), largest: Mutex::new(HashMap::new()) })
    }

    /// How many entries a tensor has along each axis.
    pub fn shape(&self, name: &str) -> Option<&[usize]> {
        self.tensors.get(name).map(|t| t.shape.as_slice())
    }

    /// Every tensor as stored in the checkpoint: name, data type and bytes, by name.
    pub fn raw_tensors(&self) -> Vec<(&str, &str, &[u8])> {
        let mut list: Vec<(&str, &str, &[u8])> = self.tensors.iter().map(|(name, t)| (name.as_str(), t.dtype.as_str(), &self.files[t.file][t.start..t.end])).collect();
        list.sort_by_key(|(name, ..)| *name);
        list
    }

    /// Makes every matrix read back as if stored in `format`.
    pub fn with_format(mut self, format: Option<WeightFormat>) -> Self {
        self.format = format;
        self
    }

    /// Runs the large matrix products on this system's GPU, if it has a usable one.
    pub fn with_gpu(mut self) -> Self {
        let files: Vec<&[u8]> = self.files.iter().map(|m| &m[..]).collect();
        self.accelerator = accel::open(&files);
        self
    }

    /// The GPU in use, if any.
    pub fn accelerator_name(&self) -> Option<String> {
        self.accelerator.as_ref().map(|a| a.name())
    }

    /// How many matrix products the GPU has completed.
    pub fn gpu_products(&self) -> usize {
        self.gpu_products.load(Ordering::Relaxed)
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
            return Ok(Matrix { rows, owner: self, name: name.to_string(), source: Source::Packed { words_per_row: last[0], groups: *ts.shape.last().unwrap_or(&1), words: bytes, scales, biases } });
        }
        let format = Plain::parse(&t.dtype).ok_or_else(|| format!("{name}: tipo {} no soportado", t.dtype))?;
        // Integer checkpoints may carry the factor that turns codes into real values.
        let factor = match self.floats(&format!("{name}.scale")) {
            Ok(values) if !values.is_empty() => values[0],
            _ => format.default_factor(),
        };
        Ok(Matrix { rows, owner: self, name: name.to_string(), source: Source::Plain { format, bytes, factor } })
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

/// The codes of a row in the cells, and each group's scale and bias.
pub type CellRow = (Vec<i64>, Vec<(i64, i64)>);

pub struct Matrix<'a> {
    rows: usize,
    owner: &'a Weights,
    name: String,
    source: Source<'a>,
}

type Requant = (WeightFormat, Arc<Reference>);

#[inline]
fn half_at(bytes: &[u8], index: usize) -> f32 {
    f16::from_le_bytes([bytes[2 * index], bytes[2 * index + 1]]).to_f32()
}

const AFFINE_EXTRA: i32 = 4;

fn mantissa_limit(bits: u32) -> i64 {
    (1i64 << (bits - 1)) - 1
}

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

    fn decode_into(&self, index: usize, y: &mut [f32]) {
        let cols = y.len();
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
                        let q = packed_code(row, at, bits);
                        y[at] = q as f32 * scale + bias;
                    }
                }
            }
            Source::Plain { format, bytes, factor } => {
                let size = format.size();
                let row = &bytes[index * cols * size..(index + 1) * cols * size];
                for (out, element) in y.iter_mut().zip(row.chunks_exact(size)) {
                    *out = format.decode(element, *factor);
                }
            }
        }
    }

    /// The format to rewrite this matrix in, or `None` when it is stored as the chip wants.
    fn requant(&self, cols: usize) -> Option<Requant> {
        let mut format = self.owner.format?;
        if let Source::Packed { words_per_row, groups, .. } = &self.source {
            let bits = (words_per_row * 32 / cols) as u32;
            let same = format.coding == WeightCoding::Affine && format.bias && format.bits == bits && format.group_size as usize == cols / groups;
            if format.keep_source || same {
                return None;
            }
        } else if format.keep_source && keeps_eight_bits(&self.name) {
            format.bits = format.bits.max(8);
            format.group_size *= 2;
        }
        let mut cache = self.owner.references.lock().expect("caché de referencias");
        let reference = cache.entry(self.name.clone()).or_insert_with(|| Arc::new(self.reference(cols, &format))).clone();
        Some((format, reference))
    }

    /// This tensor already rewritten in the chip's format, when it is narrow enough to keep.
    fn recoded(&self, cols: usize, format: &WeightFormat, reference: &Reference) -> Option<Arc<Recoded>> {
        if format.bits > 8 {
            return None;
        }
        let mut cache = self.owner.recoded.lock().expect("caché de pesos");
        if let Some(found) = cache.get(&self.name) {
            return found.clone();
        }
        let groups = if format.group_size > 0 { cols.div_ceil(format.group_size as usize) } else { 1 };
        let cost = self.rows * cols + self.rows * groups * 8;
        let budget = &self.owner.recode_budget;
        let granted = budget.try_update(Ordering::Relaxed, Ordering::Relaxed, |left| left.checked_sub(cost)).is_ok();
        let built = if granted { recode(format, reference, self.rows, cols, |index, out| self.decode_into(index, out)).map(Arc::new) } else { None };
        cache.insert(self.name.clone(), built.clone());
        built
    }

    fn reference(&self, cols: usize, format: &WeightFormat) -> Reference {
        let (min, max) = match &self.source {
            Source::Packed { words_per_row, groups, scales, biases, .. } => {
                let top = ((1u32 << (words_per_row * 32 / cols)) - 1) as f32;
                (0..self.rows * groups)
                    .into_par_iter()
                    .with_min_len(4096)
                    .map(|i| {
                        let (scale, bias) = (half_at(scales, i), half_at(biases, i));
                        let end = top * scale + bias;
                        (bias.min(end), bias.max(end))
                    })
                    .reduce(|| (f32::MAX, f32::MIN), |a, b| (a.0.min(b.0), a.1.max(b.1)))
            }
            Source::Plain { format, bytes, factor } => bytes
                .par_chunks_exact(format.size())
                .with_min_len(4096)
                .map(|e| {
                    let v = format.decode(e, *factor);
                    (v, v)
                })
                .reduce(|| (f32::MAX, f32::MIN), |a, b| (a.0.min(b.0), a.1.max(b.1))),
        };
        let mut table = Vec::new();
        if format.coding == WeightCoding::Codebook {
            let stride = (self.rows / 256).max(1);
            let mut row = vec![0f32; cols];
            let mut sample = Vec::new();
            for index in (0..self.rows).step_by(stride) {
                self.decode_into(index, &mut row);
                sample.extend_from_slice(&row);
            }
            table = train_table(sample, 1usize << format.bits.min(12));
        }
        Reference { min, max, table }
    }

    pub fn row(&self, index: usize, out: usize) -> Result<Vec<f32>, String> {
        if index >= self.rows {
            return Err("fila fuera de rango".into());
        }
        let mut y = vec![0f32; out];
        if let Some((format, reference)) = self.requant(out) {
            if let Some(recoded) = self.recoded(out, &format, &reference) {
                recoded.decode_into(index, &mut y);
            } else {
                self.decode_into(index, &mut y);
                quantize(&format, &reference, &mut y);
            }
        } else {
            self.decode_into(index, &mut y);
        }
        Ok(y)
    }

    fn gpu_for(&self, weights: usize) -> Option<&dyn Accelerator> {
        self.owner.accelerator.as_deref().filter(|_| weights >= GPU_MIN_WEIGHTS)
    }

    /// `y = M · x`, in parallel across rows. `x.len()` is the column count.
    pub fn apply(&self, x: &[f32], y: &mut [f32]) {
        self.apply_rows(0, x, y);
    }

    /// Like `apply`, over the rows starting at `first`; `y.len()` sets how many.
    pub fn apply_rows(&self, first: usize, x: &[f32], y: &mut [f32]) {
        let cols = x.len();
        if let Some((format, reference)) = self.requant(cols) {
            if let Some(recoded) = self.recoded(cols, &format, &reference) {
                let sums: Vec<f32> = x.chunks(recoded.group_size).map(|c| c.iter().sum()).collect();
                y.par_iter_mut().with_min_len(32).enumerate().for_each(|(index, out)| *out = recoded.dot(first + index, x, &sums));
                return;
            }
            y.par_iter_mut().with_min_len(8).enumerate().for_each_init(
                || vec![0f32; cols],
                |row, (index, out)| {
                    self.decode_into(first + index, row);
                    quantize(&format, &reference, row);
                    *out = row.iter().zip(x).map(|(w, v)| w * v).sum();
                },
            );
            return;
        }
        match &self.source {
            Source::Packed { words_per_row, groups, words, scales, biases } => {
                if let Some(accelerator) = self.gpu_for(y.len() * cols) {
                    let tensor = PackedTensor { words, scales, biases, bits: (words_per_row * 32 / cols) as u32, stride: words_per_row * 4, groups: *groups };
                    if accelerator.float_product(&tensor, first, x, y) {
                        self.owner.gpu_products.fetch_add(1, Ordering::Relaxed);
                        return;
                    }
                }
                let group_size = cols / groups;
                let bits = words_per_row * 32 / cols;
                let sums: Vec<f32> = x.chunks_exact(group_size).map(|c| c.iter().sum()).collect();
                let stride = words_per_row * 4;
                y.par_iter_mut().with_min_len(32).enumerate().for_each(|(index, out)| {
                    let r = first + index;
                    let row = &words[r * stride..(r + 1) * stride];
                    let mut total = 0f32;
                    for (g, xs) in x.chunks_exact(group_size).enumerate() {
                        let dot = match bits {
                            4 => dot_nibbles(&row[g * group_size / 2..(g + 1) * group_size / 2], xs),
                            8 => dot_bytes(&row[g * group_size..(g + 1) * group_size], xs),
                            _ => xs.iter().enumerate().map(|(j, v)| packed_code(row, g * group_size + j, bits) as f32 * v).sum(),
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

    /// The largest magnitude this tensor can produce, computed once.
    fn largest_weight(&self, cols: usize) -> f32 {
        if let Some(largest) = self.owner.largest.lock().expect("caché de máximos").get(&self.name) {
            return *largest;
        }
        let Source::Packed { words_per_row, groups, scales, biases, .. } = &self.source else { return 0.0 };
        let top = ((1u32 << (words_per_row * 32 / cols)) - 1) as f32;
        let largest = (0..self.rows * groups)
            .into_par_iter()
            .with_min_len(4096)
            .map(|i| {
                let (scale, bias) = (half_at(scales, i), half_at(biases, i));
                bias.abs().max((top * scale + bias).abs())
            })
            .reduce(|| 0.0, f32::max);
        self.owner.largest.lock().expect("caché de máximos").insert(self.name.clone(), largest);
        largest
    }

    /// The exponent shared by the matrix's 16-bit mantissas.
    pub fn exponent(&self, cols: usize) -> i64 {
        if let Some((_, reference)) = self.requant(cols) {
            return exponent_of(reference.largest());
        }
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
    pub fn fixed_row(&self, index: usize, cols: usize, exponent: i64, bits: u32) -> Result<Vec<i64>, String> {
        let limit = mantissa_limit(bits);
        let fraction = bits as i32 - 1;
        if self.requant(cols).is_some() {
            let scale = 2f64.powi(fraction - exponent as i32);
            return Ok(self.row(index, cols)?.iter().map(|v| ((*v as f64 * scale).round() as i64).clamp(-limit, limit)).collect());
        }
        match &self.source {
            Source::Packed { words_per_row, groups, words, scales, biases } => {
                let group_size = cols / groups;
                let bits = words_per_row * 32 / cols;
                let row = &words[index * words_per_row * 4..(index + 1) * words_per_row * 4];
                let factor = fixed_factor(exponent, fraction);
                let mut out = Vec::with_capacity(cols);
                for g in 0..*groups {
                    let (scale, bias) = affine_integers(half_at(scales, index * groups + g), half_at(biases, index * groups + g), factor);
                    for j in 0..group_size {
                        let at = g * group_size + j;
                        let q = packed_code(row, at, bits);
                        out.push(affine_weight(q, scale, bias, limit));
                    }
                }
                Ok(out)
            }
            Source::Plain { .. } => {
                let scale = 2f64.powi(fraction - exponent as i32);
                Ok(self.row(index, cols)?.iter().map(|v| ((*v as f64 * scale).round() as i64).clamp(-limit, limit)).collect())
            }
        }
    }

    /// The bits a code of this matrix takes in the cells.
    pub fn cell_code_bits(&self, cols: usize, bits: u32) -> u32 {
        match &self.source {
            Source::Packed { words_per_row, .. } if self.requant(cols).is_none() => (words_per_row * 32 / cols) as u32,
            _ => bits,
        }
    }

    /// The columns that share a scale and a bias in the cells: size and number of groups.
    pub fn cell_groups(&self, cols: usize) -> (usize, usize) {
        match &self.source {
            Source::Packed { groups, .. } if self.requant(cols).is_none() => (cols / groups, *groups),
            _ => (cols, 1),
        }
    }

    /// One row as the cells keep it: a signed code for each weight and, for each group,
    /// a scale and a bias with `FACTOR_FRACTION` extra bits.
    pub fn cell_row(&self, index: usize, cols: usize, exponent: i64, bits: u32) -> Result<CellRow, String> {
        if let Source::Packed { words_per_row, groups, words, scales, biases } = &self.source {
            if self.requant(cols).is_none() {
                let group_size = cols / groups;
                let width = words_per_row * 32 / cols;
                let row = &words[index * words_per_row * 4..(index + 1) * words_per_row * 4];
                let middle = 1i64 << (width - 1);
                let factor = fixed_factor(exponent, bits as i32 - 1);
                let code = |at: usize| packed_code(row, at, width);
                let codes = (0..cols).map(|at| code(at) as i64 - middle).collect();
                let factors = (0..*groups)
                    .map(|g| {
                        let (scale, bias) = affine_integers(half_at(scales, index * groups + g), half_at(biases, index * groups + g), factor);
                        (scale, bias + scale * middle)
                    })
                    .collect();
                debug_assert_eq!(group_size * groups, cols);
                return Ok((codes, factors));
            }
        }
        Ok((self.fixed_row(index, cols, exponent, bits)?, vec![(1 << AFFINE_EXTRA, 0)]))
    }

    /// `Σ w · x` for each row as the cells add it: the factors apply to each group's
    /// whole sum, and the extra bits go at the end, so no weight is rounded on its own.
    pub fn cell_sums(&self, first: usize, rows: usize, x: &[i32], exponent: i64, bits: u32) -> Vec<i64> {
        let cols = x.len();
        let Source::Packed { words_per_row, groups, words, scales, biases } = &self.source else {
            return self.exact_sums(first, rows, x, exponent, bits);
        };
        if self.requant(cols).is_some() {
            return self.exact_sums(first, rows, x, exponent, bits);
        }
        let mut sums = vec![0i64; rows];
        let fraction = bits as i32 - 1;
        let fits_int = |largest: f32| (largest as f64) * 2f64.powi(fraction + AFFINE_EXTRA - exponent as i32) < (1u64 << 30) as f64;
        if let Some(accelerator) = self.gpu_for(rows * cols).filter(|_| fits_int(self.largest_weight(cols))) {
            let tensor = PackedTensor { words, scales, biases, bits: (words_per_row * 32 / cols) as u32, stride: words_per_row * 4, groups: *groups };
            if accelerator.exact_product(&tensor, first, x, ExactScale { shift: fraction + AFFINE_EXTRA - exponent as i32, limit: 0 }, &mut sums) {
                self.owner.gpu_products.fetch_add(1, Ordering::Relaxed);
                return sums;
            }
        }
        let group_size = cols / groups;
        let width = words_per_row * 32 / cols;
        let stride = words_per_row * 4;
        let factor = fixed_factor(exponent, fraction);
        sums.par_iter_mut().with_min_len(16).enumerate().for_each(|(index, out)| {
            let r = first + index;
            let row = &words[r * stride..(r + 1) * stride];
            let mut total = 0i64;
            for (g, xs) in x.chunks_exact(group_size).enumerate() {
                let (scale, bias) = affine_integers(half_at(scales, r * groups + g), half_at(biases, r * groups + g), factor);
                let (mut dot, mut sum) = (0i64, 0i64);
                for (j, v) in xs.iter().enumerate() {
                    let at = g * group_size + j;
                    let q = packed_code(row, at, width);
                    dot += q as i64 * *v as i64;
                    sum += *v as i64;
                }
                total += scale * dot + bias * sum;
            }
            *out = total >> AFFINE_EXTRA;
        });
        sums
    }

    /// Full-precision `Σ w · x` for each row, using the mantissas of `fixed_row`.
    pub fn exact_sums(&self, first: usize, rows: usize, x: &[i32], exponent: i64, bits: u32) -> Vec<i64> {
        let limit = mantissa_limit(bits);
        let fraction = bits as i32 - 1;
        let cols = x.len();
        let mut sums = vec![0i64; rows];
        if let Some((format, reference)) = self.requant(cols) {
            let scale = 2f64.powi(fraction - exponent as i32);
            let recoded = self.recoded(cols, &format, &reference);
            sums.par_iter_mut().with_min_len(8).enumerate().for_each_init(
                || vec![0f32; cols],
                |row, (index, out)| {
                    if let Some(recoded) = &recoded {
                        recoded.decode_into(first + index, row);
                    } else {
                        self.decode_into(first + index, row);
                        quantize(&format, &reference, row);
                    }
                    *out = row.iter().zip(x).map(|(w, v)| ((*w as f64 * scale).round() as i64).clamp(-limit, limit) * *v as i64).sum();
                },
            );
            return sums;
        }
        match &self.source {
            Source::Packed { words_per_row, groups, words, scales, biases } => {
                let fits_int = |largest: f32| (largest as f64) * 2f64.powi(fraction + AFFINE_EXTRA - exponent as i32) < (1u64 << 30) as f64;
                if let Some(accelerator) = self.gpu_for(rows * cols).filter(|_| fits_int(self.largest_weight(cols))) {
                    let tensor = PackedTensor { words, scales, biases, bits: (words_per_row * 32 / cols) as u32, stride: words_per_row * 4, groups: *groups };
                    if accelerator.exact_product(&tensor, first, x, ExactScale { shift: fraction + AFFINE_EXTRA - exponent as i32, limit }, &mut sums) {
                        self.owner.gpu_products.fetch_add(1, Ordering::Relaxed);
                        return sums;
                    }
                }
                let group_size = cols / groups;
                let bits = words_per_row * 32 / cols;
                let stride = words_per_row * 4;
                let factor = fixed_factor(exponent, fraction);
                sums.par_iter_mut().with_min_len(16).enumerate().for_each(|(index, out)| {
                    let r = first + index;
                    let row = &words[r * stride..(r + 1) * stride];
                    let mut code = vec![0u8; group_size];
                    let mut total = 0i64;
                    for (g, xs) in x.chunks_exact(group_size).enumerate() {
                        if bits == 8 {
                            code.copy_from_slice(&row[g * group_size..(g + 1) * group_size]);
                        } else {
                            for (j, c) in code.iter_mut().enumerate() {
                                *c = packed_code(row, g * group_size + j, bits);
                            }
                        }
                        let (scale, bias) = affine_integers(half_at(scales, r * groups + g), half_at(biases, r * groups + g), factor);
                        total += dot_affine(&code, xs, scale, bias, limit);
                    }
                    *out = total;
                });
            }
            Source::Plain { format, bytes, factor, .. } => {
                let scale = 2f64.powi(fraction - exponent as i32);
                let size = format.size();
                sums.par_iter_mut().with_min_len(16).enumerate().for_each(|(index, out)| {
                    let r = first + index;
                    let row = &bytes[r * cols * size..(r + 1) * cols * size];
                    *out = row
                        .chunks_exact(size)
                        .zip(x)
                        .map(|(e, v)| ((format.decode(e, *factor) as f64 * scale).round() as i64).clamp(-limit, limit) * *v as i64)
                        .sum();
                });
            }
        }
        sums
    }
}

fn fixed_factor(exponent: i64, fraction: i32) -> f64 {
    2f64.powi(fraction + AFFINE_EXTRA - exponent as i32)
}

#[inline]
fn affine_integers(scale: f32, bias: f32, factor: f64) -> (i64, i64) {
    ((scale as f64 * factor).round() as i64, (bias as f64 * factor).round() as i64)
}

#[inline]
fn affine_weight(q: u8, scale: i64, bias: i64, limit: i64) -> i64 {
    ((q as i64 * scale + bias) >> AFFINE_EXTRA).clamp(-limit, limit)
}

/// `Σ ((q·scale + bias) >> 4) · x`, written so the compiler can vectorise it.
#[cfg_attr(target_arch = "x86_64", multiversion::multiversion(targets("x86_64+avx512f", "x86_64+avx2+fma")))]
#[cfg_attr(not(target_arch = "x86_64"), inline)]
fn dot_affine(codes: &[u8], x: &[i32], scale: i64, bias: i64, limit: i64) -> i64 {
    let mut total = 0i64;
    for (cs, xs) in codes.chunks(8).zip(x.chunks(8)) {
        let mut sum = [0i64; 8];
        for i in 0..cs.len() {
            let weight = ((cs[i] as i64 * scale + bias) >> AFFINE_EXTRA).clamp(-limit, limit);
            sum[i] = weight * xs[i] as i64;
        }
        total += sum.iter().sum::<i64>();
    }
    total
}

/// The code at `at` of a row MLX packs as one stream of `bits`-bit codes, the first in the
/// low bits of the first byte: 8, 4 or 2 bits share bytes evenly, 3, 5 or 6 cross them.
fn packed_code(row: &[u8], at: usize, bits: usize) -> u8 {
    let (byte, shift) = (at * bits / 8, at * bits % 8);
    let pair = row[byte] as u16 | (row.get(byte + 1).copied().unwrap_or(0) as u16) << 8;
    ((pair >> shift) & ((1 << bits) - 1)) as u8
}

#[cfg_attr(target_arch = "x86_64", multiversion::multiversion(targets("x86_64+avx512f", "x86_64+avx2+fma")))]
#[cfg_attr(not(target_arch = "x86_64"), inline)]
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

#[cfg_attr(target_arch = "x86_64", multiversion::multiversion(targets("x86_64+avx512f", "x86_64+avx2+fma")))]
#[cfg_attr(not(target_arch = "x86_64"), inline)]
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
