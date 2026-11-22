//! The model part of the chip image. The weights stay in the checkpoint; the manifest
//! records how each tensor is loaded and its checksum, as the real image would.

use crate::chip::{image_segments, Sharing};
use crate::lower::Layout;
use crate::tokenizer::TokenizerTables;
use crate::weights::Weights;
use compiler::{compile_model, Program};
use netlist::chip::{chip_info, GenerationSetup};
use netlist::image::{build_plan_image, ChipPart, Target};
use netlist::soc::FirmwareImage;
use netlist::{CellConfig, DieConfig, DieFeatures, Microprogram};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};

const VERSION: u32 = 2;
/// Cycles the load controller spends per 32-bit word when each element is one word.
const CYCLES_PER_WORD: u64 = 3;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImageSegment {
    pub name: String,
    pub target: u32,
    pub words: u64,
    pub crc: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Manifest {
    pub version: u32,
    pub segments: Vec<ImageSegment>,
}

/// What loading the image through the chip's controller would have done.
#[derive(Clone, Debug, Default, Serialize)]
pub struct LoadReport {
    pub segments: usize,
    pub words: u64,
    pub cycles: u64,
    pub seconds: f64,
    pub created: bool,
    pub crc_errors: Vec<String>,
}

fn target_of(name: &str, dtype: &str) -> Target {
    if name.ends_with(".scales") || name.ends_with(".biases") {
        Target::CellScale
    } else if dtype.starts_with('U') || dtype.starts_with('I') {
        Target::CellWeight
    } else {
        Target::Vector
    }
}

/// The checksum the load controller computes over a tensor, padded to whole words.
/// With `flip`, one stored bit reads wrong, as a damaged image would.
fn segment_crc(bytes: &[u8], flip: bool) -> u32 {
    let mut hasher = crc32fast::Hasher::new();
    if flip && !bytes.is_empty() {
        hasher.update(&[bytes[0] ^ 1]);
        hasher.update(&bytes[1..]);
    } else {
        hasher.update(bytes);
    }
    hasher.update(&[0u8; 3][..bytes.len().next_multiple_of(4) - bytes.len()]);
    !hasher.finalize()
}

/// The manifest of a checkpoint, computing the checksum of every tensor.
pub fn build_manifest(weights: &Weights) -> Manifest {
    let segments = weights
        .raw_tensors()
        .par_iter()
        .map(|(name, dtype, bytes)| ImageSegment { name: name.to_string(), target: target_of(name, dtype) as u32, words: bytes.len().div_ceil(4) as u64, crc: segment_crc(bytes, false) })
        .collect();
    Manifest { version: VERSION, segments }
}

/// Walks every segment as the controller would. `flip_first` damages the first segment.
pub fn verify(weights: &Weights, manifest: &Manifest, flip_first: bool) -> LoadReport {
    let start = web_time::Instant::now();
    let tensors = weights.raw_tensors();
    let results: Vec<Option<u32>> = manifest
        .segments
        .par_iter()
        .enumerate()
        .map(|(index, segment)| tensors.iter().find(|(name, ..)| *name == segment.name).map(|(_, _, bytes)| segment_crc(bytes, flip_first && index == 0)))
        .collect();
    let crc_errors = manifest.segments.iter().zip(&results).filter(|(segment, crc)| **crc != Some(segment.crc)).map(|(segment, _)| segment.name.clone()).collect();
    let words = manifest.segments.iter().map(|s| s.words).sum::<u64>();
    LoadReport { segments: manifest.segments.len(), words, cycles: words * CYCLES_PER_WORD, seconds: start.elapsed().as_secs_f64(), created: false, crc_errors }
}

/// Reads the manifest saved at `path`, or builds and saves it the first time.
pub fn load_or_build(weights: &Weights, path: &Path) -> Result<(Manifest, bool), String> {
    if let Some(manifest) = std::fs::read(path).ok().and_then(|bytes| serde_json::from_slice::<Manifest>(&bytes).ok()).filter(|m| m.version == VERSION) {
        return Ok((manifest, false));
    }
    let manifest = build_manifest(weights);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    std::fs::write(path, serde_json::to_vec(&manifest).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    Ok((manifest, true))
}

const FILE_MAGIC: &[u8; 8] = b"L2CIMG01";

/// What a chip image is built for.
#[derive(Clone, Debug, Deserialize)]
pub struct ImageRequest {
    /// The chip counts this one image can boot on.
    pub chips: Vec<usize>,
    pub context_tokens: u64,
    /// Cells that share the rows of each matrix.
    pub cells: usize,
    pub activation_bits: u32,
    /// log2 of the elements the die computes per cycle.
    pub lane_bits: u32,
    /// log2 of the copies of the vector unit.
    #[serde(default)]
    pub cluster_bits: u32,
    pub add_special: bool,
    pub stop: Vec<u32>,
    /// How the chips share the model when there are several.
    #[serde(default)]
    pub sharing: Sharing,
}

/// One tensor in the file, written cell by cell: rows `c`, `c + cells`, … for cell `c`.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ImageTensor {
    pub name: String,
    pub dtype: String,
    pub target: u32,
    pub offset: u64,
    pub bytes: u64,
    pub rows: u64,
    /// For a matrix in the cells, the rows of one use. Zero for any other tensor.
    #[serde(default)]
    pub block: u64,
    pub crc: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ImageManifest {
    pub version: u32,
    pub chips: Vec<usize>,
    pub cells: u64,
    pub boot_offset: u64,
    pub boot_words: u64,
    pub boot_crc: u32,
    pub tensors: Vec<ImageTensor>,
    #[serde(default)]
    pub sharing: Sharing,
}

#[derive(Clone, Debug, Serialize)]
pub struct ImageReport {
    pub bytes: u64,
    pub chips: Vec<usize>,
    pub tensors: usize,
    pub boot_words: u64,
    pub seconds: f64,
}

fn bits_for(count: u64) -> u32 {
    (u64::BITS - count.max(2).saturating_sub(1).leading_zeros()).max(1)
}

/// The smallest die that holds this program for this context.
pub fn fitted_die(program: &Program, weights: &Weights, request: &ImageRequest) -> Result<DieConfig, String> {
    let cell = CellConfig { weight_bits: 8, activation_bits: request.activation_bits, accumulator_bits: 2 * request.activation_bits, mac_stages: 1, adder_stages: 1, addr_bits: 3, coding: Default::default(), exponent_bits: 0, multiplier: netlist::multiplier::MultiplierKind::Array, fixed_weights: false, ecc: false, skip_zeros: false };
    // Every unit needs the same number of cells, in whole blocks of lanes.
    let share = 1usize << (request.lane_bits + request.cluster_bits);
    let rounded_cells = (request.cells.max(1) / share).max(1) as u32 * share as u32;
    let trial = DieConfig { cells: rounded_cells, cell, scale_bits: 16, reg_bits: 16, index_bits: 16, microcode_bits: 24, broadcast_stages: 1, function_bits: 12, function_select_bits: 3, features: DieFeatures::default().with_lane_bits(request.lane_bits).with_cluster_bits(request.cluster_bits).with_key_shift(bits_for(request.context_tokens.max(2)).max(request.lane_bits)).with_wide_link(true) };
    let tokens = request.context_tokens.max(1);
    let longest_store = program.kv_slots.iter().map(|s| (s.heads * s.head_dim) as u64 * tokens).max().unwrap_or(0);
    let longest_buffer = program.buffers.iter().map(|b| *b as u64).max().unwrap_or(1);
    let rope_rows = tokens * (3 * 128 + 1);
    let index_bits = bits_for(longest_store.max(longest_buffer).max(rope_rows)).min(26);
    // Measured on cells wide enough for any code: weights with no format take a whole
    // mantissa each.
    let roomy = DieConfig { index_bits: 26, cell: CellConfig { addr_bits: 26, weight_bits: cell.weight_bits.max(request.activation_bits), ..cell }, ..trial };
    let (code_bits, group) = Layout::for_image(program, weights, roomy, 1)?.cell_shape(weights)?;
    // Words as wide as the widest code: narrower codes share them.
    let roomy = DieConfig { cell: CellConfig { weight_bits: code_bits, ..roomy.cell }, ..roomy };
    let layout = Layout::for_image(program, weights, roomy, 1)?;
    // Only as many registers as the routines reach into, on any chip of any count the image
    // holds: the same silicon serves them all.
    let mut registers = 0;
    for chips in &request.chips {
        let share = crate::chip::ShareOut::new(program, weights, *chips, request.context_tokens as usize, request.activation_bits, request.sharing)?;
        for chip in 0..*chips {
            let mut part = Layout::for_image(program, weights, roomy, request.context_tokens as usize)?;
            share.microprogram(&mut part, program, chip)?;
            registers = registers.max(part.registers_needed());
        }
    }
    let (depth, groups) = layout.cell_depths();
    let a = request.activation_bits;
    let cell = CellConfig { weight_bits: code_bits, accumulator_bits: (code_bits + a + bits_for(group as u64)).max(2 * a + 4), addr_bits: bits_for(depth.max(2 * groups)), ..cell };
    Ok(DieConfig { reg_bits: bits_for(registers as u64 + 2), index_bits: index_bits.max(cell.addr_bits), cell, scale_bits: a + 5, ..trial })
}

/// One boot plan per chip count, all on the same hardware.
fn boot_plans(program: &Program, weights: &Weights, die: DieConfig, request: &ImageRequest) -> Result<Vec<Vec<ChipPart>>, String> {
    let setup = GenerationSetup { max_new: 256, greedy: true, limit: 1, temperature: 256, seed: 1, raw_tokens: false, stop: request.stop.clone(), draft_depth: 0, draft_logits: 0, think: None, think_budget: u32::MAX };
    let tokens = request.context_tokens as usize;
    let mut built: Vec<Vec<(Layout, Microprogram)>> = Vec::new();
    for chips in &request.chips {
        let share = crate::chip::ShareOut::new(program, weights, *chips, tokens, die.cell.activation_bits, request.sharing)?;
        let parts = (0..*chips)
            .map(|chip| {
                let mut layout = Layout::for_image(program, weights, die, tokens)?;
                let code = share.microprogram(&mut layout, program, chip)?;
                Ok::<_, String>((layout, code))
            })
            .collect::<Result<Vec<_>, _>>()?;
        built.push(parts);
    }
    // The same silicon serves every count, so it needs what any of the parts uses.
    let features = built.iter().flatten().map(|(_, code)| DieFeatures::of(code)).reduce(DieFeatures::union).unwrap_or_default().with_lane_bits(die.features.lane_bits).with_cluster_bits(die.features.cluster_bits).with_key_shift(die.features.key_shift).with_host_writes(program.vision.is_some());
    let words = built.iter().flatten().map(|(_, code)| code.words.len() as u64).max().unwrap_or(1);
    // Each register takes only the vector memory its routines reach, on the chip that needs most.
    let vector_bits = built.iter().flatten().map(|(layout, _)| layout.register_bases().1).max().unwrap_or(1);
    let die = DieConfig { features: features.with_vector_bits(vector_bits), microcode_bits: bits_for(words), ..die };
    Ok(built
        .iter()
        .map(|parts| parts.iter().map(|(layout, code)| ChipPart { info: chip_info(&die, code, layout.logits_register(program), program.vocab as u32, &setup), segments: image_segments(&die, code, layout) }).collect())
        .collect())
}

fn io(error: std::io::Error) -> String {
    error.to_string()
}

/// How many rows of a tensor with `rows` rows fall on `cell`.
fn rows_of_cell(rows: u64, cells: u64, cell: u64) -> u64 {
    (rows + cells - 1 - cell) / cells
}

/// Writes one file with everything a chip needs: firmware, tokenizer and microprograms
/// for each chip count, then the weights in the order the cells take them. The file
/// appears under its final name only once it is complete.
pub fn write_chip_image(dir: &Path, firmware: &FirmwareImage, request: &ImageRequest, out: &Path, done: &AtomicU64, total: &AtomicU64, cancel: &AtomicBool) -> Result<ImageReport, String> {
    let started = web_time::Instant::now();
    if request.chips.is_empty() || request.cells == 0 {
        return Err("the image needs at least one number of chips and one cell".into());
    }
    let config = std::fs::read_to_string(dir.join("config.json")).map_err(io)?;
    let program = compile_model(&config)?;
    let weights = Weights::open(dir)?;
    let die = fitted_die(&program, &weights, request)?;
    let plans = boot_plans(&program, &weights, die, request)?;
    let mut tables = TokenizerTables::from_file(&dir.join("tokenizer.json"), request.context_tokens)?;
    tables.set_add_special(request.add_special);
    let boot = build_plan_image(&firmware.program, &firmware.data, &tables.to_words(), &plans)?;
    let boot_bytes: Vec<u8> = boot.iter().flat_map(|w| w.to_le_bytes()).collect();

    let cells = request.cells as u64;
    // Matrices go in the form the cells take; only what is not in them stays as it came.
    let placed = Layout::for_image(&program, &weights, die, request.context_tokens.max(1) as usize)?;
    let in_cells = placed.cell_tensors();
    let is_in_cells = |name: &str| {
        let matrix = [".weight", ".scales", ".biases", ".scale"].iter().find_map(|end| name.strip_suffix(end)).unwrap_or(name);
        !placed.is_lookup_table(matrix) && in_cells.iter().any(|t| t.name == matrix)
    };
    let raw: Vec<_> = weights.raw_tensors().into_iter().filter(|(name, _, _)| !is_in_cells(name)).collect();
    let mut entries: Vec<ImageTensor> = raw
        .iter()
        .map(|(name, dtype, bytes)| {
            let rows = match weights.shape(name) {
                Some(shape) if shape.len() >= 2 => shape[..shape.len() - 1].iter().product::<usize>() as u64,
                _ => 1,
            };
            ImageTensor { name: name.to_string(), dtype: dtype.to_string(), target: target_of(name, dtype) as u32, offset: 0, bytes: bytes.len() as u64, rows, block: 0, crc: 0 }
        })
        .collect();
    let first_in_cells = entries.len();
    for tensor in &in_cells {
        let bytes = tensor.rows as u64 * tensor.row_bytes(&weights)? as u64;
        entries.push(ImageTensor { name: tensor.name.clone(), dtype: "cell".into(), target: Target::CellWeight as u32, offset: 0, bytes, rows: tensor.rows as u64, block: tensor.block as u64, crc: 0 });
    }
    let payload: u64 = entries.iter().map(|t| t.bytes).sum();
    total.store(payload + boot_bytes.len() as u64, Relaxed);

    // Offsets depend on the manifest's own length, so it reserves its final size first.
    let mut manifest = ImageManifest { version: VERSION, chips: request.chips.clone(), cells, boot_offset: 0, boot_words: boot.len() as u64, boot_crc: crc32fast::hash(&boot_bytes), tensors: Vec::new(), sharing: request.sharing };
    manifest.tensors = entries.clone();
    manifest.tensors.iter_mut().for_each(|t| {
        t.offset = u64::MAX;
        t.crc = u32::MAX;
    });
    manifest.boot_offset = u64::MAX;
    let reserved = serde_json::to_vec(&manifest).map_err(|e| e.to_string())?.len();
    let header = FILE_MAGIC.len() as u64 + 8 + reserved as u64;
    manifest.boot_offset = header;
    let mut offset = header + boot_bytes.len() as u64;
    for entry in &mut entries {
        entry.offset = offset;
        offset += entry.bytes;
    }

    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent).map_err(io)?;
    }
    let draft = out.with_extension("chipimg.partial");
    let mut file = BufWriter::with_capacity(1 << 22, std::fs::File::create(&draft).map_err(io)?);
    file.write_all(FILE_MAGIC).map_err(io)?;
    file.write_all(&(reserved as u64).to_le_bytes()).map_err(io)?;
    file.write_all(&vec![b' '; reserved]).map_err(io)?;
    file.write_all(&boot_bytes).map_err(io)?;
    done.store(boot_bytes.len() as u64, Relaxed);

    let (plain, encoded) = entries.split_at_mut(first_in_cells);
    for (entry, (_, _, bytes)) in plain.iter_mut().zip(&raw) {
        let row_bytes = (entry.bytes / entry.rows.max(1)) as usize;
        let spread = if row_bytes == 0 { 1 } else { cells.min(entry.rows.max(1)) };
        let mut hasher = crc32fast::Hasher::new();
        for cell in 0..spread {
            if cancel.load(Relaxed) {
                drop(file);
                let _ = std::fs::remove_file(&draft);
                return Err("cancelada".into());
            }
            let mut row = cell;
            while row < entry.rows {
                let chunk = &bytes[row as usize * row_bytes..(row as usize + 1) * row_bytes];
                file.write_all(chunk).map_err(io)?;
                hasher.update(chunk);
                row += spread;
            }
        }
        // A tensor that does not divide into rows goes as it is.
        if row_bytes * entry.rows as usize != bytes.len() {
            let tail = &bytes[row_bytes * entry.rows as usize..];
            file.write_all(tail).map_err(io)?;
            hasher.update(tail);
        }
        entry.crc = hasher.finalize();
        done.fetch_add(entry.bytes, Relaxed);
    }
    for (entry, tensor) in encoded.iter_mut().zip(&in_cells) {
        let mut hasher = crc32fast::Hasher::new();
        for cell in 0..tensor.spread() {
            if cancel.load(Relaxed) {
                drop(file);
                let _ = std::fs::remove_file(&draft);
                return Err("cancelada".into());
            }
            for row in tensor.rows_of(cell) {
                let bytes = tensor.encoded_row(&weights, row)?;
                file.write_all(&bytes).map_err(io)?;
                hasher.update(&bytes);
            }
        }
        entry.crc = hasher.finalize();
        done.fetch_add(entry.bytes, Relaxed);
    }
    file.flush().map_err(io)?;
    let mut file = file.into_inner().map_err(|e| e.to_string())?;
    manifest.tensors = entries;
    let mut text = serde_json::to_vec(&manifest).map_err(|e| e.to_string())?;
    if text.len() > reserved {
        return Err("the manifest does not fit in the space reserved".into());
    }
    text.resize(reserved, b' ');
    file.seek(SeekFrom::Start(16)).map_err(io)?;
    file.write_all(&text).map_err(io)?;
    file.sync_all().map_err(io)?;
    drop(file);
    std::fs::rename(&draft, out).map_err(io)?;
    Ok(ImageReport { bytes: offset, chips: request.chips.clone(), tensors: manifest.tensors.len(), boot_words: boot.len() as u64, seconds: started.elapsed().as_secs_f64() })
}

/// A chip image opened for checking or loading.
pub struct ChipImage {
    file: std::fs::File,
    pub manifest: ImageManifest,
}

impl ChipImage {
    pub fn open(path: &Path) -> Result<Self, String> {
        let mut file = std::fs::File::open(path).map_err(io)?;
        let mut head = [0u8; 16];
        file.read_exact(&mut head).map_err(io)?;
        if &head[..8] != FILE_MAGIC {
            return Err("not a chip image".into());
        }
        let length = u64::from_le_bytes(head[8..].try_into().expect("ocho bytes")) as usize;
        let mut text = vec![0u8; length];
        file.read_exact(&mut text).map_err(io)?;
        let manifest: ImageManifest = serde_json::from_slice(&text).map_err(|e| e.to_string())?;
        if manifest.version != VERSION {
            return Err("unknown image version".into());
        }
        Ok(Self { file, manifest })
    }

    pub fn boot(&mut self) -> Result<Vec<u32>, String> {
        self.file.seek(SeekFrom::Start(self.manifest.boot_offset)).map_err(io)?;
        let mut bytes = vec![0u8; self.manifest.boot_words as usize * 4];
        self.file.read_exact(&mut bytes).map_err(io)?;
        if crc32fast::hash(&bytes) != self.manifest.boot_crc {
            return Err("the boot part is damaged".into());
        }
        Ok(bytes.as_chunks::<4>().0.iter().map(|w| u32::from_le_bytes(*w)).collect())
    }

    /// Everything one cell stores of a tensor, in the order it takes it.
    pub fn cell_weights(&mut self, name: &str, cell: u64) -> Result<Vec<u8>, String> {
        let entry = self.manifest.tensors.iter().find(|t| t.name == name).ok_or_else(|| format!("{name} is not in the image"))?.clone();
        let (block, blocks) = if entry.block > 0 { (entry.block, entry.rows / entry.block) } else { (entry.rows.max(1), 1) };
        let spread = self.manifest.cells.min(block);
        if cell >= spread {
            return Ok(Vec::new());
        }
        let row_bytes = entry.bytes / entry.rows.max(1);
        let before: u64 = (0..cell).map(|c| blocks * rows_of_cell(block, spread, c)).sum();
        self.file.seek(SeekFrom::Start(entry.offset + before * row_bytes)).map_err(io)?;
        let mut bytes = vec![0u8; (blocks * rows_of_cell(block, spread, cell) * row_bytes) as usize];
        self.file.read_exact(&mut bytes).map_err(io)?;
        Ok(bytes)
    }

    /// What one cell of chip `index` of `chips` stores of a tensor when the chips share it
    /// by rows: each takes an equal slice and spreads it over its cells as one chip spreads
    /// the whole, so a cell's rows lie together among those the image keeps for one cell.
    pub fn cell_weights_shared(&mut self, name: &str, cell: u64, index: u64, chips: u64) -> Result<Vec<u8>, String> {
        let entry = self.manifest.tensors.iter().find(|t| t.name == name).ok_or_else(|| format!("{name} is not in the image"))?.clone();
        if entry.block > 0 && entry.block != entry.rows {
            return Err(format!("{name} is used by blocks and is not split by rows"));
        }
        let rows = entry.rows.max(1);
        let slice = rows / chips;
        let (whole, part) = (self.manifest.cells.min(rows), self.manifest.cells.min(slice));
        if cell >= part {
            return Ok(Vec::new());
        }
        let count = rows_of_cell(slice, part, cell);
        let first = index * slice + cell;
        if count > 1 && whole != part {
            return Err(format!("{name}: each chip has fewer rows than cells and the image does not group them that way"));
        }
        let (bucket, at) = (first % whole, first / whole);
        let row_bytes = entry.bytes / rows;
        let before: u64 = (0..bucket).map(|c| rows_of_cell(rows, whole, c)).sum::<u64>() + at;
        self.file.seek(SeekFrom::Start(entry.offset + before * row_bytes)).map_err(io)?;
        let mut bytes = vec![0u8; (count * row_bytes) as usize];
        self.file.read_exact(&mut bytes).map_err(io)?;
        Ok(bytes)
    }

    /// Checks every tensor and returns the names of those that do not match.
    pub fn verify(&mut self) -> Result<Vec<String>, String> {
        let mut damaged = Vec::new();
        let tensors = self.manifest.tensors.clone();
        for entry in tensors {
            self.file.seek(SeekFrom::Start(entry.offset)).map_err(io)?;
            let mut left = entry.bytes;
            let mut hasher = crc32fast::Hasher::new();
            let mut buffer = vec![0u8; 1 << 20];
            while left > 0 {
                let want = left.min(buffer.len() as u64) as usize;
                self.file.read_exact(&mut buffer[..want]).map_err(io)?;
                hasher.update(&buffer[..want]);
                left -= want as u64;
            }
            if hasher.finalize() != entry.crc {
                damaged.push(entry.name);
            }
        }
        Ok(damaged)
    }
}
