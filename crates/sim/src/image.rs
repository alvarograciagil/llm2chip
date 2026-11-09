//! The model part of the chip image. The weights stay in the checkpoint; the manifest
//! records how each tensor is loaded and its checksum, as the real image would.

use crate::weights::Weights;
use netlist::image::Target;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::path::Path;

const VERSION: u32 = 1;
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
    let start = std::time::Instant::now();
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

/// The first `count` bytes of a stored tensor, for loading a real sample through the gates.
pub fn sample_codes(weights: &Weights, name: &str, count: usize) -> Option<Vec<u32>> {
    weights.raw_tensors().into_iter().find(|(tensor, ..)| *tensor == name).map(|(_, _, bytes)| bytes[..count.min(bytes.len())].iter().map(|b| *b as u32).collect())
}
