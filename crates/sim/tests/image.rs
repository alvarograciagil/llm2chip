//! The model image: its checksums match the chip's, and damage is reported.

use netlist::image::crc32;
use sim::image::{build_manifest, load_or_build, verify};
use sim::Weights;

mod common;
use common::*;

fn setup(name: &str) -> (std::path::PathBuf, Weights) {
    let dir = std::env::temp_dir().join(format!("llm2chip-image-{name}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    make_checkpoint(&dir);
    let weights = Weights::open(&dir).unwrap();
    (dir, weights)
}

#[test]
fn checksums_match_the_load_controller() {
    let (dir, weights) = setup("crc");
    let manifest = build_manifest(&weights);
    assert!(manifest.segments.len() > 10);
    for (segment, (name, _, bytes)) in manifest.segments.iter().zip(weights.raw_tensors()) {
        assert_eq!(segment.name, name);
        let mut padded = bytes.to_vec();
        padded.resize(bytes.len().next_multiple_of(4), 0);
        let words: Vec<u32> = padded.chunks(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
        assert_eq!(segment.crc, crc32(&words), "{name}");
    }
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn a_damaged_image_is_reported() {
    let (dir, weights) = setup("verify");
    let path = dir.join("manifest.json");
    let (manifest, created) = load_or_build(&weights, &path).unwrap();
    assert!(created);
    assert!(!load_or_build(&weights, &path).unwrap().1, "la segunda vez se reutiliza");
    let clean = verify(&weights, &manifest, false);
    assert!(clean.crc_errors.is_empty());
    assert_eq!(clean.cycles, 3 * clean.words);
    let damaged = verify(&weights, &manifest, true);
    assert_eq!(damaged.crc_errors, [manifest.segments[0].name.clone()]);
    std::fs::remove_dir_all(&dir).ok();
}
