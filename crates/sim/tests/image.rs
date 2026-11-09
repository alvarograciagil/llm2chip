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

mod chip_image {
    use netlist::soc::FirmwareImage;
    use sim::image::{write_chip_image, ChipImage, ImageRequest};
    use std::path::Path;
    use std::sync::atomic::{AtomicBool, AtomicU64};

    fn firmware() -> FirmwareImage {
        FirmwareImage::from_elf(include_bytes!("../../../firmware/prebuilt/firmware.elf")).unwrap()
    }

    fn fixture(label: &str) -> std::path::PathBuf {
        let source = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden/plain/llama");
        let dir = std::env::temp_dir().join(format!("llm2chip-chip-image-{label}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        for file in ["config.json", "model.safetensors"] {
            std::fs::copy(source.join(file), dir.join(file)).unwrap();
        }
        let tokenizer = r#"{"added_tokens":[],"normalizer":null,"pre_tokenizer":null,"post_processor":null,"decoder":{"type":"Fuse"},"model":{"type":"BPE","vocab":{"a":0,"b":1},"merges":[]}}"#;
        std::fs::write(dir.join("tokenizer.json"), tokenizer).unwrap();
        dir
    }

    #[test]
    fn an_image_holds_every_chip_count_and_each_cells_weights() {
        let dir = fixture("whole");
        let out = dir.join("modelo.chipimg");
        let request = ImageRequest { chips: vec![1, 2], context_tokens: 64, cells: 3, activation_bits: 16, add_special: false, stop: vec![2] };
        let (done, total, cancel) = (AtomicU64::new(0), AtomicU64::new(0), AtomicBool::new(false));
        let report = write_chip_image(&dir, &firmware(), &request, &out, &done, &total, &cancel).unwrap();
        assert_eq!(report.chips, [1, 2]);
        assert_eq!(done.into_inner(), total.into_inner());

        let mut image = ChipImage::open(&out).unwrap();
        assert!(image.verify().unwrap().is_empty());
        assert!(!image.boot().unwrap().is_empty());

        // The rows every cell receives, put back in order, are the original matrix.
        let weights = sim::Weights::open(&dir).unwrap();
        let name = "lm_head.weight";
        let original = weights.raw_tensors().into_iter().find(|t| t.0 == name).unwrap().2.to_vec();
        let shape = weights.shape(name).unwrap().to_vec();
        let row_bytes = original.len() / shape[0];
        let mut rebuilt = vec![0u8; original.len()];
        for cell in 0..3usize {
            let bytes = image.cell_weights(name, cell as u64).unwrap();
            for (index, row) in bytes.chunks(row_bytes).enumerate() {
                let at = (cell + 3 * index) * row_bytes;
                rebuilt[at..at + row_bytes].copy_from_slice(row);
            }
        }
        assert_eq!(rebuilt, original);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn damage_to_the_file_is_found() {
        let dir = fixture("damage");
        let out = dir.join("danada.chipimg");
        let request = ImageRequest { chips: vec![1], context_tokens: 64, cells: 2, activation_bits: 16, add_special: false, stop: vec![] };
        let (done, total, cancel) = (AtomicU64::new(0), AtomicU64::new(0), AtomicBool::new(false));
        write_chip_image(&dir, &firmware(), &request, &out, &done, &total, &cancel).unwrap();
        let offset = ChipImage::open(&out).unwrap().manifest.tensors[0].offset as usize;
        let mut bytes = std::fs::read(&out).unwrap();
        bytes[offset] ^= 1;
        std::fs::write(&out, bytes).unwrap();
        assert_eq!(ChipImage::open(&out).unwrap().verify().unwrap().len(), 1);
        std::fs::remove_dir_all(dir).unwrap();
    }
}
