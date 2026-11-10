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
        let request = ImageRequest { chips: vec![1, 2], context_tokens: 64, cells: 3, activation_bits: 16, lane_bits: 0, cluster_bits: 0, add_special: false, stop: vec![2] };
        let (done, total, cancel) = (AtomicU64::new(0), AtomicU64::new(0), AtomicBool::new(false));
        let report = write_chip_image(&dir, &firmware(), &request, &out, &done, &total, &cancel).unwrap();
        assert_eq!(report.chips, [1, 2]);
        assert_eq!(done.into_inner(), total.into_inner());

        let mut image = ChipImage::open(&out).unwrap();
        assert!(image.verify().unwrap().is_empty());
        assert!(!image.boot().unwrap().is_empty());

        // Each cell receives its rows as codes plus factors, in the order it reads them.
        let weights = sim::Weights::open(&dir).unwrap();
        let matrix = weights.matrix("lm_head").unwrap();
        // This checkpoint is in floating point, so each code is a 16-bit mantissa.
        let cols = *weights.shape("lm_head.weight").unwrap().last().unwrap();
        let groups = matrix.cell_groups(cols).1;
        let exponent = matrix.exponent(cols);
        let code_bytes = 2 * cols;
        let row_bytes = code_bytes + 8 * groups;
        let mut seen = 0;
        for cell in 0..3usize {
            let bytes = image.cell_weights("lm_head", cell as u64).unwrap();
            for (index, row) in bytes.chunks(row_bytes).enumerate() {
                let (codes, factors) = matrix.cell_row(cell + 3 * index, cols, exponent, 16).unwrap();
                let decoded: Vec<i64> = row[..code_bytes].chunks(2).map(|b| i64::from(i16::from_le_bytes([b[0], b[1]]))).collect();
                assert_eq!(decoded, codes, "códigos de la fila {}", cell + 3 * index);
                for (group, (scale, bias)) in factors.iter().enumerate() {
                    let at = code_bytes + 8 * group;
                    assert_eq!(i64::from(i32::from_le_bytes(row[at..at + 4].try_into().unwrap())), *scale);
                    assert_eq!(i64::from(i32::from_le_bytes(row[at + 4..at + 8].try_into().unwrap())), *bias);
                }
                seen += 1;
            }
        }
        assert_eq!(seen, matrix.rows());
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn damage_to_the_file_is_found() {
        let dir = fixture("damage");
        let out = dir.join("danada.chipimg");
        let request = ImageRequest { chips: vec![1], context_tokens: 64, cells: 2, activation_bits: 16, lane_bits: 0, cluster_bits: 0, add_special: false, stop: vec![] };
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
