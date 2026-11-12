//! The image encoder taken to the chip, stage by stage, on a toy encoder: each stage is
//! checked against the floating-point reference of `sim::vision`.

mod common;
use common::*;
use sim::vision::{patches, VisionTower};

#[test]
fn the_toy_encoder_turns_a_picture_into_tokens_of_the_model_width() {
    let dir = std::env::temp_dir().join(format!("llm2chip-vision-toy-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    make_vision_checkpoint(&dir);
    let tower = VisionTower::open(&dir).unwrap().expect("codificador");
    // 16 × 8 pixels: 8 × 4 patches, 4 × 2 tokens.
    let image = patches(&tower.config, &test_picture(16, 8), 16, 8).unwrap();
    assert_eq!((image.rows, image.cols), (4, 8));
    let tokens = tower.encode(&image).unwrap();
    assert_eq!(tokens.len(), 8);
    assert!(tokens.iter().all(|t| t.len() == 8 && t.iter().all(|v| v.is_finite())));
    std::fs::remove_dir_all(&dir).ok();
}

/// A die with room for everything, its vectors of `bits` bits.
fn roomy_die(bits: u32) -> netlist::DieConfig {
    let base = die_config();
    netlist::DieConfig { reg_bits: 12, index_bits: 16, microcode_bits: 18, cell: netlist::CellConfig { activation_bits: bits, accumulator_bits: 2 * bits + 8, ..base.cell }, ..base }
}

#[test]
fn the_encoder_in_the_die_arithmetic_gives_the_same_tokens_within_rounding() {
    let dir = std::env::temp_dir().join(format!("llm2chip-vision-exact-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    make_vision_checkpoint(&dir);
    let tower = VisionTower::open(&dir).unwrap().unwrap();
    let image = patches(&tower.config, &test_picture(16, 8), 16, 8).unwrap();
    let reference = tower.encode(&image).unwrap();
    let config = std::fs::read_to_string(dir.join("config.json")).unwrap();
    let mut exact = sim::vision_chip::ExactVision::new(&config, sim::Weights::open_vision(&dir).unwrap(), 32, roomy_die(16)).unwrap();
    let tokens = exact.encode(&tower, &image).unwrap();
    assert_eq!(tokens.len(), reference.len());
    let largest = reference.iter().flatten().fold(0f32, |m, v| m.max(v.abs()));
    let worst = tokens.iter().flatten().zip(reference.iter().flatten()).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
    eprintln!("mayor valor {largest:.3}, mayor diferencia {worst:.4}");
    assert!(worst < 0.02 * largest, "{tokens:?}\nfrente a\n{reference:?}");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn the_language_model_reads_the_chip_encoder_s_tokens_as_the_reference_ones() {
    let dir = std::env::temp_dir().join(format!("llm2chip-vision-lm-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    make_checkpoint(&dir);
    make_vision_checkpoint(&dir);
    let tower = VisionTower::open(&dir).unwrap().unwrap();
    let image = patches(&tower.config, &test_picture(16, 8), 16, 8).unwrap();
    let reference = tower.encode(&image).unwrap();
    let config = std::fs::read_to_string(dir.join("config.json")).unwrap();
    let chip = sim::vision_chip::ExactVision::new(&config, sim::Weights::open_vision(&dir).unwrap(), 32, roomy_die(16)).unwrap().encode(&tower, &image).unwrap();
    // The same language model, in the die's arithmetic, reads either set of tokens.
    let last = |tokens: &[Vec<f32>]| {
        let mut lm = sim::ExactEngine::new(compiler::compile_qwen3_5(CONFIG).unwrap(), sim::Weights::open(&dir).unwrap(), 16).unwrap();
        let mut logits = Vec::new();
        for (position, token) in tokens.iter().enumerate() {
            logits = lm.step_embedding(token, [position as u64; 3], position + 1 == tokens.len()).unwrap().unwrap_or_default();
        }
        logits
    };
    let (from_chip, from_reference) = (last(&chip), last(&reference));
    let best = |v: &[f32]| v.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
    assert_eq!(best(&from_chip), best(&from_reference), "{from_chip:?} frente a {from_reference:?}");
    std::fs::remove_dir_all(&dir).ok();
}
