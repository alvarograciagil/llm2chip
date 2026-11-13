//! Show-o's transformer in the die's arithmetic against its floating-point reference.

mod common;

use common::{die_config, Rng, Writer};
use sim::showo::ShowO;
use sim::showo_chip::ShowOChip;
use std::path::Path;

const TEXT_VOCABULARY: usize = 50295;
const CODEBOOK: usize = 64;
const IMAGE: usize = 16;
const HIDDEN: usize = 64;
const LAYERS: usize = 2;

/// A Show-o with Phi-1.5's layout and random weights, small: two layers, two heads of 32.
fn make_toy(dir: &Path) {
    let vocab = TEXT_VOCABULARY + 10 + CODEBOOK + 1;
    let config = serde_json::json!({ "_class_name": "Showo", "llm_vocab_size": TEXT_VOCABULARY, "codebook_size": CODEBOOK, "mask_token_id": vocab - 1, "num_vq_tokens": IMAGE, "max_text": 8 });
    std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
    std::fs::create_dir_all(dir.join("phi")).unwrap();
    std::fs::write(dir.join("phi/config.json"), serde_json::json!({ "num_attention_heads": 2, "partial_rotary_factor": 0.5 }).to_string()).unwrap();
    let mut rng = Rng(0x5EED_0F5A_0000_0001);
    let mut w = Writer::default();
    let mut put = |w: &mut Writer, name: &str, shape: &[usize], spread: f32, offset: f32| {
        let values: Vec<f32> = (0..shape.iter().product()).map(|_| offset + spread * rng.unit()).collect();
        w.tensor(&format!("showo.{name}"), "F32", shape, values.iter().flat_map(|v| v.to_le_bytes()).collect());
    };
    let (h, up) = (HIDDEN, 2 * HIDDEN);
    put(&mut w, "model.embed_tokens.weight", &[vocab, h], 1.0, 0.0);
    for layer in 0..LAYERS {
        let p = |s: &str| format!("model.layers.{layer}.{s}");
        for norm in ["input_layernorm", "self_attn.q_layernorm", "self_attn.k_layernorm"] {
            let width = if norm == "input_layernorm" { h } else { h / 2 };
            put(&mut w, &p(&format!("{norm}.weight")), &[width], 0.2, 1.0);
            put(&mut w, &p(&format!("{norm}.bias")), &[width], 0.1, 0.0);
        }
        for (name, rows, cols) in [("self_attn.q_proj", h, h), ("self_attn.k_proj", h, h), ("self_attn.v_proj", h, h), ("self_attn.dense", h, h), ("mlp.fc1", up, h), ("mlp.fc2", h, up)] {
            put(&mut w, &p(&format!("{name}.weight")), &[rows, cols], 0.3, 0.0);
            put(&mut w, &p(&format!("{name}.bias")), &[rows], 0.1, 0.0);
        }
    }
    put(&mut w, "model.final_layernorm.weight", &[h], 0.2, 1.0);
    put(&mut w, "model.final_layernorm.bias", &[h], 0.1, 0.0);
    put(&mut w, "lm_head.weight", &[vocab, h], 0.3, 0.0);
    put(&mut w, "lm_head.bias", &[vocab], 0.1, 0.0);
    w.write(&dir.join("model.safetensors"));
}

/// A text, and an image with half its codes known and half masked.
fn inputs(model: &ShowO) -> (Vec<u32>, Vec<u32>) {
    let text = vec![464, 3797, 286, 257, 1097];
    let first = model.config.first_code();
    let image = (0..IMAGE as u32).map(|i| if i % 2 == 0 { model.config.mask_token } else { first + (i * 7) % CODEBOOK as u32 }).collect();
    (text, image)
}

/// Largest difference relative to the largest score, and the masked places whose best
/// code is the same.
fn compare(chip: &[f32], float: &[f32], image: &[u32], mask: u32, codebook: usize) -> (f32, usize, usize) {
    let largest = float.iter().fold(0f32, |m, v| m.max(v.abs()));
    let worst = chip.iter().zip(float).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
    let best = |row: &[f32]| row.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
    let masked: Vec<usize> = (0..image.len()).filter(|i| image[*i] == mask).collect();
    let agree = masked.iter().filter(|i| best(&chip[*i * codebook..][..codebook]) == best(&float[*i * codebook..][..codebook])).count();
    (worst / largest, agree, masked.len())
}

#[test]
fn show_o_s_scores_in_the_die_arithmetic_match_the_floating_point_ones() {
    let dir = std::env::temp_dir().join(format!("llm2chip-showo-chip-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    make_toy(&dir);
    let model = ShowO::open(&dir).unwrap().unwrap();
    let (text, image) = inputs(&model);
    let float = model.scores(&text, &image);
    let (ids, positions) = model.text_prefix(&text);
    // Vectors of 16 bits, and cells as wide, so the weights go in whole.
    let base = die_config();
    let die = netlist::DieConfig { reg_bits: 12, index_bits: 16, cell: netlist::CellConfig { weight_bits: 16, accumulator_bits: 40, ..base.cell }, ..base };
    let mut chip = ShowOChip::open(&dir, sim::Weights::open(&dir).unwrap(), die).unwrap();
    let scores = chip.scores(&ids, &positions, &image).unwrap();
    assert_eq!(scores.len(), float.len());
    let (error, agree, masked) = compare(&scores, &float, &image, model.config.mask_token, CODEBOOK);
    eprintln!("diferencia mayor {:.3} % de la mayor puntuación; mejor código igual en {agree} de {masked}", error * 100.0);
    assert!(error < 0.02, "{error}");
    assert!(agree * 10 >= masked * 9, "{agree} de {masked}");
    // The text and the image's other rows matter: with another text the scores move far
    // more than the arithmetic's error.
    let other = model.scores(&[1212, 318], &image);
    let (moved, _, _) = compare(&other, &float, &image, model.config.mask_token, CODEBOOK);
    assert!(moved > 10.0 * error, "otro texto mueve las puntuaciones {moved}");
    std::fs::remove_dir_all(&dir).ok();
}

/// The real Show-o, in `LLM2CHIP_SHOWO` or `~/.llm2chip/models/Show-o`: one pass over a
/// picture with its middle masked, on the chip's arithmetic and in floating point.
#[test]
#[ignore = "lenta: Show-o real, 1 300 M de parámetros, una pasada en la aritmética del die"]
fn the_real_show_o_scores_a_masked_picture_as_in_floating_point() {
    let dir = std::env::var_os("LLM2CHIP_SHOWO").map(std::path::PathBuf::from).unwrap_or_else(|| Path::new(&std::env::var("HOME").unwrap()).join(".llm2chip/models/Show-o"));
    if !dir.join("config.json").exists() {
        eprintln!("no está Show-o en {}: se omite", dir.display());
        return;
    }
    let model = ShowO::open(&dir).unwrap().unwrap();
    let tokenizer = tokenizers::Tokenizer::from_file(dir.join("phi/tokenizer.json")).unwrap();
    let text = tokenizer.encode("a green apple on a wooden table", false).unwrap().get_ids().to_vec();
    // A red disc on white, its codes, and the middle quarter of the grid masked.
    let side = model.config.side();
    let planes: Vec<f32> = (0..3)
        .flat_map(|c| (0..side * side).map(move |i| (c, i)))
        .map(|(c, i)| {
            let (x, y) = ((i % side) as f32 - side as f32 / 2.0, (i / side) as f32 - side as f32 / 2.0);
            let inside = x * x + y * y < (side as f32 * 0.3).powi(2);
            let value = if inside { [220.0, 30.0, 30.0][c] } else { 255.0 };
            value / 127.5 - 1.0
        })
        .collect();
    let codes = model.tokenizer().unwrap().encode(&planes).unwrap();
    let grid = model.config.grid();
    let image: Vec<u32> = codes
        .iter()
        .enumerate()
        .map(|(i, code)| if (grid / 4..3 * grid / 4).contains(&(i % grid)) && (grid / 4..3 * grid / 4).contains(&(i / grid)) { model.config.mask_token } else { model.config.first_code() + code })
        .collect();
    let started = std::time::Instant::now();
    let float = model.scores(&text, &image);
    eprintln!("coma flotante: {:.0} s", started.elapsed().as_secs_f64());
    let (ids, positions) = model.text_prefix(&text);
    let base = die_config();
    let die = netlist::DieConfig { reg_bits: 16, index_bits: 20, cell: netlist::CellConfig { weight_bits: 16, accumulator_bits: 40, ..base.cell }, ..base };
    let started = std::time::Instant::now();
    let mut chip = ShowOChip::open(&dir, sim::Weights::open(&dir).unwrap(), die).unwrap();
    eprintln!("chip preparado en {:.0} s", started.elapsed().as_secs_f64());
    let started = std::time::Instant::now();
    let scores = chip.scores(&ids, &positions, &image).unwrap();
    eprintln!("aritmética del die: {:.0} s", started.elapsed().as_secs_f64());
    let (error, agree, masked) = compare(&scores, &float, &image, model.config.mask_token, model.config.codebook as usize);
    eprintln!("diferencia mayor {:.2} % de la mayor puntuación; mejor código igual en {agree} de {masked}", error * 100.0);
    assert!(agree * 10 >= masked * 9, "{agree} de {masked}");
}
