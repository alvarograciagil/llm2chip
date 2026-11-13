//! MAGVIT-v2's decoder in the die's arithmetic against its floating-point reference.

mod common;

use common::{die_config, Rng, Writer};
use sim::magvit_chip::MagvitChip;
use sim::showo::Magvit;
use std::path::Path;

/// A MAGVIT-v2 decoder with random weights, small: 64 channels in the middle, a block that
/// goes down to 32 with its shortcut, and one doubling of the side.
fn make_toy(dir: &Path) {
    let mut rng = Rng(0x0DEC_0DE5_0000_0001);
    let mut w = Writer::default();
    let mut put = |w: &mut Writer, name: &str, shape: &[usize], spread: f32, offset: f32| {
        let values: Vec<f32> = (0..shape.iter().product()).map(|_| offset + spread * rng.unit()).collect();
        w.tensor(name, "F32", shape, values.iter().flat_map(|v| v.to_le_bytes()).collect());
    };
    let mut conv = |w: &mut Writer, name: &str, out: usize, inp: usize, k: usize| {
        let spread = 1.0 / ((inp * k * k) as f32).sqrt();
        put(w, &format!("{name}.weight"), &[out, inp, k, k], spread, 0.0);
        put(w, &format!("{name}.bias"), &[out], 0.1, 0.0);
    };
    conv(&mut w, "decoder.post_quant_conv", 13, 13, 1);
    conv(&mut w, "decoder.conv_in", 64, 13, 3);
    let blocks = [("decoder.mid.block_1", 64, 64), ("decoder.mid.block_2", 64, 64), ("decoder.up.1.block.0", 64, 64), ("decoder.up.0.block.0", 64, 32)];
    for (name, inp, out) in blocks {
        conv(&mut w, &format!("{name}.conv1"), out, inp, 3);
        conv(&mut w, &format!("{name}.conv2"), out, out, 3);
        if inp != out {
            conv(&mut w, &format!("{name}.nin_shortcut"), out, inp, 1);
        }
    }
    for part in ["q", "k", "v", "proj_out"] {
        conv(&mut w, &format!("decoder.mid.attn_1.{part}"), 64, 64, 1);
    }
    conv(&mut w, "decoder.up.1.upsample.conv", 64, 64, 3);
    conv(&mut w, "decoder.conv_out", 3, 32, 3);
    let put = |w: &mut Writer, name: &str, c: usize| {
        let values: Vec<f32> = (0..c).map(|i| 1.0 + 0.2 * ((i * 7 % 11) as f32 / 11.0 - 0.5)).collect();
        w.tensor(&format!("{name}.weight"), "F32", &[c], values.iter().flat_map(|v| v.to_le_bytes()).collect());
        let bias: Vec<f32> = (0..c).map(|i| 0.05 * ((i * 5 % 13) as f32 / 13.0 - 0.5)).collect();
        w.tensor(&format!("{name}.bias"), "F32", &[c], bias.iter().flat_map(|v| v.to_le_bytes()).collect());
    };
    for (name, c) in [("decoder.mid.block_1.norm1", 64), ("decoder.mid.block_1.norm2", 64), ("decoder.mid.block_2.norm1", 64), ("decoder.mid.block_2.norm2", 64), ("decoder.up.1.block.0.norm1", 64), ("decoder.up.1.block.0.norm2", 64), ("decoder.up.0.block.0.norm1", 64), ("decoder.up.0.block.0.norm2", 32), ("decoder.mid.attn_1.norm", 64), ("decoder.norm_out", 32)] {
        put(&mut w, name, c);
    }
    w.write(&dir.join("model.safetensors"));
}

/// An encoder with random weights: 32 channels, a block, halving, a block up to 64 with its
/// shortcut, the middle, and the 13 values of each code.
fn make_toy_encoder(dir: &Path) {
    let mut rng = Rng(0x0E4C_0DE5_0000_0001);
    let mut w = Writer::default();
    let mut put = |w: &mut Writer, name: &str, shape: &[usize], spread: f32, offset: f32| {
        let values: Vec<f32> = (0..shape.iter().product()).map(|_| offset + spread * rng.unit()).collect();
        w.tensor(name, "F32", shape, values.iter().flat_map(|v| v.to_le_bytes()).collect());
    };
    let mut conv = |w: &mut Writer, name: &str, out: usize, inp: usize, k: usize| {
        let spread = 1.0 / ((inp * k * k) as f32).sqrt();
        put(w, &format!("{name}.weight"), &[out, inp, k, k], spread, 0.0);
        put(w, &format!("{name}.bias"), &[out], 0.1, 0.0);
    };
    conv(&mut w, "encoder.conv_in", 32, 3, 3);
    let blocks = [("encoder.down.0.block.0", 32, 32), ("encoder.down.1.block.0", 32, 64), ("encoder.mid.block_1", 64, 64), ("encoder.mid.block_2", 64, 64)];
    for (name, inp, out) in blocks {
        conv(&mut w, &format!("{name}.conv1"), out, inp, 3);
        conv(&mut w, &format!("{name}.conv2"), out, out, 3);
        if inp != out {
            conv(&mut w, &format!("{name}.nin_shortcut"), out, inp, 1);
        }
    }
    conv(&mut w, "encoder.down.0.downsample.conv", 32, 32, 3);
    for part in ["q", "k", "v", "proj_out"] {
        conv(&mut w, &format!("encoder.mid.attn_1.{part}"), 64, 64, 1);
    }
    conv(&mut w, "encoder.conv_out", 13, 64, 3);
    conv(&mut w, "encoder.quant_conv", 13, 13, 1);
    let put = |w: &mut Writer, name: &str, c: usize| {
        let values: Vec<f32> = (0..c).map(|i| 1.0 + 0.2 * ((i * 7 % 11) as f32 / 11.0 - 0.5)).collect();
        w.tensor(&format!("{name}.weight"), "F32", &[c], values.iter().flat_map(|v| v.to_le_bytes()).collect());
        let bias: Vec<f32> = (0..c).map(|i| 0.05 * ((i * 5 % 13) as f32 / 13.0 - 0.5)).collect();
        w.tensor(&format!("{name}.bias"), "F32", &[c], bias.iter().flat_map(|v| v.to_le_bytes()).collect());
    };
    for (name, c) in [("encoder.down.0.block.0.norm1", 32), ("encoder.down.0.block.0.norm2", 32), ("encoder.down.1.block.0.norm1", 32), ("encoder.down.1.block.0.norm2", 64), ("encoder.mid.block_1.norm1", 64), ("encoder.mid.block_1.norm2", 64), ("encoder.mid.block_2.norm1", 64), ("encoder.mid.block_2.norm2", 64), ("encoder.mid.attn_1.norm", 64), ("encoder.norm_out", 64)] {
        put(&mut w, name, c);
    }
    w.write(&dir.join("model.safetensors"));
}

#[test]
fn magvit_s_encoder_in_the_die_arithmetic_gives_the_floating_point_codes() {
    let dir = std::env::temp_dir().join(format!("llm2chip-magvit-encoder-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    make_toy_encoder(&dir);
    let side = 8;
    let planes: Vec<f32> = (0..3 * side * side).map(|i| ((i * 37 % 101) as f32 / 50.0 - 1.0) * 0.9).collect();
    let float = Magvit::open(&dir).unwrap().encode(&planes).unwrap();
    let base = die_config();
    let die = netlist::DieConfig { reg_bits: 12, index_bits: 16, cell: netlist::CellConfig { weight_bits: 16, accumulator_bits: 40, ..base.cell }, ..base };
    let mut chip = MagvitChip::encoder(&dir, side, die).unwrap();
    let codes = chip.encode(&planes).unwrap();
    assert_eq!(codes.len(), float.len());
    let same = codes.iter().zip(&float).map(|(a, b)| 13 - (a ^ b).count_ones()).sum::<u32>();
    let total = 13 * codes.len() as u32;
    eprintln!("bits iguales: {same} de {total}; códigos iguales: {} de {}", codes.iter().zip(&float).filter(|(a, b)| a == b).count(), codes.len());
    assert!(same * 100 >= total * 97, "{same} de {total}");
    std::fs::remove_dir_all(&dir).ok();
}

fn psnr(a: &[u8], b: &[u8]) -> f64 {
    let error = a.iter().zip(b).map(|(x, y)| (*x as f64 - *y as f64).powi(2)).sum::<f64>() / a.len() as f64;
    10.0 * (255.0f64 * 255.0 / error.max(1e-9)).log10()
}

#[test]
fn magvit_s_decoder_in_the_die_arithmetic_draws_the_floating_point_picture() {
    let dir = std::env::temp_dir().join(format!("llm2chip-magvit-chip-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    make_toy(&dir);
    let grid = 4;
    let codes: Vec<u32> = (0..grid * grid).map(|i| (i as u32 * 2_654_435_761) >> 19).collect();
    let float = Magvit::open(&dir).unwrap().decode(&codes).unwrap();
    let base = die_config();
    let die = netlist::DieConfig { reg_bits: 12, index_bits: 16, cell: netlist::CellConfig { weight_bits: 16, accumulator_bits: 40, ..base.cell }, ..base };
    let mut chip = MagvitChip::decoder(&dir, grid, die).unwrap();
    assert_eq!(chip.side(), 2 * grid);
    let pixels = chip.decode(&codes).unwrap();
    assert_eq!(pixels.len(), float.len());
    let quality = psnr(&pixels, &float);
    let spread = float.iter().map(|v| *v as f64).fold((f64::MAX, f64::MIN), |(lo, hi), v| (lo.min(v), hi.max(v)));
    eprintln!("PSNR {quality:.1} dB frente a la coma flotante; valores de {} a {}", spread.0, spread.1);
    assert!(spread.1 - spread.0 > 30.0, "la imagen de referencia es casi plana");
    assert!(quality > 35.0, "{quality}");
    // Other codes give another picture, far from this one.
    let other: Vec<u32> = codes.iter().map(|c| c ^ 0x1555).collect();
    let moved = psnr(&chip.decode(&other).unwrap(), &float);
    assert!(moved < quality - 15.0, "otros códigos: {moved}");
    std::fs::remove_dir_all(&dir).ok();
}

/// The real MAGVIT-v2, in Show-o's folder (`LLM2CHIP_SHOWO` or `~/.llm2chip/models/Show-o`):
/// a picture's codes decoded on the chip's arithmetic and in floating point.
#[test]
#[ignore = "lenta: MAGVIT-v2 real, una imagen de 256 × 256 en la aritmética del die"]
fn the_real_magvit_decodes_a_picture_as_in_floating_point() {
    let dir = std::env::var_os("LLM2CHIP_SHOWO").map(std::path::PathBuf::from).unwrap_or_else(|| Path::new(&std::env::var("HOME").unwrap()).join(".llm2chip/models/Show-o")).join("magvitv2");
    if !dir.join("config.json").exists() {
        eprintln!("no está MAGVIT-v2 en {}: se omite", dir.display());
        return;
    }
    let magvit = Magvit::open(&dir).unwrap();
    // A red disc on white, its codes.
    let side = 256;
    let planes: Vec<f32> = (0..3)
        .flat_map(|c| (0..side * side).map(move |i| (c, i)))
        .map(|(c, i)| {
            let (x, y) = ((i % side) as f32 - 128.0, (i / side) as f32 - 128.0);
            (if x * x + y * y < 80.0 * 80.0 { [220.0, 30.0, 30.0][c] } else { 255.0 }) / 127.5 - 1.0
        })
        .collect();
    let codes = magvit.encode(&planes).unwrap();
    let started = std::time::Instant::now();
    let float = magvit.decode(&codes).unwrap();
    eprintln!("coma flotante: {:.0} s", started.elapsed().as_secs_f64());
    let base = die_config();
    let die = netlist::DieConfig { reg_bits: 16, index_bits: 24, cell: netlist::CellConfig { weight_bits: 16, accumulator_bits: 40, ..base.cell }, ..base };
    let started = std::time::Instant::now();
    let mut chip = MagvitChip::decoder(&dir, 16, die).unwrap();
    let pixels = chip.decode(&codes).unwrap();
    eprintln!("aritmética del die: {:.0} s", started.elapsed().as_secs_f64());
    let quality = psnr(&pixels, &float);
    eprintln!("PSNR {quality:.1} dB frente a la coma flotante");
    if let Ok(out) = std::env::var("LLM2CHIP_MAGVIT_OUT") {
        std::fs::write(out, &pixels).unwrap();
    }
    assert!(quality > 30.0, "{quality}");
}

/// The real encoder: a picture's codes on the chip's arithmetic and in floating point.
/// Values near zero may take either sign, so some bits differ.
#[test]
#[ignore = "lenta: el codificador de MAGVIT-v2 real, una imagen de 256 × 256 en la aritmética del die"]
fn the_real_magvit_encodes_a_picture_as_in_floating_point() {
    let dir = std::env::var_os("LLM2CHIP_SHOWO").map(std::path::PathBuf::from).unwrap_or_else(|| Path::new(&std::env::var("HOME").unwrap()).join(".llm2chip/models/Show-o")).join("magvitv2");
    if !dir.join("config.json").exists() {
        eprintln!("no está MAGVIT-v2 en {}: se omite", dir.display());
        return;
    }
    let magvit = Magvit::open(&dir).unwrap();
    let side = 256;
    let planes: Vec<f32> = (0..3)
        .flat_map(|c| (0..side * side).map(move |i| (c, i)))
        .map(|(c, i)| {
            let (x, y) = ((i % side) as f32 - 128.0, (i / side) as f32 - 128.0);
            (if x * x + y * y < 80.0 * 80.0 { [220.0, 30.0, 30.0][c] } else { 255.0 }) / 127.5 - 1.0
        })
        .collect();
    let float = magvit.encode(&planes).unwrap();
    let base = die_config();
    let die = netlist::DieConfig { reg_bits: 16, index_bits: 24, cell: netlist::CellConfig { weight_bits: 16, accumulator_bits: 40, ..base.cell }, ..base };
    let started = std::time::Instant::now();
    let codes = MagvitChip::encoder(&dir, side, die).unwrap().encode(&planes).unwrap();
    eprintln!("aritmética del die: {:.0} s", started.elapsed().as_secs_f64());
    let same = codes.iter().zip(&float).map(|(a, b)| 13 - (a ^ b).count_ones()).sum::<u32>();
    let total = 13 * codes.len() as u32;
    eprintln!("bits iguales: {same} de {total}; códigos iguales: {} de {}", codes.iter().zip(&float).filter(|(a, b)| a == b).count(), codes.len());
    assert!(same * 100 >= total * 95, "{same} de {total}");
}
