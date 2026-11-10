//! Shows the model an image and asks about it, all in floating point: the reference the
//! chip's image path is checked against.
//!
//! `cargo run --release -p sim --example see -- <carpeta> [figura] "<pregunta>" [tokens]`
//!
//! The figure is drawn here, 256×256 pixels on white: `rojo` (a red disc), `azul` (a blue
//! square) or `verde` (a green triangle).

use compiler::compile_qwen3_5;
use sim::vision::{image_positions, patches, VisionTower};
use sim::{Engine, Numeric, Weights};
use std::path::Path;
use tokenizers::Tokenizer;

const SIDE: usize = 256;

/// A coloured figure on white.
fn figure(kind: &str) -> Vec<u8> {
    let mut rgb = vec![255u8; SIDE * SIDE * 3];
    for y in 0..SIDE {
        for x in 0..SIDE {
            let (dx, dy) = (x as f32 - 128.0, y as f32 - 128.0);
            let (inside, colour) = match kind {
                "azul" => (dx.abs() < 70.0 && dy.abs() < 70.0, [30, 60, 220]),
                "verde" => (y > 50 && y < 210 && dx.abs() < (y as f32 - 50.0) * 0.6, [30, 170, 60]),
                _ => (dx * dx + dy * dy < 80.0 * 80.0, [220, 30, 30]),
            };
            if inside {
                rgb[(y * SIDE + x) * 3..][..3].copy_from_slice(&colour);
            }
        }
    }
    rgb
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let dir = Path::new(&args[1]);
    let kind = args.get(2).map_or("rojo", String::as_str);
    let question = args.get(3).map_or("¿De qué color es la figura? Responde con una sola palabra.", String::as_str);
    let count: usize = args.get(4).map_or(8, |n| n.parse().unwrap());
    let tower = VisionTower::open(dir)?.ok_or("este modelo no trae codificador de imágenes")?;
    let c = tower.config.clone();
    let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| e.to_string())?;
    let encode = |text: &str| tokenizer.encode(text, false).map(|e| e.get_ids().to_vec()).map_err(|e| e.to_string());

    let started = std::time::Instant::now();
    let image = patches(&c, &figure(kind), SIDE, SIDE)?;
    let tokens = tower.encode(&image)?;
    eprintln!("imagen: {} parches, {} tokens, {:.1} s", image.rows * image.cols, tokens.len(), started.elapsed().as_secs_f64());

    let program = compile_qwen3_5(&std::fs::read_to_string(dir.join("config.json"))?)?;
    let mut engine = Engine::new(program, Weights::open(dir)?, Numeric::Float)?;
    engine.set_mrope(c.mrope_section);
    let before = encode("<|im_start|>user\n<|vision_start|>")?;
    let after = encode(&format!("<|vision_end|>{question}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"))?;
    for id in &before {
        engine.step(*id, false)?;
    }
    let (positions, next) = image_positions(before.len() as u64, image.rows, image.cols, c.merge);
    for (token, position) in tokens.iter().zip(positions) {
        engine.step_embedding(token, position, false)?;
    }
    engine.continue_text_at(next);
    let mut last = 0;
    for (i, id) in after.iter().enumerate() {
        if let (Some(logits), _) = engine.step(*id, i + 1 == after.len())? {
            last = argmax(logits);
        }
    }
    let mut output = Vec::new();
    for _ in 0..count {
        output.push(last);
        if last == 248046 || last == 248044 {
            break;
        }
        last = argmax(engine.step(last, true)?.0.unwrap());
    }
    println!("{}", tokenizer.decode(&output, true).map_err(|e| e.to_string())?);
    Ok(())
}

fn argmax(values: &[f32]) -> u32 {
    values.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map_or(0, |(i, _)| i as u32)
}
