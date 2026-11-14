//! Shows the model an image and asks about it, all in floating point: the reference the
//! chip's image path is checked against. With `chip`, the die's arithmetic does it all
//! instead, the image encoder included, carried by the chip with the model.
//!
//! `cargo run --release -p sim --example see -- <folder> [shape] "<question>" [tokens] [chip]`
//!
//! The figure is drawn here, 256×256 pixels on white: `red` (a red disc), `blue` (a blue
//! square) or `green` (a green triangle).

use compiler::compile_qwen3_5;
use sim::vision::{image_positions, patches, VisionTower};
use sim::{Engine, ExactEngine, Numeric, Runner, Weights};
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
                "blue" => (dx.abs() < 70.0 && dy.abs() < 70.0, [30, 60, 220]),
                "green" => (y > 50 && y < 210 && dx.abs() < (y as f32 - 50.0) * 0.6, [30, 170, 60]),
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
    let kind = args.get(2).map_or("red", String::as_str);
    let question = args.get(3).map_or("What colour is the shape? Answer with a single word.", String::as_str);
    let count_out: usize = args.get(4).map_or(8, |n| n.parse().unwrap());
    let tower = VisionTower::open(dir)?.ok_or("this model has no image encoder")?;
    let c = tower.config.clone();
    let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| e.to_string())?;
    let encode = |text: &str| tokenizer.encode(text, false).map(|e| e.get_ids().to_vec()).map_err(|e| e.to_string());

    let on_chip = args.get(5).is_some_and(|a| a == "chip");
    let config = std::fs::read_to_string(dir.join("config.json"))?;
    let program = compile_qwen3_5(&config)?;
    let mut engine: Box<dyn Runner> = if on_chip {
        Box::new(ExactEngine::new(compiler::attach_vision(program, &config, 256)?, Weights::open_with_vision(dir)?, 16)?)
    } else {
        Box::new(Engine::new(program, Weights::open(dir)?, Numeric::Float)?)
    };
    engine.set_mrope(c.mrope_section);

    let started = std::time::Instant::now();
    let image = patches(&c.within(256), &figure(kind), SIDE, SIDE)?;
    let tokens = if on_chip { None } else { Some(tower.encode(&image)?) };
    let before = encode("<|im_start|>user\n<|vision_start|>")?;
    let after = encode(&format!("<|vision_end|>{question}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"))?;
    for id in &before {
        engine.step(*id, false)?;
    }
    let count = match &tokens {
        Some(tokens) => tokens.len(),
        None => engine.encode_image(&tower, &image)?.ok_or("the chip has no encoder")?,
    };
    eprintln!("image: {} patches, {count} tokens, {:.1} s", image.rows * image.cols, started.elapsed().as_secs_f64());
    let (positions, next) = image_positions(before.len() as u64, image.rows, image.cols, c.merge);
    for (index, position) in positions.into_iter().enumerate() {
        match &tokens {
            Some(tokens) => engine.step_embedding(&tokens[index], position, false)?,
            None => engine.step_image(index as u32, position, false)?,
        };
    }
    engine.continue_text_at(next);
    let mut last = 0;
    for (i, id) in after.iter().enumerate() {
        if let (Some(logits), _) = engine.step(*id, i + 1 == after.len())? {
            last = argmax(logits);
        }
    }
    let mut output = Vec::new();
    for _ in 0..count_out {
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
