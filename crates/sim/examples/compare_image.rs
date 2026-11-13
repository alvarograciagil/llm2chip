//! Finds the first buffer where the chip's integers drift away from floating point when
//! a picture's vector stands in for a token, on Janus.
//! `cargo run --release -p sim --example compare_image -- <folder> image.bmp`

use sim::janus::JanusEye;
use sim::{Engine, ExactEngine, Numeric, Weights};
use std::path::Path;

fn read_bmp(path: &Path) -> (Vec<u8>, usize, usize) {
    let data = std::fs::read(path).unwrap();
    let at = |o: usize| u32::from_le_bytes(data[o..o + 4].try_into().unwrap());
    let (offset, width, height) = (at(10) as usize, at(18) as usize, at(22) as i32);
    let rows = height.unsigned_abs() as usize;
    let stride = (width * 3).next_multiple_of(4);
    let mut rgb = Vec::with_capacity(width * rows * 3);
    for r in 0..rows {
        let y = if height < 0 { r } else { rows - 1 - r };
        for x in 0..width {
            let p = offset + y * stride + x * 3;
            rgb.extend([data[p + 2], data[p + 1], data[p]]);
        }
    }
    (rgb, width, rows)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let dir = Path::new(&args[1]);
    let (rgb, width, height) = read_bmp(Path::new(&args[2]));
    let eye = JanusEye::open(dir)?.ok_or("no es un Janus")?;
    let vectors = eye.encode(&rgb, width, height)?;
    let peak = |v: &[f32]| v.iter().fold(0f32, |m, x| m.max(x.abs()));
    // The vector with the largest value, where trouble would show first.
    let (largest, _) = vectors.iter().enumerate().max_by(|a, b| peak(a.1).total_cmp(&peak(b.1))).unwrap();
    let config: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json"))?)?;
    let text = config["text_config"].to_string();
    let program = compiler::compile_model(&text)?;
    let mut float = Engine::new(compiler::compile_model(&text)?, Weights::open(dir)?, Numeric::Float)?;
    let bits: u32 = std::env::var("BITS").map_or(16, |v| v.parse().unwrap());
    let mut exact = ExactEngine::new(compiler::compile_model(&text)?, Weights::open(dir)?, bits)?;
    // A few tokens of text, then the picture's largest vector.
    let all = std::env::var("ALL").is_ok();
    let steps: Vec<Option<u32>> = if all { [Some(100_000), Some(2054)].into_iter().chain(std::iter::repeat_n(None, vectors.len())).collect() } else { vec![Some(100_000), Some(2054), Some(418), None, None] };
    for (position, step) in steps.iter().enumerate() {
        match step {
            Some(token) => {
                float.step(*token, true)?;
                exact.step(*token, true)?;
            }
            None => {
                let v = &vectors[if all { position - 2 } else if position == 3 { largest } else { 0 }];
                float.step_embedding(v, [position as u64; 3], true)?;
                exact.step_embedding(v, [position as u64; 3], true)?;
            }
        }
        if all && position % 32 != 31 {
            continue;
        }
        println!("--- step {position} ({})", if step.is_some() { "text" } else { "image" });
        let mut reported = 0;
        for inst in &program.ops {
            let text = format!("{:?}", inst.op);
            let Some(start) = text.find("dst: ") else { continue };
            let buffer: usize = text[start + 5..].chars().take_while(char::is_ascii_digit).collect::<String>().parse()?;
            let (a, b) = (float.buffer(buffer), exact.buffer(buffer));
            let norm = a.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt();
            let error = a.iter().zip(&b).map(|(x, y)| (*x as f64 - *y as f64).powi(2)).sum::<f64>().sqrt();
            if error > 0.1 * norm.max(1e-9) && reported < 8 {
                reported += 1;
                let other = b.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt();
                println!("{} layer {:?} buffer {buffer} norm {norm:.3}/{other:.3} peak {:.3}/{:.3} error {error:.3}: {}", inst.name, inst.layer, peak(a), peak(&b), &text[..text.len().min(90)]);
            }
        }
    }
    Ok(())
}
