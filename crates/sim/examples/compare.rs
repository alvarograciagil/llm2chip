//! Finds the first buffer where exact integers drift away from floating point.
//! `cargo run --release -p sim --example compare -- <carpeta> [tokens]`

use compiler::compile_qwen3_5;
use sim::{Engine, ExactEngine, Numeric, Weights};
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let dir = Path::new(&args[1]);
    let tokens: usize = args.get(2).map_or(3, |n| n.parse().unwrap());
    let config = std::fs::read_to_string(dir.join("config.json"))?;
    let program = compile_qwen3_5(&config)?;
    let mut float = Engine::new(compile_qwen3_5(&config)?, Weights::open(dir)?, Numeric::Float)?;
    let bits: u32 = std::env::var("BITS").map_or(16, |v| v.parse().unwrap());
    let mut exact = ExactEngine::new(compile_qwen3_5(&config)?, Weights::open(dir)?, bits)?;
    for position in 0..tokens {
        let token = [760u32, 6511, 314, 9338, 369, 11751, 13, 1000][position % 8];
        float.step(token, true)?;
        exact.step(token, true)?;
        println!("--- token {position}");
        let mut reported = 0;
        for inst in &program.ops {
            let text = format!("{:?}", inst.op);
            let Some(start) = text.find("dst: ") else { continue };
            let buffer: usize = text[start + 5..].chars().take_while(char::is_ascii_digit).collect::<String>().parse()?;
            let (a, b) = (float.buffer(buffer), exact.buffer(buffer));
            let norm = a.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt();
            let error = a.iter().zip(&b).map(|(x, y)| (*x as f64 - *y as f64).powi(2)).sum::<f64>().sqrt();
            let limit: f64 = std::env::var("LIMIT").map_or(0.1, |v| v.parse().unwrap());
            if error > limit * norm.max(1e-9) && reported < 16 {
                reported += 1;
                let other = b.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt();
                let peak = |v: &[f32]| v.iter().fold(0f32, |m, x| m.max(x.abs()));
                println!("{} capa {:?} búfer {buffer} norma {norm:.3}/{other:.3} pico {:.3}/{:.3} error {error:.3}: {}", inst.name, inst.layer, peak(a), peak(&b), &text[..text.len().min(90)]);
            }
        }
    }
    Ok(())
}
