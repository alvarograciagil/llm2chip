//! The real image encoder in the die's arithmetic against floating point.
//!
//! `cargo run --release -p sim --example see_exact -- <carpeta> [bits]`

use sim::vision::{patches, VisionTower};
use sim::vision_chip::ExactVision;
use sim::Weights;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let dir = Path::new(&args[1]);
    let bits: u32 = args.get(2).map_or(16, |b| b.parse().unwrap());
    let mut tower = VisionTower::open(dir)?.ok_or("sin codificador")?;
    let depth: Option<usize> = args.get(3).map(|d| d.parse().unwrap());
    if let Some(d) = depth { tower.config.depth = d; }
    let side = 256;
    let rgb: Vec<u8> = (0..side * side).flat_map(|i| { let (x, y) = (i % side, i / side); let d = (x as f32 - 128.0).hypot(y as f32 - 128.0); if d < 80.0 { [220, 30, 30] } else { [255, 255, 255] } }).collect();
    let image = patches(&tower.config, &rgb, side, side)?;
    let count = image.rows * image.cols;
    let started = std::time::Instant::now();
    let float = tower.encode(&image)?;
    eprintln!("coma flotante: {count} parches, {:.1} s", started.elapsed().as_secs_f64());
    let mut config: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json"))?)?;
    if let Some(d) = depth { config["vision_config"]["depth"] = d.into(); }
    let config = config.to_string();
    let mut exact = ExactVision::exact(&config, Weights::open_with_vision(dir)?, count, bits)?;
    let started = std::time::Instant::now();
    let got = exact.encode(&tower, &image)?;
    eprintln!("aritmética del die: {:.1} s", started.elapsed().as_secs_f64());
    let (mut error, mut norm) = (0f64, 0f64);
    for (a, b) in got.iter().flatten().zip(float.iter().flatten()) {
        error += ((a - b) as f64).powi(2);
        norm += (*b as f64).powi(2);
    }
    let top = float.iter().flatten().fold(0f32, |m, v| m.max(v.abs()));
    println!("error relativo {:.4} (máximo |v| {top:.1})", (error / norm).sqrt());
    Ok(())
}
