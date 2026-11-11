//! Draws a picture from a description with a model that makes images, such as Janus-Pro.
//!
//! `cargo run --release -p sim --example draw -- <carpeta> "<descripción>" salida.bmp [temperatura] [semilla]`

use sim::janus::Janus;
use std::path::Path;
use tokenizers::Tokenizer;

/// An uncompressed 24-bit BMP, which every image viewer opens.
fn bmp(rgb: &[u8], side: usize) -> Vec<u8> {
    let row = (side * 3).next_multiple_of(4);
    let size = 54 + row * side;
    let mut out = Vec::with_capacity(size);
    out.extend_from_slice(b"BM");
    out.extend_from_slice(&(size as u32).to_le_bytes());
    out.extend_from_slice(&[0; 4]);
    out.extend_from_slice(&54u32.to_le_bytes());
    out.extend_from_slice(&40u32.to_le_bytes());
    out.extend_from_slice(&(side as i32).to_le_bytes());
    // A negative height stores the rows from the top.
    out.extend_from_slice(&(-(side as i32)).to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&24u16.to_le_bytes());
    out.extend_from_slice(&[0; 24]);
    for y in 0..side {
        for x in 0..side {
            let p = &rgb[(y * side + x) * 3..][..3];
            out.extend_from_slice(&[p[2], p[1], p[0]]);
        }
        out.resize(out.len() + row - side * 3, 0);
    }
    out
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let dir = Path::new(&args[1]);
    let temperature: f32 = args.get(4).map_or(1.0, |t| t.parse().unwrap());
    let seed: u64 = args.get(5).map_or(42, |s| s.parse().unwrap());
    let janus = Janus::open(dir)?.ok_or("este modelo no genera imágenes")?;
    let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| e.to_string())?;
    let text = format!("<|User|>: {}\n\n<|Assistant|>:", args[2]);
    let mut ids = vec![janus.config.bos];
    ids.extend(tokenizer.encode(text.as_str(), false).map_err(|e| e.to_string())?.get_ids());
    ids.push(janus.config.begin_image);
    let started = std::time::Instant::now();
    let tokens = janus.draw(&ids, temperature, seed, |done, total| {
        if done % 48 == 0 {
            eprintln!("{done} de {total} tokens de imagen, {:.0} s", started.elapsed().as_secs_f64());
        }
    })?;
    let drawn = std::time::Instant::now();
    let rgb = janus.decode(&tokens)?;
    eprintln!("decodificada en {:.0} s", drawn.elapsed().as_secs_f64());
    std::fs::write(&args[3], bmp(&rgb, janus.config.side()))?;
    println!("{}", args[3]);
    Ok(())
}
