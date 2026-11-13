//! Show-o, in floating point: retouches a region of a picture as a text describes, draws a
//! picture from a text, or checks its image tokenizer by encoding and decoding a picture.
//!
//! `cargo run --release -p sim --example inpaint -- <carpeta> retocar <imagen> x0,y0,x1,y1 "<texto>" salida.bmp [pasos] [guía] [semilla]`
//! `cargo run --release -p sim --example inpaint -- <carpeta> dibujar "<texto>" salida.bmp [pasos] [guía] [semilla]`
//! `cargo run --release -p sim --example inpaint -- <carpeta> reconstruir <imagen> salida.bmp`
//!
//! The picture is a 24-bit BMP, a binary PPM (P6), or `figura` for a red disc on white drawn here. The
//! folder holds Show-o, MAGVIT-v2 in `magvitv2` and Phi-1.5's tokenizer in `phi`.

use sim::showo::ShowO;
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

/// A binary PPM's pixels, or a red disc on white.
fn picture(source: &str) -> Result<(Vec<u8>, usize, usize), String> {
    if source == "figura" {
        let side = 256;
        let rgb = (0..side * side)
            .flat_map(|i| {
                let (x, y) = ((i % side) as f32 - 128.0, (i / side) as f32 - 128.0);
                if x * x + y * y < 80.0 * 80.0 { [220, 30, 30] } else { [255, 255, 255] }
            })
            .collect();
        return Ok((rgb, side, side));
    }
    let bytes = std::fs::read(source).map_err(|e| e.to_string())?;
    if bytes.starts_with(b"BM") {
        // An uncompressed 24-bit BMP, its rows from the bottom unless its height is negative.
        let word = |at: usize| i32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]]);
        let (offset, width, height) = (word(10) as usize, word(18) as usize, word(22));
        let rows = height.unsigned_abs() as usize;
        let stride = (width * 3).next_multiple_of(4);
        let rgb = (0..rows)
            .flat_map(|y| {
                let row = if height < 0 { y } else { rows - 1 - y };
                (0..width).flat_map(move |x| (0..3).rev().map(move |c| (row, x, c)))
            })
            .map(|(row, x, c)| bytes[offset + row * stride + x * 3 + c])
            .collect();
        return Ok((rgb, width, rows));
    }
    // "P6", width, height, the largest value, then the pixels.
    let mut fields = Vec::new();
    let mut at = 0;
    while fields.len() < 4 {
        while bytes[at].is_ascii_whitespace() {
            at += 1;
        }
        if bytes[at] == b'#' {
            while bytes[at] != b'\n' {
                at += 1;
            }
            continue;
        }
        let start = at;
        while !bytes[at].is_ascii_whitespace() {
            at += 1;
        }
        fields.push(String::from_utf8_lossy(&bytes[start..at]).to_string());
    }
    if fields[0] != "P6" || fields[3] != "255" {
        return Err("solo PPM binario (P6) de 8 bits".into());
    }
    let (width, height): (usize, usize) = (fields[1].parse().unwrap(), fields[2].parse().unwrap());
    Ok((bytes[at + 1..at + 1 + width * height * 3].to_vec(), width, height))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let dir = Path::new(&args[1]);
    let started = std::time::Instant::now();
    let model = ShowO::open(dir)?.ok_or("la carpeta no tiene Show-o")?;
    eprintln!("Show-o abierto en {:.1} s", started.elapsed().as_secs_f64());
    let side = model.config.side();
    let tokenizer = Tokenizer::from_file(dir.join("phi/tokenizer.json")).map_err(|e| e.to_string())?;
    let text = |t: &str| tokenizer.encode(t, false).map(|e| e.get_ids().to_vec()).map_err(|e| e.to_string());
    let number = |i: usize, default: f64| args.get(i).map_or(default, |v| v.parse().unwrap());
    let progress = |step: usize, steps: usize| {
        eprintln!("paso {step} de {steps}, {:.0} s", started.elapsed().as_secs_f64());
        Ok(())
    };
    let (rgb, out) = match args[2].as_str() {
        "reconstruir" => {
            let (rgb, width, height) = picture(&args[3])?;
            if (width, height) != (side, side) {
                return Err(format!("para medir, una imagen de {side} × {side}").into());
            }
            let planes: Vec<f32> = (0..3).flat_map(|c| (0..side * side).map(move |i| (c, i))).map(|(c, i)| rgb[i * 3 + c] as f32 / 127.5 - 1.0).collect();
            let codes = model.tokenizer.encode(&planes)?;
            let back = model.tokenizer.decode(&codes)?;
            let error: f64 = rgb.iter().zip(&back).map(|(a, b)| (*a as f64 - *b as f64).powi(2)).sum::<f64>() / rgb.len() as f64;
            eprintln!("{} códigos distintos de {}, PSNR {:.1} dB", codes.iter().collect::<std::collections::HashSet<_>>().len(), codes.len(), 10.0 * (255.0f64 * 255.0 / error).log10());
            (back, &args[4])
        }
        "dibujar" => (model.draw(&text(&args[3])?, number(5, 16.0) as usize, number(6, 1.75) as f32, number(7, 1.0) as u64, progress)?, &args[4]),
        "retocar" => {
            let (rgb, width, height) = picture(&args[3])?;
            let corners: Vec<usize> = args[4].split(',').map(|v| v.parse().unwrap()).collect();
            let region: Vec<bool> = (0..width * height).map(|i| (corners[0]..corners[2]).contains(&(i % width)) && (corners[1]..corners[3]).contains(&(i / width))).collect();
            (model.inpaint(&rgb, width, height, &region, &text(&args[5])?, number(7, 16.0) as usize, number(8, 1.75) as f32, number(9, 1.0) as u64, progress)?, &args[6])
        }
        other => return Err(format!("modo desconocido «{other}»: retocar, dibujar o reconstruir").into()),
    };
    std::fs::write(out, bmp(&rgb, side))?;
    eprintln!("{out}: {side} × {side}, {:.0} s", started.elapsed().as_secs_f64());
    Ok(())
}
