//! Show-o retouches a picture as the chip does it: MaskGIT's steps in integers
//! (`maskgit-rt`), the guidance in the die, every routine on the die as the chip addresses
//! it, the cells keeping the weights affine of 8 bits; MAGVIT-v2 on the host.
//!
//! `cargo run --release -p sim --example showo_fill_chip -- <carpeta de Show-o> salida.bmp [pasos] [semilla]`

use netlist::format::{WeightCoding, WeightFormat};
use sim::showo::Magvit;
use sim::showo_chip::ShowOAddressed;
use std::path::Path;

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let dir = Path::new(&args[1]);
    let steps: usize = args.get(3).map_or(4, |s| s.parse().unwrap());
    let seed: u32 = args.get(4).map_or(7, |s| s.parse().unwrap());
    let started = std::time::Instant::now();
    let format = WeightFormat { coding: WeightCoding::Affine, bits: 8, exponent_bits: 0, bias: true, group_size: 64, fine_scale: false, keep_source: false };
    let mut chip = ShowOAddressed::open(dir, format, Some(1.75), 4096)?;
    let config = chip.config.clone();
    let tokenizer = tokenizers::Tokenizer::from_file(dir.join("phi/tokenizer.json")).map_err(|e| e.to_string())?;
    let text = tokenizer.encode("a yellow lemon", false).map_err(|e| e.to_string())?.get_ids().to_vec();
    // A red disc on white, its middle masked.
    let side = config.side();
    let rgb: Vec<u8> = (0..side * side)
        .flat_map(|i| {
            let (x, y) = ((i % side) as f32 - 128.0, (i / side) as f32 - 128.0);
            if x * x + y * y < 80.0 * 80.0 { [220u8, 30, 30] } else { [255, 255, 255] }
        })
        .collect();
    let region: Vec<bool> = (0..side * side).map(|i| (48..208).contains(&(i % side)) && (48..208).contains(&(i / side))).collect();
    let magvit = Magvit::open(&dir.join("magvitv2"))?;
    let codes = config.masked_codes(&magvit, &rgb, side, side, &region)?;
    eprintln!("preparado en {:.0} s", started.elapsed().as_secs_f64());
    let filled = chip.fill(&codes, &text, steps, seed as u64, |step, steps| {
        eprintln!("paso {step} de {steps}: {:.0} s", started.elapsed().as_secs_f64());
        Ok(())
    })?;
    eprintln!("rellenado en {:.0} s", started.elapsed().as_secs_f64());
    let picture = magvit.decode(&filled)?;
    let row = (side * 3).next_multiple_of(4);
    let mut bmp = b"BM".to_vec();
    bmp.extend(((54 + row * side) as u32).to_le_bytes());
    bmp.extend([0; 4]);
    bmp.extend(54u32.to_le_bytes());
    bmp.extend(40u32.to_le_bytes());
    bmp.extend((side as i32).to_le_bytes());
    bmp.extend((-(side as i32)).to_le_bytes());
    bmp.extend(1u16.to_le_bytes());
    bmp.extend(24u16.to_le_bytes());
    bmp.extend([0; 24]);
    for y in 0..side {
        for x in 0..side {
            let p = &picture[(y * side + x) * 3..][..3];
            bmp.extend([p[2], p[1], p[0]]);
        }
        bmp.resize(bmp.len() + row - side * 3, 0);
    }
    std::fs::write(&args[2], bmp).map_err(|e| e.to_string())?;
    eprintln!("{}: {:.0} s", args[2], started.elapsed().as_secs_f64());
    Ok(())
}
