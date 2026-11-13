//! Where the chip's retouch loses colour: the same retouch with several seeds, three ways.
//!
//! - `flotante`: Show-o and MaskGIT in floating point, as the app runs them off the chip.
//! - `maskgit-entero`: Show-o in floating point, MaskGIT as the firmware runs it, in integers
//!   (`maskgit-rt`), the scores rounded to the die's 16-bit mantissas.
//! - `chip`: MaskGIT in integers and Show-o on the exact machine, weights affine of 8 bits.
//!
//! For each, the colour inside the retouched square: its mean, how yellow it is
//! (`min(R, G) − B`) and how saturated.
//!
//! `cargo run --release -p sim --example showo_fill_compare -- <carpeta de Show-o> <carpeta de salida> [semillas] [semillas en el chip] [pasos]`
//!
//! Seeds go comma-separated, odd: MaskGIT's generators make every seed odd, so 2 is 3. Those of
//! the chip take about 6 minutes each.

use netlist::format::{WeightCoding, WeightFormat};
use sim::lower::{ENTRY_TOKEN, ENTRY_VISION};
use sim::showo::{Magvit, ShowO, ShowOConfig};
use sim::showo_chip::{fill_on, schedule, Fill, FillDie, FillShape, ShowOChip};
use std::path::Path;

const GUIDANCE: f32 = 1.75;
const BITS: u32 = 16;

/// Show-o in floating point behind the die's routines: it notes the text and the image the
/// routines are given, runs the whole pass when the first place's scores are asked for, and
/// hands each place's scores rounded to mantissas of `BITS` bits.
struct FloatDie<'a> {
    model: &'a ShowO,
    text: &'a [u32],
    text_ids: Vec<u32>,
    codebook: usize,
    score_routine: usize,
    ids: Vec<u32>,
    image: Vec<u32>,
    with: Vec<f32>,
    current: Vec<f32>,
    row: usize,
}

impl FillDie for FloatDie<'_> {
    fn run(&mut self, routine: usize, token: u32, position: u32) {
        match routine {
            ENTRY_TOKEN => {
                if position == 0 {
                    self.ids.clear();
                    self.image.clear();
                }
                self.ids.push(token);
            }
            ENTRY_VISION => self.image.push(token),
            r if r == self.score_routine => {
                self.row = token as usize;
                if token == 1 {
                    let text: &[u32] = if self.ids == self.text_ids { self.text } else { &[] };
                    self.current = self.model.scores(text, &self.image[1..self.image.len() - 1]);
                }
            }
            r if r == self.score_routine + 1 && token == 1 => self.with = self.current.clone(),
            r if r == self.score_routine + 2 && token == 1 => {
                for (s, w) in self.current.iter_mut().zip(&self.with) {
                    *s = (1.0 + GUIDANCE) * w - GUIDANCE * *s;
                }
            }
            _ => {}
        }
    }

    fn rotate(&mut self, _slot: u32, _position: u32) {}

    fn scores(&mut self) -> (i32, Vec<i32>) {
        let row = &self.current[(self.row - 1) * self.codebook..][..self.codebook];
        let top = row.iter().fold(0f32, |m, s| m.max(s.abs())).max(f32::MIN_POSITIVE);
        let exponent = top.log2().floor() as i32 + 1;
        let unit = (exponent - BITS as i32 + 1) as f32;
        (exponent, row.iter().map(|s| (s / unit.exp2()).round() as i32).collect())
    }
}

/// The colour of the retouched square: mean R, G, B; mean `min(R, G) − B`; mean saturation.
fn colour(picture: &[u8], side: usize, inside: &dyn Fn(usize, usize) -> bool) -> [f64; 5] {
    let (mut sum, mut n) = ([0f64; 5], 0f64);
    for y in 0..side {
        for x in 0..side {
            if !inside(x, y) {
                continue;
            }
            let p = &picture[(y * side + x) * 3..][..3];
            let (r, g, b) = (p[0] as f64, p[1] as f64, p[2] as f64);
            let (hi, lo) = (r.max(g).max(b), r.min(g).min(b));
            for (s, v) in sum.iter_mut().zip([r, g, b, r.min(g) - b, if hi > 0.0 { (hi - lo) / hi } else { 0.0 }]) {
                *s += v;
            }
            n += 1.0;
        }
    }
    sum.map(|s| s / n)
}

fn write_bmp(path: &Path, picture: &[u8], side: usize) -> Result<(), String> {
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
    std::fs::write(path, bmp).map_err(|e| e.to_string())
}

fn seeds(arg: Option<&String>, default: &[u32]) -> Vec<u32> {
    arg.map_or(default.to_vec(), |s| s.split(',').filter(|s| !s.is_empty()).map(|s| s.parse().unwrap()).collect())
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let dir = Path::new(&args[1]);
    let out = Path::new(&args[2]);
    std::fs::create_dir_all(out).map_err(|e| e.to_string())?;
    let float_seeds = seeds(args.get(3), &[1, 3, 5, 7, 9, 11, 13, 15]);
    let chip_seeds = seeds(args.get(4), &[]);
    let steps: usize = args.get(5).map_or(4, |s| s.parse().unwrap());

    let model = ShowO::open(dir)?.ok_or("no es un Show-o")?;
    let config: ShowOConfig = model.config.clone();
    let tokenizer = tokenizers::Tokenizer::from_file(dir.join("phi/tokenizer.json")).map_err(|e| e.to_string())?;
    let text = tokenizer.encode("a yellow lemon", false).map_err(|e| e.to_string())?.get_ids().to_vec();
    let (ids, first) = config.text_prefix(&text);
    let (plain, plain_first) = config.text_prefix(&[]);
    let side = config.side();
    let rgb: Vec<u8> = (0..side * side)
        .flat_map(|i| {
            let (x, y) = ((i % side) as f32 - 128.0, (i / side) as f32 - 128.0);
            if x * x + y * y < 80.0 * 80.0 { [220u8, 30, 30] } else { [255, 255, 255] }
        })
        .collect();
    let inside = |x: usize, y: usize| (48..208).contains(&x) && (48..208).contains(&y);
    let region: Vec<bool> = (0..side * side).map(|i| inside(i % side, i / side)).collect();
    let magvit = Magvit::open(&dir.join("magvitv2"))?;
    let codes = config.masked_codes(&magvit, &rgb, side, side, &region)?;
    let plan = schedule(codes.len(), steps);
    let fill = |seed: u32| Fill { text: (&ids, first[0] as u32), plain: Some((&plain, plain_first[0] as u32)), schedule: &plan, seed };

    let mut rows: Vec<(String, u32, [f64; 5], f64)> = Vec::new();
    let mut report = |way: &str, seed: u32, filled: &[u32], seconds: f64| -> Result<(), String> {
        let picture = magvit.decode(filled)?;
        write_bmp(&out.join(format!("{way}-{seed}.bmp")), &picture, side)?;
        let c = colour(&picture, side, &inside);
        eprintln!("{way:15} semilla {seed:3}: RGB {:5.1} {:5.1} {:5.1}  amarillo {:6.1}  saturación {:.3}  ({seconds:.0} s)", c[0], c[1], c[2], c[3], c[4]);
        rows.push((way.to_string(), seed, c, seconds));
        Ok(())
    };

    for &seed in &float_seeds {
        let started = std::time::Instant::now();
        let filled = model.fill(&codes, &text, steps, GUIDANCE, seed as u64, |_, _| Ok(()))?;
        report("flotante", seed, &filled, started.elapsed().as_secs_f64())?;

        let started = std::time::Instant::now();
        // The model runs whole when the scores are asked for: one layer of routines will do.
        let score_routine = ENTRY_VISION + 1 + 2;
        let mut die = FloatDie { model: &model, text: &text, text_ids: ids.clone(), codebook: config.codebook as usize, score_routine, ids: Vec::new(), image: Vec::new(), with: Vec::new(), current: Vec::new(), row: 0 };
        let shape = FillShape { rows: codes.len() as u32 + 2, layers: 1, guided: true, bits: BITS };
        let filled = fill_on(&mut die, shape, &config, &codes, &fill(seed));
        report("maskgit-entero", seed, &filled, started.elapsed().as_secs_f64())?;
    }

    if !chip_seeds.is_empty() {
        let format = WeightFormat { coding: WeightCoding::Affine, bits: 8, exponent_bits: 0, bias: true, group_size: 64, fine_scale: false, keep_source: false };
        let mut chip = ShowOChip::exact_guided(dir, BITS, Some(format), GUIDANCE)?;
        for &seed in &chip_seeds {
            let started = std::time::Instant::now();
            chip.reset();
            let shape = chip.fill_shape();
            let filled = fill_on(&mut chip, shape, &config, &codes, &fill(seed));
            report("chip", seed, &filled, started.elapsed().as_secs_f64())?;
        }
    }

    println!("forma,semilla,R,G,B,amarillo,saturacion,segundos");
    for (way, seed, c, seconds) in &rows {
        println!("{way},{seed},{:.1},{:.1},{:.1},{:.1},{:.3},{seconds:.0}", c[0], c[1], c[2], c[3], c[4]);
    }
    for way in ["flotante", "maskgit-entero", "chip"] {
        let these: Vec<_> = rows.iter().filter(|r| r.0 == way).collect();
        if these.is_empty() {
            continue;
        }
        let mean = |i: usize| these.iter().map(|r| r.2[i]).sum::<f64>() / these.len() as f64;
        eprintln!("media {way:15} ({} semillas): RGB {:5.1} {:5.1} {:5.1}  amarillo {:6.1}  saturación {:.3}", these.len(), mean(0), mean(1), mean(2), mean(3), mean(4));
    }
    Ok(())
}
