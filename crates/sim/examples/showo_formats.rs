//! How much Show-o loses with its weights in the design's formats, in the die's arithmetic:
//! one pass over a picture with its middle masked, against floating point.
//!
//! `cargo run --release -p sim --example showo_formats -- <carpeta de Show-o>`

use netlist::format::{WeightCoding, WeightFormat};
use sim::showo::ShowO;
use sim::showo_chip::ShowOChip;
use std::path::Path;

/// A row of the table: its name and how to open the chip for it.
type Run<'a> = (&'static str, Box<dyn Fn() -> Result<ShowOChip, String> + 'a>);

fn main() -> Result<(), String> {
    let dir = std::env::args().nth(1).ok_or("falta la carpeta de Show-o")?;
    let dir = Path::new(&dir);
    let model = ShowO::open(dir)?.ok_or("la carpeta no tiene Show-o")?;
    let tokenizer = tokenizers::Tokenizer::from_file(dir.join("phi/tokenizer.json")).map_err(|e| e.to_string())?;
    let text = tokenizer.encode("a green apple on a wooden table", false).map_err(|e| e.to_string())?.get_ids().to_vec();
    let side = model.config.side();
    let planes: Vec<f32> = (0..3)
        .flat_map(|c| (0..side * side).map(move |i| (c, i)))
        .map(|(c, i)| {
            let (x, y) = ((i % side) as f32 - side as f32 / 2.0, (i / side) as f32 - side as f32 / 2.0);
            let value = if x * x + y * y < (side as f32 * 0.3).powi(2) { [220.0, 30.0, 30.0][c] } else { 255.0 };
            value / 127.5 - 1.0
        })
        .collect();
    let codes = model.tokenizer()?.encode(&planes)?;
    let grid = model.config.grid();
    let middle = |i: usize| (grid / 4..3 * grid / 4).contains(&(i % grid)) && (grid / 4..3 * grid / 4).contains(&(i / grid));
    let image: Vec<u32> = codes.iter().enumerate().map(|(i, code)| if middle(i) { model.config.mask_token } else { model.config.first_code() + code }).collect();
    let float = model.scores(&text, &image);
    let (ids, positions) = model.text_prefix(&text);
    let codebook = model.config.codebook as usize;
    let masked: Vec<usize> = (0..image.len()).filter(|i| middle(*i)).collect();
    let best = |row: &[f32]| row.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
    let softmax = |row: &[f32]| {
        let top = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp: Vec<f64> = row.iter().map(|s| ((s - top) as f64).exp()).collect();
        let total: f64 = exp.iter().sum();
        exp.into_iter().map(|e| e / total).collect::<Vec<f64>>()
    };
    // What a sampler sees: the divergence of the chip's distribution from the reference's,
    // and how likely the reference finds the chip's best code, against its own best.
    let report = |name: &str, scores: &[f32], seconds: f64| {
        let largest = float.iter().fold(0f32, |m, v| m.max(v.abs()));
        let worst = scores.iter().zip(&float).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        let agree = masked.iter().filter(|i| best(&scores[*i * codebook..][..codebook]) == best(&float[*i * codebook..][..codebook])).count();
        let (mut divergence, mut likelihood) = (0f64, 0f64);
        for i in &masked {
            let (p, q) = (softmax(&float[*i * codebook..][..codebook]), softmax(&scores[*i * codebook..][..codebook]));
            divergence += p.iter().zip(&q).filter(|(p, _)| **p > 0.0).map(|(p, q)| p * (p / q.max(1e-30)).ln()).sum::<f64>();
            likelihood += p[best(&scores[*i * codebook..][..codebook])] / p.iter().copied().fold(0.0, f64::max);
        }
        let n = masked.len() as f64;
        println!("| {name} | {:.2} % | {agree} de {} | {:.4} | {:.1} % | {seconds:.0} |", 100.0 * worst / largest, masked.len(), divergence / n, 100.0 * likelihood / n);
    };
    println!("| Pesos | Diferencia mayor | Mejor código igual | KL media (nats) | Probabilidad del mejor del chip, frente al mejor | Segundos por pasada |");
    println!("|---|---|---|---|---|---|");
    let affine = |bits, group_size| WeightFormat { coding: WeightCoding::Affine, bits, exponent_bits: 0, bias: true, group_size, fine_scale: false, keep_source: false };
    let runs: Vec<Run> = vec![
        ("enteros de 16 bits", Box::new(|| ShowOChip::exact(dir, 16))),
        ("afín de 8 bits, grupos de 64", Box::new(|| ShowOChip::with_format(dir, 16, affine(8, 64)))),
        ("afín de 4 bits, grupos de 32", Box::new(|| ShowOChip::with_format(dir, 16, affine(4, 32)))),
    ];
    for (name, open) in runs {
        let started = std::time::Instant::now();
        let mut chip = open()?;
        let scores = chip.scores(&ids, &positions, &image)?;
        report(name, &scores, started.elapsed().as_secs_f64());
    }
    Ok(())
}
