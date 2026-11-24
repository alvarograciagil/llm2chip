//! DAC's decoder in floating point against Transformers' on the codes `tools/golden/speak.py`
//! wrote, and in the die's arithmetic with `bits`-bit vectors:
//! `cargo run --release -p sim --example dac_check -- DAC_DIR golden.json [bits] [out.wav]`.
use sim::dac::Dac;
use std::path::Path;

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let dac = Dac::open(Path::new(&args[0]))?;
    let golden: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&args[1]).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    let codes: Vec<Vec<u32>> = serde_json::from_value(golden["codes"].clone()).map_err(|e| e.to_string())?;
    let audio: Vec<f32> = serde_json::from_value(golden["audio"].clone()).map_err(|e| e.to_string())?;
    let started = std::time::Instant::now();
    let ours = dac.decode(&codes)?;
    let largest = audio.iter().fold(0f32, |m, v| m.max(v.abs()));
    let worst = ours.iter().zip(&audio).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
    println!("{} frames -> {} samples (Transformers {}), {:.1} s; largest difference {:.5} of a peak of {:.3}", codes[0].len(), ours.len(), audio.len(), started.elapsed().as_secs_f32(), worst, largest);
    println!("largest |αx| {:.1}", dac.snake_reach(&codes)?);
    // Only the sine's argument as the die takes it, everything else in floating point.
    let mut phased = Dac::open(Path::new(&args[0]))?;
    for bits in [16, 20, 24] {
        phased.sine_bits = Some(bits);
        let sound = phased.decode(&codes)?;
        let error: f32 = sound.iter().zip(&ours).map(|(a, b)| (a - b) * (a - b)).sum::<f32>() / ours.iter().map(|v| v * v).sum::<f32>();
        println!("only the sine's argument with {bits}-bit vectors: {:.1} dB below the signal", -10.0 * error.log10());
    }
    // Every row as the die holds it, everything else in floating point.
    let mut held = Dac::open(Path::new(&args[0]))?;
    for bits in [16, 20, 24] {
        held.row_bits = Some(bits);
        let sound = held.decode(&codes)?;
        let error: f32 = sound.iter().zip(&ours).map(|(a, b)| (a - b) * (a - b)).sum::<f32>() / ours.iter().map(|v| v * v).sum::<f32>();
        println!("only the rows held as the die holds them, {bits}-bit vectors: {:.1} dB below the signal", -10.0 * error.log10());
    }
    let chunked = dac.decode_chunked(&codes, 16)?;
    let apart = chunked.iter().zip(&ours).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
    println!("a window at a time ({} frames each, 16 kept): {} samples, largest difference from the whole {apart:.2e}", dac.config.window_frames(codes[0].len(), 16), chunked.len());
    if let Some(bits) = args.get(2).and_then(|b| b.parse().ok()) {
        let started = std::time::Instant::now();
        let accumulator = std::env::var("ACCUMULATOR").ok().and_then(|a| a.parse().ok()).unwrap_or(2 * bits + 8);
        let mut chip = sim::dac_chip::DacChip::exact_summing(Path::new(&args[0]), codes[0].len(), bits, accumulator)?;
        let (die, _) = chip.decode(&codes)?;
        let worst = die.iter().zip(&ours).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        let error: f32 = die.iter().zip(&ours).map(|(a, b)| (a - b) * (a - b)).sum::<f32>() / ours.iter().map(|v| v * v).sum::<f32>();
        println!("die's arithmetic, {bits}-bit vectors ({:.1} s): largest difference {worst:.4}, {:.1} dB below the signal", started.elapsed().as_secs_f32(), -10.0 * error.log10());
        if let Some(out) = args.get(3) {
            sim::audio::Clip { rate: dac.config.sampling_rate, samples: die }.write(Path::new(out))?;
        }
    }
    Ok(())
}
