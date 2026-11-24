//! A chip that listens, thinks and speaks, sized as a system of four stages, each on a die
//! of its own: Moonshine, a language model, OuteTTS and DAC. Every stage's cycles come from
//! its own microprogram and its silicon from the same counts as the optimiser's, in N6.
//!
//!     cargo run --release -p sim --example voice_system -- MOONSHINE LLM OUTETTS DAC [SECONDS] [WORDS]
//!
//! `SECONDS` is the question's length (3) and `WORDS` how long the answer's first sentence
//! is in the language model's tokens (12). The time to the first sound counts the question
//! heard whole, the answer's first sentence, OuteTTS writing half a second of speech after
//! its prompt, and DAC decoding its first window. `LISTEN` and `SPEAK` set the recogniser's and the
//! decoder's dies as `UNITS,LANES,CELLS` in powers of two (`0,5,11` and `0,6,11`).

use netlist::format::{WeightCoding, WeightFormat};
use sim::dac::DacConfig;
use sim::dac_chip::DacChip;
use sim::moonshine::MoonshineConfig;
use sim::moonshine_chip::MoonshineChip;
use sim::sizing::{die_of, measure, reference_die, routines_die, Costs, Sharing, Target, N6};
use sim::system::{Stage, System};
use sim::Weights;
use std::path::Path;

const CLOCK: f64 = 1e9;

/// 8-bit codes with a scale and a bias per group of `group` columns: 64 is the chip's
/// usual; Moonshine's widths (288) need 32, and DAC's codebooks, 8 values wide, need 8.
/// A die's shape, `(log2 units, log2 lanes, log2 cells)`, from `UNITS,LANES,CELLS` in the
/// environment variable `name` or the default.
fn shape_of(name: &str, default: (u32, u32, u32)) -> (u32, u32, u32) {
    let given: Option<Vec<u32>> = std::env::var(name).ok().map(|v| v.split(',').filter_map(|x| x.parse().ok()).collect());
    match given.as_deref() {
        Some([units, lanes, cells]) => (*units, *lanes, *cells),
        _ => default,
    }
}

fn weights(dir: &Path, group: u32) -> Result<Weights, String> {
    let format = WeightFormat { coding: WeightCoding::Affine, bits: 8, exponent_bits: 0, bias: true, group_size: group, fine_scale: false, keep_source: false };
    Ok(Weights::open(dir)?.with_format(Some(format)))
}

/// A language model on `chips` dies of `shape`, a prompt of `prompt` tokens before its
/// first: its stage.
fn language_model(name: &str, dir: &Path, costs: &Costs, chips: u32, shape: (u32, u32, u32), prompt: u32) -> Result<Stage, String> {
    let program = compiler::compile_model(&std::fs::read_to_string(dir.join("config.json")).map_err(|e| e.to_string())?)?;
    let target = Target { tokens_per_second: 0.0, prompt_seconds: f64::INFINITY, prompt, chips: vec![chips], clock_hz: CLOCK, process: N6, context: 1024, lane_bits: shape.1..=shape.1, cluster_bits: shape.0..=shape.0, cell_bits: shape.2..=shape.2, sharing: vec![Sharing::Layers] };
    let candidate = measure(&program, &weights(dir, 64)?, reference_die(8), costs, &target, Sharing::Layers, (chips, shape.0, shape.1, shape.2))?;
    Ok(Stage::from_candidate(name, &candidate))
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [listen, think, tts, dac, rest @ ..] = args.as_slice() else {
        return Err("usage: voice_system MOONSHINE LLM OUTETTS DAC [SECONDS] [WORDS]".into());
    };
    let (listen, think, tts, dac) = (Path::new(listen), Path::new(think), Path::new(tts), Path::new(dac));
    let seconds: f64 = rest.first().and_then(|s| s.parse().ok()).unwrap_or(3.0);
    let words: u32 = rest.get(1).and_then(|s| s.parse().ok()).unwrap_or(12);
    let costs = Costs::measure(reference_die(8));

    // Speech to text: a die of one unit of 32 lanes and 2,048 cells. Routines over rows
    // keep whole maps, which every unit of a die would hold a copy of; and with more cells
    // than its matrices have rows, the rest would stand empty, as deep as the deepest.
    let shape = shape_of("LISTEN", (0, 5, 11));
    let samples = (seconds * 16_000.0) as usize;
    let die = die_of(reference_die(8), 1024, shape);
    let recogniser = MoonshineChip::from_parts(MoonshineConfig::read(listen)?, weights(listen, 32)?, samples, die, Some(1024))?;
    let calls: Vec<_> = [recogniser.encoder_calls(), recogniser.token_calls(0, 0)].concat();
    let code = recogniser.microprogram(&calls)?.1;
    let (parts, clock) = routines_die(recogniser.parts().1, &code, &costs, &N6, shape, CLOCK);
    println!("speech to text: {} vector words a unit", recogniser.parts().1.vector_words_used());
    println!("speech to text: logic {:.0}, cells {:.0}, vectors {:.0}, microcode {:.0} mm²", parts[0], parts[1], parts[2], parts[3]);
    let hear = recogniser.stage(1, parts.iter().sum(), clock)?;

    // The answer and the speech's tokens: language models on one die each.
    let answer = language_model("language model", think, &costs, 1, (4, 6, 16), 40)?;
    let writer = language_model("speech writer", tts, &costs, 1, (4, 6, 16), 40)?;

    // The sound decoded a window at a time, on a die with 20-bit vectors: each window keeps
    // `CHUNK_FRAMES` frames and reaches a frame's reach further on both sides.
    let config = DacConfig::read(dac)?;
    let chunk = sim::voice::CHUNK_FRAMES;
    let frames = chunk + 2 * config.reach_frames();
    let base = reference_die(8);
    let wide = netlist::DieConfig { cell: netlist::CellConfig { activation_bits: 20, accumulator_bits: 2 * 20 + 16, ..base.cell }, ..base };
    let wide_costs = Costs::measure(wide);
    let shape = shape_of("SPEAK", (0, 6, 11));
    let decoder = DacChip::from_parts(config.clone(), weights(dac, 8)?, frames, die_of(wide, 32_768, shape), true)?;
    let calls = decoder.calls(&vec![vec![0; frames]; decoder.config.codebooks])?;
    let (parts, clock) = routines_die(decoder.parts().1, &decoder.microprogram()?, &wide_costs, &N6, shape, CLOCK);
    println!("sound decoder: windows of {frames} frames, {chunk} kept; logic {:.0}, cells {:.0}, vectors {:.0}, microcode {:.0} mm²", parts[0], parts[1], parts[2], parts[3]);
    let cycles = decoder.cycles(&calls)?;
    // Its outputs are frames of sound: the first window gives `chunk` of them, as does each after.
    let speak = Stage { name: "sound decoder".into(), dies: 1, area_mm2: parts.iter().sum(), clock_hz: clock, first_cycles: cycles, output_cycles: cycles / chunk as u64 };

    let system = System::new(vec![hear, answer, writer, speak]);
    for stage in &system.stages {
        println!("{:>15}: {} die of {:>5.0} mm² at {:>4.0} MHz, first output after {:>7.1} ms, then {:>6.0} a second", stage.name, stage.dies, stage.area_mm2, stage.clock_hz / 1e6, stage.seconds_to(1) * 1e3, stage.outputs_per_second());
    }
    // OuteTTS writes about 2.4 tokens per frame of speech (codes, and words' own tokens),
    // and the decoder's first window needs `frames` of them.
    let speech_tokens = (frames as f64 * 2.4) as u32;
    // The question's text: about three tokens a second of speech.
    let first = system.first_output_seconds(&[(seconds * 3.0).ceil() as u32, words, speech_tokens])?;
    println!("{} dies, {:.0} mm²; first sound {:.0} ms after the question ends", system.dies(), system.area_mm2(), first * 1e3);
    // Real time: the writer 75 frames a second at 2.4 tokens each, the decoder 75 frames.
    println!("speech keeps up: {}", system.keeps_up(&[0.0, 0.0, 75.0 * 2.4, 75.0]));
    Ok(())
}
