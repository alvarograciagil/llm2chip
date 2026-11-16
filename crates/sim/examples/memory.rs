//! What each chip of a set holds: the words of its microprogram, and the vector memory of
//! each unit on the chip that needs most, by what it holds.
//!
//! `cargo run --release -p sim --example memory -- MODEL_DIR [CONTEXT]`
//!
//! `CHIPS` (16 by default), and `UNITS`, `LANES` and `CELLS` (log2: 6, 6 and 15 by default)
//! give the set: the frontier's fastest point for Qwen3.8-27B.

use netlist::multiplier::MultiplierKind;
use netlist::{CellConfig, DieConfig, DieFeatures};
use sim::sizing::{built, die_of, Sharing};
use sim::Weights;
use std::path::Path;

fn env(name: &str, default: u32) -> u32 {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = Path::new(args.get(1).expect("model folder"));
    let context: u32 = args.get(2).map_or(8192, |a| a.parse().expect("context"));
    let chips = env("CHIPS", 16);
    let config = std::fs::read_to_string(dir.join("config.json")).expect("config.json");
    let program = compiler::compile_model(&config).unwrap();
    let weights = Weights::open(dir).unwrap();
    let cell = CellConfig { weight_bits: 8, activation_bits: 16, accumulator_bits: 36, mac_stages: 1, adder_stages: 1, addr_bits: 24, coding: Default::default(), exponent_bits: 0, multiplier: MultiplierKind::Booth, fixed_weights: false, ecc: false, skip_zeros: false };
    let base = DieConfig { cells: 1, cell, scale_bits: 21, reg_bits: 16, index_bits: 24, microcode_bits: 24, broadcast_stages: 1, function_bits: 12, function_select_bits: 3, features: DieFeatures::default().with_factor_steps(8) };
    let die = die_of(base, context, (env("UNITS", 6), env("LANES", 6), env("CELLS", 15)));
    let set = built(&program, &weights, die, context, chips, Sharing::Layers).unwrap();
    let words: Vec<usize> = set.programs.iter().map(|p| p.words.len()).collect();
    println!("microprogram words of each chip: {words:?}");
    println!("vector memory of each unit, on the chip that needs most: {} words", set.vector_words);
    for (kind, count) in &set.vector_kinds {
        println!("  {kind}: {count} ({:.0} %)", 100.0 * *count as f64 / set.vector_words as f64);
    }
    println!("microcode memory: {:.2} Mbit", set.microcode_bits as f64 / 1e6);
    println!("cell memory: {} bits", set.cell_bits);
}
