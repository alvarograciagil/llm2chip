//! The chips a model needs for a speed and a wait: the least logic whose own microprograms
//! give at least the tokens per second asked for at the last position of the context, and
//! the first token within the time asked for after a prompt.
//!
//! `cargo run --release -p sim --example size -- MODEL_DIR TOKENS_PER_SECOND [FIRST_TOKEN_MS] [PROMPT] [CONTEXT] [MAX_GHZ]`
//!
//! Each die runs at its own fastest clock in N6, from its critical path, or at `MAX_GHZ`
//! if that is lower.
//!
//! `K` sets the cycles a cell takes to apply its factors (8 by default).

use netlist::multiplier::MultiplierKind;
use netlist::{CellConfig, DieConfig, DieFeatures};
use sim::sizing::{search, Costs, Target};
use sim::Weights;
use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = Path::new(args.get(1).expect("carpeta del modelo"));
    let number = |at: usize, default: f64| args.get(at).map_or(default, |a| a.parse().expect("un número"));
    let wanted = number(2, 1000.0);
    let first_ms = number(3, f64::INFINITY);
    let prompt = number(4, 1024.0) as u32;
    let context = number(5, 8192.0) as u32;
    let ghz = number(6, 5.0);
    let config = std::fs::read_to_string(dir.join("config.json")).unwrap();
    let program = compiler::compile_model(&config).unwrap();
    let weights = Weights::open(dir).unwrap();
    let steps = std::env::var("K").ok().and_then(|k| k.parse().ok()).unwrap_or(8);
    let cell = CellConfig { weight_bits: 8, activation_bits: 16, accumulator_bits: 36, mac_stages: 1, adder_stages: 1, addr_bits: 24, coding: Default::default(), exponent_bits: 0, multiplier: MultiplierKind::Array, fixed_weights: false, ecc: false, skip_zeros: false };
    let base = DieConfig { cells: 1, cell, scale_bits: 21, reg_bits: 16, index_bits: 24, microcode_bits: 24, broadcast_stages: 1, function_bits: 12, function_select_bits: 3, features: DieFeatures::default().with_factor_steps(steps) };
    let costs = Costs::measure(base);
    println!("puertas: fijo {:.0}, unidad {:.0} + {:.0}/carril, nivel {:.0} + {:.0}/carril, celda {:.0}", costs.fixed, costs.unit, costs.lane, costs.level, costs.level_lane, costs.cell);
    println!("camino crítico: {:.1} sumadores completos + {:.1} por cada doble de carriles + {:.1} por cada doble de unidades", costs.path, costs.path_lane_bit, costs.path_unit_bit);
    // N6: the delay of a full adder in that process.
    let target = Target { tokens_per_second: wanted, prompt_seconds: first_ms / 1e3, prompt, chips: vec![1, 2, 4, 8], clock_hz: ghz * 1e9, full_adder_ns: 0.04, context, lane_bits: 6..=7, cluster_bits: 5..=11, cell_bits: 16..=25 };
    let found = search(&program, &weights, base, &costs, &target);
    if found.is_empty() {
        println!("ningún conjunto de chips del rango alcanza {wanted} tok/s y el primer token en {first_ms} ms");
    }
    for c in found.iter().take(12) {
        println!("{} chips de {:>5} unidades x {:>3} carriles, {:>9} celdas: {:>7} ciclos a {:>4.0} MHz, {:>6.0} tok/s, primer token tras {prompt} en {:>6.1} ms, {:>8.1} M puertas", c.chips, 1u64 << c.cluster_bits, 1u64 << c.lane_bits, 1u64 << c.cell_bits, c.cycles, c.clock_hz / 1e6, c.tokens_per_second, c.prompt_seconds * 1e3, c.gates / 1e6);
    }
}
