//! The die a model needs for a speed: the least logic whose own microprogram gives at
//! least the tokens per second asked for, at the last position of the context.
//!
//! `cargo run --release -p sim --example size -- MODEL_DIR TOKENS_PER_SECOND [CONTEXT] [GHZ]`

use netlist::multiplier::MultiplierKind;
use netlist::{CellConfig, DieConfig, DieFeatures};
use sim::sizing::{search, Costs, Target};
use sim::Weights;
use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = Path::new(args.get(1).expect("carpeta del modelo"));
    let wanted: f64 = args.get(2).expect("tokens por segundo").parse().expect("un número");
    let context: u32 = args.get(3).map_or(8192, |a| a.parse().expect("contexto"));
    let ghz: f64 = args.get(4).map_or(1.0, |a| a.parse().expect("frecuencia"));
    let config = std::fs::read_to_string(dir.join("config.json")).unwrap();
    let program = compiler::compile_model(&config).unwrap();
    let weights = Weights::open(dir).unwrap();
    let cell = CellConfig { weight_bits: 8, activation_bits: 16, accumulator_bits: 36, mac_stages: 1, adder_stages: 1, addr_bits: 24, coding: Default::default(), exponent_bits: 0, multiplier: MultiplierKind::Array, fixed_weights: false, ecc: false, skip_zeros: false };
    let base = DieConfig { cells: 1, cell, scale_bits: 21, reg_bits: 16, index_bits: 24, microcode_bits: 24, broadcast_stages: 1, function_bits: 12, function_select_bits: 3, features: DieFeatures::default().with_factor_steps(std::env::var("K").ok().and_then(|k| k.parse().ok()).unwrap_or(8)) };
    let costs = Costs::measure(base);
    println!("puertas: fijo {:.0}, unidad {:.0} + {:.0}/carril, nivel {:.0} + {:.0}/carril, celda {:.0}", costs.fixed, costs.unit, costs.lane, costs.level, costs.level_lane, costs.cell);
    let target = Target { tokens_per_second: wanted, clock_hz: ghz * 1e9, context, lane_bits: 5..=8, cluster_bits: 5..=11, cell_bits: 16..=25 };
    let found = search(&program, &weights, base, &costs, &target);
    if found.is_empty() {
        println!("ningún die del rango alcanza {wanted} tok/s");
    }
    for c in found.iter().take(10) {
        println!("{:>5} unidades x {:>3} carriles, {:>9} celdas: {:>8} ciclos, {:>6.0} tok/s, {:>8.1} M puertas", 1u64 << c.cluster_bits, 1u64 << c.lane_bits, 1u64 << c.cell_bits, c.cycles, c.tokens_per_second, c.gates / 1e6);
    }
}
