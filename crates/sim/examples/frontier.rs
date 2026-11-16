//! The speeds a model can have and the logic each costs, and where more speed stops being
//! worth its logic.
//!
//! `cargo run --release -p sim --example frontier -- MODEL_DIR [PROMPT] [CONTEXT]`
//!
//! `CHIPS` lists the numbers of chips to try, `8,12,16` by default: each die must fit in
//! the reticle. `SHARING` is `layers` (the default), `rows` or `mixed:N`, groups of N chips
//! that share each layer, its products and its heads.

use netlist::multiplier::MultiplierKind;
use netlist::{CellConfig, DieConfig, DieFeatures};
use sim::sizing::{Process, balanced, efficient, frontier, Costs, Target};
use sim::Weights;
use std::path::Path;

/// TSMC N6 as `tech/n6.json` estimates it: an AND2 is 1.33 NAND2 equivalents, and wiring
/// adds a fifth. The reticle is 26 by 33 mm.
const N6: Process = Process { full_adder_ns: 0.04, gate_um2: 0.0875 / 1.33 * 1.2, sram_cell_um2: 0.027, sram_periphery_k: 5.0, reticle_mm2: 858.0 };

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = Path::new(args.get(1).expect("model folder"));
    let prompt: u32 = args.get(2).map_or(1024, |a| a.parse().expect("prompt"));
    let context: u32 = args.get(3).map_or(8192, |a| a.parse().expect("context"));
    let config = std::fs::read_to_string(dir.join("config.json")).unwrap();
    let program = compiler::compile_model(&config).unwrap();
    let weights = Weights::open(dir).unwrap();
    let cell = CellConfig { weight_bits: 8, activation_bits: 16, accumulator_bits: 36, mac_stages: 1, adder_stages: 1, addr_bits: 24, coding: Default::default(), exponent_bits: 0, multiplier: MultiplierKind::Booth, fixed_weights: false, ecc: false, skip_zeros: false };
    let base = DieConfig { cells: 1, cell, scale_bits: 21, reg_bits: 16, index_bits: 24, microcode_bits: 24, broadcast_stages: 1, function_bits: 12, function_select_bits: 3, features: DieFeatures::default().with_factor_steps(8) };
    let costs = Costs::measure(base);
    let target = Target { tokens_per_second: 0.0, prompt_seconds: f64::INFINITY, prompt, chips: std::env::var("CHIPS").unwrap_or("8,12,16".into()).split(',').map(|n| n.parse().expect("CHIPS")).collect(), clock_hz: 5e9, process: N6, context, lane_bits: range("LANES", 6..=7), cluster_bits: range("UNITS", 5..=11), cell_bits: range("CELLS", 13..=24), sharing: sharing_from(&std::env::var("SHARING").unwrap_or_default()) };
    let points = frontier(&program, &weights, base, &costs, &target);
    let first = points.first().copied();
    for c in &points {
        let extra = first.map_or(0.0, |f| c.area_mm2 / f.area_mm2);
        println!("{} chip(s) {:>5} x {:>3}, {:>9} cells: {:>6.0} tok/s at {:>4.0} MHz, first token {:>6.1} ms, {:>9.1} M gates, {:>7.0} mm² ({extra:>4.2}x, {:>4.0} per chip), {:>5.3} mm² per tok/s", c.chips, 1u64 << c.cluster_bits, 1u64 << c.lane_bits, 1u64 << c.cell_bits, c.tokens_per_second, c.clock_hz / 1e6, c.prompt_seconds * 1e3, c.gates / 1e6, c.area_mm2, c.die_mm2, c.area_mm2 / c.tokens_per_second);
        let [logic, cells, vectors, code] = c.parts_mm2;
        println!("    per chip: logic {logic:.0} mm², cell memories {cells:.0}, vector memories {vectors:.0}, microcode {code:.1}");
    }
    if let Some(c) = efficient(&points) {
        println!("most speed per area: {} chip(s) of {} x {}, {} cells, {:.0} tok/s, {:.0} mm², {:.3} mm² per tok/s", c.chips, 1u64 << c.cluster_bits, 1u64 << c.lane_bits, 1u64 << c.cell_bits, c.tokens_per_second, c.area_mm2, c.area_mm2 / c.tokens_per_second);
    }
    for ratio in [1.5, 2.0, 3.0] {
        if let Some(c) = balanced(&points, ratio) {
            println!("balance with each extra tok/s at no more than {ratio}x the mean area per tok/s: {} chip(s) of {} x {}, {} cells, {:.0} tok/s, {:.0} mm²", c.chips, 1u64 << c.cluster_bits, 1u64 << c.lane_bits, 1u64 << c.cell_bits, c.tokens_per_second, c.area_mm2);
        }
    }
}

/// `layers`, `rows` or `mixed:N`.
fn sharing_from(text: &str) -> sim::sizing::Sharing {
    match text.split_once(':') {
        Some(("mixed", n)) => sim::sizing::Sharing::Mixed(n.parse().expect("the size of the groups")),
        _ if text == "rows" => sim::sizing::Sharing::Rows,
        _ => sim::sizing::Sharing::Layers,
    }
}

/// A range of log2 sizes from the environment, `low..high` or one value, or the default.
fn range(name: &str, default: std::ops::RangeInclusive<u32>) -> std::ops::RangeInclusive<u32> {
    let Ok(text) = std::env::var(name) else { return default };
    match text.split_once("..") {
        Some((low, high)) => low.parse().expect(name)..=high.parse().expect(name),
        None => {
            let value = text.parse().expect(name);
            value..=value
        }
    }
}
