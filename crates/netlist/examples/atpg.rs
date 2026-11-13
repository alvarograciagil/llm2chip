//! Scan chain and test patterns for the blocks of the chip, with their fault coverage.
//!
//! `cargo run --release -p netlist --example atpg [bloque…]`, the blocks among `celda`, `cpu`,
//! `muestreo` and `die`; without any, all but the die, which takes much longer.

use netlist::atpg;
use netlist::format::WeightCoding;
use netlist::multiplier::MultiplierKind;
use netlist::sampler::{sampler_design, SamplerConfig};
use netlist::scan::insert_scan;
use netlist::{cell_design, die_design, CellConfig, Design, DieConfig, DieFeatures};

type Block = (&'static str, Box<dyn Fn() -> Design>);

fn main() {
    let wanted: Vec<String> = std::env::args().skip(1).collect();
    let blocks: Vec<Block> = vec![
        ("cell", Box::new(|| cell_design(CellConfig { weight_bits: 4, activation_bits: 16, accumulator_bits: 32, mac_stages: 2, adder_stages: 2, addr_bits: 5, coding: WeightCoding::Affine, exponent_bits: 0, multiplier: MultiplierKind::Array, fixed_weights: false, ecc: false, skip_zeros: false }))),
        ("cpu", Box::new(netlist::rv32::core_design)),
        ("sampler", Box::new(|| sampler_design(SamplerConfig { logit_bits: 16, index_bits: 12, entries: 8, lanes: 1 }))),
        // A small die whole: three cells, the sequencer, the vector unit and the tables.
        ("die", Box::new(|| {
            die_design(DieConfig {
                cells: 3,
                cell: CellConfig { weight_bits: 8, activation_bits: 16, accumulator_bits: 32, mac_stages: 2, adder_stages: 2, addr_bits: 4, coding: Default::default(), exponent_bits: 0, multiplier: MultiplierKind::Array, fixed_weights: false, ecc: false, skip_zeros: false },
                scale_bits: 16,
                reg_bits: 3,
                index_bits: 4,
                microcode_bits: 9,
                broadcast_stages: 2,
                function_bits: 8,
                function_select_bits: 3,
                features: DieFeatures::without_register_windows(),
            })
        })),
    ];
    println!("| Block | Flip-flops in the chain | Faults | Detected | By the chain | Undetectable | Aborted | Patterns | Fault coverage | Test coverage | Seconds |");
    println!("|---|---|---|---|---|---|---|---|---|---|---|");
    for (name, build) in blocks {
        if if wanted.is_empty() { name == "die" } else { !wanted.iter().any(|w| w == name) } {
            continue;
        }
        let started = std::time::Instant::now();
        let scanned = insert_scan(build());
        let r = atpg::generate(&scanned.design, 1000).expect("ATPG");
        println!(
            "| {name} | {} | {} | {} | {} | {} | {} | {} + {} | {:.2} % | {:.2} % | {:.1} |",
            r.chain_length, r.faults, r.detected, r.chain, r.untestable, r.aborted, r.random_patterns, r.deterministic_patterns,
            r.fault_coverage() * 100.0, r.test_coverage() * 100.0, started.elapsed().as_secs_f64()
        );
        // With ATPG_PERDIDOS, where the faults no pattern finds sit, by module.
        if std::env::var("ATPG_PERDIDOS").is_ok() {
            let lost: Vec<_> = r.outcomes.iter().filter(|(_, o)| *o != atpg::Outcome::Detected).collect();
            let names = atpg::describe(&scanned.design, &lost.iter().map(|(f, _)| *f).collect::<Vec<_>>()).unwrap();
            let mut count: std::collections::BTreeMap<String, usize> = Default::default();
            for ((_, outcome), name) in lost.iter().zip(&names) {
                let place: String = name.split_whitespace().next().unwrap_or("").rsplit_once('/').map_or("", |(dir, _)| dir).to_string();
                *count.entry(format!("{outcome:?} {place}")).or_default() += 1;
            }
            let mut sorted: Vec<_> = count.into_iter().collect();
            sorted.sort_by_key(|(_, n)| std::cmp::Reverse(*n));
            for (place, n) in sorted.iter().take(25) {
                eprintln!("{n:6} {place}");
            }
        }
    }
}
