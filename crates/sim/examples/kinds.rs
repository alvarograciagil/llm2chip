//! Where the cycles of one token go, operation by operation, on a die sized for a large
//! model at its full context. It follows the microprogram's own jumps, so the counts are
//! those the circuit spends.
//!
//! `cargo run --release -p sim --example kinds -- MODEL_DIR [OP_INDEX | list]`
//!
//! `UNITS`, `LANES` and `CELLS` (log2) size the die. With an operation index it prints
//! each word of that operation and how many cycles it took.

use netlist::multiplier::MultiplierKind;
use netlist::{CellConfig, ControlWord, DieConfig, DieFeatures, Microprogram};
use sim::lower::Layout;
use sim::Weights;
use std::collections::BTreeMap;
use std::path::Path;

const CONTEXT: u32 = 8192;

fn env(name: &str, default: u32) -> u32 {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn kind(w: &ControlWord) -> &'static str {
    if w.cell_run || w.cell_flush || w.cell_feed || w.chain_shift || w.chain_load || w.cell_clear_group {
        "celdas"
    } else if w.x_write {
        "bucles"
    } else if w.read_step && w.vector {
        "vector"
    } else if w.read_step {
        "1elem"
    } else if w.use_literal && (w.write_en || w.op != 0 || w.literal_a != 0 || w.reg_a != 0) {
        "alu"
    } else {
        "espera"
    }
}

/// Cycles and visits of each word, for the last token of the context.
fn profile(code: &Microprogram, die: &DieConfig) -> BTreeMap<usize, (u64, u64)> {
    let mask = (1u32 << die.index_bits) - 1;
    let mut index = vec![0u32; 1 << 16];
    for p in &code.parameters {
        index[p.register as usize] = p.value(1, CONTEXT - 1) & mask;
    }
    let mut words: BTreeMap<usize, (u64, u64)> = BTreeMap::new();
    let mut pc = code.entries[0] as usize;
    loop {
        let w = &code.words[pc];
        let cycles = if w.repeat_x { ((index[w.x_limit as usize] >> if w.vector { die.features.lane_bits } else { 0 }) + 1) as u64 } else { w.repeat as u64 + 1 };
        let entry = words.entry(pc).or_default();
        (entry.0, entry.1) = (entry.0 + 1, entry.1 + cycles);
        if w.halt {
            return words;
        }
        let mut next = pc + 1;
        if w.x_write {
            let value = index[w.x_src as usize].wrapping_add(w.literal_b) & mask;
            if w.x_dst != 0 && w.x_only <= 1 {
                index[w.x_dst as usize] = value;
            }
            if w.branch && value != index[w.x_limit as usize] {
                next = w.target as usize;
            }
        }
        pc = next;
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = Path::new(args.get(1).expect("carpeta del modelo"));
    let config = std::fs::read_to_string(dir.join("config.json")).unwrap();
    let program = compiler::compile_model(&config).unwrap();
    if args.get(2).map(String::as_str) == Some("list") {
        for (i, inst) in program.ops.iter().enumerate() {
            println!("{i} {:?}", inst.op);
        }
        return;
    }
    let weights = Weights::open(dir).unwrap();
    let cell = CellConfig { weight_bits: 8, activation_bits: 16, accumulator_bits: 36, mac_stages: 1, adder_stages: 1, addr_bits: 24, coding: Default::default(), exponent_bits: 0, multiplier: MultiplierKind::Array, fixed_weights: false, ecc: false, skip_zeros: false };
    let features = DieFeatures::default().with_lane_bits(env("LANES", 7)).with_cluster_bits(env("UNITS", 7)).with_key_shift(CONTEXT.trailing_zeros());
    let die = DieConfig { cells: 1 << env("CELLS", 18), cell, scale_bits: 21, reg_bits: 16, index_bits: 24, microcode_bits: 24, broadcast_stages: 1, function_bits: 12, function_select_bits: 3, features };
    let layout = Layout::for_image(&program, &weights, die, CONTEXT as usize).unwrap();
    let pick: Option<usize> = args.get(2).map(|a| a.parse().expect("índice de operación"));

    let mut table: BTreeMap<String, BTreeMap<&str, u64>> = BTreeMap::new();
    for (i, inst) in program.ops.iter().enumerate() {
        if pick.is_some_and(|p| p != i) {
            continue;
        }
        let code = layout.op_microprogram(&program, inst, i == 0).unwrap();
        let words = profile(&code, &die);
        let name = format!("{:?}", inst.op).split([' ', '{']).next().unwrap_or("").to_string();
        let tally = table.entry(name).or_default();
        for (pc, (visits, cycles)) in &words {
            let w = &code.words[*pc];
            *tally.entry(kind(w)).or_default() += cycles;
            // Capitalised: not cycles of their own, but where some of them went.
            if *visits == 1 && w.repeat == 0 {
                *tally.entry("RECTA").or_default() += cycles;
            }
            if w.load_shift_a || w.load_shift_b || w.load_shift_left || w.load_wide_shift {
                *tally.entry("CARGAS").or_default() += visits;
            }
        }
        if pick.is_some() {
            println!("{:?}", inst.op);
            for (pc, (visits, cycles)) in &words {
                println!("{pc:4} x{visits:<5} {cycles:>6}  {:?}", code.words[*pc]);
            }
        }
    }
    let mut rows: Vec<_> = table.into_iter().map(|(name, t)| (t.iter().filter(|(k, _)| !k.starts_with(char::is_uppercase)).map(|(_, v)| v).sum::<u64>(), name, t)).collect();
    rows.sort_by_key(|r| std::cmp::Reverse(r.0));
    let total: u64 = rows.iter().map(|r| r.0).sum();
    println!("TOTAL {total} ciclos, {:.0} tok/s a 1 GHz", 1e9 / total as f64);
    for (cycles, name, t) in rows.iter().take(12) {
        println!("{cycles:>9} {name:<14} {}", t.iter().map(|(k, v)| format!("{k} {v}")).collect::<Vec<_>>().join(", "));
    }
}
