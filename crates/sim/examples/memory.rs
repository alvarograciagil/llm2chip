//! What each chip of a set holds: the words of its microprogram, and the vector memory of
//! each unit on the chip that needs most, by what it holds.
//!
//! `cargo run --release -p sim --example memory -- MODEL_DIR [CONTEXT]`
//!
//! `CHIPS` (16 by default), and `UNITS`, `LANES` and `CELLS` (log2: 6, 6 and 15 by default)
//! give the set: the frontier's fastest point for Qwen3.8-27B. `SHARING` is `layers` (the
//! default), `rows` or `mixed:N` for groups of N chips. It also gives the cycles of a token at
//! the last position of the context, through every chip. With `PROFILE` set to a chip's
//! number it also gives where that chip's cycles go, by kind of operation. `FACTOR_STEPS`
//! (8 by default) is how many cycles each cell takes to apply a group's scale and bias.

use netlist::multiplier::MultiplierKind;
use netlist::{CellConfig, DieConfig, DieFeatures};
use sim::sizing::{built, die_of, timing, Sharing};
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
    let base = DieConfig { cells: 1, cell, scale_bits: 21, reg_bits: 16, index_bits: 24, microcode_bits: 24, broadcast_stages: 1, function_bits: 12, function_select_bits: 3, features: DieFeatures::default().with_factor_steps(std::env::var("FACTOR_STEPS").map_or(8, |v| v.parse().expect("FACTOR_STEPS"))) };
    let die = die_of(base, context, (env("UNITS", 6), env("LANES", 6), env("CELLS", 15)));
    let sharing = sharing_from(&std::env::var("SHARING").unwrap_or_default());
    let set = built(&program, &weights, die, context, chips, sharing).unwrap();
    let words: Vec<usize> = set.programs.iter().map(|p| p.words.len()).collect();
    println!("microprogram words of each chip: {words:?}");
    println!("vector memory of each unit, on the chip that needs most: {} words", set.vector_words);
    for (kind, count) in &set.vector_kinds {
        println!("  {kind}: {count} ({:.0} %)", 100.0 * *count as f64 / set.vector_words as f64);
    }
    println!("microcode memory: {:.2} Mbit", set.microcode_bits as f64 / 1e6);
    println!("cell memory: {} bits", set.cell_bits);
    let time = timing(&set.programs, die, context, 1024, 256, sharing);
    println!("a token at position {}: {} cycles, {:.0} tok/s at 1 GHz", context - 1, time.token_cycles, 1e9 / time.token_cycles as f64);
    if let Ok(chip) = std::env::var("PROFILE") {
        let chip: usize = chip.parse().expect("a chip's number");
        let share = sim::chip::ShareOut::new(&program, &weights, chips as usize, context as usize, die.cell.activation_bits, sharing).unwrap();
        let mut layout = sim::lower::Layout::for_image_table_outside(&program, &weights, die, context as usize).unwrap();
        share.microprogram(&mut layout, &program, chip).unwrap();
        if std::env::var("CONSTANTS").is_ok() {
            let named = layout.largest_constants(12);
            let total: u64 = named.iter().map(|(_, w)| w).sum();
            println!("largest constants of chip {chip} ({total} words among them):");
            for (name, words) in named {
                println!("  {words:>8} {name}");
            }
        }
        if let Ok(op) = std::env::var("WORDS") {
            let inst = &program.ops[op.parse::<usize>().expect("an operation's number")];
            let code = layout.op_microprogram(&program, inst, false).unwrap();
            for (pc, (visits, cycles)) in profile(&code, &die, context - 1) {
                println!("{pc:4} x{visits:<5} {cycles:>6}  {:?}", code.words[pc]);
            }
        }
        if std::env::var("KINDS").is_ok() {
            // Each word by what it does: the cells, the link, a block at a time or alone.
            let start = share.plan()[chip / share.group().unwrap_or(1)].start;
            let ops = share.cycles_by_op(&layout, &program, chip, context - 1).unwrap().len();
            let mut totals: std::collections::BTreeMap<&str, u64> = Default::default();
            let mut signatures: std::collections::BTreeMap<String, u64> = Default::default();
            let mut by_op: std::collections::BTreeMap<(String, &str), u64> = Default::default();
            for inst in &program.ops[start..start + ops] {
                let name = format!("{:?}", inst.op).split([' ', '{', '(']).next().unwrap().to_string();
                let code = layout.op_microprogram(&program, inst, false).unwrap();
                for (pc, (_, cycles)) in profile(&code, &die, context - 1) {
                    let w = &code.words[pc];
                    let kind = if w.cell_feed || w.cell_run || w.cell_flush || w.chain_shift || w.chain_load || w.cell_clear_group || w.cell_seek_weights || w.cell_seek_factors {
                        "cells"
                    } else if w.link_send || w.op == netlist::die::OP_LINK {
                        "link"
                    } else if w.vector && (w.read_step || w.repeat_x || w.repeat > 0) {
                        "blocks"
                    } else if w.read_step || w.repeat > 0 || w.repeat_x {
                        "streams of one"
                    } else if w.x_write || w.set_clusters {
                        "single: index"
                    } else if w.use_literal && w.op == 10 {
                        "single: immediate"
                    } else if w.op == 0 && !w.write_en && !w.write_now && !w.magnitude_scan && !w.wide_scan && !w.max_scan && !w.accumulate && !w.wide_write && !w.load_shift_left && !w.load_shift_a && !w.wide_clear && !w.magnitude_clear && !w.halt {
                        "single: wait"
                    } else if w.vector {
                        "single: one block"
                    } else {
                        "single: one element"
                    };
                    *totals.entry(kind).or_default() += cycles;
                    *by_op.entry((name.clone(), if kind.starts_with("single") { "single" } else { kind })).or_default() += cycles;
                    if std::env::var("KINDS").as_deref() == Ok(kind) {
                        let text = format!("{w:?}");
                        let fields: Vec<&str> = text.trim_start_matches("ControlWord { ").trim_end_matches(" }").split(", ").filter(|f| !f.ends_with(": false") && !f.ends_with(": 0")).map(|f| f.split(':').next().unwrap()).collect();
                        *signatures.entry(fields.join(" ")).or_default() += cycles;
                    }
                }
            }
            let mut common: Vec<_> = signatures.into_iter().collect();
            common.sort_by_key(|(_, c)| std::cmp::Reverse(*c));
            for (fields, cycles) in common.iter().take(25) {
                println!("  {cycles:>7} {fields}");
            }
            let mut rows: Vec<_> = by_op.into_iter().collect();
            rows.sort_by_key(|(_, c)| std::cmp::Reverse(*c));
            for ((name, kind), cycles) in rows.iter().take(30) {
                println!("  {name:<16} {kind:<16} {cycles:>8}");
            }
            let total: u64 = totals.values().sum();
            println!("chip {chip}, by kind of word: {total} cycles");
            for (kind, cycles) in totals {
                println!("  {kind:<16} {cycles:>9} ({:.0} %)", 100.0 * cycles as f64 / total as f64);
            }
        }
        let mut kinds: std::collections::BTreeMap<String, (u64, u64)> = Default::default();
        let list = share.cycles_by_op(&layout, &program, chip, context - 1).unwrap();
        if std::env::var("EACH").is_ok() {
            let start = share.plan()[chip / share.group().unwrap_or(1)].start;
            for (i, (name, cycles)) in list.iter().enumerate().take(40) {
                let op = &program.ops[start + i].op;
                let shape = match op {
                    compiler::Op::Matvec { src, dst, matrix } => format!("{matrix}: {} -> {}", program.buffers[*src], program.buffers[*dst]),
                    _ => String::new(),
                };
                println!("  {:>5} {name:<16} {cycles:>7} {shape}", start + i);
            }
        }
        for (name, cycles) in list {
            let entry = kinds.entry(name).or_default();
            (entry.0, entry.1) = (entry.0 + 1, entry.1 + cycles);
        }
        let mut rows: Vec<_> = kinds.into_iter().collect();
        rows.sort_by_key(|(_, (_, cycles))| std::cmp::Reverse(*cycles));
        let total: u64 = rows.iter().map(|(_, (_, c))| c).sum();
        println!("chip {chip}, by operation: {total} cycles");
        for (name, (count, cycles)) in rows {
            println!("  {name:<16} {count:>4} ops {cycles:>9} cycles ({:.0} %)", 100.0 * cycles as f64 / total as f64);
        }
    }
}

/// `layers`, `rows` or `mixed:N`.
fn sharing_from(text: &str) -> Sharing {
    match text.split_once(':') {
        Some(("mixed", n)) => Sharing::Mixed(n.parse().expect("the size of the groups")),
        _ if text == "rows" => Sharing::Rows,
        _ => Sharing::Layers,
    }
}

/// Cycles and visits of each word of a routine, for a token at `position`.
fn profile(code: &netlist::Microprogram, die: &DieConfig, position: u32) -> std::collections::BTreeMap<usize, (u64, u64)> {
    let mask = (1u32 << die.index_bits) - 1;
    let mut index = vec![0u32; 1 << 16];
    for p in &code.parameters {
        index[p.register as usize] = p.value(1, position) & mask;
    }
    let mut words: std::collections::BTreeMap<usize, (u64, u64)> = Default::default();
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
