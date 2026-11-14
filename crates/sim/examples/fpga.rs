//! The whole chip, in Verilog, to synthesise for an FPGA: the RISC-V controller, the die
//! with only the logic its microprogram uses, and the sampling unit.
//!
//! `cargo run --release -p sim --example fpga -- OUT.v [--dsp] [--model DIR]... [--context N] [--cells N] [--lanes N] [--table-inside] [--no-dictionary] [--factor-steps N] [--group N] [--tokenizer FILE] [--chain] [--path NET] [--groups]`
//!
//! With `--dsp`, each one-cycle multiplier is written as a signed product so the FPGA
//! tools put it in their multiplier blocks. Without `--model`, the chip is the one
//! `the_chip_sized_for_an_fpga_says_the_same_in_gates` runs for the test model; with it,
//! a chip of the same shape sized for that model's weights and `--context` tokens; with
//! several, one chip that runs any of them, each booting from its own image. The
//! embedding table stays in the image, and the firmware brings each token's row; with
//! `--table-inside` it goes into the vector memory instead.

#[path = "../tests/common/mod.rs"]
mod common;

use common::*;
use compiler::compile_qwen3_5;
use netlist::{to_verilog, to_verilog_for_fpga, CellConfig, Design, DieConfig, InstKind, Prim};
use sim::chip::{fpga_chip_for, FpgaChip, FpgaOptions};
use sim::lower::Layout;
use sim::Weights;
use std::collections::HashMap;
use std::path::Path;

/// Each memory of the design by its instance name: how many there are, their words and width.
fn memory_list(design: &Design) -> Vec<(String, u64, u32, u32)> {
    fn walk(design: &Design, module: &str, times: u64, out: &mut HashMap<(String, u32, u32), u64>) {
        for inst in &design.modules[module].instances {
            match &inst.kind {
                InstKind::Module(child) => walk(design, child, times, out),
                InstKind::Prim(Prim::Sram { addr_bits, width }) | InstKind::Prim(Prim::Ram2p { addr_bits, width }) | InstKind::Prim(Prim::Rom { addr_bits, width }) => {
                    let name = inst.name.trim_end_matches(|c: char| c.is_ascii_digit()).to_string();
                    *out.entry((name, *addr_bits, *width)).or_default() += times;
                }
                InstKind::Prim(_) => {}
            }
        }
    }
    let mut out = HashMap::new();
    walk(design, &design.top, 1, &mut out);
    let mut list: Vec<_> = out.into_iter().map(|((name, addr, width), count)| (name, count, addr, width)).collect();
    list.sort_by_key(|(_, count, addr, width)| std::cmp::Reverse(count * (1u64 << addr) * *width as u64));
    list
}

/// Bits of every memory in the design, by kind, counting each instance of each module.
fn memory_bits(design: &Design) -> HashMap<&'static str, u64> {
    fn walk(design: &Design, module: &str, times: u64, out: &mut HashMap<&'static str, u64>) {
        for inst in &design.modules[module].instances {
            match &inst.kind {
                InstKind::Module(child) => walk(design, child, times, out),
                InstKind::Prim(Prim::Sram { addr_bits, width }) | InstKind::Prim(Prim::Ram2p { addr_bits, width }) | InstKind::Prim(Prim::Rom { addr_bits, width }) => {
                    let kind = match inst.name.as_str() {
                        n if n.starts_with("weight") || n.contains("scale_mem") || n.contains("bias_mem") => "weights and factors",
                        n if n.starts_with("vectors") || n.starts_with("wide_memory") => "vector memory",
                        n if n.starts_with("microcode") => "microcode",
                        _ => "CPU, tables and the rest",
                    };
                    *out.entry(kind).or_default() += times * (1u64 << addr_bits) * *width as u64;
                }
                InstKind::Prim(_) => {}
            }
        }
    }
    let mut out = HashMap::new();
    walk(design, &design.top, 1, &mut out);
    out
}

fn bits_for(count: u64) -> u32 {
    (u64::BITS - count.max(2).saturating_sub(1).leading_zeros()).max(1)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let out = args.get(1).expect("output file");
    let option = |name: &str| args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned();
    let test_dir = std::env::temp_dir().join(format!("llm2chip-fpga-{}", std::process::id()));
    // Every `--model DIR`: one chip that runs them all, each from its own image.
    let dirs: Vec<&String> = args.iter().enumerate().filter(|(_, a)| *a == "--model").filter_map(|(i, _)| args.get(i + 1)).collect();
    let context: usize = if dirs.is_empty() { 8 } else { option("--context").map_or(128, |n| n.parse().unwrap()) };
    let (models, trial): (Vec<(compiler::Program, Weights)>, DieConfig) = if dirs.is_empty() {
        std::fs::create_dir_all(&test_dir).unwrap();
        // `--group N`: columns that share a scale and a bias in the test model, so a cell
        // can take up to N cycles to apply them.
        if let Some(group) = option("--group") {
            GROUP.with(|g| g.set(group.parse().unwrap()));
        }
        make_checkpoint(&test_dir);
        (vec![(compile_qwen3_5(CONFIG).unwrap(), Weights::open(&test_dir).unwrap())], fpga_die_config())
    } else {
        let cells: u32 = option("--cells").map_or(8, |n| n.parse().unwrap());
        let lane_bits = option("--lanes").map_or(0, |n| n.parse::<u32>().unwrap().max(1).ilog2());
        // Room for everything first, then just the bits the layouts need, the most of any.
        let base = fpga_die_config();
        let roomy = DieConfig { cells, reg_bits: 12, index_bits: 20, cell: CellConfig { addr_bits: 20, weight_bits: base.cell.activation_bits, ..base.cell }, features: base.features.with_key_shift(bits_for(context as u64)).with_lane_bits(lane_bits), ..base };
        let mut trial = DieConfig { index_bits: 1, cell: CellConfig { weight_bits: 2, addr_bits: 1, ..roomy.cell }, ..roomy };
        let models = dirs
            .iter()
            .map(|dir| {
                let dir = Path::new(dir.as_str());
                let program = compiler::compile_model(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
                let weights = Weights::open(dir).unwrap();
                let (code_bits, _) = Layout::with_cells(&program, &weights, roomy, context).unwrap().cell_shape(&weights).unwrap();
                // The cells' depth with words as wide as the codes.
                let narrow = DieConfig { cell: CellConfig { weight_bits: code_bits, ..roomy.cell }, ..roomy };
                let (depth, groups) = Layout::with_cells(&program, &weights, narrow, context).unwrap().cell_depths();
                let addr_bits = bits_for(depth.max(2 * groups));
                let index_bits = bits_for(program.buffers.iter().map(|b| *b as u64).max().unwrap_or(1).max(context as u64 * 512)).max(addr_bits);
                trial = DieConfig { index_bits: trial.index_bits.max(index_bits), cell: CellConfig { weight_bits: trial.cell.weight_bits.max(code_bits), addr_bits: trial.cell.addr_bits.max(addr_bits), ..trial.cell }, ..trial };
                (program, weights)
            })
            .collect();
        (models, trial)
    };
    let tokenizer_words = option("--tokenizer").map_or(0, |path| sim::tokenizer::TokenizerTables::from_file(Path::new(&path), context as u64).unwrap().to_words().len());
    let options = FpgaOptions {
        factor_steps: option("--factor-steps").map_or(trial.features.factor_steps, |n| n.parse().unwrap()),
        table_inside: args.iter().any(|a| a == "--table-inside"),
        dictionary: !args.iter().any(|a| a == "--no-dictionary"),
        tokenizer_words,
        chain: args.iter().any(|a| a == "--chain"),
    };
    let pairs: Vec<(&compiler::Program, &Weights)> = models.iter().map(|(p, w)| (p, w)).collect();
    let FpgaChip { design, config, models: built } = fpga_chip_for(&pairs, trial, context, options).unwrap();
    let die = config.die;
    let last = context as u32 - 1;
    for ((layout, code), name) in built.iter().zip(dirs.iter().map(|d| d.as_str()).chain(["test model"])) {
        eprintln!("== {name}");
        if layout.outside_table().is_some() {
            eprintln!("the embedding table stays in the image");
        }
        eprintln!("longest registers: {:?}", layout.longest_registers(6));
        let (depth, groups) = layout.cell_depths();
        eprintln!("{} registers used (2^{} in the table), {} weight words and {} factor groups per cell", layout.registers_needed(), die.reg_bits, depth, groups);
        let distinct: std::collections::HashSet<_> = code.words.iter().map(|w| format!("{w:?}")).collect();
        eprintln!("microprogram: {} words, {} distinct", code.words.len(), distinct.len());
        // The die's cycles for one generated token at the last position, with its logits.
        let token = code.cycles(&die, sim::lower::ENTRY_TOKEN, 1, last) + code.cycles(&die, sim::lower::ENTRY_LOGITS, 1, last);
        eprintln!("{token} die cycles per token generated at position {last}");
    }
    eprintln!("compact vector memory of 2^{} words, against 2^{} without a table", die.features.vector_bits, die.reg_bits + die.index_bits);
    eprintln!("{:.0} NAND2 equivalents, microcode of 2^{} words", design.gate_equivalents(), die.microcode_bits);
    let mut memories: Vec<_> = memory_bits(&design).into_iter().collect();
    memories.sort();
    let total: u64 = memories.iter().map(|(_, bits)| bits).sum();
    eprintln!("memory: {:.2} Mbit ({})", total as f64 / 1e6, memories.iter().map(|(kind, bits)| format!("{kind} {:.2}", *bits as f64 / 1e6)).collect::<Vec<_>>().join(", "));
    if args.iter().any(|a| a == "--memories") {
        for (name, count, addr, width) in memory_list(&design) {
            eprintln!("  {count} × {name}: 2^{addr} × {width} bits = {:.3} Mbit", (count * (1u64 << addr) * width as u64) as f64 / 1e6);
        }
    }
    // `--groups`: the die's gates and flip-flops by the area they are drawn in.
    if args.iter().any(|a| a == "--groups") {
        fn gates(design: &Design, module: &str) -> (u64, u64) {
            design.modules[module].instances.iter().fold((0, 0), |(g, f), inst| match &inst.kind {
                InstKind::Module(child) => {
                    let (cg, cf) = gates(design, child);
                    (g + cg, f + cf)
                }
                InstKind::Prim(Prim::Dff) => (g, f + 1),
                InstKind::Prim(Prim::Sram { .. } | Prim::Ram2p { .. } | Prim::Rom { .. }) => (g, f),
                InstKind::Prim(_) => (g + 1, f),
            })
        }
        let mut groups: std::collections::BTreeMap<String, (u64, u64)> = Default::default();
        // The die by its areas, and the rest of the chip by its top-level parts.
        let parts = design.modules["llm2chip_top"].instances.iter().filter(|i| !matches!(&i.kind, InstKind::Module(m) if m == "llm2chip_die")).map(|i| (format!("chip: {}", if i.group.is_empty() { &i.name } else { &i.group }), i));
        for (group, inst) in design.modules["llm2chip_die"].instances.iter().map(|i| (i.group.clone(), i)).chain(parts) {
            let (g, f) = match &inst.kind {
                InstKind::Module(child) => gates(&design, child),
                InstKind::Prim(Prim::Dff) => (0, 1),
                InstKind::Prim(Prim::Sram { .. } | Prim::Ram2p { .. } | Prim::Rom { .. }) => (0, 0),
                InstKind::Prim(_) => (1, 0),
            };
            let entry = groups.entry(group).or_default();
            (entry.0, entry.1) = (entry.0 + g, entry.1 + f);
        }
        let mut sorted: Vec<_> = groups.into_iter().collect();
        sorted.sort_by_key(|(_, (g, _))| std::cmp::Reverse(*g));
        for (group, (g, f)) in sorted {
            eprintln!("  {g:8} gates {f:7} flip-flops  {group}");
        }
    }
    // `--path NET`: the deepest path into a net whose name has NET, in gates, as the FPGA
    // tools start from it before packing gates into look-up tables.
    if let Some(net) = args.iter().position(|a| a == "--path").and_then(|i| args.get(i + 1)) {
        let trace = netlist::Simulator::with_names(&design).unwrap().trace_to([1.0; 5], |name| name.contains(net.as_str()));
        let mut hidden = 0;
        for (name, depth) in trace {
            if name.starts_with('#') {
                hidden += 1;
                continue;
            }
            eprintln!("  {depth:4.0} {name}{}", if hidden > 0 { format!("  (+{hidden} unnamed)") } else { String::new() });
            hidden = 0;
        }
    }
    let text = if args.iter().any(|a| a == "--dsp") { to_verilog_for_fpga(&design) } else { to_verilog(&design) };
    std::fs::write(out, text).unwrap();
    std::fs::remove_dir_all(test_dir).ok();
}
