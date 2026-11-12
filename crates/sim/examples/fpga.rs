//! The whole chip, in Verilog, to synthesise for an FPGA: the RISC-V controller, the die
//! with only the logic its microprogram uses, and the sampling unit.
//!
//! `cargo run --release -p sim --example fpga -- OUT.v [--dsp] [--model DIR] [--context N] [--cells N] [--lanes N] [--table-inside] [--no-dictionary] [--factor-steps N] [--tokenizer FILE] [--chain] [--path NET] [--groups]`
//!
//! With `--dsp`, each one-cycle multiplier is written as a signed product so the FPGA
//! tools put it in their multiplier blocks. Without `--model`, the chip is the one
//! `the_chip_sized_for_an_fpga_says_the_same_in_gates` runs for the test model; with it,
//! a chip of the same shape sized for that model's weights and `--context` tokens. The
//! embedding table stays in the image, and the firmware brings each token's row; with
//! `--table-inside` it goes into the vector memory instead.

#[path = "../tests/common/mod.rs"]
mod common;

use common::*;
use compiler::compile_qwen3_5;
use netlist::chip::{chip_design, chip_design_alone, ChipConfig};
use netlist::sampler::SamplerConfig;
use netlist::soc::SocConfig;
use netlist::{to_verilog, to_verilog_for_fpga, CellConfig, Design, DieConfig, DieFeatures, InstKind, Prim};
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
                        n if n.starts_with("weight") || n.contains("scale_mem") || n.contains("bias_mem") => "pesos y factores",
                        n if n.starts_with("vectors") || n.starts_with("wide_memory") => "memoria vectorial",
                        n if n.starts_with("microcode") => "microcódigo",
                        _ => "CPU, tablas y resto",
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
    let out = args.get(1).expect("archivo de salida");
    let option = |name: &str| args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned();
    let test_dir = std::env::temp_dir().join(format!("llm2chip-fpga-{}", std::process::id()));
    let (program, weights, trial, context) = match option("--model") {
        Some(dir) => {
            let dir = Path::new(&dir);
            let program = compiler::compile_model(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
            let weights = Weights::open(dir).unwrap();
            let context: usize = option("--context").map_or(128, |n| n.parse().unwrap());
            let cells: u32 = option("--cells").map_or(8, |n| n.parse().unwrap());
            let lane_bits = option("--lanes").map_or(0, |n| n.parse::<u32>().unwrap().max(1).ilog2());
            // Room for everything first, then just the bits the layout needs.
            let base = fpga_die_config();
            let roomy = DieConfig { cells, reg_bits: 12, index_bits: 20, cell: CellConfig { addr_bits: 20, ..base.cell }, features: base.features.with_key_shift(bits_for(context as u64)).with_lane_bits(lane_bits), ..base };
            let layout = Layout::with_cells(&program, &weights, roomy, context).unwrap();
            let (depth, groups) = layout.cell_depths();
            let (code_bits, _) = layout.cell_shape(&weights).unwrap();
            let addr_bits = bits_for(depth.max(2 * groups));
            let index_bits = bits_for(program.buffers.iter().map(|b| *b as u64).max().unwrap_or(1).max(context as u64 * 512)).max(addr_bits);
            // Only as many registers as the routines reach into.
            layout.microprogram(&program).unwrap();
            let trial = DieConfig { reg_bits: bits_for(layout.registers_needed() as u64 + 2), index_bits, cell: CellConfig { weight_bits: code_bits.max(2), addr_bits, ..roomy.cell }, ..roomy };
            (program, weights, trial, context)
        }
        None => {
            std::fs::create_dir_all(&test_dir).unwrap();
            make_checkpoint(&test_dir);
            (compile_qwen3_5(CONFIG).unwrap(), Weights::open(&test_dir).unwrap(), fpga_die_config(), 8)
        }
    };
    // With `--factor-steps`, scale and bias multiply a few bits a cycle from registers, which
    // cuts the cell's longest path: factor memory, product and total in one cycle. The
    // microprogram is laid out for it.
    let steps: u32 = option("--factor-steps").map_or(trial.features.factor_steps, |n| n.parse().unwrap());
    let trial = DieConfig { features: trial.features.with_factor_steps(steps), ..trial };
    let mut layout = Layout::with_cells(&program, &weights, trial, context).unwrap();
    if !args.iter().any(|a| a == "--table-inside") && layout.table_outside(&program) {
        eprintln!("la tabla de embedding se queda en la imagen");
    }
    let code = layout.microprogram(&program).unwrap();
    eprintln!("registros más largos: {:?}", layout.longest_registers(6));
    let (depth, groups) = layout.cell_depths();
    eprintln!("{} registros (2^{} en la tabla), {} palabras de pesos y {} grupos de factores por celda", layout.registers_used(), trial.reg_bits, depth, groups);
    // Only what this microprogram uses.
    let features = DieFeatures::of(&code).with_lane_bits(trial.features.lane_bits).with_cluster_bits(trial.features.cluster_bits).with_key_shift(trial.features.key_shift).with_factor_steps(steps);
    let die = sim::chip::compact_die(DieConfig { features, microcode_bits: bits_for(code.words.len() as u64), ..trial }, &layout);
    let die = if args.iter().any(|a| a == "--no-dictionary") { die } else { sim::chip::with_code_dictionary(die, &code) };
    let distinct: std::collections::HashSet<_> = code.words.iter().map(|w| format!("{w:?}")).collect();
    eprintln!("microprograma: {} palabras, {} distintas", code.words.len(), distinct.len());
    eprintln!("memoria vectorial compacta de 2^{} palabras, frente a 2^{} sin tabla", die.features.vector_bits, die.reg_bits + die.index_bits);
    // The firmware's data, its stack and the chip's description take the first 32 KB; with
    // `--tokenizer DIR/tokenizer.json` the CPU also keeps that model's tables, after them.
    let tokenizer_words = option("--tokenizer").map_or(0, |path| sim::tokenizer::TokenizerTables::from_file(Path::new(&path), context as u64).unwrap().to_words().len());
    let data_bits = (netlist::soc::TOKENIZER_WORD + tokenizer_words).next_power_of_two().ilog2();
    let config = ChipConfig { soc: SocConfig { program_bits: 13, data_bits, fifo_bits: 6 }, die, sampler: SamplerConfig { logit_bits: 16, index_bits: bits_for(program.vocab as u64).max(4), entries: 4, lanes: 1 } };
    // A board on its own needs neither the manufacturing tests nor the chain to other chips;
    // `--chain` keeps them, for boards joined to each other.
    let design = if args.iter().any(|a| a == "--chain") { chip_design(config) } else { chip_design_alone(config) };
    eprintln!("{:.0} equivalentes NAND2, microprograma de {} palabras", design.gate_equivalents(), code.words.len());
    let mut memories: Vec<_> = memory_bits(&design).into_iter().collect();
    memories.sort();
    let total: u64 = memories.iter().map(|(_, bits)| bits).sum();
    eprintln!("memoria: {:.2} Mbit ({})", total as f64 / 1e6, memories.iter().map(|(kind, bits)| format!("{kind} {:.2}", *bits as f64 / 1e6)).collect::<Vec<_>>().join(", "));
    if args.iter().any(|a| a == "--memories") {
        for (name, count, addr, width) in memory_list(&design) {
            eprintln!("  {count} × {name}: 2^{addr} × {width} bits = {:.3} Mbit", (count * (1u64 << addr) * width as u64) as f64 / 1e6);
        }
    }
    // The die's cycles for one generated token at the last position, with its logits.
    let last = context as u32 - 1;
    let token = code.cycles(&die, sim::lower::ENTRY_TOKEN, 1, last) + code.cycles(&die, sim::lower::ENTRY_LOGITS, 1, last);
    eprintln!("{token} ciclos del die por token generado en la posición {last}");
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
            eprintln!("  {g:8} puertas {f:7} biestables  {group}");
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
            eprintln!("  {depth:4.0} {name}{}", if hidden > 0 { format!("  (+{hidden} sin nombre)") } else { String::new() });
            hidden = 0;
        }
    }
    let text = if args.iter().any(|a| a == "--dsp") { to_verilog_for_fpga(&design) } else { to_verilog(&design) };
    std::fs::write(out, text).unwrap();
    std::fs::remove_dir_all(test_dir).ok();
}
