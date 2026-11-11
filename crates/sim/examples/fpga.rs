//! The whole chip for the small model of the tests, in Verilog, to synthesise for an FPGA:
//! the RISC-V controller, the die with only the logic its microprogram uses, and the
//! sampling unit.
//!
//! `cargo run --release -p sim --example fpga -- OUT.v [--dsp]`
//!
//! With `--dsp`, each one-cycle multiplier is written as a signed product so the FPGA
//! tools put it in their multiplier blocks.
//!
//! The die is the one `the_chip_sized_for_an_fpga_says_the_same_in_gates` runs.

#[path = "../tests/common/mod.rs"]
mod common;

use common::*;
use compiler::compile_qwen3_5;
use netlist::chip::{chip_design, ChipConfig};
use netlist::sampler::SamplerConfig;
use netlist::soc::SocConfig;
use netlist::{to_verilog, to_verilog_for_fpga, DieConfig, DieFeatures};
use sim::lower::Layout;
use sim::Weights;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let out = args.get(1).expect("archivo de salida");
    let dir = std::env::temp_dir().join(format!("llm2chip-fpga-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    make_checkpoint(&dir);
    let program = compile_qwen3_5(CONFIG).unwrap();
    let trial = fpga_die_config();
    let layout = Layout::with_cells(&program, &Weights::open(&dir).unwrap(), trial, 8).unwrap();
    let code = layout.microprogram(&program).unwrap();
    let (depth, groups) = layout.cell_depths();
    eprintln!("{} registros, {} palabras de pesos y {} grupos de factores por celda", layout.registers_used(), depth, groups);
    // Only what this microprogram uses.
    let features = DieFeatures::of(&code).with_lane_bits(trial.features.lane_bits).with_cluster_bits(trial.features.cluster_bits).with_key_shift(trial.features.key_shift);
    let die = DieConfig { features, ..trial };
    let config = ChipConfig { soc: SocConfig { program_bits: 13, data_bits: 14, fifo_bits: 6 }, die, sampler: SamplerConfig { logit_bits: 16, index_bits: 12, entries: 4, lanes: 1 } };
    let design = chip_design(config);
    eprintln!("{:.0} equivalentes NAND2, microprograma de {} palabras", design.gate_equivalents(), code.words.len());
    let text = if args.iter().any(|a| a == "--dsp") { to_verilog_for_fpga(&design) } else { to_verilog(&design) };
    std::fs::write(out, text).unwrap();
    std::fs::remove_dir_all(dir).ok();
}
