//! The whole chip for the small model of the tests, in Verilog, to synthesise for an FPGA:
//! the RISC-V controller, the die with only the logic its microprogram uses, and the
//! sampling unit.
//!
//! `cargo run --release -p sim --example fpga -- OUT.v [CELLS] [LANE_BITS] [CLUSTER_BITS]`

#[path = "../tests/common/mod.rs"]
mod common;

use common::*;
use compiler::compile_qwen3_5;
use netlist::chip::{chip_design, ChipConfig};
use netlist::sampler::SamplerConfig;
use netlist::soc::SocConfig;
use netlist::{to_verilog, DieConfig, DieFeatures};
use sim::lower::Layout;
use sim::Weights;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let out = args.get(1).expect("archivo de salida");
    let number = |at: usize, default: u32| args.get(at).map_or(default, |a| a.parse().expect("un número"));
    let (cells, lane_bits, cluster_bits) = (number(2, 1), number(3, 0), number(4, 0));
    let dir = std::env::temp_dir().join(format!("llm2chip-fpga-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    make_checkpoint(&dir);
    let program = compile_qwen3_5(CONFIG).unwrap();
    let base = cell_die_config();
    let trial = DieConfig { cells, microcode_bits: 15, features: base.features.with_lane_bits(lane_bits).with_cluster_bits(cluster_bits), ..base };
    let layout = Layout::with_cells(&program, &Weights::open(&dir).unwrap(), trial, 16).unwrap();
    let code = layout.microprogram(&program).unwrap();
    // Only what this microprogram uses.
    let features = DieFeatures::of(&code).with_lane_bits(lane_bits).with_cluster_bits(cluster_bits).with_key_shift(trial.features.key_shift);
    let die = DieConfig { features, ..trial };
    let config = ChipConfig { soc: SocConfig { program_bits: 13, data_bits: 14, fifo_bits: 6 }, die, sampler: SamplerConfig { logit_bits: 16, index_bits: 12, entries: 4, lanes: 1 } };
    let design = chip_design(config);
    eprintln!("{:.0} equivalentes NAND2, microprograma de {} palabras", design.gate_equivalents(), code.words.len());
    std::fs::write(out, to_verilog(&design)).unwrap();
    std::fs::remove_dir_all(dir).ok();
}
