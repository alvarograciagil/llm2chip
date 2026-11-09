//! Writes a cell's Verilog to stdout.
//! `cargo run -p netlist --example export -- [fixed|affine|float|posit|log] [bits] [exponent] [array|wallace|booth]`

use netlist::format::WeightCoding;
use netlist::multiplier::MultiplierKind;
use netlist::{cell_design, to_verilog, CellConfig, Simulator};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let coding = match args.get(1).map(String::as_str) {
        Some("affine") => WeightCoding::Affine,
        Some("float") => WeightCoding::Float,
        Some("posit") => WeightCoding::Posit,
        Some("log") => WeightCoding::Log,
        _ => WeightCoding::Fixed,
    };
    let weight_bits = args.get(2).map_or(8, |v| v.parse().expect("bits"));
    let exponent_bits = args.get(3).map_or(0, |v| v.parse().expect("exponente"));
    let multiplier = match args.get(4).map(String::as_str) {
        Some("wallace") => MultiplierKind::Wallace,
        Some("booth") => MultiplierKind::Booth,
        Some("multiples") => MultiplierKind::Multiples,
        _ => MultiplierKind::Array,
    };
    let mut config = CellConfig { weight_bits, activation_bits: 16, accumulator_bits: 32, mac_stages: 1, adder_stages: 1, addr_bits: 2, coding, exponent_bits, multiplier, fixed_weights: false, ecc: false, skip_zeros: false };
    config.accumulator_bits = config.required_accumulator().max(32);
    let design = cell_design(config);
    let path = Simulator::new(&design).expect("simulador").critical_path([0.25, 0.25, 0.5, 0.12, 0.4]);
    eprintln!("{}", serde_json::json!({ "counts": design.flat_counts(), "path": path }));
    print!("{}", to_verilog(&design));
}
