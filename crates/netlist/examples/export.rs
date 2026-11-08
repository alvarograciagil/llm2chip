//! Writes a cell's Verilog to stdout.
//! `cargo run -p netlist --example export -- [fixed|affine|float|posit|log] [bits] [exponent]`

use netlist::format::WeightCoding;
use netlist::{cell_design, to_verilog, CellConfig};

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
    let mut config = CellConfig { weight_bits, activation_bits: 16, accumulator_bits: 32, mac_stages: 1, adder_stages: 1, addr_bits: 2, coding, exponent_bits };
    config.accumulator_bits = config.required_accumulator().max(32);
    print!("{}", to_verilog(&cell_design(config)));
}
