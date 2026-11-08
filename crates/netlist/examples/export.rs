//! Writes the cell's Verilog to stdout: `cargo run -p netlist --example export`.

use netlist::{cell_design, to_verilog, CellConfig};

fn main() {
    let config = CellConfig { weight_bits: 8, activation_bits: 16, accumulator_bits: 32, mac_stages: 2, adder_stages: 2, addr_bits: 15, coding: Default::default(), exponent_bits: 0 };
    print!("{}", to_verilog(&cell_design(config)));
}
