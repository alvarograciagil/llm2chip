//! Prints switching per multiplication for reference cells.

use netlist::format::{switching_per_mac, WeightCoding};
use netlist::multiplier::MultiplierKind;
use netlist::CellConfig;

fn main() {
    for (kind, coding, bits, exponent) in [
        (MultiplierKind::Array, WeightCoding::Fixed, 8, 0),
        (MultiplierKind::Wallace, WeightCoding::Fixed, 4, 0),
        (MultiplierKind::Wallace, WeightCoding::Fixed, 8, 0),
        (MultiplierKind::Booth, WeightCoding::Fixed, 8, 0),
        (MultiplierKind::Wallace, WeightCoding::Fixed, 16, 0),
        (MultiplierKind::Wallace, WeightCoding::Float, 16, 5),
        (MultiplierKind::Wallace, WeightCoding::Float, 32, 8),
        (MultiplierKind::Wallace, WeightCoding::Posit, 16, 1),
        (MultiplierKind::Wallace, WeightCoding::Log, 8, 4),
        (MultiplierKind::Wallace, WeightCoding::Codebook, 4, 0),
    ] {
        let mut c = CellConfig { weight_bits: bits, activation_bits: 16, accumulator_bits: 32, mac_stages: 2, adder_stages: 2, addr_bits: 3, coding, exponent_bits: exponent, multiplier: kind, fixed_weights: false, ecc: false, skip_zeros: false };
        c.accumulator_bits = c.accumulator_bits.max(c.required_accumulator());
        println!("{kind:?} {coding:?} {bits}: {:.1}", switching_per_mac(c, 0.0));
    }
}
