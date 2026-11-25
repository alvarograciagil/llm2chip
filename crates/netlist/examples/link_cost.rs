//! What each host link adds to the control system, in gates and flip-flops.
//!
//! `cargo run --release -p netlist --example link_cost`

use netlist::soc::{soc_design, HostLinks, SocConfig};

fn main() {
    let count = |links: HostLinks| {
        let design = soc_design(SocConfig { program_words: 1 << 10, data_bits: 10, fifo_bits: 6, links });
        let counts = design.flat_counts();
        let gates: u64 = ["and", "or", "xor", "not", "mux"].iter().map(|g| counts.get(*g).copied().unwrap_or(0)).sum();
        (gates as i64, counts.get("dff").copied().unwrap_or(0) as i64)
    };
    let base = count(HostLinks::default());
    for (name, links) in [("serial port, 333 clocks a bit", HostLinks { uart: Some(333), qspi: false }), ("quad SPI", HostLinks { uart: None, qspi: true })] {
        let (gates, flops) = count(links);
        println!("{name}: {} gates, {} flip-flops more", gates - base.0, flops - base.1);
    }
}
