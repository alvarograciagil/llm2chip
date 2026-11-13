//! The slowest combinational path of dies of several sizes, in full-adder delays.
use netlist::multiplier::MultiplierKind;
use netlist::{die_design, CellConfig, DieConfig, DieFeatures, Simulator};
fn main() {
    let cell = CellConfig { weight_bits: 8, activation_bits: 16, accumulator_bits: 36, mac_stages: 1, adder_stages: 1, addr_bits: 12, coding: Default::default(), exponent_bits: 0, multiplier: match std::env::var("MULT").as_deref() { Ok("wallace") => MultiplierKind::Wallace, Ok("booth") => MultiplierKind::Booth, _ => MultiplierKind::Array }, fixed_weights: false, ecc: false, skip_zeros: false };
    let base = DieConfig { cells: 1, cell, scale_bits: 21, reg_bits: 12, index_bits: 16, microcode_bits: 16, broadcast_stages: 1, function_bits: 12, function_select_bits: 3, features: DieFeatures::default().with_factor_steps(8) };
    for (units, lanes) in [(0, 2)] {
        let c = DieConfig { cells: 1 << (units + lanes), features: base.features.with_lane_bits(lanes).with_cluster_bits(units), ..base };
        let path = Simulator::new(&die_design(c)).unwrap().critical_path([0.25, 0.25, 0.5, 0.12, 0.4]);
        if std::env::var("TRACE").is_ok() {
            let trace = Simulator::with_names(&die_design(c)).unwrap().critical_trace([0.25, 0.25, 0.5, 0.12, 0.4]);
            let mut hidden = 0;
            for (net, time) in trace {
                if net.starts_with('#') { hidden += 1; if hidden % 10 == 1 { println!("   {time:6.1} {net}"); } continue; }
                println!("   {time:6.1} {net}  (+{hidden} unnamed)");
                hidden = 0;
            }
        }
        println!("{} unidades x {} carriles: {path:.1} sumadores completos, {:.2} ns en N6", 1 << units, 1 << lanes, path * 0.04);
    }
}
