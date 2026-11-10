//! Gate equivalents of the scaled cell, by part.
use netlist::multiplier::MultiplierKind;
use netlist::{die_design, CellConfig, DieConfig, DieFeatures, Design, InstKind, Prim};
fn ge_prim(p: &Prim) -> f64 { match p { Prim::And2 | Prim::Or2 => 1.33, Prim::Not => 0.67, Prim::Xor2 => 2.0, Prim::Mux2 => 2.33, Prim::Dff => 4.5, _ => 0.0 } }
fn ge(design: &Design, module: &str) -> f64 {
    design.modules[module].instances.iter().map(|i| match &i.kind { InstKind::Prim(p) => ge_prim(p), InstKind::Module(m) => ge(design, m) }).sum()
}
fn main() {
    let k = std::env::var("K").ok().and_then(|k| k.parse().ok()).unwrap_or(8);
    let cell = CellConfig { weight_bits: 8, activation_bits: 16, accumulator_bits: 36, mac_stages: 1, adder_stages: 1, addr_bits: 24, coding: Default::default(), exponent_bits: 0, multiplier: match std::env::var("MULT").as_deref() { Ok("wallace") => MultiplierKind::Wallace, Ok("booth") => MultiplierKind::Booth, _ => MultiplierKind::Array }, fixed_weights: false, ecc: false, skip_zeros: false };
    let c = DieConfig { cells: 1, cell, scale_bits: 21, reg_bits: 16, index_bits: 24, microcode_bits: 24, broadcast_stages: 1, function_bits: 12, function_select_bits: 3, features: DieFeatures::default().with_factor_steps(k) };
    let d = die_design(c);
    for module in ["scaled_cell", "compute_cell"] {
        println!("{module} {:.0}", ge(&d, module));
        let mut by: std::collections::BTreeMap<String, f64> = Default::default();
        for i in &d.modules[module].instances {
            let (key, value) = match &i.kind { InstKind::Module(n) => (n.clone(), ge(&d, n)), InstKind::Prim(p) => (format!("{p:?}"), ge_prim(p)) };
            *by.entry(key).or_default() += value;
        }
        let mut v: Vec<_> = by.into_iter().collect();
        v.sort_by(|a, b| b.1.total_cmp(&a.1));
        for (key, g) in v.iter().take(8) { println!("  {g:>8.0} {key}"); }
    }
}
