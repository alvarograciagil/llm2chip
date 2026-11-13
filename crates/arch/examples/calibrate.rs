//! Compares the estimator with the layouts of tests/golden/asic.json and fits each process's
//! full-adder delay to them: `cargo run -p arch --example calibrate`.

use arch::{laid_out_cell, node};
use netlist::CellConfig;
use std::collections::BTreeMap;

/// The estimator's process for an OpenROAD platform.
pub fn node_of(platform: &str) -> &str {
    if platform == "sky130hd" { "sky130" } else { platform }
}

fn main() {
    let golden: serde_json::Value = serde_json::from_str(include_str!("../../../tests/golden/asic.json")).expect("asic.json");
    let mut fits: BTreeMap<String, Vec<(f64, f64, f64)>> = BTreeMap::new();
    println!("| Cell | Process | Measured area µm² | Estimated | Ratio | Measured clock MHz | Estimated | Ratio |");
    println!("|---|---|---|---|---|---|---|---|");
    for m in golden["measures"].as_array().expect("medidas") {
        let platform = m["platform"].as_str().unwrap();
        let tech = node(node_of(platform)).expect("technology");
        let config: CellConfig = serde_json::from_value(m["config"].clone()).expect("cell");
        let (area, fmax_ghz) = laid_out_cell(tech, config);
        let (measured_area, measured_mhz) = (m["area_um2"].as_f64().unwrap(), m["fmax_mhz"].as_f64().unwrap_or(0.0));
        println!(
            "| {} | {platform} | {measured_area:.0} | {area:.0} | {:.2} | {measured_mhz:.0} | {:.0} | {:.2} |",
            m["cell"].as_str().unwrap(),
            area / measured_area,
            fmax_ghz * 1000.0,
            fmax_ghz * 1000.0 / measured_mhz
        );
        // The path in full adders, to fit: period = path · delay + overhead.
        let path = (1.0 / fmax_ghz - tech.ff_overhead_ns) / tech.fa_delay_ns;
        fits.entry(tech.id.clone()).or_default().push((path, 1000.0 / measured_mhz, area / measured_area));
    }
    for (id, points) in fits {
        let tech = node(&id).unwrap();
        // Least squares of period − overhead against the path, through the origin.
        let delay = points.iter().map(|(x, y, _)| x * (y - tech.ff_overhead_ns)).sum::<f64>() / points.iter().map(|(x, _, _)| x * x).sum::<f64>();
        let area = points.iter().map(|(_, _, r)| r.ln()).sum::<f64>() / points.len() as f64;
        // And with the flip-flop overhead free too: period = a · path + b.
        let n = points.len() as f64;
        let (mx, my) = (points.iter().map(|p| p.0).sum::<f64>() / n, points.iter().map(|p| p.1).sum::<f64>() / n);
        let slope = points.iter().map(|(x, y, _)| (x - mx) * (y - my)).sum::<f64>() / points.iter().map(|(x, _, _)| (x - mx).powi(2)).sum::<f64>();
        println!("{id}: with both free, delay {slope:.3} ns and flip-flop {:.3} ns (now {:.3})", my - slope * mx, tech.ff_overhead_ns);
        println!("{id}: fitted adder delay {delay:.3} ns (now {:.3}); estimated / measured area, geometric mean {:.2}", tech.fa_delay_ns, area.exp());
    }
}
