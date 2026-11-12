//! The estimator against the layouts OpenROAD measured (tests/golden/asic.json, written by
//! tools/asic/calibrate.sh): what it expects of each cell laid out on its own.

use arch::{laid_out_cell, node};
use netlist::CellConfig;

#[test]
fn the_estimate_follows_the_measured_layouts() {
    let golden: serde_json::Value = serde_json::from_str(include_str!("../../../tests/golden/asic.json")).unwrap();
    let measures = golden["measures"].as_array().unwrap();
    assert!(measures.len() >= 15, "{} medidas", measures.len());
    for m in measures {
        let platform = m["platform"].as_str().unwrap();
        let tech = node(if platform == "sky130hd" { "sky130" } else { platform }).unwrap();
        let config: CellConfig = serde_json::from_value(m["config"].clone()).unwrap();
        let (area, fmax_ghz) = laid_out_cell(tech, config);
        let area_ratio = area / m["area_um2"].as_f64().unwrap();
        let clock_ratio = fmax_ghz * 1000.0 / m["fmax_mhz"].as_f64().unwrap();
        let name = format!("{} en {platform}", m["cell"]);
        // The clock within 15 %; the area within the spread of synthesis, worst for floats.
        assert!((0.85..1.15).contains(&clock_ratio), "{name}: reloj {clock_ratio:.2}");
        assert!((0.6..1.25).contains(&area_ratio), "{name}: área {area_ratio:.2}");
        // No process caps its clock below what its cells reached.
        assert!(tech.fmax_ghz * 1000.0 >= m["fmax_mhz"].as_f64().unwrap(), "{name}: tope {}", tech.fmax_ghz);
    }
}

#[test]
fn the_estimate_follows_the_tile_with_a_real_sram_macro() {
    let golden: serde_json::Value = serde_json::from_str(include_str!("../../../tests/golden/asic.json")).unwrap();
    let tile = &golden["tile"];
    let tech = node("sky130").unwrap();
    // 32 Kbit in one macro: the estimator's density for a macro that size.
    let macro_um2 = 32.0 * 1024.0 / tech.sram_density(32.0);
    let measured = tile["macro_um2"].as_f64().unwrap();
    assert!((macro_um2 / measured - 1.0).abs() < 0.05, "{macro_um2} frente a {measured}");
    // The macro, not the cell's logic, sets the clock.
    let ghz = 1.0 / tech.sram_min_period_ns.unwrap();
    assert!((ghz * 1000.0 / tile["fmax_mhz"].as_f64().unwrap() - 1.0).abs() < 0.01);
}
