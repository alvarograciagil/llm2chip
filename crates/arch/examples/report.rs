//! Prints the estimate of a design: `cargo run -p arch --example report -- [model.json] ['{"node": "sky130", ...}']`.

use arch::{estimate, ChipParams};
use model_ir::ModelGraph;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).map_or("models/qwen3.8-27b.json", String::as_str);
    let config = std::fs::read_to_string(path).expect("modelo");
    let model = ModelGraph::from_config_json(path, &config).unwrap();
    let mut params = serde_json::to_value(ChipParams::default()).unwrap();
    if let Some(patch) = args.get(2) {
        for (key, value) in serde_json::from_str::<serde_json::Map<String, serde_json::Value>>(patch).expect("parámetros") {
            params[key] = value;
        }
    }
    let p: ChipParams = serde_json::from_value(params).unwrap();
    let r = estimate(&model, &p).unwrap();
    for m in &r.metrics { println!("{:<16} {:<40} {:>14.3} {}", m.section, m.label, m.value, m.unit); }
    println!();
    println!("{} dies de {:.2} mm²", r.dies, r.die_area_mm2);
    for b in &r.blocks { println!("{:<36} {:>9.2} mm²", b.label, b.per_die_mm2); }
    for v in &r.violations { println!("{:?}: {}", v.severity, v.message); }
    println!();
    println!("{}", serde_json::to_string_pretty(&arch::feasibility::advise(&model, &p, true).unwrap()).unwrap());
}
