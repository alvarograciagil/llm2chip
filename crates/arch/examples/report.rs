use arch::{estimate, ChipParams};
use model_ir::ModelGraph;

fn main() {
    let model = ModelGraph::from_config_json("Qwen3.8-27B", include_str!("../../../models/qwen3.8-27b.json")).unwrap();
    let r = estimate(&model, &ChipParams::default()).unwrap();
    for m in &r.metrics { println!("{:<16} {:<40} {:>14.3} {}", m.section, m.label, m.value, m.unit); }
    println!();
    for b in &r.blocks { println!("{:<36} {:>9.2} mm²", b.label, b.per_die_mm2); }
    for v in &r.violations { println!("{:?}: {}", v.severity, v.message); }
}
