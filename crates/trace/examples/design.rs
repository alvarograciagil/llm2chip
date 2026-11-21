//! Traces a whole design operation by operation, from each chip's microprogram, for the
//! 3D viewer: a token at the start, the middle and the end of the context, and its logits.
//!
//!   cargo run --release -p chip-trace --example design -- MODEL_DIR [file]
//!
//! The design comes from CHIPS (16), SHARING (mixed:8), UNITS, LANES and CELLS as log2
//! (7, 7, 16), CONTEXT (8192) and FACTOR_STEPS (8).

use chip_trace::design::{design_trace, Design};
use sim::Weights;
use std::path::Path;

fn number(name: &str, default: u32) -> u32 {
    std::env::var(name).ok().map_or(default, |v| v.parse().unwrap_or_else(|_| panic!("{name}")))
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let dir = Path::new(args.first().expect("model folder"));
    let file = args.get(1).cloned().unwrap_or_else(|| "design.json".into());
    let chips = number("CHIPS", 16);
    let sharing = sim::sizing::parse_sharings(&std::env::var("SHARING").unwrap_or_else(|_| "mixed:8".into()), &[chips]).expect("sharing")[0];
    let design = Design { chips, sharing, cluster_bits: number("UNITS", 7), lane_bits: number("LANES", 7), cell_bits: number("CELLS", 16), context: number("CONTEXT", 8192), factor_steps: number("FACTOR_STEPS", 8) };
    let program = compiler::compile_model(&std::fs::read_to_string(dir.join("config.json")).expect("config.json")).expect("the model");
    let weights = Weights::open(dir).expect("the weights");
    let started = std::time::Instant::now();
    let last = design.context - 1;
    let trace = design_trace("design", &program, &weights, design, &[0, last / 2, last]).expect("the trace");
    let json = serde_json::to_string(&trace).unwrap();
    std::fs::write(&file, &json).unwrap();
    for s in &trace.steps {
        println!("{} at {}: {} cycles", s.routine, s.position, s.cycles);
    }
    println!("{file}: {} KB, {:.1} s", json.len() / 1024, started.elapsed().as_secs_f64());
}
