//! The set of chips that does best what is asked of it for a model, each design measured
//! on its own generated circuit.
//!
//! `cargo run --release -p sim --example optimize -- MODEL_DIR [options]`
//!
//! - `--goal speed|area|efficiency|energy`: what to make best (speed by default).
//! - `--min-tok-s N`, `--max-first-token-ms N`, `--max-chips N`, `--max-area N` (mm²),
//!   `--max-power N` (W): limits every design must meet.
//! - `--chips 8,12,16`, `--sharing layers,rows,mixed:auto`, `--units 5..11`, `--lanes 6..7`,
//!   `--cells 13..24` (log2): what may be tried.
//! - `--context N`, `--prompt N`, `--max-ghz N`, `--thorough`, and `--factor-steps N`, the
//!   cycles each cell takes to apply a group's scale and bias (8 by default; a model with
//!   smaller groups needs fewer).
//!
//! Measurements are kept in `~/.llm2chip/cache`, so asking again measures only what is new.

use sim::optimize::{cache_path, optimize, Design, Effort, Goal, Model, Request, Silicon};
use sim::sizing::{reference_die, Costs, N6};
use sim::Weights;
use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = Path::new(args.get(1).expect("model folder"));
    let mut request = Request::default();
    let mut factor_steps = 8;
    let mut i = 2;
    while i < args.len() {
        let flag = args[i].as_str();
        if flag == "--thorough" {
            request.effort = Effort::Thorough;
            i += 1;
            continue;
        }
        let value = args.get(i + 1).unwrap_or_else(|| panic!("{flag} needs a value")).as_str();
        let number = || value.parse::<f64>().unwrap_or_else(|_| panic!("{flag} takes a number"));
        let span = || match value.split_once("..") {
            Some((a, b)) => (a.parse().expect(flag), b.parse().expect(flag)),
            None => (value.parse().expect(flag), value.parse().expect(flag)),
        };
        match flag {
            "--goal" => request.goal = serde_json::from_value(serde_json::Value::String(value.into())).expect("speed, area, efficiency or energy"),
            "--min-tok-s" => request.limits.min_tok_s = number(),
            "--max-first-token-ms" => request.limits.max_first_token_ms = number(),
            "--max-chips" => request.limits.max_chips = number() as u32,
            "--max-area" => request.limits.max_area_mm2 = number(),
            "--max-power" => request.limits.max_power_w = number(),
            "--chips" => request.space.chips = value.split(',').map(|n| n.parse().expect("--chips")).collect(),
            "--sharing" => request.space.sharing = value.into(),
            "--units" => request.space.cluster_bits = span(),
            "--lanes" => request.space.lane_bits = span(),
            "--cells" => request.space.cell_bits = span(),
            "--context" => request.context = number() as u32,
            "--prompt" => request.prompt = number() as u32,
            "--max-ghz" => request.max_clock_ghz = number(),
            "--factor-steps" => factor_steps = number() as u32,
            _ => panic!("unknown option {flag}"),
        }
        i += 2;
    }
    let config = std::fs::read_to_string(dir.join("config.json")).expect("config.json");
    let program = compiler::compile_model(&config).unwrap();
    let weights = Weights::open(dir).unwrap();
    let name = dir.file_name().map_or("model".into(), |n| n.to_string_lossy().into_owned());
    let graph = model_ir::ModelGraph::from_config_json(&name, &config).ok();
    let home = std::env::var("HOME").map(std::path::PathBuf::from).unwrap_or_default();
    let model = Model { program: &program, weights: &weights, graph: graph.as_ref(), cache: Some(cache_path(&home.join(".llm2chip/cache"), &config)) };
    let base = reference_die(factor_steps);
    let silicon = Silicon { base, costs: Costs::measure(base), process: N6 };
    let cancel = std::sync::atomic::AtomicBool::new(false);
    let last = std::sync::Mutex::new((String::new(), 0usize));
    let report = optimize(&model, &silicon, &request, &cancel, &|p| {
        let mut last = last.lock().unwrap();
        if last.0 != p.phase || p.done >= last.1 + 10 {
            let best = p.best.as_ref().map_or(String::new(), |d| format!(", best so far {}", line(d)));
            eprintln!("{}: {} of about {} measured{best}", p.phase, p.done, p.planned);
            *last = (p.phase.to_string(), p.done);
        }
    })
    .unwrap_or_else(|e| panic!("{e}"));
    let goal = match report.goal {
        Goal::Speed => "speed",
        Goal::Area => "area",
        Goal::Efficiency => "area per tok/s",
        Goal::Energy => "energy per token",
    };
    println!("{} designs measured, {} taken from earlier searches", report.measured, report.from_cache);
    if !report.too_few_chips.is_empty() {
        println!("too few chips to hold the model: {:?}", report.too_few_chips);
    }
    match &report.recommended {
        Some(d) => {
            println!("recommended for {goal}: {}", line(d));
            if !report.limiting.is_empty() {
                println!("  held back by: {}", report.limiting.join(", "));
            }
        }
        None => match &report.failure {
            Some(why) => println!("no design in the space meets every limit; one could not be built: {why}"),
            None => println!("no design in the space meets every limit"),
        },
    }
    for a in &report.alternatives {
        println!("{}: {}", a.label, line(&a.design));
        if let Some(r) = &report.recommended {
            let (speed, area) = (a.design.candidate.tokens_per_second / r.candidate.tokens_per_second - 1.0, a.design.candidate.area_mm2 / r.candidate.area_mm2 - 1.0);
            println!("  {:+.0} % tok/s for {:+.0} % silicon against the recommended", speed * 100.0, area * 100.0);
        }
    }
    println!("frontier of what was measured:");
    for d in &report.frontier {
        println!("  {}", line(d));
    }
}

fn line(d: &Design) -> String {
    let c = d.candidate;
    let energy = d.energy_mj.map_or(String::new(), |e| format!(", ~{e:.1} mJ/token"));
    let power = d.power_w.map_or(String::new(), |w| format!(", ~{w:.0} W"));
    format!("{} chips {}, {} x {} units, {} cells: {:.0} tok/s at {:.0} MHz, first token {:.0} ms, {:.0} mm² ({:.0} per chip), {:.3} mm² per tok/s{energy}{power}", c.chips, d.sharing_name, 1u64 << c.cluster_bits, 1u64 << c.lane_bits, 1u64 << c.cell_bits, c.tokens_per_second, c.clock_hz / 1e6, c.prompt_seconds * 1e3, c.area_mm2, c.die_mm2, d.area_per_tok_s)
}
