//! The search for the best set of chips, on the tiny test model with small dies.

use compiler::compile_qwen3_5;
use sim::optimize::{optimize, Goal, Model, Report, Request, Silicon, Space};
use sim::sizing::{reference_die, Costs, N6};
use sim::Weights;
use std::path::Path;
use std::sync::atomic::AtomicBool;

mod common;
use common::*;

fn search(dir: &Path, request: &Request, cache: Option<&Path>) -> Report {
    let program = compile_qwen3_5(CONFIG).unwrap();
    let weights = Weights::open(dir).unwrap();
    let graph = model_ir::ModelGraph::from_config_json("tiny", CONFIG).ok();
    let model = Model { program: &program, weights: &weights, graph: graph.as_ref(), cache: cache.map(Path::to_path_buf) };
    let base = reference_die(1);
    let silicon = Silicon { base, costs: Costs::measure(base), process: N6 };
    optimize(&model, &silicon, request, &AtomicBool::new(false), &|_| {}).unwrap()
}

fn small(goal: Goal) -> Request {
    let space = Space { chips: vec![1, 2], sharing: "layers,rows".into(), cluster_bits: (0, 2), lane_bits: (1, 2), cell_bits: (3, 8) };
    Request { goal, space, context: 64, prompt: 4, ..Default::default() }
}

fn checkpoint(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("llm2chip-optimize-{name}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    make_checkpoint(&dir);
    dir
}

#[test]
fn the_least_silicon_that_reaches_a_speed_reaches_it_and_nothing_smaller_does() {
    let dir = checkpoint("area");
    let fastest = search(&dir, &small(Goal::Speed), None).recommended.expect("a design");
    let mut request = small(Goal::Area);
    request.limits.min_tok_s = fastest.candidate.tokens_per_second / 2.0;
    let report = search(&dir, &request, None);
    let chosen = report.recommended.expect("a design that reaches half the top speed");
    assert!(chosen.candidate.tokens_per_second >= request.limits.min_tok_s);
    assert!(chosen.candidate.area_mm2 <= fastest.candidate.area_mm2);
    for d in report.frontier.iter().filter(|d| d.candidate.area_mm2 < chosen.candidate.area_mm2) {
        assert!(d.candidate.tokens_per_second < request.limits.min_tok_s, "{:?} is smaller and fast enough", d.candidate);
    }
    std::fs::remove_dir_all(dir).ok();
}

#[test]
fn the_fastest_design_keeps_to_the_silicon_it_is_given() {
    let dir = checkpoint("speed");
    let free = search(&dir, &small(Goal::Speed), None).recommended.expect("a design");
    let mut request = small(Goal::Speed);
    request.limits.max_area_mm2 = free.candidate.area_mm2 / 2.0;
    let chosen = search(&dir, &request, None).recommended.expect("a design in half the silicon");
    assert!(chosen.candidate.area_mm2 <= request.limits.max_area_mm2);
    assert!(chosen.candidate.tokens_per_second <= free.candidate.tokens_per_second);
    std::fs::remove_dir_all(dir).ok();
}

#[test]
fn asking_again_measures_nothing_new() {
    let dir = checkpoint("cache");
    let cache = dir.join("sizing.json");
    let first = search(&dir, &small(Goal::Efficiency), Some(&cache));
    assert!(first.measured > 0 && first.from_cache == 0);
    let again = search(&dir, &small(Goal::Efficiency), Some(&cache));
    assert_eq!(again.measured, 0, "everything was measured before");
    assert!(again.from_cache > 0);
    // The same design: what the disk keeps may differ in a number's last bit.
    let shape = |r: &Report| r.recommended.as_ref().map(|d| (d.candidate.chips, d.candidate.sharing, d.candidate.cluster_bits, d.candidate.lane_bits, d.candidate.cell_bits));
    assert_eq!(shape(&again), shape(&first));
    std::fs::remove_dir_all(dir).ok();
}

#[test]
fn a_search_uses_only_what_answers_its_own_question() {
    let dir = checkpoint("scope");
    let cache = dir.join("sizing.json");
    let shape = |r: &Report| r.recommended.as_ref().map(|d| (d.candidate.chips, d.candidate.sharing, d.candidate.cluster_bits, d.candidate.lane_bits, d.candidate.cell_bits));
    // A longer context first, kept; then a shorter one, with the same file and without one.
    search(&dir, &small(Goal::Speed), Some(&cache));
    let mut shorter = small(Goal::Speed);
    shorter.context = 32;
    let kept = search(&dir, &shorter, Some(&cache));
    let fresh = search(&dir, &shorter, None);
    assert_eq!(shape(&kept), shape(&fresh));
    assert_eq!(kept.frontier.len(), fresh.frontier.len(), "nothing from the other context");
    // A narrower space only recommends from within it.
    let mut narrow = small(Goal::Speed);
    narrow.space.chips = vec![1];
    narrow.space.cell_bits = (3, 5);
    let report = search(&dir, &narrow, Some(&cache));
    assert!(report.frontier.iter().all(|d| d.candidate.chips == 1 && d.candidate.cell_bits <= 5));
    std::fs::remove_dir_all(dir).ok();
}
