//! Records a tiny model running on joined dies, gate by gate, into a trace for the 3D
//! viewer.
//!
//!   cargo run --release -p chip-trace --example trace -- [file] [chips] [sharing] [model]
//!
//! `sharing` is layers, rows or mixed:N (mixed:2 by default, on 4 chips); `model` plain or
//! grouped (grouped by default). The trace goes to `trace.json` unless a file is given.

use chip_trace::{sharing_label, tiny_board, TinyDies, TinyModel};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let file = args.first().cloned().unwrap_or_else(|| "trace.json".into());
    let chips: usize = args.get(1).map_or(4, |c| c.parse().expect("chips"));
    let sharing = sim::sizing::parse_sharings(args.get(2).map_or("mixed:2", String::as_str), &[chips as u32]).expect("sharing")[0];
    let model = match args.get(3).map_or("grouped", String::as_str) {
        "plain" => TinyModel::Plain,
        "grouped" => TinyModel::Grouped,
        other => panic!("{other} is not plain or grouped"),
    };
    let dir = std::env::temp_dir().join(format!("llm2chip-trace-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    model.write(&dir);
    let mut board = tiny_board(model, &dir, chips, sharing, TinyDies::default()).expect("the board");
    let title = format!("Tiny model on {chips} dies, {}", sharing_label(board.sharing));
    let started = std::time::Instant::now();
    let trace = board.record(&title, &[3, 5, 1]).expect("the trace");
    std::fs::remove_dir_all(&dir).ok();
    let cycles: u64 = trace.steps.iter().map(|s| s.cycles).sum();
    let stalled: Vec<u64> = trace.dies.iter().map(|d| d.stalls.iter().map(|s| s.1).sum()).collect();
    let json = serde_json::to_string(&trace).unwrap();
    std::fs::write(&file, &json).unwrap();
    println!("{file}: {cycles} cycles, {} steps, stalled per die {stalled:?}, {} KB, {:.1} s", trace.steps.len(), json.len() / 1024, started.elapsed().as_secs_f64());
}
