//! Recording a small model on dies joined by their links, gate by gate, for the 3D viewer,
//! each routine checked against the exact machine.

use chip_trace::design::{design_trace, Design};
use chip_trace::{tiny_board, TinyDies, TinyModel, TINY_CONTEXT};
use serde::Deserialize;
use std::path::Path;
use std::sync::Mutex;

/// The most dies a recording may join: each is simulated gate by gate.
const MOST_CHIPS: usize = 8;

#[derive(Deserialize)]
pub struct TraceStart {
    /// "grouped" or "plain".
    #[serde(default)]
    pub model: String,
    pub chips: usize,
    /// "layers", "rows" or "mixed:N".
    pub sharing: String,
    #[serde(default)]
    pub tokens: Vec<u32>,
    /// A whole design of a real model, traced by operations instead of a small model on
    /// the gates.
    #[serde(default)]
    pub design: Option<DesignStart>,
}

/// The design to trace operation by operation: the model's folder and each die's shape.
#[derive(Deserialize, serde::Serialize)]
pub struct DesignStart {
    pub path: String,
    /// log2 of the vector units, of the lanes of each and of the cells of each die.
    pub cluster_bits: u32,
    pub lane_bits: u32,
    pub cell_bits: u32,
    pub context: u32,
}

/// The most chips a design trace may have: each is laid out whole.
const MOST_DESIGN_CHIPS: usize = 64;

#[derive(Default)]
pub struct TraceJob {
    pub outcome: Mutex<Option<Result<serde_json::Value, String>>>,
}

impl TraceJob {
    pub fn status(&self) -> serde_json::Value {
        match self.outcome.lock().unwrap().clone() {
            None => serde_json::json!({ "state": "running" }),
            Some(Ok(trace)) => serde_json::json!({ "state": "done", "trace": trace }),
            Some(Err(message)) => serde_json::json!({ "state": "error", "error": message }),
        }
    }
}

/// Checks a request before it starts.
pub fn check(start: &TraceStart) -> Result<(), String> {
    if let Some(design) = &start.design {
        if start.chips == 0 || start.chips > MOST_DESIGN_CHIPS {
            return Err(format!("from 1 to {MOST_DESIGN_CHIPS} dies"));
        }
        if design.cluster_bits > 11 || design.lane_bits > 8 || design.cell_bits > 24 || !(16..=1 << 20).contains(&design.context) {
            return Err("a die shape or context out of range".into());
        }
        return Ok(());
    }
    if start.chips == 0 || start.chips > MOST_CHIPS {
        return Err(format!("from 1 to {MOST_CHIPS} dies"));
    }
    if start.tokens.is_empty() || start.tokens.len() > TINY_CONTEXT {
        return Err(format!("from 1 to {TINY_CONTEXT} tokens"));
    }
    model_of(&start.model)?;
    Ok(())
}

fn model_of(name: &str) -> Result<TinyModel, String> {
    match name {
        "grouped" => Ok(TinyModel::Grouped),
        "plain" => Ok(TinyModel::Plain),
        other => Err(format!("{other} is not grouped or plain")),
    }
}

/// Traces a design of the model in `dir` operation by operation: a token at the start, the
/// middle and the end of the context. Kept under `cache`, named after the request, the
/// model's configuration and this build of the engine.
pub fn run_design(start: &TraceStart, design: &DesignStart, dir: &Path, cache: &Path) -> Result<serde_json::Value, String> {
    let config = std::fs::read_to_string(dir.join("config.json")).map_err(|e| format!("config.json: {e}"))?;
    let key = format!("{config}{}{}{}", start.chips, start.sharing, serde_json::to_string(design).unwrap_or_default());
    let sized = sim::optimize::cache_path(cache, &key);
    let file = sized.with_file_name(sized.file_name().and_then(|n| n.to_str()).unwrap_or("sizing.json").replacen("sizing", "trace", 1));
    if let Some(trace) = std::fs::read(&file).ok().and_then(|bytes| serde_json::from_slice(&bytes).ok()) {
        return Ok(trace);
    }
    let sharing = *sim::sizing::parse_sharings(&start.sharing, &[start.chips as u32])?.first().ok_or("no way of sharing")?;
    let program = compiler::compile_model(&config)?;
    let weights = sim::Weights::open(dir)?;
    let shape = Design { chips: start.chips as u32, sharing, cluster_bits: design.cluster_bits, lane_bits: design.lane_bits, cell_bits: design.cell_bits, context: design.context, factor_steps: 8 };
    let name = dir.file_name().and_then(|n| n.to_str()).unwrap_or("model");
    let title = format!("{name} on {} dies, {}", start.chips, chip_trace::sharing_label(sharing.for_chips(start.chips)));
    let last = design.context - 1;
    let trace = serde_json::to_value(design_trace(&title, &program, &weights, shape, &[0, last / 2, last])?).map_err(|e| e.to_string())?;
    std::fs::create_dir_all(cache).ok();
    std::fs::write(&file, trace.to_string()).ok();
    Ok(trace)
}

/// Builds the dies, runs the tokens and gives the trace. `scratch` is a folder of its own
/// for the model's checkpoint, removed afterwards.
pub fn run(start: &TraceStart, scratch: &Path) -> Result<serde_json::Value, String> {
    let model = model_of(&start.model)?;
    let sharing = *sim::sizing::parse_sharings(&start.sharing, &[start.chips as u32])?.first().ok_or("no way of sharing")?;
    std::fs::create_dir_all(scratch).map_err(|e| e.to_string())?;
    model.write(scratch);
    let vocab = compiler::compile_qwen3_5(&model.config())?.vocab as u32;
    let tokens: Vec<u32> = start.tokens.iter().map(|t| t % vocab).collect();
    let made = tiny_board(model, scratch, start.chips, sharing, TinyDies::default()).and_then(|mut board| {
        let title = format!("Small model on {} dies, {}", start.chips, chip_trace::sharing_label(board.sharing));
        board.record_checked(&title, &tokens, scratch)
    });
    std::fs::remove_dir_all(scratch).ok();
    serde_json::to_value(made?).map_err(|e| e.to_string())
}
