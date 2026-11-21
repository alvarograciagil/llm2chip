//! Recording a small model on dies joined by their links, gate by gate, for the 3D viewer.

use chip_trace::{tiny_board, TinyDies, TinyModel, TINY_CONTEXT};
use serde::Deserialize;
use std::path::Path;
use std::sync::Mutex;

/// The most dies a recording may join: each is simulated gate by gate.
const MOST_CHIPS: usize = 8;

#[derive(Deserialize)]
pub struct TraceStart {
    /// "grouped" or "plain".
    pub model: String,
    pub chips: usize,
    /// "layers", "rows" or "mixed:N".
    pub sharing: String,
    pub tokens: Vec<u32>,
}

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
        board.record(&title, &tokens)
    });
    std::fs::remove_dir_all(scratch).ok();
    serde_json::to_value(made?).map_err(|e| e.to_string())
}
