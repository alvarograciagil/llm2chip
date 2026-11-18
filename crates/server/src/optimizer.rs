//! The search for the set of chips that does best what the person asks for a model, run
//! in the background: each design it tries is measured on its own generated circuit.

use crate::catalog::ModelEntry;
use serde::Deserialize;
use sim::optimize::{cache_path, optimize, Model, Progress, Request, Silicon};
use sim::sizing::{reference_die, Costs, N6};
use sim::Weights;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

/// What to search for, and for which model.
#[derive(Deserialize)]
pub struct OptimizeStart {
    pub path: String,
    #[serde(default)]
    pub request: Request,
    /// Cycles each cell takes to apply a group's scale and bias: a model whose quantisation
    /// groups are smaller than this cannot be built, so it is lowered to fit them.
    #[serde(default = "eight")]
    pub factor_steps: u32,
}

fn eight() -> u32 {
    8
}

/// A search under way: how far it has gone and, in the end, what it found.
#[derive(Default)]
pub struct OptimizeJob {
    pub name: Mutex<String>,
    pub progress: Mutex<Option<Progress>>,
    pub cancel: Arc<AtomicBool>,
    pub outcome: Mutex<Option<Result<serde_json::Value, String>>>,
}

impl OptimizeJob {
    /// Where the search is, as the status request shows it.
    pub fn status(&self) -> serde_json::Value {
        let name = self.name.lock().unwrap().clone();
        match self.outcome.lock().unwrap().clone() {
            None => serde_json::json!({ "state": "running", "name": name, "progress": *self.progress.lock().unwrap() }),
            Some(Ok(report)) => serde_json::json!({ "state": "done", "name": name, "report": report }),
            Some(Err(message)) => serde_json::json!({ "state": "error", "name": name, "error": message }),
        }
    }
}

/// Runs the search for `entry`, keeping its measurements under `home`.
pub fn run(entry: &ModelEntry, start: &OptimizeStart, job: &OptimizeJob, home: &Path) -> Result<serde_json::Value, String> {
    let dir = PathBuf::from(&entry.path);
    let program = compiler::compile_model(&entry.config)?;
    let weights = Weights::open(&dir)?;
    let graph = model_ir::ModelGraph::from_config_json(&entry.name, &entry.config).ok();
    let model = Model { program: &program, weights: &weights, graph: graph.as_ref(), cache: Some(cache_path(&home.join(".llm2chip/cache"), &entry.config)) };
    let base = reference_die(start.factor_steps.clamp(1, 64));
    let silicon = Silicon { base, costs: Costs::measure(base), process: N6 };
    let report = optimize(&model, &silicon, &start.request, &job.cancel, &|p| *job.progress.lock().unwrap() = Some(p))?;
    serde_json::to_value(report).map_err(|e| e.to_string())
}
