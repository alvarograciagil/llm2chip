//! The chip simulator in the browser. The checkpoint's files are written into WebAssembly
//! memory as they download, so no second copy of the weights lives in JavaScript; events
//! cross the boundary as JSON, the same ones the local engine sends.

mod generator;

pub use generator::{Generator, Request, Setup};
#[cfg(all(feature = "threads", target_arch = "wasm32"))]
pub use wasm_bindgen_rayon::init_thread_pool;
use std::collections::HashMap;
use wasm_bindgen::prelude::*;

fn error(e: impl ToString) -> JsError {
    JsError::new(&e.to_string())
}

/// A checkpoint arriving: its text files, and room for each weights file filled by parts.
#[wasm_bindgen]
#[derive(Default)]
pub struct Checkpoint {
    texts: HashMap<String, String>,
    weights: Vec<(String, Vec<u8>)>,
}

#[wasm_bindgen]
impl Checkpoint {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Checkpoint {
        Checkpoint::default()
    }

    pub fn add_text(&mut self, name: String, text: String) {
        self.texts.insert(name, text);
    }

    /// Makes room for a weights file of `size` bytes and returns its number.
    pub fn reserve(&mut self, name: String, size: usize) -> Result<usize, JsError> {
        let mut bytes = Vec::new();
        bytes.try_reserve_exact(size).map_err(|_| error(format!("there is not enough memory for {name}")))?;
        bytes.resize(size, 0);
        self.weights.push((name, bytes));
        Ok(self.weights.len() - 1)
    }

    /// Writes a part of weights file `file` at `offset`.
    pub fn write(&mut self, file: usize, offset: usize, part: &[u8]) -> Result<(), JsError> {
        let bytes = &mut self.weights.get_mut(file).ok_or_else(|| error("no such file"))?.1;
        let end = offset.checked_add(part.len()).filter(|end| *end <= bytes.len()).ok_or_else(|| error("the file is larger than announced"))?;
        bytes[offset..end].copy_from_slice(part);
        Ok(())
    }
}

/// A model loaded on the chip being designed.
#[wasm_bindgen]
pub struct BrowserChip {
    generator: Generator,
}

#[wasm_bindgen]
impl BrowserChip {
    /// Builds the chip's engine from a checkpoint, with the design's arithmetic (`setup`, JSON).
    #[wasm_bindgen(constructor)]
    pub fn new(checkpoint: Checkpoint, setup: &str) -> Result<BrowserChip, JsError> {
        let setup: Setup = serde_json::from_str(setup).map_err(error)?;
        let Checkpoint { texts, mut weights } = checkpoint;
        weights.sort_by(|a, b| a.0.cmp(&b.0));
        let weights = weights.into_iter().map(|(_, bytes)| bytes.into_boxed_slice()).collect();
        Ok(BrowserChip { generator: Generator::new(&texts, weights, &setup).map_err(error)? })
    }

    /// Starts a run (`request`, JSON) and returns its first event.
    pub fn start(&mut self, request: &str) -> Result<String, JsError> {
        let request: Request = serde_json::from_str(request).map_err(error)?;
        Ok(self.generator.start(request).map_err(error)?.to_string())
    }

    /// Runs one step and returns its event; `done` ends the run.
    pub fn advance(&mut self) -> Result<String, JsError> {
        Ok(self.generator.advance().map_err(error)?.to_string())
    }
}

/// Whether the simulator can run a model of this `config.json`, and why not.
#[wasm_bindgen]
pub fn simulation_refusal(config: &str) -> Option<String> {
    compiler::compile_model(config).err()
}
