//! A model run on the chip's integer arithmetic from files held in memory, a step at a time,
//! with the engine's rules: the same framing, sampling, stops and endings as the local engine.

use netlist::format::WeightFormat;
use serde::Deserialize;
use serde_json::{json, Value};
use sim::prompt::{PromptStyle, ReasoningBudget};
use sim::tokenizer::TokenizerTables;
use sim::{sample, ExactEngine, Runner, Weights};
use std::collections::HashMap;
use tokenizer_rt::{memory_for_context, LoopWatch};

/// How the chip being designed computes: as the Simulator sends it to the engine.
#[derive(Clone, Deserialize)]
pub struct Setup {
    pub activation_bits: u32,
    pub format: Option<WeightFormat>,
    /// The context the design remembers; the answer ends there.
    pub context_tokens: u64,
}

/// What a run asks for.
#[derive(Clone, Deserialize)]
pub struct Request {
    pub prompt: String,
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_k: usize,
    #[serde(default)]
    pub thinking: bool,
    #[serde(default)]
    pub think_budget: Option<u32>,
    /// Starts the sampler's random sequence.
    pub seed: u64,
    /// The project's fixed system prompt, read before the question as the chip runs it.
    #[serde(default)]
    pub system: String,
}

pub struct Generator {
    engine: ExactEngine,
    tokenizer: TokenizerTables,
    prompt: PromptStyle,
    context: usize,
    run: Option<Run>,
}

struct Run {
    request: Request,
    ids: Vec<u32>,
    fed: usize,
    next: u32,
    output: Vec<u32>,
    shown: usize,
    watch: LoopWatch,
    reasoning: ReasoningBudget,
    seed: u64,
    /// Why the run ended, once it has.
    ending: Option<&'static str>,
}

/// The ids of a `tokenizer.json`'s tokens by their text: the vocabulary and the added tokens.
fn token_ids(tokenizer: &str) -> HashMap<String, u32> {
    let json: Value = serde_json::from_str(tokenizer).unwrap_or(Value::Null);
    let mut ids: HashMap<String, u32> = json["model"]["vocab"].as_object().map(|vocab| vocab.iter().filter_map(|(text, id)| Some((text.clone(), id.as_u64()? as u32))).collect()).unwrap_or_default();
    for added in json["added_tokens"].as_array().into_iter().flatten() {
        if let (Some(text), Some(id)) = (added["content"].as_str(), added["id"].as_u64()) {
            ids.insert(text.to_string(), id as u32);
        }
    }
    ids
}

impl Generator {
    /// `texts` holds the checkpoint's text files by name (config.json and tokenizer.json at
    /// least); `weights` its safetensors files in the order of their names.
    pub fn new(texts: &HashMap<String, String>, weights: Vec<Box<[u8]>>, setup: &Setup) -> Result<Self, String> {
        let config = texts.get("config.json").ok_or("the model has no config.json")?;
        let tokenizer_json = texts.get("tokenizer.json").ok_or("the model has no tokenizer.json")?;
        let program = compiler::compile_model(config)?;
        let family = program.family.clone();
        let (text_bytes, arena_words) = memory_for_context(setup.context_tokens);
        let tokenizer = TokenizerTables::from_json(tokenizer_json, text_bytes, arena_words)?;
        let ids = token_ids(tokenizer_json);
        let prompt = PromptStyle::read(&family, |name| texts.get(name).cloned(), |token| ids.get(token).copied());
        let weights = Weights::from_buffers(weights)?.with_format(setup.format);
        let engine = ExactEngine::new(program, weights, setup.activation_bits.clamp(8, 24))?;
        Ok(Self { engine, tokenizer, prompt, context: setup.context_tokens as usize, run: None })
    }

    /// Frames and tokenizes the prompt and starts from an empty context.
    pub fn start(&mut self, request: Request) -> Result<Value, String> {
        let (text, add_special) = self.prompt.render_with_system(request.system.trim(), &request.prompt, request.thinking)?;
        self.tokenizer.set_add_special(add_special);
        let ids = self.tokenizer.encode(&text)?;
        if ids.is_empty() {
            return Err("the prompt has no tokens".into());
        }
        if self.context > 0 && ids.len() >= self.context {
            return Err(format!("the prompt takes {} tokens and the chip remembers {}: raise the design's context", ids.len(), self.context));
        }
        self.engine.reset();
        let reasoning = ReasoningBudget::new(self.prompt.think, request.think_budget);
        // A zero seed would stay zero through the xorshift.
        let seed = 0x9E37_79B9_7F4A_7C15 ^ request.seed.max(1);
        let event = json!({ "type": "start", "prompt_tokens": ids.len() });
        self.run = Some(Run { request, ids, fed: 0, next: 0, output: Vec::new(), shown: 0, watch: LoopWatch::default(), reasoning, seed, ending: None });
        Ok(event)
    }

    /// One step of the chip: a token of the prompt, a token of the answer, or the end.
    pub fn advance(&mut self) -> Result<Value, String> {
        let run = self.run.as_mut().ok_or("nothing is running")?;
        if let Some(reason) = run.ending {
            self.run = None;
            return Ok(json!({ "type": "done", "reason": reason }));
        }
        if run.fed < run.ids.len() {
            let id = run.ids[run.fed];
            run.reasoning.see(id);
            run.fed += 1;
            let last = run.fed == run.ids.len();
            let (logits, stats) = Runner::step(&mut self.engine, id, last)?;
            if let (true, Some(logits)) = (last, logits) {
                let request = &run.request;
                run.next = run.reasoning.steer(sample(logits, request.temperature, request.top_k, &mut run.seed));
            }
            return Ok(json!({ "type": "prefill", "done": run.fed, "total": run.ids.len(), "seconds": stats.seconds }));
        }
        let reason = if run.output.len() >= run.request.max_tokens {
            Some("limit")
        } else if self.prompt.stop.contains(&run.next) {
            Some("stop")
        } else if self.context > 0 && run.ids.len() + run.output.len() >= self.context {
            Some("context")
        } else {
            None
        };
        if let Some(reason) = reason {
            self.run = None;
            return Ok(json!({ "type": "done", "reason": reason }));
        }
        let token = run.next;
        run.output.push(token);
        let looping = run.watch.push(token);
        // Text goes out once its characters are whole.
        let decoded = String::from_utf8_lossy(&self.tokenizer.decode(&run.output)).into_owned();
        let piece = if decoded.ends_with('\u{FFFD}') {
            String::new()
        } else {
            let piece = decoded.get(run.shown..).unwrap_or("").to_string();
            run.shown = decoded.len();
            piece
        };
        let (logits, stats) = Runner::step(&mut self.engine, token, true)?;
        let logits = logits.ok_or("no logits")?;
        let request = &run.request;
        run.next = run.reasoning.steer(sample(logits, request.temperature, request.top_k, &mut run.seed));
        if looping {
            run.ending = Some("loop");
        }
        Ok(json!({ "type": "token", "id": token, "text": piece, "seconds": stats.seconds, "saturated": stats.saturated, "index": run.output.len(), "hits": [], "tries": [] }))
    }
}
