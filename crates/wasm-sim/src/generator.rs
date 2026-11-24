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
    /// The question goes on from the last answer, over the cache, as a chip that holds a
    /// conversation takes its next turn; otherwise the cache is emptied first, as the clear
    /// pin does.
    #[serde(default)]
    pub conversation: bool,
}

pub struct Generator {
    engine: ExactEngine,
    tokenizer: TokenizerTables,
    prompt: PromptStyle,
    context: usize,
    run: Option<Run>,
    /// Tokens the cache holds: every earlier turn of the conversation and its answers.
    held: usize,
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
        Ok(Self { engine, tokenizer, prompt, context: setup.context_tokens as usize, run: None, held: 0 })
    }

    /// Frames and tokenizes the prompt and starts from an empty context, or, in a
    /// conversation, from the cache its earlier turns left.
    pub fn start(&mut self, request: Request) -> Result<Value, String> {
        // A run left halfway leaves an answer the conversation does not know of.
        let going_on = request.conversation && self.held > 0 && self.run.is_none();
        let (text, add_special) = if going_on {
            (self.prompt.next_turn(request.system.trim(), &request.prompt, request.thinking)?, false)
        } else {
            self.prompt.render_with_system(request.system.trim(), &request.prompt, request.thinking)?
        };
        self.tokenizer.set_add_special(add_special);
        let ids = self.tokenizer.encode(&text)?;
        if ids.is_empty() {
            return Err("the prompt has no tokens".into());
        }
        let before = if going_on { self.held } else { 0 };
        if self.context > 0 && before + ids.len() >= self.context {
            return Err(if going_on {
                format!("the conversation takes {before} tokens and the question {} more, and the chip remembers {}: start a new conversation", ids.len(), self.context)
            } else {
                format!("the prompt takes {} tokens and the chip remembers {}: raise the design's context", ids.len(), self.context)
            });
        }
        if !going_on {
            self.engine.reset();
        }
        self.held = before;
        let reasoning = ReasoningBudget::new(self.prompt.think, request.think_budget);
        // A zero seed would stay zero through the xorshift.
        let seed = 0x9E37_79B9_7F4A_7C15 ^ request.seed.max(1);
        let event = json!({ "type": "start", "prompt_tokens": ids.len(), "held": before });
        self.run = Some(Run { request, ids, fed: 0, next: 0, output: Vec::new(), shown: 0, watch: LoopWatch::default(), reasoning, seed, ending: None });
        Ok(event)
    }

    /// One step of the chip: a token of the prompt, a token of the answer, or the end.
    pub fn advance(&mut self) -> Result<Value, String> {
        let run = self.run.as_mut().ok_or("nothing is running")?;
        if let Some(reason) = run.ending {
            self.run = None;
            return Ok(json!({ "type": "done", "reason": reason, "held": self.held }));
        }
        if run.fed < run.ids.len() {
            let id = run.ids[run.fed];
            run.reasoning.see(id);
            run.fed += 1;
            let last = run.fed == run.ids.len();
            let (logits, stats) = Runner::step(&mut self.engine, id, last)?;
            self.held += 1;
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
        } else if self.context > 0 && self.held >= self.context {
            Some("context")
        } else {
            None
        };
        if let Some(reason) = reason {
            self.run = None;
            return Ok(json!({ "type": "done", "reason": reason, "held": self.held }));
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
        // The chip writes the token that closes a loop and stops without running it, so the
        // cache a conversation keeps ends before it, as the firmware's does.
        if looping {
            run.ending = Some("loop");
            return Ok(json!({ "type": "token", "id": token, "text": piece, "seconds": 0.0, "saturated": 0, "index": run.output.len(), "hits": [], "tries": [] }));
        }
        let (logits, stats) = Runner::step(&mut self.engine, token, true)?;
        self.held += 1;
        let logits = logits.ok_or("no logits")?;
        let request = &run.request;
        run.next = run.reasoning.steer(sample(logits, request.temperature, request.top_k, &mut run.seed));
        Ok(json!({ "type": "token", "id": token, "text": piece, "seconds": stats.seconds, "saturated": stats.saturated, "index": run.output.len(), "hits": [], "tries": [] }))
    }
}
