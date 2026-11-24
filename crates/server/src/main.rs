//! Local engine for the app. Listens on 127.0.0.1 and requires the session token.

mod catalog;
mod hub;
mod layout;
mod optimizer;
mod trace_job;
mod prompt;
mod retouch;
#[cfg(any(feature = "listen", feature = "speak"))]
mod voice_job;

use axum::{
    extract::{ws::{Message, WebSocket}, Query, Request, State, WebSocketUpgrade},
    http::{header, StatusCode},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{get, post, put},
    Json, Router,
};
use catalog::{Catalog, ModelEntry, Settings};
use compiler::compile_model;
use hub::Hub;
use prompt::PromptStyle;
use serde::Deserialize;
use netlist::activity::ChipActivity;
use netlist::format::WeightFormat;
use netlist::block::encode_vector;
use netlist::chip::{chip_info, ChipModel, GenerationSetup};
use netlist::image::build_image;
use netlist::rv32::Rv32Iss;
use netlist::sampler::{exponential_table, SamplerConfig, SamplerModel};
use netlist::soc::{FirmwareImage, SocBus, SocConfig};
use netlist::{CellConfig, DieConfig, Microprogram, Parameter};
use sim::chip::{RunnerEngine, FLOW_DRAFT_REGISTER, FLOW_REGISTER_BITS, FLOW_INDEX_BITS};
use sim::image::{fitted_die, load_or_build, verify, write_chip_image, ImageRequest, LoadReport};
use sim::lower::Layout;
use sim::tokenizer::TokenizerTables;
use sim::chip::{ImageTokens, Placement};
use sim::{sample, Engine, ExactEngine, Numeric, Runner, Weights};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, Mutex};
use tokenizers::Tokenizer;
use tokio::sync::mpsc;

/// The engine's port unless `LLM2CHIP_ENGINE_PORT` gives another.
const DEFAULT_PORT: u16 = 7878;

fn engine_port() -> u16 {
    std::env::var("LLM2CHIP_ENGINE_PORT").ok().and_then(|p| p.trim().parse().ok()).unwrap_or(DEFAULT_PORT)
}
const FIRMWARE: &[u8] = include_bytes!("../../../firmware/prebuilt/firmware.elf");
const FLOW_LOGITS_REGISTER: u32 = 1;
const DEFAULT_CONTEXT: u64 = 8192;
/// The most patches of a picture the chip's own image encoder takes: 256 of 16×16 pixels,
/// the least the Qwen encoder asks for. Larger pictures are resized down to fit.
const CHIP_IMAGE_PATCHES: usize = 256;

struct Loaded {
    name: String,
    engine: Box<dyn Runner>,
    tokenizer: Tokenizer,
    accelerator: Option<String>,
    dir: PathBuf,
    /// The firmware's tokenizer tables, built the first time they are needed.
    /// The tokenizer's tables, and the context and the special tokens they were built for.
    firmware_tables: Option<((u64, bool), Vec<u32>)>,
    load_report: Option<LoadReport>,
    prompt: PromptStyle,
    /// The image encoder, opened the first time an image arrives.
    vision: Option<Seer>,
    /// The image generator, opened the first time a picture is asked for.
    painter: Option<sim::janus::Janus>,
    /// The bits of the die's arithmetic when the model runs on it, for drawing too.
    exact_bits: Option<u32>,
    /// The conversation the cache holds.
    talk: Talk,
}

impl Loaded {
    /// Takes the engine back from the chip that kept it, which forgets its conversation.
    fn reclaim(&mut self) {
        if let Some(chip) = self.talk.chip.take() {
            if let Some(engine) = chip.bus.peripheral.engine.into_engine() {
                self.engine = engine;
            }
        }
    }
}

/// The conversation the simulator holds, as a chip keeps one between questions.
#[derive(Default)]
struct Talk {
    /// Tokens the cache holds: none when it is empty, or when the last turn was left halfway.
    held: usize,
    /// What it is held with: a question that goes on must come with the same.
    signature: String,
    /// The chip that holds it, when the firmware runs: it keeps the engine meanwhile.
    chip: Option<KeptChip>,
}

/// A chip at instruction level, with the engine it keeps, waiting for its next turn.
struct KeptChip {
    bus: SocBus<ChipModel<RunnerEngine<'static>>>,
    cpu: Rv32Iss,
    profile: netlist::activity::CpuProfile,
    signature: String,
    /// It writes tokens, not text.
    raw: bool,
    /// It holds a conversation; otherwise it stops after one answer.
    conversation: bool,
    /// What its CPU is, for the simulator.
    about: serde_json::Value,
}

/// What stands for the engine while a chip keeps it.
struct Lent;

impl Runner for Lent {
    fn reset(&mut self) {}
    fn vocabulary(&self) -> usize {
        0
    }
    fn step(&mut self, _token: u32, _want_logits: bool) -> Result<(Option<&[f32]>, sim::StepStats), String> {
        Err("the engine is in the chip that holds the conversation".into())
    }
    fn has_draft_head(&self) -> bool {
        false
    }
    fn draft_tokens(&self) -> usize {
        0
    }
    fn truncate_draft(&mut self, _tokens: usize) {}
    fn feed_draft_from_main(&mut self) {}
    fn feed_draft_from_draft(&mut self) {}
    fn draft(&mut self, _token: u32, _want_logits: bool) -> Result<Option<&[f32]>, String> {
        Err("the engine is in the chip that holds the conversation".into())
    }
}

#[derive(Clone)]
struct App {
    token: Arc<String>,
    loaded: Arc<Mutex<Option<Loaded>>>,
    dist: Option<Arc<PathBuf>>,
    catalog: Arc<Catalog>,
    hub: Arc<Hub>,
    image: Arc<Mutex<Option<Arc<ImageJob>>>>,
    layout: Arc<Mutex<Option<Arc<layout::LayoutJob>>>>,
    retouch: Arc<Mutex<Option<Arc<retouch::RetouchJob>>>>,
    optimize: Arc<Mutex<Option<Arc<optimizer::OptimizeJob>>>>,
    trace: Arc<Mutex<Option<Arc<trace_job::TraceJob>>>>,
    replay: Arc<Mutex<Option<trace_job::Replayed>>>,
    #[cfg(any(feature = "listen", feature = "speak"))]
    voice: Arc<Mutex<Option<Arc<voice_job::VoiceJob>>>>,
    #[cfg(any(feature = "listen", feature = "speak"))]
    voice_plan: Arc<Mutex<Option<Arc<voice_job::PlanJob>>>>,
    /// The spoken conversation going on, its language model's cache kept between turns.
    #[cfg(any(feature = "listen", feature = "speak"))]
    conversation: Arc<Mutex<Option<sim::voice::Conversation>>>,
    painter: retouch::Painter,
    chip_painter: retouch::ChipPainter,
}

fn home() -> PathBuf {
    PathBuf::from(std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE")).unwrap_or_default())
}

async fn models(State(app): State<App>, Query(query): Query<HashMap<String, String>>) -> Json<Vec<ModelEntry>> {
    if query.contains_key("refresh") {
        app.catalog.forget();
    }
    Json(tokio::task::spawn_blocking(move || app.catalog.models()).await.unwrap_or_default())
}

fn refused(message: String) -> Response {
    (StatusCode::BAD_REQUEST, message).into_response()
}

async fn read_settings(State(app): State<App>) -> Response {
    Json(app.catalog.view()).into_response()
}

async fn write_settings(State(app): State<App>, Json(settings): Json<Settings>) -> Response {
    match tokio::task::spawn_blocking(move || app.catalog.update(settings)).await {
        Ok(Ok(view)) => Json(view).into_response(),
        Ok(Err(message)) => refused(message),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    }
}

async fn hub_search(State(app): State<App>, Query(query): Query<HashMap<String, String>>) -> Response {
    match app.hub.search(query.get("q").map_or("", String::as_str)).await {
        Ok(hits) => Json(hits).into_response(),
        Err(message) => refused(message),
    }
}

async fn hub_details(State(app): State<App>, Query(query): Query<HashMap<String, String>>) -> Response {
    match app.hub.details(query.get("id").map_or("", String::as_str), &app.catalog.download_dir()).await {
        Ok(details) => Json(details).into_response(),
        Err(message) => refused(message),
    }
}

#[derive(Deserialize)]
struct RepoRequest {
    id: String,
}

async fn hub_download(State(app): State<App>, Json(request): Json<RepoRequest>) -> Response {
    let catalog = app.catalog.clone();
    match app.hub.start(request.id, app.catalog.download_dir(), move || catalog.forget()).await {
        Ok(job) => Json(job).into_response(),
        Err(message) => refused(message),
    }
}

async fn hub_jobs(State(app): State<App>) -> Response {
    Json(app.hub.jobs()).into_response()
}

async fn hub_cancel(State(app): State<App>, Json(request): Json<RepoRequest>) -> Response {
    Json(serde_json::json!({ "cancelled": app.hub.cancel(&request.id) })).into_response()
}

async fn hub_discard(State(app): State<App>, Json(request): Json<RepoRequest>) -> Response {
    match app.hub.discard(&request.id, &app.catalog.download_dir()).await {
        Ok(()) => StatusCode::NO_CONTENT.into_response(),
        Err(message) => refused(message),
    }
}

async fn hub_token(State(app): State<App>) -> Response {
    Json(app.hub.token_status()).into_response()
}

#[derive(Deserialize)]
struct TokenRequest {
    token: String,
}

async fn hub_set_token(State(app): State<App>, Json(request): Json<TokenRequest>) -> Response {
    match app.hub.set_token(&request.token) {
        Ok(()) => Json(app.hub.token_status()).into_response(),
        Err(message) => refused(message),
    }
}

async fn hub_clear_token(State(app): State<App>) -> Response {
    match app.hub.clear_token() {
        Ok(()) => Json(app.hub.token_status()).into_response(),
        Err(message) => refused(message),
    }
}

#[derive(Deserialize)]
struct LabRequest {
    path: String,
    prompt: String,
    new_tokens: usize,
    activation_bits: u32,
    state_bits: u32,
    table_bits: Option<u32>,
}

fn relative_error(fixed: &[f32], float: &[f32]) -> f64 {
    let (mut difference, mut size) = (0f64, 0f64);
    for (a, b) in fixed.iter().zip(float) {
        difference += ((a - b) as f64).powi(2);
        size += (*b as f64).powi(2);
    }
    (difference / size.max(1e-30)).sqrt()
}

/// Runs the same tokens in floating point and in the chip's arithmetic and compares them.
fn run_lab(catalog: &Catalog, request: LabRequest) -> Result<serde_json::Value, String> {
    let dir = PathBuf::from(&request.path);
    let entry = catalog.find(&request.path).ok_or("the model is not in the known folders")?;
    if !entry.supported {
        return Err(entry.adaptation.reason.unwrap_or_else(|| "this architecture cannot run yet".into()));
    }
    let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| e.to_string())?;
    let program = compile_model(&entry.config)?;
    let prompt = prompt::read(&dir, &program.family, &tokenizer);
    let (text, add_special) = prompt.render(&request.prompt, false)?;
    let mut tokens = tokenizer.encode(text.as_str(), add_special).map_err(|e| e.to_string())?.get_ids().to_vec();
    let prompt_tokens = tokens.len();
    if prompt_tokens == 0 {
        return Err("the text produces no token".into());
    }
    // The output of each layer is what its last operation writes.
    let mut layer_outputs: Vec<(u32, usize)> = Vec::new();
    for inst in &program.ops {
        if let (Some(layer), Some(buffer)) = (inst.layer, inst.op.buffers().1.first().copied()) {
            match layer_outputs.last_mut() {
                Some(last) if last.0 == layer => last.1 = buffer,
                _ => layer_outputs.push((layer, buffer)),
            }
        }
    }
    let mode = Numeric::Block { activation_bits: request.activation_bits.clamp(4, 24), state_bits: request.state_bits.clamp(4, 24), table_bits: request.table_bits.map(|b| b.clamp(4, 16)) };
    let mut float = Engine::new(compile_model(&entry.config)?, Weights::open(&dir)?, Numeric::Float)?;
    let mut fixed = Engine::new(compile_model(&entry.config)?, Weights::open(&dir)?, mode)?;
    let mut layer_errors = vec![0f64; layer_outputs.len()];
    let (mut by_token, mut state_by_token, mut agreements, mut first_divergence) = (Vec::new(), Vec::new(), 0usize, None);
    let total = prompt_tokens + request.new_tokens.clamp(1, 256);
    let mut position = 0;
    while position < total {
        let token = *tokens.get(position).ok_or("tokens are missing")?;
        let wanted = position + 1 >= prompt_tokens;
        let best = |logits: &[f32]| logits.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map_or(0, |(i, _)| i as u32);
        let float_logits = float.step(token, wanted)?.0.map(<[f32]>::to_vec);
        let fixed_logits = fixed.step(token, wanted)?.0.map(<[f32]>::to_vec);
        for (slot, (_, buffer)) in layer_errors.iter_mut().zip(&layer_outputs) {
            *slot += relative_error(fixed.buffer(*buffer), float.buffer(*buffer));
        }
        let drift: Vec<f64> = fixed.recurrent_state().zip(float.recurrent_state()).map(|(a, b)| relative_error(a, b)).collect();
        state_by_token.push(if drift.is_empty() { 0.0 } else { drift.iter().sum::<f64>() / drift.len() as f64 });
        if let (Some(a), Some(b)) = (&fixed_logits, &float_logits) {
            by_token.push(relative_error(a, b));
            let (mine, theirs) = (best(a), best(b));
            if mine == theirs {
                agreements += 1;
            } else if first_divergence.is_none() {
                first_divergence = Some(position + 1 - prompt_tokens);
            }
            if position + 1 == tokens.len() {
                tokens.push(theirs);
            }
        }
        position += 1;
    }
    let steps = total as f64;
    let compared = by_token.len();
    Ok(serde_json::json!({
        "layers": layer_outputs.iter().zip(&layer_errors).map(|((layer, _), error)| serde_json::json!({ "layer": layer, "error": error / steps })).collect::<Vec<_>>(),
        "logits_error": by_token,
        "state_error": state_by_token,
        "prompt_tokens": prompt_tokens,
        "agreements": agreements,
        "compared": compared,
        "first_divergence": first_divergence,
        "has_state": !program.delta_slots.is_empty(),
    }))
}

#[derive(Deserialize)]
struct SpeedRequest {
    path: String,
    context_tokens: u64,
    cells: usize,
    activation_bits: u32,
    #[serde(default = "one_lane")]
    vector_lanes: u32,
    #[serde(default = "one_lane")]
    vector_units: u32,
}

/// Counts the cycles of a token at the last position of the context, on the die the image
/// would be built for. The result is what the circuit does, not what the estimator expects.
fn run_speed(catalog: &Catalog, request: SpeedRequest) -> Result<serde_json::Value, String> {
    let dir = PathBuf::from(&request.path);
    let entry = catalog.find(&request.path).ok_or("the model is not in the known folders")?;
    if !entry.supported {
        return Err(entry.adaptation.reason.unwrap_or_else(|| "this architecture cannot run yet".into()));
    }
    let program = compile_model(&entry.config)?;
    let weights = Weights::open(&dir)?;
    let context = request.context_tokens.max(1);
    let image = ImageRequest { sharing: Default::default(), chips: vec![1], context_tokens: context, cells: request.cells.max(1), activation_bits: request.activation_bits.clamp(8, 24), lane_bits: request.vector_lanes.max(1).ilog2().min(MAX_IMAGE_LANE_BITS), cluster_bits: request.vector_units.max(1).ilog2().min(MAX_UNIT_BITS), add_special: false, stop: Vec::new(), system: Vec::new(), conversation: false };
    let die = fitted_die(&program, &weights, &image)?;
    let layout = Layout::for_image(&program, &weights, die, context as usize)?;
    let mut totals: Vec<(String, u64, u32)> = Vec::new();
    for (name, cycles) in layout.cycles_by_op(&program, 0, (context - 1) as u32)? {
        match totals.iter_mut().find(|row| row.0 == name) {
            Some(row) => (row.1, row.2) = (row.1 + cycles, row.2 + 1),
            None => totals.push((name, cycles, 1)),
        }
    }
    totals.sort_by_key(|row| std::cmp::Reverse(row.1));
    let total: u64 = totals.iter().map(|row| row.1).sum();
    let ops: Vec<_> = totals.iter().map(|(name, cycles, count)| serde_json::json!({ "name": name, "cycles": cycles, "count": count })).collect();
    Ok(serde_json::json!({ "name": entry.name, "total": total, "position": context - 1, "cells": die.cells, "lanes": 1u32 << die.features.lane_bits, "units": 1u32 << die.features.cluster_bits, "ops": ops }))
}

async fn speed(State(app): State<App>, Json(request): Json<SpeedRequest>) -> Response {
    let catalog = app.catalog.clone();
    match tokio::task::spawn_blocking(move || run_speed(&catalog, request)).await {
        Ok(Ok(result)) => Json(result).into_response(),
        Ok(Err(message)) => refused(message),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    }
}

async fn lab(State(app): State<App>, Json(request): Json<LabRequest>) -> Response {
    let catalog = app.catalog.clone();
    match tokio::task::spawn_blocking(move || run_lab(&catalog, request)).await {
        Ok(Ok(result)) => Json(result).into_response(),
        Ok(Err(message)) => refused(message),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    }
}

/// A chip image being written in the background.
#[derive(Default)]
struct ImageJob {
    name: Mutex<String>,
    done: Arc<AtomicU64>,
    total: Arc<AtomicU64>,
    cancel: Arc<AtomicBool>,
    outcome: Mutex<Option<Result<serde_json::Value, String>>>,
}

#[derive(Deserialize)]
struct ImageStart {
    path: String,
    chips: Vec<usize>,
    /// How the chips share the model: by layers or by rows of every matrix.
    #[serde(default)]
    sharing: sim::chip::Sharing,
    context_tokens: u64,
    cells: usize,
    activation_bits: u32,
    #[serde(default = "one_lane")]
    vector_lanes: u32,
    #[serde(default = "one_lane")]
    vector_units: u32,
    /// The project's fixed system prompt, which the chip carries and runs before every
    /// question.
    #[serde(default)]
    system: String,
}

async fn image_status(State(app): State<App>) -> Response {
    let job = app.image.lock().unwrap().clone();
    let Some(job) = job else { return Json(serde_json::json!({ "state": "idle" })).into_response() };
    let outcome = job.outcome.lock().unwrap().clone();
    let name = job.name.lock().unwrap().clone();
    let (done, total) = (job.done.load(Relaxed), job.total.load(Relaxed));
    Json(match outcome {
        None => serde_json::json!({ "state": "writing", "name": name, "done": done, "total": total }),
        Some(Ok(report)) => serde_json::json!({ "state": "done", "name": name, "report": report }),
        Some(Err(message)) => serde_json::json!({ "state": "error", "name": name, "error": message }),
    })
    .into_response()
}

/// Starts a real layout of the design's compute cell with OpenROAD, one at a time.
async fn layout_start(State(app): State<App>, Json(request): Json<layout::LayoutRequest>) -> Response {
    let running = app.layout.lock().unwrap().as_ref().is_some_and(|job| job.outcome.lock().unwrap().is_none());
    if running {
        return (StatusCode::CONFLICT, "a layout is already running").into_response();
    }
    let job = Arc::new(layout::LayoutJob::default());
    *app.layout.lock().unwrap() = Some(job.clone());
    tokio::task::spawn_blocking(move || {
        let outcome = layout::run(&request, &job, &home());
        *job.outcome.lock().unwrap() = Some(outcome);
    });
    StatusCode::ACCEPTED.into_response()
}

#[cfg(any(feature = "listen", feature = "speak"))]
/// What the spoken turn is doing, or what it gave.
async fn voice_status(State(app): State<App>) -> Response {
    let job = app.voice.lock().unwrap().clone();
    // What this engine was built to do: listen, speak, or both.
    let Some(job) = job else { return Json(serde_json::json!({ "state": "idle", "listen": cfg!(feature = "listen"), "speak": cfg!(feature = "speak") })).into_response() };
    let outcome = job.outcome.lock().unwrap().clone();
    Json(match outcome {
        None => serde_json::json!({ "state": "running", "phase": job.phase.lock().unwrap().clone(), "partial": job.partial.lock().unwrap().clone() }),
        Some(Ok(report)) => serde_json::json!({ "state": "done", "result": report }),
        Some(Err(message)) => serde_json::json!({ "state": "error", "error": message, "partial": job.partial.lock().unwrap().clone() }),
    })
    .into_response()
}

#[cfg(any(feature = "listen", feature = "speak"))]
/// Starts a spoken turn: hear the clip or take the text, answer, say the answer, as far as
/// the chip listens and speaks. One at a time, and only with models of the known folders.
async fn voice_start(State(app): State<App>, Json(start): Json<voice_job::VoiceStart>) -> Response {
    let running = app.voice.lock().unwrap().as_ref().is_some_and(|job| job.outcome.lock().unwrap().is_none());
    if running {
        return refused("a spoken turn is already running".into());
    }
    if let Some(why) = start.refusal() {
        return refused(why);
    }
    for (path, role) in start.models() {
        let Some(entry) = app.catalog.find(path) else { return refused("a model is not in the known folders".into()) };
        if role.is_some() && entry.voice != role {
            return refused(format!("{} does not do that part of a spoken turn", entry.name));
        }
        if role.is_none() && !entry.supported {
            return refused(entry.adaptation.reason.unwrap_or_else(|| "this architecture cannot run yet".into()));
        }
    }
    let job = Arc::new(voice_job::VoiceJob::default());
    *app.voice.lock().unwrap() = Some(job.clone());
    let kept = app.conversation.clone();
    tokio::task::spawn_blocking(move || {
        let outcome = voice_job::run(&start, &job, &kept);
        *job.outcome.lock().unwrap() = Some(outcome);
    });
    StatusCode::ACCEPTED.into_response()
}

#[cfg(any(feature = "listen", feature = "speak"))]
/// What the voice system's planning is doing, or what it chose.
async fn voice_plan_status(State(app): State<App>) -> Response {
    let job = app.voice_plan.lock().unwrap().clone();
    let Some(job) = job else { return Json(serde_json::json!({ "state": "idle" })).into_response() };
    let outcome = job.outcome.lock().unwrap().clone();
    Json(match outcome {
        None => serde_json::json!({ "state": "running", "doing": job.doing.lock().unwrap().clone() }),
        Some(Ok(result)) => serde_json::json!({ "state": "done", "result": result }),
        Some(Err(message)) => serde_json::json!({ "state": "error", "error": message }),
    })
    .into_response()
}

#[cfg(any(feature = "listen", feature = "speak"))]
/// Plans a chip that listens, speaks or both, with its language model: one at a time, with
/// models of the known folders.
async fn voice_plan_start(State(app): State<App>, Json(start): Json<voice_job::PlanStart>) -> Response {
    let running = app.voice_plan.lock().unwrap().as_ref().is_some_and(|job| job.outcome.lock().unwrap().is_none());
    if running {
        return refused("a plan is already being made".into());
    }
    let named = [(&start.listen, Some("listens")), (&start.speak, Some("speaks")), (&start.sound, Some("sounds"))];
    let roles = named.into_iter().filter_map(|(path, role)| path.as_deref().map(|p| (p, role))).chain([(start.think.as_str(), None)]);
    for (path, role) in roles {
        let Some(entry) = app.catalog.find(path) else { return refused("a model is not in the known folders".into()) };
        if role.is_some() && entry.voice != role {
            return refused(format!("{} does not do that part of a spoken turn", entry.name));
        }
    }
    let job = Arc::new(voice_job::PlanJob::default());
    *app.voice_plan.lock().unwrap() = Some(job.clone());
    tokio::task::spawn_blocking(move || {
        let outcome = voice_job::run_plan(&start, &job, &home());
        *job.outcome.lock().unwrap() = Some(outcome);
    });
    StatusCode::ACCEPTED.into_response()
}

/// Starts the search for the best set of chips for a model, one at a time.
async fn optimize_start(State(app): State<App>, Json(start): Json<optimizer::OptimizeStart>) -> Response {
    let running = app.optimize.lock().unwrap().as_ref().is_some_and(|job| job.outcome.lock().unwrap().is_none());
    if running {
        return refused("a search is already running".into());
    }
    let Some(entry) = app.catalog.find(&start.path) else { return refused("the model is not in the known folders".into()) };
    if !entry.supported {
        return refused(entry.adaptation.reason.unwrap_or_else(|| "this architecture cannot run yet".into()));
    }
    let job = Arc::new(optimizer::OptimizeJob { name: Mutex::new(entry.name.clone()), ..Default::default() });
    *app.optimize.lock().unwrap() = Some(job.clone());
    tokio::task::spawn_blocking(move || {
        let outcome = optimizer::run(&entry, &start, &job, &home());
        *job.outcome.lock().unwrap() = Some(outcome);
    });
    StatusCode::ACCEPTED.into_response()
}

/// Records a small model running on dies joined by their links, gate by gate, for the
/// 3D viewer: one recording at a time.
async fn trace_start(State(app): State<App>, Json(start): Json<trace_job::TraceStart>) -> Response {
    let running = app.trace.lock().unwrap().as_ref().is_some_and(|job| job.outcome.lock().unwrap().is_none());
    if running {
        return refused("a recording is already running".into());
    }
    if let Err(message) = trace_job::check(&start) {
        return refused(message);
    }
    // A design's model must be one of the known folders, as for the optimiser.
    let model_dir = match &start.design {
        Some(design) => match app.catalog.find(&design.path) {
            Some(entry) if entry.supported => Some(std::path::PathBuf::from(&design.path)),
            Some(entry) => return refused(entry.adaptation.reason.unwrap_or_else(|| "this architecture cannot run yet".into())),
            None => return refused("the model is not in the known folders".into()),
        },
        None => None,
    };
    let job = Arc::new(trace_job::TraceJob::default());
    *app.trace.lock().unwrap() = Some(job.clone());
    tokio::task::spawn_blocking(move || {
        let scratch = std::env::temp_dir().join(format!("llm2chip-trace-{}-{}", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_millis())));
        let run = || match (&start.design, &model_dir) {
            (Some(design), Some(dir)) => trace_job::run_design(&start, design, dir, &home().join("cache")),
            _ => trace_job::run(&start, &scratch),
        };
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(run)).unwrap_or_else(|_| Err("the recording stopped with an internal error".into()));
        *job.outcome.lock().unwrap() = Some(outcome);
    });
    StatusCode::ACCEPTED.into_response()
}

/// Looks inside a die of a recorded run: replays the run (once) and answers with `answer`.
async fn trace_look(app: App, look: trace_job::Inspect, answer: fn(&mut Option<trace_job::Replayed>, &trace_job::Inspect, &std::path::Path) -> Result<serde_json::Value, String>) -> Response {
    let scratch = std::env::temp_dir().join(format!("llm2chip-replay-{}", std::process::id()));
    let done = tokio::task::spawn_blocking(move || {
        let mut kept = app.replay.lock().unwrap();
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| answer(&mut kept, &look, &scratch))).unwrap_or_else(|_| Err("the replay stopped with an internal error".into()))
    })
    .await;
    match done {
        Ok(Ok(value)) => Json(value).into_response(),
        Ok(Err(message)) => refused(message),
        Err(_) => refused("the replay stopped with an internal error".into()),
    }
}

async fn trace_design(State(app): State<App>, Json(look): Json<trace_job::Inspect>) -> Response {
    trace_look(app, look, trace_job::design).await
}

async fn trace_inspect(State(app): State<App>, Json(look): Json<trace_job::Inspect>) -> Response {
    trace_look(app, look, trace_job::inspect).await
}

async fn trace_status(State(app): State<App>) -> Response {
    let job = app.trace.lock().unwrap().clone();
    Json(job.map_or_else(|| serde_json::json!({ "state": "idle" }), |job| job.status())).into_response()
}

async fn optimize_status(State(app): State<App>) -> Response {
    let job = app.optimize.lock().unwrap().clone();
    Json(job.map_or_else(|| serde_json::json!({ "state": "idle" }), |job| job.status())).into_response()
}

async fn optimize_cancel(State(app): State<App>) -> Response {
    if let Some(job) = app.optimize.lock().unwrap().as_ref() {
        job.cancel.store(true, Relaxed);
    }
    StatusCode::NO_CONTENT.into_response()
}

async fn layout_status(State(app): State<App>) -> Response {
    let job = app.layout.lock().unwrap().clone();
    let Some(job) = job else { return Json(serde_json::json!({ "state": "idle" })).into_response() };
    let outcome = job.outcome.lock().unwrap().clone();
    let phase = job.phase.lock().unwrap().clone();
    Json(match outcome {
        None => serde_json::json!({ "state": "running", "phase": phase }),
        Some(Ok(result)) => serde_json::json!({ "state": "done", "result": result }),
        Some(Err(message)) => serde_json::json!({ "state": "error", "error": message }),
    })
    .into_response()
}

/// Starts drawing or retouching a picture with Show-o, one at a time.
async fn retouch_start(State(app): State<App>, Json(request): Json<retouch::RetouchRequest>) -> Response {
    let running = app.retouch.lock().unwrap().as_ref().is_some_and(|job| job.outcome.lock().unwrap().is_none());
    if running {
        return refused("an image is already being written".into());
    }
    let Some(entry) = app.catalog.find(&request.path) else { return refused("the model is not in the known folders".into()) };
    if !entry.images.edits {
        return refused(format!("{} does not retouch images", entry.name));
    }
    let job = Arc::new(retouch::RetouchJob::default());
    *app.retouch.lock().unwrap() = Some(job.clone());
    let (painter, chip_painter) = (app.painter.clone(), app.chip_painter.clone());
    tokio::task::spawn_blocking(move || {
        let outcome = retouch::run(&request, &job, &painter, &chip_painter);
        *job.outcome.lock().unwrap() = Some(outcome);
    });
    StatusCode::ACCEPTED.into_response()
}

async fn retouch_status(State(app): State<App>) -> Response {
    let job = app.retouch.lock().unwrap().clone();
    let Some(job) = job else { return Json(serde_json::json!({ "state": "idle" })).into_response() };
    let outcome = job.outcome.lock().unwrap().clone();
    let phase = job.phase.lock().unwrap().clone();
    let (step, steps) = *job.step.lock().unwrap();
    Json(match outcome {
        None => serde_json::json!({ "state": "running", "phase": phase, "step": step, "steps": steps }),
        Some(Ok(result)) => serde_json::json!({ "state": "done", "result": result }),
        Some(Err(message)) => serde_json::json!({ "state": "error", "error": message }),
    })
    .into_response()
}

async fn retouch_cancel(State(app): State<App>) -> Response {
    if let Some(job) = app.retouch.lock().unwrap().as_ref() {
        job.cancel.store(true, Relaxed);
    }
    StatusCode::NO_CONTENT.into_response()
}

async fn image_cancel(State(app): State<App>) -> Response {
    if let Some(job) = app.image.lock().unwrap().as_ref() {
        job.cancel.store(true, Relaxed);
    }
    StatusCode::NO_CONTENT.into_response()
}

/// Writes the chip image of a model next to the others, in the background.
async fn image_start(State(app): State<App>, Json(request): Json<ImageStart>) -> Response {
    let running = app.image.lock().unwrap().as_ref().is_some_and(|job| job.outcome.lock().unwrap().is_none());
    if running {
        return refused("an image is already being generated".into());
    }
    let catalog = app.catalog.clone();
    let Some(entry) = catalog.find(&request.path) else { return refused("the model is not in the known folders".into()) };
    if !entry.supported {
        return refused(entry.adaptation.reason.unwrap_or_else(|| "this architecture cannot run yet".into()));
    }
    let job = Arc::new(ImageJob { name: Mutex::new(entry.name.clone()), ..Default::default() });
    *app.image.lock().unwrap() = Some(job.clone());
    tokio::task::spawn_blocking(move || {
        let dir = PathBuf::from(&entry.path);
        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"));
        let outcome = tokenizer.map_err(|e| e.to_string()).and_then(|tokenizer| {
            let family = compile_model(&entry.config)?.family;
            let prompt = prompt::read(&dir, &family, &tokenizer);
            let firmware = FirmwareImage::from_elf(FIRMWARE)?;
            let out = home().join(".llm2chip/images").join(format!("{}.chipimg", entry.name.replace('/', "--")));
            // A chip's image has one framing of the text: with its template, nothing is added.
            // A system prompt takes the special tokens with it, into the chip's description.
            let add_special = prompt.render("", false)?.1;
            let system = request.system.trim();
            let (add_special, system) = if system.is_empty() {
                (add_special, Vec::new())
            } else {
                let (framed, special) = prompt.render_with_system(system, "?", false)?;
                let encode = |text: &str, special: bool| tokenizer.encode(text, special).map(|e| e.get_ids().to_vec()).map_err(|e| e.to_string());
                (false, prompt.split_system(system, &framed, false, special, encode)?.0)
            };
            let image = ImageRequest { chips: request.chips, context_tokens: request.context_tokens.max(1), cells: request.cells.max(1), activation_bits: request.activation_bits, lane_bits: request.vector_lanes.max(1).ilog2().min(MAX_IMAGE_LANE_BITS), cluster_bits: request.vector_units.max(1).ilog2().min(MAX_UNIT_BITS), add_special, stop: prompt.stop.clone(), sharing: request.sharing, system, conversation: true };
            let report = write_chip_image(&dir, &firmware, &image, &out, &job.done, &job.total, &job.cancel)?;
            Ok(serde_json::json!({ "path": out.to_string_lossy(), "bytes": report.bytes, "chips": report.chips, "tensors": report.tensors, "boot_words": report.boot_words, "seconds": report.seconds }))
        });
        *job.outcome.lock().unwrap() = Some(outcome);
    });
    StatusCode::ACCEPTED.into_response()
}

async fn status(State(app): State<App>) -> Json<serde_json::Value> {
    let name = app.loaded.lock().unwrap().as_ref().map(|l| l.name.clone());
    Json(serde_json::json!({ "loaded": name }))
}

#[derive(Deserialize)]
struct LoadRequest {
    path: String,
    numeric: Option<NumericRequest>,
    /// Runs the die's exact integer arithmetic instead of a software model of it.
    #[serde(default)]
    exact: bool,
    /// The chip's weight format; `None` keeps the checkpoint as it is.
    #[serde(default)]
    format: Option<WeightFormat>,
    /// Runs the large matrix products on this system's GPU when it has one.
    #[serde(default)]
    gpu: bool,
    /// Walks the whole image as the load controller would and checks its checksums.
    #[serde(default)]
    transaction: bool,
}

#[derive(Clone, Copy, Deserialize)]
struct NumericRequest {
    activation_bits: u32,
    state_bits: u32,
    table_bits: Option<u32>,
}

fn load_model(catalog: &Catalog, request: LoadRequest) -> Result<Loaded, String> {
    let LoadRequest { path, numeric, exact, format, gpu, transaction } = request;
    let path = path.as_str();
    let dir = Path::new(path);
    let entry = catalog.find(path).ok_or("the model is not in the known folders")?;
    if !entry.supported {
        let reason = entry.adaptation.reason.unwrap_or_else(|| format!("the “{}” architecture cannot run yet", entry.architecture));
        return Err(format!("{}: {reason}", entry.name));
    }
    let config = std::fs::read_to_string(dir.join("config.json")).map_err(|e| e.to_string())?;
    let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| e.to_string())?;
    let mode = match numeric {
        None => Numeric::Float,
        Some(n) => Numeric::Block { activation_bits: n.activation_bits.clamp(4, 24), state_bits: n.state_bits.clamp(4, 24), table_bits: n.table_bits.map(|b| b.clamp(4, 16)) },
    };
    let exact_bits = exact.then(|| numeric.map_or(16, |n| n.activation_bits.clamp(8, 24)));
    // On the die's arithmetic, the chip carries the model's image encoder, Qwen's, with it.
    let janus = serde_json::from_str::<serde_json::Value>(&config).map_err(|e| e.to_string())?["model_type"] == "janus";
    let carries_encoder = exact_bits.is_some() && !janus && sim::vision::VisionConfig::read(dir)?.is_some();
    let weights = if carries_encoder { Weights::open_with_vision(dir)? } else { Weights::open(dir)? }.with_format(format);
    let weights = if gpu { weights.with_gpu() } else { weights };
    let accelerator = weights.accelerator_name();
    let load_report = if transaction {
        let manifest_path = home().join(".llm2chip/images").join(format!("{}.json", entry.name.replace('/', "--")));
        let (manifest, created) = load_or_build(&weights, &manifest_path)?;
        let mut report = verify(&weights, &manifest, false);
        report.created = created;
        Some(report)
    } else {
        None
    };
    let program = compile_model(&config)?;
    let program = if carries_encoder { compiler::attach_vision(program, &config, CHIP_IMAGE_PATCHES)? } else { program };
    let prompt = prompt::read(dir, &program.family, &tokenizer);
    let engine: Box<dyn Runner> = match exact_bits {
        Some(bits) => Box::new(ExactEngine::new(program, weights, bits)?),
        None => Box::new(Engine::new(program, weights, mode)?),
    };
    Ok(Loaded { name: entry.name, engine, tokenizer, accelerator, dir: dir.to_path_buf(), firmware_tables: None, load_report, prompt, vision: None, painter: None, exact_bits, talk: Talk::default() })
}

async fn load(State(app): State<App>, Json(request): Json<LoadRequest>) -> Response {
    let result = tokio::task::spawn_blocking(move || {
        let loaded = load_model(&app.catalog, request)?;
        let reply = (loaded.name.clone(), loaded.accelerator.clone(), loaded.load_report.clone());
        *app.loaded.lock().unwrap() = Some(loaded);
        Ok::<_, String>(reply)
    })
    .await;
    match result {
        Ok(Ok((name, accelerator, load_report))) => Json(serde_json::json!({ "loaded": name, "accelerator": accelerator, "load_report": load_report })).into_response(),
        Ok(Err(e)) => (StatusCode::BAD_REQUEST, e).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    }
}

#[derive(Deserialize)]
struct GenerateRequest {
    prompt: String,
    /// The project's fixed system prompt, which the chip runs before every question.
    #[serde(default)]
    system: String,
    max_tokens: usize,
    temperature: f32,
    top_k: usize,
    #[serde(default)]
    thinking: bool,
    /// The most tokens the model may reason before its reasoning is closed for it.
    #[serde(default)]
    think_budget: Option<u32>,
    /// A picture the question is about, as RGB pixels row by row.
    #[serde(default)]
    image: Option<ImageInput>,
    /// Draw a picture of the prompt instead of answering it, with a model that draws.
    #[serde(default)]
    draw: bool,
    /// Also runs the draft head and reports how often it guessed right.
    #[serde(default)]
    mtp_depth: usize,
    /// How much of the chip around the die to simulate.
    #[serde(default)]
    fidelity: Fidelity,
    /// The question goes on from the last answer, over the cache, as a chip that holds a
    /// conversation takes its next turn; otherwise the cache is emptied first, as the clear
    /// pin does.
    #[serde(default)]
    conversation: bool,
}

#[derive(Deserialize)]
struct ImageInput {
    width: usize,
    height: usize,
    rgb: Vec<u8>,
}

#[derive(Clone, Copy, Default, Deserialize)]
struct Fidelity {
    /// The chip's real firmware runs on an instruction-level CPU and drives the engine.
    #[serde(default)]
    firmware: bool,
    /// Sampling goes through the model of the chip's unit instead of software.
    #[serde(default)]
    sampler_unit: bool,
    /// How many cycles the die takes per token, from the design being evaluated.
    #[serde(default)]
    cycles_per_token: f64,
    /// Logits the sampling unit takes in per cycle, from the design being evaluated.
    #[serde(default)]
    sampler_lanes: u32,
    /// The firmware turns the text into tokens instead of receiving them ready.
    #[serde(default)]
    tokenizer: bool,
    /// The context of the design being evaluated, which sizes the firmware's memory.
    #[serde(default)]
    context_tokens: u64,
}

async fn generate_socket(ws: WebSocketUpgrade, State(app): State<App>) -> Response {
    ws.on_upgrade(move |socket| handle_socket(socket, app))
}

async fn handle_socket(mut socket: WebSocket, app: App) {
    let Some(Ok(Message::Text(text))) = socket.recv().await else { return };
    let Ok(request) = serde_json::from_str::<GenerateRequest>(&text) else {
        let _ = socket.send(Message::Text(r#"{"type":"error","message":"invalid request"}"#.into())).await;
        return;
    };

    let (sender, mut receiver) = mpsc::channel::<serde_json::Value>(64);
    let cancelled = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let flag = cancelled.clone();
    let worker = tokio::task::spawn_blocking(move || {
        if let Err(message) = run_generation(&app, request, &sender, &flag) {
            let _ = sender.blocking_send(serde_json::json!({ "type": "error", "message": message }));
        }
    });

    loop {
        tokio::select! {
            event = receiver.recv() => match event {
                Some(event) => {
                    if socket.send(Message::Text(event.to_string().into())).await.is_err() {
                        break;
                    }
                }
                None => break,
            },
            incoming = socket.recv() => {
                if matches!(incoming, None | Some(Err(_)) | Some(Ok(Message::Close(_)))) {
                    break;
                }
            }
        }
    }
    cancelled.store(true, std::sync::atomic::Ordering::Relaxed);
    let _ = worker.await;
}

const SAMPLER_ENTRIES: u32 = 64;
const FLOW_BITS: u32 = 16;
/// Measured on Qwen3.8: past 128 lanes, too many loops are shorter than a block.
const MAX_IMAGE_LANE_BITS: u32 = 7;
const MAX_UNIT_BITS: u32 = 8;

fn one_lane() -> u32 {
    1
}

fn sampler_model(vocabulary: usize, seed: u32, lanes: u32) -> SamplerModel {
    let config = SamplerConfig { logit_bits: FLOW_BITS, index_bits: usize::BITS - vocabulary.leading_zeros(), entries: SAMPLER_ENTRIES, lanes: lanes.max(1).next_power_of_two() };
    SamplerModel { config, rng: seed.max(1), table: exponential_table() }
}

/// The same choice the chip's sampling unit would make for these logits.
fn sample_on_unit(unit: &mut SamplerModel, logits: &[f32], temperature: f32, top_k: usize) -> u32 {
    let values: Vec<f64> = logits.iter().map(|v| *v as f64).collect();
    let (mantissas, exponent) = encode_vector(&values, FLOW_BITS);
    let inverse = (2f64.powi(exponent as i32 + 1 + 16 - FLOW_BITS as i32) / temperature.max(1e-3) as f64).min(65535.0) as u32;
    unit.sample(&mantissas, inverse, top_k.clamp(1, SAMPLER_ENTRIES as usize), temperature <= 0.0).token
}

/// The bytes after `shown` that already form complete UTF-8 characters.
fn take_text(bytes: &[u8], shown: &mut usize) -> String {
    let pending = &bytes[*shown..];
    let valid = match std::str::from_utf8(pending) {
        Ok(_) => pending.len(),
        Err(e) => e.valid_up_to(),
    };
    *shown += valid;
    String::from_utf8_lossy(&pending[..valid]).into_owned()
}

/// The first `depth` counters the firmware reported for the draft head.
fn draft_counts(counts: &[u32; 4], depth: usize) -> Vec<u32> {
    counts[..depth.min(4)].to_vec()
}

/// The die's routines the firmware starts, by their entry number.
const ROUTINE_NAMES: [&str; 6] = ["token", "logits", "feed_from_main", "feed_from_draft", "draft", "draft_logits"];

/// What a step did, for the simulator: the CPU's cycles and instructions by task, and the
/// runs and cycles of each routine of the die and of the sampling unit.
fn activity_json(a: &netlist::activity::ChipActivity) -> serde_json::Value {
    let cpu: serde_json::Map<String, serde_json::Value> = a.cpu.entries().into_iter().map(|(task, cycles, instructions)| (serde_json::to_value(task).unwrap().as_str().unwrap().to_string(), serde_json::json!([cycles, instructions]))).collect();
    let routines: serde_json::Map<String, serde_json::Value> = a.routines.iter().enumerate().filter(|(_, r)| r.0 > 0).map(|(i, r)| (ROUTINE_NAMES.get(i).map_or_else(|| i.to_string(), |n| n.to_string()), serde_json::json!([r.0, r.1]))).collect();
    serde_json::json!({ "cpu": cpu, "routines": routines, "samples": [a.samples.0, a.samples.1] })
}

/// An image for the firmware: its tokens and their places, as the engine reads them, and
/// what the host sends the chip with the chip's description of its encoder.
type ImageForFirmware = (ImageTokens, Placement, u32, (Vec<u8>, netlist::chip::VisionInfo));

/// Where a question stands in the conversation the simulator holds: whether it goes on
/// from the last answer, and what the conversation is held with.
struct Turn {
    going_on: bool,
    signature: String,
}

/// Runs the chip's real firmware at instruction level around the loaded engine. Without a
/// picture the chip holds a conversation: it is kept, with the engine, waiting for the
/// next turn, and a new conversation on the same chip raises its clear pin.
#[allow(clippy::too_many_arguments)]
fn run_firmware(loaded: &mut Loaded, text: &str, add_special: bool, ids: &[u32], request: &GenerateRequest, sender: &mpsc::Sender<serde_json::Value>, cancelled: &AtomicBool, picture: Option<ImageForFirmware>, turn: Turn) -> Result<(), String> {
    let use_tables = request.fidelity.tokenizer;
    let system = request.system.trim();
    // What the chip carries and runs before the host's text, and what the host sends. The
    // system prompt's tokens go in the chip's description, with the special tokens, which
    // open a conversation once and not every turn; a turn that goes on sends only its own.
    let (carried, rest, sent) = if turn.going_on {
        (Vec::new(), text.to_string(), ids.to_vec())
    } else {
        let tokenizer = &loaded.tokenizer;
        let encode = |text: &str, special: bool| tokenizer.encode(text, special).map(|e| e.get_ids().to_vec()).map_err(|e| e.to_string());
        if !system.is_empty() {
            let (tokens, rest) = loaded.prompt.split_system(system, text, picture.is_some(), add_special, encode)?;
            let sent = ids[tokens.len()..].to_vec();
            (tokens, rest, sent)
        } else if add_special && picture.is_none() {
            let bare = encode(text, false)?;
            match ids.len().checked_sub(bare.len()).filter(|at| ids[*at..] == bare[..]) {
                Some(at) => (ids[..at].to_vec(), text.to_string(), bare),
                None => (Vec::new(), text.to_string(), ids.to_vec()),
            }
        } else {
            (Vec::new(), text.to_string(), ids.to_vec())
        }
    };
    // The chip that held the conversation, if it is this same chip.
    let kept = match loaded.talk.chip.take() {
        Some(chip) if chip.signature == turn.signature && picture.is_none() => Some(chip),
        other => {
            // Another chip: its engine comes back, emptied.
            if other.is_some() {
                loaded.talk.chip = other;
                loaded.reclaim();
                loaded.engine.reset();
            }
            None
        }
    };
    if turn.going_on && kept.is_none() {
        return Err("the chip that held the conversation is gone: start a new one".into());
    }
    // The firmware's tokenizer adds the special tokens itself only when they did not go first.
    let special = !turn.going_on && carried.is_empty() && add_special;
    let context = if request.fidelity.context_tokens > 0 { request.fidelity.context_tokens } else { DEFAULT_CONTEXT };
    let prompt: Vec<u8> = if use_tables {
        rest.bytes().chain([0]).collect()
    } else {
        sent.iter().flat_map(|t| t.to_le_bytes()).chain(u32::MAX.to_le_bytes()).collect()
    };
    let clear = kept.is_some() && !turn.going_on;
    let (mut chip, input) = match kept {
        Some(chip) => (chip, prompt),
        None => build_chip(loaded, &carried, special, context, request, picture, &turn.signature, prompt)?,
    };
    let outcome = run_turn(&mut chip, &loaded.tokenizer, &input, ids, request, sender, cancelled, clear, context);
    match outcome {
        // A chip that holds a conversation waits for the next turn with the engine.
        Ok(Some(held)) if chip.conversation => loaded.talk = Talk { held, signature: turn.signature, chip: Some(chip) },
        _ => {
            if let Some(engine) = chip.bus.peripheral.engine.into_engine() {
                loaded.engine = engine;
            }
        }
    }
    outcome.map(|_| ())
}

/// A chip at instruction level around the loaded engine, which it keeps, and what the host
/// sends it first: a picture, if there is one, before the prompt.
#[allow(clippy::too_many_arguments)]
fn build_chip(loaded: &mut Loaded, carried: &[u32], special: bool, context: u64, request: &GenerateRequest, picture: Option<ImageForFirmware>, signature: &str, prompt: Vec<u8>) -> Result<(KeptChip, Vec<u8>), String> {
    if request.mtp_depth > 0 && !loaded.engine.has_draft_head() {
        return Err("this model has no MTP head".into());
    }
    let firmware = FirmwareImage::from_elf(FIRMWARE)?;
    let vocabulary = loaded.engine.vocabulary();
    let cell = CellConfig { weight_bits: 8, activation_bits: FLOW_BITS, accumulator_bits: 32, mac_stages: 1, adder_stages: 1, addr_bits: 3, coding: Default::default(), exponent_bits: 0, multiplier: netlist::multiplier::MultiplierKind::Array, fixed_weights: false, ecc: false, skip_zeros: false };
    let die = DieConfig { cells: 1, cell, scale_bits: 16, reg_bits: FLOW_REGISTER_BITS, index_bits: FLOW_INDEX_BITS, microcode_bits: 2, broadcast_stages: 1, function_bits: 12, function_select_bits: 3, features: Default::default() };
    // With an image, the routines that take it: its tokens, each patch and their fusion.
    let routines = if picture.is_some() { sim::lower::ENTRY_VISION as u32 + 2 } else { 6 };
    let code = Microprogram { words: Vec::new(), entries: (0..routines).collect(), parameters: vec![Parameter { register: 1, token: 1, position: 0, constant: 0, window_length: 0, window_slot: 0, window: 0 }, Parameter { register: 2, token: 0, position: 1, constant: 0, window_length: 0, window_slot: 0, window: 0 }], group: (0, 0), marks: Vec::new() };
    let seed = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(1, |d| d.as_nanos() as u32);
    let use_tables = request.fidelity.tokenizer;
    let setup = GenerationSetup {
        max_new: request.max_tokens as u32,
        greedy: request.temperature <= 0.0,
        limit: request.top_k.clamp(1, SAMPLER_ENTRIES as usize) as u32,
        temperature: (request.temperature.max(0.0) * 256.0).round().max(1.0) as u32,
        seed,
        raw_tokens: !use_tables,
        stop: loaded.prompt.stop.clone(),
        draft_depth: request.mtp_depth.min(3) as u32,
        draft_logits: FLOW_DRAFT_REGISTER << FLOW_INDEX_BITS,
        think: loaded.prompt.think,
        think_budget: request.think_budget.unwrap_or(u32::MAX),
    };
    let mut info = chip_info(&die, &code, FLOW_LOGITS_REGISTER, vocabulary as u32, &setup);
    if !carried.is_empty() {
        netlist::chip::add_system_prompt(&mut info, carried)?;
    }
    let conversation = picture.is_none();
    if conversation {
        // It waits for the next turn instead of stopping, and its clear pin empties the
        // engine's cache.
        netlist::chip::add_conversation(&mut info, context.min(u32::MAX as u64) as u32, &sim::chip::runner_conversation())?;
    }
    if let Some((.., (_, vision))) = &picture {
        netlist::chip::add_vision(&mut info, vision);
    }
    if use_tables && loaded.firmware_tables.as_ref().is_none_or(|(built_for, _)| *built_for != (context, special)) {
        let mut tables = TokenizerTables::from_file(&loaded.dir.join("tokenizer.json"), context)?;
        tables.set_add_special(special);
        loaded.firmware_tables = Some(((context, special), tables.to_words()));
    }
    let tables: &[u32] = match &loaded.firmware_tables {
        Some((_, words)) if use_tables => words,
        _ => &[],
    };
    let image = build_image(&firmware.program, &firmware.data, &info, tables, &[])?;
    // The memory the chip needs for this model's tables and its image layout.
    let data_bits = (image.len().max(1 << 13).next_power_of_two().ilog2()).max(13);
    let soc = SocConfig { program_bits: 14, data_bits, fifo_bits: 6 };
    // The image must boot before the engine goes into the chip.
    SocBus::booted(soc, &image, &[])?;
    let unit = sampler_model(vocabulary, seed, request.fidelity.sampler_lanes);
    let engine = std::mem::replace(&mut loaded.engine, Box::new(Lent));
    let mut runner = RunnerEngine::keeping(engine, FLOW_BITS, request.fidelity.cycles_per_token.max(0.0) as u64);
    // The image goes first, before the prompt that places it.
    let input = match picture {
        Some((tokens, placement, placeholder, (bytes, _))) => {
            runner = runner.with_image(tokens, placement, placeholder);
            bytes.into_iter().chain(prompt).collect()
        }
        None => prompt,
    };
    let bus = SocBus::booted_with(soc, &image, &[], ChipModel::new(runner, code, unit))?;
    // What the CPU is, for the simulator to show beside what it does.
    let about = serde_json::json!({ "type": "cpu", "isa": "RV32IM", "program_bytes": 4u64 << soc.program_bits, "data_bytes": 4u64 << soc.data_bits, "firmware_bytes": 4 * firmware.program.len(), "image_bytes": 4 * image.len(), "tokenizer": use_tables });
    let chip = KeptChip { bus, cpu: Rv32Iss::default(), profile: netlist::activity::CpuProfile::from_elf(FIRMWARE), signature: signature.to_string(), raw: !use_tables, conversation, about };
    Ok((chip, input))
}

/// One turn on a chip: the clear pin first if `clear`, then `input`, until the chip ends its
/// answer (or stops, without a conversation). Gives the tokens its cache holds then, or
/// nothing if the run was cancelled.
#[allow(clippy::too_many_arguments)]
fn run_turn(chip: &mut KeptChip, tokenizer: &Tokenizer, input: &[u8], ids: &[u32], request: &GenerateRequest, sender: &mpsc::Sender<serde_json::Value>, cancelled: &AtomicBool, clear: bool, context: u64) -> Result<Option<usize>, String> {
    let KeptChip { bus, cpu, profile, raw, about, conversation, .. } = chip;
    let raw = *raw;
    let _ = sender.blocking_send(about.clone());
    if clear {
        // The clear pin, while the chip waits: it empties the cache and the recurrent state,
        // and reads its system prompt again when the next turn comes.
        let from = cpu.cycle;
        bus.clear();
        while bus.clearing() {
            if cancelled.load(Relaxed) {
                return Ok(None);
            }
            let pc = cpu.pc;
            cpu.step(bus);
            profile.record(pc, bus.last_access.take(), cpu.cycle);
            if cpu.halted {
                return Err("the chip stopped instead of clearing its conversation".into());
            }
        }
        let _ = sender.blocking_send(serde_json::json!({ "type": "cleared", "cycles": cpu.cycle - from }));
    }
    bus.send(input);
    let engine = &bus.peripheral.engine;
    let (first_cycle, first_die) = (cpu.cycle, engine.die_cycles);
    let (steps, generated, tokens, start) = (engine.steps, engine.generated, engine.tokens.len(), bus.tx.len());
    let mut last = ChipActivity::of(profile, &bus.peripheral);
    let (mut emitted, mut reported, mut shown) = (0, 0, start);
    let mut output: Vec<u32> = Vec::new();
    let mut decoded_shown = 0;
    // The text the chip writes ends with a zero byte, its tokens with `END_OF_TOKENS`.
    let mut ended = false;
    while !cpu.halted && !ended {
        if cancelled.load(Relaxed) {
            return Ok(None);
        }
        // A token's step ends when the firmware orders the next one, so it is cut just before
        // that order: the order and its run belong to the next step.
        let before = ChipActivity::of(profile, &bus.peripheral);
        let pc = cpu.pc;
        cpu.step(bus);
        profile.record(pc, bus.last_access.take(), cpu.cycle);
        if let Some(error) = bus.peripheral.engine.error.take() {
            return Err(error);
        }
        while reported < bus.peripheral.engine.steps - steps && reported < ids.len() {
            reported += 1;
            let _ = sender.blocking_send(serde_json::json!({ "type": "prefill", "done": reported, "total": ids.len(), "seconds": bus.peripheral.engine.last_seconds, "cycles": cpu.cycle, "activity": activity_json(&before.since(&last)) }));
            last = before;
        }
        if !raw {
            let end = bus.tx[start..].iter().position(|b| *b == 0).map(|at| start + at);
            ended = end.is_some();
            let text = &bus.tx[..end.unwrap_or(bus.tx.len())];
            // A token's text is complete once the firmware asks for the next token's logits.
            while emitted + 1 < bus.peripheral.engine.generated - generated {
                emitted += 1;
                let piece = take_text(text, &mut shown);
                let _ = sender.blocking_send(serde_json::json!({ "type": "token", "text": piece, "seconds": bus.peripheral.engine.last_seconds, "saturated": 0, "index": emitted, "cycles": cpu.cycle, "hits": draft_counts(&bus.peripheral.draft_hits, request.mtp_depth), "tries": draft_counts(&bus.peripheral.draft_tries, request.mtp_depth), "activity": activity_json(&before.since(&last)) }));
                last = before;
            }
            continue;
        }
        while !ended && (bus.tx.len() - start) / 4 > emitted {
            let at = start + emitted * 4;
            let token = u32::from_le_bytes([bus.tx[at], bus.tx[at + 1], bus.tx[at + 2], bus.tx[at + 3]]);
            if token == u32::MAX {
                ended = true;
                break;
            }
            output.push(token);
            emitted += 1;
            let decoded = tokenizer.decode(&output, true).map_err(|e| e.to_string())?;
            let piece = if decoded.ends_with('\u{FFFD}') { String::new() } else { let part = decoded[decoded_shown..].to_string(); decoded_shown = decoded.len(); part };
            // Here the step ends with the token written to the host, this very instruction.
            let now = ChipActivity::of(profile, &bus.peripheral);
            let _ = sender.blocking_send(serde_json::json!({ "type": "token", "id": token, "text": piece, "seconds": bus.peripheral.engine.last_seconds, "saturated": 0, "index": emitted, "cycles": cpu.cycle, "hits": draft_counts(&bus.peripheral.draft_hits, request.mtp_depth), "tries": draft_counts(&bus.peripheral.draft_tries, request.mtp_depth), "activity": activity_json(&now.since(&last)) }));
            last = now;
        }
    }
    profile.settle();
    if !raw {
        let end = bus.tx[start..].iter().position(|b| *b == 0).map_or(bus.tx.len(), |at| start + at);
        let piece = take_text(&bus.tx[..end], &mut shown);
        if !piece.is_empty() {
            emitted += 1;
            let now = ChipActivity::of(profile, &bus.peripheral);
            let _ = sender.blocking_send(serde_json::json!({ "type": "token", "text": piece, "seconds": bus.peripheral.engine.last_seconds, "saturated": 0, "index": emitted, "cycles": cpu.cycle, "hits": draft_counts(&bus.peripheral.draft_hits, request.mtp_depth), "tries": draft_counts(&bus.peripheral.draft_tries, request.mtp_depth), "activity": activity_json(&now.since(&last)) }));
            last = now;
        }
    }
    let runner = &bus.peripheral.engine;
    let fed = &runner.tokens[tokens..];
    if !raw && fed.is_empty() && !ids.is_empty() {
        return Err("the text does not fit in the chip tokenizer's working memory".into());
    }
    let tokenizer_match = (!raw).then(|| fed.starts_with(ids));
    // A chip without a conversation keeps nothing for the next question.
    let held = if *conversation { runner.cache_tokens() } else { 0 };
    // Short of the limit, the firmware stopped by its own rules: an end token, a loop or a
    // full cache.
    let reason = if emitted >= request.max_tokens {
        "limit"
    } else if runner.cache_tokens() as u64 >= context {
        "context"
    } else {
        "chip"
    };
    let now = ChipActivity::of(profile, &bus.peripheral);
    let _ = sender.blocking_send(serde_json::json!({ "type": "done", "tokens": emitted, "reason": reason, "held": held, "chip": { "cycles": cpu.cycle - first_cycle, "die_cycles": runner.die_cycles - first_die, "tokenizer_match": tokenizer_match }, "activity": activity_json(&now.since(&last)), "totals": activity_json(&now) }));
    Ok(Some(held))
}

/// Base64, for a picture sent inside a message.
fn base64(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let n = chunk.iter().enumerate().fold(0u32, |n, (i, b)| n | (*b as u32) << (16 - 8 * i));
        for i in 0..4 {
            out.push(if i <= chunk.len() { DIGITS[(n >> (18 - 6 * i) & 63) as usize] as char } else { '=' });
        }
    }
    out
}

/// An uncompressed 24-bit BMP, which browsers show as it is.
fn bmp(rgb: &[u8], side: usize) -> Vec<u8> {
    let row = (side * 3).next_multiple_of(4);
    let mut out = Vec::with_capacity(54 + row * side);
    out.extend_from_slice(b"BM");
    out.extend_from_slice(&((54 + row * side) as u32).to_le_bytes());
    out.extend_from_slice(&[0; 4]);
    out.extend_from_slice(&54u32.to_le_bytes());
    out.extend_from_slice(&40u32.to_le_bytes());
    out.extend_from_slice(&(side as i32).to_le_bytes());
    out.extend_from_slice(&(-(side as i32)).to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&24u16.to_le_bytes());
    out.extend_from_slice(&[0; 24]);
    for y in 0..side {
        for p in rgb[y * side * 3..(y + 1) * side * 3].chunks(3) {
            out.extend_from_slice(&[p[2], p[1], p[0]]);
        }
        out.resize(out.len() + row - side * 3, 0);
    }
    out
}

/// Draws the prompt with the loaded model's image generator: its language model writes
/// the codebook tokens, then its decoder turns them into pixels.
fn run_drawing(loaded: &mut Loaded, request: &GenerateRequest, sender: &mpsc::Sender<serde_json::Value>, cancelled: &std::sync::atomic::AtomicBool) -> Result<(), String> {
    use std::sync::atomic::Ordering::Relaxed;
    if loaded.painter.is_none() {
        loaded.painter = Some(sim::janus::Janus::open(&loaded.dir)?.ok_or("this model does not generate images")?);
    }
    let painter = loaded.painter.as_ref().unwrap();
    let text = format!("<|User|>: {}\n\n<|Assistant|>:", request.prompt);
    let mut ids = vec![painter.config.bos];
    ids.extend(loaded.tokenizer.encode(text.as_str(), false).map_err(|e| e.to_string())?.get_ids());
    ids.push(painter.config.begin_image);
    let _ = sender.blocking_send(serde_json::json!({ "type": "start", "prompt_tokens": ids.len() }));
    let seed = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(1, |d| d.as_nanos() as u64);
    let started = std::time::Instant::now();
    let tokens = painter.draw(&ids, request.temperature.max(0.0), seed, loaded.exact_bits, |done, total| {
        let _ = sender.blocking_send(serde_json::json!({ "type": "drawing", "done": done, "total": total, "seconds": started.elapsed().as_secs_f64() }));
    })?;
    if cancelled.load(Relaxed) {
        return Ok(());
    }
    let _ = sender.blocking_send(serde_json::json!({ "type": "decoding" }));
    let rgb = painter.decode(&tokens)?;
    let side = painter.config.side();
    let _ = sender.blocking_send(serde_json::json!({ "type": "picture", "side": side, "tokens": tokens.len(), "seconds": started.elapsed().as_secs_f64(), "data": format!("data:image/bmp;base64,{}", base64(&bmp(&rgb, side))) }));
    let _ = sender.blocking_send(serde_json::json!({ "type": "done", "tokens": tokens.len() }));
    Ok(())
}

fn run_generation(app: &App, request: GenerateRequest, sender: &mpsc::Sender<serde_json::Value>, cancelled: &std::sync::atomic::AtomicBool) -> Result<(), String> {
    use std::sync::atomic::Ordering::Relaxed;
    let mut guard = app.loaded.lock().unwrap();
    let loaded = guard.as_mut().ok_or("load a model first")?;
    if request.draw {
        loaded.reclaim();
        loaded.talk = Talk::default();
        return run_drawing(loaded, &request, sender, cancelled);
    }
    let system = request.system.trim();
    let fidelity = request.fidelity;
    // What a conversation is held with: the same system prompt, and through the firmware
    // the same chip, whose image carries the generation's settings.
    let signature = if fidelity.firmware {
        serde_json::json!(["firmware", system, request.max_tokens, request.temperature, request.top_k, request.think_budget, request.mtp_depth, fidelity.cycles_per_token, fidelity.sampler_lanes, fidelity.tokenizer, fidelity.context_tokens])
    } else {
        serde_json::json!(["engine", system, request.mtp_depth])
    }
    .to_string();
    // A question about a picture starts a conversation of its own and ends there.
    let going_on = request.conversation && request.image.is_none() && loaded.talk.held > 0 && loaded.talk.signature == signature;
    let before = if going_on { loaded.talk.held } else { 0 };
    let (text, add_special) = match request.image {
        Some(_) => (loaded.prompt.render_with_image(system, &request.prompt, request.thinking)?, false),
        None if going_on => (loaded.prompt.next_turn(system, &request.prompt, request.thinking)?, false),
        None => loaded.prompt.render_with_system(system, &request.prompt, request.thinking)?,
    };
    let ids = loaded.tokenizer.encode(text.as_str(), add_special).map_err(|e| e.to_string())?.get_ids().to_vec();
    // The engine leaves a chip that held a conversation unless the chip goes on with it.
    if request.image.is_some() || !fidelity.firmware {
        loaded.reclaim();
    }
    if !going_on && loaded.talk.chip.is_none() {
        loaded.engine.reset();
    }
    // The image becomes the tokens that take the place of its `<|image_pad|>`.
    let image = match &request.image {
        Some(input) => {
            if request.mtp_depth > 0 {
                return Err("images do not go through the MTP head yet".into());
            }
            if loaded.vision.is_none() {
                loaded.vision = Some(Seer::open(&loaded.dir)?.ok_or("this model has no image encoder")?);
            }
            let started = std::time::Instant::now();
            let plan = match loaded.vision.as_ref().unwrap() {
                Seer::Qwen(tower) => {
                    loaded.engine.set_mrope(tower.config.mrope_section);
                    // A chip that carries the encoder runs it in its own arithmetic and keeps
                    // the tokens; otherwise the host encodes in floating point.
                    let config = loaded.engine.image_patches().map_or_else(|| tower.config.clone(), |most| tower.config.within(most));
                    let patches = sim::vision::patches(&config, &input.rgb, input.width, input.height)?;
                    let tokens = match loaded.engine.encode_image(tower, &patches)? {
                        Some(count) => ImageTokens::OnChip(count),
                        None => ImageTokens::Vectors(tower.encode(&patches)?),
                    };
                    // Through the firmware, the host sends the patches over the chip's port.
                    let wire = (sim::vision_chip::picture_bytes(tower, Some(&patches), FLOW_BITS)?, sim::vision_chip::software_vision_info(&tower.config, patches.rows * patches.cols, FLOW_INDEX_BITS, FLOW_REGISTER_BITS));
                    (tokens, patches.rows * patches.cols, Placement::Grid(patches.rows, patches.cols, tower.config.merge), tower.config.image_token, wire)
                }
                // Janus reads its patches one after another, as text.
                Seer::Janus(eye) => {
                    let tokens = eye.encode(&input.rgb, input.width, input.height)?;
                    let wire = line_wire(tokens.len(), eye.placeholder);
                    (ImageTokens::Vectors(tokens), eye.tokens(), Placement::Line, eye.placeholder, wire)
                }
            };
            let on_chip = matches!(plan.0, ImageTokens::OnChip(_));
            let _ = sender.blocking_send(serde_json::json!({ "type": "image", "patches": plan.1, "tokens": plan.0.len(), "seconds": started.elapsed().as_secs_f64(), "on_chip": on_chip }));
            Some((plan.0, plan.2, plan.3, plan.4))
        }
        None => None,
    };
    let image_tokens = image.as_ref().map_or(0, |(tokens, ..)| tokens.len().saturating_sub(1));
    // The chip remembers its context and no more: the prompt must fit, and the answer ends there.
    let context = request.fidelity.context_tokens as usize;
    if context > 0 && before + ids.len() + image_tokens >= context {
        return Err(if going_on {
            format!("the conversation takes {before} tokens and the question {} more, and the chip remembers {context}: start a new conversation", ids.len())
        } else {
            format!("the prompt takes {} tokens and the chip remembers {context}: raise the design's context", ids.len() + image_tokens)
        });
    }
    let _ = sender.blocking_send(serde_json::json!({ "type": "start", "prompt_tokens": ids.len() + image_tokens, "held": before }));
    // Until this turn ends well, the conversation is left halfway.
    loaded.talk.held = 0;
    if fidelity.firmware {
        return run_firmware(loaded, &text, add_special, &ids, &request, sender, cancelled, image, Turn { going_on, signature });
    }

    let mut unit = sampler_model(loaded.engine.vocabulary(), 0x1234_5679, request.fidelity.sampler_lanes);
    let use_unit = request.fidelity.sampler_unit;
    let mut seed = 0x9E3779B97F4A7C15u64 ^ std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(1, |d| d.as_nanos() as u64);
    let mut next = 0;
    let mut reasoning = prompt::ReasoningBudget::new(loaded.prompt.think, request.think_budget);
    for (i, id) in ids.iter().enumerate() {
        reasoning.see(*id);
        if cancelled.load(Relaxed) {
            return Ok(());
        }
        let last = i + 1 == ids.len();
        let wants_mtp = request.mtp_depth > 0;
        if let Some((tokens, layout, ..)) = image.as_ref().filter(|image| *id == image.2) {
            // In a grid, the image's tokens all take the position the placeholder had, plus
            // their row and column, and the text goes on past the longer side; in a line,
            // each takes the next position.
            let (positions, after) = layout.positions(i as u64, tokens.len());
            for (index, position) in positions.into_iter().enumerate().take(tokens.len()) {
                match tokens {
                    ImageTokens::Vectors(vectors) => loaded.engine.step_embedding(&vectors[index], position, false)?,
                    ImageTokens::OnChip(_) => loaded.engine.step_image(index as u32, position, false)?,
                };
            }
            loaded.engine.continue_text_at(after);
            let _ = sender.blocking_send(serde_json::json!({ "type": "prefill", "done": i + 1 + image_tokens, "total": ids.len() + image_tokens, "seconds": 0.0 }));
            continue;
        }
        let (logits, stats) = loaded.engine.step(*id, last || wants_mtp)?;
        if last {
            if let Some(l) = logits {
                next = reasoning.steer(if use_unit { sample_on_unit(&mut unit, l, request.temperature, request.top_k) } else { sample(l, request.temperature, request.top_k, &mut seed) });
            }
        }
        if wants_mtp && loaded.engine.has_draft_head() {
            let following = if last { next } else { ids[i + 1] };
            loaded.engine.feed_draft_from_main();
            loaded.engine.draft(following, false)?;
        }
        let done = if image.as_ref().is_some_and(|image| ids[..i].contains(&image.2)) { i + 1 + image_tokens } else { i + 1 };
        let _ = sender.blocking_send(serde_json::json!({ "type": "prefill", "done": done, "total": ids.len() + image_tokens, "seconds": stats.seconds }));
    }

    if request.mtp_depth > 0 && !loaded.engine.has_draft_head() {
        return Err("this model has no MTP head".into());
    }
    let mut output: Vec<u32> = Vec::new();
    let mut shown = 0;
    let (mut hits, mut tries) = (vec![0u32; request.mtp_depth], vec![0u32; request.mtp_depth]);
    let mut pending: Vec<(usize, Vec<u32>)> = Vec::new();
    // The chip's own rule: its firmware stops a text that goes round in circles.
    let mut watch = tokenizer_rt::LoopWatch::default();
    let mut reason = "limit";
    for _ in 0..request.max_tokens {
        if cancelled.load(Relaxed) {
            reason = "cancelled";
            break;
        }
        if loaded.prompt.stop.contains(&next) {
            reason = "stop";
            break;
        }
        if context > 0 && before + ids.len() + image_tokens + output.len() >= context {
            reason = "context";
            break;
        }
        output.push(next);
        let looping = watch.push(next);
        let decoded = loaded.tokenizer.decode(&output, true).map_err(|e| e.to_string())?;
        let piece = if decoded.ends_with('\u{FFFD}') { String::new() } else { let part = decoded[shown..].to_string(); shown = decoded.len(); part };
        // The chip writes the token that closes a loop and stops without running it, so the
        // cache a conversation keeps ends before it, as the firmware's does.
        if looping {
            let _ = sender.blocking_send(serde_json::json!({ "type": "token", "id": next, "text": piece, "seconds": 0.0, "saturated": 0, "index": output.len(), "hits": hits, "tries": tries }));
            reason = "loop";
            break;
        }
        let (logits, stats) = loaded.engine.step(next, true)?;
        let saturated = stats.saturated;
        let seconds = stats.seconds;
        let token = next;
        let logits = logits.ok_or("no logits")?;
        next = reasoning.steer(if use_unit { sample_on_unit(&mut unit, logits, request.temperature, request.top_k) } else { sample(logits, request.temperature, request.top_k, &mut seed) });

        if request.mtp_depth > 0 {
            // Score earlier guesses against what the main model actually produced.
            let position = output.len();
            pending.retain(|(from, guesses)| {
                let depth = position - from;
                if depth >= 1 && depth <= guesses.len() {
                    tries[depth - 1] += 1;
                    if guesses[depth - 1] == output[position - 1] {
                        hits[depth - 1] += 1;
                    }
                }
                depth < guesses.len()
            });
            let mark = loaded.engine.draft_tokens();
            let mut guess = next;
            let mut guesses = Vec::new();
            loaded.engine.feed_draft_from_main();
            for _ in 0..request.mtp_depth {
                let logits = loaded.engine.draft(guess, true)?.ok_or("no logits")?;
                guess = logits.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map_or(0, |(i, _)| i as u32);
                guesses.push(guess);
                loaded.engine.feed_draft_from_draft();
            }
            loaded.engine.truncate_draft(mark);
            pending.push((position + 1, guesses));
            loaded.engine.feed_draft_from_main();
            loaded.engine.draft(next, false)?;
        }
        let _ = sender.blocking_send(serde_json::json!({ "type": "token", "id": token, "text": piece, "seconds": seconds, "saturated": saturated, "index": output.len(), "hits": hits, "tries": tries }));
    }
    // The cache holds the conversation for the next question, unless it was left halfway or
    // the question was about a picture.
    let kept = reason != "cancelled" && image.is_none();
    let held = if kept { before + ids.len() + image_tokens + output.len() - usize::from(reason == "loop") } else { 0 };
    if kept {
        loaded.talk = Talk { held, signature, chip: None };
    }
    let _ = sender.blocking_send(serde_json::json!({ "type": "done", "tokens": output.len(), "reason": reason, "held": held }));
    Ok(())
}

/// The image encoder of the loaded model.
enum Seer {
    Qwen(sim::vision::VisionTower),
    Janus(sim::janus::JanusEye),
}

impl Seer {
    fn open(dir: &Path) -> Result<Option<Self>, String> {
        if let Some(eye) = sim::janus::JanusEye::open(dir)? {
            return Ok(Some(Seer::Janus(eye)));
        }
        Ok(sim::vision::VisionTower::open(dir)?.map(Seer::Qwen))
    }
}

/// What the host sends a software die for an image whose tokens are already vectors, one
/// after another: as many patches as tokens, each with nothing in it.
fn line_wire(tokens: usize, placeholder: u32) -> (Vec<u8>, netlist::chip::VisionInfo) {
    let words = [tokens as u32, 1, tokens as u32].into_iter().chain((0..tokens).flat_map(|_| [0, 0]));
    let bytes = words.flat_map(u32::to_le_bytes).collect();
    let exponents = ((1u32 << FLOW_REGISTER_BITS) - 1) << FLOW_INDEX_BITS;
    let info = netlist::chip::VisionInfo { patches: tokens as u32, image_token: placeholder, merge: 1, blocks: 0, pixels: (3 << FLOW_INDEX_BITS, exponents | 3, 0), position: (4 << FLOW_INDEX_BITS, exponents | 4, 0), patch_rope: (5 << FLOW_INDEX_BITS, 0), rope: 0, rope_half: 0, mrope_rows: 0, mrope_cols: 0 };
    (bytes, info)
}

async fn guard(State(app): State<App>, request: Request, next: Next) -> Response {
    let headers = request.headers();
    let origin_ok = headers.get(header::ORIGIN).and_then(|o| o.to_str().ok()).is_none_or(|o| {
        o.starts_with("http://localhost:") || o.starts_with("http://127.0.0.1:")
    });
    let cookie = format!("llm2chip={}", app.token);
    let cookie_ok = headers.get(header::COOKIE).and_then(|c| c.to_str().ok()).is_some_and(|c| c.split("; ").any(|part| part == cookie));
    let token_ok = cookie_ok || headers.get("x-llm2chip-token").and_then(|t| t.to_str().ok()) == Some(app.token.as_str());
    if !origin_ok || !token_ok {
        return StatusCode::FORBIDDEN.into_response();
    }
    next.run(request).await
}

/// Serves the built interface and hands the browser its session cookie.
async fn static_file(State(app): State<App>, uri: axum::http::Uri) -> Response {
    let Some(root) = app.dist.as_ref() else {
        return (StatusCode::NOT_FOUND, "the interface is not built: run npm --prefix web run build").into_response();
    };
    let wanted = uri.path().trim_start_matches('/');
    let wanted = if wanted.is_empty() { "index.html" } else { wanted };
    if wanted.split('/').any(|part| part == "..") {
        return StatusCode::FORBIDDEN.into_response();
    }
    let path = root.join(wanted);
    let Ok(bytes) = tokio::fs::read(&path).await else {
        return StatusCode::NOT_FOUND.into_response();
    };
    let kind = match path.extension().and_then(|e| e.to_str()) {
        Some("html") => "text/html; charset=utf-8",
        Some("js") => "text/javascript",
        Some("css") => "text/css",
        Some("wasm") => "application/wasm",
        Some("svg") => "image/svg+xml",
        _ => "application/octet-stream",
    };
    let cookie = format!("llm2chip={}; Path=/; HttpOnly; SameSite=Strict", app.token);
    ([(header::CONTENT_TYPE, kind.to_string()), (header::SET_COOKIE, cookie)], bytes).into_response()
}

fn dist_directory() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("LLM2CHIP_DIST") {
        return Some(PathBuf::from(path));
    }
    let exe = std::env::current_exe().ok()?;
    let candidate = exe.parent()?.parent()?.parent()?.join("web/dist");
    candidate.join("index.html").exists().then_some(candidate)
}

fn session_token() -> String {
    let dir = home().join(".llm2chip");
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join("session-token");
    let token = format!(
        "{:x}{:x}",
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_nanos()),
        std::process::id() as u128 * 0x9E3779B97F4A7C15
    );
    if std::fs::write(&path, &token).is_err() {
        eprintln!("Could not write the token to {}", path.display());
    }
    token
}

#[tokio::main]
async fn main() {
    let app = App { token: Arc::new(session_token()), loaded: Arc::new(Mutex::new(None)), dist: dist_directory().map(Arc::new), catalog: Arc::new(Catalog::open(home())), hub: Hub::new(home()), image: Arc::new(Mutex::new(None)), layout: Arc::new(Mutex::new(None)), retouch: Arc::new(Mutex::new(None)), optimize: Arc::new(Mutex::new(None)), trace: Arc::new(Mutex::new(None)), replay: Arc::new(Mutex::new(None)), painter: Arc::new(Mutex::new(None)), chip_painter: Arc::new(Mutex::new(None)),
        #[cfg(any(feature = "listen", feature = "speak"))]
        voice: Arc::new(Mutex::new(None)),
        #[cfg(any(feature = "listen", feature = "speak"))]
        voice_plan: Arc::new(Mutex::new(None)),
        #[cfg(any(feature = "listen", feature = "speak"))]
        conversation: Arc::new(Mutex::new(None)),
    };
    let routes = Router::new();
    // Voice only in an engine built with listening, speaking or both.
    #[cfg(any(feature = "listen", feature = "speak"))]
    let routes = routes
        .route("/api/voice", get(voice_status).post(voice_start).layer(axum::extract::DefaultBodyLimit::max(16 << 20)))
        .route("/api/voice/plan", get(voice_plan_status).post(voice_plan_start));
    let routes = routes
        .route("/api/models", get(models))
        .route("/api/status", get(status))
        .route("/api/settings", get(read_settings).put(write_settings))
        .route("/api/hf/search", get(hub_search))
        .route("/api/hf/model", get(hub_details))
        .route("/api/hf/downloads", get(hub_jobs).post(hub_download))
        .route("/api/hf/cancel", post(hub_cancel))
        .route("/api/hf/discard", post(hub_discard))
        .route("/api/lab", post(lab))
        .route("/api/speed", post(speed))
        .route("/api/image", get(image_status).post(image_start))
        .route("/api/image/cancel", post(image_cancel))
        .route("/api/layout", get(layout_status).post(layout_start))
        .route("/api/optimize", get(optimize_status).post(optimize_start))
        .route("/api/optimize/cancel", post(optimize_cancel))
        .route("/api/trace", get(trace_status).post(trace_start))
        .route("/api/trace/design", post(trace_design))
        .route("/api/trace/inspect", post(trace_inspect))
        // A picture of 512 × 512 travels as some 3 MB of JSON, past the default limit.
        .route("/api/retouch", get(retouch_status).post(retouch_start).layer(axum::extract::DefaultBodyLimit::max(32 << 20)))
        .route("/api/retouch/cancel", post(retouch_cancel))
        .route("/api/hf/token", put(hub_set_token).get(hub_token).delete(hub_clear_token))
        .route("/api/load", post(load))
        .route("/api/generate", get(generate_socket))
        .layer(middleware::from_fn_with_state(app.clone(), guard))
        .fallback(static_file)
        .with_state(app);
    let port = engine_port();
    let listener = match tokio::net::TcpListener::bind(("127.0.0.1", port)).await {
        Ok(listener) => listener,
        Err(e) => {
            eprintln!("cannot listen on 127.0.0.1:{port} ({e}): another llm2chip may be running, or set LLM2CHIP_ENGINE_PORT");
            std::process::exit(1);
        }
    };
    println!("llm2chip at http://127.0.0.1:{port}");
    axum::serve(listener, routes).await.expect("the server failed");
}

#[cfg(test)]
mod picture_tests {
    use super::*;

    #[test]
    fn a_picture_travels_as_base64_of_a_bmp() {
        assert_eq!([base64(b"Man"), base64(b"Ma"), base64(b"M"), base64(b"")], ["TWFu", "TWE=", "TQ==", ""]);
        let image = bmp(&[255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255], 2);
        assert_eq!(&image[..2], b"BM");
        assert_eq!(image.len(), 54 + 8 * 2);
        // Blue, green, red: the first pixel is red.
        assert_eq!(&image[54..57], &[0, 0, 255]);
    }
}
