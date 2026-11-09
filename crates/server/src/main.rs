//! Local engine for the app. Listens on 127.0.0.1 and requires the session token.

mod catalog;
mod hub;
mod prompt;

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
use netlist::format::WeightFormat;
use netlist::block::encode_vector;
use netlist::chip::{chip_info, ChipModel, GenerationSetup};
use netlist::image::build_image;
use netlist::rv32::Rv32Iss;
use netlist::sampler::{exponential_table, SamplerConfig, SamplerModel};
use netlist::soc::{FirmwareImage, SocBus, SocConfig};
use netlist::{CellConfig, DieConfig, Microprogram, Parameter};
use sim::chip::{RunnerEngine, FLOW_REGISTER_BITS, FLOW_INDEX_BITS};
use sim::lower::{ENTRY_LOGITS, ENTRY_TOKEN};
use sim::image::{load_or_build, verify, LoadReport};
use sim::tokenizer::TokenizerTables;
use sim::{Engine, ExactEngine, Numeric, Runner, Weights};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use tokenizers::Tokenizer;
use tokio::sync::mpsc;

const PORT: u16 = 7878;
const FIRMWARE: &[u8] = include_bytes!("../../../firmware/prebuilt/firmware.elf");
const FLOW_LOGITS_REGISTER: u32 = 1;

struct Loaded {
    name: String,
    engine: Box<dyn Runner>,
    tokenizer: Tokenizer,
    accelerator: Option<String>,
    dir: PathBuf,
    /// The firmware's tokenizer tables, built the first time they are needed.
    firmware_tables: Option<Vec<u32>>,
    load_report: Option<LoadReport>,
    prompt: PromptStyle,
}

#[derive(Clone)]
struct App {
    token: Arc<String>,
    loaded: Arc<Mutex<Option<Loaded>>>,
    dist: Option<Arc<PathBuf>>,
    catalog: Arc<Catalog>,
    hub: Arc<Hub>,
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
    let entry = catalog.find(path).ok_or("el modelo no está en las carpetas conocidas")?;
    if !entry.supported {
        let reason = entry.adaptation.reason.unwrap_or_else(|| format!("la arquitectura «{}» todavía no se puede ejecutar", entry.architecture));
        return Err(format!("{}: {reason}", entry.name));
    }
    let config = std::fs::read_to_string(dir.join("config.json")).map_err(|e| e.to_string())?;
    let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| e.to_string())?;
    let mode = match numeric {
        None => Numeric::Float,
        Some(n) => Numeric::Block { activation_bits: n.activation_bits.clamp(4, 24), state_bits: n.state_bits.clamp(4, 24), table_bits: n.table_bits.map(|b| b.clamp(4, 16)) },
    };
    let weights = Weights::open(dir)?.with_format(format);
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
    let prompt = PromptStyle::read(dir, &program.family, &tokenizer);
    let engine: Box<dyn Runner> = if exact {
        Box::new(ExactEngine::new(program, weights, numeric.map_or(16, |n| n.activation_bits.clamp(8, 24)))?)
    } else {
        Box::new(Engine::new(program, weights, mode)?)
    };
    Ok(Loaded { name: entry.name, engine, tokenizer, accelerator, dir: dir.to_path_buf(), firmware_tables: None, load_report, prompt })
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
    max_tokens: usize,
    temperature: f32,
    top_k: usize,
    #[serde(default)]
    thinking: bool,
    /// Also runs the draft head and reports how often it guessed right.
    #[serde(default)]
    mtp_depth: usize,
    /// How much of the chip around the die to simulate.
    #[serde(default)]
    fidelity: Fidelity,
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
}

async fn generate_socket(ws: WebSocketUpgrade, State(app): State<App>) -> Response {
    ws.on_upgrade(move |socket| handle_socket(socket, app))
}

async fn handle_socket(mut socket: WebSocket, app: App) {
    let Some(Ok(Message::Text(text))) = socket.recv().await else { return };
    let Ok(request) = serde_json::from_str::<GenerateRequest>(&text) else {
        let _ = socket.send(Message::Text(r#"{"type":"error","message":"petición inválida"}"#.into())).await;
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

fn sample(logits: &[f32], temperature: f32, top_k: usize, seed: &mut u64) -> u32 {
    if temperature <= 0.0 {
        return logits.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map_or(0, |(i, _)| i as u32);
    }
    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    let k = top_k.clamp(1, indexed.len());
    indexed.select_nth_unstable_by(k - 1, |a, b| b.1.total_cmp(&a.1));
    indexed.truncate(k);
    let top = indexed.iter().map(|(_, v)| *v).fold(f32::MIN, f32::max);
    let weights: Vec<f64> = indexed.iter().map(|(_, v)| (((v - top) / temperature) as f64).exp()).collect();
    let total: f64 = weights.iter().sum();
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    let mut point = (*seed as f64 / u64::MAX as f64) * total;
    for ((i, _), w) in indexed.iter().zip(&weights) {
        point -= w;
        if point <= 0.0 {
            return *i as u32;
        }
    }
    indexed[0].0 as u32
}

const SAMPLER_ENTRIES: u32 = 64;
const FLOW_BITS: u32 = 16;

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

/// Runs the chip's real firmware at instruction level around the loaded engine.
fn run_firmware(loaded: &mut Loaded, text: &str, add_special: bool, ids: &[u32], request: &GenerateRequest, sender: &mpsc::Sender<serde_json::Value>, cancelled: &std::sync::atomic::AtomicBool) -> Result<(), String> {
    use std::sync::atomic::Ordering::Relaxed;
    if request.mtp_depth > 0 {
        return Err("la cabeza MTP todavía no funciona con la CPU simulada".into());
    }
    let firmware = FirmwareImage::from_elf(FIRMWARE)?;
    let vocabulary = loaded.engine.vocabulary();
    let cell = CellConfig { weight_bits: 8, activation_bits: FLOW_BITS, accumulator_bits: 32, mac_stages: 1, adder_stages: 1, addr_bits: 3, coding: Default::default(), exponent_bits: 0, multiplier: netlist::multiplier::MultiplierKind::Array, fixed_weights: false, ecc: false, skip_zeros: false };
    let die = DieConfig { cells: 1, cell, scale_bits: 16, reg_bits: FLOW_REGISTER_BITS, index_bits: FLOW_INDEX_BITS, microcode_bits: 2, broadcast_stages: 1, function_bits: 12, function_select_bits: 3 };
    let code = Microprogram { words: Vec::new(), entries: vec![ENTRY_TOKEN as u32, ENTRY_LOGITS as u32], parameters: vec![Parameter { register: 1, token: 1, position: 0, constant: 0 }, Parameter { register: 2, token: 0, position: 1, constant: 0 }] };
    let seed = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(1, |d| d.as_nanos() as u32);
    let setup = GenerationSetup {
        max_new: request.max_tokens as u32,
        greedy: request.temperature <= 0.0,
        limit: request.top_k.clamp(1, SAMPLER_ENTRIES as usize) as u32,
        temperature: (request.temperature.max(0.0) * 256.0).round().max(1.0) as u32,
        seed,
        raw_tokens: !request.fidelity.tokenizer,
        stop: loaded.prompt.stop_pair(),
    };
    let info = chip_info(&die, &code, FLOW_LOGITS_REGISTER, vocabulary as u32, &setup);
    let use_tables = request.fidelity.tokenizer;
    if use_tables && loaded.firmware_tables.is_none() {
        let mut tables = TokenizerTables::from_file(&loaded.dir.join("tokenizer.json"))?;
        tables.set_add_special(add_special);
        loaded.firmware_tables = Some(tables.to_words());
    }
    let tables: &[u32] = if use_tables { loaded.firmware_tables.as_deref().unwrap_or_default() } else { &[] };
    let image = build_image(&firmware.program, &firmware.data, &info, tables, &[])?;
    let input: Vec<u8> = if use_tables {
        text.bytes().chain([0]).collect()
    } else {
        ids.iter().flat_map(|t| t.to_le_bytes()).chain(u32::MAX.to_le_bytes()).collect()
    };

    let Loaded { engine, tokenizer, .. } = loaded;
    // The memory the chip needs for this model's tables and its image layout.
    let data_bits = (image.len().max(1 << 13).next_power_of_two().ilog2()).max(13);
    let unit = sampler_model(vocabulary, seed, request.fidelity.sampler_lanes);
    let runner = RunnerEngine::new(engine.as_mut(), FLOW_BITS, request.fidelity.cycles_per_token.max(0.0) as u64);
    let soc = SocConfig { program_bits: 14, data_bits, fifo_bits: 6 };
    let mut bus = SocBus::booted_with(soc, &image, &input, ChipModel::new(runner, code, unit))?;
    let mut cpu = Rv32Iss::default();
    let (mut emitted, mut reported, mut shown) = (0, 0, 0);
    let mut output: Vec<u32> = Vec::new();
    while !cpu.halted {
        if cancelled.load(Relaxed) {
            return Ok(());
        }
        cpu.step(&mut bus);
        let runner = &mut bus.peripheral.engine;
        if let Some(error) = runner.error.take() {
            return Err(error);
        }
        while reported < runner.steps && reported < ids.len() {
            reported += 1;
            let _ = sender.blocking_send(serde_json::json!({ "type": "prefill", "done": reported, "total": ids.len(), "seconds": runner.last_seconds }));
        }
        if use_tables {
            // A token's text is complete once the firmware asks for the next token's logits.
            while emitted + 1 < bus.peripheral.engine.generated {
                emitted += 1;
                let piece = take_text(&bus.tx, &mut shown);
                let _ = sender.blocking_send(serde_json::json!({ "type": "token", "text": piece, "seconds": bus.peripheral.engine.last_seconds, "saturated": 0, "index": emitted, "cycles": cpu.cycle }));
            }
            continue;
        }
        while bus.tx.len() / 4 > emitted {
            let at = emitted * 4;
            output.push(u32::from_le_bytes([bus.tx[at], bus.tx[at + 1], bus.tx[at + 2], bus.tx[at + 3]]));
            emitted += 1;
            let decoded = tokenizer.decode(&output, true).map_err(|e| e.to_string())?;
            let piece = if decoded.ends_with('\u{FFFD}') { String::new() } else { let part = decoded[shown..].to_string(); shown = decoded.len(); part };
            let _ = sender.blocking_send(serde_json::json!({ "type": "token", "id": output[emitted - 1], "text": piece, "seconds": bus.peripheral.engine.last_seconds, "saturated": 0, "index": emitted, "cycles": cpu.cycle }));
        }
    }
    if use_tables {
        let piece = take_text(&bus.tx, &mut shown);
        if !piece.is_empty() {
            emitted += 1;
            let _ = sender.blocking_send(serde_json::json!({ "type": "token", "text": piece, "seconds": bus.peripheral.engine.last_seconds, "saturated": 0, "index": emitted, "cycles": cpu.cycle }));
        }
    } else {
        emitted = output.len();
    }
    let runner = &bus.peripheral.engine;
    if use_tables && runner.tokens.is_empty() && !ids.is_empty() {
        return Err("el texto no cabe en la memoria de trabajo del tokenizador del chip".into());
    }
    let tokenizer_match = use_tables.then(|| runner.tokens.len() >= ids.len() && runner.tokens[..ids.len()] == *ids);
    let _ = sender.blocking_send(serde_json::json!({ "type": "done", "tokens": emitted, "chip": { "cycles": cpu.cycle, "die_cycles": runner.die_cycles, "tokenizer_match": tokenizer_match } }));
    Ok(())
}

fn run_generation(app: &App, request: GenerateRequest, sender: &mpsc::Sender<serde_json::Value>, cancelled: &std::sync::atomic::AtomicBool) -> Result<(), String> {
    use std::sync::atomic::Ordering::Relaxed;
    let mut guard = app.loaded.lock().unwrap();
    let loaded = guard.as_mut().ok_or("carga un modelo primero")?;
    let (text, add_special) = loaded.prompt.render(&request.prompt, request.thinking)?;
    let ids = loaded.tokenizer.encode(text.as_str(), add_special).map_err(|e| e.to_string())?.get_ids().to_vec();
    loaded.engine.reset();
    let _ = sender.blocking_send(serde_json::json!({ "type": "start", "prompt_tokens": ids.len() }));
    if request.fidelity.firmware {
        return run_firmware(loaded, &text, add_special, &ids, &request, sender, cancelled);
    }

    let mut unit = sampler_model(loaded.engine.vocabulary(), 0x1234_5679, request.fidelity.sampler_lanes);
    let use_unit = request.fidelity.sampler_unit;
    let mut seed = 0x9E3779B97F4A7C15u64 ^ std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(1, |d| d.as_nanos() as u64);
    let mut next = 0;
    for (i, id) in ids.iter().enumerate() {
        if cancelled.load(Relaxed) {
            return Ok(());
        }
        let last = i + 1 == ids.len();
        let wants_mtp = request.mtp_depth > 0;
        let (logits, stats) = loaded.engine.step(*id, last || wants_mtp)?;
        if last {
            if let Some(l) = logits {
                next = if use_unit { sample_on_unit(&mut unit, l, request.temperature, request.top_k) } else { sample(l, request.temperature, request.top_k, &mut seed) };
            }
        }
        if wants_mtp && loaded.engine.has_draft_head() {
            let following = if last { next } else { ids[i + 1] };
            loaded.engine.feed_draft_from_main();
            loaded.engine.draft(following, false)?;
        }
        let _ = sender.blocking_send(serde_json::json!({ "type": "prefill", "done": i + 1, "total": ids.len(), "seconds": stats.seconds }));
    }

    if request.mtp_depth > 0 && !loaded.engine.has_draft_head() {
        return Err("este modelo no trae cabeza MTP".into());
    }
    let mut output: Vec<u32> = Vec::new();
    let mut shown = 0;
    let (mut hits, mut tries) = (vec![0u32; request.mtp_depth], vec![0u32; request.mtp_depth]);
    let mut pending: Vec<(usize, Vec<u32>)> = Vec::new();
    for _ in 0..request.max_tokens {
        if cancelled.load(Relaxed) || loaded.prompt.stop.contains(&next) {
            break;
        }
        output.push(next);
        let decoded = loaded.tokenizer.decode(&output, true).map_err(|e| e.to_string())?;
        let piece = if decoded.ends_with('\u{FFFD}') { String::new() } else { let part = decoded[shown..].to_string(); shown = decoded.len(); part };
        let (logits, stats) = loaded.engine.step(next, true)?;
        let saturated = stats.saturated;
        let seconds = stats.seconds;
        let token = next;
        let logits = logits.ok_or("sin logits")?;
        next = if use_unit { sample_on_unit(&mut unit, logits, request.temperature, request.top_k) } else { sample(logits, request.temperature, request.top_k, &mut seed) };

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
                let logits = loaded.engine.draft(guess, true)?.ok_or("sin logits")?;
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
    let _ = sender.blocking_send(serde_json::json!({ "type": "done", "tokens": output.len() }));
    Ok(())
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
        return (StatusCode::NOT_FOUND, "interfaz no compilada").into_response();
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
        eprintln!("No se pudo escribir el token en {}", path.display());
    }
    token
}

#[tokio::main]
async fn main() {
    let app = App { token: Arc::new(session_token()), loaded: Arc::new(Mutex::new(None)), dist: dist_directory().map(Arc::new), catalog: Arc::new(Catalog::open(home())), hub: Hub::new(home()) };
    let routes = Router::new()
        .route("/api/models", get(models))
        .route("/api/status", get(status))
        .route("/api/settings", get(read_settings).put(write_settings))
        .route("/api/hf/search", get(hub_search))
        .route("/api/hf/model", get(hub_details))
        .route("/api/hf/downloads", get(hub_jobs).post(hub_download))
        .route("/api/hf/cancel", post(hub_cancel))
        .route("/api/hf/discard", post(hub_discard))
        .route("/api/hf/token", put(hub_set_token).get(hub_token).delete(hub_clear_token))
        .route("/api/load", post(load))
        .route("/api/generate", get(generate_socket))
        .layer(middleware::from_fn_with_state(app.clone(), guard))
        .fallback(static_file)
        .with_state(app);
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", PORT)).await.expect("el puerto 7878 está ocupado");
    println!("llm2chip en http://127.0.0.1:{PORT}");
    axum::serve(listener, routes).await.expect("el servidor ha fallado");
}
