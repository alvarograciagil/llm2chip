//! Local engine for the app. Listens on 127.0.0.1 and requires the session token.

use axum::{
    extract::{ws::{Message, WebSocket}, Request, State, WebSocketUpgrade},
    http::{header, StatusCode},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use compiler::compile_qwen3_5;
use serde::{Deserialize, Serialize};
use netlist::format::WeightFormat;
use sim::{Engine, ExactEngine, Numeric, Runner, Weights};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use tokenizers::Tokenizer;
use tokio::sync::mpsc;

const PORT: u16 = 7878;
const STOP_TOKENS: [u32; 2] = [248046, 248044];

struct Loaded {
    name: String,
    engine: Box<dyn Runner>,
    tokenizer: Tokenizer,
    accelerator: Option<String>,
}

#[derive(Clone)]
struct App {
    token: Arc<String>,
    loaded: Arc<Mutex<Option<Loaded>>>,
    dist: Option<Arc<PathBuf>>,
}

#[derive(Serialize)]
struct ModelEntry {
    name: String,
    path: String,
    architecture: String,
    supported: bool,
    size_gb: f64,
}

fn home() -> PathBuf {
    PathBuf::from(std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE")).unwrap_or_default())
}

fn describe(dir: &Path) -> Option<ModelEntry> {
    let config: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).ok()?).ok()?;
    let mut size = 0u64;
    let mut has_weights = false;
    for entry in std::fs::read_dir(dir).ok()?.flatten() {
        if entry.path().extension().is_some_and(|e| e == "safetensors") {
            has_weights = true;
            size += entry.metadata().map_or(0, |m| m.len());
        }
    }
    if !has_weights || !dir.join("tokenizer.json").exists() {
        return None;
    }
    let architecture = config.get("model_type").and_then(|v| v.as_str()).unwrap_or("?").to_string();
    let name = dir.file_name()?.to_string_lossy().replace("Youssofal--", "");
    Some(ModelEntry { name, path: dir.to_string_lossy().into(), supported: matches!(architecture.as_str(), "qwen3_5" | "qwen3_5_moe"), architecture, size_gb: size as f64 / 1e9 })
}

fn scan_models() -> Vec<ModelEntry> {
    let mut found = Vec::new();
    let mtplx = home().join(".mtplx/models");
    if let Ok(entries) = std::fs::read_dir(mtplx) {
        found.extend(entries.flatten().filter_map(|e| describe(&e.path())));
    }
    if let Ok(entries) = std::fs::read_dir(home().join(".cache/huggingface/hub")) {
        for model in entries.flatten() {
            if let Ok(snapshots) = std::fs::read_dir(model.path().join("snapshots")) {
                found.extend(snapshots.flatten().filter_map(|s| describe(&s.path())));
            }
        }
    }
    found.sort_by(|a, b| b.supported.cmp(&a.supported).then(a.name.cmp(&b.name)));
    found
}

async fn models() -> Json<Vec<ModelEntry>> {
    Json(tokio::task::spawn_blocking(scan_models).await.unwrap_or_default())
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
}

#[derive(Clone, Copy, Deserialize)]
struct NumericRequest {
    activation_bits: u32,
    state_bits: u32,
    table_bits: Option<u32>,
}

fn load_model(request: LoadRequest) -> Result<Loaded, String> {
    let LoadRequest { path, numeric, exact, format, gpu } = request;
    let path = path.as_str();
    let dir = Path::new(path);
    let entry = scan_models().into_iter().find(|m| m.path == path).ok_or("el modelo no está en las carpetas conocidas")?;
    if !entry.supported {
        return Err(format!("la arquitectura «{}» todavía no se puede ejecutar", entry.architecture));
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
    let engine: Box<dyn Runner> = if exact {
        Box::new(ExactEngine::new(compile_qwen3_5(&config)?, weights, numeric.map_or(16, |n| n.activation_bits.clamp(8, 24)))?)
    } else {
        Box::new(Engine::new(compile_qwen3_5(&config)?, weights, mode)?)
    };
    Ok(Loaded { name: entry.name, engine, tokenizer, accelerator })
}

async fn load(State(app): State<App>, Json(request): Json<LoadRequest>) -> Response {
    let result = tokio::task::spawn_blocking(move || {
        let loaded = load_model(request)?;
        let reply = (loaded.name.clone(), loaded.accelerator.clone());
        *app.loaded.lock().unwrap() = Some(loaded);
        Ok::<_, String>(reply)
    })
    .await;
    match result {
        Ok(Ok((name, accelerator))) => Json(serde_json::json!({ "loaded": name, "accelerator": accelerator })).into_response(),
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

fn run_generation(app: &App, request: GenerateRequest, sender: &mpsc::Sender<serde_json::Value>, cancelled: &std::sync::atomic::AtomicBool) -> Result<(), String> {
    use std::sync::atomic::Ordering::Relaxed;
    let mut guard = app.loaded.lock().unwrap();
    let loaded = guard.as_mut().ok_or("carga un modelo primero")?;
    let think = if request.thinking { "<think>\n" } else { "<think>\n\n</think>\n\n" };
    let text = format!("<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n{think}", request.prompt);
    let ids = loaded.tokenizer.encode(text, false).map_err(|e| e.to_string())?.get_ids().to_vec();
    loaded.engine.reset();
    let _ = sender.blocking_send(serde_json::json!({ "type": "start", "prompt_tokens": ids.len() }));

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
                next = sample(l, request.temperature, request.top_k, &mut seed);
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
        if cancelled.load(Relaxed) || STOP_TOKENS.contains(&next) {
            break;
        }
        output.push(next);
        let decoded = loaded.tokenizer.decode(&output, true).map_err(|e| e.to_string())?;
        let piece = if decoded.ends_with('\u{FFFD}') { String::new() } else { let part = decoded[shown..].to_string(); shown = decoded.len(); part };
        let (logits, stats) = loaded.engine.step(next, true)?;
        let saturated = stats.saturated;
        let seconds = stats.seconds;
        let token = next;
        next = sample(logits.ok_or("sin logits")?, request.temperature, request.top_k, &mut seed);

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
    let app = App { token: Arc::new(session_token()), loaded: Arc::new(Mutex::new(None)), dist: dist_directory().map(Arc::new) };
    let routes = Router::new()
        .route("/api/models", get(models))
        .route("/api/status", get(status))
        .route("/api/load", post(load))
        .route("/api/generate", get(generate_socket))
        .layer(middleware::from_fn_with_state(app.clone(), guard))
        .fallback(static_file)
        .with_state(app);
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", PORT)).await.expect("el puerto 7878 está ocupado");
    println!("llm2chip en http://127.0.0.1:{PORT}");
    axum::serve(listener, routes).await.expect("el servidor ha fallado");
}
