//! Finds checkpoints on Hugging Face and brings them to this machine.
//!
//! Only safetensors weights and plain metadata are fetched. Pickled weights and a
//! repository's own Python are never downloaded, so nothing from it can execute here.

use crate::catalog::{adapt_config, adapt_family, Adaptation, PARTIAL_SUFFIX};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::sync::Semaphore;

const DEFAULT_ENDPOINT: &str = "https://huggingface.co";
const METADATA: [&str; 8] = ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja", "chat_template.json", "model.safetensors.index.json"];
const ATTEMPTS: u32 = 4;
/// Left free on the disk, so that a download never fills it to the last byte.
const SPARE_BYTES: u64 = 1 << 30;

#[derive(Clone, Debug, Deserialize)]
struct Lfs {
    sha256: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
struct Sibling {
    rfilename: String,
    #[serde(default)]
    size: Option<u64>,
    #[serde(default)]
    lfs: Option<Lfs>,
}

#[derive(Deserialize)]
struct RepoInfo {
    #[serde(default)]
    sha: Option<String>,
    #[serde(default)]
    gated: Value,
    #[serde(default)]
    siblings: Vec<Sibling>,
}

#[derive(Serialize)]
pub struct SearchHit {
    id: String,
    downloads: u64,
    likes: u64,
    architecture: Option<String>,
    adaptation: Adaptation,
}

#[derive(Serialize)]
pub struct FileView {
    name: String,
    size: u64,
}

#[derive(Serialize)]
pub struct Details {
    id: String,
    gated: bool,
    files: Vec<FileView>,
    total_bytes: u64,
    config: Option<String>,
    architecture: Option<String>,
    adaptation: Adaptation,
    target: String,
    present: bool,
    resumable_bytes: u64,
    free_bytes: Option<u64>,
}

#[derive(Clone, Debug, Serialize)]
pub struct JobView {
    id: String,
    state: String,
    total_bytes: u64,
    done_bytes: u64,
    file: String,
    error: Option<String>,
    target: String,
}

struct Job {
    id: String,
    target: PathBuf,
    total: u64,
    done: AtomicU64,
    cancel: AtomicBool,
    file: Mutex<String>,
    state: Mutex<(String, Option<String>)>,
}

impl Job {
    fn view(&self) -> JobView {
        let (state, error) = self.state.lock().unwrap().clone();
        JobView { id: self.id.clone(), state, total_bytes: self.total, done_bytes: self.done.load(Relaxed), file: self.file.lock().unwrap().clone(), error, target: self.target.to_string_lossy().into_owned() }
    }

    fn set_state(&self, state: &str, error: Option<String>) {
        *self.state.lock().unwrap() = (state.to_string(), error);
    }

    fn active(&self) -> bool {
        matches!(self.state.lock().unwrap().0.as_str(), "queued" | "downloading")
    }
}

enum Stop {
    Cancelled,
    /// Trying again cannot help: the bytes are wrong or access is denied.
    Fatal(String),
    Retry(String),
}

pub struct Hub {
    endpoint: String,
    home: PathBuf,
    client: reqwest::Client,
    jobs: Mutex<HashMap<String, Arc<Job>>>,
    slots: Semaphore,
}

/// `owner/name`, as a repository is named; this also keeps it safe as a folder name.
fn valid_repo(id: &str) -> bool {
    let parts: Vec<&str> = id.split('/').collect();
    parts.len() == 2 && parts.iter().all(|p| !p.is_empty() && !p.starts_with('.') && !p.contains("..") && p.len() <= 96 && p.chars().all(|c| c.is_ascii_alphanumeric() || "-_.".contains(c)))
}

fn folder_for(root: &Path, id: &str) -> PathBuf {
    root.join(id.replace('/', "--"))
}

fn partial_of(target: &Path) -> PathBuf {
    let mut name = target.as_os_str().to_owned();
    name.push(PARTIAL_SUFFIX);
    PathBuf::from(name)
}

/// The files that make up a usable checkpoint, without duplicates of the weights.
fn select_files(siblings: &[Sibling], shards: Option<&[String]>) -> Result<Vec<Sibling>, String> {
    let root: Vec<&Sibling> = siblings.iter().filter(|s| !s.rfilename.contains('/')).collect();
    let is_sidecar = |name: &str| name.starts_with("mtp") && name.ends_with(".safetensors");
    let is_weights = |name: &str| name.ends_with(".safetensors");
    let single = root.iter().any(|s| s.rfilename == "model.safetensors");
    let keep = |name: &str| {
        if METADATA.contains(&name) || is_sidecar(name) {
            return true;
        }
        if !is_weights(name) {
            return false;
        }
        match shards {
            Some(list) => list.iter().any(|s| s == name),
            None if single => name == "model.safetensors",
            None => !name.starts_with("consolidated"),
        }
    };
    let chosen: Vec<Sibling> = root.into_iter().filter(|s| keep(&s.rfilename)).cloned().collect();
    if !chosen.iter().any(|s| is_weights(&s.rfilename)) {
        return Err("este repositorio no publica pesos en safetensors".into());
    }
    if !chosen.iter().any(|s| s.rfilename == "config.json") {
        return Err("este repositorio no tiene config.json".into());
    }
    Ok(chosen)
}

fn read_trimmed(path: &Path) -> Option<String> {
    let text = std::fs::read_to_string(path).ok()?;
    let text = text.trim();
    (!text.is_empty()).then(|| text.to_string())
}

fn free_space(path: &Path) -> Option<u64> {
    path.ancestors().find(|p| p.exists()).and_then(|p| fs4::available_space(p).ok())
}

impl Hub {
    pub fn new(home: PathBuf) -> Arc<Self> {
        let endpoint = std::env::var("HF_ENDPOINT").unwrap_or_else(|_| DEFAULT_ENDPOINT.into());
        Self::with_endpoint(home, endpoint)
    }

    fn with_endpoint(home: PathBuf, endpoint: String) -> Arc<Self> {
        let client = reqwest::Client::builder().user_agent(concat!("llm2chip/", env!("CARGO_PKG_VERSION"))).connect_timeout(Duration::from_secs(30)).read_timeout(Duration::from_secs(60)).build().expect("cliente HTTP");
        Arc::new(Self { endpoint: endpoint.trim_end_matches('/').to_string(), home, client, jobs: Mutex::new(HashMap::new()), slots: Semaphore::new(1) })
    }

    fn token_file(&self) -> PathBuf {
        self.home.join(".llm2chip/hf-token")
    }

    /// The token and where it came from. What is set in the app wins.
    fn token(&self) -> Option<(String, &'static str)> {
        if let Some(token) = read_trimmed(&self.token_file()) {
            return Some((token, "app"));
        }
        for name in ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"] {
            if let Some(token) = std::env::var(name).ok().filter(|t| !t.trim().is_empty()) {
                return Some((token.trim().to_string(), "entorno"));
            }
        }
        let cache = std::env::var_os("HF_HOME").map_or_else(|| self.home.join(".cache/huggingface"), PathBuf::from);
        read_trimmed(&cache.join("token")).map(|token| (token, "huggingface-cli"))
    }

    pub fn token_status(&self) -> Value {
        serde_json::json!({ "source": self.token().map(|(_, source)| source) })
    }

    pub fn set_token(&self, token: &str) -> Result<(), String> {
        let token = token.trim();
        if token.is_empty() || !token.chars().all(|c| c.is_ascii_graphic()) {
            return Err("el token no tiene un formato válido".into());
        }
        let path = self.token_file();
        std::fs::create_dir_all(path.parent().unwrap()).map_err(|e| e.to_string())?;
        let mut options = std::fs::OpenOptions::new();
        options.write(true).create(true).truncate(true);
        #[cfg(unix)]
        std::os::unix::fs::OpenOptionsExt::mode(&mut options, 0o600);
        std::io::Write::write_all(&mut options.open(&path).map_err(|e| e.to_string())?, token.as_bytes()).map_err(|e| e.to_string())
    }

    pub fn clear_token(&self) -> Result<(), String> {
        match std::fs::remove_file(self.token_file()) {
            Err(e) if e.kind() != std::io::ErrorKind::NotFound => Err(e.to_string()),
            _ => Ok(()),
        }
    }

    fn request(&self, url: &str) -> reqwest::RequestBuilder {
        let request = self.client.get(url);
        match self.token() {
            // The client drops this header when a file redirects to another host.
            Some((token, _)) => request.bearer_auth(token),
            None => request,
        }
    }

    async fn checked(&self, request: reqwest::RequestBuilder, id: &str) -> Result<reqwest::Response, String> {
        let response = request.send().await.map_err(|e| format!("no se pudo conectar con Hugging Face: {e}"))?;
        match response.status().as_u16() {
            200..=299 => Ok(response),
            401 | 403 => Err(format!("{id} necesita acceso: acepta su licencia en huggingface.co/{id} y añade un token de lectura")),
            404 => Err(format!("{id} no existe o es privado")),
            429 => Err("Hugging Face está limitando las peticiones; prueba dentro de un rato".into()),
            code => Err(format!("Hugging Face respondió con el error {code}")),
        }
    }

    pub async fn search(&self, query: &str) -> Result<Vec<SearchHit>, String> {
        let url = format!("{}/api/models", self.endpoint);
        let request = self.request(&url).query(&[("search", query), ("filter", "safetensors"), ("pipeline_tag", "text-generation"), ("sort", "downloads"), ("limit", "30"), ("config", "true")]);
        let found: Vec<Value> = self.checked(request, query).await?.json().await.map_err(|e| e.to_string())?;
        Ok(found
            .iter()
            .filter_map(|item| {
                let architecture = item["config"]["model_type"].as_str().map(String::from);
                let adaptation = architecture.as_deref().map(adapt_family).unwrap_or_default();
                Some(SearchHit { id: item["id"].as_str()?.to_string(), downloads: item["downloads"].as_u64().unwrap_or(0), likes: item["likes"].as_u64().unwrap_or(0), architecture, adaptation })
            })
            .collect())
    }

    async fn info(&self, id: &str) -> Result<(RepoInfo, String), String> {
        if !valid_repo(id) {
            return Err(format!("«{id}» no es un repositorio válido; se escribe autor/modelo"));
        }
        let url = format!("{}/api/models/{id}", self.endpoint);
        let info: RepoInfo = self.checked(self.request(&url).query(&[("blobs", "true")]), id).await?.json().await.map_err(|e| e.to_string())?;
        let revision = info.sha.clone().ok_or("Hugging Face no indica la revisión del repositorio")?;
        Ok((info, revision))
    }

    fn file_url(&self, id: &str, revision: &str, name: &str) -> String {
        format!("{}/{id}/resolve/{revision}/{name}", self.endpoint)
    }

    async fn plan(&self, id: &str, info: &RepoInfo, revision: &str) -> Result<Vec<Sibling>, String> {
        let index = "model.safetensors.index.json";
        let mut shards = None;
        if info.siblings.iter().any(|s| s.rfilename == index) {
            let text = self.checked(self.request(&self.file_url(id, revision, index)), id).await?.text().await.map_err(|e| e.to_string())?;
            let value: Value = serde_json::from_str(&text).map_err(|e| format!("{index}: {e}"))?;
            let map = value["weight_map"].as_object().ok_or_else(|| format!("{index} no tiene weight_map"))?;
            let mut list: Vec<String> = map.values().filter_map(|v| v.as_str().map(String::from)).collect();
            list.sort();
            list.dedup();
            shards = Some(list);
        }
        select_files(&info.siblings, shards.as_deref())
    }

    /// Everything known about a repository without fetching its weights.
    pub async fn details(&self, id: &str, root: &Path) -> Result<Details, String> {
        let (info, revision) = self.info(id).await?;
        let files = self.plan(id, &info, &revision).await;
        let has_config = info.siblings.iter().any(|s| s.rfilename == "config.json");
        let config = if has_config { Some(self.checked(self.request(&self.file_url(id, &revision, "config.json")), id).await?.text().await.map_err(|e| e.to_string())?) } else { None };
        let architecture = config.as_deref().and_then(|c| serde_json::from_str::<Value>(c).ok()).and_then(|v| v.get("model_type")?.as_str().map(String::from));
        let mut adaptation = config.as_deref().map(adapt_config).unwrap_or_default();
        let (views, total) = match &files {
            Ok(list) => (list.iter().map(|s| FileView { name: s.rfilename.clone(), size: s.size.unwrap_or(0) }).collect(), list.iter().map(|s| s.size.unwrap_or(0)).sum()),
            Err(reason) => {
                adaptation.simulate = false;
                adaptation.reason = Some(reason.clone());
                (Vec::new(), 0)
            }
        };
        if files.as_ref().is_ok_and(|list| !list.iter().any(|s| s.rfilename == "tokenizer.json")) && adaptation.simulate {
            adaptation.simulate = false;
            adaptation.reason = Some("no publica tokenizer.json, que es lo que convierte el texto en tokens".into());
        }
        let target = folder_for(root, id);
        Ok(Details {
            id: id.to_string(),
            gated: !matches!(info.gated, Value::Bool(false) | Value::Null),
            files: views,
            total_bytes: total,
            config,
            architecture,
            adaptation,
            present: target.join("config.json").exists(),
            resumable_bytes: folder_size(&partial_of(&target)),
            free_bytes: free_space(root),
            target: target.to_string_lossy().into_owned(),
        })
    }

    pub fn jobs(&self) -> Vec<JobView> {
        let mut list: Vec<JobView> = self.jobs.lock().unwrap().values().map(|j| j.view()).collect();
        list.sort_by(|a, b| a.id.cmp(&b.id));
        list
    }

    /// Starts in the background and returns at once. `finished` runs after a success.
    pub async fn start(self: &Arc<Self>, id: String, root: PathBuf, finished: impl FnOnce() + Send + 'static) -> Result<JobView, String> {
        if self.jobs.lock().unwrap().get(&id).is_some_and(|j| j.active()) {
            return Err("ese modelo ya se está descargando".into());
        }
        let (info, revision) = self.info(&id).await?;
        let files = self.plan(&id, &info, &revision).await?;
        let target = folder_for(&root, &id);
        if target.join("config.json").exists() {
            return Err(format!("ya está en {}", target.display()));
        }
        let job = Arc::new(Job {
            id: id.clone(),
            target,
            total: files.iter().map(|f| f.size.unwrap_or(0)).sum(),
            done: AtomicU64::new(0),
            cancel: AtomicBool::new(false),
            file: Mutex::new(String::new()),
            state: Mutex::new(("queued".into(), None)),
        });
        self.jobs.lock().unwrap().insert(id.clone(), job.clone());
        let view = job.view();
        let hub = self.clone();
        tokio::spawn(async move {
            let Ok(_turn) = hub.slots.acquire().await else { return };
            if job.cancel.load(Relaxed) {
                job.set_state("cancelled", None);
                return;
            }
            job.set_state("downloading", None);
            match hub.transfer(&job, &id, &revision, &files).await {
                Ok(()) => {
                    job.set_state("done", None);
                    finished();
                }
                Err(Stop::Cancelled) => job.set_state("cancelled", None),
                Err(Stop::Fatal(message) | Stop::Retry(message)) => job.set_state("error", Some(message)),
            }
        });
        Ok(view)
    }

    pub fn cancel(&self, id: &str) -> bool {
        match self.jobs.lock().unwrap().get(id) {
            Some(job) if job.active() => {
                job.cancel.store(true, Relaxed);
                true
            }
            _ => false,
        }
    }

    /// Drops what an interrupted download left behind.
    pub async fn discard(&self, id: &str, root: &Path) -> Result<(), String> {
        if !valid_repo(id) {
            return Err("repositorio no válido".into());
        }
        if self.jobs.lock().unwrap().get(id).is_some_and(|j| j.active()) {
            return Err("detén antes la descarga".into());
        }
        let partial = partial_of(&folder_for(root, id));
        if partial.exists() {
            tokio::fs::remove_dir_all(&partial).await.map_err(|e| e.to_string())?;
        }
        self.jobs.lock().unwrap().remove(id);
        Ok(())
    }

    async fn transfer(&self, job: &Job, id: &str, revision: &str, files: &[Sibling]) -> Result<(), Stop> {
        let fatal = |e: std::io::Error| Stop::Fatal(e.to_string());
        let partial = partial_of(&job.target);
        let stamp = partial.join(".revision");
        // Pieces of another revision must not be joined with this one's.
        if tokio::fs::read_to_string(&stamp).await.ok().as_deref() != Some(revision) && partial.exists() {
            tokio::fs::remove_dir_all(&partial).await.map_err(fatal)?;
        }
        tokio::fs::create_dir_all(&partial).await.map_err(fatal)?;
        tokio::fs::write(&stamp, revision).await.map_err(fatal)?;

        let held = folder_size(&partial);
        if let Some(free) = free_space(&partial) {
            let needed = job.total.saturating_sub(held) + SPARE_BYTES;
            if free < needed {
                return Err(Stop::Fatal(format!("faltan {:.1} GB libres en el disco para descargarlo", (needed - free) as f64 / 1e9)));
            }
        }

        let mut finished = 0u64;
        for file in files {
            *job.file.lock().unwrap() = file.rfilename.clone();
            let url = self.file_url(id, revision, &file.rfilename);
            let dest = partial.join(&file.rfilename);
            let mut attempt = 0;
            loop {
                attempt += 1;
                match self.fetch(job, &url, &dest, file, finished).await {
                    Ok(()) => break,
                    Err(Stop::Retry(message)) if attempt >= ATTEMPTS => return Err(Stop::Fatal(message)),
                    Err(Stop::Retry(_)) => {
                        for _ in 0..attempt * 10 {
                            if job.cancel.load(Relaxed) {
                                return Err(Stop::Cancelled);
                            }
                            tokio::time::sleep(Duration::from_millis(100)).await;
                        }
                    }
                    Err(other) => return Err(other),
                }
            }
            finished += file.size.unwrap_or(0);
            job.done.store(finished, Relaxed);
        }
        tokio::fs::remove_file(&stamp).await.map_err(fatal)?;
        if let Some(parent) = job.target.parent() {
            tokio::fs::create_dir_all(parent).await.map_err(fatal)?;
        }
        tokio::fs::rename(&partial, &job.target).await.map_err(fatal)
    }

    /// Brings one file, carrying on from what is already there and checking the result.
    async fn fetch(&self, job: &Job, url: &str, dest: &Path, file: &Sibling, before: u64) -> Result<(), Stop> {
        let io = |e: std::io::Error| Stop::Fatal(format!("{}: {e}", file.rfilename));
        let name = &file.rfilename;
        let expected = file.size;
        let digest = file.lfs.as_ref().and_then(|l| l.sha256.clone());

        // Only a file with a published digest can be trusted when joined from pieces.
        let mut hasher = Sha256::new();
        let mut held = 0u64;
        if let (Some(size), true) = (expected, digest.is_some()) {
            if let Ok(mut existing) = tokio::fs::File::open(dest).await {
                let mut chunk = vec![0u8; 1 << 20];
                loop {
                    if job.cancel.load(Relaxed) {
                        return Err(Stop::Cancelled);
                    }
                    let read = existing.read(&mut chunk).await.map_err(io)?;
                    if read == 0 {
                        break;
                    }
                    hasher.update(&chunk[..read]);
                    held += read as u64;
                    job.done.store(before + held, Relaxed);
                }
            }
            if held > size {
                held = 0;
                hasher = Sha256::new();
                tokio::fs::remove_file(dest).await.map_err(io)?;
            }
        }

        if digest.is_none() || expected != Some(held) {
            let mut request = self.request(url);
            if held > 0 {
                request = request.header(reqwest::header::RANGE, format!("bytes={held}-"));
            }
            let response = request.send().await.map_err(|e| Stop::Retry(format!("{name}: {e}")))?;
            let status = response.status().as_u16();
            match status {
                200 => {
                    held = 0;
                    hasher = Sha256::new();
                }
                206 => {}
                401 | 403 => return Err(Stop::Fatal(format!("sin acceso a {name}"))),
                416 => {
                    tokio::fs::remove_file(dest).await.map_err(io)?;
                    return Err(Stop::Retry(format!("{name}: rango no válido")));
                }
                408 | 429 | 500..=599 => return Err(Stop::Retry(format!("{name}: error {status}"))),
                _ => return Err(Stop::Fatal(format!("{name}: error {status}"))),
            }
            let mut output = tokio::fs::OpenOptions::new().create(true).write(true).append(status == 206).truncate(status == 200).open(dest).await.map_err(io)?;
            let mut stream = response.bytes_stream();
            while let Some(piece) = stream.next().await {
                if job.cancel.load(Relaxed) {
                    output.flush().await.map_err(io)?;
                    return Err(Stop::Cancelled);
                }
                let piece = match piece {
                    Ok(piece) => piece,
                    Err(e) => {
                        output.flush().await.map_err(io)?;
                        return Err(Stop::Retry(format!("{name}: {e}")));
                    }
                };
                output.write_all(&piece).await.map_err(io)?;
                hasher.update(&piece);
                held += piece.len() as u64;
                job.done.store(before + held, Relaxed);
            }
            output.flush().await.map_err(io)?;
        }

        if expected.is_some_and(|size| held < size) {
            return Err(Stop::Retry(format!("{name}: la conexión se cortó")));
        }
        let wrong_digest = digest.is_some_and(|wanted| !wanted.eq_ignore_ascii_case(&hex(&hasher.finalize())));
        if wrong_digest || expected.is_some_and(|size| held != size) {
            tokio::fs::remove_file(dest).await.map_err(io)?;
            return Err(Stop::Retry(format!("{name} no coincide con lo publicado")));
        }
        Ok(())
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn folder_size(dir: &Path) -> u64 {
    std::fs::read_dir(dir).map_or(0, |entries| entries.flatten().filter_map(|e| e.metadata().ok()).filter(|m| m.is_file()).map(|m| m.len()).sum())
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        body::Body,
        extract::{Path as Route, State},
        http::{header, HeaderMap, StatusCode},
        response::{IntoResponse, Response},
        routing::get,
        Json, Router,
    };
    use std::sync::atomic::AtomicUsize;

    const LLAMA: &str = r#"{"model_type":"llama","hidden_size":16,"num_attention_heads":2,"intermediate_size":32,"vocab_size":8,"num_hidden_layers":1}"#;
    const REPO: &str = "autor/modelo";

    #[derive(Default)]
    struct Fake {
        /// The next downloads of the weights end after this many bytes.
        cut_after: Mutex<Vec<usize>>,
        corrupt: AtomicBool,
        ranges: Mutex<Vec<String>>,
        fetched: Mutex<Vec<String>>,
        authorized: AtomicUsize,
    }

    fn weights() -> Vec<u8> {
        (0..200_000u32).map(|i| (i % 251) as u8).collect()
    }

    fn sha(bytes: &[u8]) -> String {
        hex(&Sha256::digest(bytes))
    }

    async fn repo(headers: HeaderMap, Route((owner, name)): Route<(String, String)>, State(fake): State<Arc<Fake>>) -> Response {
        if headers.contains_key(header::AUTHORIZATION) {
            fake.authorized.fetch_add(1, Relaxed);
        }
        if owner == "cerrado" {
            return StatusCode::UNAUTHORIZED.into_response();
        }
        if name != "modelo" {
            return StatusCode::NOT_FOUND.into_response();
        }
        let weights = weights();
        Json(serde_json::json!({
            "sha": "abc123",
            "gated": false,
            "siblings": [
                { "rfilename": "config.json", "size": LLAMA.len() },
                { "rfilename": "tokenizer.json", "size": 2 },
                { "rfilename": "modeling_custom.py", "size": 10 },
                { "rfilename": "pytorch_model.bin", "size": 10, "lfs": { "sha256": "00" } },
                { "rfilename": "onnx/model.safetensors", "size": 10 },
                { "rfilename": "model.safetensors", "size": weights.len(), "lfs": { "sha256": sha(&weights) } },
            ],
        }))
        .into_response()
    }

    async fn resolve(Route((_, _, revision, file)): Route<(String, String, String, String)>, State(fake): State<Arc<Fake>>) -> Response {
        assert_eq!(revision, "abc123", "todo se pide a la revisión fijada");
        fake.fetched.lock().unwrap().push(file.clone());
        match file.as_str() {
            "config.json" => LLAMA.into_response(),
            "tokenizer.json" => "{}".into_response(),
            "model.safetensors" => (StatusCode::FOUND, [(header::LOCATION, "/cdn/peso")]).into_response(),
            _ => StatusCode::NOT_FOUND.into_response(),
        }
    }

    async fn cdn(headers: HeaderMap, State(fake): State<Arc<Fake>>) -> Response {
        let mut data = weights();
        if fake.corrupt.swap(false, Relaxed) {
            data[10] ^= 1;
        }
        let start = headers.get(header::RANGE).and_then(|r| r.to_str().ok()).map(|r| {
            fake.ranges.lock().unwrap().push(r.to_string());
            r.trim_start_matches("bytes=").trim_end_matches('-').parse::<usize>().unwrap()
        });
        let body = data[start.unwrap_or(0)..].to_vec();
        let limit = { let mut cuts = fake.cut_after.lock().unwrap(); if cuts.is_empty() { None } else { Some(cuts.remove(0)) } };
        let status = if start.is_some() { StatusCode::PARTIAL_CONTENT } else { StatusCode::OK };
        let Some(limit) = limit else {
            return (status, body).into_response();
        };
        let head = body[..limit].to_vec();
        // The pause lets the first part reach the client before the connection breaks.
        let stream = futures_util::stream::iter([Some(head), None]).then(|piece| async move {
            if piece.is_none() {
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
            piece.ok_or_else(|| std::io::Error::other("corte"))
        });
        (status, [(header::CONTENT_LENGTH, body.len().to_string())], Body::from_stream(stream)).into_response()
    }

    async fn serve() -> (String, Arc<Fake>) {
        let fake = Arc::new(Fake::default());
        let routes = Router::new().route("/api/models/{owner}/{name}", get(repo)).route("/{owner}/{name}/resolve/{revision}/{file}", get(resolve)).route("/cdn/peso", get(cdn)).with_state(fake.clone());
        let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
        let address = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, routes).await.unwrap() });
        (format!("http://{address}"), fake)
    }

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("llm2chip-hub-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    async fn until_over(hub: &Hub) -> JobView {
        for _ in 0..1000 {
            let view = hub.jobs().remove(0);
            if !matches!(view.state.as_str(), "queued" | "downloading") {
                return view;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        panic!("la descarga no termina");
    }

    fn sibling(name: &str) -> Sibling {
        Sibling { rfilename: name.into(), size: Some(1), lfs: None }
    }

    #[test]
    fn weights_are_taken_once_and_nothing_executable_is() {
        let names = |list: Vec<Sibling>| list.into_iter().map(|s| s.rfilename).collect::<Vec<_>>();
        let mistral = ["config.json", "consolidated.safetensors", "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors", "model.safetensors.index.json", "modeling.py", "pytorch_model.bin"].map(sibling);
        let shards = ["model-00001-of-00002.safetensors".to_string(), "model-00002-of-00002.safetensors".to_string()];
        assert_eq!(names(select_files(&mistral, Some(&shards)).unwrap()), ["config.json", "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors", "model.safetensors.index.json"]);

        let sidecar = ["config.json", "model.safetensors", "model.fp16.safetensors", "mtp.safetensors"].map(sibling);
        assert_eq!(names(select_files(&sidecar, None).unwrap()), ["config.json", "model.safetensors", "mtp.safetensors"]);

        assert!(select_files(&["config.json", "pytorch_model.bin"].map(sibling), None).unwrap_err().contains("safetensors"));
    }

    #[test]
    fn repository_names_cannot_leave_the_folder() {
        for bad in ["../secreto", "a/../../b", "a/b/c", "/a", "a/.b", "a", "a/b c", ""] {
            assert!(!valid_repo(bad), "{bad}");
        }
        assert!(valid_repo("HuggingFaceTB/SmolLM2-135M") && valid_repo("a_b/c.d-e"));
    }

    #[tokio::test]
    async fn a_repository_is_described_without_fetching_its_weights() {
        let (endpoint, fake) = serve().await;
        let home = scratch("details");
        let hub = Hub::with_endpoint(home.clone(), endpoint);
        let details = hub.details(REPO, &home).await.unwrap();
        assert_eq!(details.total_bytes as usize, 200_000 + LLAMA.len() + 2);
        assert!(details.adaptation.design && details.adaptation.simulate);
        assert_eq!(*fake.fetched.lock().unwrap(), ["config.json"]);
        assert!(hub.details("cerrado/modelo", &home).await.err().unwrap().contains("token"));
        assert!(hub.details("autor/otro", &home).await.err().unwrap().contains("no existe"));
        std::fs::remove_dir_all(home).unwrap();
    }

    #[tokio::test]
    async fn a_cut_connection_resumes_and_only_a_checked_model_appears() {
        let (endpoint, fake) = serve().await;
        fake.cut_after.lock().unwrap().push(70_000);
        let home = scratch("resume");
        let hub = Hub::with_endpoint(home.clone(), endpoint);
        hub.start(REPO.into(), home.clone(), || {}).await.unwrap();
        assert!(hub.start(REPO.into(), home.clone(), || {}).await.is_err() || hub.jobs()[0].state == "done");
        let view = until_over(&hub).await;
        assert_eq!(view.state, "done", "{:?}", view.error);
        assert_eq!(*fake.ranges.lock().unwrap(), ["bytes=70000-"]);
        let target = home.join("autor--modelo");
        assert_eq!(std::fs::read(target.join("model.safetensors")).unwrap(), weights());
        assert!(!target.join(".revision").exists() && !target.join("pytorch_model.bin").exists() && !target.join("modeling_custom.py").exists());
        assert!(!partial_of(&target).exists());
        assert!(hub.start(REPO.into(), home.clone(), || {}).await.unwrap_err().contains("ya está"));
        std::fs::remove_dir_all(home).unwrap();
    }

    #[tokio::test]
    async fn corrupted_bytes_are_thrown_away_and_fetched_again() {
        let (endpoint, fake) = serve().await;
        fake.corrupt.store(true, Relaxed);
        let home = scratch("corrupt");
        let hub = Hub::with_endpoint(home.clone(), endpoint);
        hub.start(REPO.into(), home.clone(), || {}).await.unwrap();
        let view = until_over(&hub).await;
        assert_eq!(view.state, "done", "{:?}", view.error);
        assert_eq!(std::fs::read(home.join("autor--modelo/model.safetensors")).unwrap(), weights());
        std::fs::remove_dir_all(home).unwrap();
    }

    #[tokio::test]
    async fn a_download_can_be_stopped_resumed_or_discarded() {
        let (endpoint, fake) = serve().await;
        // Enough cuts to outlast the retries: the job ends in error with its bytes kept.
        fake.cut_after.lock().unwrap().extend([50_000, 10_000, 10_000, 10_000]);
        let home = scratch("stop");
        let hub = Hub::with_endpoint(home.clone(), endpoint);
        hub.start(REPO.into(), home.clone(), || {}).await.unwrap();
        tokio::time::sleep(Duration::from_millis(200)).await;
        assert!(hub.cancel(REPO));
        let view = until_over(&hub).await;
        assert_eq!(view.state, "cancelled");
        let details = hub.details(REPO, &home).await.unwrap();
        assert!(details.resumable_bytes >= 50_000 && !details.present);

        hub.discard(REPO, &home).await.unwrap();
        assert!(!partial_of(&home.join("autor--modelo")).exists() && hub.jobs().is_empty());
        std::fs::remove_dir_all(home).unwrap();
    }

    #[tokio::test]
    async fn the_token_is_sent_kept_private_and_can_be_removed() {
        let (endpoint, fake) = serve().await;
        let home = scratch("token");
        let hub = Hub::with_endpoint(home.clone(), endpoint);
        hub.set_token(" hf_prueba\n").unwrap();
        assert_eq!(hub.token_status()["source"], "app");
        assert_eq!(std::fs::read_to_string(hub.token_file()).unwrap(), "hf_prueba");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(std::fs::metadata(hub.token_file()).unwrap().permissions().mode() & 0o777, 0o600);
        }
        hub.details(REPO, &home).await.unwrap();
        assert!(fake.authorized.load(Relaxed) > 0);
        hub.clear_token().unwrap();
        assert!(!hub.token_file().exists());
        assert!(hub.set_token("con espacio").is_err());
        std::fs::remove_dir_all(home).unwrap();
    }
}
