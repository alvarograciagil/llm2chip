//! Where models live on this machine, and what each one can be turned into.

use compiler::{compile_model, SIMULATED_FAMILIES};
use model_ir::{approximate_family, designable_family, ModelGraph};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// A user's folder may hold models a few levels down, grouped by author or by project.
const SEARCH_DEPTH: usize = 3;
/// Stops a folder as wide as a whole disk from stalling every listing.
const MAX_FOLDERS: usize = 4000;
const KEEP_LISTING: Duration = Duration::from_secs(30);
pub const PARTIAL_SUFFIX: &str = ".partial";

#[derive(Clone, Default, Serialize)]
pub struct Adaptation {
    /// A circuit can be estimated from its `config.json`.
    pub design: bool,
    /// The weights can run on the simulated chip.
    pub simulate: bool,
    /// The estimate takes the layer shape only; the family differs in other ways.
    pub approximate: bool,
    /// Why it cannot go further, in the user's language.
    pub reason: Option<String>,
    pub parameters: Option<u64>,
    pub layers: Option<u32>,
}

/// Decides from a full `config.json`.
pub fn adapt_config(config: &str) -> Adaptation {
    let family = serde_json::from_str::<Value>(config).ok().and_then(|v| v.get("model_type")?.as_str().map(String::from)).unwrap_or_default();
    let graph = ModelGraph::from_config_json("", config);
    let compiled = compile_model(config);
    let mut result = Adaptation { design: graph.is_ok(), simulate: compiled.is_ok(), approximate: approximate_family(&family), ..Default::default() };
    if let Ok(graph) = &graph {
        result.parameters = Some(graph.total_params());
        result.layers = Some(graph.num_layers);
    }
    result.reason = match (&graph, &compiled) {
        (Err(e), _) => Some(e.0.clone()),
        (Ok(_), Err(e)) => Some(e.clone()),
        _ => None,
    };
    result
}

/// Decides from the architecture name alone, as listings give nothing else.
pub fn adapt_family(model_type: &str) -> Adaptation {
    Adaptation { design: designable_family(model_type), simulate: SIMULATED_FAMILIES.contains(&model_type), approximate: approximate_family(model_type), ..Default::default() }
}

#[derive(Clone, Serialize)]
pub struct ModelEntry {
    pub name: String,
    pub path: String,
    pub architecture: String,
    /// It can be loaded and run now.
    pub supported: bool,
    pub size_gb: f64,
    pub config: String,
    pub adaptation: Adaptation,
}

#[derive(Clone, Default, Serialize, Deserialize)]
pub struct Settings {
    #[serde(default)]
    pub model_dirs: Vec<String>,
    #[serde(default)]
    pub download_dir: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct SettingsView {
    model_dirs: Vec<String>,
    download_dir: String,
    download_is_default: bool,
    /// Searched whether or not the user adds anything.
    default_dirs: Vec<String>,
}

pub struct Catalog {
    home: PathBuf,
    settings: Mutex<Settings>,
    listing: Mutex<Option<(Instant, Vec<ModelEntry>)>>,
}

fn describe(dir: &Path, name: String) -> Option<ModelEntry> {
    let config = std::fs::read_to_string(dir.join("config.json")).ok()?;
    let value: Value = serde_json::from_str(&config).ok()?;
    let mut size = 0u64;
    let mut has_weights = false;
    for entry in std::fs::read_dir(dir).ok()?.flatten() {
        if entry.path().extension().is_some_and(|e| e == "safetensors") {
            has_weights = true;
            // The Hugging Face cache links its files, and the size is the target's.
            size += std::fs::metadata(entry.path()).map_or(0, |m| m.len());
        }
    }
    if !has_weights {
        return None;
    }
    let architecture = value.get("model_type").and_then(Value::as_str).unwrap_or("?").to_string();
    let mut adaptation = adapt_config(&config);
    if adaptation.simulate && !dir.join("tokenizer.json").exists() {
        adaptation.simulate = false;
        adaptation.reason = Some("falta tokenizer.json, que es lo que convierte el texto en tokens".into());
    }
    Some(ModelEntry { name, path: dir.to_string_lossy().into(), supported: adaptation.simulate, architecture, size_gb: size as f64 / 1e9, config, adaptation })
}

fn folder_name(dir: &Path) -> String {
    let name = dir.file_name().map_or_else(String::new, |n| n.to_string_lossy().into_owned());
    name.trim_start_matches("models--").replace("Youssofal--", "").replacen("--", "/", 1)
}

fn visit(dir: &Path, depth: usize, budget: &mut usize, found: &mut Vec<ModelEntry>) {
    if *budget == 0 {
        return;
    }
    *budget -= 1;
    if let Some(model) = describe(dir, folder_name(dir)) {
        found.push(model);
        return;
    }
    if let Ok(snapshots) = std::fs::read_dir(dir.join("snapshots")) {
        let repository = folder_name(dir);
        found.extend(snapshots.flatten().filter_map(|s| describe(&s.path(), repository.clone())));
        return;
    }
    if depth == 0 {
        return;
    }
    let Ok(children) = std::fs::read_dir(dir) else { return };
    for child in children.flatten() {
        let name = child.file_name();
        let name = name.to_string_lossy();
        if name.starts_with('.') || name.ends_with(PARTIAL_SUFFIX) || name == "node_modules" || name == "blobs" {
            continue;
        }
        if child.path().is_dir() {
            visit(&child.path(), depth - 1, budget, found);
        }
    }
}

/// Every model under the folders, once each, those that can run first.
pub fn scan(roots: &[PathBuf]) -> Vec<ModelEntry> {
    let mut found = Vec::new();
    let mut budget = MAX_FOLDERS;
    for root in roots {
        visit(root, SEARCH_DEPTH, &mut budget, &mut found);
    }
    let mut seen = HashSet::new();
    found.retain(|m| seen.insert(std::fs::canonicalize(&m.path).unwrap_or_else(|_| PathBuf::from(&m.path))));
    found.sort_by(|a, b| b.supported.cmp(&a.supported).then(b.adaptation.design.cmp(&a.adaptation.design)).then(a.name.cmp(&b.name)));
    found
}

impl Catalog {
    pub fn open(home: PathBuf) -> Self {
        let saved = std::fs::read_to_string(home.join(".llm2chip/settings.json")).ok().and_then(|text| serde_json::from_str(&text).ok()).unwrap_or_default();
        Self { home, settings: Mutex::new(saved), listing: Mutex::new(None) }
    }

    fn expand(&self, raw: &str) -> PathBuf {
        match raw.trim().strip_prefix('~') {
            Some(rest) if rest.is_empty() || rest.starts_with(['/', '\\']) => self.home.join(rest.trim_start_matches(['/', '\\'])),
            _ => PathBuf::from(raw.trim()),
        }
    }

    fn default_roots(&self) -> Vec<PathBuf> {
        let hub = std::env::var_os("HF_HUB_CACHE")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HF_HOME").map(|home| PathBuf::from(home).join("hub")))
            .unwrap_or_else(|| self.home.join(".cache/huggingface/hub"));
        vec![self.default_download(), self.home.join(".mtplx/models"), hub]
    }

    fn default_download(&self) -> PathBuf {
        self.home.join(".llm2chip/models")
    }

    pub fn download_dir(&self) -> PathBuf {
        let saved = self.settings.lock().unwrap().download_dir.clone();
        saved.map_or_else(|| self.default_download(), |dir| self.expand(&dir))
    }

    pub fn view(&self) -> SettingsView {
        let settings = self.settings.lock().unwrap().clone();
        SettingsView {
            model_dirs: settings.model_dirs,
            download_dir: self.download_dir().to_string_lossy().into_owned(),
            download_is_default: settings.download_dir.is_none(),
            default_dirs: self.default_roots().iter().map(|p| p.to_string_lossy().into_owned()).collect(),
        }
    }

    /// Replaces the settings if every folder is usable, and leaves them alone if not.
    pub fn update(&self, wanted: Settings) -> Result<SettingsView, String> {
        let mut model_dirs: Vec<String> = Vec::new();
        for raw in wanted.model_dirs.iter().filter(|d| !d.trim().is_empty()) {
            let path = self.expand(raw);
            if !path.is_absolute() {
                return Err(format!("«{raw}» no es una ruta completa"));
            }
            if !path.is_dir() {
                return Err(format!("la carpeta «{raw}» no existe"));
            }
            let text = path.to_string_lossy().into_owned();
            if !model_dirs.contains(&text) {
                model_dirs.push(text);
            }
        }
        let download_dir = wanted.download_dir.filter(|d| !d.trim().is_empty());
        if let Some(raw) = &download_dir {
            let path = self.expand(raw);
            if !path.is_absolute() {
                return Err(format!("«{raw}» no es una ruta completa"));
            }
            std::fs::create_dir_all(&path).map_err(|e| format!("no se puede usar «{raw}»: {e}"))?;
        }
        let settings = Settings { model_dirs, download_dir };
        let dir = self.home.join(".llm2chip");
        std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        let draft = dir.join("settings.json.tmp");
        std::fs::write(&draft, serde_json::to_string_pretty(&settings).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
        std::fs::rename(&draft, dir.join("settings.json")).map_err(|e| e.to_string())?;
        *self.settings.lock().unwrap() = settings;
        self.forget();
        Ok(self.view())
    }

    pub fn forget(&self) {
        *self.listing.lock().unwrap() = None;
    }

    /// The models known now; a recent answer is reused, as the interface polls.
    pub fn models(&self) -> Vec<ModelEntry> {
        if let Some((taken, list)) = self.listing.lock().unwrap().as_ref() {
            if taken.elapsed() < KEEP_LISTING {
                return list.clone();
            }
        }
        let mut roots = self.default_roots();
        roots.push(self.download_dir());
        roots.extend(self.settings.lock().unwrap().model_dirs.iter().map(|d| self.expand(d)));
        let list = scan(&roots);
        *self.listing.lock().unwrap() = Some((Instant::now(), list.clone()));
        list
    }

    pub fn find(&self, path: &str) -> Option<ModelEntry> {
        self.models().into_iter().find(|m| m.path == path).or_else(|| {
            self.forget();
            self.models().into_iter().find(|m| m.path == path)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const LLAMA: &str = r#"{"model_type":"llama","hidden_size":16,"num_attention_heads":2,"intermediate_size":32,"vocab_size":8,"num_hidden_layers":1}"#;

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("llm2chip-catalog-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write_model(dir: &Path, config: &str, tokenizer: bool) {
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(dir.join("config.json"), config).unwrap();
        std::fs::write(dir.join("model.safetensors"), [0u8; 2_000]).unwrap();
        if tokenizer {
            std::fs::write(dir.join("tokenizer.json"), "{}").unwrap();
        }
    }

    #[test]
    fn folders_are_searched_a_few_levels_down_and_partial_downloads_are_skipped() {
        let root = scratch("depth");
        write_model(&root.join("autor/grupo/uno"), LLAMA, true);
        write_model(&root.join("demasiado/profundo/por/aqui/dos"), LLAMA, true);
        write_model(&root.join("a--b.partial"), LLAMA, true);
        let names: Vec<String> = scan(std::slice::from_ref(&root)).into_iter().map(|m| m.name).collect();
        assert_eq!(names, ["uno"]);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn the_hugging_face_cache_is_named_after_the_repository_and_measured_through_links() {
        let root = scratch("cache");
        let snapshot = root.join("models--org--modelo/snapshots/abc123");
        std::fs::create_dir_all(root.join("models--org--modelo/blobs")).unwrap();
        write_model(&snapshot, LLAMA, true);
        let blob = root.join("models--org--modelo/blobs/peso");
        std::fs::write(&blob, vec![0u8; 5_000]).unwrap();
        std::fs::remove_file(snapshot.join("model.safetensors")).unwrap();
        #[cfg(unix)]
        std::os::unix::fs::symlink(&blob, snapshot.join("model.safetensors")).unwrap();
        #[cfg(not(unix))]
        std::fs::copy(&blob, snapshot.join("model.safetensors")).unwrap();
        let list = scan(std::slice::from_ref(&root));
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].name, "org/modelo");
        assert_eq!(list[0].size_gb, 5e-6);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn each_model_says_how_far_it_can_be_adapted() {
        let root = scratch("tiers");
        write_model(&root.join("corre"), LLAMA, true);
        write_model(&root.join("sin-tokenizador"), LLAMA, false);
        write_model(&root.join("solo-diseno"), &LLAMA.replace("llama", "gemma2"), true);
        write_model(&root.join("desconocido"), r#"{"model_type":"bert"}"#, true);
        let list = scan(std::slice::from_ref(&root));
        let of = |name: &str| list.iter().find(|m| m.name == name).unwrap();
        assert!(of("corre").supported && of("corre").adaptation.parameters.is_some());
        assert!(!of("sin-tokenizador").supported && of("sin-tokenizador").adaptation.reason.as_ref().unwrap().contains("tokenizer"));
        assert!(of("solo-diseno").adaptation.design && of("solo-diseno").adaptation.approximate && !of("solo-diseno").supported);
        assert!(!of("desconocido").adaptation.design && of("desconocido").adaptation.reason.as_ref().unwrap().contains("bert"));
        assert_eq!(list[0].name, "corre");
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn settings_are_checked_kept_and_survive_a_restart() {
        let home = scratch("settings");
        let models = home.join("mis modelos");
        write_model(&models.join("uno"), LLAMA, true);
        let catalog = Catalog::open(home.clone());
        assert!(catalog.models().is_empty());

        let bad = Settings { model_dirs: vec!["~/no-existe".into()], download_dir: None };
        assert!(catalog.update(bad).unwrap_err().contains("no existe"));
        assert!(catalog.update(Settings { model_dirs: vec!["relativa".into()], download_dir: None }).is_err());

        let good = Settings { model_dirs: vec!["~/mis modelos".into(), models.to_string_lossy().into()], download_dir: Some("~/bajadas".into()) };
        catalog.update(good).unwrap();
        assert_eq!(catalog.models().len(), 1);
        assert_eq!(catalog.download_dir(), home.join("bajadas"));
        assert!(home.join("bajadas").is_dir());

        let again = Catalog::open(home.clone());
        assert_eq!(again.view().model_dirs.len(), 1);
        assert_eq!(again.models().len(), 1);
        std::fs::remove_dir_all(home).unwrap();
    }
}
