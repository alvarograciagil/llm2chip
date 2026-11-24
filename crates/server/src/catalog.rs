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
    pub images: ImageAbility,
    /// What it does in a spoken turn, if anything: `listens` (Moonshine), `speaks` (OuteTTS,
    /// a language model that writes speech's codes) or `sounds` (DAC's decoder, converted
    /// with tools/convert/dac.py).
    pub voice: Option<&'static str>,
}

/// What the checkpoint in `dir` does in a spoken turn, from its configuration and files.
pub fn voice_role(dir: &Path, config: &Value) -> Option<&'static str> {
    // Only the parts this engine was built with.
    match config.get("model_type").and_then(Value::as_str) {
        Some("moonshine" | "moonshine_streaming") if cfg!(feature = "listen") => Some("listens"),
        Some("dac") if cfg!(feature = "speak") => Some("sounds"),
        _ if cfg!(feature = "speak") && std::fs::read_to_string(dir.join("added_tokens.json")).is_ok_and(|t| t.contains("<|c1_0|>")) => Some("speaks"),
        _ => None,
    }
}

/// What a model does with images, and whether the simulator can follow it.
#[derive(Clone, Default, Serialize)]
pub struct ImageAbility {
    /// It reads images and the simulator runs its encoder.
    pub reads: bool,
    /// It makes images and the simulator runs its decoder.
    pub makes: bool,
    /// It draws from a text and retouches a region of a picture, apart from the chip: Show-o.
    pub edits: bool,
    /// What the model could do that the simulator cannot yet.
    pub note: Option<String>,
}

/// What the checkpoint in `dir` does with images, from its configuration and files.
pub fn image_ability(dir: &Path, config: &Value) -> ImageAbility {
    let model_type = config.get("model_type").and_then(Value::as_str).unwrap_or("");
    // Qwen3.5's encoder and Janus's SigLIP are the two the simulator runs.
    let reads = (sim::vision::VisionConfig::read(dir).ok().flatten().is_some() && model_type.starts_with("qwen3_5")) || (model_type == "janus" && config.get("vision_config").is_some());
    // Models that draw emit image tokens a decoder turns into pixels: Janus describes its
    // generator apart, Chameleon and Emu3 a VQ model.
    let draws = ["gen_vision_config", "gen_head_config", "gen_aligner_config", "vq_config", "vqmodel_config"].iter().any(|key| config.get(key).is_some());
    // Janus draws with a decoder the simulator runs.
    let makes = sim::janus::JanusConfig::read(dir).ok().flatten().is_some();
    let edits = sim::showo::ShowOConfig::read(dir).ok().flatten().is_some() && dir.join("magvitv2").is_dir();
    let note = if edits {
        Some("Show-o draws and retouches images on the Images screen, in floating point or as the chip would.".into())
    } else if draws && !makes {
        Some("The model generates images, but the simulator does not have its image decoder yet.".into())
    } else if config.get("vision_config").is_some() && !reads {
        Some("The model reads images with an encoder the simulator does not reproduce yet.".into())
    } else {
        None
    };
    ImageAbility { reads, makes, edits, note }
}

#[derive(Clone, Default, Serialize, Deserialize)]
pub struct Settings {
    #[serde(default)]
    pub model_dirs: Vec<String>,
    #[serde(default)]
    pub download_dir: Option<String>,
    /// Folders searched by default that the user turned off.
    #[serde(default)]
    pub disabled_defaults: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct SettingsView {
    model_dirs: Vec<String>,
    download_dir: String,
    download_is_default: bool,
    /// Searched unless turned off: the download folder, MTPLX's, the Hugging Face cache and
    /// those in `LLM2CHIP_MODEL_DIRS`.
    default_dirs: Vec<String>,
    disabled_defaults: Vec<String>,
    /// The home folder, so the interface can show paths under it as `~`.
    home: String,
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
        adaptation.reason = Some("tokenizer.json is missing, which is what turns the text into tokens".into());
    }
    let images = image_ability(dir, &value);
    let voice = voice_role(dir, &value);
    Some(ModelEntry { name, path: dir.to_string_lossy().into(), supported: adaptation.simulate, architecture, size_gb: size as f64 / 1e9, config, adaptation, images, voice })
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
        let mut roots = vec![self.default_download(), self.home.join(".mtplx/models"), hub];
        // Folders set outside the interface, separated as in PATH.
        if let Some(extra) = std::env::var_os("LLM2CHIP_MODEL_DIRS") {
            roots.extend(std::env::split_paths(&extra).filter(|p| !p.as_os_str().is_empty()).map(|p| self.expand(&p.to_string_lossy())));
        }
        roots
    }

    /// The default folders the user has not turned off.
    fn enabled_defaults(&self) -> Vec<PathBuf> {
        let disabled = self.settings.lock().unwrap().disabled_defaults.clone();
        self.default_roots().into_iter().filter(|p| !disabled.iter().any(|d| self.expand(d) == *p)).collect()
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
            disabled_defaults: settings.disabled_defaults.iter().map(|d| self.expand(d).to_string_lossy().into_owned()).collect(),
            home: self.home.to_string_lossy().into_owned(),
        }
    }

    /// Replaces the settings if every folder is usable, and leaves them alone if not.
    pub fn update(&self, wanted: Settings) -> Result<SettingsView, String> {
        let mut model_dirs: Vec<String> = Vec::new();
        for raw in wanted.model_dirs.iter().filter(|d| !d.trim().is_empty()) {
            let path = self.expand(raw);
            if !path.is_absolute() {
                return Err(format!("“{raw}” is not a full path"));
            }
            if !path.is_dir() {
                return Err(format!("the folder “{raw}” does not exist"));
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
                return Err(format!("“{raw}” is not a full path"));
            }
            std::fs::create_dir_all(&path).map_err(|e| format!("“{raw}” cannot be used: {e}"))?;
        }
        let defaults = self.default_roots();
        let mut disabled_defaults: Vec<String> = Vec::new();
        for raw in &wanted.disabled_defaults {
            let path = self.expand(raw);
            let text = path.to_string_lossy().into_owned();
            if defaults.contains(&path) && !disabled_defaults.contains(&text) {
                disabled_defaults.push(text);
            }
        }
        let settings = Settings { model_dirs, download_dir, disabled_defaults };
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
        let mut roots = self.enabled_defaults();
        // Where downloads go is always searched: a model just downloaded must show up.
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
        write_model(&root.join("author/group/one"), LLAMA, true);
        write_model(&root.join("too/deep/down/here/two"), LLAMA, true);
        write_model(&root.join("a--b.partial"), LLAMA, true);
        let names: Vec<String> = scan(std::slice::from_ref(&root)).into_iter().map(|m| m.name).collect();
        assert_eq!(names, ["one"]);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn the_hugging_face_cache_is_named_after_the_repository_and_measured_through_links() {
        let root = scratch("cache");
        let snapshot = root.join("models--org--model/snapshots/abc123");
        std::fs::create_dir_all(root.join("models--org--model/blobs")).unwrap();
        write_model(&snapshot, LLAMA, true);
        let blob = root.join("models--org--model/blobs/weight");
        std::fs::write(&blob, vec![0u8; 5_000]).unwrap();
        std::fs::remove_file(snapshot.join("model.safetensors")).unwrap();
        #[cfg(unix)]
        std::os::unix::fs::symlink(&blob, snapshot.join("model.safetensors")).unwrap();
        #[cfg(not(unix))]
        std::fs::copy(&blob, snapshot.join("model.safetensors")).unwrap();
        let list = scan(std::slice::from_ref(&root));
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].name, "org/model");
        assert_eq!(list[0].size_gb, 5e-6);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn each_model_says_how_far_it_can_be_adapted() {
        let root = scratch("tiers");
        write_model(&root.join("runs"), LLAMA, true);
        write_model(&root.join("no-tokenizer"), LLAMA, false);
        write_model(&root.join("design-only"), &LLAMA.replace("llama", "phi3"), true);
        write_model(&root.join("unknown"), r#"{"model_type":"bert"}"#, true);
        let list = scan(std::slice::from_ref(&root));
        let of = |name: &str| list.iter().find(|m| m.name == name).unwrap();
        assert!(of("runs").supported && of("runs").adaptation.parameters.is_some());
        assert!(!of("no-tokenizer").supported && of("no-tokenizer").adaptation.reason.as_ref().unwrap().contains("tokenizer"));
        assert!(of("design-only").adaptation.design && of("design-only").adaptation.approximate && !of("design-only").supported);
        assert!(!of("unknown").adaptation.design && of("unknown").adaptation.reason.as_ref().unwrap().contains("bert"));
        assert_eq!(list[0].name, "runs");
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn settings_are_checked_kept_and_survive_a_restart() {
        let home = scratch("settings");
        let models = home.join("my models");
        write_model(&models.join("one"), LLAMA, true);
        let catalog = Catalog::open(home.clone());
        assert!(catalog.models().is_empty());

        let bad = Settings { model_dirs: vec!["~/does-not-exist".into()], ..Settings::default() };
        assert!(catalog.update(bad).unwrap_err().contains("does not exist"));
        assert!(catalog.update(Settings { model_dirs: vec!["relative".into()], ..Settings::default() }).is_err());

        let good = Settings { model_dirs: vec!["~/my models".into(), models.to_string_lossy().into()], download_dir: Some("~/downloads".into()), ..Settings::default() };
        catalog.update(good).unwrap();
        assert_eq!(catalog.models().len(), 1);
        assert_eq!(catalog.download_dir(), home.join("downloads"));
        assert!(home.join("downloads").is_dir());

        let again = Catalog::open(home.clone());
        assert_eq!(again.view().model_dirs.len(), 1);
        assert_eq!(again.models().len(), 1);

        // A default folder can be turned off, and only default folders are kept as such.
        write_model(&home.join(".mtplx/models/two"), LLAMA, true);
        again.forget();
        assert_eq!(again.models().len(), 2);
        let off = Settings { model_dirs: vec![models.to_string_lossy().into()], download_dir: Some("~/downloads".into()), disabled_defaults: vec!["~/.mtplx/models".into(), "~/elsewhere".into()] };
        again.update(off).unwrap();
        assert_eq!(again.view().disabled_defaults, vec![home.join(".mtplx/models").to_string_lossy().into_owned()]);
        assert_eq!(again.models().len(), 1);
        std::fs::remove_dir_all(home).unwrap();
    }
}

#[cfg(test)]
mod image_tests {
    use super::*;

    #[test]
    fn a_model_says_what_it_does_with_images() {
        let dir = std::env::temp_dir().join(format!("llm2chip-images-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let ability = |config: &str| {
            std::fs::write(dir.join("config.json"), config).unwrap();
            image_ability(&dir, &serde_json::from_str(config).unwrap())
        };
        assert!(!ability(r#"{"model_type":"qwen3_5"}"#).reads);
        std::fs::write(dir.join("model-vision.safetensors"), b"").unwrap();
        let qwen = ability(r#"{"model_type":"qwen3_5","vision_config":{"depth":2}}"#);
        assert!(qwen.reads && qwen.note.is_none());
        let other = ability(r#"{"model_type":"llava","vision_config":{"depth":2}}"#);
        assert!(!other.reads && other.note.unwrap().contains("encoder"));
        let other = ability(r#"{"model_type":"multi_modality","gen_vision_config":{}}"#);
        assert!(!other.makes && other.note.unwrap().contains("generates images"));
        let janus = ability(r#"{"model_type":"janus","vq_config":{"num_patches":24},"vision_config":{"image_size":384}}"#);
        assert!(janus.makes && janus.reads && janus.note.is_none(), "Janus reads and draws");
        assert!(!janus.edits);
        let showo = r#"{"_class_name":"Showo","llm_vocab_size":50295,"codebook_size":8192,"mask_token_id":58497,"num_vq_tokens":256}"#;
        assert!(!ability(showo).edits, "without its MAGVIT-v2 it cannot");
        std::fs::create_dir_all(dir.join("magvitv2")).unwrap();
        let showo = ability(showo);
        assert!(showo.edits && !showo.makes && showo.note.unwrap().contains("Images"));
        std::fs::remove_dir_all(dir).ok();
    }
}
