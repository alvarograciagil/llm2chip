//! Hardware-independent description of a language model: its weight tensors,
//! the ordered stages of one forward pass, and the mutable state it keeps.

mod deepseek;
mod llama;
mod qwen3_5;

use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Block {
    Embedding,
    Head,
    FullAttention,
    DeltaNet,
    Mlp,
    Norm,
    Mtp,
    Vision,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WeightTensor {
    pub name: String,
    pub rows: u64,
    pub cols: u64,
    pub block: Block,
    pub layer: Option<u32>,
    /// Kept at 8 bits by the mixed quantisation recipes used on real checkpoints.
    pub wide: bool,
}

impl WeightTensor {
    pub fn params(&self) -> u64 {
        self.rows * self.cols
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StageKind {
    /// Matrix-vector products that use `weights` stored weights once each.
    Matvec { weights: u64 },
    /// Activation or state arithmetic: `ops` plus `ops_per_ctx` for every context token.
    Vector { ops: u64, ops_per_ctx: u64 },
    /// One row of an embedding table is read; nothing is multiplied.
    EmbeddingRead { values: u64 },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Stage {
    pub name: String,
    pub layer: Option<u32>,
    pub block: Block,
    pub kind: StageKind,
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct StateInfo {
    /// Key and value entries appended to the cache for each context token.
    pub kv_values_per_token: u64,
    /// Recurrent matrix entries held across all linear-attention layers.
    pub delta_values: u64,
    /// Input history held by the causal convolutions.
    pub conv_values: u64,
}

/// The shape of a model's image encoder: a ViT on patches, then a merger of
/// `merge × merge` of them into each token of the language model.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct VisionShape {
    pub depth: u64,
    pub hidden: u64,
    pub intermediate: u64,
    pub patch: u64,
    pub temporal: u64,
    pub merge: u64,
    pub out_hidden: u64,
}

impl VisionShape {
    /// Multiplications to encode an image of `width × height` pixels, which the encoder
    /// first rounds to whole merged tokens: every patch through every block, attention
    /// against every other patch, and the merger once per token.
    pub fn macs_per_image(&self, width: u64, height: u64) -> u64 {
        let side = self.patch * self.merge;
        let patches = width.div_ceil(side) * height.div_ceil(side) * self.merge * self.merge;
        let embed = 3 * self.temporal * self.patch * self.patch * self.hidden;
        let block = 4 * self.hidden * self.hidden + 2 * self.hidden * self.intermediate + 2 * patches * self.hidden;
        let merged = self.hidden * self.merge * self.merge;
        let merger = (merged * merged + merged * self.out_hidden) / (self.merge * self.merge);
        patches * (embed + self.depth * block + merger)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelGraph {
    pub name: String,
    pub family: String,
    pub hidden_size: u64,
    pub num_layers: u32,
    pub vocab_size: u64,
    pub max_context: u64,
    pub tensors: Vec<WeightTensor>,
    pub stages: Vec<Stage>,
    pub state: StateInfo,
    /// The image encoder, for models that read images.
    #[serde(default)]
    pub vision: Option<VisionShape>,
}

/// The `model_type`s a circuit can be estimated for.
pub fn designable_family(model_type: &str) -> bool {
    matches!(model_type, "qwen3_5" | "qwen3_5_text" | "qwen3_5_moe" | "qwen3_5_moe_text" | "deepseek_v3") || llama::FAMILIES.contains(&model_type)
}

/// Families estimated from their shape alone: what sets them apart is not modelled.
pub fn approximate_family(model_type: &str) -> bool {
    llama::APPROXIMATE.contains(&model_type)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelError(pub String);

impl std::fmt::Display for ModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ModelError {}

impl ModelGraph {
    /// Builds the graph from a checkpoint's `config.json`, without touching its weights.
    pub fn from_config_json(name: &str, json: &str) -> Result<Self, ModelError> {
        let config: Value =
            serde_json::from_str(json).map_err(|e| ModelError(format!("config inválido: {e}")))?;
        let model_type = config.get("model_type").and_then(Value::as_str);
        match model_type {
            Some("qwen3_5") | Some("qwen3_5_text") | Some("qwen3_5_moe") | Some("qwen3_5_moe_text") => qwen3_5::build(name, &config),
            Some("deepseek_v3") => deepseek::build(name, &config),
            Some(family) if llama::FAMILIES.contains(&family) => llama::build_hf(name, family, &config),
            Some(other) => Err(ModelError(format!(
                "la arquitectura «{other}» todavía no está soportada"
            ))),
            None if config.get("n_layers").is_some() => llama::build_llama2c(name, &config),
            None => Err(ModelError("el config no indica model_type".into())),
        }
    }

    pub fn params_where(&self, keep: impl Fn(&WeightTensor) -> bool) -> u64 {
        self.tensors.iter().filter(|t| keep(t)).map(WeightTensor::params).sum()
    }

    pub fn total_params(&self) -> u64 {
        self.params_where(|_| true)
    }

    pub fn params_in(&self, block: Block) -> u64 {
        self.params_where(|t| t.block == block)
    }
}

pub(crate) fn require_u64(config: &Value, key: &str) -> Result<u64, ModelError> {
    config
        .get(key)
        .and_then(Value::as_u64)
        .ok_or_else(|| ModelError(format!("falta el campo numérico «{key}» en el config")))
}

pub(crate) fn tensor(
    name: String,
    rows: u64,
    cols: u64,
    block: Block,
    layer: Option<u32>,
    wide: bool,
) -> WeightTensor {
    WeightTensor { name, rows, cols, block, layer, wide }
}

pub(crate) fn matvec(name: &str, layer: Option<u32>, block: Block, weights: u64) -> Stage {
    Stage { name: name.into(), layer, block, kind: StageKind::Matvec { weights } }
}

pub(crate) fn vector(
    name: &str,
    layer: Option<u32>,
    block: Block,
    ops: u64,
    ops_per_ctx: u64,
) -> Stage {
    Stage { name: name.into(), layer, block, kind: StageKind::Vector { ops, ops_per_ctx } }
}

#[cfg(test)]
mod tests {
    use super::*;

    const QWEN: &str = include_str!("../../../models/qwen3.8-27b.json");

    #[test]
    fn qwen_matches_published_parameter_count() {
        let m = ModelGraph::from_config_json("qwen", QWEN).unwrap();
        let total = m.total_params() as f64;
        assert!((total / 27.78e9 - 1.0).abs() < 0.005, "total {total}");
    }

    #[test]
    fn qwen_mtp_head_matches_released_sidecar() {
        // The BF16 sidecar is 849,400,388 bytes, about 424.7M parameters.
        let m = ModelGraph::from_config_json("qwen", QWEN).unwrap();
        let mtp = m.params_in(Block::Mtp) as f64;
        assert!((mtp / 424.7e6 - 1.0).abs() < 0.005, "mtp {mtp}");
    }

    #[test]
    fn qwen_state_matches_reference_code() {
        let m = ModelGraph::from_config_json("qwen", QWEN).unwrap();
        assert_eq!(m.state.kv_values_per_token, 32_768);
        assert_eq!(m.state.delta_values, 48 * 48 * 128 * 128);
    }

    #[test]
    fn llama_8b_and_stories_match_known_sizes() {
        let llama = include_str!("../../../models/llama-3.1-8b.json");
        let total = ModelGraph::from_config_json("l", llama).unwrap().total_params() as f64;
        assert!((total / 8.03e9 - 1.0).abs() < 0.01, "llama {total}");

        let stories = include_str!("../../../models/stories260k.json");
        let total = ModelGraph::from_config_json("s", stories).unwrap().total_params() as f64;
        assert!((total / 260e3 - 1.0).abs() < 0.02, "stories {total}");
    }

    #[test]
    fn mixture_of_experts_counts_all_weights_but_reads_only_the_active_ones() {
        let m = ModelGraph::from_config_json("moe", include_str!("../../../models/qwen3.6-35b-a3b.json")).unwrap();
        let total = m.total_params() as f64;
        assert!((total / 35e9 - 1.0).abs() < 0.08, "total {total}");
        let read: u64 = m.stages.iter().filter(|s| s.block != Block::Mtp).map(|s| if let StageKind::Matvec { weights } = s.kind { weights } else { 0 }).sum();
        assert!(read as f64 / total < 0.2, "lee {read} de {total}");
    }

    #[test]
    fn other_architectures_take_the_shape_their_config_states() {
        let mistral = r#"{"model_type":"mistral","hidden_size":4096,"intermediate_size":14336,"vocab_size":32000,"num_hidden_layers":32,"num_attention_heads":32,"num_key_value_heads":8}"#;
        let m = ModelGraph::from_config_json("m", mistral).unwrap();
        assert_eq!(m.family, "mistral");
        assert!((m.total_params() as f64 / 7.3e9 - 1.0).abs() < 0.03, "{}", m.total_params());

        let gemma = r#"{"model_type":"gemma2","hidden_size":2304,"intermediate_size":9216,"vocab_size":256000,"num_hidden_layers":26,"num_attention_heads":8,"head_dim":288}"#;
        let g = ModelGraph::from_config_json("g", gemma).unwrap();
        assert!(!g.tensors.iter().any(|t| t.block == Block::Head), "Gemma comparte embedding y salida salvo que el config diga otra cosa");
        assert!(approximate_family("phi3") && !approximate_family("gemma2"));
    }

    #[test]
    fn unknown_architecture_is_reported() {
        let err = ModelGraph::from_config_json("x", r#"{"model_type":"bert"}"#).unwrap_err();
        assert!(err.0.contains("bert"));
    }
}
