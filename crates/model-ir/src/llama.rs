//! Adapters for plain Llama-style transformers: Hugging Face configs and llama2.c arguments.

use crate::{
    matvec, require_u64, tensor, vector, Block, ModelError, ModelGraph, Stage, StageKind,
    StateInfo,
};
use serde_json::Value;

struct Dims {
    hidden: u64,
    intermediate: u64,
    vocab: u64,
    layers: u64,
    heads: u64,
    kv_heads: u64,
    tied_embeddings: bool,
    max_context: u64,
}

pub(crate) fn build_hf(name: &str, config: &Value) -> Result<ModelGraph, ModelError> {
    let heads = require_u64(config, "num_attention_heads")?;
    let dims = Dims {
        hidden: require_u64(config, "hidden_size")?,
        intermediate: require_u64(config, "intermediate_size")?,
        vocab: require_u64(config, "vocab_size")?,
        layers: require_u64(config, "num_hidden_layers")?,
        heads,
        kv_heads: config.get("num_key_value_heads").and_then(Value::as_u64).unwrap_or(heads),
        tied_embeddings: config
            .get("tie_word_embeddings")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        max_context: config.get("max_position_embeddings").and_then(Value::as_u64).unwrap_or(0),
    };
    Ok(build(name, "llama", dims))
}

pub(crate) fn build_llama2c(name: &str, config: &Value) -> Result<ModelGraph, ModelError> {
    let hidden = require_u64(config, "dim")?;
    let multiple_of = require_u64(config, "multiple_of")?;
    let heads = require_u64(config, "n_heads")?;
    let rounded = 2 * (4 * hidden) / 3;
    let dims = Dims {
        hidden,
        intermediate: multiple_of * rounded.div_ceil(multiple_of),
        vocab: require_u64(config, "vocab_size")?,
        layers: require_u64(config, "n_layers")?,
        heads,
        kv_heads: config.get("n_kv_heads").and_then(Value::as_u64).unwrap_or(heads),
        tied_embeddings: true,
        max_context: config.get("max_seq_len").and_then(Value::as_u64).unwrap_or(0),
    };
    Ok(build(name, "llama2c", dims))
}

fn build(name: &str, family: &str, d: Dims) -> ModelGraph {
    let head_dim = d.hidden / d.heads;
    let kv = d.kv_heads * head_dim;
    let mut tensors = vec![tensor("tok_embeddings".into(), d.vocab, d.hidden, Block::Embedding, None, true)];
    let mut stages = vec![Stage {
        name: "embedding".into(),
        layer: None,
        block: Block::Embedding,
        kind: StageKind::EmbeddingRead { values: d.hidden },
    }];

    for i in 0..d.layers {
        let layer = i as u32;
        let l = Some(layer);
        let p = |s: &str| format!("layers.{layer}.{s}");
        let a = Block::FullAttention;
        let m = Block::Mlp;
        tensors.push(tensor(p("wq"), d.hidden, d.hidden, a, l, false));
        tensors.push(tensor(p("wk"), d.hidden, kv, a, l, false));
        tensors.push(tensor(p("wv"), d.hidden, kv, a, l, false));
        tensors.push(tensor(p("wo"), d.hidden, d.hidden, a, l, false));
        tensors.push(tensor(p("w1"), d.hidden, d.intermediate, m, l, false));
        tensors.push(tensor(p("w2"), d.intermediate, d.hidden, m, l, false));
        tensors.push(tensor(p("w3"), d.hidden, d.intermediate, m, l, false));
        tensors.push(tensor(p("attention_norm"), 1, d.hidden, Block::Norm, l, false));
        tensors.push(tensor(p("ffn_norm"), 1, d.hidden, Block::Norm, l, false));

        stages.push(vector("norm de entrada", l, Block::Norm, 3 * d.hidden, 0));
        stages.push(matvec("q/k/v", l, a, d.hidden * (d.hidden + 2 * kv)));
        stages.push(vector("RoPE", l, a, 4 * (d.hidden + kv), 0));
        stages.push(vector("atención (Q·K, softmax, suma V)", l, a, 3 * d.hidden, 2 * d.hidden));
        stages.push(matvec("salida", l, a, d.hidden * d.hidden));
        stages.push(vector("residual y norm", l, Block::Norm, 4 * d.hidden, 0));
        stages.push(matvec("gate/up", l, m, 2 * d.hidden * d.intermediate));
        stages.push(vector("SiLU(gate)·up", l, m, 2 * d.intermediate, 0));
        stages.push(matvec("down", l, m, d.intermediate * d.hidden));
        stages.push(vector("residual", l, Block::Norm, d.hidden, 0));
    }

    tensors.push(tensor("norm".into(), 1, d.hidden, Block::Norm, None, false));
    if !d.tied_embeddings {
        tensors.push(tensor("output".into(), d.vocab, d.hidden, Block::Head, None, true));
    }
    stages.push(vector("norm final", None, Block::Norm, 3 * d.hidden, 0));
    stages.push(matvec("salida de vocabulario", None, Block::Head, d.vocab * d.hidden));

    ModelGraph {
        name: name.into(),
        family: family.into(),
        hidden_size: d.hidden,
        num_layers: d.layers as u32,
        vocab_size: d.vocab,
        max_context: d.max_context,
        tensors,
        stages,
        state: StateInfo { kv_values_per_token: d.layers * 2 * kv, delta_values: 0, conv_values: 0 },
    }
}
