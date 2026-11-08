//! Adapter for `Qwen3_5ForConditionalGeneration`, following `modeling_qwen3_5.py`.

use crate::{
    matvec, require_u64, tensor, vector, Block, ModelError, ModelGraph, Stage, StageKind,
    StateInfo, WeightTensor,
};
use serde_json::Value;

/// Layers whose MLP keeps 8-bit weights in the released mixed quantisation.
const WIDE_MLP_FROM_LAYER: u32 = 56;

struct Dims {
    hidden: u64,
    intermediate: u64,
    vocab: u64,
    heads: u64,
    kv_heads: u64,
    head_dim: u64,
    linear_key_heads: u64,
    linear_key_dim: u64,
    linear_value_heads: u64,
    linear_value_dim: u64,
    conv_kernel: u64,
    moe: Option<Moe>,
}

#[derive(Clone, Copy)]
struct Moe {
    experts: u64,
    top_k: u64,
    intermediate: u64,
    shared: u64,
}

pub(crate) fn build(name: &str, config: &Value) -> Result<ModelGraph, ModelError> {
    let text = config.get("text_config").unwrap_or(config);
    let moe = if text.get("num_experts").is_some() {
        Some(Moe {
            experts: require_u64(text, "num_experts")?,
            top_k: require_u64(text, "num_experts_per_tok")?,
            intermediate: require_u64(text, "moe_intermediate_size")?,
            shared: require_u64(text, "shared_expert_intermediate_size")?,
        })
    } else {
        None
    };
    let d = Dims {
        hidden: require_u64(text, "hidden_size")?,
        intermediate: if moe.is_some() { 0 } else { require_u64(text, "intermediate_size")? },
        vocab: require_u64(text, "vocab_size")?,
        heads: require_u64(text, "num_attention_heads")?,
        kv_heads: require_u64(text, "num_key_value_heads")?,
        head_dim: require_u64(text, "head_dim")?,
        linear_key_heads: require_u64(text, "linear_num_key_heads")?,
        linear_key_dim: require_u64(text, "linear_key_head_dim")?,
        linear_value_heads: require_u64(text, "linear_num_value_heads")?,
        linear_value_dim: require_u64(text, "linear_value_head_dim")?,
        conv_kernel: require_u64(text, "linear_conv_kernel_dim")?,
        moe,
    };
    let layer_types: Vec<Value> = match text.get("layer_types").and_then(Value::as_array) {
        Some(list) => list.clone(),
        None => {
            let interval = require_u64(text, "full_attention_interval")?;
            (1..=require_u64(text, "num_hidden_layers")?)
                .map(|i| Value::from(if i % interval == 0 { "full_attention" } else { "linear_attention" }))
                .collect()
        }
    };

    let mut tensors = vec![tensor(
        "embed_tokens".into(),
        d.vocab,
        d.hidden,
        Block::Embedding,
        None,
        true,
    )];
    let mut stages = vec![Stage {
        name: "embedding".into(),
        layer: None,
        block: Block::Embedding,
        kind: StageKind::EmbeddingRead { values: d.hidden },
    }];
    let mut state = StateInfo::default();

    for (i, kind) in layer_types.iter().enumerate() {
        let layer = i as u32;
        match kind.as_str() {
            Some("full_attention") => {
                add_full_attention(&d, layer, &mut tensors, &mut stages, &mut state)
            }
            Some("linear_attention") => {
                add_delta_net(&d, layer, &mut tensors, &mut stages, &mut state)
            }
            other => return Err(ModelError(format!("tipo de capa desconocido: {other:?}"))),
        }
        add_mlp(&d, layer, &mut tensors, &mut stages);
    }

    tensors.push(tensor("norm".into(), 1, d.hidden, Block::Norm, None, false));
    tensors.push(tensor("lm_head".into(), d.vocab, d.hidden, Block::Head, None, true));
    stages.push(vector("norm final", None, Block::Norm, 3 * d.hidden, 0));
    stages.push(matvec("lm_head", None, Block::Head, d.vocab * d.hidden));

    if text.get("mtp_num_hidden_layers").and_then(Value::as_u64).unwrap_or(0) > 0 {
        add_mtp_head(&d, &mut tensors, &mut stages);
    }
    if let Some(vision) = config.get("vision_config") {
        tensors.extend(vision_tensors(vision)?);
    }

    Ok(ModelGraph {
        name: name.into(),
        family: "qwen3_5".into(),
        hidden_size: d.hidden,
        num_layers: layer_types.len() as u32,
        vocab_size: d.vocab,
        max_context: text.get("max_position_embeddings").and_then(Value::as_u64).unwrap_or(0),
        tensors,
        stages,
        state,
    })
}

fn add_full_attention(
    d: &Dims,
    layer: u32,
    tensors: &mut Vec<WeightTensor>,
    stages: &mut Vec<Stage>,
    state: &mut StateInfo,
) {
    let l = Some(layer);
    let b = Block::FullAttention;
    let query = d.heads * d.head_dim;
    let kv = d.kv_heads * d.head_dim;
    let p = |s: &str| format!("layers.{layer}.self_attn.{s}");

    tensors.push(tensor(p("q_proj"), d.hidden, query * 2, b, l, false));
    tensors.push(tensor(p("k_proj"), d.hidden, kv, b, l, false));
    tensors.push(tensor(p("v_proj"), d.hidden, kv, b, l, false));
    tensors.push(tensor(p("o_proj"), query, d.hidden, b, l, false));
    tensors.push(tensor(p("q_norm"), 1, d.head_dim, Block::Norm, l, false));
    tensors.push(tensor(p("k_norm"), 1, d.head_dim, Block::Norm, l, false));
    add_layer_norms(d, layer, tensors);

    let qkv = d.hidden * (query * 2 + 2 * kv);
    stages.push(vector("norm de entrada", l, Block::Norm, 3 * d.hidden, 0));
    stages.push(matvec("q/k/v_proj", l, b, qkv));
    stages.push(vector("norm q/k y RoPE", l, b, 4 * (query + kv), 0));
    // Score against every cached key, then sum every cached value.
    stages.push(vector("atención (Q·K, softmax, suma V)", l, b, 3 * query, 2 * query));
    stages.push(vector("puerta sigmoide", l, b, 2 * query, 0));
    stages.push(matvec("o_proj", l, b, query * d.hidden));

    state.kv_values_per_token += 2 * kv;
}

fn add_delta_net(
    d: &Dims,
    layer: u32,
    tensors: &mut Vec<WeightTensor>,
    stages: &mut Vec<Stage>,
    state: &mut StateInfo,
) {
    let l = Some(layer);
    let b = Block::DeltaNet;
    let key_dim = d.linear_key_heads * d.linear_key_dim;
    let value_dim = d.linear_value_heads * d.linear_value_dim;
    let conv_dim = key_dim * 2 + value_dim;
    let p = |s: &str| format!("layers.{layer}.linear_attn.{s}");

    tensors.push(tensor(p("in_proj_qkv"), d.hidden, conv_dim, b, l, false));
    tensors.push(tensor(p("in_proj_z"), d.hidden, value_dim, b, l, false));
    tensors.push(tensor(p("in_proj_a"), d.hidden, d.linear_value_heads, b, l, false));
    tensors.push(tensor(p("in_proj_b"), d.hidden, d.linear_value_heads, b, l, false));
    tensors.push(tensor(p("out_proj"), value_dim, d.hidden, b, l, true));
    tensors.push(tensor(p("conv1d"), conv_dim, d.conv_kernel, b, l, false));
    tensors.push(tensor(p("A_log"), 1, d.linear_value_heads, b, l, false));
    tensors.push(tensor(p("dt_bias"), 1, d.linear_value_heads, b, l, false));
    tensors.push(tensor(p("norm"), 1, d.linear_value_dim, Block::Norm, l, false));
    add_layer_norms(d, layer, tensors);

    let state_values = d.linear_value_heads * d.linear_key_dim * d.linear_value_dim;
    let input = d.hidden * (conv_dim + value_dim + 2 * d.linear_value_heads);
    stages.push(vector("norm de entrada", l, Block::Norm, 3 * d.hidden, 0));
    stages.push(matvec("in_proj qkv/z/a/b", l, b, input));
    stages.push(vector("convolución causal", l, b, conv_dim * (d.conv_kernel + 1), 0));
    stages.push(vector("β y decaimiento", l, b, 3 * d.linear_value_heads, 0));
    // decay, kv_mem, state update and output each touch the whole state once.
    let l2 = 4 * d.linear_value_heads * d.linear_key_dim;
    stages.push(vector("regla delta", l, b, 4 * state_values + l2, 0));
    stages.push(vector("norm con puerta", l, b, 4 * value_dim, 0));
    stages.push(matvec("out_proj", l, b, value_dim * d.hidden));

    state.delta_values += state_values;
    state.conv_values += conv_dim * (d.conv_kernel - 1);
}

fn add_layer_norms(d: &Dims, layer: u32, tensors: &mut Vec<WeightTensor>) {
    let l = Some(layer);
    tensors.push(tensor(format!("layers.{layer}.input_layernorm"), 1, d.hidden, Block::Norm, l, false));
    tensors.push(tensor(
        format!("layers.{layer}.post_attention_layernorm"),
        1,
        d.hidden,
        Block::Norm,
        l,
        false,
    ));
}

fn add_moe(d: &Dims, m: Moe, layer: u32, tensors: &mut Vec<WeightTensor>, stages: &mut Vec<Stage>) {
    let l = Some(layer);
    let b = Block::Mlp;
    let p = |s: &str| format!("layers.{layer}.mlp.{s}");
    tensors.push(tensor(p("gate"), d.hidden, m.experts, b, l, false));
    tensors.push(tensor(p("switch_mlp.gate_proj"), d.hidden, m.experts * m.intermediate, b, l, false));
    tensors.push(tensor(p("switch_mlp.up_proj"), d.hidden, m.experts * m.intermediate, b, l, false));
    tensors.push(tensor(p("switch_mlp.down_proj"), m.experts * m.intermediate, d.hidden, b, l, false));
    tensors.push(tensor(p("shared_expert.gate_proj"), d.hidden, m.shared, b, l, false));
    tensors.push(tensor(p("shared_expert.up_proj"), d.hidden, m.shared, b, l, false));
    tensors.push(tensor(p("shared_expert.down_proj"), m.shared, d.hidden, b, l, false));
    tensors.push(tensor(p("shared_expert_gate"), d.hidden, 1, b, l, false));

    stages.push(vector("residual y norm", l, Block::Norm, 4 * d.hidden, 0));
    stages.push(matvec("enrutador de expertos", l, b, d.hidden * m.experts));
    stages.push(vector("softmax y selección", l, b, 3 * m.experts, 0));
    // Only the chosen experts' weights are read for each token.
    stages.push(matvec("expertos activos", l, b, m.top_k * 3 * d.hidden * m.intermediate));
    stages.push(vector("SiLU(gate)·up", l, b, 2 * m.top_k * m.intermediate, 0));
    stages.push(matvec("experto compartido", l, b, 3 * d.hidden * m.shared + d.hidden));
    stages.push(vector("residual", l, Block::Norm, d.hidden, 0));
}

fn add_mlp(d: &Dims, layer: u32, tensors: &mut Vec<WeightTensor>, stages: &mut Vec<Stage>) {
    if let Some(m) = d.moe {
        add_moe(d, m, layer, tensors, stages);
        return;
    }
    let l = Some(layer);
    let b = Block::Mlp;
    let wide = layer >= WIDE_MLP_FROM_LAYER;
    let p = |s: &str| format!("layers.{layer}.mlp.{s}");

    tensors.push(tensor(p("gate_proj"), d.hidden, d.intermediate, b, l, wide));
    tensors.push(tensor(p("up_proj"), d.hidden, d.intermediate, b, l, wide));
    tensors.push(tensor(p("down_proj"), d.intermediate, d.hidden, b, l, wide));

    stages.push(vector("residual y norm", l, Block::Norm, 4 * d.hidden, 0));
    stages.push(matvec("gate/up_proj", l, b, 2 * d.hidden * d.intermediate));
    stages.push(vector("SiLU(gate)·up", l, b, 2 * d.intermediate, 0));
    stages.push(matvec("down_proj", l, b, d.intermediate * d.hidden));
    stages.push(vector("residual", l, Block::Norm, d.hidden, 0));
}

/// The draft head stored in `mtp.safetensors`: `fc` over `embedding_hidden` and
/// one attention plus MLP layer. It shares `lm_head` with the main model.
fn add_mtp_head(d: &Dims, tensors: &mut Vec<WeightTensor>, stages: &mut Vec<Stage>) {
    let b = Block::Mtp;
    let query = d.heads * d.head_dim;
    let kv = d.kv_heads * d.head_dim;
    let fc = 2 * d.hidden;
    for (name, rows, cols) in [
        ("mtp.fc", fc, d.hidden),
        ("mtp.q_proj", d.hidden, query * 2),
        ("mtp.k_proj", d.hidden, kv),
        ("mtp.v_proj", d.hidden, kv),
        ("mtp.o_proj", query, d.hidden),
        ("mtp.gate_proj", d.hidden, d.moe.map_or(d.intermediate, |m| m.experts * m.intermediate)),
        ("mtp.up_proj", d.hidden, d.moe.map_or(d.intermediate, |m| m.experts * m.intermediate)),
        ("mtp.down_proj", d.moe.map_or(d.intermediate, |m| m.experts * m.intermediate), d.hidden),
    ] {
        tensors.push(tensor(name.into(), rows, cols, b, None, false));
    }

    let attn = d.hidden * (query * 2 + 2 * kv) + query * d.hidden;
    let mlp = 3 * d.hidden * d.moe.map_or(d.intermediate, |m| m.top_k * m.intermediate);
    stages.push(vector("mtp: normas", None, b, 8 * d.hidden, 0));
    stages.push(matvec("mtp: fc", None, b, fc * d.hidden));
    stages.push(matvec("mtp: atención", None, b, attn));
    stages.push(vector("mtp: cálculo de atención", None, b, 8 * query, 0));
    stages.push(matvec("mtp: mlp", None, b, mlp));
    stages.push(vector("mtp: activación", None, b, 2 * d.moe.map_or(d.intermediate, |m| m.top_k * m.intermediate), 0));
    stages.push(matvec("mtp: lm_head", None, b, d.vocab * d.hidden));
}

fn vision_tensors(vision: &Value) -> Result<Vec<WeightTensor>, ModelError> {
    let depth = require_u64(vision, "depth")?;
    let hidden = require_u64(vision, "hidden_size")?;
    let intermediate = require_u64(vision, "intermediate_size")?;
    let attention = depth * 4 * hidden * hidden;
    let mlp = depth * 2 * hidden * intermediate;
    Ok(vec![tensor("visual".into(), 1, attention + mlp, Block::Vision, None, false)])
}
