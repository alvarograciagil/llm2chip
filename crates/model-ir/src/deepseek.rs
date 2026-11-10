//! Adapter for DeepSeek V3 and R1: latent attention and a sigmoid-routed mixture of experts.

use crate::{matvec, require_u64, tensor, vector, Block, ModelError, ModelGraph, Stage, StageKind, StateInfo};
use serde_json::Value;

pub(crate) fn build(name: &str, c: &Value) -> Result<ModelGraph, ModelError> {
    let get = |key: &str| require_u64(c, key);
    let (hidden, vocab, layers, heads) = (get("hidden_size")?, get("vocab_size")?, get("num_hidden_layers")?, get("num_attention_heads")?);
    let (lora, rope, nope, v_head) = (get("kv_lora_rank")?, get("qk_rope_head_dim")?, get("qk_nope_head_dim")?, get("v_head_dim")?);
    let q_rank = c.get("q_lora_rank").and_then(Value::as_u64);
    let intermediate = get("intermediate_size")?;
    let first_dense = c.get("first_k_dense_replace").and_then(Value::as_u64).unwrap_or(0);
    let frequency = c.get("moe_layer_freq").and_then(Value::as_u64).unwrap_or(1).max(1);
    let routed = c.get("n_routed_experts").and_then(Value::as_u64);
    let shared = c.get("n_shared_experts").and_then(Value::as_u64).unwrap_or(0);
    let top_k = c.get("num_experts_per_tok").and_then(Value::as_u64).unwrap_or(1);
    let moe_size = c.get("moe_intermediate_size").and_then(Value::as_u64).unwrap_or(intermediate);
    let q_head = nope + rope;

    let mut tensors = vec![tensor("model.embed_tokens".into(), vocab, hidden, Block::Embedding, None, true)];
    let mut stages = vec![Stage { name: "embedding".into(), layer: None, block: Block::Embedding, kind: StageKind::EmbeddingRead { values: hidden } }];
    for i in 0..layers {
        let layer = i as u32;
        let l = Some(layer);
        let p = |s: &str| format!("model.layers.{i}.{s}");
        let a = Block::FullAttention;
        let q_weights = match q_rank {
            Some(rank) => {
                tensors.push(tensor(p("self_attn.q_a_proj"), rank, hidden, a, l, false));
                tensors.push(tensor(p("self_attn.q_b_proj"), heads * q_head, rank, a, l, false));
                rank * hidden + heads * q_head * rank
            }
            None => {
                tensors.push(tensor(p("self_attn.q_proj"), heads * q_head, hidden, a, l, false));
                heads * q_head * hidden
            }
        };
        tensors.push(tensor(p("self_attn.kv_a_proj_with_mqa"), lora + rope, hidden, a, l, false));
        tensors.push(tensor(p("self_attn.embed_q"), heads * lora, nope, a, l, false));
        tensors.push(tensor(p("self_attn.unembed_out"), heads * v_head, lora, a, l, false));
        tensors.push(tensor(p("self_attn.o_proj"), hidden, heads * v_head, a, l, false));
        tensors.push(tensor(p("input_layernorm"), 1, hidden, Block::Norm, l, false));
        tensors.push(tensor(p("post_attention_layernorm"), 1, hidden, Block::Norm, l, false));
        stages.push(vector("norm de entrada", l, Block::Norm, 3 * hidden, 0));
        stages.push(matvec("q y kv latentes", l, a, q_weights + hidden * (lora + rope)));
        stages.push(vector("RoPE", l, a, 4 * rope * (heads + 1), 0));
        stages.push(matvec("embed_q", l, a, heads * lora * nope));
        stages.push(vector("atención (latente, softmax, suma)", l, a, 3 * heads, heads * (lora + rope)));
        stages.push(matvec("unembed y salida", l, a, heads * v_head * lora + heads * v_head * hidden));
        stages.push(vector("residual y norm", l, Block::Norm, 4 * hidden, 0));

        let m = Block::Mlp;
        if let Some(experts) = routed.filter(|_| i >= first_dense && i % frequency == 0) {
            tensors.push(tensor(p("mlp.gate"), experts, hidden, m, l, false));
            tensors.push(tensor(p("mlp.switch_mlp.gate_proj"), experts * moe_size, hidden, m, l, false));
            tensors.push(tensor(p("mlp.switch_mlp.up_proj"), experts * moe_size, hidden, m, l, false));
            tensors.push(tensor(p("mlp.switch_mlp.down_proj"), experts * hidden, moe_size, m, l, false));
            stages.push(matvec("enrutador de expertos", l, m, experts * hidden));
            stages.push(vector("sigmoide y selección", l, m, 3 * experts, 0));
            stages.push(matvec("expertos elegidos", l, m, top_k * 3 * hidden * moe_size));
            stages.push(vector("activación de expertos", l, m, 2 * top_k * moe_size, 0));
            if shared > 0 {
                let size = moe_size * shared;
                tensors.push(tensor(p("mlp.shared_experts.gate_proj"), size, hidden, m, l, false));
                tensors.push(tensor(p("mlp.shared_experts.up_proj"), size, hidden, m, l, false));
                tensors.push(tensor(p("mlp.shared_experts.down_proj"), hidden, size, m, l, false));
                stages.push(matvec("experto compartido", l, m, 3 * hidden * size));
            }
        } else {
            tensors.push(tensor(p("mlp.gate_proj"), intermediate, hidden, m, l, false));
            tensors.push(tensor(p("mlp.up_proj"), intermediate, hidden, m, l, false));
            tensors.push(tensor(p("mlp.down_proj"), hidden, intermediate, m, l, false));
            stages.push(matvec("mlp", l, m, 3 * hidden * intermediate));
            stages.push(vector("activación", l, m, 2 * intermediate, 0));
        }
        stages.push(vector("residual", l, Block::Norm, hidden, 0));
    }
    tensors.push(tensor("model.norm".into(), 1, hidden, Block::Norm, None, false));
    tensors.push(tensor("lm_head".into(), vocab, hidden, Block::Head, None, true));
    stages.push(vector("norm final", None, Block::Norm, 3 * hidden, 0));
    stages.push(matvec("salida de vocabulario", None, Block::Head, vocab * hidden));
    Ok(ModelGraph {
        name: name.into(),
        family: "deepseek_v3".into(),
        hidden_size: hidden,
        num_layers: layers as u32,
        vocab_size: vocab,
        max_context: c.get("max_position_embeddings").and_then(Value::as_u64).unwrap_or(0),
        tensors,
        stages,
        state: StateInfo { kv_values_per_token: layers * (lora + rope), delta_values: 0, conv_values: 0 },
        vision: None,
    })
}
