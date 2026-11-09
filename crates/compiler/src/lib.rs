//! Turns a model into the microprogram the chip executes, one token at a time.

use serde::Serialize;
use serde_json::Value;

mod schedule;
pub use schedule::{schedule, Schedule, ScheduledStage};

pub type Buf = usize;

#[derive(Clone, Debug, Serialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum Op {
    EmbRead { table: String, dst: Buf },
    Matvec { matrix: String, src: Buf, dst: Buf },
    Copy { src: Buf, dst: Buf },
    Add { a: Buf, b: Buf, dst: Buf },
    MulElem { a: Buf, b: Buf, dst: Buf },
    Silu { src: Buf, dst: Buf },
    /// `x · rsqrt(mean(x²) + eps) · (offset + w)`, per block of `block` values.
    RmsNorm { src: Buf, weight: String, offset: f32, block: usize, dst: Buf },
    /// `q_proj` interleaves each head's query and gate.
    SplitQueryGate { src: Buf, query: Buf, gate: Buf, block: usize },
    Rope { buf: Buf, block: usize, rotary: usize },
    KvAppend { slot: usize, key: Buf, value: Buf },
    AttnSum { slot: usize, query: Buf, head_dim: usize, heads: usize, kv_heads: usize, dst: Buf },
    Sigmoid { src: Buf, dst: Buf },
    Dconv { slot: usize, weight: String, src: Buf, dst: Buf },
    DeltaStep { slot: usize, mixed: Buf, a: Buf, b: Buf, a_log: String, dt_bias: String, key_dim: usize, key_heads: usize, value_dim: usize, value_heads: usize, dst: Buf },
    RmsGated { src: Buf, z: Buf, weight: String, block: usize, dst: Buf },
    Concat { a: Buf, b: Buf, dst: Buf },
    /// Sparse mixture of experts: routes to `top_k` experts plus a shared expert.
    Moe { prefix: String, src: Buf, dst: Buf, experts: usize, top_k: usize, intermediate: usize, shared_intermediate: usize },
}

impl Op {
    /// The buffers this operation reads and the ones it writes.
    pub fn buffers(&self) -> (Vec<Buf>, Vec<Buf>) {
        match self {
            Op::EmbRead { dst, .. } => (vec![], vec![*dst]),
            Op::Matvec { src, dst, .. } | Op::Copy { src, dst } | Op::Silu { src, dst } | Op::Sigmoid { src, dst } | Op::RmsNorm { src, dst, .. } | Op::Dconv { src, dst, .. } | Op::Moe { src, dst, .. } => (vec![*src], vec![*dst]),
            Op::Add { a, b, dst } | Op::MulElem { a, b, dst } | Op::Concat { a, b, dst } => (vec![*a, *b], vec![*dst]),
            Op::SplitQueryGate { src, query, gate, .. } => (vec![*src], vec![*query, *gate]),
            Op::Rope { buf, .. } => (vec![*buf], vec![*buf]),
            Op::KvAppend { key, value, .. } => (vec![*key, *value], vec![]),
            Op::AttnSum { query, dst, .. } => (vec![*query], vec![*dst]),
            Op::DeltaStep { mixed, a, b, dst, .. } => (vec![*mixed, *a, *b], vec![*dst]),
            Op::RmsGated { src, z, dst, .. } => (vec![*src, *z], vec![*dst]),
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct Inst {
    pub op: Op,
    pub layer: Option<u32>,
    pub name: &'static str,
}

#[derive(Clone, Debug, Serialize)]
pub struct Program {
    pub family: String,
    pub vocab: usize,
    pub hidden: usize,
    pub num_layers: usize,
    pub buffers: Vec<usize>,
    pub ops: Vec<Inst>,
    /// Index into `ops` where producing the vocabulary logits begins.
    pub logits_from: usize,
    pub logits: Buf,
    pub kv_slots: Vec<KvShape>,
    pub delta_slots: Vec<usize>,
    pub conv_slots: Vec<(usize, usize)>,
    pub rope_theta: f32,
    /// The hidden state after the final norm, the input of a draft step.
    pub final_hidden: Buf,
    pub draft: Option<DraftProgram>,
}

/// The MTP head: predicts the token after next from a hidden state and a token.
#[derive(Clone, Debug, Serialize)]
pub struct DraftProgram {
    pub ops: Vec<Inst>,
    pub hidden_in: Buf,
    pub hidden_out: Buf,
    pub logits_from: usize,
    pub logits: Buf,
    pub kv_slot: usize,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct KvShape {
    pub head_dim: usize,
    pub heads: usize,
}

impl Program {
    /// Splits the work before the logits among `dies`, cutting between layers.
    pub fn split(&self, dies: usize) -> Result<Vec<std::ops::Range<usize>>, String> {
        let dies = dies.max(1);
        if dies > self.num_layers.max(1) {
            return Err(format!("{dies} dies para {} capas", self.num_layers));
        }
        let mut ranges = Vec::with_capacity(dies);
        let mut start = 0;
        for die in 1..dies {
            let boundary = (die * self.num_layers / dies) as u32;
            let end = self.ops.iter().position(|inst| inst.layer == Some(boundary)).ok_or("capa sin operaciones")?;
            ranges.push(start..end);
            start = end;
        }
        ranges.push(start..self.logits_from);
        Ok(ranges)
    }

    /// The one buffer that has to cross the cut before operation `cut`.
    pub fn crossing(&self, cut: usize) -> Result<Buf, String> {
        let mut last_writer = vec![None; self.buffers.len()];
        let mut crossing: Vec<Buf> = Vec::new();
        for (index, inst) in self.ops.iter().enumerate() {
            let (reads, writes) = inst.op.buffers();
            if index >= cut {
                for buffer in reads {
                    if last_writer[buffer].is_some_and(|w: usize| w < cut) && !crossing.contains(&buffer) {
                        crossing.push(buffer);
                    }
                }
            }
            for buffer in writes {
                last_writer[buffer] = Some(index);
            }
        }
        match crossing.as_slice() {
            [one] => Ok(*one),
            other => Err(format!("el corte en {cut} cruza {} vectores", other.len())),
        }
    }
}

struct Emit {
    program: Program,
    layer: Option<u32>,
}

impl Emit {
    fn buf(&mut self, len: usize) -> Buf {
        self.program.buffers.push(len);
        self.program.buffers.len() - 1
    }

    fn op(&mut self, name: &'static str, op: Op) {
        self.program.ops.push(Inst { op, layer: self.layer, name });
    }

    fn matvec(&mut self, name: &'static str, matrix: String, src: Buf, len: usize) -> Buf {
        let dst = self.buf(len);
        self.op(name, Op::Matvec { matrix, src, dst });
        dst
    }

    fn norm(&mut self, name: &'static str, src: Buf, weight: String, block: usize, len: usize) -> Buf {
        let dst = self.buf(len);
        self.op(name, Op::RmsNorm { src, weight, offset: 1.0, block, dst });
        dst
    }
}

fn dim(text: &Value, key: &str) -> Result<usize, String> {
    text.get(key).and_then(Value::as_u64).map(|v| v as usize).ok_or_else(|| format!("falta «{key}» en el config"))
}

struct Shape {
    hidden: usize,
    intermediate: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    rotary: usize,
    moe: Option<MoeShape>,
}

#[derive(Clone, Copy)]
struct MoeShape {
    experts: usize,
    top_k: usize,
    intermediate: usize,
    shared_intermediate: usize,
}

fn emit_attention(e: &mut Emit, s: &Shape, prefix: &str, normed: Buf) -> Buf {
    let p = |n: &str| format!("{prefix}.self_attn.{n}");
    let query_len = s.heads * s.head_dim;
    let kv_len = s.kv_heads * s.head_dim;
    let qg = e.matvec("Proyección Q", p("q_proj"), normed, query_len * 2);
    let k = e.matvec("Proyección K", p("k_proj"), normed, kv_len);
    let v = e.matvec("Proyección V", p("v_proj"), normed, kv_len);
    let query = e.buf(query_len);
    let gate = e.buf(query_len);
    e.op("Separar consulta y puerta", Op::SplitQueryGate { src: qg, query, gate, block: s.head_dim });
    let query = e.norm("Normalizar Q", query, p("q_norm"), s.head_dim, query_len);
    let key = e.norm("Normalizar K", k, p("k_norm"), s.head_dim, kv_len);
    e.op("RoPE en Q", Op::Rope { buf: query, block: s.head_dim, rotary: s.rotary });
    e.op("RoPE en K", Op::Rope { buf: key, block: s.head_dim, rotary: s.rotary });
    let slot = e.program.kv_slots.len();
    e.program.kv_slots.push(KvShape { head_dim: s.head_dim, heads: s.kv_heads });
    e.op("Guardar en caché KV", Op::KvAppend { slot, key, value: v });
    let attn = e.buf(query_len);
    e.op("Atención", Op::AttnSum { slot, query, head_dim: s.head_dim, heads: s.heads, kv_heads: s.kv_heads, dst: attn });
    let gate_on = e.buf(query_len);
    e.op("Sigmoide de puerta", Op::Sigmoid { src: gate, dst: gate_on });
    let gated = e.buf(query_len);
    e.op("Aplicar puerta", Op::MulElem { a: attn, b: gate_on, dst: gated });
    e.matvec("Proyección de salida", p("o_proj"), gated, s.hidden)
}

fn emit_feed_forward(e: &mut Emit, s: &Shape, prefix: &str, normed: Buf) -> Buf {
    if let Some(m) = s.moe {
        let dst = e.buf(s.hidden);
        e.op(
            "Mezcla de expertos",
            Op::Moe { prefix: format!("{prefix}.mlp"), src: normed, dst, experts: m.experts, top_k: m.top_k, intermediate: m.intermediate, shared_intermediate: m.shared_intermediate },
        );
        return dst;
    }
    let p = |n: &str| format!("{prefix}.mlp.{n}");
    let gate = e.matvec("Proyección gate", p("gate_proj"), normed, s.intermediate);
    let up = e.matvec("Proyección up", p("up_proj"), normed, s.intermediate);
    let silu = e.buf(s.intermediate);
    e.op("Activación SiLU", Op::Silu { src: gate, dst: silu });
    let product = e.buf(s.intermediate);
    e.op("Puerta × up", Op::MulElem { a: silu, b: up, dst: product });
    e.matvec("Proyección down", p("down_proj"), product, s.hidden)
}

fn emit_layer_tail(e: &mut Emit, s: &Shape, prefix: &str, x: Buf, mixed: Buf) -> Buf {
    let after_mixer = e.buf(s.hidden);
    e.op("Residual", Op::Add { a: x, b: mixed, dst: after_mixer });
    let normed = e.norm("Normalizar", after_mixer, format!("{prefix}.post_attention_layernorm"), s.hidden, s.hidden);
    let down = emit_feed_forward(e, s, prefix, normed);
    let next = e.buf(s.hidden);
    e.op("Residual", Op::Add { a: after_mixer, b: down, dst: next });
    next
}

/// Compiles a `Qwen3_5` text model, dense or with a mixture of experts. The program
/// follows `modeling_qwen3_5.py`, always taking the per-token path of Gated DeltaNet.
pub fn compile_qwen3_5(config: &str) -> Result<Program, String> {
    let config: Value = serde_json::from_str(config).map_err(|e| e.to_string())?;
    let text = config.get("text_config").unwrap_or(&config);
    let hidden = dim(text, "hidden_size")?;
    let vocab = dim(text, "vocab_size")?;
    let head_dim = dim(text, "head_dim")?;
    let key_heads = dim(text, "linear_num_key_heads")?;
    let key_dim = dim(text, "linear_key_head_dim")?;
    let value_heads = dim(text, "linear_num_value_heads")?;
    let value_dim = dim(text, "linear_value_head_dim")?;
    let kernel = dim(text, "linear_conv_kernel_dim")?;
    let rope_factor = text.get("partial_rotary_factor").and_then(Value::as_f64).unwrap_or(0.25);
    let rope_theta = text.get("rope_parameters").and_then(|r| r.get("rope_theta")).and_then(Value::as_f64).unwrap_or(1e7) as f32;
    let layer_types: Vec<String> = match text.get("layer_types").and_then(Value::as_array) {
        Some(list) => list.iter().filter_map(|v| v.as_str().map(String::from)).collect(),
        None => {
            let interval = dim(text, "full_attention_interval")?;
            (1..=dim(text, "num_hidden_layers")?).map(|i| if i % interval == 0 { "full_attention".to_string() } else { "linear_attention".to_string() }).collect()
        }
    };
    let moe = if text.get("num_experts").is_some() {
        Some(MoeShape {
            experts: dim(text, "num_experts")?,
            top_k: dim(text, "num_experts_per_tok")?,
            intermediate: dim(text, "moe_intermediate_size")?,
            shared_intermediate: dim(text, "shared_expert_intermediate_size")?,
        })
    } else {
        None
    };
    let shape = Shape {
        hidden,
        intermediate: if moe.is_some() { 0 } else { dim(text, "intermediate_size")? },
        heads: dim(text, "num_attention_heads")?,
        kv_heads: dim(text, "num_key_value_heads")?,
        head_dim,
        rotary: (head_dim as f64 * rope_factor) as usize,
        moe,
    };

    let mut e = Emit {
        program: Program {
            family: "qwen3_5".into(),
            vocab,
            hidden,
            num_layers: layer_types.len(),
            buffers: Vec::new(),
            ops: Vec::new(),
            logits_from: 0,
            logits: 0,
            kv_slots: Vec::new(),
            delta_slots: Vec::new(),
            conv_slots: Vec::new(),
            rope_theta,
            final_hidden: 0,
            draft: None,
        },
        layer: None,
    };

    let mut x = e.buf(hidden);
    e.op("Leer embedding", Op::EmbRead { table: "model.embed_tokens".into(), dst: x });

    for (i, kind) in layer_types.iter().enumerate() {
        e.layer = Some(i as u32);
        let prefix = format!("model.layers.{i}");
        let p = |s: &str| format!("{prefix}.{s}");

        let normed = e.norm("Normalizar entrada", x, p("input_layernorm"), hidden, hidden);
        let mixed = match kind.as_str() {
            "full_attention" => emit_attention(&mut e, &shape, &prefix, normed),
            "linear_attention" => {
                let key_len = key_heads * key_dim;
                let value_len = value_heads * value_dim;
                let conv_dim = 2 * key_len + value_len;
                let qkv = e.matvec("Proyección QKV", p("linear_attn.in_proj_qkv"), normed, conv_dim);
                let z = e.matvec("Proyección Z", p("linear_attn.in_proj_z"), normed, value_len);
                let b = e.matvec("Proyección β", p("linear_attn.in_proj_b"), normed, value_heads);
                let a = e.matvec("Proyección a", p("linear_attn.in_proj_a"), normed, value_heads);
                let slot = e.program.conv_slots.len();
                e.program.conv_slots.push((conv_dim, kernel));
                e.program.delta_slots.push(value_heads * key_dim * value_dim);
                let conv = e.buf(conv_dim);
                e.op("Convolución causal", Op::Dconv { slot, weight: p("linear_attn.conv1d"), src: qkv, dst: conv });
                let silu = e.buf(conv_dim);
                e.op("Activación SiLU", Op::Silu { src: conv, dst: silu });
                let out = e.buf(value_len);
                e.op(
                    "Regla delta",
                    Op::DeltaStep { slot, mixed: silu, a, b, a_log: p("linear_attn.A_log"), dt_bias: p("linear_attn.dt_bias"), key_dim, key_heads, value_dim, value_heads, dst: out },
                );
                let gated = e.buf(value_len);
                e.op("Norma con puerta", Op::RmsGated { src: out, z, weight: p("linear_attn.norm"), block: value_dim, dst: gated });
                e.matvec("Proyección de salida", p("linear_attn.out_proj"), gated, hidden)
            }
            other => return Err(format!("tipo de capa desconocido: {other:?}")),
        };
        x = emit_layer_tail(&mut e, &shape, &prefix, x, mixed);
    }

    e.layer = None;
    e.program.logits_from = e.program.ops.len();
    let normed = e.norm("Norma final", x, "model.norm".into(), hidden, hidden);
    e.program.final_hidden = normed;
    let logits = e.matvec("Salida de vocabulario", "lm_head".into(), normed, vocab);
    e.program.logits = logits;

    if text.get("mtp_num_hidden_layers").and_then(Value::as_u64).unwrap_or(0) > 0 {
        let main_ops = std::mem::take(&mut e.program.ops);
        let hidden_in = e.buf(hidden);
        let token = e.buf(hidden);
        e.op("Leer embedding", Op::EmbRead { table: "model.embed_tokens".into(), dst: token });
        let en = e.norm("Normalizar embedding", token, "mtp.pre_fc_norm_embedding".into(), hidden, hidden);
        let hn = e.norm("Normalizar estado", hidden_in, "mtp.pre_fc_norm_hidden".into(), hidden, hidden);
        let joined = e.buf(2 * hidden);
        e.op("Unir embedding y estado", Op::Concat { a: en, b: hn, dst: joined });
        let mixed_in = e.matvec("Proyección fc", "mtp.fc".into(), joined, hidden);
        let kv_slot = e.program.kv_slots.len();
        let prefix = "mtp.layers.0";
        let normed = e.norm("Normalizar entrada", mixed_in, format!("{prefix}.input_layernorm"), hidden, hidden);
        let attention = emit_attention(&mut e, &shape, prefix, normed);
        let output = emit_layer_tail(&mut e, &shape, prefix, mixed_in, attention);
        let logits_from = e.program.ops.len();
        let hidden_out = e.norm("Norma final", output, "mtp.norm".into(), hidden, hidden);
        let logits = e.matvec("Salida de vocabulario", "lm_head".into(), hidden_out, vocab);
        let ops = std::mem::replace(&mut e.program.ops, main_ops);
        e.program.draft = Some(DraftProgram { ops, hidden_in, hidden_out, logits_from, logits, kv_slot });
    }
    Ok(e.program)
}

#[cfg(test)]
mod tests {
    use super::*;
    use model_ir::{Block, ModelGraph, StageKind};

    const QWEN: &str = include_str!("../../../models/qwen3.8-27b.json");

    #[test]
    fn program_covers_exactly_the_stages_the_estimator_counts() {
        let program = compile_qwen3_5(QWEN).unwrap();
        let graph = ModelGraph::from_config_json("qwen", QWEN).unwrap();
        let expected: u64 = graph
            .stages
            .iter()
            .filter(|s| s.block != Block::Mtp)
            .map(|s| if let StageKind::Matvec { weights } = s.kind { weights } else { 0 })
            .sum();
        let in_program: u64 = program
            .ops
            .iter()
            .map(|inst| {
                if let Op::Matvec { src, dst, .. } = &inst.op {
                    (program.buffers[*src] * program.buffers[*dst]) as u64
                } else {
                    0
                }
            })
            .sum();
        assert_eq!(in_program, expected);
    }

    #[test]
    fn state_slots_match_the_model() {
        let program = compile_qwen3_5(QWEN).unwrap();
        assert_eq!(program.kv_slots.len(), 16 + 1, "las capas de atención y la cabeza MTP");
        assert_eq!(program.delta_slots.len(), 48);
        assert_eq!(program.delta_slots.iter().sum::<usize>(), 48 * 48 * 128 * 128);
        assert_eq!(program.buffers[program.logits], 248_320);
    }

    #[test]
    fn every_buffer_is_written_before_it_is_read() {
        let program = compile_qwen3_5(QWEN).unwrap();
        let mut written = vec![false; program.buffers.len()];
        for inst in &program.ops {
            let (reads, writes): (Vec<Buf>, Vec<Buf>) = match &inst.op {
                Op::EmbRead { dst, .. } => (vec![], vec![*dst]),
                Op::Matvec { src, dst, .. } | Op::Copy { src, dst } | Op::Silu { src, dst } | Op::Sigmoid { src, dst } | Op::RmsNorm { src, dst, .. } | Op::Dconv { src, dst, .. } => (vec![*src], vec![*dst]),
                Op::Add { a, b, dst } | Op::MulElem { a, b, dst } => (vec![*a, *b], vec![*dst]),
                Op::SplitQueryGate { src, query, gate, .. } => (vec![*src], vec![*query, *gate]),
                Op::Rope { buf, .. } => (vec![*buf], vec![*buf]),
                Op::KvAppend { key, value, .. } => (vec![*key, *value], vec![]),
                Op::AttnSum { query, dst, .. } => (vec![*query], vec![*dst]),
                Op::DeltaStep { mixed, a, b, dst, .. } => (vec![*mixed, *a, *b], vec![*dst]),
                Op::RmsGated { src, z, dst, .. } => (vec![*src, *z], vec![*dst]),
                Op::Concat { a, b, dst } => (vec![*a, *b], vec![*dst]),
                Op::Moe { src, dst, .. } => (vec![*src], vec![*dst]),
            };
            for r in reads {
                assert!(written[r], "{} lee un búfer sin escribir", inst.name);
            }
            for w in writes {
                written[w] = true;
            }
        }
    }
}
