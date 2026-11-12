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
    /// `x · Φ(x)`, with the normal distribution or, when `approximate`, its tanh fit.
    Gelu { src: Buf, approximate: bool, dst: Buf },
    /// Every element times a constant.
    Scale { src: Buf, factor: f32, dst: Buf },
    /// Adds a stored vector, such as a projection's bias.
    AddWeight { src: Buf, weight: String, dst: Buf },
    /// `x · rsqrt(mean(x²) + eps) · (offset + w)`, per block of `block` values.
    RmsNorm { src: Buf, weight: String, offset: f32, block: usize, dst: Buf },
    /// `q_proj` interleaves each head's query and gate.
    SplitQueryGate { src: Buf, query: Buf, gate: Buf, block: usize },
    Rope { buf: Buf, block: usize, rotary: usize, table: usize },
    KvAppend { slot: usize, key: Buf, value: Buf, window: Option<usize> },
    AttnSum { slot: usize, query: Buf, head_dim: usize, scale_dim: usize, heads: usize, kv_heads: usize, window: Option<usize>, softcap: Option<f32>, dst: Buf },
    /// `cap · tanh(x / cap)`, which keeps every value inside `±cap`.
    SoftCap { src: Buf, cap: f32, dst: Buf },
    Sigmoid { src: Buf, dst: Buf },
    Dconv { slot: usize, weight: String, src: Buf, dst: Buf },
    DeltaStep { slot: usize, mixed: Buf, a: Buf, b: Buf, a_log: String, dt_bias: String, key_dim: usize, key_heads: usize, value_dim: usize, value_heads: usize, dst: Buf },
    RmsGated { src: Buf, z: Buf, weight: String, block: usize, dst: Buf },
    Concat { a: Buf, b: Buf, dst: Buf },
    /// Copies the part of `src` starting at `from` that fits `dst`.
    Slice { src: Buf, from: usize, dst: Buf },
    /// DeepSeek's multi-head latent attention, one step: stores the token's latent and
    /// rotary key, then attends over them all for every head.
    Mla { slot: usize, query: Buf, latent: Buf, rope_key: Buf, embed: String, unembed: String, heads: usize, nope: usize, rope: usize, lora: usize, value_dim: usize, scale: f32, dst: Buf },
    /// DeepSeek's mixture: experts chosen by sigmoid scores plus their correction.
    SigmoidMoe { prefix: String, src: Buf, dst: Buf, experts: usize, top_k: usize, intermediate: usize, shared_intermediate: usize, scaling: f32, normalize: bool },
    /// Sparse mixture of experts: routes to `top_k` experts plus a shared expert.
    Moe { prefix: String, src: Buf, dst: Buf, experts: usize, top_k: usize, intermediate: usize, shared_intermediate: usize },
}

impl Op {
    /// The buffers this operation reads and the ones it writes.
    pub fn buffers(&self) -> (Vec<Buf>, Vec<Buf>) {
        match self {
            Op::EmbRead { dst, .. } => (vec![], vec![*dst]),
            Op::Matvec { src, dst, .. } | Op::Copy { src, dst } | Op::Silu { src, dst } | Op::Gelu { src, dst, .. } | Op::Scale { src, dst, .. } | Op::SoftCap { src, dst, .. } | Op::AddWeight { src, dst, .. } | Op::Sigmoid { src, dst } | Op::RmsNorm { src, dst, .. } | Op::Dconv { src, dst, .. } | Op::Moe { src, dst, .. } => (vec![*src], vec![*dst]),
            Op::Add { a, b, dst } | Op::MulElem { a, b, dst } | Op::Concat { a, b, dst } => (vec![*a, *b], vec![*dst]),
            Op::SplitQueryGate { src, query, gate, .. } => (vec![*src], vec![*query, *gate]),
            Op::Slice { src, dst, .. } => (vec![*src], vec![*dst]),
            Op::Mla { query, latent, rope_key, dst, .. } => (vec![*query, *latent, *rope_key], vec![*dst]),
            Op::SigmoidMoe { src, dst, .. } => (vec![*src], vec![*dst]),
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
    pub rope_scaling: Option<RopeScaling>,
    /// The base of a second RoPE table, for layers that see only recent tokens.
    pub local_rope_theta: Option<f32>,
    /// The small constant that keeps the RMS norms' square root defined.
    pub rms_epsilon: f64,
    /// The hidden state after the final norm, the input of a draft step.
    pub final_hidden: Buf,
    pub draft: Option<DraftProgram>,
    /// An image generator's heads, for models that draw: see `ImageHead`.
    pub image: Option<ImageHead>,
}

/// The two small heads with which a language model draws, as Janus-Pro does: one turns
/// the final hidden state into the scores of each code of the image's codebook, the other
/// turns a drawn code into the vector that goes in, in place of a token's embedding.
#[derive(Clone, Debug, Serialize)]
pub struct ImageHead {
    /// From `final_hidden` to `logits`, one score per code.
    pub logits_ops: Vec<Inst>,
    pub logits: Buf,
    pub codes: usize,
    /// From the code, taken as the token, to `input`, as long as the hidden state.
    pub input_ops: Vec<Inst>,
    pub input: Buf,
}

/// Llama 3's lengthening of the slow rotary waves, for contexts beyond its training.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct RopeScaling {
    pub factor: f64,
    pub low: f64,
    pub high: f64,
    pub trained_context: f64,
}

impl RopeScaling {
    pub fn frequency(&self, base: f64) -> f64 {
        let wavelength = std::f64::consts::TAU / base;
        let (low_wave, high_wave) = (self.trained_context / self.low, self.trained_context / self.high);
        if wavelength > low_wave {
            base / self.factor
        } else if wavelength > high_wave {
            let smooth = (self.trained_context / wavelength - self.low) / (self.high - self.low);
            base * ((1.0 - smooth) / self.factor + smooth)
        } else {
            base
        }
    }
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
    /// Rows of an embedding table: the vocabulary's, or the codebook's for an image head.
    pub fn table_rows(&self, table: &str) -> usize {
        match &self.image {
            Some(image) if image.input_ops.iter().any(|i| matches!(&i.op, Op::EmbRead { table: t, .. } if t == table)) => image.codes,
            _ => self.vocab,
        }
    }

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

    /// Splits the work before the logits among `dies`, cutting between layers so that the
    /// die with most `cost` has as little as can be. The first die also counts the cost
    /// of the logits, which it makes.
    pub fn split_by(&self, dies: usize, cost: impl Fn(usize) -> u64) -> Result<Vec<std::ops::Range<usize>>, String> {
        let dies = dies.max(1);
        let layers = self.num_layers.max(1);
        if dies > layers {
            return Err(format!("{dies} dies para {} capas", self.num_layers));
        }
        // Where each layer starts, and the cost of each: what comes before the first
        // layer and the logits go with the first.
        let mut starts = Vec::with_capacity(layers);
        for layer in 1..layers as u32 {
            starts.push(self.ops.iter().position(|inst| inst.layer == Some(layer)).ok_or("capa sin operaciones")?);
        }
        let bounds: Vec<usize> = std::iter::once(0).chain(starts).chain(std::iter::once(self.logits_from)).collect();
        let mut load: Vec<u64> = bounds.windows(2).map(|w| (w[0]..w[1]).map(&cost).sum()).collect();
        load[0] += (self.logits_from..self.ops.len()).map(&cost).sum::<u64>();
        // best[k][i]: for `i` layers on `k + 1` dies, the least largest load and then the
        // least sum of squared loads, so that the dies the largest does not bound are even
        // too; and where the last die starts.
        let prefix: Vec<u64> = std::iter::once(0).chain(load.iter().scan(0, |sum, l| {
            *sum += l;
            Some(*sum)
        })).collect();
        let square = |load: u64| (load as f64).powi(2);
        let mut best = vec![vec![((u64::MAX, f64::INFINITY), 0usize); layers + 1]; dies];
        for i in 1..=layers {
            best[0][i] = ((prefix[i], square(prefix[i])), 0);
        }
        for k in 1..dies {
            for i in k + 1..=layers {
                best[k][i] = (k..i)
                    .map(|j| {
                        let ((largest, squares), _) = best[k - 1][j];
                        let here = prefix[i] - prefix[j];
                        ((largest.max(here), squares + square(here)), j)
                    })
                    .min_by(|a, b| a.0 .0.cmp(&b.0 .0).then(a.0 .1.total_cmp(&b.0 .1)))
                    .unwrap();
            }
        }
        let mut cuts = vec![layers];
        for k in (1..dies).rev() {
            cuts.push(best[k][*cuts.last().unwrap()].1);
        }
        cuts.push(0);
        cuts.reverse();
        Ok(cuts.windows(2).map(|w| bounds[w[0]]..bounds[w[1]]).collect())
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
    /// What is added to each stored norm weight before it multiplies.
    norm_offset: f32,
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
        self.op(name, Op::RmsNorm { src, weight, offset: self.norm_offset, block, dst });
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
    e.op("RoPE en Q", Op::Rope { buf: query, block: s.head_dim, rotary: s.rotary, table: 0 });
    e.op("RoPE en K", Op::Rope { buf: key, block: s.head_dim, rotary: s.rotary, table: 0 });
    let slot = e.program.kv_slots.len();
    e.program.kv_slots.push(KvShape { head_dim: s.head_dim, heads: s.kv_heads });
    e.op("Guardar en caché KV", Op::KvAppend { slot, key, value: v, window: None });
    let attn = e.buf(query_len);
    e.op("Atención", Op::AttnSum { slot, query, head_dim: s.head_dim, scale_dim: s.head_dim, heads: s.heads, kv_heads: s.kv_heads, window: None, softcap: None, dst: attn });
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

/// The checkpoints `compile_model` can turn into a program, by `model_type`.
pub const SIMULATED_FAMILIES: [&str; 13] = ["qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text", "llama", "mistral", "qwen2", "qwen3", "gemma", "gemma2", "gemma3_text", "deepseek_v3", "janus"];

/// Compiles any supported checkpoint from its `config.json`.
pub fn compile_model(config: &str) -> Result<Program, String> {
    let value: Value = serde_json::from_str(config).map_err(|e| e.to_string())?;
    if let Some(method) = value.get("quantization_config").and_then(|q| q.get("quant_method")).and_then(Value::as_str) {
        return Err(format!("sus pesos están cuantizados con «{method}», un formato que no se lee todavía"));
    }
    match value.get("model_type").and_then(Value::as_str) {
        Some("qwen3_5" | "qwen3_5_text" | "qwen3_5_moe" | "qwen3_5_moe_text") => compile_qwen3_5(config),
        Some("llama" | "mistral" | "qwen2" | "qwen3" | "gemma" | "gemma2" | "gemma3_text") => compile_llama(&value),
        Some("deepseek_v3") => compile_deepseek(&value),
        // Janus writes text and images with the Llama in its text configuration, and draws
        // with two small heads around it.
        Some("janus") => {
            let mut program = compile_model(&value["text_config"].to_string())?;
            if let Some(vq) = value.get("vq_config") {
                add_janus_image_head(&mut program, vq)?;
            }
            Ok(program)
        }
        Some(other) => Err(format!("no hay compilador para la arquitectura «{other}»")),
        // llama2.c describes its models with its own names: they are Llamas.
        None if value.get("dim").is_some() && value.get("n_layers").is_some() => compile_llama(&llama2c_as_llama(&value)?),
        None => Err("el config no indica model_type".into()),
    }
}

/// A llama2.c configuration (`dim`, `n_layers`, …) as the Llama configuration it stands for,
/// with the hidden size of its MLP worked out as llama2.c does.
fn llama2c_as_llama(config: &Value) -> Result<Value, String> {
    let number = |key: &str| config.get(key).and_then(Value::as_u64).ok_or_else(|| format!("al config de llama2.c le falta «{key}»"));
    let (dim, heads, multiple_of) = (number("dim")?, number("n_heads")?, number("multiple_of").unwrap_or(32));
    let hidden = config.get("hidden_dim").and_then(Value::as_u64).unwrap_or_else(|| multiple_of * (2 * (4 * dim) / 3).div_ceil(multiple_of));
    Ok(serde_json::json!({
        "model_type": "llama",
        "hidden_size": dim,
        "intermediate_size": hidden,
        "num_hidden_layers": number("n_layers")?,
        "num_attention_heads": heads,
        "num_key_value_heads": config.get("n_kv_heads").and_then(Value::as_u64).unwrap_or(heads),
        "vocab_size": number("vocab_size")?,
        "max_position_embeddings": config.get("max_seq_len").and_then(Value::as_u64).unwrap_or(2048),
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "hidden_act": "silu",
        "tie_word_embeddings": true,
    }))
}

/// How a layer of the plain family differs from the others.
struct PlainLayer {
    head_norms: bool,
    attention_bias: bool,
    /// Qwen2 has biases on its inputs but none on the output projection.
    output_bias: bool,
    scale_dim: usize,
    rope_table: usize,
    window: Option<usize>,
    softcap: Option<f32>,
}

fn with_bias(e: &mut Emit, name: &'static str, matrix: String, src: Buf, len: usize, bias: bool) -> Buf {
    let out = e.matvec(name, matrix.clone(), src, len);
    if !bias {
        return out;
    }
    let dst = e.buf(len);
    e.op("Sumar sesgo", Op::AddWeight { src: out, weight: format!("{matrix}.bias"), dst });
    dst
}

fn plain_attention(e: &mut Emit, s: &Shape, prefix: &str, normed: Buf, layer: &PlainLayer) -> Buf {
    let p = |n: &str| format!("{prefix}.self_attn.{n}");
    let query_len = s.heads * s.head_dim;
    let kv_len = s.kv_heads * s.head_dim;
    let mut query = with_bias(e, "Proyección Q", p("q_proj"), normed, query_len, layer.attention_bias);
    let mut key = with_bias(e, "Proyección K", p("k_proj"), normed, kv_len, layer.attention_bias);
    let value = with_bias(e, "Proyección V", p("v_proj"), normed, kv_len, layer.attention_bias);
    if layer.head_norms {
        query = e.norm("Normalizar Q", query, p("q_norm"), s.head_dim, query_len);
        key = e.norm("Normalizar K", key, p("k_norm"), s.head_dim, kv_len);
    }
    e.op("RoPE en Q", Op::Rope { buf: query, block: s.head_dim, rotary: s.rotary, table: layer.rope_table });
    e.op("RoPE en K", Op::Rope { buf: key, block: s.head_dim, rotary: s.rotary, table: layer.rope_table });
    let slot = e.program.kv_slots.len();
    e.program.kv_slots.push(KvShape { head_dim: s.head_dim, heads: s.kv_heads });
    e.op("Guardar en caché KV", Op::KvAppend { slot, key, value, window: layer.window });
    let attn = e.buf(query_len);
    e.op("Atención", Op::AttnSum { slot, query, head_dim: s.head_dim, scale_dim: layer.scale_dim, heads: s.heads, kv_heads: s.kv_heads, window: layer.window, softcap: layer.softcap, dst: attn });
    with_bias(e, "Proyección de salida", p("o_proj"), attn, s.hidden, layer.output_bias)
}

/// The gated feed-forward of the plain family, with its activation and optional bias.
fn plain_mlp(e: &mut Emit, s: &Shape, prefix: &str, normed: Buf, gelu: Option<bool>, bias: bool) -> Buf {
    let p = |n: &str| format!("{prefix}.mlp.{n}");
    let gate = with_bias(e, "Proyección gate", p("gate_proj"), normed, s.intermediate, bias);
    let up = with_bias(e, "Proyección up", p("up_proj"), normed, s.intermediate, bias);
    let active = e.buf(s.intermediate);
    match gelu {
        Some(approximate) => e.op("Activación GELU", Op::Gelu { src: gate, approximate, dst: active }),
        None => e.op("Activación SiLU", Op::Silu { src: gate, dst: active }),
    }
    let product = e.buf(s.intermediate);
    e.op("Puerta × up", Op::MulElem { a: active, b: up, dst: product });
    with_bias(e, "Proyección down", p("down_proj"), product, s.hidden, bias)
}

/// Compiles the architectures made of plain attention and a gated MLP. They differ
/// from each other only in what a config states, and what they cannot do is refused
/// here so that a checkpoint never runs with a different meaning than it was trained.
fn compile_llama(config: &Value) -> Result<Program, String> {
    let family = config["model_type"].as_str().unwrap_or("llama");
    let gemma = family.starts_with("gemma");
    let four_norms = matches!(family, "gemma2" | "gemma3_text");
    let hidden = dim(config, "hidden_size")?;
    let heads = dim(config, "num_attention_heads")?;
    let kv_heads = config.get("num_key_value_heads").and_then(Value::as_u64).map_or(heads, |v| v as usize);
    if heads == 0 || kv_heads == 0 || heads % kv_heads != 0 {
        return Err(format!("{heads} cabezas no se reparten en {kv_heads} grupos KV"));
    }
    let head_dim = config.get("head_dim").and_then(Value::as_u64).map_or(hidden / heads, |v| v as usize);
    if head_dim == 0 || (config.get("head_dim").is_none() && hidden % heads != 0) {
        return Err(format!("{hidden} no se reparte entre {heads} cabezas"));
    }
    let on = |key: &str, default: bool| config.get(key).and_then(Value::as_bool).unwrap_or(default);
    let attention_bias = on("attention_bias", family == "qwen2");
    let output_bias = attention_bias && family != "qwen2";
    let mlp_bias = on("mlp_bias", false);
    let sliding = family == "gemma3_text";
    let layer_kinds = config.get("layer_types").and_then(Value::as_array);
    if !sliding && (on("use_sliding_window", false) || layer_kinds.is_some_and(|l| l.iter().any(|t| t != "full_attention"))) {
        return Err("usa capas con ventana deslizante y el compilador todavía no las genera".into());
    }
    let activation = if gemma { config.get("hidden_activation").or_else(|| config.get("hidden_act")) } else { config.get("hidden_act") }.and_then(Value::as_str);
    let gelu = match (gemma, activation) {
        (false, None | Some("silu" | "swish")) => None,
        (true, None) => Some(family != "gemma"),
        (true, Some("gelu")) => Some(false),
        (true, Some("gelu_pytorch_tanh")) => Some(true),
        (_, Some(other)) => return Err(format!("usa la activación «{other}», que el chip no tiene")),
    };
    let rope = config.get("rope_parameters").or_else(|| config.get("rope_scaling"));
    let rope_scaling = match rope.and_then(|r| r.get("rope_type").or_else(|| r.get("type"))).and_then(Value::as_str) {
        None | Some("default") => None,
        Some("llama3") => {
            let rope = rope.unwrap_or(&Value::Null);
            let number = |key: &str, default: f64| rope.get(key).and_then(Value::as_f64).unwrap_or(default);
            Some(RopeScaling { factor: number("factor", 1.0), low: number("low_freq_factor", 1.0), high: number("high_freq_factor", 4.0), trained_context: number("original_max_position_embeddings", 8192.0) })
        }
        Some(kind) => return Err(format!("usa un RoPE escalado («{kind}») que el compilador no genera")),
    };
    let default_theta = if sliding { 1_000_000.0 } else { 10_000.0 };
    let rope_theta = config.get("rope_theta").or_else(|| rope.and_then(|r| r.get("rope_theta"))).and_then(Value::as_f64).unwrap_or(default_theta) as f32;
    let vocab = dim(config, "vocab_size")?;
    let layers = dim(config, "num_hidden_layers")?;
    let shape = Shape { hidden, intermediate: dim(config, "intermediate_size")?, heads, kv_heads, head_dim, rotary: head_dim, moe: None };
    let window = config.get("sliding_window").and_then(Value::as_u64).unwrap_or(512) as usize;
    let pattern = config.get("sliding_window_pattern").or_else(|| config.get("_sliding_window_pattern")).and_then(Value::as_u64).unwrap_or(6) as usize;
    let scale_dim = if four_norms { config.get("query_pre_attn_scalar").and_then(Value::as_f64).unwrap_or(head_dim as f64) as usize } else { head_dim };

    // Gemma 2 caps its scores unless its config says null; nothing else does.
    let softcap = |key: &str, default: f64| -> Option<f32> {
        match config.get(key) {
            Some(Value::Null) => None,
            Some(value) => value.as_f64().map(|v| v as f32),
            None => (family == "gemma2").then_some(default as f32),
        }
    };
    let mut e = Emit { program: empty_program(family, vocab, hidden, layers, rope_theta), layer: None, norm_offset: if gemma { 1.0 } else { 0.0 } };
    e.program.rope_scaling = rope_scaling;
    if sliding {
        e.program.local_rope_theta = Some(config.get("rope_local_base_freq").and_then(Value::as_f64).unwrap_or(10_000.0) as f32);
    }
    e.program.rms_epsilon = config.get("rms_norm_eps").and_then(Value::as_f64).unwrap_or(1e-6);
    let mut x = e.buf(hidden);
    e.op("Leer embedding", Op::EmbRead { table: "model.embed_tokens".into(), dst: x });
    if gemma {
        let scaled = e.buf(hidden);
        e.op("Escalar embedding", Op::Scale { src: x, factor: (hidden as f32).sqrt(), dst: scaled });
        x = scaled;
    }
    for i in 0..layers {
        e.layer = Some(i as u32);
        let prefix = format!("model.layers.{i}");
        let local = sliding && match layer_kinds {
            Some(kinds) => kinds.get(i).is_some_and(|k| k == "sliding_attention"),
            None => !(i + 1).is_multiple_of(pattern),
        };
        let layer = PlainLayer {
            head_norms: family == "qwen3" || family == "gemma3_text",
            attention_bias,
            output_bias,
            scale_dim,
            rope_table: local as usize,
            window: local.then_some(window),
            softcap: softcap("attn_logit_softcapping", 50.0),
        };
        let normed = e.norm("Normalizar entrada", x, format!("{prefix}.input_layernorm"), hidden, hidden);
        let mixed = plain_attention(&mut e, &shape, &prefix, normed, &layer);
        if four_norms {
            let after_attention = e.norm("Normalizar atención", mixed, format!("{prefix}.post_attention_layernorm"), hidden, hidden);
            let h = e.buf(hidden);
            e.op("Residual", Op::Add { a: x, b: after_attention, dst: h });
            let before_mlp = e.norm("Normalizar antes de la MLP", h, format!("{prefix}.pre_feedforward_layernorm"), hidden, hidden);
            let down = plain_mlp(&mut e, &shape, &prefix, before_mlp, gelu, mlp_bias);
            let after_mlp = e.norm("Normalizar después de la MLP", down, format!("{prefix}.post_feedforward_layernorm"), hidden, hidden);
            let next = e.buf(hidden);
            e.op("Residual", Op::Add { a: h, b: after_mlp, dst: next });
            x = next;
        } else {
            let after_mixer = e.buf(hidden);
            e.op("Residual", Op::Add { a: x, b: mixed, dst: after_mixer });
            let before_mlp = e.norm("Normalizar", after_mixer, format!("{prefix}.post_attention_layernorm"), hidden, hidden);
            let down = plain_mlp(&mut e, &shape, &prefix, before_mlp, gelu, mlp_bias);
            let next = e.buf(hidden);
            e.op("Residual", Op::Add { a: after_mixer, b: down, dst: next });
            x = next;
        }
    }
    e.layer = None;
    e.program.logits_from = e.program.ops.len();
    let normed = e.norm("Norma final", x, "model.norm".into(), hidden, hidden);
    e.program.final_hidden = normed;
    // With tied embeddings there is one table in memory, read both ways.
    let head = if on("tie_word_embeddings", gemma) { "model.embed_tokens" } else { "lm_head" };
    let mut logits = e.matvec("Salida de vocabulario", head.into(), normed, vocab);
    if let Some(cap) = softcap("final_logit_softcapping", 30.0) {
        let capped = e.buf(vocab);
        e.op("Limitar logits", Op::SoftCap { src: logits, cap, dst: capped });
        logits = capped;
    }
    e.program.logits = logits;
    Ok(e.program)
}

/// Compiles DeepSeek V3 and R1, following their reference code.
fn compile_deepseek(config: &Value) -> Result<Program, String> {
    let (hidden, vocab, layers, heads) = (dim(config, "hidden_size")?, dim(config, "vocab_size")?, dim(config, "num_hidden_layers")?, dim(config, "num_attention_heads")?);
    let (lora, rope, nope, value_dim) = (dim(config, "kv_lora_rank")?, dim(config, "qk_rope_head_dim")?, dim(config, "qk_nope_head_dim")?, dim(config, "v_head_dim")?);
    let q_rank = config.get("q_lora_rank").and_then(Value::as_u64).map(|v| v as usize);
    if config.get("rope_scaling").is_some_and(|r| !r.is_null()) {
        return Err("usa un RoPE escalado que el compilador no genera en esta arquitectura".into());
    }
    let groups = config.get("n_group").and_then(Value::as_u64).unwrap_or(1);
    if groups > 1 && config.get("topk_group").and_then(Value::as_u64).unwrap_or(1) != groups {
        return Err("elige expertos por grupos y el compilador todavía no lo genera".into());
    }
    let wanted = |key: &str, value: &str| config.get(key).and_then(Value::as_str).is_none_or(|v| v == value);
    if !wanted("topk_method", "noaux_tc") || !wanted("scoring_func", "sigmoid") {
        return Err("usa un método de selección de expertos que el compilador no genera".into());
    }
    let bias = config.get("attention_bias").and_then(Value::as_bool).unwrap_or(false);
    let q_head = nope + rope;
    let intermediate = dim(config, "intermediate_size")?;
    let routed = config.get("n_routed_experts").and_then(Value::as_u64).map(|v| v as usize);
    let first_dense = config.get("first_k_dense_replace").and_then(Value::as_u64).unwrap_or(0) as usize;
    let frequency = config.get("moe_layer_freq").and_then(Value::as_u64).unwrap_or(1).max(1) as usize;
    let rope_theta = config.get("rope_theta").and_then(Value::as_f64).unwrap_or(10_000.0) as f32;
    let shape = Shape { hidden, intermediate, heads, kv_heads: heads, head_dim: q_head, rotary: rope, moe: None };

    let mut e = Emit { program: empty_program("deepseek_v3", vocab, hidden, layers, rope_theta), layer: None, norm_offset: 0.0 };
    e.program.rms_epsilon = config.get("rms_norm_eps").and_then(Value::as_f64).unwrap_or(1e-6);
    let mut x = e.buf(hidden);
    e.op("Leer embedding", Op::EmbRead { table: "model.embed_tokens".into(), dst: x });
    for i in 0..layers {
        e.layer = Some(i as u32);
        let prefix = format!("model.layers.{i}");
        let p = |s: &str| format!("{prefix}.self_attn.{s}");
        let normed = e.norm("Normalizar entrada", x, format!("{prefix}.input_layernorm"), hidden, hidden);
        let query = match q_rank {
            Some(rank) => {
                let compact = with_bias(&mut e, "Proyección Q latente", p("q_a_proj"), normed, rank, bias);
                let compact = e.norm("Normalizar Q latente", compact, p("q_a_layernorm"), rank, rank);
                e.matvec("Expandir Q", p("q_b_proj"), compact, heads * q_head)
            }
            None => e.matvec("Proyección Q", p("q_proj"), normed, heads * q_head),
        };
        let compressed = with_bias(&mut e, "Proyección KV latente", p("kv_a_proj_with_mqa"), normed, lora + rope, bias);
        let raw_latent = e.buf(lora);
        e.op("Separar latente", Op::Slice { src: compressed, from: 0, dst: raw_latent });
        let rope_key = e.buf(rope);
        e.op("Separar clave rotada", Op::Slice { src: compressed, from: lora, dst: rope_key });
        let latent = e.norm("Normalizar latente", raw_latent, p("kv_a_layernorm"), lora, lora);
        let slot = e.program.kv_slots.len();
        e.program.kv_slots.push(KvShape { head_dim: lora + rope, heads: 1 });
        let attended = e.buf(heads * value_dim);
        e.op("Atención latente", Op::Mla { slot, query, latent, rope_key, embed: p("embed_q"), unembed: p("unembed_out"), heads, nope, rope, lora, value_dim, scale: (q_head as f32).powf(-0.5), dst: attended });
        let mixed = with_bias(&mut e, "Proyección de salida", p("o_proj"), attended, hidden, bias);
        let after_attention = e.buf(hidden);
        e.op("Residual", Op::Add { a: x, b: mixed, dst: after_attention });
        let before_mlp = e.norm("Normalizar", after_attention, format!("{prefix}.post_attention_layernorm"), hidden, hidden);
        let is_moe = i >= first_dense && i.is_multiple_of(frequency);
        let feed = match routed.filter(|_| is_moe) {
            Some(experts) => {
                let dst = e.buf(hidden);
                let moe_size = dim(config, "moe_intermediate_size")?;
                let shared = config.get("n_shared_experts").and_then(Value::as_u64).unwrap_or(0) as usize;
                e.op("Mezcla de expertos", Op::SigmoidMoe {
                    prefix: format!("{prefix}.mlp"),
                    src: before_mlp,
                    dst,
                    experts,
                    top_k: config.get("num_experts_per_tok").and_then(Value::as_u64).unwrap_or(1) as usize,
                    intermediate: moe_size,
                    shared_intermediate: moe_size * shared,
                    scaling: config.get("routed_scaling_factor").and_then(Value::as_f64).unwrap_or(1.0) as f32,
                    normalize: config.get("norm_topk_prob").and_then(Value::as_bool).unwrap_or(true),
                });
                dst
            }
            None => plain_mlp(&mut e, &shape, &prefix, before_mlp, None, false),
        };
        let next = e.buf(hidden);
        e.op("Residual", Op::Add { a: after_attention, b: feed, dst: next });
        x = next;
    }
    e.layer = None;
    e.program.logits_from = e.program.ops.len();
    let normed = e.norm("Norma final", x, "model.norm".into(), hidden, hidden);
    e.program.final_hidden = normed;
    e.program.logits = e.matvec("Salida de vocabulario", "lm_head".into(), normed, vocab);
    Ok(e.program)
}

fn empty_program(family: &str, vocab: usize, hidden: usize, num_layers: usize, rope_theta: f32) -> Program {
    Program {
        family: family.into(),
        vocab,
        hidden,
        num_layers,
        buffers: Vec::new(),
        ops: Vec::new(),
        logits_from: 0,
        logits: 0,
        kv_slots: Vec::new(),
        delta_slots: Vec::new(),
        conv_slots: Vec::new(),
        rope_theta,
        rope_scaling: None,
        local_rope_theta: None,
        rms_epsilon: 1e-6,
        final_hidden: 0,
        draft: None,
        image: None,
    }
}

/// Adds Janus-Pro's image heads to its language model's program, from the checkpoint's
/// `vq_config`: the generation head (`proj_out`, GELU, `vision_head`) and the aligner
/// (`generation_embeddings`, `fc1`, GELU, `hidden_layers.0`), each layer with its bias.
fn add_janus_image_head(program: &mut Program, vq: &Value) -> Result<(), String> {
    let codes = dim(vq, "num_embeddings")?;
    let code_dim = dim(vq, "embed_dim")?;
    let width = dim(vq, "image_token_embed_dim")?;
    let hidden = program.hidden;
    let buffer = |program: &mut Program, len: usize| {
        program.buffers.push(len);
        program.buffers.len() - 1
    };
    let inst = |name: &'static str, op: Op| Inst { op, layer: None, name };
    // A product and its bias, into a buffer of its own.
    let layer = |program: &mut Program, ops: &mut Vec<Inst>, matrix: &str, src: Buf, len: usize| -> Buf {
        let product = buffer(program, len);
        ops.push(inst("Producto de la cabeza de imagen", Op::Matvec { matrix: matrix.into(), src, dst: product }));
        let biased = buffer(program, len);
        ops.push(inst("Sumar sesgo", Op::AddWeight { src: product, weight: format!("{matrix}.bias"), dst: biased }));
        biased
    };
    let mut logits_ops = Vec::new();
    let projected = layer(program, &mut logits_ops, "model.generation_head.proj_out", program.final_hidden, hidden);
    let active = buffer(program, hidden);
    logits_ops.push(inst("Activación GELU", Op::Gelu { src: projected, approximate: false, dst: active }));
    let logits = layer(program, &mut logits_ops, "model.generation_head.vision_head", active, codes);

    let mut input_ops = Vec::new();
    let code = buffer(program, code_dim);
    input_ops.push(inst("Vector del código", Op::EmbRead { table: "model.generation_embeddings".into(), dst: code }));
    let widened = layer(program, &mut input_ops, "model.generation_aligner.fc1", code, width);
    let active = buffer(program, width);
    input_ops.push(inst("Activación GELU", Op::Gelu { src: widened, approximate: false, dst: active }));
    let input = layer(program, &mut input_ops, "model.generation_aligner.hidden_layers.0", active, hidden);
    program.image = Some(ImageHead { logits_ops, logits, codes, input_ops, input });
    Ok(())
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

    let mut e = Emit { program: empty_program("qwen3_5", vocab, hidden, layer_types.len(), rope_theta), layer: None, norm_offset: 1.0 };

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
    fn every_model_of_the_catalogue_compiles_including_llama2c_ones() {
        for config in [include_str!("../../../models/stories110m.json"), include_str!("../../../models/stories15m.json"), include_str!("../../../models/stories260k.json"), include_str!("../../../models/llama-3.1-8b.json")] {
            let program = compile_model(config).unwrap_or_else(|e| panic!("{e}: {config}"));
            let graph = ModelGraph::from_config_json("modelo", config).unwrap();
            assert_eq!(program.hidden as u64, graph.hidden_size, "{config}");
        }
    }

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
    fn plain_models_cost_what_the_estimator_counts() {
        let config = include_str!("../../../models/llama-3.1-8b.json");
        let program = compile_model(config).unwrap();
        let graph = ModelGraph::from_config_json("llama", config).unwrap();
        let expected: u64 = graph.stages.iter().map(|s| if let StageKind::Matvec { weights } = s.kind { weights } else { 0 }).sum();
        let found: u64 = program.ops.iter().map(|i| if let Op::Matvec { src, dst, .. } = &i.op { (program.buffers[*src] * program.buffers[*dst]) as u64 } else { 0 }).sum();
        assert_eq!(found, expected);
        assert_eq!(program.kv_slots.len(), graph.num_layers as usize);
    }

    #[test]
    fn what_a_checkpoint_needs_and_the_chip_lacks_is_refused() {
        let base = r#""model_type":"llama","hidden_size":16,"num_attention_heads":2,"intermediate_size":32,"vocab_size":8,"num_hidden_layers":1"#;
        assert!(compile_model(&format!("{{{base}}}")).is_ok());
        for (extra, word) in [(r#""hidden_act":"gelu""#, "gelu"), (r#""rope_scaling":{"rope_type":"yarn","factor":8.0}"#, "yarn"), (r#""quantization_config":{"quant_method":"gptq"}"#, "gptq")] {
            let error = compile_model(&format!("{{{base},{extra}}}")).unwrap_err();
            assert!(error.contains(word), "{error}");
        }
        assert!(compile_model(r#"{"model_type":"bert"}"#).unwrap_err().contains("bert"));
    }

    #[test]
    fn a_split_by_cost_gives_the_first_die_fewer_layers_when_it_also_makes_the_logits() {
        let program = compile_qwen3_5(QWEN).unwrap();
        let matrices = |i: usize| match &program.ops[i].op {
            Op::Matvec { src, dst, .. } => (program.buffers[*src] * program.buffers[*dst]) as u64,
            _ => 0,
        };
        let even = program.split(8).unwrap();
        let balanced = program.split_by(8, matrices).unwrap();
        assert_eq!(balanced.len(), 8);
        assert_eq!((balanced[0].start, balanced[7].end), (even[0].start, even[7].end));
        assert!(balanced.windows(2).all(|w| w[0].end == w[1].start && !w[0].is_empty()));
        let layers = |r: &std::ops::Range<usize>| program.ops[r.clone()].iter().filter_map(|i| i.layer).collect::<std::collections::HashSet<_>>().len();
        assert!(layers(&balanced[0]) < layers(&even[0]), "{balanced:?}");
        // No other cut leaves less on the die with most.
        let load = |plan: &[std::ops::Range<usize>]| plan.iter().enumerate().map(|(d, r)| r.clone().map(matrices).sum::<u64>() + if d == 0 { (program.logits_from..program.ops.len()).map(matrices).sum() } else { 0 }).max().unwrap();
        assert!(load(&balanced) < load(&even));
        assert_eq!(program.split_by(1, matrices).unwrap(), program.split(1).unwrap());
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
                Op::Matvec { src, dst, .. } | Op::Copy { src, dst } | Op::Silu { src, dst } | Op::Gelu { src, dst, .. } | Op::Scale { src, dst, .. } | Op::SoftCap { src, dst, .. } | Op::AddWeight { src, dst, .. } | Op::Sigmoid { src, dst } | Op::RmsNorm { src, dst, .. } | Op::Dconv { src, dst, .. } => (vec![*src], vec![*dst]),
                Op::Add { a, b, dst } | Op::MulElem { a, b, dst } => (vec![*a, *b], vec![*dst]),
                Op::SplitQueryGate { src, query, gate, .. } => (vec![*src], vec![*query, *gate]),
                Op::Slice { src, dst, .. } | Op::SigmoidMoe { src, dst, .. } => (vec![*src], vec![*dst]),
                Op::Mla { query, latent, rope_key, dst, .. } => (vec![*query, *latent, *rope_key], vec![*dst]),
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
