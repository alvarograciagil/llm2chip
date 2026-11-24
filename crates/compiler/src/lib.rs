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
    /// `(x − mean(x)) · rsqrt(var(x) + eps) · w + b`, per block of `block` values, with
    /// `{weight}.weight` and `{weight}.bias`.
    LayerNorm { src: Buf, weight: String, block: usize, dst: Buf },
    /// Row `token` of a store of rows as long as `dst`: an image's patch, kept between the
    /// routines that go through the encoder block by block.
    LoadRow { store: Buf, dst: Buf },
    /// `src` into row `token` of a store.
    StoreRow { src: Buf, store: Buf },
    /// The rows `token + offset` of a store, one after another: the neighbours of a pixel
    /// in a map kept with a border of zeros, for a convolution.
    LoadRows { store: Buf, dst: Buf, offsets: Vec<usize> },
    /// GroupNorm with statistics gathered over many rows: `sums` and `squares` hold, for
    /// each of `groups` groups of channels, the sum of its values and of their squares over
    /// `count` values. `(x − mean) · rsqrt(var + eps) · w + b`, `w` and `b` per channel.
    GroupNorm { src: Buf, sums: Buf, squares: Buf, count: usize, weight: String, groups: usize, dst: Buf },
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
            Op::LayerNorm { src, dst, .. } => (vec![*src], vec![*dst]),
            Op::LoadRow { store, dst } | Op::LoadRows { store, dst, .. } => (vec![*store], vec![*dst]),
            // A store keeps its other rows: writing one also reads it.
            Op::StoreRow { src, store } => (vec![*src, *store], vec![*store]),
            Op::GroupNorm { src, sums, squares, dst, .. } => (vec![*src, *sums, *squares], vec![*dst]),
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
    /// What the model has besides its token routine: a second codebook (`CodeHead`) or
    /// routines over a store of rows (`RowPart`). At most one of each: see `codes` and
    /// `rows`.
    pub parts: Vec<Part>,
    /// Rows of the embedding table when they are not the vocabulary's: Show-o reads text,
    /// its markers and the image's codes from one table, and scores only the codes.
    pub embedding_rows: Option<usize>,
}

/// A part of a program besides its token routine.
#[derive(Clone, Debug, Serialize)]
#[serde(tag = "part", rename_all = "snake_case")]
pub enum Part {
    Codes(CodeHead),
    Rows(RowPart),
}

/// A second codebook, with two small heads around the model: one turns the final hidden
/// state into the scores of each code, the other turns a chosen code into the vector that
/// goes in, in place of a token's embedding. Janus-Pro draws with the codes of an image.
#[derive(Clone, Debug, Serialize)]
pub struct CodeHead {
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
    /// The second codebook's heads, if the program has them.
    pub fn codes(&self) -> Option<&CodeHead> {
        self.parts.iter().find_map(|part| if let Part::Codes(codes) = part { Some(codes) } else { None })
    }

    /// The routines over a store of rows, if the program has them.
    pub fn rows(&self) -> Option<&RowPart> {
        self.parts.iter().find_map(|part| if let Part::Rows(rows) = part { Some(rows) } else { None })
    }

    /// Adds a part, in place of one of the same kind.
    pub fn set_part(&mut self, part: Part) {
        self.parts.retain(|p| std::mem::discriminant(p) != std::mem::discriminant(&part));
        self.parts.push(part);
    }

    /// Rows of an embedding table: the vocabulary's, or the second codebook's.
    pub fn table_rows(&self, table: &str) -> usize {
        match self.codes() {
            Some(codes) if codes.input_ops.iter().any(|i| matches!(&i.op, Op::EmbRead { table: t, .. } if t == table)) => codes.codes,
            _ => self.embedding_rows.unwrap_or(self.vocab),
        }
    }

    /// Splits the work before the logits among `dies`, cutting between layers.
    pub fn split(&self, dies: usize) -> Result<Vec<std::ops::Range<usize>>, String> {
        let dies = dies.max(1);
        if dies > self.num_layers.max(1) {
            return Err(format!("{dies} dies for {} layers", self.num_layers));
        }
        let mut ranges = Vec::with_capacity(dies);
        let mut start = 0;
        for die in 1..dies {
            let boundary = (die * self.num_layers / dies) as u32;
            let end = self.ops.iter().position(|inst| inst.layer == Some(boundary)).ok_or("a layer with no operations")?;
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
            return Err(format!("{dies} dies for {} layers", self.num_layers));
        }
        // Where each layer starts, and the cost of each: what comes before the first
        // layer and the logits go with the first.
        let mut starts = Vec::with_capacity(layers);
        for layer in 1..layers as u32 {
            starts.push(self.ops.iter().position(|inst| inst.layer == Some(layer)).ok_or("a layer with no operations")?);
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
            other => Err(format!("the cut at {cut} crosses {} vectors", other.len())),
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
    text.get(key).and_then(Value::as_u64).map(|v| v as usize).ok_or_else(|| format!("“{key}” is missing from the config"))
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
    let qg = e.matvec("Q projection", p("q_proj"), normed, query_len * 2);
    let k = e.matvec("K projection", p("k_proj"), normed, kv_len);
    let v = e.matvec("V projection", p("v_proj"), normed, kv_len);
    let query = e.buf(query_len);
    let gate = e.buf(query_len);
    e.op("Split query and gate", Op::SplitQueryGate { src: qg, query, gate, block: s.head_dim });
    let query = e.norm("Normalize Q", query, p("q_norm"), s.head_dim, query_len);
    let key = e.norm("Normalize K", k, p("k_norm"), s.head_dim, kv_len);
    e.op("RoPE on Q", Op::Rope { buf: query, block: s.head_dim, rotary: s.rotary, table: 0 });
    e.op("RoPE on K", Op::Rope { buf: key, block: s.head_dim, rotary: s.rotary, table: 0 });
    let slot = e.program.kv_slots.len();
    e.program.kv_slots.push(KvShape { head_dim: s.head_dim, heads: s.kv_heads });
    e.op("Store in the KV cache", Op::KvAppend { slot, key, value: v, window: None });
    let attn = e.buf(query_len);
    e.op("Attention", Op::AttnSum { slot, query, head_dim: s.head_dim, scale_dim: s.head_dim, heads: s.heads, kv_heads: s.kv_heads, window: None, softcap: None, dst: attn });
    let gate_on = e.buf(query_len);
    e.op("Gate sigmoid", Op::Sigmoid { src: gate, dst: gate_on });
    let gated = e.buf(query_len);
    e.op("Apply gate", Op::MulElem { a: attn, b: gate_on, dst: gated });
    e.matvec("Output projection", p("o_proj"), gated, s.hidden)
}

fn emit_feed_forward(e: &mut Emit, s: &Shape, prefix: &str, normed: Buf) -> Buf {
    if let Some(m) = s.moe {
        let dst = e.buf(s.hidden);
        e.op(
            "Mixture of experts",
            Op::Moe { prefix: format!("{prefix}.mlp"), src: normed, dst, experts: m.experts, top_k: m.top_k, intermediate: m.intermediate, shared_intermediate: m.shared_intermediate },
        );
        return dst;
    }
    let p = |n: &str| format!("{prefix}.mlp.{n}");
    let gate = e.matvec("Gate projection", p("gate_proj"), normed, s.intermediate);
    let up = e.matvec("Up projection", p("up_proj"), normed, s.intermediate);
    let silu = e.buf(s.intermediate);
    e.op("SiLU activation", Op::Silu { src: gate, dst: silu });
    let product = e.buf(s.intermediate);
    e.op("Gate × up", Op::MulElem { a: silu, b: up, dst: product });
    e.matvec("Down projection", p("down_proj"), product, s.hidden)
}

fn emit_layer_tail(e: &mut Emit, s: &Shape, prefix: &str, x: Buf, mixed: Buf) -> Buf {
    let after_mixer = e.buf(s.hidden);
    e.op("Residual", Op::Add { a: x, b: mixed, dst: after_mixer });
    let normed = e.norm("Normalize", after_mixer, format!("{prefix}.post_attention_layernorm"), s.hidden, s.hidden);
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
        return Err(format!("its weights are quantized with “{method}”, a format that cannot be read yet"));
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
        Some(other) => Err(format!("there is no compiler for the “{other}” architecture")),
        // llama2.c describes its models with its own names: they are Llamas.
        None if value.get("dim").is_some() && value.get("n_layers").is_some() => compile_llama(&llama2c_as_llama(&value)?),
        None => Err("the config does not give a model_type".into()),
    }
}

/// A llama2.c configuration (`dim`, `n_layers`, …) as the Llama configuration it stands for,
/// with the hidden size of its MLP worked out as llama2.c does.
fn llama2c_as_llama(config: &Value) -> Result<Value, String> {
    let number = |key: &str| config.get(key).and_then(Value::as_u64).ok_or_else(|| format!("the llama2.c config lacks “{key}”"));
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
    e.op("Add bias", Op::AddWeight { src: out, weight: format!("{matrix}.bias"), dst });
    dst
}

fn plain_attention(e: &mut Emit, s: &Shape, prefix: &str, normed: Buf, layer: &PlainLayer) -> Buf {
    let p = |n: &str| format!("{prefix}.self_attn.{n}");
    let query_len = s.heads * s.head_dim;
    let kv_len = s.kv_heads * s.head_dim;
    let mut query = with_bias(e, "Q projection", p("q_proj"), normed, query_len, layer.attention_bias);
    let mut key = with_bias(e, "K projection", p("k_proj"), normed, kv_len, layer.attention_bias);
    let value = with_bias(e, "V projection", p("v_proj"), normed, kv_len, layer.attention_bias);
    if layer.head_norms {
        query = e.norm("Normalize Q", query, p("q_norm"), s.head_dim, query_len);
        key = e.norm("Normalize K", key, p("k_norm"), s.head_dim, kv_len);
    }
    e.op("RoPE on Q", Op::Rope { buf: query, block: s.head_dim, rotary: s.rotary, table: layer.rope_table });
    e.op("RoPE on K", Op::Rope { buf: key, block: s.head_dim, rotary: s.rotary, table: layer.rope_table });
    let slot = e.program.kv_slots.len();
    e.program.kv_slots.push(KvShape { head_dim: s.head_dim, heads: s.kv_heads });
    e.op("Store in the KV cache", Op::KvAppend { slot, key, value, window: layer.window });
    let attn = e.buf(query_len);
    e.op("Attention", Op::AttnSum { slot, query, head_dim: s.head_dim, scale_dim: layer.scale_dim, heads: s.heads, kv_heads: s.kv_heads, window: layer.window, softcap: layer.softcap, dst: attn });
    with_bias(e, "Output projection", p("o_proj"), attn, s.hidden, layer.output_bias)
}

/// The gated feed-forward of the plain family, with its activation and optional bias.
fn plain_mlp(e: &mut Emit, s: &Shape, prefix: &str, normed: Buf, gelu: Option<bool>, bias: bool) -> Buf {
    let p = |n: &str| format!("{prefix}.mlp.{n}");
    let gate = with_bias(e, "Gate projection", p("gate_proj"), normed, s.intermediate, bias);
    let up = with_bias(e, "Up projection", p("up_proj"), normed, s.intermediate, bias);
    let active = e.buf(s.intermediate);
    match gelu {
        Some(approximate) => e.op("GELU activation", Op::Gelu { src: gate, approximate, dst: active }),
        None => e.op("SiLU activation", Op::Silu { src: gate, dst: active }),
    }
    let product = e.buf(s.intermediate);
    e.op("Gate × up", Op::MulElem { a: active, b: up, dst: product });
    with_bias(e, "Down projection", p("down_proj"), product, s.hidden, bias)
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
        return Err(format!("{heads} heads do not divide into {kv_heads} KV groups"));
    }
    let head_dim = config.get("head_dim").and_then(Value::as_u64).map_or(hidden / heads, |v| v as usize);
    if head_dim == 0 || (config.get("head_dim").is_none() && hidden % heads != 0) {
        return Err(format!("{hidden} does not divide among {heads} heads"));
    }
    let on = |key: &str, default: bool| config.get(key).and_then(Value::as_bool).unwrap_or(default);
    let attention_bias = on("attention_bias", family == "qwen2");
    let output_bias = attention_bias && family != "qwen2";
    let mlp_bias = on("mlp_bias", false);
    let sliding = family == "gemma3_text";
    let layer_kinds = config.get("layer_types").and_then(Value::as_array);
    if !sliding && (on("use_sliding_window", false) || layer_kinds.is_some_and(|l| l.iter().any(|t| t != "full_attention"))) {
        return Err("it uses sliding-window layers and the compiler does not generate them yet".into());
    }
    let activation = if gemma { config.get("hidden_activation").or_else(|| config.get("hidden_act")) } else { config.get("hidden_act") }.and_then(Value::as_str);
    let gelu = match (gemma, activation) {
        (false, None | Some("silu" | "swish")) => None,
        (true, None) => Some(family != "gemma"),
        (true, Some("gelu")) => Some(false),
        (true, Some("gelu_pytorch_tanh")) => Some(true),
        (_, Some(other)) => return Err(format!("it uses the “{other}” activation, which the chip does not have")),
    };
    let rope = config.get("rope_parameters").or_else(|| config.get("rope_scaling"));
    let rope_scaling = match rope.and_then(|r| r.get("rope_type").or_else(|| r.get("type"))).and_then(Value::as_str) {
        None | Some("default") => None,
        Some("llama3") => {
            let rope = rope.unwrap_or(&Value::Null);
            let number = |key: &str, default: f64| rope.get(key).and_then(Value::as_f64).unwrap_or(default);
            Some(RopeScaling { factor: number("factor", 1.0), low: number("low_freq_factor", 1.0), high: number("high_freq_factor", 4.0), trained_context: number("original_max_position_embeddings", 8192.0) })
        }
        Some(kind) => return Err(format!("it uses a scaled RoPE (“{kind}”) the compiler does not generate")),
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
    e.op("Read embedding", Op::EmbRead { table: "model.embed_tokens".into(), dst: x });
    if gemma {
        let scaled = e.buf(hidden);
        e.op("Scale embedding", Op::Scale { src: x, factor: (hidden as f32).sqrt(), dst: scaled });
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
        let normed = e.norm("Normalize input", x, format!("{prefix}.input_layernorm"), hidden, hidden);
        let mixed = plain_attention(&mut e, &shape, &prefix, normed, &layer);
        if four_norms {
            let after_attention = e.norm("Normalize attention", mixed, format!("{prefix}.post_attention_layernorm"), hidden, hidden);
            let h = e.buf(hidden);
            e.op("Residual", Op::Add { a: x, b: after_attention, dst: h });
            let before_mlp = e.norm("Normalize before the MLP", h, format!("{prefix}.pre_feedforward_layernorm"), hidden, hidden);
            let down = plain_mlp(&mut e, &shape, &prefix, before_mlp, gelu, mlp_bias);
            let after_mlp = e.norm("Normalize after the MLP", down, format!("{prefix}.post_feedforward_layernorm"), hidden, hidden);
            let next = e.buf(hidden);
            e.op("Residual", Op::Add { a: h, b: after_mlp, dst: next });
            x = next;
        } else {
            let after_mixer = e.buf(hidden);
            e.op("Residual", Op::Add { a: x, b: mixed, dst: after_mixer });
            let before_mlp = e.norm("Normalize", after_mixer, format!("{prefix}.post_attention_layernorm"), hidden, hidden);
            let down = plain_mlp(&mut e, &shape, &prefix, before_mlp, gelu, mlp_bias);
            let next = e.buf(hidden);
            e.op("Residual", Op::Add { a: after_mixer, b: down, dst: next });
            x = next;
        }
    }
    e.layer = None;
    e.program.logits_from = e.program.ops.len();
    let normed = e.norm("Final norm", x, "model.norm".into(), hidden, hidden);
    e.program.final_hidden = normed;
    // With tied embeddings there is one table in memory, read both ways.
    let head = if on("tie_word_embeddings", gemma) { "model.embed_tokens" } else { "lm_head" };
    let mut logits = e.matvec("Vocabulary output", head.into(), normed, vocab);
    if let Some(cap) = softcap("final_logit_softcapping", 30.0) {
        let capped = e.buf(vocab);
        e.op("Cap logits", Op::SoftCap { src: logits, cap, dst: capped });
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
        return Err("it uses a scaled RoPE the compiler does not generate for this architecture".into());
    }
    let groups = config.get("n_group").and_then(Value::as_u64).unwrap_or(1);
    if groups > 1 && config.get("topk_group").and_then(Value::as_u64).unwrap_or(1) != groups {
        return Err("it chooses experts by groups and the compiler does not generate that yet".into());
    }
    let wanted = |key: &str, value: &str| config.get(key).and_then(Value::as_str).is_none_or(|v| v == value);
    if !wanted("topk_method", "noaux_tc") || !wanted("scoring_func", "sigmoid") {
        return Err("it uses an expert selection method the compiler does not generate".into());
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
    e.op("Read embedding", Op::EmbRead { table: "model.embed_tokens".into(), dst: x });
    for i in 0..layers {
        e.layer = Some(i as u32);
        let prefix = format!("model.layers.{i}");
        let p = |s: &str| format!("{prefix}.self_attn.{s}");
        let normed = e.norm("Normalize input", x, format!("{prefix}.input_layernorm"), hidden, hidden);
        let query = match q_rank {
            Some(rank) => {
                let compact = with_bias(&mut e, "Latent Q projection", p("q_a_proj"), normed, rank, bias);
                let compact = e.norm("Normalize latent Q", compact, p("q_a_layernorm"), rank, rank);
                e.matvec("Q expansion", p("q_b_proj"), compact, heads * q_head)
            }
            None => e.matvec("Q projection", p("q_proj"), normed, heads * q_head),
        };
        let compressed = with_bias(&mut e, "Latent KV projection", p("kv_a_proj_with_mqa"), normed, lora + rope, bias);
        let raw_latent = e.buf(lora);
        e.op("Split latent", Op::Slice { src: compressed, from: 0, dst: raw_latent });
        let rope_key = e.buf(rope);
        e.op("Split rotated key", Op::Slice { src: compressed, from: lora, dst: rope_key });
        let latent = e.norm("Normalize latent", raw_latent, p("kv_a_layernorm"), lora, lora);
        let slot = e.program.kv_slots.len();
        e.program.kv_slots.push(KvShape { head_dim: lora + rope, heads: 1 });
        let attended = e.buf(heads * value_dim);
        e.op("Latent attention", Op::Mla { slot, query, latent, rope_key, embed: p("embed_q"), unembed: p("unembed_out"), heads, nope, rope, lora, value_dim, scale: (q_head as f32).powf(-0.5), dst: attended });
        let mixed = with_bias(&mut e, "Output projection", p("o_proj"), attended, hidden, bias);
        let after_attention = e.buf(hidden);
        e.op("Residual", Op::Add { a: x, b: mixed, dst: after_attention });
        let before_mlp = e.norm("Normalize", after_attention, format!("{prefix}.post_attention_layernorm"), hidden, hidden);
        let is_moe = i >= first_dense && i.is_multiple_of(frequency);
        let feed = match routed.filter(|_| is_moe) {
            Some(experts) => {
                let dst = e.buf(hidden);
                let moe_size = dim(config, "moe_intermediate_size")?;
                let shared = config.get("n_shared_experts").and_then(Value::as_u64).unwrap_or(0) as usize;
                e.op("Mixture of experts", Op::SigmoidMoe {
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
    let normed = e.norm("Final norm", x, "model.norm".into(), hidden, hidden);
    e.program.final_hidden = normed;
    e.program.logits = e.matvec("Vocabulary output", "lm_head".into(), normed, vocab);
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
        embedding_rows: None,
        draft: None,
        parts: Vec::new(),
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
        ops.push(inst("Image head product", Op::Matvec { matrix: matrix.into(), src, dst: product }));
        let biased = buffer(program, len);
        ops.push(inst("Add bias", Op::AddWeight { src: product, weight: format!("{matrix}.bias"), dst: biased }));
        biased
    };
    let mut logits_ops = Vec::new();
    let projected = layer(program, &mut logits_ops, "model.generation_head.proj_out", program.final_hidden, hidden);
    let active = buffer(program, hidden);
    logits_ops.push(inst("GELU activation", Op::Gelu { src: projected, approximate: false, dst: active }));
    let logits = layer(program, &mut logits_ops, "model.generation_head.vision_head", active, codes);

    let mut input_ops = Vec::new();
    let code = buffer(program, code_dim);
    input_ops.push(inst("Code vector", Op::EmbRead { table: "model.generation_embeddings".into(), dst: code }));
    let widened = layer(program, &mut input_ops, "model.generation_aligner.fc1", code, width);
    let active = buffer(program, width);
    input_ops.push(inst("GELU activation", Op::Gelu { src: widened, approximate: false, dst: active }));
    let input = layer(program, &mut input_ops, "model.generation_aligner.hidden_layers.0", active, hidden);
    program.set_part(Part::Codes(CodeHead { logits_ops, logits, codes, input_ops, input }));
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
    e.op("Read embedding", Op::EmbRead { table: "model.embed_tokens".into(), dst: x });

    for (i, kind) in layer_types.iter().enumerate() {
        e.layer = Some(i as u32);
        let prefix = format!("model.layers.{i}");
        let p = |s: &str| format!("{prefix}.{s}");

        let normed = e.norm("Normalize input", x, p("input_layernorm"), hidden, hidden);
        let mixed = match kind.as_str() {
            "full_attention" => emit_attention(&mut e, &shape, &prefix, normed),
            "linear_attention" => {
                let key_len = key_heads * key_dim;
                let value_len = value_heads * value_dim;
                let conv_dim = 2 * key_len + value_len;
                let qkv = e.matvec("QKV projection", p("linear_attn.in_proj_qkv"), normed, conv_dim);
                let z = e.matvec("Z projection", p("linear_attn.in_proj_z"), normed, value_len);
                let b = e.matvec("β projection", p("linear_attn.in_proj_b"), normed, value_heads);
                let a = e.matvec("a projection", p("linear_attn.in_proj_a"), normed, value_heads);
                let slot = e.program.conv_slots.len();
                e.program.conv_slots.push((conv_dim, kernel));
                e.program.delta_slots.push(value_heads * key_dim * value_dim);
                let conv = e.buf(conv_dim);
                e.op("Causal convolution", Op::Dconv { slot, weight: p("linear_attn.conv1d"), src: qkv, dst: conv });
                let silu = e.buf(conv_dim);
                e.op("SiLU activation", Op::Silu { src: conv, dst: silu });
                let out = e.buf(value_len);
                e.op(
                    "Delta rule",
                    Op::DeltaStep { slot, mixed: silu, a, b, a_log: p("linear_attn.A_log"), dt_bias: p("linear_attn.dt_bias"), key_dim, key_heads, value_dim, value_heads, dst: out },
                );
                let gated = e.buf(value_len);
                e.op("Gated norm", Op::RmsGated { src: out, z, weight: p("linear_attn.norm"), block: value_dim, dst: gated });
                e.matvec("Output projection", p("linear_attn.out_proj"), gated, hidden)
            }
            other => return Err(format!("unknown layer type: {other:?}")),
        };
        x = emit_layer_tail(&mut e, &shape, &prefix, x, mixed);
    }

    e.layer = None;
    e.program.logits_from = e.program.ops.len();
    let normed = e.norm("Final norm", x, "model.norm".into(), hidden, hidden);
    e.program.final_hidden = normed;
    let logits = e.matvec("Vocabulary output", "lm_head".into(), normed, vocab);
    e.program.logits = logits;

    if text.get("mtp_num_hidden_layers").and_then(Value::as_u64).unwrap_or(0) > 0 {
        let main_ops = std::mem::take(&mut e.program.ops);
        let hidden_in = e.buf(hidden);
        let token = e.buf(hidden);
        e.op("Read embedding", Op::EmbRead { table: "model.embed_tokens".into(), dst: token });
        let en = e.norm("Normalize embedding", token, "mtp.pre_fc_norm_embedding".into(), hidden, hidden);
        let hn = e.norm("Normalize state", hidden_in, "mtp.pre_fc_norm_hidden".into(), hidden, hidden);
        let joined = e.buf(2 * hidden);
        e.op("Join embedding and state", Op::Concat { a: en, b: hn, dst: joined });
        let mixed_in = e.matvec("fc projection", "mtp.fc".into(), joined, hidden);
        let kv_slot = e.program.kv_slots.len();
        let prefix = "mtp.layers.0";
        let normed = e.norm("Normalize input", mixed_in, format!("{prefix}.input_layernorm"), hidden, hidden);
        let attention = emit_attention(&mut e, &shape, prefix, normed);
        let output = emit_layer_tail(&mut e, &shape, prefix, mixed_in, attention);
        let logits_from = e.program.ops.len();
        let hidden_out = e.norm("Final norm", output, "mtp.norm".into(), hidden, hidden);
        let logits = e.matvec("Vocabulary output", "lm_head".into(), hidden_out, vocab);
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
            let graph = ModelGraph::from_config_json("model", config).unwrap();
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
        assert_eq!(program.kv_slots.len(), 16 + 1, "the attention layers and the MTP head");
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
            Op::LayerNorm { src, dst, .. } => (vec![*src], vec![*dst]),
            Op::LoadRow { store, dst } | Op::LoadRows { store, dst, .. } => (vec![*store], vec![*dst]),
            // A store keeps its other rows: writing one also reads it.
            Op::StoreRow { src, store } => (vec![*src, *store], vec![*store]),
            Op::GroupNorm { src, sums, squares, dst, .. } => (vec![*src, *sums, *squares], vec![*dst]),
                Op::Concat { a, b, dst } => (vec![*a, *b], vec![*dst]),
                Op::Moe { src, dst, .. } => (vec![*src], vec![*dst]),
            };
            for r in reads {
                assert!(written[r], "{} reads a buffer never written", inst.name);
            }
            for w in writes {
                written[w] = true;
            }
        }
    }
}

/// Routines over a store of rows, as the chip runs them: operations that form routines,
/// and the order the host or the firmware runs them in. A routine runs once for each row,
/// with the row as the token; one that attends over every row takes the last row as its
/// position. Its buffers and key slots are the program's it belongs to. The image encoder
/// is one, with a row per patch: the names of its fields come from it.
#[derive(Clone, Debug, Serialize)]
pub struct RowPart {
    pub ops: Vec<Inst>,
    /// Turns a patch's pixels, with its learned position, into the first hidden state.
    pub embed: std::ops::Range<usize>,
    /// For each block: the routine that stores the patch's keys, values and query, and
    /// the one that attends over every patch and runs the rest of the block.
    pub blocks: Vec<(std::ops::Range<usize>, std::ops::Range<usize>)>,
    /// Joins `merge × merge` patches into one token of the language model.
    pub merge: std::ops::Range<usize>,
    /// What the host writes before the patch embedding: the pixels and the learned
    /// position, already resampled to the image's grid.
    pub pixels: Buf,
    pub position: Buf,
    /// The token for the language model each merge routine leaves.
    pub output: Buf,
    pub head_dim: usize,
    /// The most patches an image may have, which sizes the stores.
    pub patches: usize,
    /// Routines of its own after the merge, each with an entry: Show-o keeps the scores
    /// with the text and combines them with those without it.
    pub extra: Vec<std::ops::Range<usize>>,
}

impl RowPart {
    /// Every routine in the order of its entries: embedding, each block's two, fusion.
    pub fn routines(&self) -> Vec<std::ops::Range<usize>> {
        std::iter::once(self.embed.clone()).chain(self.blocks.iter().flat_map(|(a, b)| [a.clone(), b.clone()])).chain([self.merge.clone()]).chain(self.extra.iter().cloned()).collect()
    }
}

/// An image encoder on its own: a program whose only operations are the encoder's.
#[derive(Clone, Debug, Serialize)]
pub struct VisionProgram {
    pub program: Program,
    pub vision: RowPart,
}

/// The encoder in `vision_config`, for images of at most `patches` patches, on its own.
pub fn compile_vision(config: &str, patches: usize) -> Result<VisionProgram, String> {
    let root: Value = serde_json::from_str(config).map_err(|e| e.to_string())?;
    let mut program = Program {
        family: "vision".into(),
        vocab: 1,
        hidden: 0,
        num_layers: 0,
        buffers: Vec::new(),
        logits_from: 0,
        ops: Vec::new(),
        logits: 0,
        kv_slots: Vec::new(),
        delta_slots: Vec::new(),
        conv_slots: Vec::new(),
        rope_theta: 10_000.0,
        rope_scaling: None,
        local_rope_theta: None,
        rms_epsilon: 1e-6,
        final_hidden: 0,
        embedding_rows: None,
        draft: None,
        parts: Vec::new(),
    };
    let vision = vision_part(&root, patches, &mut program)?;
    program.ops = vision.ops.clone();
    program.logits_from = program.ops.len();
    program.hidden = root["vision_config"]["hidden_size"].as_u64().unwrap_or(0) as usize;
    program.num_layers = vision.blocks.len();
    (program.logits, program.final_hidden) = (vision.output, vision.output);
    Ok(VisionProgram { program, vision })
}

/// A language model that also reads images: `program` with the encoder of `config`, its
/// buffers and key slots after the model's. Its layer norms keep their own epsilon.
pub fn attach_vision(mut program: Program, config: &str, patches: usize) -> Result<Program, String> {
    let root: Value = serde_json::from_str(config).map_err(|e| e.to_string())?;
    let vision = vision_part(&root, patches, &mut program)?;
    program.set_part(Part::Rows(vision));
    Ok(program)
}

fn vision_part(root: &Value, patches: usize, program: &mut Program) -> Result<RowPart, String> {
    let v = &root["vision_config"];
    let number = |key: &str| v[key].as_u64().map(|n| n as usize).ok_or(format!("vision_config has no {key}"));
    let (hidden, heads, depth, intermediate) = (number("hidden_size")?, number("num_heads")?, number("depth")?, number("intermediate_size")?);
    let (patch, temporal, merge, out) = (number("patch_size")?, number("temporal_patch_size")?, number("spatial_merge_size")?, number("out_hidden_size")?);
    let head_dim = hidden / heads;
    let merged = hidden * merge * merge;
    let patches = patches.next_multiple_of(merge * merge);
    let first_slot = program.kv_slots.len();

    let buffers = &mut program.buffers;
    let mut buf = |len: usize| {
        buffers.push(len);
        buffers.len() - 1
    };
    let mut ops: Vec<Inst> = Vec::new();
    let mut emit = |op: Op, name: &'static str, layer: Option<u32>| ops.push(Inst { op, layer, name });
    let t = |name: &str| format!("vision_tower.{name}");

    let pixels = buf(3 * temporal * patch * patch);
    let position = buf(hidden);
    let state = buf(patches * hidden);
    let (x, biased, placed) = (buf(hidden), buf(hidden), buf(hidden));
    emit(Op::Matvec { matrix: t("patch_embed.proj"), src: pixels, dst: x }, "patch embedding", None);
    emit(Op::AddWeight { src: x, weight: t("patch_embed.proj.bias"), dst: biased }, "patch bias", None);
    emit(Op::Add { a: biased, b: position, dst: placed }, "learned position", None);
    emit(Op::StoreRow { src: placed, store: state }, "store patch", None);
    let embed = 0..4;

    let mut blocks = Vec::new();
    for block in 0..depth {
        let b = |name: &str| t(&format!("blocks.{block}.{name}"));
        let layer = Some(block as u32);
        let queries = buf(patches * hidden);
        let start = 4 + block * 25;
        let (h, normed, qkv, qkv_b, q, k, v_) = (buf(hidden), buf(hidden), buf(3 * hidden), buf(3 * hidden), buf(hidden), buf(hidden), buf(hidden));
        emit(Op::LoadRow { store: state, dst: h }, "read patch", layer);
        emit(Op::LayerNorm { src: h, weight: b("norm1"), block: hidden, dst: normed }, "norm 1", layer);
        emit(Op::Matvec { matrix: b("attn.qkv"), src: normed, dst: qkv }, "q, k and v", layer);
        emit(Op::AddWeight { src: qkv, weight: b("attn.qkv.bias"), dst: qkv_b }, "q, k and v bias", layer);
        emit(Op::Slice { src: qkv_b, from: 0, dst: q }, "query", layer);
        emit(Op::Slice { src: qkv_b, from: hidden, dst: k }, "key", layer);
        emit(Op::Slice { src: qkv_b, from: 2 * hidden, dst: v_ }, "value", layer);
        emit(Op::Rope { buf: q, block: head_dim, rotary: head_dim, table: 2 }, "2D RoPE of the query", layer);
        emit(Op::Rope { buf: k, block: head_dim, rotary: head_dim, table: 2 }, "2D RoPE of the key", layer);
        emit(Op::KvAppend { slot: first_slot + block, key: k, value: v_, window: None }, "store key and value", layer);
        emit(Op::StoreRow { src: q, store: queries }, "store query", layer);
        let middle = start + 11;
        let (q2, attended, projected, projected_b, h2, h3, normed2, up, up_b, gelu, down, down_b, h4) =
            (buf(hidden), buf(hidden), buf(hidden), buf(hidden), buf(hidden), buf(hidden), buf(hidden), buf(intermediate), buf(intermediate), buf(intermediate), buf(hidden), buf(hidden), buf(hidden));
        emit(Op::LoadRow { store: queries, dst: q2 }, "read query", layer);
        emit(Op::AttnSum { slot: first_slot + block, query: q2, head_dim, scale_dim: head_dim, heads, kv_heads: heads, window: None, softcap: None, dst: attended }, "attention over all the patches", layer);
        emit(Op::Matvec { matrix: b("attn.proj"), src: attended, dst: projected }, "projection", layer);
        emit(Op::AddWeight { src: projected, weight: b("attn.proj.bias"), dst: projected_b }, "projection bias", layer);
        emit(Op::LoadRow { store: state, dst: h2 }, "read patch", layer);
        emit(Op::Add { a: h2, b: projected_b, dst: h3 }, "residual", layer);
        emit(Op::LayerNorm { src: h3, weight: b("norm2"), block: hidden, dst: normed2 }, "norm 2", layer);
        emit(Op::Matvec { matrix: b("mlp.linear_fc1"), src: normed2, dst: up }, "MLP, up", layer);
        emit(Op::AddWeight { src: up, weight: b("mlp.linear_fc1.bias"), dst: up_b }, "up bias", layer);
        emit(Op::Gelu { src: up_b, approximate: true, dst: gelu }, "GELU", layer);
        emit(Op::Matvec { matrix: b("mlp.linear_fc2"), src: gelu, dst: down }, "MLP, down", layer);
        emit(Op::AddWeight { src: down, weight: b("mlp.linear_fc2.bias"), dst: down_b }, "down bias", layer);
        emit(Op::Add { a: h3, b: down_b, dst: h4 }, "residual", layer);
        if block + 1 < depth {
            emit(Op::StoreRow { src: h4, store: state }, "store patch", layer);
            blocks.push((start..middle, middle..start + 25));
        } else {
            // The fusion's norm is each patch's own, so the last block applies it before
            // storing: the patches of a token then line up without one of them, much larger
            // than its neighbours, drowning their small values.
            let normed = buf(hidden);
            emit(Op::LayerNorm { src: h4, weight: t("merger.norm"), block: hidden, dst: normed }, "merger norm", layer);
            emit(Op::StoreRow { src: normed, store: state }, "store patch", layer);
            blocks.push((start..middle, middle..start + 26));
        }
    }

    let start = 4 + depth * 25 + usize::from(depth > 0);
    let (joined, up, up_b, gelu, down, output) = (buf(merged), buf(merged), buf(merged), buf(merged), buf(out), buf(out));
    let normed = if depth == 0 {
        let normed = buf(merged);
        emit(Op::LoadRow { store: state, dst: joined }, "read a token's patches", None);
        emit(Op::LayerNorm { src: joined, weight: t("merger.norm"), block: hidden, dst: normed }, "merger norm", None);
        normed
    } else {
        emit(Op::LoadRow { store: state, dst: joined }, "read a token's patches", None);
        joined
    };
    emit(Op::Matvec { matrix: t("merger.linear_fc1"), src: normed, dst: up }, "merger, up", None);
    emit(Op::AddWeight { src: up, weight: t("merger.linear_fc1.bias"), dst: up_b }, "up bias", None);
    emit(Op::Gelu { src: up_b, approximate: false, dst: gelu }, "GELU", None);
    emit(Op::Matvec { matrix: t("merger.linear_fc2"), src: gelu, dst: down }, "merger, down", None);
    emit(Op::AddWeight { src: down, weight: t("merger.linear_fc2.bias"), dst: output }, "down bias", None);
    let merge_range = start..start + if depth == 0 { 7 } else { 6 };
    debug_assert_eq!(ops.len(), merge_range.end);

    program.kv_slots.extend((0..depth).map(|_| KvShape { head_dim, heads }));
    Ok(RowPart { ops, embed, blocks, merge: merge_range, pixels, position, output, head_dim, patches, extra: Vec::new() })
}
