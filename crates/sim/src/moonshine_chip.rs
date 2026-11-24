//! Moonshine in the die's arithmetic: speech to text with the chip's operations, run in
//! the order the host gives them, as MAGVIT-v2's decoder and the image encoder are.
//!
//! Nothing here is new to the die. The program is routines over stores of rows:
//!
//! - **The waveform** goes in 64 samples to a row. The first convolution (kernel 127,
//!   stride 64) reads rows `t` and `t + 1` side by side, with a zero for the 128th weight.
//!   The other two read 7 and 3 rows from every third and every other row: the host gives
//!   the first row read as the token, and a second routine stores the result at its own.
//! - **The GroupNorm** of one group adds up every row's values and squares with a product
//!   by a row of ones, then adds the rows in pairs until one holds the totals, as MAGVIT-v2's.
//! - **Each encoder block** goes twice over the frames: the first stores each frame's key
//!   and value and its query; the second attends over every frame, with the position at
//!   the last one, and runs the rest of the block.
//! - **Cross-attention** needs no new operation. Once the encoder is done, a routine per
//!   frame stores each decoder layer's key and value of it in a slot of its own. The
//!   decoder's cross-attention is then a routine of its own that runs with the position at
//!   the last frame: the same attention, reading as far as the frames go.
//! - **Each decoder token** runs its embedding, then for each layer a routine at its own
//!   position (self-attention, with RoPE) and one at the last frame's (cross-attention and
//!   the MLP), then the logits.
//!
//! Two things are done to the weights as they are read, which change nothing: Moonshine
//! rotates adjacent pairs of each head and the die the two halves of the rotated part, so
//! the rows of every query and key projection are reordered alike, which leaves their
//! products as they were; and its LayerNorms, which have no bias, get one of zeros.
//!
//! A chip is built for clips of up to some length and takes any shorter one: the stores are
//! sized for the longest, the host runs as many rows as the clip has, and the GroupNorm's
//! totals are multiplied by `1 / count`, which the host writes for each clip, before a
//! GroupNorm of one value.

use crate::chip::RoutinesDie;
use crate::lower::Layout;
use crate::moonshine::{argmax, MoonshineConfig};
use crate::vm::BlockVm;
use crate::weights::{Tensor, Weights};
use compiler::{Buf, Inst, KvShape, Op, Part, Program, RowPart};
use netlist::block::{encode_vector, exponent_register, BlockOps};
use netlist::schedule::{Record, FIRST_FREE, NO_ROUTINE, POSITION, TOKEN, ZERO};
use netlist::DieConfig;
use std::ops::Range;
use std::path::Path;
use std::sync::Arc;

/// A routine's operations and whether it loads the rotation of its position first.
pub type Routine = (Range<usize>, bool);

/// A routine that computes and one that stores what it computed.
type Pair = (Range<usize>, Range<usize>);

/// A vector in the die's format: its mantissas and its exponent.
pub type Vector = (Vec<i64>, i64);

/// Samples in a row of the waveform's store.
pub const SAMPLES_PER_ROW: usize = 64;

/// One routine as the host runs it: its operations, its token and its position, whether
/// it loads the rotation of its position first, and whether the logits are read after it.
#[derive(Clone, Debug, PartialEq)]
pub struct Call {
    pub routine: Range<usize>,
    pub token: u32,
    pub position: u32,
    pub rope: bool,
    pub read: bool,
}

pub(crate) struct Builder<'a> {
    pub(crate) weights: &'a Weights,
    pub(crate) buffers: Vec<usize>,
    pub(crate) ops: Vec<Inst>,
    pub(crate) derived: Vec<Tensor>,
}

impl Builder<'_> {
    pub(crate) fn buf(&mut self, len: usize) -> Buf {
        self.buffers.push(len);
        self.buffers.len() - 1
    }

    pub(crate) fn op(&mut self, name: &'static str, op: Op) {
        self.ops.push(Inst { op, layer: None, name });
    }

    pub(crate) fn routine(&mut self, body: impl FnOnce(&mut Self) -> Result<(), String>) -> Result<Range<usize>, String> {
        let start = self.ops.len();
        body(self)?;
        Ok(start..self.ops.len())
    }

    pub(crate) fn floats(&self, name: &str) -> Result<Vec<f32>, String> {
        self.weights.floats(name)
    }

    pub(crate) fn derive(&mut self, name: String, shape: Vec<usize>, values: Vec<f32>) -> String {
        if !self.derived.iter().any(|(n, _, _)| *n == name) {
            self.derived.push((name.clone(), shape, values));
        }
        name
    }

    /// `matrix · src` into a new buffer, plus `{bias}` when given.
    pub(crate) fn product(&mut self, matrix: &str, src: Buf, outputs: usize, bias: Option<String>) -> Buf {
        let out = self.buf(outputs);
        self.op("product", Op::Matvec { matrix: matrix.into(), src, dst: out });
        match bias {
            Some(weight) => {
                let biased = self.buf(outputs);
                self.op("bias", Op::AddWeight { src: out, weight, dst: biased });
                biased
            }
            None => out,
        }
    }

    /// A linear layer of the checkpoint, with its bias if it has one.
    pub(crate) fn linear(&mut self, name: &str, src: Buf, outputs: usize) -> Buf {
        let bias = format!("{name}.bias");
        let bias = self.weights.shape(&bias).is_some().then_some(bias);
        self.product(name, src, outputs, bias)
    }

    /// A LayerNorm of the checkpoint, its missing bias zeros.
    pub(crate) fn layer_norm(&mut self, name: &str, src: Buf, width: usize) -> Buf {
        if self.weights.shape(&format!("{name}.bias")).is_none() {
            self.derive(format!("{name}.bias"), vec![width], vec![0.0; width]);
        }
        let dst = self.buf(width);
        self.op("LayerNorm", Op::LayerNorm { src, weight: name.into(), block: width, dst });
        dst
    }

    /// A query or key projection with its rows reordered for the die's rotation: in each
    /// head, the rotated pairs' first elements, then their second ones, then the rest.
    pub(crate) fn rotated(&mut self, name: &str, heads: usize, head_dim: usize, rotary: usize) -> Result<String, String> {
        let weight = self.floats(&format!("{name}.weight"))?;
        let cols = weight.len() / (heads * head_dim);
        let order: Vec<usize> = (0..rotary / 2).map(|i| 2 * i).chain((0..rotary / 2).map(|i| 2 * i + 1)).chain(rotary..head_dim).collect();
        let mut laid = Vec::with_capacity(weight.len());
        for head in 0..heads {
            for &row in &order {
                let row = head * head_dim + row;
                laid.extend_from_slice(&weight[row * cols..(row + 1) * cols]);
            }
        }
        let derived = format!("{name}.rotated");
        Ok(self.derive(format!("{derived}.weight"), vec![heads * head_dim, cols], laid).trim_end_matches(".weight").to_string())
    }

    /// A convolution's kernel `[out, in, kernel]` as one matrix over `kernel` rows side by
    /// side, padded with zero columns to `width` rows.
    pub(crate) fn kernel(&mut self, name: &str, kernel: usize, width: usize) -> Result<(String, usize), String> {
        let weight = self.floats(&format!("{name}.weight"))?;
        let shape = self.weights.shape(&format!("{name}.weight")).ok_or(format!("{name} is missing"))?.to_vec();
        let (outputs, inputs) = (shape[0], shape[1]);
        let mut laid = vec![0f32; outputs * width * inputs];
        for o in 0..outputs {
            for c in 0..inputs {
                for k in 0..kernel {
                    laid[(o * width + k) * inputs + c] = weight[(o * inputs + c) * kernel + k];
                }
            }
        }
        let derived = format!("{name}.rows");
        self.derive(format!("{derived}.weight"), vec![outputs, width * inputs], laid);
        Ok((derived, outputs))
    }
}

/// A convolution with GELU over `kernel` rows of `src`, `inputs` wide: a routine that reads
/// from the first row and leaves the result, and one that stores it in a store of `rows`.
fn strided(b: &mut Builder, name: &str, src: Buf, inputs: usize, kernel: usize, rows: usize) -> Result<(Buf, Pair), String> {
    let (matrix, outputs) = b.kernel(name, kernel, kernel)?;
    let near = b.buf(kernel * inputs);
    let result = b.buf(outputs);
    let dst = b.buf(rows * outputs);
    let read = b.routine(|b| {
        b.op("read rows", Op::LoadRows { store: src, dst: near, offsets: (0..kernel).collect() });
        let product = b.product(&matrix, near, outputs, Some(format!("{name}.bias")));
        b.op("GELU", Op::Gelu { src: product, approximate: false, dst: result });
        Ok(())
    })?;
    let store = b.routine(|b| {
        b.op("store", Op::StoreRow { src: result, store: dst });
        Ok(())
    })?;
    Ok((dst, (read, store)))
}

/// The routines of a decoder token.
#[derive(Clone, Debug)]
pub(crate) struct DecoderRoutines {
    pub(crate) embed: Range<usize>,
    /// Each layer's self-attention, at the token's position.
    pub(crate) own: Vec<Range<usize>>,
    /// Each layer's cross-attention and MLP, at the last frame's.
    pub(crate) cross: Vec<Range<usize>>,
    pub(crate) logits: Range<usize>,
}

/// Moonshine's decoder as routines: its layers' own keys in `own_slot(layer)`, the frames'
/// in `cross_slot(layer)`. The routines, the logits and the decoder's last output.
pub(crate) fn decoder_routines(b: &mut Builder, config: &MoonshineConfig, own_slot: &dyn Fn(usize) -> usize, cross_slot: &dyn Fn(usize) -> usize) -> Result<(DecoderRoutines, Buf, Buf), String> {
    let (h, heads, head_dim, rotary) = (config.hidden, config.heads, config.head_dim, config.rotary);
    let mut x = b.buf(h);
    let embed = b.routine(|b| {
        b.op("read embedding", Op::EmbRead { table: "model.decoder.embed_tokens".into(), dst: x });
        Ok(())
    })?;
    let (mut own, mut cross) = (Vec::new(), Vec::new());
    for layer in 0..config.decoder_layers {
        let l = |n: &str| format!("model.decoder.layers.{layer}.{n}");
        let (q_proj, k_proj) = (b.rotated(&l("self_attn.q_proj"), heads, head_dim, rotary)?, b.rotated(&l("self_attn.k_proj"), heads, head_dim, rotary)?);
        let mut after = 0;
        own.push(b.routine(|b| {
            let normed = b.layer_norm(&l("input_layernorm"), x, h);
            let q = b.product(&q_proj, normed, h, None);
            let k = b.product(&k_proj, normed, h, None);
            let v = b.linear(&l("self_attn.v_proj"), normed, h);
            b.op("RoPE on Q", Op::Rope { buf: q, block: head_dim, rotary, table: 0 });
            b.op("RoPE on K", Op::Rope { buf: k, block: head_dim, rotary, table: 0 });
            b.op("store key and value", Op::KvAppend { slot: own_slot(layer), key: k, value: v, window: None });
            let attended = b.buf(h);
            b.op("attention", Op::AttnSum { slot: own_slot(layer), query: q, head_dim, scale_dim: head_dim, heads, kv_heads: heads, window: None, softcap: None, dst: attended });
            let out = b.linear(&l("self_attn.o_proj"), attended, h);
            after = b.buf(h);
            b.op("residual", Op::Add { a: x, b: out, dst: after });
            Ok(())
        })?);
        let mut next = 0;
        cross.push(b.routine(|b| {
            let normed = b.layer_norm(&l("post_attention_layernorm"), after, h);
            let q = b.linear(&l("encoder_attn.q_proj"), normed, h);
            let attended = b.buf(h);
            b.op("cross-attention", Op::AttnSum { slot: cross_slot(layer), query: q, head_dim, scale_dim: head_dim, heads, kv_heads: heads, window: None, softcap: None, dst: attended });
            let out = b.linear(&l("encoder_attn.o_proj"), attended, h);
            let x1 = b.buf(h);
            b.op("residual", Op::Add { a: after, b: out, dst: x1 });
            let normed = b.layer_norm(&l("final_layernorm"), x1, h);
            let both = b.linear(&l("mlp.fc1"), normed, 2 * config.intermediate);
            let (value, gate) = (b.buf(config.intermediate), b.buf(config.intermediate));
            b.op("value half", Op::Slice { src: both, from: 0, dst: value });
            b.op("gate half", Op::Slice { src: both, from: config.intermediate, dst: gate });
            let bent = b.buf(config.intermediate);
            b.op("SiLU", Op::Silu { src: gate, dst: bent });
            let gated = b.buf(config.intermediate);
            b.op("gate × value", Op::MulElem { a: bent, b: value, dst: gated });
            let out = b.linear(&l("mlp.fc2"), gated, h);
            next = b.buf(h);
            b.op("residual", Op::Add { a: x1, b: out, dst: next });
            Ok(())
        })?);
        x = next;
    }
    let head = if b.weights.shape("proj_out.weight").is_some() { "proj_out" } else { "model.decoder.embed_tokens" };
    let mut logits = 0;
    let logits_routine = b.routine(|b| {
        let normed = b.layer_norm("model.decoder.norm", x, h);
        logits = b.product(head, normed, config.vocab, None);
        Ok(())
    })?;
    Ok((DecoderRoutines { embed, own, cross, logits: logits_routine }, logits, x))
}

/// How long a clip is, in samples, rows of samples, outputs of the first and second
/// convolutions, and frames.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Lengths {
    pub samples: usize,
    pub sample_rows: usize,
    pub first: usize,
    pub second: usize,
    pub frames: usize,
}

impl Lengths {
    pub fn of(samples: usize) -> Result<Self, String> {
        let frames = MoonshineConfig::frames(samples);
        if frames == 0 {
            return Err("the clip is too short for one frame".into());
        }
        let first = (samples - 127) / 64 + 1;
        Ok(Self { samples, sample_rows: first + 1, first, second: (first - 7) / 3 + 1, frames })
    }
}

/// Moonshine as the chip's program, for clips of up to some length, and what the host runs.
pub struct MoonshineChip {
    pub config: MoonshineConfig,
    program: Program,
    layout: Layout,
    vm: BlockVm,
    exponents: u32,
    bits: u32,
    weights: Arc<Weights>,
    /// The longest clip it takes, and the one it is taking.
    capacity: Lengths,
    now: Lengths,
    input: Buf,
    /// Where the host writes `1 / count` for the GroupNorm of the clip it is taking.
    inverse: Buf,
    store_samples: Range<usize>,
    conv1: Range<usize>,
    sums: Range<usize>,
    reduce: (Range<usize>, Range<usize>, Range<usize>),
    totals: Range<usize>,
    normalize: Range<usize>,
    conv2: Pair,
    conv3: Pair,
    blocks: Vec<Pair>,
    cross_keys: Range<usize>,
    decoder: DecoderRoutines,
    logits: Buf,
}

impl MoonshineChip {
    /// The checkpoint in `dir` for clips of up to `samples` samples, on a die like `die` with
    /// vectors of its activation bits; with the weights in its cells and caches for
    /// `tokens` places when `tokens` is given, as a chip holds them.
    pub fn new(dir: &Path, samples: usize, die: DieConfig, tokens: Option<usize>) -> Result<Self, String> {
        Self::from_parts(MoonshineConfig::read(dir)?, Weights::open(dir)?, samples, die, tokens)
    }

    /// The exact machine's die with vectors of `bits` bits and cells as wide.
    pub fn exact(dir: &Path, samples: usize, bits: u32) -> Result<Self, String> {
        let base = crate::exact::exact_config(bits);
        let die = DieConfig { cell: netlist::CellConfig { weight_bits: bits, accumulator_bits: 2 * bits + 8, ..base.cell }, ..base };
        Self::new(dir, samples, die, None)
    }

    pub fn from_parts(config: MoonshineConfig, weights: Weights, samples: usize, die: DieConfig, tokens: Option<usize>) -> Result<Self, String> {
        if config.rope_dim != config.rotary {
            return Err("RoPE over an odd number of values is not taken to the die".into());
        }
        let capacity = Lengths::of(samples)?;
        let Lengths { sample_rows, first, second, frames, .. } = capacity;
        let (h, heads, head_dim, rotary) = (config.hidden, config.heads, config.head_dim, config.rotary);
        let mut b = Builder { weights: &weights, buffers: Vec::new(), ops: Vec::new(), derived: Vec::new() };
        let p = |n: &str| format!("model.encoder.{n}");

        // The waveform, a row at a time.
        let input = b.buf(SAMPLES_PER_ROW);
        let wave = b.buf(sample_rows * SAMPLES_PER_ROW);
        let store_samples = b.routine(|b| {
            b.op("store samples", Op::StoreRow { src: input, store: wave });
            Ok(())
        })?;

        // conv1 with tanh, kept for the GroupNorm.
        let (kernel1, _) = b.kernel(&p("conv1"), 127, 2 * SAMPLES_PER_ROW)?;
        let a1 = b.buf(first * h);
        let conv1 = b.routine(|b| {
            let near = b.buf(2 * SAMPLES_PER_ROW);
            b.op("read samples", Op::LoadRows { store: wave, dst: near, offsets: vec![0, 1] });
            let product = b.product(&kernel1, near, h, None);
            let bent = b.buf(h);
            b.op("tanh", Op::SoftCap { src: product, cap: 1.0, dst: bent });
            b.op("store", Op::StoreRow { src: bent, store: a1 });
            Ok(())
        })?;

        // The GroupNorm's totals: each row's sum and sum of squares, then rows in pairs.
        let ones = b.derive("moonshine.ones.weight".into(), vec![1, h], vec![1.0; h]);
        let ones = ones.trim_end_matches(".weight").to_string();
        let (sum_store, square_store) = (b.buf(first), b.buf(first));
        let sums = b.routine(|b| {
            let x = b.buf(h);
            b.op("read", Op::LoadRow { store: a1, dst: x });
            let x2 = b.buf(h);
            b.op("squares", Op::MulElem { a: x, b: x, dst: x2 });
            let s = b.product(&ones, x, 1, None);
            let q = b.product(&ones, x2, 1, None);
            b.op("store sum", Op::StoreRow { src: s, store: sum_store });
            b.op("store squares", Op::StoreRow { src: q, store: square_store });
            Ok(())
        })?;
        let (a, a2, c, c2) = (b.buf(1), b.buf(1), b.buf(1), b.buf(1));
        let load = b.routine(|b| {
            b.op("read sum", Op::LoadRow { store: sum_store, dst: a });
            b.op("read squares", Op::LoadRow { store: square_store, dst: a2 });
            Ok(())
        })?;
        let load_second = b.routine(|b| {
            b.op("read sum", Op::LoadRow { store: sum_store, dst: c });
            b.op("read squares", Op::LoadRow { store: square_store, dst: c2 });
            Ok(())
        })?;
        let add = b.routine(|b| {
            let (e, e2) = (b.buf(1), b.buf(1));
            b.op("add", Op::Add { a, b: c, dst: e });
            b.op("add", Op::Add { a: a2, b: c2, dst: e2 });
            b.op("store sum", Op::StoreRow { src: e, store: sum_store });
            b.op("store squares", Op::StoreRow { src: e2, store: square_store });
            Ok(())
        })?;
        // The totals over this clip's values: times `1 / count`, which the host writes.
        let inverse = b.buf(1);
        let (total, total_squares, mean, mean_square) = (b.buf(1), b.buf(1), b.buf(1), b.buf(1));
        let totals = b.routine(|b| {
            b.op("read totals", Op::LoadRow { store: sum_store, dst: total });
            b.op("read totals", Op::LoadRow { store: square_store, dst: total_squares });
            b.op("mean", Op::MulElem { a: total, b: inverse, dst: mean });
            b.op("mean of squares", Op::MulElem { a: total_squares, b: inverse, dst: mean_square });
            Ok(())
        })?;
        let normalize = b.routine(|b| {
            let x = b.buf(h);
            b.op("read", Op::LoadRow { store: a1, dst: x });
            let y = b.buf(h);
            b.op("GroupNorm", Op::GroupNorm { src: x, sums: mean, squares: mean_square, count: 1, weight: p("groupnorm"), groups: 1, dst: y });
            b.op("store", Op::StoreRow { src: y, store: a1 });
            Ok(())
        })?;

        // conv2 and conv3 with GELU: one routine reads from the first row, one stores.
        let (a2s, conv2) = strided(&mut b, &p("conv2"), a1, h, 7, second)?;
        let (x_store, conv3) = strided(&mut b, &p("conv3"), a2s, 2 * h, 3, frames)?;

        // The encoder's blocks, in two passes each, over one key slot they all share.
        let queries = b.buf(frames * h);
        let mut blocks = Vec::new();
        for layer in 0..config.encoder_layers {
            let l = |n: &str| format!("model.encoder.layers.{layer}.{n}");
            let (q_proj, k_proj) = (b.rotated(&l("self_attn.q_proj"), heads, head_dim, rotary)?, b.rotated(&l("self_attn.k_proj"), heads, head_dim, rotary)?);
            let project = b.routine(|b| {
                let x = b.buf(h);
                b.op("read frame", Op::LoadRow { store: x_store, dst: x });
                let normed = b.layer_norm(&l("input_layernorm"), x, h);
                let q = b.product(&q_proj, normed, h, None);
                let k = b.product(&k_proj, normed, h, None);
                let v = b.linear(&l("self_attn.v_proj"), normed, h);
                b.op("RoPE on Q", Op::Rope { buf: q, block: head_dim, rotary, table: 0 });
                b.op("RoPE on K", Op::Rope { buf: k, block: head_dim, rotary, table: 0 });
                b.op("store key and value", Op::KvAppend { slot: 0, key: k, value: v, window: None });
                b.op("store query", Op::StoreRow { src: q, store: queries });
                Ok(())
            })?;
            let rest = b.routine(|b| {
                let q = b.buf(h);
                b.op("read query", Op::LoadRow { store: queries, dst: q });
                let attended = b.buf(h);
                b.op("attention", Op::AttnSum { slot: 0, query: q, head_dim, scale_dim: head_dim, heads, kv_heads: heads, window: None, softcap: None, dst: attended });
                let out = b.linear(&l("self_attn.o_proj"), attended, h);
                let x = b.buf(h);
                b.op("read frame", Op::LoadRow { store: x_store, dst: x });
                let x1 = b.buf(h);
                b.op("residual", Op::Add { a: x, b: out, dst: x1 });
                let normed = b.layer_norm(&l("post_attention_layernorm"), x1, h);
                let inner = b.linear(&l("mlp.fc1"), normed, config.intermediate);
                let bent = b.buf(config.intermediate);
                b.op("GELU", Op::Gelu { src: inner, approximate: false, dst: bent });
                let out = b.linear(&l("mlp.fc2"), bent, h);
                let x2 = b.buf(h);
                b.op("residual", Op::Add { a: x1, b: out, dst: x2 });
                b.op("store frame", Op::StoreRow { src: x2, store: x_store });
                Ok(())
            })?;
            blocks.push((project, rest));
        }

        // Every frame's key and value of each decoder layer's cross-attention.
        let cross_slot = |layer: usize| 1 + layer;
        let own_slot = |layer: usize| 1 + config.decoder_layers + layer;
        let cross_keys = b.routine(|b| {
            let x = b.buf(h);
            b.op("read frame", Op::LoadRow { store: x_store, dst: x });
            let frame = b.layer_norm(&p("layer_norm"), x, h);
            for layer in 0..config.decoder_layers {
                let l = |n: &str| format!("model.decoder.layers.{layer}.encoder_attn.{n}");
                let k = b.linear(&l("k_proj"), frame, h);
                let v = b.linear(&l("v_proj"), frame, h);
                b.op("store key and value", Op::KvAppend { slot: cross_slot(layer), key: k, value: v, window: None });
            }
            Ok(())
        })?;

        // A decoder token.
        let (decoder, logits, x) = decoder_routines(&mut b, &config, &own_slot, &cross_slot)?;

        let Builder { buffers, ops, derived, .. } = b;
        let reach = [sample_rows, first, second, frames, config.max_tokens].into_iter().max().unwrap_or(1);
        let mut kv_slots = vec![KvShape { head_dim, heads }];
        kv_slots.extend((0..2 * config.decoder_layers).map(|_| KvShape { head_dim, heads }));
        let program = Program {
            family: "moonshine".into(),
            vocab: config.vocab,
            hidden: h,
            num_layers: 0,
            buffers,
            ops: Vec::new(),
            logits_from: 0,
            logits,
            kv_slots,
            delta_slots: Vec::new(),
            conv_slots: Vec::new(),
            rope_theta: config.rope_theta,
            rope_scaling: None,
            local_rope_theta: None,
            rms_epsilon: 1e-5,
            final_hidden: x,
            embedding_rows: None,
            draft: None,
            parts: vec![Part::Rows(RowPart { ops, embed: 0..0, blocks: Vec::new(), merge: 0..0, pixels: input, position: input, output: logits, head_dim, patches: reach, extra: Vec::new(), tables: Vec::new() })],
        };
        let weights = Arc::new(weights.with_floats(derived)?);
        let layout = match tokens {
            Some(tokens) => Layout::with_cells(&program, &weights, die, tokens)?,
            None => Layout::new(&program, &weights, die, None)?,
        };
        let mut vm = BlockVm::new(&die, layout.registers_used() as usize + 8, Some(weights.clone()));
        vm.compute_in_cells();
        Ok(Self {
            config,
            program,
            layout,
            vm,
            exponents: exponent_register(&die),
            bits: die.cell.activation_bits,
            weights,
            capacity,
            now: capacity,
            input,
            inverse,
            store_samples,
            conv1,
            sums,
            reduce: (load, load_second, add),
            totals,
            normalize,
            conv2,
            conv3,
            blocks,
            cross_keys,
            decoder,
            logits,
        })
    }

    /// Samples in the longest clip it takes.
    pub fn samples(&self) -> usize {
        self.capacity.samples
    }

    /// The frames of the clip it is taking.
    pub fn frames(&self) -> usize {
        self.now.frames
    }

    /// Takes a clip of `samples` samples from now on: what the host runs follows its length.
    pub fn clip(&mut self, samples: usize) -> Result<(), String> {
        if samples > self.capacity.samples {
            return Err(format!("this chip takes clips of up to {} samples", self.capacity.samples));
        }
        self.now = Lengths::of(samples)?;
        Ok(())
    }

    /// What the host writes before the encoder runs: `1 / count` of this clip's GroupNorm.
    pub fn inverse_write(&self) -> (u32, Vector) {
        let count = (self.now.first * self.config.hidden) as f64;
        (self.layout.register_of(self.inverse), encode_vector(&[1.0 / count], self.bits))
    }

    /// The program, its layout and its weights, to run it elsewhere, such as on gates.
    pub fn parts(&self) -> (&Program, &Layout, &Weights) {
        (&self.program, &self.layout, &self.weights)
    }

    pub fn logits_register(&self) -> u32 {
        self.layout.register_of(self.logits)
    }

    /// Where the host writes a row of samples, and the rows in the die's format: the
    /// clip padded with zeros to whole rows.
    pub fn sample_rows(&self, samples: &[f32]) -> Result<(u32, Vec<Vector>), String> {
        if samples.len() != self.now.samples {
            return Err(format!("the clip has {} samples and the chip takes {} now: call clip first", samples.len(), self.now.samples));
        }
        let rows = (0..self.now.sample_rows)
            .map(|r| {
                let row: Vec<f64> = (0..SAMPLES_PER_ROW).map(|i| samples.get(r * SAMPLES_PER_ROW + i).copied().unwrap_or(0.0) as f64).collect();
                encode_vector(&row, self.bits)
            })
            .collect();
        Ok((self.layout.register_of(self.input), rows))
    }

    fn call(routine: &Range<usize>, token: usize, position: usize, rope: bool) -> Call {
        Call { routine: routine.clone(), token: token as u32, position: position as u32, rope, read: false }
    }

    /// The routine that stores each row of samples, run after the host writes it.
    pub fn store_samples_call(&self, row: usize) -> Call {
        Self::call(&self.store_samples, row, 0, false)
    }

    /// Everything the encoder runs once the samples are in, up to every frame's keys and
    /// values for the decoder's cross-attention.
    pub fn encoder_calls(&self) -> Vec<Call> {
        let mut calls = Vec::new();
        let Lengths { first, second, frames, .. } = self.now;
        let last = frames - 1;
        for t in 0..first {
            calls.push(Self::call(&self.conv1, t, 0, false));
            calls.push(Self::call(&self.sums, t, 0, false));
        }
        let (load, load_second, add) = &self.reduce;
        let mut count = first;
        while count > 1 {
            let half = count / 2;
            for i in 0..half {
                calls.push(Self::call(load, i, 0, false));
                calls.push(Self::call(load_second, i + half, 0, false));
                calls.push(Self::call(add, i, 0, false));
            }
            if count % 2 == 1 {
                calls.push(Self::call(load, 0, 0, false));
                calls.push(Self::call(load_second, count - 1, 0, false));
                calls.push(Self::call(add, 0, 0, false));
            }
            count = half;
        }
        calls.push(Self::call(&self.totals, 0, 0, false));
        for t in 0..first {
            calls.push(Self::call(&self.normalize, t, 0, false));
        }
        for u in 0..second {
            calls.push(Self::call(&self.conv2.0, 3 * u, 0, false));
            calls.push(Self::call(&self.conv2.1, u, 0, false));
        }
        for t in 0..frames {
            calls.push(Self::call(&self.conv3.0, 2 * t, 0, false));
            calls.push(Self::call(&self.conv3.1, t, 0, false));
        }
        for (project, rest) in &self.blocks {
            for t in 0..frames {
                calls.push(Self::call(project, t, t, true));
            }
            for t in 0..frames {
                calls.push(Self::call(rest, t, last, false));
            }
        }
        for t in 0..frames {
            calls.push(Self::call(&self.cross_keys, t, t, false));
        }
        calls
    }

    /// What a decoder token runs: `token` at `position`, the logits read at the end.
    pub fn token_calls(&self, token: u32, position: usize) -> Vec<Call> {
        let d = &self.decoder;
        let mut calls = vec![Self::call(&d.embed, token as usize, position, false)];
        for (own, cross) in d.own.iter().zip(&d.cross) {
            calls.push(Self::call(own, 0, position, true));
            calls.push(Self::call(cross, 0, self.now.frames - 1, false));
        }
        calls.push(Call { read: true, ..Self::call(&d.logits, 0, position, false) });
        calls
    }

    /// The routines `calls` run, each once, and the microprogram with an entry for each,
    /// as a chip's firmware would have it.
    pub fn microprogram(&self, calls: &[Call]) -> Result<(Vec<Routine>, netlist::Microprogram), String> {
        let mut routines: Vec<Routine> = Vec::new();
        for call in calls {
            if !routines.iter().any(|(r, _)| *r == call.routine) {
                routines.push((call.routine.clone(), call.rope));
            }
        }
        let part = self.program.rows().ok_or("the routines are missing")?;
        let lists: Vec<&[Inst]> = routines.iter().map(|(r, _)| &part.ops[r.clone()]).collect();
        let code = self.layout.routines_microprogram(&self.program, &lists.iter().zip(&routines).map(|(l, (_, rope))| (*l, *rope)).collect::<Vec<_>>())?;
        Ok((routines, code))
    }

    /// The die's cycles for `calls`, counted from the microprogram.
    pub fn cycles(&self, calls: &[Call]) -> Result<u64, String> {
        let (routines, code) = self.microprogram(calls)?;
        let config = self.layout.config();
        Ok(calls.iter().map(|call| code.cycles(&config, routines.iter().position(|(r, _)| *r == call.routine).unwrap(), call.token, call.position)).sum())
    }

    /// This recogniser as a stage of a system of `dies` dies of `die_mm2` at `clock_hz`:
    /// a clip's samples written and encoded, then its first token; then each token, at the
    /// middle of the text it can write.
    pub fn stage(&self, dies: u32, die_mm2: f64, clock_hz: f64) -> Result<crate::system::Stage, String> {
        let input: Vec<Call> = (0..self.now.sample_rows).map(|r| self.store_samples_call(r)).collect();
        let first = self.cycles(&[input, self.encoder_calls(), self.token_calls(self.config.start, 0)].concat())?;
        let middle = self.limit() / 2;
        let each = self.cycles(&self.token_calls(self.config.start, middle))?;
        Ok(crate::system::Stage { name: "speech to text".into(), dies, area_mm2: dies as f64 * die_mm2, clock_hz, first_cycles: first, output_cycles: each })
    }

    /// Every routine the chip runs, with the microprogram that has an entry for each in
    /// that order, and the schedule its firmware follows to transcribe a clip by itself: it
    /// receives the clip's length in samples, `1 / count` for the GroupNorm and the rows of
    /// samples (see `schedule_input`), works out the lengths, runs the encoder and writes
    /// the tokens greedily, as `transcribe`, ending at the end token.
    pub fn schedule(&self) -> Result<(Vec<Routine>, netlist::Microprogram, Vec<Record>), String> {
        let all = [vec![self.store_samples_call(0)], self.encoder_calls(), self.token_calls(self.config.start, 0)].concat();
        let (routines, code) = self.microprogram(&all)?;
        let index = |r: &Range<usize>| routines.iter().position(|(x, _)| x == r).map(|i| i as u32).ok_or("a routine the schedule needs is not in the microprogram");
        let config = self.layout.config();
        let (ib, exponents) = (config.index_bits, self.exponents);
        let vector = |buf: Buf| {
            let register = self.layout.register_of(buf);
            (register << ib, exponents << ib | register)
        };
        // The variables after the firmware's own.
        let [samples, rows, first, second, frames, last, limit] = std::array::from_fn(|i| FIRST_FREE + i as u32);
        let at = |var: u32| (0, 0, var);
        let i = |times: u32| (times, 0, ZERO);
        let call = |r: &Range<usize>, token: (u32, u32, u32), position: (u32, u32, u32)| -> Result<Record, String> { Ok(Record::Call { routine: index(r)?, token, position }) };
        let repeat = |count: u32, body: Vec<Record>| [vec![Record::Repeat { body: body.len() as u32, count, add: 0 }], body].concat();
        let compute = |dst, src, mul, pre, div, post| Record::Compute { dst, src, mul, pre, div, post, max: i32::MAX };
        let mut s = vec![
            Record::Receive { var: samples },
            // As `Lengths::of`.
            compute(first, samples, 1, -127, 64, 1),
            compute(rows, first, 1, 0, 1, 1),
            compute(second, first, 1, -7, 3, 1),
            compute(frames, second, 1, -3, 2, 1),
            compute(last, frames, 1, -1, 1, 0),
            // As `limit`.
            Record::Compute { dst: limit, src: samples, mul: 6, pre: 0, div: 16_000, post: 1, max: self.config.max_tokens as i32 - 1 },
        ];
        let (address, exponent) = vector(self.inverse);
        s.push(Record::Vectors { count: ZERO, add: 1, address, exponent, size: 1, routine: NO_ROUTINE });
        let (address, exponent) = vector(self.input);
        s.push(Record::Vectors { count: rows, add: 0, address, exponent, size: SAMPLES_PER_ROW as u32, routine: index(&self.store_samples)? });
        // As `encoder_calls`.
        s.extend(repeat(first, vec![call(&self.conv1, i(1), at(ZERO))?, call(&self.sums, i(1), at(ZERO))?]));
        let (load, second_load, add) = &self.reduce;
        s.push(Record::Halve { count: first, load: index(load)?, second: index(second_load)?, add: index(add)? });
        s.push(call(&self.totals, at(ZERO), at(ZERO))?);
        s.extend(repeat(first, vec![call(&self.normalize, i(1), at(ZERO))?]));
        s.extend(repeat(second, vec![call(&self.conv2.0, i(3), at(ZERO))?, call(&self.conv2.1, i(1), at(ZERO))?]));
        s.extend(repeat(frames, vec![call(&self.conv3.0, i(2), at(ZERO))?, call(&self.conv3.1, i(1), at(ZERO))?]));
        for (project, rest) in &self.blocks {
            s.extend(repeat(frames, vec![call(project, i(1), i(1))?]));
            s.extend(repeat(frames, vec![call(rest, i(1), at(last))?]));
        }
        s.extend(repeat(frames, vec![call(&self.cross_keys, i(1), i(1))?]));
        // As `token_calls`, a token at a time.
        let d = &self.decoder;
        let mut body = vec![call(&d.embed, at(TOKEN), at(POSITION))?];
        for (own, cross) in d.own.iter().zip(&d.cross) {
            body.push(call(own, at(ZERO), at(POSITION))?);
            body.push(call(cross, at(ZERO), at(last))?);
        }
        body.push(call(&d.logits, at(ZERO), at(POSITION))?);
        s.push(Record::Generate { body: body.len() as u32, limit, start: self.config.start });
        s.extend(body);
        Ok((routines, code, s))
    }

    /// The control CPU's cycles beyond the die's to follow `schedule` for the clip it is
    /// taking (`crate::chip::firmware_cycles`): up to its first token, and for each token
    /// after. The rows and the pairwise sums start their routines alone, the encoder's
    /// repeats on the lean path, and each token's routines alone.
    pub fn firmware_cycles(&self) -> Result<(u64, u64), String> {
        let (load, second, add) = &self.reduce;
        let encoder = self.encoder_calls();
        let (summing, lean): (Vec<Call>, Vec<Call>) = encoder.into_iter().partition(|c| [load, second, add].contains(&&c.routine));
        let rows = self.now.sample_rows as u64;
        let words = 1 + 2 + rows * (SAMPLES_PER_ROW as u64 + 1);
        let token = self.token_calls(self.config.start, 0).len() as u64;
        let each = crate::chip::firmware_cycles(0, token, 1);
        let (_, code, _) = self.schedule()?;
        let config = self.layout.config();
        let (queued, parameters) = (crate::chip::queued(&config, &code), code.parameters.len() as u64);
        let lean = lean.iter().map(|call| self.cycles(std::slice::from_ref(call)).map(|die| crate::chip::lean_cycles(1, die, parameters, queued))).sum::<Result<u64, String>>()?;
        Ok((lean + crate::chip::firmware_cycles(0, rows + summing.len() as u64, words) + each, each))
    }

    /// What the host sends a chip that follows `schedule`, as words: the clip's length, then
    /// each vector's exponent and elements, `1 / count` first and then the rows of samples.
    pub fn schedule_input(&mut self, samples: &[f32]) -> Result<Vec<u32>, String> {
        self.clip(samples.len())?;
        let (_, inverse) = self.inverse_write();
        let (_, rows) = self.sample_rows(samples)?;
        let mut words = vec![samples.len() as u32];
        for (mantissas, exponent) in std::iter::once(&inverse).chain(&rows) {
            words.push(*exponent as u32);
            words.extend(mantissas.iter().map(|m| *m as u32));
        }
        Ok(words)
    }

    /// The die as its firmware sees it, run by the exact machine: `routines` numbered as in
    /// the microprogram `code`.
    pub fn die_engine(&mut self, routines: &[Routine], code: netlist::Microprogram) -> RoutinesDie<'_> {
        self.reset();
        let part = self.program.rows().expect("the routines");
        let lists = routines.iter().map(|(r, rope)| (part.ops[r.clone()].to_vec(), *rope)).collect();
        RoutinesDie::new(&self.layout, &self.program, &mut self.vm, lists, code)
    }

    /// The most tokens it writes for a clip: six a second, as `Moonshine::transcribe`.
    pub fn limit(&self) -> usize {
        (self.now.samples * 6 / 16_000 + 1).min(self.config.max_tokens - 1)
    }

    fn reset(&mut self) {
        self.vm.clear();
        crate::chip::load_initial(&mut self.vm, &self.layout.setup().initial, self.exponents);
    }

    fn run(&mut self, call: &Call) -> Result<(), String> {
        let part = self.program.rows().expect("the routines");
        let list = part.ops[call.routine.clone()].to_vec();
        self.layout.run_ops(&mut self.vm, &self.program, &list, call.token, call.position, call.rope)
    }

    /// Writes `samples` in and runs the encoder.
    pub fn encode(&mut self, samples: &[f32]) -> Result<(), String> {
        self.clip(samples.len())?;
        let (register, rows) = self.sample_rows(samples)?;
        self.reset();
        let (inverse, (mantissas, exponent)) = self.inverse_write();
        self.vm.load_mantissas(inverse, &mantissas, exponent);
        for (row, (mantissas, exponent)) in rows.iter().enumerate() {
            self.vm.load_mantissas(register, mantissas, *exponent);
            self.run(&self.store_samples_call(row))?;
        }
        for call in self.encoder_calls() {
            self.run(&call)?;
        }
        Ok(())
    }

    /// The scores of the token after `token`, at `position`, as values and as the die's
    /// mantissas.
    pub fn step(&mut self, token: u32, position: usize) -> Result<(Vec<f32>, Vec<i64>), String> {
        for call in self.token_calls(token, position) {
            self.run(&call)?;
        }
        let register = self.logits_register();
        let mantissas = (0..self.config.vocab as u32).map(|i| self.vm.mantissa(register, i)).collect();
        Ok((self.vm.floats(register, self.config.vocab), mantissas))
    }

    /// The text of `samples`, greedily, as `Moonshine::transcribe` writes it.
    pub fn transcribe(&mut self, samples: &[f32]) -> Result<Vec<u32>, String> {
        self.encode(samples)?;
        let mut token = self.config.start;
        let mut text = Vec::new();
        for position in 0..self.limit() {
            token = argmax(&self.step(token, position)?.0);
            if token == self.config.end {
                break;
            }
            text.push(token);
        }
        Ok(text)
    }
}

