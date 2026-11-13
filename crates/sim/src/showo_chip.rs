//! Show-o's transformer in the die's arithmetic: Phi-1.5 with Show-o's changes, as the
//! chip's operations, run by the exact machine in the order the host gives them.
//!
//! - **The text** goes in as ordinary tokens, each seeing those before it.
//! - **The image** sees itself whole, in both directions, so it goes through each layer in
//!   two passes over its rows, as the image encoder does (`compiler::VisionPart`):
//!   1. each row stores its key and value in the layer's cache, after the text's, and its
//!      query;
//!   2. each row attends over the text and every row of the image, then runs the rest of
//!      the layer: attention and MLP side by side on the same normalised input.
//! - **The codes' scores** come from the output's rows of the codes and their biases.
//!
//! MAGVIT-v2, which turns pictures into codes and back, stays on the host in floating point.

use crate::lower::Layout;
use crate::showo::{fill_with, Magvit, ShowOConfig};
use crate::vm::BlockVm;
use crate::weights::Weights;
use compiler::{Buf, Inst, KvShape, Op, Program, VisionPart};
use netlist::block::exponent_register;
use netlist::DieConfig;
use std::path::Path;
use std::sync::Arc;

/// The shape of Show-o's transformer.
#[derive(Clone, Copy, Debug)]
pub struct Shape {
    pub hidden: usize,
    pub heads: usize,
    pub layers: usize,
    pub intermediate: usize,
    /// Rows of the output, the text's tokens and Show-o's with them.
    pub vocab: usize,
    /// Elements of each head that rotate with the position.
    pub rotary: usize,
    /// The first code's row of the output, and how many codes.
    pub first_code: usize,
    pub codebook: usize,
    /// Rows of an image: its start, its codes and its end.
    pub image_rows: usize,
}

impl Shape {
    /// From Show-o's folder: its configuration, Phi's in `phi`, and its weights' shapes.
    pub fn read(dir: &Path, weights: &Weights) -> Result<(Self, ShowOConfig), String> {
        let config = ShowOConfig::read(dir)?.ok_or("esta carpeta no tiene Show-o")?;
        let shape = |name: &str| weights.shape(name).map(<[usize]>::to_vec).ok_or_else(|| format!("falta {name}"));
        let embed = shape("showo.model.embed_tokens.weight")?;
        let up = shape("showo.model.layers.0.mlp.fc1.weight")?;
        let layers = (0..).take_while(|i| weights.shape(&format!("showo.model.layers.{i}.input_layernorm.weight")).is_some()).count();
        let head = embed[1] / config.heads;
        Ok((
            Self {
                hidden: embed[1],
                heads: config.heads,
                layers,
                intermediate: up[0],
                vocab: shape("showo.lm_head.weight")?[0],
                rotary: (head as f64 * config.rotary_fraction).round() as usize,
                first_code: config.first_code() as usize,
                codebook: config.codebook as usize,
                image_rows: config.image_tokens + 2,
            },
            config,
        ))
    }

    fn head_dim(&self) -> usize {
        self.hidden / self.heads
    }
}

struct Emit {
    buffers: Vec<usize>,
    ops: Vec<Inst>,
    layer: Option<u32>,
}

impl Emit {
    fn buf(&mut self, len: usize) -> Buf {
        self.buffers.push(len);
        self.buffers.len() - 1
    }

    fn op(&mut self, name: &'static str, op: Op) {
        self.ops.push(Inst { op, layer: self.layer, name });
    }

    /// `x · W` plus its bias.
    fn linear(&mut self, name: &'static str, matrix: String, src: Buf, rows: usize) -> Buf {
        let (product, biased) = (self.buf(rows), self.buf(rows));
        self.op(name, Op::Matvec { matrix: matrix.clone(), src, dst: product });
        self.op("sesgo", Op::AddWeight { src: product, weight: format!("{matrix}.bias"), dst: biased });
        biased
    }

    fn norm(&mut self, name: &'static str, src: Buf, weight: String, block: usize) -> Buf {
        let dst = self.buf(self.buffers[src]);
        self.op(name, Op::LayerNorm { src, weight, block, dst });
        dst
    }
}

fn layer_name(layer: usize, name: &str) -> String {
    format!("showo.model.layers.{layer}.{name}")
}

/// The norm, queries, keys and values of a layer, the first two normalised per head and
/// rotated: `(normalised input, query, key, value)`.
fn attention_inputs(e: &mut Emit, s: &Shape, layer: usize, x: Buf) -> (Buf, Buf, Buf, Buf) {
    let p = |name: &str| layer_name(layer, name);
    let normed = e.norm("norma de entrada", x, p("input_layernorm"), s.hidden);
    let q = e.linear("consulta", p("self_attn.q_proj"), normed, s.hidden);
    let k = e.linear("clave", p("self_attn.k_proj"), normed, s.hidden);
    let v = e.linear("valor", p("self_attn.v_proj"), normed, s.hidden);
    let q = e.norm("norma de la consulta", q, p("self_attn.q_layernorm"), s.head_dim());
    let k = e.norm("norma de la clave", k, p("self_attn.k_layernorm"), s.head_dim());
    e.op("RoPE de la consulta", Op::Rope { buf: q, block: s.head_dim(), rotary: s.rotary, table: 0 });
    e.op("RoPE de la clave", Op::Rope { buf: k, block: s.head_dim(), rotary: s.rotary, table: 0 });
    (normed, q, k, v)
}

/// The rest of a layer once its query is at hand: attention over the cache, its projection,
/// the MLP on the same normalised input, and both added to the input. Gives the output.
fn attention_and_mlp(e: &mut Emit, s: &Shape, layer: usize, x: Buf, normed: Buf, query: Buf) -> Buf {
    let p = |name: &str| layer_name(layer, name);
    let attended = e.buf(s.hidden);
    let head_dim = s.head_dim();
    e.op("atención", Op::AttnSum { slot: layer, query, head_dim, scale_dim: head_dim, heads: s.heads, kv_heads: s.heads, window: None, softcap: None, dst: attended });
    let projected = e.linear("proyección", p("self_attn.dense"), attended, s.hidden);
    let up = e.linear("MLP, subida", p("mlp.fc1"), normed, s.intermediate);
    let activated = e.buf(s.intermediate);
    e.op("GELU", Op::Gelu { src: up, approximate: true, dst: activated });
    let down = e.linear("MLP, bajada", p("mlp.fc2"), activated, s.hidden);
    let (with_attention, out) = (e.buf(s.hidden), e.buf(s.hidden));
    e.op("residual", Op::Add { a: x, b: projected, dst: with_attention });
    e.op("residual", Op::Add { a: with_attention, b: down, dst: out });
    out
}

/// The final norm and the scores of the codes.
fn code_scores(e: &mut Emit, s: &Shape, x: Buf) -> (Buf, Buf) {
    let normed = e.norm("norma final", x, "showo.model.final_layernorm".into(), s.hidden);
    let all = e.linear("salida", "showo.lm_head".into(), normed, s.vocab);
    let codes = e.buf(s.codebook);
    e.op("puntuaciones de los códigos", Op::Slice { src: all, from: s.first_code, dst: codes });
    (normed, codes)
}

/// Show-o as the chip's program: the text's token in `ops`, the image's routines in
/// `vision`. In the image's part, `embed` reads a token's row (run with the token) and
/// `merge` stores it in the image's rows (run with the row); each block's two routines and
/// the last one, which scores the codes, take the row.
pub fn program(s: &Shape) -> Program {
    let mut e = Emit { buffers: Vec::new(), ops: Vec::new(), layer: None };
    let mut x = e.buf(s.hidden);
    e.op("leer embedding", Op::EmbRead { table: "showo.model.embed_tokens".into(), dst: x });
    for layer in 0..s.layers {
        e.layer = Some(layer as u32);
        let (normed, q, k, v) = attention_inputs(&mut e, s, layer, x);
        e.op("guardar clave y valor", Op::KvAppend { slot: layer, key: k, value: v, window: None });
        x = attention_and_mlp(&mut e, s, layer, x, normed, q);
    }
    e.layer = None;
    let logits_from = e.ops.len();
    let (final_hidden, logits) = code_scores(&mut e, s, x);
    let text_ops = std::mem::take(&mut e.ops);

    // The image: its rows' states and queries live in stores between the passes.
    let rows = s.image_rows;
    let (state, queries) = (e.buf(rows * s.hidden), e.buf(rows * s.hidden));
    let row = e.buf(s.hidden);
    e.op("leer embedding", Op::EmbRead { table: "showo.model.embed_tokens".into(), dst: row });
    let embed = 0..e.ops.len();
    let start = e.ops.len();
    e.op("guardar fila", Op::StoreRow { src: row, store: state });
    let store = start..e.ops.len();
    let mut blocks = Vec::new();
    for layer in 0..s.layers {
        e.layer = Some(layer as u32);
        let first = e.ops.len();
        let h = e.buf(s.hidden);
        e.op("leer fila", Op::LoadRow { store: state, dst: h });
        let (_, q, k, v) = attention_inputs(&mut e, s, layer, h);
        e.op("guardar clave y valor", Op::KvAppend { slot: layer, key: k, value: v, window: None });
        e.op("guardar consulta", Op::StoreRow { src: q, store: queries });
        let middle = e.ops.len();
        let (q, h) = (e.buf(s.hidden), e.buf(s.hidden));
        e.op("leer consulta", Op::LoadRow { store: queries, dst: q });
        e.op("leer fila", Op::LoadRow { store: state, dst: h });
        // The norm again, rather than another store: it costs little next to the products.
        let normed = e.norm("norma de entrada", h, layer_name(layer, "input_layernorm"), s.hidden);
        let out = attention_and_mlp(&mut e, s, layer, h, normed, q);
        e.op("guardar fila", Op::StoreRow { src: out, store: state });
        blocks.push((first..middle, middle..e.ops.len()));
    }
    e.layer = None;
    let h = e.buf(s.hidden);
    e.op("leer fila", Op::LoadRow { store: state, dst: h });
    let (_, scores) = code_scores(&mut e, s, h);
    let image_ops = std::mem::take(&mut e.ops);
    let unused = e.buf(1);

    Program {
        family: "showo".into(),
        vocab: s.codebook,
        hidden: s.hidden,
        num_layers: s.layers,
        buffers: e.buffers,
        ops: text_ops,
        logits_from,
        logits,
        kv_slots: (0..s.layers).map(|_| KvShape { head_dim: s.head_dim(), heads: s.heads }).collect(),
        delta_slots: Vec::new(),
        conv_slots: Vec::new(),
        rope_theta: 10_000.0,
        rope_scaling: None,
        local_rope_theta: None,
        rms_epsilon: 1e-5,
        final_hidden,
        draft: None,
        image: None,
        // The routine that scores a row's codes is the last, after the blocks.
        vision: Some(VisionPart { ops: image_ops, embed, blocks, merge: store, pixels: unused, position: unused, output: scores, head_dim: s.head_dim(), patches: rows }),
    }
}

/// Show-o's transformer on the exact machine.
pub struct ShowOChip {
    pub shape: Shape,
    pub config: ShowOConfig,
    program: Program,
    layout: Layout,
    vm: BlockVm,
    exponents: u32,
}

impl ShowOChip {
    /// From Show-o's folder, on a die like `die` with vectors of its activation bits.
    pub fn open(dir: &Path, weights: Weights, die: DieConfig) -> Result<Self, String> {
        let (shape, config) = Shape::read(dir, &weights)?;
        let program = program(&shape);
        let weights = Arc::new(weights);
        let layout = Layout::new(&program, &weights, die, None)?;
        let mut vm = BlockVm::new(&die, layout.registers_used() as usize + 8, Some(weights));
        vm.compute_in_cells();
        Ok(Self { shape, config, program, layout, vm, exponents: exponent_register(&die) })
    }

    /// On the exact machine's large die, with vectors of `bits` bits and cells as wide, so
    /// the weights, in floating point, go in whole.
    pub fn exact(dir: &Path, bits: u32) -> Result<Self, String> {
        let base = crate::exact::exact_config(bits);
        let die = DieConfig { cell: netlist::CellConfig { weight_bits: bits, accumulator_bits: 2 * bits + 8, ..base.cell }, ..base };
        Self::open(dir, Weights::open(dir)?, die)
    }

    /// On the exact machine's die, with vectors of `bits` bits and the weights in `format`,
    /// as the design keeps them in its cells.
    pub fn with_format(dir: &Path, bits: u32, format: netlist::format::WeightFormat) -> Result<Self, String> {
        Self::open(dir, Weights::open(dir)?.with_format(Some(format)), crate::exact::exact_config(bits))
    }

    /// The same as `ShowO::scores`, in the die's arithmetic: each code's score at every
    /// place of the image (the codes' rows of the output, a place after another), after
    /// `text`. The image is the model's tokens, a code's or the mask.
    pub fn scores(&mut self, text_ids: &[u32], text_positions: &[u64], image: &[u32]) -> Result<Vec<f32>, String> {
        let part = self.program.vision.clone().ok_or("falta la parte de la imagen")?;
        if image.len() + 2 != self.shape.image_rows {
            return Err(format!("una imagen de Show-o tiene {} códigos", self.shape.image_rows - 2));
        }
        self.vm.clear();
        crate::chip::load_initial(&mut self.vm, &self.layout.setup().initial, self.exponents);
        self.layout.clear_rotations();
        // The text, a token at a time; the padding before it is left out, as nothing reads it.
        for (slot, (id, position)) in text_ids.iter().zip(text_positions).enumerate() {
            self.layout.place_rotation(&self.program, slot as u32, [*position; 3], None)?;
            self.layout.run_token(&mut self.vm, &self.program, *id, slot as u32, false)?;
        }
        let text = text_ids.len() as u32;
        let first_position = text_positions.last().map_or(0, |p| p + 1);
        let ids: Vec<u32> = std::iter::once(self.config.start_of_image()).chain(image.iter().copied()).chain([self.config.end_of_image()]).collect();
        let rows = ids.len() as u32;
        for (row, id) in ids.iter().enumerate() {
            self.layout.run_ops(&mut self.vm, &self.program, &part.ops[part.embed.clone()], *id, 0, false)?;
            self.layout.run_ops(&mut self.vm, &self.program, &part.ops[part.merge.clone()], row as u32, 0, false)?;
        }
        for row in 0..rows {
            self.layout.place_rotation(&self.program, text + row, [first_position + row as u64; 3], None)?;
        }
        for (first, rest) in &part.blocks {
            for row in 0..rows {
                self.layout.run_ops(&mut self.vm, &self.program, &part.ops[first.clone()], row, text + row, true)?;
            }
            for row in 0..rows {
                self.layout.run_ops(&mut self.vm, &self.program, &part.ops[rest.clone()], row, text + rows - 1, false)?;
            }
        }
        let score = part.blocks.last().map_or(part.merge.end, |(_, rest)| rest.end)..part.ops.len();
        let (output, width) = (self.layout.register_of(part.output), self.program.buffers[part.output]);
        let mut out = Vec::with_capacity(image.len() * width);
        // The image's places only, without its start and end.
        for row in 1..rows - 1 {
            self.layout.run_ops(&mut self.vm, &self.program, &part.ops[score.clone()], row, 0, false)?;
            out.extend(self.vm.floats(output, width));
        }
        Ok(out)
    }

    /// Fills the masked places of `codes` (`None`) as `ShowO::fill` does, with the scores of
    /// the die's arithmetic: two passes each step with guidance, with the text and without.
    #[allow(clippy::too_many_arguments)]
    pub fn fill(&mut self, codes: &[Option<u32>], text: &[u32], steps: usize, guidance: f32, seed: u64, progress: impl FnMut(usize, usize) -> Result<(), String>) -> Result<Vec<u32>, String> {
        let config = self.config.clone();
        let (with, without) = (config.text_prefix(text), config.text_prefix(&[]));
        fill_with(&config, codes, steps, guidance, seed, progress, |tokens, conditioned| {
            let (ids, positions) = if conditioned { &with } else { &without };
            self.scores(ids, positions, tokens)
        })
    }

    /// Retouches the `region` of a picture as `ShowO::inpaint` does, the transformer in the
    /// die's arithmetic and MAGVIT-v2 on the host.
    #[allow(clippy::too_many_arguments)]
    pub fn inpaint(&mut self, magvit: &Magvit, rgb: &[u8], width: usize, height: usize, region: &[bool], text: &[u32], steps: usize, guidance: f32, seed: u64, progress: impl FnMut(usize, usize) -> Result<(), String>) -> Result<Vec<u8>, String> {
        let known = self.config.masked_codes(magvit, rgb, width, height, region)?;
        let filled = self.fill(&known, text, steps, guidance, seed, progress)?;
        magvit.decode(&filled)
    }

    /// A new picture from `text` alone, every code masked.
    pub fn draw(&mut self, magvit: &Magvit, text: &[u32], steps: usize, guidance: f32, seed: u64, progress: impl FnMut(usize, usize) -> Result<(), String>) -> Result<Vec<u8>, String> {
        let filled = self.fill(&vec![None; self.config.image_tokens], text, steps, guidance, seed, progress)?;
        magvit.decode(&filled)
    }
}
