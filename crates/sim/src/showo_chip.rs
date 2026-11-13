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
use crate::weights::{Tensor, Weights};
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

/// The output's rows of the codes, as their own matrix with their biases: the rest of the
/// output is never needed.
const CODE_HEAD: &str = "showo.cabeza_de_codigos";

/// The final norm and the scores of the codes.
fn code_scores(e: &mut Emit, s: &Shape, x: Buf) -> (Buf, Buf) {
    let normed = e.norm("norma final", x, "showo.model.final_layernorm".into(), s.hidden);
    let codes = e.linear("puntuaciones de los códigos", CODE_HEAD.into(), normed, s.codebook);
    (normed, codes)
}

/// Show-o's shape and configuration from its folder, and its weights with the code head
/// cut out, as `program` reads them.
pub fn weights(dir: &Path, weights: Weights) -> Result<(Shape, ShowOConfig, Weights), String> {
    let (shape, config) = Shape::read(dir, &weights)?;
    let head = code_head(&weights, &shape)?;
    Ok((shape, config, weights.with_floats(head)?))
}

/// The code head's matrix and biases, cut from the output's.
fn code_head(weights: &Weights, s: &Shape) -> Result<Vec<Tensor>, String> {
    let (weight, bias) = (weights.floats("showo.lm_head.weight")?, weights.floats("showo.lm_head.bias")?);
    let rows = s.first_code..s.first_code + s.codebook;
    Ok(vec![
        (format!("{CODE_HEAD}.weight"), vec![s.codebook, s.hidden], weight[rows.start * s.hidden..rows.end * s.hidden].to_vec()),
        (format!("{CODE_HEAD}.bias"), vec![s.codebook], bias[rows].to_vec()),
    ])
}

/// Show-o as the chip's program: the text's token in `ops`, the image's routines in
/// `vision`. In the image's part, `embed` reads a token's row (run with the token) and
/// `merge` stores it in the image's rows (run with the row); each block's two routines and
/// the last one, which scores the codes, take the row.
pub fn program(s: &Shape) -> Program {
    program_guided(s, None)
}

/// The same with `guidance` in the die: after the merge, a routine that keeps a row's scores
/// with the text, and one that gives in `output` the guided ones from those without it.
pub fn program_guided(s: &Shape, guidance: Option<f32>) -> Program {
    let mut guided = None;
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
    let mut blocks = Vec::new();
    for layer in 0..s.layers {
        e.layer = Some(layer as u32);
        let first = e.ops.len();
        // The first layer's first pass starts by keeping the row the embedding left.
        if layer == 0 {
            e.op("guardar fila", Op::StoreRow { src: row, store: state });
        }
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
    let score_start = e.ops.len();
    let h = e.buf(s.hidden);
    e.op("leer fila", Op::LoadRow { store: state, dst: h });
    let (normed, scores) = code_scores(&mut e, s, h);
    let score = score_start..e.ops.len();
    // With guidance, two more routines: one keeps a row's final state with the text, the
    // other combines it with the one without, `(1 + g) · with − g · without`, by the row,
    // and scores that. The code head is affine, so that gives the same as combining the
    // scores, `(1 + g) · (W·a + b) − g · (W·c + b)`; a row's state is a quarter of its
    // scores, and the whole image's fits the die's addresses where its scores do not.
    let mut extra = Vec::new();
    if let Some(g) = guidance {
        let kept = e.buf(rows * s.hidden);
        let start = e.ops.len();
        e.op("guardar estado con texto", Op::StoreRow { src: normed, store: kept });
        extra.push(start..e.ops.len());
        let start = e.ops.len();
        let (with, without, difference, pushed, combined) = (e.buf(s.hidden), e.buf(s.hidden), e.buf(s.hidden), e.buf(s.hidden), e.buf(s.hidden));
        e.op("leer estado con texto", Op::LoadRow { store: kept, dst: with });
        e.op("cambiar de signo", Op::Scale { src: normed, factor: -1.0, dst: without });
        e.op("diferencia", Op::Add { a: with, b: without, dst: difference });
        e.op("guía", Op::Scale { src: difference, factor: g, dst: pushed });
        e.op("estado guiado", Op::Add { a: with, b: pushed, dst: combined });
        let scored = e.linear("puntuaciones guiadas", CODE_HEAD.into(), combined, s.codebook);
        extra.push(start..e.ops.len());
        guided = Some(scored);
    }
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
        embedding_rows: Some(s.vocab),
        draft: None,
        image: None,
        // The image's routines, as the chip numbers them: the embedding (by the token), each
        // layer's two passes (by the row), and the codes' scores (by the row).
        vision: Some(VisionPart { ops: image_ops, embed, blocks, merge: score, pixels: unused, position: unused, output: guided.unwrap_or(scores), head_dim: s.head_dim(), patches: rows, extra }),
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
        Self::open_guided(dir, weights, die, None)
    }

    /// The same with `guidance` in the die (`program_guided`).
    pub fn open_guided(dir: &Path, weights: Weights, die: DieConfig, guidance: Option<f32>) -> Result<Self, String> {
        let (shape, config, weights) = self::weights(dir, weights)?;
        let program = program_guided(&shape, guidance);
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

    /// On the exact machine's die with vectors of `bits` bits, the weights in `format` (or
    /// whole, in cells as wide as the vectors), and `guidance` in the die.
    pub fn exact_guided(dir: &Path, bits: u32, format: Option<netlist::format::WeightFormat>, guidance: f32) -> Result<Self, String> {
        let base = crate::exact::exact_config(bits);
        let die = match format {
            Some(_) => base,
            None => DieConfig { cell: netlist::CellConfig { weight_bits: bits, accumulator_bits: 2 * bits + 8, ..base.cell }, ..base },
        };
        Self::open_guided(dir, Weights::open(dir)?.with_format(format), die, Some(guidance))
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
        for row in 0..rows {
            self.layout.place_rotation(&self.program, text + row, [first_position + row as u64; 3], None)?;
        }
        for (layer, (first, rest)) in part.blocks.iter().enumerate() {
            for row in 0..rows {
                // Before the first layer, the row's embedding, which its first pass keeps.
                if layer == 0 {
                    self.layout.run_ops(&mut self.vm, &self.program, &part.ops[part.embed.clone()], ids[row as usize], 0, false)?;
                }
                self.layout.run_ops(&mut self.vm, &self.program, &part.ops[first.clone()], row, text + row, true)?;
            }
            for row in 0..rows {
                self.layout.run_ops(&mut self.vm, &self.program, &part.ops[rest.clone()], row, text + rows - 1, false)?;
            }
        }
        let score = part.merge.clone();
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

    /// Retouches as `inpaint` does with MAGVIT-v2 on the chip too: `magvit` encodes the
    /// picture and decodes the result in the die's arithmetic.
    #[allow(clippy::too_many_arguments)]
    pub fn inpaint_on_chip(&mut self, magvit: &mut crate::magvit_chip::MagvitPair, rgb: &[u8], width: usize, height: usize, region: &[bool], text: &[u32], steps: usize, guidance: f32, seed: u64, mut progress: impl FnMut(usize, usize) -> Result<(), String>) -> Result<Vec<u8>, String> {
        let known = self.config.clone().masked_codes_with(|picture| magvit.encoder.encode(picture), rgb, width, height, region)?;
        let filled = self.fill(&known, text, steps, guidance, seed, &mut progress)?;
        magvit.decoder.decode(&filled)
    }

    /// Draws as `draw` does, decoding with MAGVIT-v2 on the chip.
    pub fn draw_on_chip(&mut self, magvit: &mut crate::magvit_chip::MagvitPair, text: &[u32], steps: usize, guidance: f32, seed: u64, progress: impl FnMut(usize, usize) -> Result<(), String>) -> Result<Vec<u8>, String> {
        let filled = self.fill(&vec![None; self.config.image_tokens], text, steps, guidance, seed, progress)?;
        magvit.decoder.decode(&filled)
    }

    /// A new picture from `text` alone, every code masked.
    pub fn draw(&mut self, magvit: &Magvit, text: &[u32], steps: usize, guidance: f32, seed: u64, progress: impl FnMut(usize, usize) -> Result<(), String>) -> Result<Vec<u8>, String> {
        let filled = self.fill(&vec![None; self.config.image_tokens], text, steps, guidance, seed, progress)?;
        magvit.decode(&filled)
    }
}

/// What the firmware needs to run a pass of `program`, Show-o's, laid out by `layout`.
pub fn firmware_info(program: &Program, layout: &Layout, die: &DieConfig) -> Result<netlist::chip::ShowOInfo, String> {
    let part = program.vision.as_ref().ok_or("no es el programa de Show-o")?;
    let (rope, rope_half) = layout.rope_table().ok_or("la tabla de rotaciones no está en registros")?;
    let scores = layout.register_of(part.output);
    Ok(netlist::chip::ShowOInfo {
        rows: part.patches as u32,
        layers: part.blocks.len() as u32,
        codebook: program.buffers[part.output] as u32,
        scores: (scores << die.index_bits, exponent_register(die) << die.index_bits | scores),
        rope,
        rope_half,
        guided: !part.extra.is_empty() as u32,
    })
}

/// A die with cells for Show-o's `program`, addressed as the firmware addresses it, with
/// `base`'s cells and arithmetic. The embedding table stays in the image and the loader
/// brings each token's row, the text's and the image's: with the real Show-o it alone would
/// take 27 bits of element. Every other memory is just as large as the routines reach, so
/// that register and element fit together in the CPU's 32-bit addresses; if they do not,
/// more cells make each one's weights shallower.
pub fn addressed_die(program: &Program, weights: &Weights, base: DieConfig, context: usize) -> Result<DieConfig, String> {
    let bits_for = |count: u64| (u64::BITS - count.max(2).saturating_sub(1).leading_zeros()).max(1);
    let layout = |die: DieConfig| Layout::with_cells_table_outside(program, weights, die, context);
    let roomy = DieConfig { microcode_bits: 20, reg_bits: 12, index_bits: 30, cell: netlist::CellConfig { addr_bits: 30, ..base.cell }, features: base.features.with_key_shift(bits_for(context as u64)), ..base };
    let reach = layout(roomy)?;
    let (depth, groups) = reach.cell_depths();
    // A product seeks its weights by the element's field, so it holds the cells' depth too.
    let addr_bits = bits_for(depth.max(2 * groups));
    reach.microprogram(program)?;
    let furthest = reach.longest_registers(1).first().map_or(1, |(_, extent)| *extent as u64);
    let initial = reach.setup().initial.keys().map(|(_, index)| *index as u64 + 1).max().unwrap_or(1);
    let index_bits = bits_for(furthest.max(initial)).max(addr_bits);
    let sized = DieConfig { index_bits, cell: netlist::CellConfig { addr_bits, ..roomy.cell }, ..roomy };
    let wide = layout(sized)?;
    let code = wide.microprogram(program)?;
    let die = DieConfig { microcode_bits: bits_for(code.words.len() as u64), ..crate::chip::with_registers_needed(sized, &wide) };
    if die.reg_bits + die.index_bits > 32 {
        return Err(format!("{} bits de registro y {} de elemento no caben en 32: hacen falta más celdas", die.reg_bits, die.index_bits));
    }
    Ok(die)
}

/// A MaskGIT fill as the chip's firmware runs it.
pub struct Fill<'a> {
    /// The text's tokens and the position of the first, and the same without the text
    /// when the die guides: `ShowOConfig::text_prefix` gives them.
    pub text: (&'a [u32], u32),
    pub plain: Option<(&'a [u32], u32)>,
    /// For each step, how many places the schedule leaves masked and the noise's
    /// temperature, with `maskgit_rt::FRACTION` fractional bits.
    pub schedule: &'a [(i64, i64)],
    pub seed: u32,
}

/// The schedule of `steps` steps for `places` places as Show-o's: masked along a cosine,
/// the noise cooling as the steps go.
pub fn schedule(places: usize, steps: usize) -> Vec<(i64, i64)> {
    let mut temperature = 1f64;
    (0..steps)
        .map(|step| {
            let ratio = (step + 1) as f64 / steps as f64;
            let remaining = (places as f64 * (ratio * std::f64::consts::FRAC_PI_2).cos()).floor() as i64;
            temperature *= 1.0 - ratio;
            (remaining, (temperature * (1u64 << maskgit_rt::FRACTION) as f64).round() as i64)
        })
        .collect()
}

/// What a fill needs from a die: run its routines, rotate a place of the cache as another
/// position, and read the scores the scores' routine leaves, as signed mantissas with their
/// exponent.
pub trait FillDie {
    fn run(&mut self, routine: usize, token: u32, position: u32);
    fn rotate(&mut self, slot: u32, position: u32);
    fn scores(&mut self) -> (i32, Vec<i32>);
}

/// A die addressed as the firmware addresses it, through a `DieEngine`.
pub struct Addressed<'a> {
    pub engine: &'a mut dyn netlist::chip::DieEngine,
    pub layout: &'a Layout,
    pub program: &'a Program,
    pub info: &'a netlist::chip::ShowOInfo,
    pub bits: u32,
}

impl FillDie for Addressed<'_> {
    fn run(&mut self, routine: usize, token: u32, position: u32) {
        // With the embedding table left in the image, its row goes into the table's
        // register first, as the firmware's loader brings it, for the text and the image.
        if let Some(table) = self.layout.outside_table().filter(|_| routine == crate::lower::ENTRY_TOKEN || routine == crate::lower::ENTRY_VISION) {
            let base = table.register << self.layout.config().index_bits;
            let row = &table.mantissas[(token * table.cols) as usize..][..table.cols as usize];
            row.iter().enumerate().for_each(|(i, value)| self.engine.write(base + i as u32, *value));
        }
        self.engine.run(routine, token, position);
    }

    fn rotate(&mut self, slot: u32, position: u32) {
        let width = 3 * self.info.rope_half + 1;
        let row = self.layout.text_rotation(self.program, [position as u64; 3], [self.info.rope_half as usize, 0, 0]);
        row.iter().enumerate().for_each(|(i, value)| self.engine.write(self.info.rope + slot * width + i as u32, *value));
    }

    fn scores(&mut self) -> (i32, Vec<i32>) {
        let bits = self.bits;
        let signed = |v: i64| ((v as u32) << (32 - bits)) as i32 >> (32 - bits);
        let exponent = signed(self.engine.read(self.info.scores.1));
        (exponent, (0..self.info.codebook).map(|c| signed(self.engine.read(self.info.scores.0 + c))).collect())
    }
}

impl FillDie for ShowOChip {
    fn run(&mut self, routine: usize, token: u32, position: u32) {
        self.layout.run_routine(&mut self.vm, &self.program, routine, token, position).expect("rutina de Show-o");
    }

    fn rotate(&mut self, slot: u32, position: u32) {
        self.layout.place_rotation(&self.program, slot, [position as u64; 3], None).expect("rotación");
    }

    fn scores(&mut self) -> (i32, Vec<i32>) {
        let part = self.program.vision.as_ref().expect("la imagen");
        let (reg, width) = (self.layout.register_of(part.output), self.program.buffers[part.output] as u32);
        (self.vm.register_exponent(reg) as i32, (0..width).map(|i| self.vm.mantissa(reg, i) as i32).collect())
    }
}

impl ShowOChip {
    /// Starts the machine over, as before a fill.
    pub fn reset(&mut self) {
        self.vm.clear();
        crate::chip::load_initial(&mut self.vm, &self.layout.setup().initial, self.exponents);
        self.layout.clear_rotations();
    }

    /// The shape of a fill on this machine: its rows, layers, codes and whether it guides.
    pub fn fill_shape(&self) -> FillShape {
        let part = self.program.vision.as_ref().expect("la imagen");
        FillShape { rows: part.patches as u32, layers: part.blocks.len() as u32, guided: !part.extra.is_empty(), bits: self.layout.config().cell.activation_bits }
    }
}

impl ShowOChip {
    /// Fills the masked places of `codes` as the chip's firmware does (`fill_on`): MaskGIT
    /// in integers and, if this machine was opened with guidance, the guidance in the die.
    pub fn fill_firmware(&mut self, codes: &[Option<u32>], text: &[u32], steps: usize, seed: u64, mut progress: impl FnMut(usize, usize) -> Result<(), String>) -> Result<Vec<u32>, String> {
        let config = self.config.clone();
        let shape = self.fill_shape();
        self.reset();
        firmware_fill(self, shape, &config, codes, text, steps, seed, &mut progress)
    }
}

/// The firmware's fill of `codes` with `text`, in `steps` steps, on any die.
#[allow(clippy::too_many_arguments)]
fn firmware_fill(die: &mut dyn FillDie, shape: FillShape, config: &ShowOConfig, codes: &[Option<u32>], text: &[u32], steps: usize, seed: u64, progress: &mut dyn FnMut(usize, usize) -> Result<(), String>) -> Result<Vec<u32>, String> {
    if codes.len() != config.image_tokens {
        return Err(format!("una imagen de Show-o tiene {} códigos", config.image_tokens));
    }
    let (ids, first) = config.text_prefix(text);
    let (plain, plain_first) = config.text_prefix(&[]);
    let plan = schedule(codes.len(), steps);
    // The firmware's generator takes a word; the seed's two halves both count.
    let seed = (seed ^ (seed >> 32)) as u32;
    let fill = Fill { text: (&ids, first[0] as u32), plain: shape.guided.then_some((&plain[..], plain_first[0] as u32)), schedule: &plan, seed };
    fill_on_with(die, shape, config, codes, &fill, progress)
}

/// Show-o on the die as a chip addresses it (`addressed_die`): the weights in its cells,
/// the embedding table in the image, a row at a time, and the firmware's fill, MaskGIT in
/// integers with the guidance in the die. The closest the host gets to the chip itself.
pub struct ShowOAddressed {
    pub config: ShowOConfig,
    pub die: DieConfig,
    engine: crate::chip::VmEngine,
    layout: Layout,
    program: Program,
    info: netlist::chip::ShowOInfo,
}

impl ShowOAddressed {
    /// From Show-o's folder, the weights in `format` (affine, as the cells keep it), with
    /// `cells` cells and `guidance` in the die, if any.
    pub fn open(dir: &Path, format: netlist::format::WeightFormat, guidance: Option<f32>, cells: u32) -> Result<Self, String> {
        let open = || -> Result<_, String> { weights(dir, Weights::open(dir)?.with_format(Some(format))) };
        let (shape, config, w) = open()?;
        let program = program_guided(&shape, guidance);
        let context = (config.max_text + 1 + shape.image_rows).next_power_of_two();
        let base = DieConfig {
            cells,
            cell: netlist::CellConfig { weight_bits: format.bits, activation_bits: 16, accumulator_bits: 48, mac_stages: 1, adder_stages: 1, addr_bits: 3, coding: format.coding, exponent_bits: 0, multiplier: netlist::multiplier::MultiplierKind::Array, fixed_weights: false, ecc: false, skip_zeros: false },
            scale_bits: 21,
            reg_bits: 12,
            index_bits: 12,
            microcode_bits: 18,
            broadcast_stages: 1,
            function_bits: 12,
            function_select_bits: 3,
            features: Default::default(),
        };
        let die = addressed_die(&program, &w, base, context)?;
        let layout = Layout::with_cells_table_outside(&program, &w, die, context)?;
        let info = firmware_info(&program, &layout, &die)?;
        drop(w);
        let engine = crate::chip::VmEngine::new_table_outside(program.clone(), open()?.2, die, context)?;
        Ok(Self { config, die, engine, layout, program, info })
    }

    /// Fills as `ShowOChip::fill_firmware` does, on this die.
    pub fn fill(&mut self, codes: &[Option<u32>], text: &[u32], steps: usize, seed: u64, mut progress: impl FnMut(usize, usize) -> Result<(), String>) -> Result<Vec<u32>, String> {
        self.engine.reset();
        let shape = FillShape { rows: self.info.rows, layers: self.info.layers, guided: self.info.guided != 0, bits: self.die.cell.activation_bits };
        let config = self.config.clone();
        let mut die = Addressed { engine: &mut self.engine, layout: &self.layout, program: &self.program, info: &self.info, bits: shape.bits };
        firmware_fill(&mut die, shape, &config, codes, text, steps, seed, &mut progress)
    }
}

/// What the fill needs to know of the program: the image's rows, the layers, whether the
/// die guides, and the bits of its mantissas.
#[derive(Clone, Copy, Debug)]
pub struct FillShape {
    pub rows: u32,
    pub layers: u32,
    pub guided: bool,
    pub bits: u32,
}

/// The firmware's MaskGIT fill, ordered from the host on an addressed die: the same
/// routines, the same draws and the same choices. `codes` are the image's places, `None`
/// where masked; `bits` are the die's mantissas'. Gives every place's code.
#[allow(clippy::too_many_arguments)]
pub fn fill_on_die(engine: &mut dyn netlist::chip::DieEngine, layout: &Layout, program: &Program, info: &netlist::chip::ShowOInfo, config: &ShowOConfig, bits: u32, codes: &[Option<u32>], fill: &Fill) -> Vec<u32> {
    let shape = FillShape { rows: info.rows, layers: info.layers, guided: info.guided != 0, bits };
    fill_on(&mut Addressed { engine, layout, program, info, bits }, shape, config, codes, fill)
}

/// The fill on any die, as the firmware runs it.
pub fn fill_on(die: &mut dyn FillDie, shape: FillShape, config: &ShowOConfig, codes: &[Option<u32>], fill: &Fill) -> Vec<u32> {
    fill_on_with(die, shape, config, codes, fill, &mut |_, _| Ok(())).expect("sin nada que lo pare")
}

/// The same, telling `progress` of each step before it runs, and of the end: an error from
/// it stops the fill with that error.
pub fn fill_on_with(die: &mut dyn FillDie, shape: FillShape, config: &ShowOConfig, codes: &[Option<u32>], fill: &Fill, progress: &mut dyn FnMut(usize, usize) -> Result<(), String>) -> Result<Vec<u32>, String> {
    use crate::lower::{ENTRY_TOKEN, ENTRY_VISION};
    let score_routine = ENTRY_VISION + 1 + 2 * shape.layers as usize;
    let mut current = codes.to_vec();
    let mut sampled: Vec<u32> = codes.iter().map(|c| c.unwrap_or(0)).collect();
    let mut rng = maskgit_rt::Rng(fill.seed | 1);
    let rows = shape.rows;
    let pass = |die: &mut dyn FillDie, (ids, first): (&[u32], u32), image: &[u32], after: &mut dyn FnMut(&mut dyn FillDie, u32)| {
        let text = ids.len() as u32;
        for slot in 0..text + rows {
            die.rotate(slot, first + slot);
        }
        for (slot, id) in ids.iter().enumerate() {
            die.run(ENTRY_TOKEN, *id, slot as u32);
        }
        for layer in 0..shape.layers as usize {
            for row in 0..rows {
                if layer == 0 {
                    die.run(ENTRY_VISION, image[row as usize], 0);
                }
                die.run(ENTRY_VISION + 1 + 2 * layer, row, text + row);
            }
            for row in 0..rows {
                die.run(ENTRY_VISION + 2 + 2 * layer, row, text + rows - 1);
            }
        }
        for row in 1..rows - 1 {
            die.run(score_routine, row, 0);
            after(die, row);
        }
    };
    for (step, (remaining, temperature)) in fill.schedule.iter().enumerate() {
        progress(step, fill.schedule.len())?;
        let image: Vec<u32> = std::iter::once(config.start_of_image()).chain(current.iter().map(|c| c.map_or(config.mask_token, |c| config.first_code() + c))).chain([config.end_of_image()]).collect();
        let mut confidence = vec![i64::MAX; current.len()];
        let mut draw = |die: &mut dyn FillDie, row: u32, current: &[Option<u32>], sampled: &mut [u32], confidence: &mut [i64]| {
            let place = row as usize - 1;
            if let Some(code) = current[place] {
                sampled[place] = code;
                return;
            }
            let (exponent, scores) = die.scores();
            let (code, sure) = maskgit_rt::draw(scores.len() as u32, &|c| scores[c as usize], exponent, shape.bits, *temperature, &mut rng);
            sampled[place] = code;
            confidence[place] = sure;
        };
        match (fill.plain, shape.guided) {
            (Some(plain), true) => {
                pass(die, fill.text, &image, &mut |die, row| die.run(score_routine + 1, row, 0));
                pass(die, plain, &image, &mut |die, row| {
                    die.run(score_routine + 2, row, 0);
                    draw(die, row, &current, &mut sampled, &mut confidence)
                });
            }
            _ => pass(die, fill.text, &image, &mut |die, row| draw(die, row, &current, &mut sampled, &mut confidence)),
        }
        let unknown = current.iter().filter(|c| c.is_none()).count();
        let cut = maskgit_rt::cut(&confidence, maskgit_rt::masked(*remaining, unknown));
        for (place, code) in current.iter_mut().enumerate() {
            *code = (confidence[place] >= cut).then_some(sampled[place]);
        }
    }
    progress(fill.schedule.len(), fill.schedule.len())?;
    Ok(sampled)
}
