//! MAGVIT-v2's decoder in the die's arithmetic: Show-o's codes to pixels, with the chip's
//! operations, run by the exact machine in the order the host gives them.
//!
//! Its maps are stores of rows, one row a pixel with all its channels, kept with a border
//! of zero rows: a pixel's eight neighbours are then rows at fixed distances, whatever the
//! pixel, and a 3 × 3 convolution is one product of the nine rows side by side with the
//! kernel laid out the same way.
//!
//! - **GroupNorm** needs each group's mean and spread over the whole map. A first pass
//!   writes each pixel's sums per group (a product with a matrix of ones, in the cells'
//!   wide accumulators) and of its squares; the rows are then added in pairs, a level at a
//!   time, until one row holds the totals; a last pass normalises every pixel with them.
//! - **The attention in the middle** goes in two passes over the pixels, as Show-o's image.
//! - **Doubling the side** copies each row to its four places.

use crate::lower::Layout;
use crate::vm::BlockVm;
use crate::weights::{Tensor, Weights};
use compiler::{Buf, Inst, KvShape, Op, Program, VisionPart};
use netlist::block::{encode_vector, exponent_register, BlockOps};
use netlist::DieConfig;
use std::collections::HashMap;
use std::ops::Range;
use std::path::Path;
use std::sync::Arc;

const GROUPS: usize = 32;
/// Each code's bits, which the decoder reads as ±1.
const CODE_BITS: usize = 13;

/// Which row of a map a routine works on, for the pixel at `(y, x)` of a map of side `side`.
#[derive(Clone, Copy, Debug)]
enum Row {
    /// The pixel's own row, in the map with its border.
    Centre,
    /// The row of its upper-left neighbour: the first of the nine a convolution reads.
    Corner,
    /// The pixel's place in a map without border, a row after another.
    Compact,
    /// The pixel's own row in a map of half the side: where it is copied from when doubling.
    Half,
    /// The last pixel's place without border: an attention over every pixel reaches it.
    Last,
    /// The upper-left neighbour of the pixel's place in a map of twice the side, every
    /// other pixel: a convolution that halves the side, padded on the right and below.
    Stride,
}

/// A routine and the rows it takes as its token and its position.
type Step = (Range<usize>, Row, Row);

/// What the host runs: routines one after another for each pixel of a map, pairs of rows
/// added, or a routine once.
#[derive(Clone, Debug)]
enum Pass {
    Pixels { side: usize, steps: Vec<Step> },
    /// Rows `i` and `i + half` added into row `i`, halving until one is left.
    Sums { load: Range<usize>, load_second: Range<usize>, add: Range<usize>, rows: usize },
    Once { routine: Range<usize> },
}

fn padded(y: usize, x: usize, side: usize) -> usize {
    (y + 1) * (side + 2) + x + 1
}

fn neighbours(side: usize) -> Vec<usize> {
    let w = side + 2;
    vec![0, 1, 2, w, w + 1, w + 2, 2 * w, 2 * w + 1, 2 * w + 2]
}

struct Builder<'a> {
    weights: &'a Weights,
    buffers: Vec<usize>,
    ops: Vec<Inst>,
    passes: Vec<Pass>,
    /// Kernels laid out as the chip reads them, and the matrices of ones of the groups.
    derived: Vec<Tensor>,
    /// Stores by side, channels and use, kept for every map of that shape.
    stores: HashMap<(usize, usize, &'static str), Buf>,
    /// Buffers of one row, by width and use.
    rows: HashMap<(usize, &'static str), Buf>,
}

impl Builder<'_> {
    fn buf(&mut self, len: usize) -> Buf {
        self.buffers.push(len);
        self.buffers.len() - 1
    }

    fn store(&mut self, side: usize, channels: usize, name: &'static str) -> Buf {
        if let Some(b) = self.stores.get(&(side, channels, name)) {
            return *b;
        }
        let b = self.buf((side + 2) * (side + 2) * channels);
        self.stores.insert((side, channels, name), b);
        b
    }

    fn row(&mut self, width: usize, name: &'static str) -> Buf {
        if let Some(b) = self.rows.get(&(width, name)) {
            return *b;
        }
        let b = self.buf(width);
        self.rows.insert((width, name), b);
        b
    }

    fn op(&mut self, name: &'static str, op: Op) {
        self.ops.push(Inst { op, layer: None, name });
    }

    /// The operations `body` adds, as one routine.
    fn routine(&mut self, body: impl FnOnce(&mut Self)) -> Range<usize> {
        let start = self.ops.len();
        body(self);
        start..self.ops.len()
    }

    fn channels(&self, name: &str) -> Result<(usize, usize), String> {
        let shape = self.weights.shape(&format!("{name}.weight")).ok_or_else(|| format!("falta {name}"))?;
        Ok((shape[0], shape[1]))
    }

    /// A convolution's kernel as one matrix: for a 1 × 1, its rows; for a 3 × 3, each
    /// output's weights in the order of the nine rows side by side.
    fn kernel(&mut self, name: &str, size: usize) -> Result<(String, usize, usize), String> {
        let (outputs, inputs) = self.channels(name)?;
        let weight = self.weights.floats(&format!("{name}.weight"))?;
        let derived = format!("{name}.filas");
        let taps = size * size;
        let mut laid = vec![0f32; outputs * taps * inputs];
        for o in 0..outputs {
            for c in 0..inputs {
                for k in 0..taps {
                    laid[(o * taps + k) * inputs + c] = weight[(o * inputs + c) * taps + k];
                }
            }
        }
        self.derived.push((format!("{derived}.weight"), vec![outputs, taps * inputs], laid));
        Ok((derived, outputs, inputs))
    }

    /// `matrix · src` plus the convolution's bias, into a buffer of its own use.
    fn product(&mut self, name: &str, matrix: String, src: Buf, outputs: usize, use_: &'static str) -> Buf {
        let (out, biased) = (self.row(outputs, "producto"), self.row(outputs, use_));
        self.op("producto", Op::Matvec { matrix, src, dst: out });
        self.op("sesgo", Op::AddWeight { src: out, weight: format!("{name}.bias"), dst: biased });
        biased
    }

    /// The matrix that adds each group's channels.
    fn group_matrix(&mut self, channels: usize) -> String {
        let name = format!("magvit.grupos.{channels}");
        if !self.derived.iter().any(|(n, _, _)| *n == format!("{name}.weight")) {
            let per = channels / GROUPS;
            let ones = (0..GROUPS * channels).map(|i| (i % channels / per == i / channels) as u8 as f32).collect();
            self.derived.push((format!("{name}.weight"), vec![GROUPS, channels], ones));
        }
        name
    }

    /// GroupNorm of the map in `src` into `dst`, followed by swish when `swish`.
    fn group_norm(&mut self, name: &str, src: Buf, dst: Buf, side: usize, channels: usize, swish: bool) {
        let matrix = self.group_matrix(channels);
        let (sums, squares) = (self.buf(side * side * GROUPS), self.buf(side * side * GROUPS));
        let (x, x2, s, q) = (self.row(channels, "x"), self.row(channels, "cuadrados"), self.row(GROUPS, "sumas"), self.row(GROUPS, "sumas de cuadrados"));
        let gather = self.routine(|b| {
            b.op("leer píxel", Op::LoadRow { store: src, dst: x });
            b.op("cuadrados", Op::MulElem { a: x, b: x, dst: x2 });
            b.op("sumas por grupo", Op::Matvec { matrix: matrix.clone(), src: x, dst: s });
            b.op("sumas de cuadrados por grupo", Op::Matvec { matrix: matrix.clone(), src: x2, dst: q });
        });
        let keep = self.routine(|b| {
            b.op("guardar sumas", Op::StoreRow { src: s, store: sums });
            b.op("guardar sumas", Op::StoreRow { src: q, store: squares });
        });
        self.passes.push(Pass::Pixels { side, steps: vec![(gather, Row::Centre, Row::Centre), (keep, Row::Compact, Row::Compact)] });
        // Pairs of rows added, a level at a time.
        let (a, a2, c, c2, e, e2) = (self.row(GROUPS, "suma a"), self.row(GROUPS, "cuadrados a"), self.row(GROUPS, "suma b"), self.row(GROUPS, "cuadrados b"), self.row(GROUPS, "suma"), self.row(GROUPS, "cuadrados"));
        let load = self.routine(|b| {
            b.op("leer sumas", Op::LoadRow { store: sums, dst: a });
            b.op("leer sumas", Op::LoadRow { store: squares, dst: a2 });
        });
        let load_second = self.routine(|b| {
            b.op("leer sumas", Op::LoadRow { store: sums, dst: c });
            b.op("leer sumas", Op::LoadRow { store: squares, dst: c2 });
        });
        let add = self.routine(|b| {
            b.op("sumar", Op::Add { a, b: c, dst: e });
            b.op("sumar", Op::Add { a: a2, b: c2, dst: e2 });
            b.op("guardar sumas", Op::StoreRow { src: e, store: sums });
            b.op("guardar sumas", Op::StoreRow { src: e2, store: squares });
        });
        self.passes.push(Pass::Sums { load, load_second, add, rows: side * side });
        let (total, total_squares) = (self.buf(GROUPS), self.buf(GROUPS));
        let once = self.routine(|b| {
            b.op("leer totales", Op::LoadRow { store: sums, dst: total });
            b.op("leer totales", Op::LoadRow { store: squares, dst: total_squares });
        });
        self.passes.push(Pass::Once { routine: once });
        let (y, z) = (self.row(channels, "normalizada"), self.row(channels, "swish"));
        let count = side * side * channels / GROUPS;
        let weight = name.to_string();
        let apply = self.routine(|b| {
            b.op("leer píxel", Op::LoadRow { store: src, dst: x });
            b.op("GroupNorm", Op::GroupNorm { src: x, sums: total, squares: total_squares, count, weight, groups: GROUPS, dst: y });
            let out = if swish {
                b.op("swish", Op::Silu { src: y, dst: z });
                z
            } else {
                y
            };
            b.op("guardar píxel", Op::StoreRow { src: out, store: dst });
        });
        self.passes.push(Pass::Pixels { side, steps: vec![(apply, Row::Centre, Row::Centre)] });
    }

    /// A 3 × 3 convolution of the map in `src` into `dst`, plus the map `residual` when
    /// given, through its 1 × 1 `shortcut` when the channels change.
    fn conv3(&mut self, name: &str, src: Buf, dst: Buf, side: usize, residual: Option<(Buf, Option<String>)>) -> Result<usize, String> {
        let (matrix, outputs, inputs) = self.kernel(name, 3)?;
        let near = self.row(9 * inputs, "vecinos");
        let shortcut = match &residual {
            Some((_, Some(short))) => Some((short.clone(), self.kernel(short, 1)?)),
            _ => None,
        };
        let width = side + 2;
        let name = name.to_string();
        let convolve = self.routine(|b| {
            b.op("leer vecinos", Op::LoadRows { store: src, dst: near, offsets: neighbours(side) });
            b.product(&name, matrix, near, outputs, "convolución");
        });
        let out = self.row(outputs, "convolución");
        let finish = self.routine(|b| {
            let stored = match residual {
                None => out,
                Some((map, _)) => {
                    let x = b.row(b.buffers[map] / (width * width), "residual");
                    b.op("leer residual", Op::LoadRow { store: map, dst: x });
                    let x = match shortcut {
                        Some((short, (matrix, outputs, _))) => b.product(&short, matrix, x, outputs, "atajo"),
                        None => x,
                    };
                    let result = b.row(outputs, "resultado");
                    b.op("residual", Op::Add { a: out, b: x, dst: result });
                    result
                }
            };
            b.op("guardar píxel", Op::StoreRow { src: stored, store: dst });
        });
        self.passes.push(Pass::Pixels { side, steps: vec![(convolve, Row::Corner, Row::Corner), (finish, Row::Centre, Row::Centre)] });
        Ok(outputs)
    }

    /// A 3 × 3 convolution every other pixel of `src`, of side `side`, into `dst`, of half
    /// the side, the map padded by one on the right and below.
    fn conv3_halving(&mut self, name: &str, src: Buf, dst: Buf, side: usize) -> Result<(), String> {
        let (matrix, outputs, inputs) = self.kernel(name, 3)?;
        let near = self.row(9 * inputs, "vecinos");
        let name = name.to_string();
        let convolve = self.routine(|b| {
            b.op("leer vecinos", Op::LoadRows { store: src, dst: near, offsets: neighbours(side) });
            b.product(&name, matrix, near, outputs, "convolución");
        });
        let out = self.row(outputs, "convolución");
        let finish = self.routine(|b| b.op("guardar píxel", Op::StoreRow { src: out, store: dst }));
        self.passes.push(Pass::Pixels { side: side / 2, steps: vec![(convolve, Row::Stride, Row::Stride), (finish, Row::Centre, Row::Centre)] });
        Ok(())
    }

    /// A ResNet block: GroupNorm and swish, convolution, twice, plus its input.
    fn resnet(&mut self, name: &str, x: Buf, side: usize, channels: usize) -> Result<(Buf, usize), String> {
        let (outputs, _) = self.channels(&format!("{name}.conv1"))?;
        let h = self.store(side, channels, "normalizada");
        self.group_norm(&format!("{name}.norm1"), x, h, side, channels, true);
        let h2 = self.store(side, outputs, "intermedia");
        self.conv3(&format!("{name}.conv1"), h, h2, side, None)?;
        let h3 = self.store(side, outputs, "normalizada");
        self.group_norm(&format!("{name}.norm2"), h2, h3, side, outputs, true);
        let shortcut = self.weights.shape(&format!("{name}.nin_shortcut.weight")).map(|_| format!("{name}.nin_shortcut"));
        // The output replaces the input where the channels stay: each pixel reads only its
        // own row of the input.
        let out = if shortcut.is_some() { self.store(side, outputs, "principal") } else { x };
        self.conv3(&format!("{name}.conv2"), h3, out, side, Some((x, shortcut)))?;
        Ok((out, outputs))
    }

    /// One head of attention over every pixel, with its residual.
    fn attention(&mut self, name: &str, x: Buf, side: usize, channels: usize) -> Result<(), String> {
        let h = self.store(side, channels, "normalizada");
        self.group_norm(&format!("{name}.norm"), x, h, side, channels, false);
        let (queries, outcome) = (self.buf(side * side * channels), self.buf(side * side * channels));
        let row = self.row(channels, "x");
        let mut kernels = Vec::new();
        for part in ["q", "k", "v", "proj_out"] {
            let full = format!("{name}.{part}");
            let (matrix, _, _) = self.kernel(&full, 1)?;
            kernels.push((full, matrix));
        }
        // First pass: each pixel's key and value to the cache, its query to a store.
        let (q, k) = (self.row(channels, "consulta"), self.row(channels, "clave"));
        let project = self.routine(|b| {
            b.op("leer píxel", Op::LoadRow { store: h, dst: row });
            b.product(&kernels[0].0, kernels[0].1.clone(), row, channels, "consulta");
            b.product(&kernels[1].0, kernels[1].1.clone(), row, channels, "clave");
            let v = b.product(&kernels[2].0, kernels[2].1.clone(), row, channels, "valor");
            b.op("guardar clave y valor", Op::KvAppend { slot: 0, key: k, value: v, window: None });
        });
        let keep = self.routine(|b| b.op("guardar consulta", Op::StoreRow { src: q, store: queries }));
        self.passes.push(Pass::Pixels { side, steps: vec![(project, Row::Centre, Row::Compact), (keep, Row::Compact, Row::Compact)] });
        // Second pass: each query over every pixel, projected, stored; then added to its pixel.
        let (read, attended) = (self.row(channels, "consulta leída"), self.row(channels, "atendida"));
        let attend = self.routine(|b| {
            b.op("leer consulta", Op::LoadRow { store: queries, dst: read });
            b.op("atención", Op::AttnSum { slot: 0, query: read, head_dim: channels, scale_dim: channels, heads: 1, kv_heads: 1, window: None, softcap: None, dst: attended });
            let projected = b.product(&kernels[3].0, kernels[3].1.clone(), attended, channels, "proyección");
            b.op("guardar resultado", Op::StoreRow { src: projected, store: outcome });
        });
        let (got, sum) = (self.row(channels, "proyección leída"), self.row(channels, "suma"));
        let add = self.routine(|b| {
            b.op("leer resultado", Op::LoadRow { store: outcome, dst: got });
        });
        let place = self.routine(|b| {
            b.op("leer píxel", Op::LoadRow { store: x, dst: row });
            b.op("residual", Op::Add { a: row, b: got, dst: sum });
            b.op("guardar píxel", Op::StoreRow { src: sum, store: x });
        });
        self.passes.push(Pass::Pixels { side, steps: vec![(attend, Row::Compact, Row::Last), (add, Row::Compact, Row::Compact), (place, Row::Centre, Row::Centre)] });
        Ok(())
    }
}

/// MAGVIT-v2's decoder as the chip's program, and what the host runs of it.
pub struct MagvitChip {
    program: Program,
    layout: Layout,
    vm: BlockVm,
    exponents: u32,
    bits: u32,
    passes: Vec<Pass>,
    /// Where each input row goes in, and the map the output comes out of. The decoder
    /// reads its codes' values from a table instead: `input_read` runs with the code.
    input: Buf,
    input_read: Option<Range<usize>>,
    input_routine: Range<usize>,
    output: Buf,
    output_routine: Range<usize>,
    grid: usize,
    side: usize,
    weights: Arc<Weights>,
}

/// One routine as the host runs it: its operations, its token and its position, and
/// whether the output is read after it.
#[derive(Clone, Debug, PartialEq)]
pub struct Call {
    pub routine: Range<usize>,
    pub token: u32,
    pub position: u32,
    pub read: bool,
}

/// An output read: its mantissas, as the die holds them, and its values.
type Read = (Vec<i64>, Vec<f32>);

/// The table of each code's 13 values, ±1, its bits from the highest.
const BITS_TABLE: &str = "magvit.bits";

impl MagvitChip {
    /// MAGVIT-v2's decoder from its folder, for a grid of `grid` × `grid` codes, on a die
    /// like `die` with vectors of its activation bits.
    pub fn decoder(dir: &Path, grid: usize, die: DieConfig) -> Result<Self, String> {
        Self::decoder_on(dir, grid, die, None)
    }

    /// The same with the weights in the die's cells, as a chip holds them, and the cache of
    /// the attention for `tokens` places, when `tokens` is given.
    pub fn decoder_on(dir: &Path, grid: usize, die: DieConfig, tokens: Option<usize>) -> Result<Self, String> {
        let weights = Weights::open(dir)?;
        let mut b = Builder { weights: &weights, buffers: Vec::new(), ops: Vec::new(), passes: Vec::new(), derived: Vec::new(), stores: HashMap::new(), rows: HashMap::new() };
        // Each code's values come from a table, as a token's embedding: the decoder's work is
        // all the chip's. Through the 1 × 1 convolution they go into the first map.
        let values = (0..1usize << CODE_BITS).flat_map(|code| (0..CODE_BITS).map(move |c| if code >> (CODE_BITS - 1 - c) & 1 == 1 { 1.0 } else { -1.0 })).collect();
        b.derived.push((format!("{BITS_TABLE}.weight"), vec![1 << CODE_BITS, CODE_BITS], values));
        let input = b.buf(CODE_BITS);
        let (post, post_out, _) = b.kernel("decoder.post_quant_conv", 1)?;
        let z = b.store(grid, post_out, "códigos");
        let projected = b.row(post_out, "códigos");
        let input_read = b.routine(|b| {
            b.op("leer los valores del código", Op::EmbRead { table: BITS_TABLE.into(), dst: input });
            b.product("decoder.post_quant_conv", post, input, post_out, "códigos");
        });
        let input_routine = b.routine(|b| b.op("guardar píxel", Op::StoreRow { src: projected, store: z }));
        let (channels, _) = b.channels("decoder.conv_in")?;
        let mut x = b.store(grid, channels, "principal");
        b.conv3("decoder.conv_in", z, x, grid, None)?;
        let mut side = grid;
        let mut channels = channels;
        (x, channels) = b.resnet("decoder.mid.block_1", x, side, channels)?;
        b.attention("decoder.mid.attn_1", x, side, channels)?;
        (x, channels) = b.resnet("decoder.mid.block_2", x, side, channels)?;
        let levels = (0..).take_while(|l| weights.shape(&format!("decoder.up.{l}.block.0.conv1.weight")).is_some()).count();
        for level in (0..levels).rev() {
            let mut block = 0;
            while weights.shape(&format!("decoder.up.{level}.block.{block}.conv1.weight")).is_some() {
                (x, channels) = b.resnet(&format!("decoder.up.{level}.block.{block}"), x, side, channels)?;
                block += 1;
            }
            let upsample = format!("decoder.up.{level}.upsample.conv");
            if weights.shape(&format!("{upsample}.weight")).is_some() {
                // Each row to its four places in a map of twice the side.
                let wide = b.store(2 * side, channels, "ampliada");
                let row = b.row(channels, "x");
                let read = b.routine(|b| b.op("leer píxel", Op::LoadRow { store: x, dst: row }));
                let write = b.routine(|b| b.op("guardar píxel", Op::StoreRow { src: row, store: wide }));
                b.passes.push(Pass::Pixels { side: 2 * side, steps: vec![(read, Row::Half, Row::Half), (write, Row::Centre, Row::Centre)] });
                side *= 2;
                let next = b.store(side, channels, "principal");
                b.conv3(&upsample, wide, next, side, None)?;
                x = next;
            }
        }
        let normed = b.store(side, channels, "normalizada");
        b.group_norm("decoder.norm_out", x, normed, side, channels, true);
        let (out_matrix, colours, _) = b.kernel("decoder.conv_out", 3)?;
        let near = b.row(9 * channels, "vecinos");
        let mut output = 0;
        let output_routine = b.routine(|b| {
            b.op("leer vecinos", Op::LoadRows { store: normed, dst: near, offsets: neighbours(side) });
            output = b.product("decoder.conv_out", out_matrix, near, colours, "color");
        });
        let Builder { buffers, ops, passes, derived, .. } = b;
        Self::finish(weights, (buffers, ops, passes, derived), "decoder.mid.attn_1", (input, Some(input_read), input_routine, grid), (output, output_routine, side), die, tokens)
    }

    /// MAGVIT-v2's encoder from its folder, for pictures of `side` × `side`, on a die like
    /// `die` with vectors of its activation bits.
    pub fn encoder(dir: &Path, side: usize, die: DieConfig) -> Result<Self, String> {
        let weights = Weights::open(dir)?;
        let mut b = Builder { weights: &weights, buffers: Vec::new(), ops: Vec::new(), passes: Vec::new(), derived: Vec::new(), stores: HashMap::new(), rows: HashMap::new() };
        // The pixels' colours, a row each, into the first map.
        let input = b.buf(3);
        let picture = b.store(side, 3, "imagen");
        let input_routine = b.routine(|b| b.op("guardar píxel", Op::StoreRow { src: input, store: picture }));
        let (channels, _) = b.channels("encoder.conv_in")?;
        let mut x = b.store(side, channels, "principal");
        b.conv3("encoder.conv_in", picture, x, side, None)?;
        let (mut side_now, mut channels) = (side, channels);
        let mut level = 0;
        while weights.shape(&format!("encoder.down.{level}.block.0.conv1.weight")).is_some() {
            let mut block = 0;
            while weights.shape(&format!("encoder.down.{level}.block.{block}.conv1.weight")).is_some() {
                (x, channels) = b.resnet(&format!("encoder.down.{level}.block.{block}"), x, side_now, channels)?;
                block += 1;
            }
            let downsample = format!("encoder.down.{level}.downsample.conv");
            if weights.shape(&format!("{downsample}.weight")).is_some() {
                let next = b.store(side_now / 2, channels, "principal");
                b.conv3_halving(&downsample, x, next, side_now)?;
                side_now /= 2;
                x = next;
            }
            level += 1;
        }
        (x, channels) = b.resnet("encoder.mid.block_1", x, side_now, channels)?;
        b.attention("encoder.mid.attn_1", x, side_now, channels)?;
        (x, channels) = b.resnet("encoder.mid.block_2", x, side_now, channels)?;
        let normed = b.store(side_now, channels, "normalizada");
        b.group_norm("encoder.norm_out", x, normed, side_now, channels, true);
        let (out_matrix, out_channels, _) = b.kernel("encoder.conv_out", 3)?;
        let (quant, bits, _) = b.kernel("encoder.quant_conv", 1)?;
        let near = b.row(9 * channels, "vecinos");
        let mut output = 0;
        let output_routine = b.routine(|b| {
            b.op("leer vecinos", Op::LoadRows { store: normed, dst: near, offsets: neighbours(side_now) });
            let out = b.product("encoder.conv_out", out_matrix, near, out_channels, "salida");
            output = b.product("encoder.quant_conv", quant, out, bits, "códigos");
        });
        let Builder { buffers, ops, passes, derived, .. } = b;
        Self::finish(weights, (buffers, ops, passes, derived), "encoder.mid.attn_1", (input, None, input_routine, side), (output, output_routine, side_now), die, None)
    }

    /// The program from what `b` built, its layout and its machine.
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn finish(weights: Weights, (buffers, ops, passes, derived): (Vec<usize>, Vec<Inst>, Vec<Pass>, Vec<Tensor>), attention: &str, (input, input_read, input_routine, input_side): (Buf, Option<Range<usize>>, Range<usize>, usize), (output, output_routine, side): (Buf, Range<usize>, usize), die: DieConfig, tokens: Option<usize>) -> Result<Self, String> {
        // The rows a routine reaches: the largest map with its border, or the codes' table.
        let reach = passes.iter().map(|p| match p {
            Pass::Pixels { side, .. } => (side + 2) * (side + 2),
            Pass::Sums { rows, .. } => *rows,
            Pass::Once { .. } => 1,
        });
        let reach = reach.chain([(side + 2) * (side + 2), (input_side + 2) * (input_side + 2)]).max().unwrap_or(1);
        let attention_channels = weights.shape(&format!("{attention}.q.weight")).map(|s| s[0]).ok_or("falta la atención del centro")?;
        let program = Program {
            family: "magvit".into(),
            vocab: 0,
            hidden: attention_channels,
            num_layers: 0,
            buffers,
            ops: Vec::new(),
            logits_from: 0,
            logits: output,
            kv_slots: vec![KvShape { head_dim: attention_channels, heads: 1 }],
            delta_slots: Vec::new(),
            conv_slots: Vec::new(),
            rope_theta: 10_000.0,
            rope_scaling: None,
            local_rope_theta: None,
            rms_epsilon: 1e-6,
            final_hidden: output,
            embedding_rows: input_read.is_some().then_some(1 << CODE_BITS),
            draft: None,
            image: None,
            vision: Some(VisionPart { ops, embed: 0..0, blocks: Vec::new(), merge: 0..0, pixels: input, position: input, output, head_dim: attention_channels, patches: reach }),
        };
        let weights = Arc::new(weights.with_floats(derived)?);
        let layout = match tokens {
            Some(tokens) => Layout::with_cells(&program, &weights, die, tokens)?,
            None => Layout::new(&program, &weights, die, None)?,
        };
        let mut vm = BlockVm::new(&die, layout.registers_used() as usize + 8, Some(weights.clone()));
        vm.compute_in_cells();
        Ok(Self { program, layout, vm, exponents: exponent_register(&die), bits: die.cell.activation_bits, passes, input, input_read, input_routine, output, output_routine, grid: input_side, side, weights })
    }

    /// Side of the pictures it makes.
    pub fn side(&self) -> usize {
        self.side
    }

    fn run(&mut self, routine: &Range<usize>, token: usize, position: usize) -> Result<(), String> {
        let part = self.program.vision.as_ref().expect("las rutinas");
        let list = part.ops[routine.clone()].to_vec();
        self.layout.run_ops(&mut self.vm, &self.program, &list, token as u32, position as u32, false)
    }

    /// The program, its layout and its weights, to run it elsewhere, such as on gates.
    pub fn parts(&self) -> (&Program, &Layout, &Weights) {
        (&self.program, &self.layout, &self.weights)
    }

    /// The register the output is read from.
    pub fn output_register(&self) -> u32 {
        self.layout.register_of(self.output)
    }

    /// Every routine the host runs after the input is in, in order.
    fn pass_calls(&self) -> Vec<Call> {
        let mut calls = Vec::new();
        let mut call = |routine: &Range<usize>, token: usize, position: usize| calls.push(Call { routine: routine.clone(), token: token as u32, position: position as u32, read: false });
        for pass in &self.passes {
            match pass {
                Pass::Pixels { side, steps } => {
                    let side = *side;
                    let place = |row: Row, y: usize, x: usize| match row {
                        Row::Centre => padded(y, x, side),
                        Row::Corner => y * (side + 2) + x,
                        Row::Compact => y * side + x,
                        Row::Half => padded(y / 2, x / 2, side / 2),
                        Row::Last => side * side - 1,
                        Row::Stride => (2 * y + 1) * (2 * side + 2) + 2 * x + 1,
                    };
                    for y in 0..side {
                        for x in 0..side {
                            for (routine, token, position) in steps {
                                call(routine, place(*token, y, x), place(*position, y, x));
                            }
                        }
                    }
                }
                Pass::Sums { load, load_second, add, rows } => {
                    let mut count = *rows;
                    while count > 1 {
                        let half = count / 2;
                        for i in 0..half {
                            call(load, i, i);
                            call(load_second, i + half, i + half);
                            call(add, i, i);
                        }
                        // An odd row left over goes with the first.
                        if count % 2 == 1 {
                            call(load, 0, 0);
                            call(load_second, count - 1, count - 1);
                            call(add, 0, 0);
                        }
                        count = half;
                    }
                }
                Pass::Once { routine } => call(routine, 0, 0),
            }
        }
        calls
    }

    /// The output's routine for each pixel, the output read after each.
    fn output_calls(&self) -> Vec<Call> {
        let side = self.side;
        (0..side * side).map(|i| Call { routine: self.output_routine.clone(), token: ((i / side) * (side + 2) + i % side) as u32, position: 0, read: true }).collect()
    }

    /// Everything the decoder runs for `codes`: each code's values read and kept, every
    /// pass, and each pixel's colours.
    pub fn decoder_calls(&self, codes: &[u32]) -> Result<Vec<Call>, String> {
        let read = self.input_read.clone().ok_or("esto no es el decodificador")?;
        if codes.len() != self.grid * self.grid {
            return Err(format!("el decodificador espera {} códigos", self.grid * self.grid));
        }
        let mut calls = Vec::new();
        for (i, code) in codes.iter().enumerate() {
            calls.push(Call { routine: read.clone(), token: *code, position: 0, read: false });
            calls.push(Call { routine: self.input_routine.clone(), token: padded(i / self.grid, i % self.grid, self.grid) as u32, position: 0, read: false });
        }
        calls.extend(self.pass_calls());
        calls.extend(self.output_calls());
        Ok(calls)
    }

    /// Runs `calls` from a clean machine; gives the output after each that reads it, as
    /// mantissas and as values.
    fn run_calls(&mut self, calls: &[Call]) -> Result<Vec<Read>, String> {
        let (output, width) = (self.output_register(), self.program.buffers[self.output]);
        let mut out = Vec::new();
        for call in calls {
            self.run(&call.routine, call.token as usize, call.position as usize)?;
            if call.read {
                out.push(((0..width as u32).map(|i| self.vm.mantissa(output, i)).collect(), self.vm.floats(output, width)));
            }
        }
        Ok(out)
    }

    /// The decoder's output mantissas for `codes`, a pixel after another, as the die holds
    /// them: what its gates must give too.
    pub fn decode_mantissas(&mut self, codes: &[u32]) -> Result<Vec<Vec<i64>>, String> {
        let calls = self.decoder_calls(codes)?;
        self.reset();
        Ok(self.run_calls(&calls)?.into_iter().map(|(m, _)| m).collect())
    }

    fn reset(&mut self) {
        self.vm.clear();
        crate::chip::load_initial(&mut self.vm, &self.layout.setup().initial, self.exponents);
    }

    /// The picture of `codes`, a square grid, as 8-bit RGB a row after another, as
    /// `Magvit::decode` gives it.
    pub fn decode(&mut self, codes: &[u32]) -> Result<Vec<u8>, String> {
        let calls = self.decoder_calls(codes)?;
        self.reset();
        Ok(self.run_calls(&calls)?.iter().flat_map(|(_, pixel)| pixel.iter().map(|v| (((v + 1.0) / 2.0).clamp(0.0, 1.0) * 255.0) as u8).collect::<Vec<_>>()).collect())
    }

    /// The codes of a picture of 3 planes of `side × side` in [-1, 1], row after row, as
    /// `Magvit::encode` takes it: each pixel of the grid's 13 values, their signs the bits.
    /// The host writes each pixel's colours, as it does a picture's patches.
    pub fn encode(&mut self, planes: &[f32]) -> Result<Vec<u32>, String> {
        let area = self.grid * self.grid;
        if planes.len() != 3 * area {
            return Err(format!("el codificador espera una imagen de {} × {}", self.grid, self.grid));
        }
        self.reset();
        let reg = self.layout.register_of(self.input);
        let routine = self.input_routine.clone();
        for i in 0..area {
            let values: Vec<f64> = (0..3).map(|c| planes[c * area + i] as f64).collect();
            let (mantissas, exponent) = encode_vector(&values, self.bits);
            self.vm.load_mantissas(reg, &mantissas, exponent);
            self.run(&routine, padded(i / self.grid, i % self.grid, self.grid), 0)?;
        }
        let calls: Vec<Call> = self.pass_calls().into_iter().chain(self.output_calls()).collect();
        Ok(self.run_calls(&calls)?.iter().map(|(_, values)| values.iter().fold(0u32, |code, v| (code << 1) | (*v > 0.0) as u32)).collect())
    }
}

/// MAGVIT-v2's encoder and decoder on the chip, for Show-o's pictures.
pub struct MagvitPair {
    pub encoder: MagvitChip,
    pub decoder: MagvitChip,
}

impl MagvitPair {
    /// For pictures of `side` × `side` and grids of a sixteenth of the side, on the exact
    /// machine's die with vectors of `bits` bits and cells as wide.
    pub fn exact(dir: &Path, side: usize, bits: u32) -> Result<Self, String> {
        let base = crate::exact::exact_config(bits);
        let die = DieConfig { cell: netlist::CellConfig { weight_bits: bits, accumulator_bits: 2 * bits + 8, ..base.cell }, ..base };
        Ok(Self { encoder: MagvitChip::encoder(dir, side, die)?, decoder: MagvitChip::decoder(dir, side / 16, die)? })
    }
}
