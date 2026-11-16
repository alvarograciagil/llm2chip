//! Expresses a compiled program in the die's vector arithmetic, token by token.
//! The same code drives the gates (through microcode) and the exact virtual machine.

use crate::weights::{CellRow, Weights};
use compiler::{Inst, Op, Program, RopeScaling};
use netlist::block::{FN_GELU, FN_GELU_TANH, FN_TANH};
use netlist::block::{chained_rows, encode_vector, exponent_register, register_extents, Assembly, AttentionRegs, BlockOps, CellMatrix, DeltaRegs, Expr, KeyStore, MatvecPart, Step, Var, VarRanges, FN_EXP, FN_SIGMOID, FN_SOFTPLUS};
use netlist::{DieConfig, DieProgram, Microprogram};
use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};
use std::ops::Range;

/// KV entries, convolution history and expert outputs use this fixed exponent.
const STATE_EXPONENT: i64 = 6;
/// The rotation table of an image encoder's patches, after the text's two.
pub const VISION_TABLE: usize = 2;
/// Expert outputs are small; a coarser exponent would leave them only a few bits.
const EXPERT_EXPONENT: i64 = 2;
/// Ones kept when the layout is not for a chip with a known context: attention adds up
/// as many tokens as there are ones, so this is the longest text such a layout can read.
const ONES_LENGTH: usize = 1 << 17;
const TEMPORARIES: usize = 64;
const PARTS: usize = 128;

/// Words of vector memory the operation keeps from one token to the next: the keys and
/// values of every position it can attend to, and the recurrent and convolution states.
fn state_words(program: &Program, op: &Op, context: usize) -> u64 {
    match op {
        Op::KvAppend { slot, window, .. } => {
            let shape = program.kv_slots[*slot];
            2 * (shape.heads * shape.head_dim * window.map_or(context, |w| w.min(context))) as u64
        }
        Op::Mla { slot, .. } => (program.kv_slots[*slot].head_dim * context) as u64,
        Op::DeltaStep { slot, .. } => program.delta_slots[*slot] as u64,
        Op::Dconv { slot, .. } => {
            let (dim, kernel) = program.conv_slots[*slot];
            (dim * kernel) as u64
        }
        _ => 0,
    }
}

/// The layers each of `chips` chips runs, cut so the chip with most memory has as little as
/// can be: its weights at their own widths, and its states for `context` tokens.
pub fn chip_plan(program: &Program, weights: &Weights, chips: usize, context: usize, activation_bits: u32) -> Result<Vec<Range<usize>>, String> {
    let bits = |index: usize| -> u64 {
        let op = &program.ops[index].op;
        // A lookup table never goes in the cells: it stays in the image, or in vector memory.
        if matches!(op, Op::EmbRead { .. }) {
            return 0;
        }
        // Each code, and each group's scale and bias, about as wide as an activation each.
        let matrices: u64 = matrices_of(program, op)
            .iter()
            .map(|(name, rows, cols)| match weights.matrix(name) {
                Ok(m) => (rows * cols) as u64 * m.cell_code_bits(*cols, activation_bits) as u64 + (rows * m.cell_groups(*cols).1) as u64 * 2 * activation_bits as u64,
                Err(_) => (rows * cols) as u64 * activation_bits as u64,
            })
            .sum();
        matrices + state_words(program, op, context) * activation_bits as u64
    };
    program.split_by(chips, bits)
}

/// The per-layer values an operation keeps among the small constants: the ones shared by
/// every layer, such as scales, are not counted.
fn small_of(op: &Op) -> Vec<String> {
    match op {
        Op::RmsNorm { weight, .. } | Op::RmsGated { weight, .. } | Op::Dconv { weight, .. } => vec![format!("{weight}.weight")],
        Op::AddWeight { weight, .. } => vec![weight.clone()],
        Op::LayerNorm { weight, .. } | Op::GroupNorm { weight, .. } => vec![format!("{weight}.weight"), format!("{weight}.bias")],
        Op::DeltaStep { a_log, dt_bias, .. } => vec![a_log.clone(), dt_bias.clone()],
        _ => Vec::new(),
    }
}

fn all_ops(program: &Program) -> impl Iterator<Item = &Inst> {
    program.ops.iter().chain(program.draft.iter().flat_map(|d| d.ops.iter())).chain(program.image.iter().flat_map(|i| i.logits_ops.iter().chain(&i.input_ops))).chain(program.vision.iter().flat_map(|v| v.ops.iter()))
}

/// Which buffers can share a register: each is alive from its first write to its last read
/// within a token, and two whose lives never meet take the same one. The logits, the final
/// hidden state, whatever the draft or image heads touch and anything read before it is
/// written (carried from one token to the next) keep a register of their own. A run of
/// products of the same vector writes all its outputs at its first product, so their lives
/// start there. Returns, for each buffer, a group (buffers of a group share a register) and
/// whether a later buffer of its group takes the register over.
fn buffer_groups(program: &Program) -> (Vec<usize>, Vec<bool>) {
    let count = program.buffers.len();
    let mut first: Vec<Option<usize>> = vec![None; count];
    let mut last = vec![0usize; count];
    let mut alone = vec![false; count];
    let mut run_start = 0;
    let mut previous_src = None;
    for (index, inst) in program.ops.iter().enumerate() {
        let (reads, writes) = inst.op.buffers();
        let src = if let Op::Matvec { src, .. } = inst.op { Some(src) } else { None };
        if src.is_none() || src != previous_src {
            run_start = index;
        }
        previous_src = src;
        for buffer in reads {
            if first[buffer].is_none() {
                alone[buffer] = true;
            }
            last[buffer] = last[buffer].max(index);
        }
        for buffer in writes {
            let start = if src.is_some() { run_start } else { index };
            first[buffer] = Some(first[buffer].map_or(start, |f| f.min(start)));
            last[buffer] = last[buffer].max(index);
        }
    }
    let mut kept = vec![program.logits, program.final_hidden];
    if let Some(draft) = &program.draft {
        kept.extend([draft.hidden_in, draft.hidden_out, draft.logits]);
        kept.extend(draft.ops.iter().flat_map(|i| { let (r, w) = i.op.buffers(); r.into_iter().chain(w) }));
    }
    if let Some(image) = &program.image {
        kept.extend([image.logits, image.input]);
        kept.extend(image.logits_ops.iter().chain(&image.input_ops).flat_map(|i| { let (r, w) = i.op.buffers(); r.into_iter().chain(w) }));
    }
    for buffer in kept {
        alone[buffer] = true;
    }
    let mut order: Vec<usize> = (0..count).collect();
    order.sort_by_key(|b| first[*b].unwrap_or(0));
    let mut groups = vec![usize::MAX; count];
    let mut overwritten = vec![false; count];
    // The last use of each group's current buffer, and that buffer.
    let mut busy_until: Vec<usize> = Vec::new();
    let mut occupant: Vec<usize> = Vec::new();
    for buffer in order {
        let shareable = !alone[buffer] && first[buffer].is_some();
        let free = shareable.then(|| busy_until.iter().position(|until| *until < first[buffer].unwrap_or(0))).flatten();
        let group = match free {
            Some(group) => group,
            None => {
                busy_until.push(0);
                occupant.push(buffer);
                busy_until.len() - 1
            }
        };
        if occupant[group] != buffer {
            overwritten[occupant[group]] = true;
            occupant[group] = buffer;
        }
        // A buffer kept alone takes a group that nothing else will join.
        busy_until[group] = if shareable { last[buffer] } else { usize::MAX };
        groups[buffer] = group;
    }
    (groups, overwritten)
}

fn matrices_of(program: &Program, op: &Op) -> Vec<(String, usize, usize)> {
    match op {
        Op::Matvec { matrix, src, dst } => vec![(matrix.clone(), program.buffers[*dst], program.buffers[*src])],
        Op::EmbRead { table, dst } => vec![(table.clone(), program.table_rows(table), program.buffers[*dst])],
        Op::Moe { prefix, src, experts, intermediate, shared_intermediate, .. } => {
            let hidden = program.buffers[*src];
            vec![
                (format!("{prefix}.gate"), *experts, hidden),
                (format!("{prefix}.switch_mlp.gate_proj"), experts * intermediate, hidden),
                (format!("{prefix}.switch_mlp.up_proj"), experts * intermediate, hidden),
                (format!("{prefix}.switch_mlp.down_proj"), experts * hidden, *intermediate),
                (format!("{prefix}.shared_expert.gate_proj"), *shared_intermediate, hidden),
                (format!("{prefix}.shared_expert.up_proj"), *shared_intermediate, hidden),
                (format!("{prefix}.shared_expert.down_proj"), hidden, *shared_intermediate),
                (format!("{prefix}.shared_expert_gate"), 1, hidden),
            ]
        }
        Op::Mla { embed, unembed, heads, nope, lora, value_dim, .. } => vec![(embed.clone(), heads * lora, *nope), (unembed.clone(), heads * value_dim, *lora)],
        Op::SigmoidMoe { prefix, src, experts, intermediate, shared_intermediate, .. } => {
            let hidden = program.buffers[*src];
            let mut list = vec![
                (format!("{prefix}.gate"), *experts, hidden),
                (format!("{prefix}.switch_mlp.gate_proj"), experts * intermediate, hidden),
                (format!("{prefix}.switch_mlp.up_proj"), experts * intermediate, hidden),
                (format!("{prefix}.switch_mlp.down_proj"), experts * hidden, *intermediate),
            ];
            if *shared_intermediate > 0 {
                list.extend([
                    (format!("{prefix}.shared_experts.gate_proj"), *shared_intermediate, hidden),
                    (format!("{prefix}.shared_experts.up_proj"), *shared_intermediate, hidden),
                    (format!("{prefix}.shared_experts.down_proj"), hidden, *shared_intermediate),
                ]);
            }
            list
        }
        _ => Vec::new(),
    }
}

/// How many rows of `name` one use takes, when it does not take them all.
fn program_dst(op: &Op) -> usize {
    match op {
        Op::Matvec { dst, .. } => *dst,
        _ => unreachable!("only products are chained"),
    }
}

fn rows_per_use(program: &Program, op: &Op, name: &str, rows: usize) -> usize {
    match op {
        Op::Moe { src, intermediate, .. } | Op::SigmoidMoe { src, intermediate, .. } if name.contains("switch_mlp.") => {
            if name.ends_with("down_proj") { program.buffers[*src] } else { *intermediate }
        }
        Op::Mla { embed, lora, value_dim, .. } => if name == embed { *lora } else { *value_dim },
        _ => rows,
    }
}

/// A matrix as the image stores it for the cells.
#[derive(Clone, Debug)]
pub struct CellTensor {
    pub name: String,
    pub rows: usize,
    pub cols: usize,
    /// Rows of one use, each starting again at the first cell.
    pub block: usize,
    pub exponent: i64,
    cells: usize,
    bits: u32,
    /// The first row of the whole matrix that this chip holds, when chips share it by rows.
    first_row: usize,
    /// Runs of rows of the whole matrix, `(first, count)`, when this chip's are not one.
    pieces: Vec<(usize, usize)>,
}

/// Row `row` of a chip's share of a matrix, in the whole matrix: from `first_row` on, or
/// through the runs `pieces` when it holds several.
fn whole_row(first_row: usize, pieces: &[(usize, usize)], row: usize) -> usize {
    let mut rest = row;
    for (first, count) in pieces {
        if rest < *count {
            return first + rest;
        }
        rest -= count;
    }
    first_row + row
}

impl CellTensor {
    /// The cells that hold part of this matrix.
    pub fn spread(&self) -> usize {
        self.cells.min(self.block)
    }

    /// The rows `cell` holds, in the order it reads them.
    pub fn rows_of(&self, cell: usize) -> impl Iterator<Item = usize> + '_ {
        let rounds = self.block.div_ceil(self.cells);
        (0..self.rows / self.block).flat_map(move |block| (0..rounds).map(move |round| (block, round * self.cells + cell))).filter(|(_, row)| *row < self.block).map(|(block, row)| block * self.block + row)
    }

    /// One row in the bytes of the image: its codes, then each group's scale and bias.
    pub fn encoded_row(&self, weights: &Weights, row: usize) -> Result<Vec<u8>, String> {
        let matrix = weights.matrix(&self.name)?;
        let (codes, factors) = matrix.cell_row(whole_row(self.first_row, &self.pieces, row), self.cols, self.exponent, self.bits)?;
        let code_bits = matrix.cell_code_bits(self.cols, self.bits);
        let mut bytes = Vec::with_capacity(self.row_bytes(weights)?);
        match code_bits {
            0..=4 => bytes.extend(codes.chunks(2).map(|pair| (pair[0] as u8 & 0xF) | (pair.get(1).map_or(0, |c| *c as u8 & 0xF) << 4))),
            5..=8 => bytes.extend(codes.iter().map(|c| *c as u8)),
            _ => bytes.extend(codes.iter().flat_map(|c| (*c as i16).to_le_bytes())),
        }
        for (scale, bias) in factors {
            bytes.extend((scale as i32).to_le_bytes());
            bytes.extend((bias as i32).to_le_bytes());
        }
        Ok(bytes)
    }

    /// The size of one row in the image.
    pub fn row_bytes(&self, weights: &Weights) -> Result<usize, String> {
        let matrix = weights.matrix(&self.name)?;
        let codes = match matrix.cell_code_bits(self.cols, self.bits) {
            0..=4 => self.cols.div_ceil(2),
            5..=8 => self.cols,
            _ => 2 * self.cols,
        };
        Ok(codes + 8 * matrix.cell_groups(self.cols).1)
    }
}

/// One matrix in the cells, in the order they hold them.
#[derive(Clone, Debug)]
struct CellEntry {
    name: String,
    rows: usize,
    cols: usize,
    block: usize,
    /// Columns fed to the cells between two flushes, and how many such groups there are.
    group: usize,
    groups: usize,
    /// Columns that share one scale and bias: `group` itself, or a multiple of it when
    /// the quantisation groups are cut so that the units share the columns evenly.
    quant: usize,
    /// log2 of the codes each word of the cells holds: narrower codes share a word.
    packing: u32,
    exponent: i64,
    /// Its rows follow those of the entry before, in the same passes of the cells.
    chained: bool,
    /// The first row of the whole matrix that this chip holds, when chips share it by rows.
    first_row: usize,
    /// Runs of rows of the whole matrix, `(first, count)`, when this chip's are not one:
    /// those of its heads in each part of a projection that makes several.
    pieces: Vec<(usize, usize)>,
}

/// Entries that share passes of the cells, with where each one's rows start and how many
/// rows a pass covers: a single matrix used a block at a time passes over each block.
fn cell_runs(plan: &[CellEntry], lanes: u32) -> Vec<(&[CellEntry], Vec<usize>, usize)> {
    let mut runs = Vec::new();
    let mut start = 0;
    for end in 1..=plan.len() {
        if end == plan.len() || !plan[end].chained {
            let parts = &plan[start..end];
            if let [single] = parts {
                runs.push((parts, vec![0], single.block));
            } else {
                let (starts, total) = chained_rows(&parts.iter().map(|e| e.rows as u32).collect::<Vec<_>>(), lanes);
                runs.push((parts, starts.into_iter().map(|s| s as usize).collect(), total as usize));
            }
            start = end;
        }
    }
    runs
}

/// The buffers an operation reads.
fn buffers_read(op: &Op) -> Vec<usize> {
    match op {
        Op::EmbRead { .. } => vec![],
        Op::Matvec { src, .. } | Op::Copy { src, .. } | Op::Silu { src, .. } | Op::Gelu { src, .. } | Op::Scale { src, .. } | Op::AddWeight { src, .. } | Op::RmsNorm { src, .. } | Op::SplitQueryGate { src, .. } | Op::SoftCap { src, .. } | Op::Sigmoid { src, .. } | Op::Dconv { src, .. } | Op::Slice { src, .. } | Op::SigmoidMoe { src, .. } | Op::Moe { src, .. } => vec![*src],
        Op::Add { a, b, .. } | Op::MulElem { a, b, .. } | Op::Concat { a, b, .. } => vec![*a, *b],
        Op::Rope { buf, .. } => vec![*buf],
        Op::KvAppend { key, value, .. } => vec![*key, *value],
        Op::AttnSum { query, .. } => vec![*query],
        Op::DeltaStep { mixed, a, b, .. } => vec![*mixed, *a, *b],
        Op::RmsGated { src, z, .. } => vec![*src, *z],
        Op::Mla { query, latent, rope_key, .. } => vec![*query, *latent, *rope_key],
        Op::LayerNorm { src, .. } => vec![*src],
        Op::LoadRow { store, .. } | Op::LoadRows { store, .. } => vec![*store],
        Op::StoreRow { src, store } => vec![*src, *store],
        Op::GroupNorm { src, sums, squares, .. } => vec![*src, *sums, *squares],
    }
}

/// Buffers that can stay a slice per unit: each is made by a step that works in slices
/// of its length, and read only by such steps, which take the same slices.
fn spread_buffers(program: &Program) -> HashSet<usize> {
    let ops: Vec<&Op> = all_ops(program).map(|i| &i.op).collect();
    // A norm over the whole vector at once also works in slices.
    let in_slices = |op: &Op| match op {
        Op::Silu { .. } | Op::Sigmoid { .. } | Op::MulElem { .. } | Op::Add { .. } | Op::Dconv { .. } => true,
        Op::RmsNorm { src, block, .. } => program.buffers[*src] == *block,
        _ => false,
    };
    let mut kept: HashSet<usize> = [program.logits, program.final_hidden].into_iter().collect();
    if let Some(draft) = &program.draft {
        kept.extend([draft.hidden_in, draft.hidden_out]);
    }
    if let Some(image) = &program.image {
        kept.extend([image.logits, image.input]);
    }
    let made: HashMap<usize, &Op> = ops.iter().filter_map(|op| match op {
        Op::Matvec { dst, .. } | Op::Silu { dst, .. } | Op::Sigmoid { dst, .. } | Op::MulElem { dst, .. } | Op::Add { dst, .. } | Op::Dconv { dst, .. } => Some((*dst, *op)),
        _ => None,
    }).collect();
    made.keys().copied().filter(|buffer| {
        let length = program.buffers[*buffer];
        let mut readers = ops.iter().filter(|op| buffers_read(op).contains(buffer)).peekable();
        !kept.contains(buffer) && readers.peek().is_some() && readers.all(|op| in_slices(op) && buffers_read(op).iter().all(|b| program.buffers[*b] == length))
    }).collect()
}

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 { a } else { gcd(b, a % b) }
}

impl CellEntry {
    /// The quantisation groups that unit `unit` has a part of, in its share of the columns.
    fn quant_groups_of(&self, unit: usize, units: usize) -> Range<usize> {
        let share = self.groups.div_ceil(units) * self.group;
        let first = unit * share;
        first / self.quant..(first + share).div_ceil(self.quant)
    }

    /// Scales and biases each cell keeps for one pass: as many as the unit with most.
    fn factors_per_pass(&self, units: usize) -> usize {
        if self.quant == self.group {
            return self.groups.div_ceil(units);
        }
        (0..units).map(|unit| self.quant_groups_of(unit, units).len()).max().unwrap_or(0)
    }
}

/// The address of each matrix in the cells, and how deep their two memories must be.
fn place_in_cells(plan: &[CellEntry], cells: usize, units: usize, lanes: u32) -> (HashMap<String, CellMatrix>, u64, u64) {
    let per_unit = (cells / units).max(1);
    let (mut weights, mut factors) = (0u64, 0u64);
    let mut places = HashMap::new();
    for (parts, _, block) in cell_runs(plan, lanes) {
        let e = &parts[0];
        let quant_bits = if e.quant == e.group { 0 } else { e.quant.trailing_zeros() };
        for part in parts {
            places.insert(part.name.clone(), CellMatrix { weights: weights as u32, factors: factors as u32, group: e.group as u32, groups: e.groups as u32, quant_bits, packing: e.packing });
        }
        let blocks = if parts.len() == 1 { e.rows / e.block } else { 1 };
        let rounds = (blocks * block.div_ceil(per_unit)) as u64;
        let groups = e.groups.div_ceil(units) as u64;
        weights += (rounds * groups * e.group as u64) >> e.packing;
        factors += rounds * e.factors_per_pass(units) as u64;
    }
    (places, weights, factors)
}

fn t_gelu(approximate: bool) -> u32 {
    if approximate { FN_GELU_TANH } else { FN_GELU }
}

/// Routines of the fixed microprogram, in the order of `Microprogram::entries`.
pub const ENTRY_TOKEN: usize = 0;
pub const ENTRY_LOGITS: usize = 1;
pub const ENTRY_FEED_FROM_MAIN: usize = 2;
pub const ENTRY_FEED_FROM_DRAFT: usize = 3;
pub const ENTRY_DRAFT: usize = 4;
pub const ENTRY_DRAFT_LOGITS: usize = 5;
/// A token that comes from an image, then the encoder's routines in `VisionPart::routines` order.
pub const ENTRY_TOKEN_FROM_IMAGE: usize = 6;
pub const ENTRY_VISION: usize = 7;

/// Cosines and sines for one position, with the 1 that leaves the other dimensions alone.
fn rope_row(theta: f32, scaling: Option<RopeScaling>, half: usize, position: u32) -> Vec<f64> {
    let rotary = 2 * half;
    let (mut cos, mut sin) = (Vec::new(), Vec::new());
    for i in 0..half {
        let base = 1.0 / (theta as f64).powf((2 * i) as f64 / rotary as f64);
        let frequency = scaling.map_or(base, |s| s.frequency(base));
        cos.push((position as f64 * frequency).cos());
        sin.push((position as f64 * frequency).sin());
    }
    let mut table = cos;
    table.extend(sin.iter().map(|s| -s));
    table.extend(sin.iter());
    table.push(1.0);
    table
}

/// Values as mantissas of `bits` bits at a fixed `exponent`, the largest kept in range.
pub fn encode_at(values: &[f64], bits: u32, exponent: i64) -> Vec<i64> {
    let top = (1i64 << (bits - 1)) - 1;
    let scale = 2f64.powi(bits as i32 - 1 - exponent as i32);
    values.iter().map(|v| ((v * scale).round() as i64).clamp(-top, top)).collect()
}

/// The row of the encoder's rotation table for a patch at `angles`: their cosines, minus
/// their sines, their sines and a one for the elements past the rotated ones.
pub fn patch_rotation_row(angles: &[f64]) -> Vec<f64> {
    let mut row: Vec<f64> = angles.iter().map(|a| a.cos()).collect();
    row.extend(angles.iter().map(|a| -a.sin()));
    row.extend(angles.iter().map(|a| a.sin()));
    row.push(1.0);
    row
}

/// The exponent the encoder's rotation table keeps: its values reach 1.
pub const PATCH_ROTATION_EXPONENT: i64 = 1;

/// Cosines and sines for a token whose rotary frequencies follow three positions, as M-RoPE
/// interleaves them: frequency `i` takes the row's if `i % 3 == 1` and the column's if
/// `i % 3 == 2`, as far as their sections reach, and time's otherwise.
fn mixed_rope_row(theta: f32, scaling: Option<RopeScaling>, half: usize, positions: [u64; 3], section: [usize; 3]) -> Vec<f64> {
    let rows: Vec<Vec<f64>> = positions.iter().map(|p| rope_row(theta, scaling, half, *p as u32)).collect();
    let mut table = rows[0].clone();
    for i in 0..half {
        let from = match i % 3 {
            1 if i < 3 * section[1] => 1,
            2 if i < 3 * section[2] => 2,
            _ => 0,
        };
        for offset in [0, half, 2 * half] {
            table[offset + i] = rows[from][offset + i];
        }
    }
    table
}

/// Where everything lives in the register file, stable across tokens.
pub struct Layout {
    config: DieConfig,
    buffers: Vec<u32>,
    /// Buffers whose register a later buffer takes over within the token.
    overwritten: Vec<bool>,
    weights: HashMap<String, u32>,
    small: HashMap<String, u32>,
    keys: Vec<u32>,
    values: Vec<u32>,
    conv: Vec<u32>,
    delta: Vec<Vec<u32>>,
    next: u32,
    ones: u32,
    minus_one: u32,
    epsilon: u32,
    temporaries: Vec<u32>,
    parts: Vec<u32>,
    /// Each head of a recurrent output keeps its own exponent until it is normalised.
    delta_parts: Vec<u32>,
    delta_outputs: HashSet<usize>,
    /// Buffers that stay a slice per unit: made and read only by element-wise steps
    /// that take the same slices.
    spread: HashSet<usize>,
    rope: [u32; 2],
    rope_table: [u32; 2],
    rope_half: usize,
    /// With every rotation on heads of one size, the cosines and sines of each element
    /// of a head, built once a token: `(block, [(cos, sin)] for each table)`.
    rope_factors: Option<(u32, [(u32, u32); 2])>,
    inverse_root: HashMap<usize, u32>,
    inverse_len: HashMap<usize, u32>,
    setup: Assembly,
    cell_plan: Vec<CellEntry>,
    in_cells: HashMap<String, CellMatrix>,
    lookup_tables: HashSet<String>,
    key_capacity: Option<u32>,
    /// Products of one vector run together in the cells: the first matrix of each chain
    /// gives every matrix of it, with its output buffer; the others do nothing alone.
    chains: HashMap<String, Vec<(String, usize)>>,
    followers: HashSet<String>,
    /// How far into each register the routines assembled so far reach, and the registers
    /// their loops step through by number: what a compact vector memory has to hold.
    extents: RefCell<HashMap<u32, u32>>,
    windows: RefCell<HashMap<u32, u32>>,
    /// Rotations written for particular positions of the cache, in place of the table's: an
    /// image's tokens under M-RoPE, and the text after it, which goes on from further on.
    /// The host writes them into the table the chip reads.
    rotations: RefCell<HashMap<(usize, u32), Vec<f64>>>,
    /// The image encoder's rotations, table 2: the register its row goes into, the table the
    /// host writes for each patch, and half a head.
    vision_rope: Option<(u32, u32, usize)>,
    /// Set while laying out or running a token whose input is the image encoder's output.
    from_image: Cell<bool>,
    /// For each store of rows, the register that keeps each row's exponent, and how wide
    /// its rows are.
    row_exponents: HashMap<usize, (u32, u32)>,
    /// The most tokens and positions the routines can see: a token is also an image
    /// encoder's patch.
    tokens: u32,
    positions: u32,
    /// This chip's place among those that share every product by rows, and the products
    /// they share: the others every chip computes whole.
    shard: Option<Shard>,
    by_rows: HashSet<String>,
    /// The lookup table that stays in the image, its register and its row length: the
    /// register holds only the row the firmware brought in for this token.
    outside: Option<OutsideTable>,
    /// The rotation table when it stays in the image: its register holds only the row of the
    /// token's position, which the firmware brings before each routine that rotates.
    rotation_outside: Option<OutsideTable>,
    /// The group field the routines read across, once one has: see `Assembly::group_field`.
    group_field: Cell<Option<(u32, u32)>>,
    group_used: Cell<bool>,
    /// With the products shared by rows, the attention's heads shared too: this chip's place
    /// in the ring and how many chips, the buffers each chip holds only its heads' slice of,
    /// and those of them that are gathered whole as soon as they are made.
    heads_shard: Option<(u32, u32)>,
    local: HashSet<usize>,
    gather_local: HashSet<usize>,
}

/// Where each register of a compact vector memory starts and its loop step, the words of
/// the whole memory, their log2, and the words each register takes.
struct RegisterPlan {
    bases: Vec<(u32, u32)>,
    words: u64,
    bits: u32,
    sizes: Vec<(u32, u64)>,
}

/// A lookup table the firmware reads from the image a row at a time.
#[derive(Clone, Debug)]
pub struct OutsideTable {
    pub name: String,
    pub register: u32,
    pub cols: u32,
    /// Every row, one after another, as mantissas with the register's exponent.
    pub mantissas: Vec<i64>,
}

/// One of `chips` chips that share each product by rows, in a ring: chip `index` holds
/// rows from `index * rows / chips` of every matrix it shares.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shard {
    pub index: u32,
    pub chips: u32,
    /// The ring is one group of several, joined in a chain: its last chip closes it over
    /// the loop back, and the logits' matrices stay whole on the first chip of all.
    pub grouped: bool,
}

impl Layout {
    /// Allocates registers and stores constants. With `chip_tokens`, the matrix weights
    /// and a RoPE table for that many tokens go into registers for the gates; otherwise
    /// the weights stay in the checkpoint and RoPE is computed directly.
    pub fn new(program: &Program, weights: &Weights, config: DieConfig, chip_tokens: Option<usize>) -> Result<Self, String> {
        Self::build(program, weights, config, chip_tokens, true, false, false)
    }

    /// For the gates: matrices compute in the cells and only lookup tables stay in registers.
    pub fn with_cells(program: &Program, weights: &Weights, config: DieConfig, chip_tokens: usize) -> Result<Self, String> {
        Self::build(program, weights, config, Some(chip_tokens), true, true, false)
    }

    /// The same with the embedding table left in the image from the start, as
    /// `table_outside` leaves it, without ever holding it in the vector memory: a table
    /// larger than the die's addresses reach, such as Show-o's, can only go this way.
    pub fn with_cells_table_outside(program: &Program, weights: &Weights, config: DieConfig, chip_tokens: usize) -> Result<Self, String> {
        Self::build(program, weights, config, Some(chip_tokens), true, true, true)
    }

    /// As a real chip is laid out: matrices live in the cells, not in vector registers.
    pub fn for_image(program: &Program, weights: &Weights, config: DieConfig, chip_tokens: usize) -> Result<Self, String> {
        Self::build(program, weights, config, Some(chip_tokens), false, true, false)
    }

    /// As a large model's chip is laid out: matrices in the cells and the embedding table
    /// in the image, a row brought before each token. Only the layout is wanted, so the
    /// table's rows are not read: the outside table keeps none of them.
    pub fn for_image_table_outside(program: &Program, weights: &Weights, config: DieConfig, chip_tokens: usize) -> Result<Self, String> {
        Self::build(program, weights, config, Some(chip_tokens), false, true, true)
    }

    fn build(program: &Program, weights: &Weights, config: DieConfig, chip_tokens: Option<usize>, matrices_in_registers: bool, cells: bool, table_out: bool) -> Result<Self, String> {
        let mut outside: Option<OutsideTable> = None;
        let next = Cell::new(1u32);
        let alloc = |count: u32| next.replace(next.get() + count);
        // A real chip shares registers among buffers that are never alive at once; the
        // exact engine keeps one each, so every buffer can be read after a token.
        let (buffers, overwritten): (Vec<u32>, Vec<bool>) = if cells {
            let (groups, overwritten) = buffer_groups(program);
            let registers: Vec<u32> = (0..groups.iter().max().map_or(0, |g| g + 1)).map(|_| alloc(1)).collect();
            (groups.iter().map(|g| registers[*g]).collect(), overwritten)
        } else {
            (program.buffers.iter().map(|_| alloc(1)).collect(), vec![false; program.buffers.len()])
        };
        let mut setup = Assembly::new(&config);
        // A store of rows keeps each row with its own exponent, beside it: the rows of an
        // image encoder's state range from a few units to thousands.
        let mut row_exponents: HashMap<usize, (u32, u32)> = HashMap::new();
        for inst in all_ops(program) {
            if let Op::StoreRow { src, store } = inst.op {
                row_exponents.entry(store).or_insert_with(|| (alloc(1), program.buffers[src] as u32));
            }
        }
        let mut matrices: HashMap<String, u32> = HashMap::new();
        let mut by_exponent: HashMap<i64, u32> = HashMap::new();
        let mut cell_plan: Vec<CellEntry> = Vec::new();
        // Consecutive products of the same vector run in one pass of the cells.
        let mut chains: HashMap<String, Vec<(String, usize)>> = HashMap::new();
        let mut leader_of: HashMap<String, String> = HashMap::new();
        let mut uses: HashMap<String, usize> = HashMap::new();
        for inst in all_ops(program) {
            for (name, ..) in matrices_of(program, &inst.op) {
                *uses.entry(name).or_default() += 1;
            }
        }
        let mut previous: Option<&Op> = None;
        let mut lookup_tables: HashSet<String> = HashSet::new();
        let mut in_registers: HashSet<String> = HashSet::new();
        let mut small: HashMap<String, u32> = HashMap::new();
        let mut inverse_root: HashMap<usize, u32> = HashMap::new();
        let mut inverse_len: HashMap<usize, u32> = HashMap::new();

        let root_of = |dim: usize, setup: &mut Assembly, map: &mut HashMap<usize, u32>| {
            map.entry(dim).or_insert_with(|| {
                let reg = alloc(1);
                setup.load_vector(reg, &[1.0 / (dim as f64).sqrt()]);
                reg
            });
        };
        let store = |name: String, values: Vec<f64>, setup: &mut Assembly, map: &mut HashMap<String, u32>| {
            map.entry(name).or_insert_with(|| {
                let reg = alloc(1);
                setup.load_vector(reg, &values);
                reg
            });
        };

        for inst in all_ops(program) {
            for (name, rows, cols) in matrices_of(program, &inst.op) {
                let table = matches!(inst.op, Op::EmbRead { .. });
                if table {
                    lookup_tables.insert(name.clone());
                }
                // A matrix in the cells keeps only its exponent in its register: matrices with
                // the same exponent share one on a real chip.
                let reg = match matrices.get(&name) {
                    Some(reg) => *reg,
                    None => {
                        let reg = if cells && !table { *by_exponent.entry(weights.matrix(&name)?.exponent(cols)).or_insert_with(|| alloc(1)) } else { alloc(1) };
                        matrices.insert(name.clone(), reg);
                        reg
                    }
                };
                if table_out && table && in_registers.insert(name.clone()) {
                    if outside.as_ref().is_some_and(|t| t.name != name) {
                        return Err("only one embedding table can stay in the image".into());
                    }
                    if cols > 1 << config.index_bits {
                        return Err(format!("a row of {name} has {cols} elements and addressing only reaches {}", 1usize << config.index_bits));
                    }
                    let matrix = weights.matrix(&name)?;
                    let exponent = matrix.exponent(cols);
                    // A layout for the image alone reads only the first row.
                    let kept = if matrices_in_registers { rows } else { 1 };
                    let mut mantissas = Vec::with_capacity(kept * cols);
                    for r in 0..kept {
                        mantissas.extend(matrix.fixed_row(r, cols, exponent, config.cell.activation_bits)?);
                    }
                    // The register holds one row, the one the firmware brings before each token.
                    setup.load_mantissas(reg, &mantissas[..cols], exponent);
                    if !matrices_in_registers {
                        mantissas.clear();
                    }
                    outside = Some(OutsideTable { name: name.clone(), register: reg, cols: cols as u32, mantissas });
                } else if chip_tokens.is_some() && matrices_in_registers && (!cells || table) && in_registers.insert(name.clone()) {
                    if rows * cols > 1 << config.index_bits {
                        return Err(format!("matrix {name} has {} elements and addressing only reaches {}", rows * cols, 1usize << config.index_bits));
                    }
                    let matrix = weights.matrix(&name)?;
                    let exponent = matrix.exponent(cols);
                    let mut flat = Vec::with_capacity(rows * cols);
                    for r in 0..rows {
                        flat.extend(matrix.fixed_row(r, cols, exponent, config.cell.activation_bits)?);
                    }
                    setup.load_mantissas(reg, &flat, exponent);
                }
                if cells && !table && cell_plan.iter().all(|e| e.name != name) {
                    let matrix = weights.matrix(&name)?;
                    let (group, groups) = matrix.cell_groups(cols);
                    // A matrix with no groups of its own is split so every unit gets a part.
                    let units = 1usize << config.features.cluster_bits;
                    let (group, groups) = if groups == 1 && units > 1 && cols.is_multiple_of(units) { (cols / units, units) } else { (group, groups) };
                    let block = rows_per_use(program, &inst.op, &name, rows);
                    // Quantisation groups that do not divide among the units are cut into
                    // parts that do: each unit's cells then feed the same number of columns,
                    // and move on to the next scale and bias where a whole group ends.
                    let share = cols / units;
                    let part = gcd(group, share);
                    let cut = config.features.feeds_cells() && units > 1 && block == rows && !groups.is_multiple_of(units) && cols.is_multiple_of(units) && group.is_power_of_two() && group <= 128 && (2..group).contains(&part) && part >= config.features.factor_steps as usize;
                    let (group, groups, quant) = if cut { (part, cols / part, group) } else { (group, groups, group) };
                    // As many codes in each word of the cells as fit, while each group fills whole words.
                    let code_bits = matrix.cell_code_bits(cols, config.cell.activation_bits);
                    // Weights with no format of their own go in whole, one mantissa each.
                    if code_bits > config.cell.weight_bits {
                        return Err(format!("the cells keep {}-bit weights and {name}, unquantized, needs {code_bits}: give it a weight format or wider cells", config.cell.weight_bits));
                    }
                    let packing = (0..=config.cell.most_packing()).rev().find(|p| code_bits << p <= config.cell.weight_bits && group.is_multiple_of(1 << p)).unwrap_or(0);
                    let exponent = matrix.exponent(cols);
                    setup.load_mantissas(reg, &[], exponent);
                    let chained = match (previous, &inst.op, cell_plan.last()) {
                        (Some(Op::Matvec { matrix: before, src: from, .. }), Op::Matvec { src, dst, .. }, Some(last)) => {
                            last.name == *before && from == src && dst != src && uses[before] == 1 && uses[&name] == 1 && block == rows && last.block == last.rows && (last.cols, last.group, last.groups, last.quant, last.packing) == (cols, group, groups, quant, packing)
                        }
                        _ => false,
                    };
                    if chained {
                        let leader = leader_of.get(&cell_plan.last().expect("previous").name).cloned().expect("chain");
                        chains.get_mut(&leader).expect("chain").push((name.clone(), program_dst(&inst.op)));
                        leader_of.insert(name.clone(), leader);
                    } else if let Op::Matvec { dst, .. } = &inst.op {
                        chains.insert(name.clone(), vec![(name.clone(), *dst)]);
                        leader_of.insert(name.clone(), name.clone());
                    }
                    cell_plan.push(CellEntry { block, name, rows, cols, group, groups, quant, packing, exponent, chained, first_row: 0, pieces: Vec::new() });
                }
            }
            previous = Some(&inst.op);
            // A layer norm's mean needs `1 / block` whatever the block.
            let mean = matches!(inst.op, Op::LayerNorm { .. });
            if let Op::RmsNorm { block, .. } | Op::RmsGated { block, .. } | Op::LayerNorm { block, .. } = &inst.op {
                if mean || !block.is_power_of_two() {
                    inverse_len.entry(*block).or_insert_with(|| {
                        let reg = alloc(1);
                        setup.load_vector(reg, &[1.0 / *block as f64]);
                        reg
                    });
                }
            }
            match &inst.op {
                Op::RmsNorm { weight, offset, .. } => {
                    let name = format!("{weight}.weight");
                    let raw = weights.floats(&name)?;
                    let centred = program.family == "qwen3_5" && raw.iter().sum::<f32>() / raw.len() as f32 > 0.5;
                    let values = raw.iter().map(|w| (*w + if centred { 0.0 } else { *offset }) as f64).collect();
                    store(name, values, &mut setup, &mut small);
                }
                Op::Scale { factor, .. } => store(format!("escala:{factor}"), vec![*factor as f64], &mut setup, &mut small),
                Op::AddWeight { weight, .. } => store(weight.clone(), weights.floats(weight)?.iter().map(|v| *v as f64).collect(), &mut setup, &mut small),
                Op::RmsGated { weight, .. } => {
                    let name = format!("{weight}.weight");
                    let values = weights.floats(&name)?.iter().map(|w| *w as f64).collect();
                    store(name, values, &mut setup, &mut small);
                }
                Op::LayerNorm { weight, .. } => {
                    for part in ["weight", "bias"] {
                        let name = format!("{weight}.{part}");
                        let values = weights.floats(&name)?.iter().map(|w| *w as f64).collect();
                        store(name, values, &mut setup, &mut small);
                    }
                }
                Op::GroupNorm { weight, count, .. } => {
                    for part in ["weight", "bias"] {
                        let name = format!("{weight}.{part}");
                        let values = weights.floats(&name)?.iter().map(|w| *w as f64).collect();
                        store(name, values, &mut setup, &mut small);
                    }
                    store(format!("escala:{}", 1.0 / *count as f32), vec![1.0 / *count as f64], &mut setup, &mut small);
                }
                Op::Dconv { slot, weight, .. } => {
                    // Kept as one row per tap, which is how the history is stored too.
                    let (dim, kernel) = program.conv_slots[*slot];
                    let name = format!("{weight}.weight");
                    let taps = weights.floats(&name)?;
                    let values = (0..kernel).flat_map(|k| (0..dim).map(move |c| (c, k))).map(|(c, k)| taps[c * kernel + k] as f64).collect();
                    store(name, values, &mut setup, &mut small);
                }
                Op::DeltaStep { a_log, dt_bias, key_dim, .. } => {
                    store(a_log.clone(), weights.floats(a_log)?.iter().map(|v| v.exp() as f64).collect(), &mut setup, &mut small);
                    store(dt_bias.clone(), weights.floats(dt_bias)?.iter().map(|v| *v as f64).collect(), &mut setup, &mut small);
                    root_of(*key_dim, &mut setup, &mut inverse_root);
                }
                Op::AttnSum { scale_dim, softcap, .. } => {
                    root_of(*scale_dim, &mut setup, &mut inverse_root);
                    if let Some(cap) = softcap {
                        store(format!("escala:{}", 1.0 / cap), vec![1.0 / *cap as f64], &mut setup, &mut small);
                        store(format!("escala:{cap}"), vec![*cap as f64], &mut setup, &mut small);
                    }
                }
                Op::Mla { nope, rope, .. } => root_of(nope + rope, &mut setup, &mut inverse_root),
                Op::SigmoidMoe { prefix, experts, scaling, .. } => {
                    let name = format!("{prefix}.gate.e_score_correction_bias");
                    // An all-zero correction has no exponent worth aligning to, so it is left out.
                    let correction: Vec<f64> = weights.floats(&name).map_or_else(|_| vec![0.0; *experts], |v| v.iter().map(|x| *x as f64).collect());
                    if correction.iter().any(|v| *v != 0.0) {
                        store(name, correction, &mut setup, &mut small);
                    }
                    store(format!("escala:{scaling}"), vec![*scaling as f64], &mut setup, &mut small);
                }
                Op::SoftCap { cap, .. } => {
                    store(format!("escala:{}", 1.0 / cap), vec![1.0 / *cap as f64], &mut setup, &mut small);
                    store(format!("escala:{cap}"), vec![*cap as f64], &mut setup, &mut small);
                }
                _ => {}
            }
        }

        let state_register = |exponent: i64, setup: &mut Assembly| {
            let reg = alloc(1);
            setup.load_mantissas(reg, &[], exponent);
            reg
        };
        let keys: Vec<u32> = program.kv_slots.iter().map(|_| state_register(STATE_EXPONENT, &mut setup)).collect();
        let values: Vec<u32> = program.kv_slots.iter().map(|_| state_register(STATE_EXPONENT, &mut setup)).collect();
        let conv: Vec<u32> = program.conv_slots.iter().map(|_| state_register(STATE_EXPONENT, &mut setup)).collect();
        let delta: Vec<Vec<u32>> = program
            .ops
            .iter()
            .filter_map(|i| if let Op::DeltaStep { value_heads, .. } = i.op { Some(value_heads) } else { None })
            .map(|heads| (0..heads).map(|_| state_register(0, &mut setup)).collect())
            .collect();

        // Attention adds its weights over every token with the ones, and a mixture over
        // every expert: as many ones as either.
        let experts = all_ops(program).filter_map(|i| if let Op::Moe { experts, .. } | Op::SigmoidMoe { experts, .. } = i.op { Some(experts) } else { None }).max().unwrap_or(0);
        let ones = alloc(1);
        // A layer norm adds up its block with them too.
        let norm_block = all_ops(program).filter_map(|i| if let Op::LayerNorm { block, .. } = i.op { Some(block) } else { None }).max().unwrap_or(0);
        // An image's attention adds up over every patch, which may be more than the text's tokens.
        let patches = program.vision.as_ref().map_or(0, |v| v.patches);
        setup.load_vector(ones, &vec![1.0; chip_tokens.unwrap_or(ONES_LENGTH).max(experts).max(norm_block).max(patches).max(1)]);
        let minus_one = alloc(1);
        setup.load_vector(minus_one, &[-1.0]);
        let epsilon = alloc(1);
        setup.load_vector(epsilon, &[program.rms_epsilon]);
        let rope = [alloc(1), alloc(1)];
        let rope_half = all_ops(program)
            .find_map(|i| match i.op {
                Op::Rope { rotary, table, .. } if table < VISION_TABLE => Some(rotary / 2),
                Op::Mla { rope, .. } => Some(rope / 2),
                _ => None,
            })
            .unwrap_or(0);
        let rope_table = [alloc(1), alloc(1)];
        let mut rope_blocks = all_ops(program).filter_map(|i| if let Op::Rope { block, table, .. } = i.op { (table < VISION_TABLE).then_some(block as u32) } else { None });
        let rope_factors = rope_blocks.next().filter(|first| rope_blocks.all(|b| b == *first)).map(|block| (block, [(alloc(1), alloc(1)), (alloc(1), alloc(1))]));
        if let (Some(tokens), true) = (chip_tokens, rope_half > 0) {
            let tables = [(program.rope_theta, program.rope_scaling), (program.local_rope_theta.unwrap_or(program.rope_theta), None)];
            for (index, (theta, scaling)) in tables.iter().enumerate().take(1 + program.local_rope_theta.is_some() as usize) {
                let mut flat = Vec::with_capacity(tokens * (3 * rope_half + 1));
                let mut exponent = 0;
                for position in 0..tokens {
                    let (row, row_exponent) = encode_vector(&rope_row(*theta, *scaling, rope_half, position as u32), config.cell.activation_bits);
                    flat.extend(row);
                    exponent = row_exponent;
                }
                setup.load_mantissas(rope_table[index], &flat, exponent);
            }
        }
        // The host writes a row of the encoder's table for each patch; its values reach 1.
        let vision_rope = all_ops(program).find_map(|i| if let Op::Rope { rotary, table: VISION_TABLE, .. } = i.op { Some(rotary / 2) } else { None }).map(|half| {
            let (row, table) = (alloc(1), alloc(1));
            setup.load_mantissas(table, &[], PATCH_ROTATION_EXPONENT);
            (row, table, half)
        });
        // Working registers come last, so a die with fewer registers keeps the first of
        // them. Those that do not fit are numbered past every register the die can name:
        // a routine that reaches one fails to assemble.
        let mut room = ((1u32 << config.reg_bits) - 3).saturating_sub(next.get());
        let mut beyond = 1u32 << config.reg_bits;
        let mut working = |count: usize| -> Vec<u32> {
            (0..count)
                .map(|_| {
                    if room > 0 {
                        room -= 1;
                        alloc(1)
                    } else {
                        beyond += 1;
                        beyond - 1
                    }
                })
                .collect()
        };
        let temporaries = working(TEMPORARIES);
        let parts = working(PARTS);
        let has_delta = all_ops(program).any(|i| matches!(i.op, Op::DeltaStep { .. }));
        let delta_parts = working(if has_delta { PARTS } else { 0 });
        let delta_outputs = all_ops(program).filter_map(|i| if let Op::DeltaStep { dst, .. } = i.op { Some(dst) } else { None }).collect();
        let spread = spread_buffers(program);
        let next = next.get();
        if next >= (1 << config.reg_bits) - 2 {
            return Err(format!("registers are missing: {next} are needed"));
        }
        let key_capacity = chip_tokens.filter(|_| cells && config.features.combines()).map(|tokens| match config.features.key_shift {
            0 => (tokens as u32).next_multiple_of(1 << config.features.lane_bits),
            shift => 1 << shift,
        });
        if let (Some(tokens), Some(capacity)) = (chip_tokens, key_capacity) {
            if tokens as u32 > capacity {
                return Err(format!("each key column has room for {capacity} tokens and the context is {tokens}"));
            }
            if let Some(vision) = program.vision.as_ref().filter(|v| v.patches as u32 > capacity) {
                return Err(format!("each key column has room for {capacity} tokens and an image can have {} patches", vision.patches));
            }
        }
        // A pass of the cells drains whole blocks: fewer cells than lanes on a unit would
        // leave lanes idle and its rows out of place.
        let (units, lanes) = (1u32 << config.features.cluster_bits, 1u32 << config.features.lane_bits);
        if cells && config.cells / units < lanes {
            return Err(format!("each unit needs at least {lanes} cells, one per lane, and has {}", config.cells / units));
        }
        // A cell applies one group's factors while it adds up the next, so no group is shorter.
        let steps = config.features.factor_steps as usize;
        if let Some(short) = cell_plan.iter().find(|e| e.group < steps) {
            return Err(format!("{} has groups of {} columns and each cell takes {steps} cycles to apply its factors", short.name, short.group));
        }
        let (in_cells, cell_weights, cell_factors) = place_in_cells(&cell_plan, config.cells as usize, 1 << config.features.cluster_bits, 1 << config.features.lane_bits);
        chains.retain(|_, parts| parts.len() > 1);
        let followers: HashSet<String> = chains.values().flat_map(|parts| parts[1..].iter().map(|(name, _)| name.clone())).collect();
        let reach = 1u64 << config.index_bits.min(config.cell.addr_bits);
        if cell_weights > reach || 2 * cell_factors > 1 << config.cell.addr_bits {
            return Err(format!("the cells need {cell_weights} weight positions and {cell_factors} groups, and only reach {reach}"));
        }
        // The token a routine takes is a row of an embedding table, a patch, or one of the
        // vocabulary: as far as the largest of them, which is how far its registers reach.
        let table_rows = all_ops(program).filter_map(|i| match &i.op {
            Op::EmbRead { table, .. } => Some(program.table_rows(table)),
            _ => None,
        });
        let token_range = table_rows.chain([program.vocab, program.vision.as_ref().map_or(0, |v| v.patches)]).max().unwrap_or(0) as u32;
        let mut layout = Self { config, buffers, overwritten, weights: matrices, small, keys, values, conv, delta, next, ones, minus_one, epsilon, temporaries, parts, delta_parts, delta_outputs, spread, rope, rope_table, rope_half, rope_factors, vision_rope, from_image: Cell::new(false), row_exponents, inverse_root, inverse_len, setup, cell_plan, in_cells, lookup_tables, key_capacity, chains, followers, extents: RefCell::default(), windows: RefCell::default(), rotations: RefCell::default(), tokens: if outside.is_some() { program.vision.as_ref().map_or(program.vocab, |v| v.patches) as u32 } else { token_range }, positions: chip_tokens.unwrap_or(1) as u32, shard: None, by_rows: HashSet::new(), outside, rotation_outside: None, group_field: Cell::new(None), group_used: Cell::new(false), heads_shard: None, local: HashSet::new(), gather_local: HashSet::new() };
        if table_out {
            layout.rotation_out(program, matrices_in_registers);
        }
        Ok(layout)
    }

    pub fn setup(&self) -> &Assembly {
        &self.setup
    }

    pub fn registers_used(&self) -> u32 {
        self.next
    }

    pub fn logits_register(&self, program: &Program) -> u32 {
        self.buffers[program.logits]
    }

    pub fn config(&self) -> DieConfig {
        self.config
    }

    pub fn register_of(&self, buffer: usize) -> u32 {
        self.buffers[buffer]
    }

    /// Whether another buffer takes this one's register later in the token, so it no longer
    /// holds its value once the token is done.
    pub fn is_overwritten(&self, buffer: usize) -> bool {
        self.overwritten[buffer]
    }

    /// Whether a die with several units leaves `buffer` a slice on each, never whole.
    pub fn is_spread(&self, buffer: usize) -> bool {
        self.spread.contains(&buffer)
    }

    /// The whole microprogram: loaded once, then entered with the token and position
    /// held in its parameter registers.
    pub fn microprogram(&self, program: &Program) -> Result<Microprogram, String> {
        let token = Expr::var(Var::Token);
        let position = Expr::var(Var::Position);
        let mut code = DieProgram::new(self.config);
        let routine = |code: &mut DieProgram, build: &mut dyn FnMut(&mut Assembly) -> Result<(), String>| {
            let mut asm = self.assembly();
            build(&mut asm)?;
            self.assemble(code, &asm)
        };
        routine(&mut code, &mut |asm| self.run_range(asm, program, 0..program.logits_from, token, position, true))?;
        code.next_entry();
        routine(&mut code, &mut |asm| self.run_range(asm, program, program.logits_from..program.ops.len(), token, position, false))?;
        code.next_entry();
        routine(&mut code, &mut |asm| {
            self.feed_draft(asm, program, true);
            Ok(())
        })?;
        code.next_entry();
        routine(&mut code, &mut |asm| {
            self.feed_draft(asm, program, false);
            Ok(())
        })?;
        code.next_entry();
        if let Some(draft) = &program.draft {
            routine(&mut code, &mut |asm| self.run_list(asm, program, &draft.ops[..draft.logits_from], token, position, true))?;
            code.next_entry();
            routine(&mut code, &mut |asm| self.run_list(asm, program, &draft.ops[draft.logits_from..], token, position, false))?;
        } else {
            code.next_entry();
        }
        if let Some(vision) = &program.vision {
            // A token that comes from an image: the encoder's output in place of the embedding.
            code.next_entry();
            self.from_image.set(true);
            let made = routine(&mut code, &mut |asm| self.run_range(asm, program, 0..program.logits_from, token, position, true));
            self.from_image.set(false);
            made?;
            for (i, range) in vision.routines().into_iter().enumerate() {
                code.next_entry();
                let rope = i % 2 == 1 && i < 2 * vision.blocks.len();
                routine(&mut code, &mut |asm| self.run_list(asm, program, &vision.ops[range.clone()], token, position, rope))?;
            }
        }
        Ok(self.finish(code))
    }

    /// A token whose input is the image encoder's last output, at `position`.
    pub fn run_token_from_image<O: BlockOps>(&self, ops: &mut O, program: &Program, position: O::N, want_logits: bool) -> Result<(), String> {
        self.from_image.set(true);
        let done = self.run_token(ops, program, 0.into(), position, want_logits);
        self.from_image.set(false);
        done
    }

    /// The microprogram of one die of several: its part of the model plus the transfers.
    pub fn die_microprogram(&self, program: &Program, plan: &[Range<usize>], die: usize) -> Result<Microprogram, String> {
        let token = Expr::var(Var::Token);
        let position = Expr::var(Var::Position);
        let last = die + 1 == plan.len();
        let first = die == 0;
        let several = plan.len() > 1;
        let mut code = DieProgram::new(self.config);

        let mut asm = self.assembly();
        if !first {
            let input = program.crossing(plan[die].start)?;
            asm.receive(self.reg(input), (program.buffers[input] as u32).into());
        }
        self.run_list(&mut asm, program, &program.ops[plan[die].clone()], token, position, true)?;
        // A token's work only flows down the chain, so the first chip can start the next
        // token of a prompt while the others are still on this one.
        if !last {
            let output = program.crossing(plan[die + 1].start)?;
            asm.send(self.reg(output), (program.buffers[output] as u32).into());
        }
        self.assemble(&mut code, &asm)?;
        code.next_entry();

        // The logits are made on the first chip, which samples them: the last one sends it
        // the final hidden state, far shorter than the logits.
        let mut asm = self.assembly();
        let tail = &program.ops[program.logits_from..];
        if several && (first || last) {
            let hidden = program.crossing(program.logits_from)?;
            let length: Expr = (program.buffers[hidden] as u32).into();
            if last {
                asm.send(self.reg(hidden), length);
            } else {
                asm.receive(self.reg(hidden), length);
            }
        }
        if first {
            self.run_list(&mut asm, program, tail, token, position, false)?;
        }
        self.assemble(&mut code, &asm)?;
        Ok(self.finish(code))
    }

    /// The microcode of member `shard.index` of group `stage` when the chips are in groups
    /// that share each product by rows, joined in a chain by layers (`plan`, one range per
    /// group). The first of a group takes the hidden vector from the group before and every
    /// member passes it on to the next; the last of a group sends its result on to the next
    /// group. The logits are made on the first chip of all, from the final hidden state the
    /// last chip of all sends it.
    pub fn group_microprogram(&self, program: &Program, plan: &[Range<usize>], stage: usize) -> Result<Microprogram, String> {
        let shard = self.shard.filter(|s| s.grouped).ok_or("the mixed split needs the chip's share within its group")?;
        let (member, members) = (shard.index as usize, shard.chips as usize);
        let token = Expr::var(Var::Token);
        let position = Expr::var(Var::Position);
        let (first_stage, last_stage) = (stage == 0, stage + 1 == plan.len());
        let last_member = member + 1 == members;
        let mut code = DieProgram::new(self.config);

        let mut asm = self.assembly();
        if !first_stage {
            let input = program.crossing(plan[stage].start)?;
            let length: Expr = (program.buffers[input] as u32).into();
            asm.receive(self.reg(input), length);
            if !last_member {
                asm.send(self.reg(input), length);
            }
        }
        self.run_list(&mut asm, program, &program.ops[plan[stage].clone()], token, position, true)?;
        if !last_stage && last_member {
            let output = program.crossing(plan[stage + 1].start)?;
            asm.send(self.reg(output), (program.buffers[output] as u32).into());
        }
        self.assemble(&mut code, &asm)?;
        code.next_entry();

        let mut asm = self.assembly();
        let hidden = program.crossing(program.logits_from)?;
        let length: Expr = (program.buffers[hidden] as u32).into();
        if last_stage && last_member {
            asm.send(self.reg(hidden), length);
        }
        // The first group makes the logits together: the final hidden state comes to its first
        // chip, which passes it down the group, and each member makes its rows.
        if first_stage {
            asm.receive(self.reg(hidden), length);
            if !last_member {
                asm.send(self.reg(hidden), length);
            }
            self.run_list(&mut asm, program, &program.ops[program.logits_from..], token, position, false)?;
        }
        self.assemble(&mut code, &asm)?;
        Ok(self.finish(code))
    }

    /// The microprogram laid out so far, with the group field its words read across.
    fn finish(&self, code: DieProgram) -> Microprogram {
        let mut program = code.finish_program();
        if self.group_used.get() {
            program.group = self.group_field.get().unwrap_or_default();
        }
        program
    }

    /// The group field the routines assembled so far read across, `(shift, bits)`: the units
    /// that run the query heads of one key head keep its cache once between them.
    pub fn group_field(&self) -> Option<(u32, u32)> {
        self.group_field.get().filter(|_| self.group_used.get())
    }

    /// Lays out a routine's steps, noting how far into each register they reach.
    fn assemble(&self, code: &mut DieProgram, asm: &Assembly) -> Result<(), String> {
        if asm.group_used {
            self.group_field.set(asm.group_field);
            self.group_used.set(true);
        }
        let steps = &asm.steps;
        let mut ranges = VarRanges::new(self.tokens, self.positions);
        let lanes = 1u32 << self.config.features.lane_bits;
        register_extents(steps, &mut ranges, lanes, self.config.features.key_shift, &mut self.extents.borrow_mut(), &mut self.windows.borrow_mut());
        match self.highest_register() {
            Some(r) if r >= self.next => Err(format!("registers are missing: the routines reach {r} and the die has {}", self.next)),
            _ => {
                code.assemble(steps);
                Ok(())
            }
        }
    }

    /// The highest register the routines assembled so far use, leaving out the wide
    /// memory and the two the die keeps for itself.
    fn highest_register(&self) -> Option<u32> {
        let reserved = [exponent_register(&self.config), (1u32 << self.config.reg_bits) - 2];
        let windows = self.windows.borrow();
        let reached = windows.iter().map(|(first, reach)| first + reach);
        self.extents.borrow().keys().copied().chain(reached).filter(|r| *r != 0 && !reserved.contains(r)).max()
    }

    /// Registers a die needs for the routines assembled so far: every buffer, matrix and
    /// constant, and the working registers as far as the routines reach into them.
    pub fn registers_needed(&self) -> u32 {
        let first_working = self.temporaries.first().copied().unwrap_or(self.next);
        first_working.max(self.highest_register().map_or(0, |r| r + 1))
    }

    /// How many elements of the wide memory the routines assembled so far reach: the steps
    /// that read it name it as register 0, which no buffer takes.
    pub fn wide_reach(&self) -> u32 {
        self.extents.borrow().get(&0).copied().unwrap_or(0)
    }

    /// Where each register starts in a compact vector memory with the step of the loops
    /// that visit it by number, and log2 of the words the memory needs, for the routines
    /// assembled so far. Each register takes only the words they reach, in whole blocks of
    /// lanes; registers a loop steps through sit one after another, a power of two apart.
    pub fn register_bases(&self) -> (Vec<(u32, u32)>, u32) {
        let (bases, _, bits) = self.register_layout();
        (bases, bits)
    }

    /// The registers that reach furthest, for looking into a layout.
    pub fn longest_registers(&self, count: usize) -> Vec<(u32, u32)> {
        let mut list: Vec<(u32, u32)> = self.extents.borrow().iter().map(|(r, e)| (*r, *e)).collect();
        list.sort_by_key(|(_, e)| std::cmp::Reverse(*e));
        list.truncate(count);
        list
    }

    /// Words of vector memory each unit needs for the routines assembled so far.
    pub fn vector_words_used(&self) -> u64 {
        self.register_layout().1
    }

    /// Words of each unit's vector memory by what they hold, for the routines assembled so
    /// far: the cache's keys and values, the rotation table, the recurrent and convolution
    /// state, the constants and the rest.
    pub fn vector_words_by_kind(&self) -> Vec<(&'static str, u64)> {
        let sizes = self.register_layout_sizes().sizes;
        let kind = |r: u32| -> &'static str {
            if self.keys.contains(&r) {
                "keys"
            } else if self.values.contains(&r) {
                "values"
            } else if self.rope_table.contains(&r) || self.vision_rope.is_some_and(|(_, table, _)| table == r) {
                "rotation table"
            } else if self.delta.iter().flatten().any(|d| *d == r) || self.conv.contains(&r) {
                "recurrent state"
            } else if self.small.values().any(|s| *s == r) || self.weights.values().any(|w| *w == r) {
                "constants and tables"
            } else {
                "working registers"
            }
        };
        let mut totals: Vec<(&'static str, u64)> = Vec::new();
        for (register, words) in sizes {
            let name = kind(register);
            match totals.iter_mut().find(|(n, _)| *n == name) {
                Some((_, total)) => *total += words,
                None => totals.push((name, words)),
            }
        }
        totals.sort_by_key(|(_, words)| std::cmp::Reverse(*words));
        totals
    }

    /// The constants and tables that take most of a vector unit's memory, by name, with
    /// the words of each.
    pub fn largest_constants(&self, count: usize) -> Vec<(String, u64)> {
        let sizes: HashMap<u32, u64> = self.register_layout_sizes().sizes.into_iter().collect();
        let mut named: Vec<(String, u64)> = self.small.iter().chain(self.weights.iter()).filter_map(|(name, r)| sizes.get(r).map(|words| (name.clone(), *words))).collect();
        named.sort_by_key(|(_, words)| std::cmp::Reverse(*words));
        named.truncate(count);
        named
    }

    fn register_layout(&self) -> (Vec<(u32, u32)>, u64, u32) {
        let plan = self.register_layout_sizes();
        (plan.bases, plan.words, plan.bits)
    }

    fn register_layout_sizes(&self) -> RegisterPlan {
        let lanes = 1u64 << self.config.features.lane_bits;
        let mut extents = self.extents.borrow().clone();
        // What is loaded before the first routine, and the exponent of every register.
        for (register, index) in self.setup.initial.keys() {
            let entry = extents.entry(*register).or_default();
            *entry = (*entry).max(index + 1);
        }
        let exponents = extents.entry(exponent_register(&self.config)).or_default();
        *exponents = (*exponents).max(self.next);
        let words = |r: u32| (extents.get(&r).copied().unwrap_or(0) as u64).max(1).next_multiple_of(lanes);
        // Runs of registers a loop visits by number; runs that overlap become one.
        let mut runs: Vec<(u32, u32)> = self.windows.borrow().iter().map(|(first, reach)| (*first, first + reach)).collect();
        runs.sort_unstable();
        let mut merged: Vec<(u32, u32)> = Vec::new();
        for (first, last) in runs {
            match merged.last_mut() {
                Some((_, end)) if first <= *end => *end = (*end).max(last),
                _ => merged.push((first, last)),
            }
        }
        let mut bases = vec![(0u32, 0u32); 1 << self.config.reg_bits];
        let mut placed = vec![false; bases.len()];
        let mut sizes = Vec::new();
        let mut next = 0u64;
        for (first, last) in merged {
            let stride = (first..=last).map(words).max().unwrap_or(lanes).next_power_of_two();
            for r in first..=last {
                bases[r as usize] = ((next + (r - first) as u64 * stride) as u32, stride.trailing_zeros());
                placed[r as usize] = true;
                sizes.push((r, stride));
            }
            next += (last - first + 1) as u64 * stride;
        }
        let mut alone: Vec<u32> = extents.keys().copied().filter(|r| !placed[*r as usize]).collect();
        alone.sort_unstable();
        for r in alone {
            bases[r as usize] = (next as u32, 0);
            sizes.push((r, words(r)));
            next += words(r);
        }
        let bits = (u64::BITS - next.max(lanes * 2).saturating_sub(1).leading_zeros()).max(self.config.features.lane_bits + 1);
        RegisterPlan { bases, words: next, bits, sizes }
    }

    /// Forgets the weights of every matrix that the given operations do not use.
    /// The steps a routine of `list` assembles to, to run them on `netlist::block::Behavior`.
    pub fn routine_steps(&self, program: &Program, list: &[Inst], rope: bool) -> Result<Vec<Step>, String> {
        let mut asm = self.assembly();
        self.run_list(&mut asm, program, list, Expr::var(Var::Token), Expr::var(Var::Position), rope)?;
        Ok(asm.steps)
    }

    fn assembly(&self) -> Assembly {
        let mut asm = Assembly::new(&self.config);
        // Every routine's groups use the field the first one chose.
        asm.group_field = asm.group_field.or(self.group_field.get());
        asm.in_cells = self.in_cells.clone();
        asm.key_capacity = self.key_capacity;
        asm.ring = self.shard.map_or(1, |s| s.chips);
        if let Some(shard) = self.shard.filter(|s| s.grouped) {
            (asm.ring_first, asm.ring_last) = (shard.index == 0, shard.index + 1 == shard.chips);
        }
        asm
    }

    /// The microcode of one operation on its own, to see what it costs.
    /// One operation on its own; `first` also loads the token's rotary table, as the
    /// first operation of every token does.
    pub fn op_microprogram(&self, program: &Program, inst: &Inst, first: bool) -> Result<Microprogram, String> {
        let mut asm = self.assembly();
        self.run_list(&mut asm, program, std::slice::from_ref(inst), Expr::var(Var::Token), Expr::var(Var::Position), first)?;
        let mut code = DieProgram::new(self.config);
        self.assemble(&mut code, &asm)?;
        Ok(self.finish(code))
    }

    /// The cycles each operation of a token takes on this die, in program order.
    pub fn cycles_by_op(&self, program: &Program, token: u32, position: u32) -> Result<Vec<(String, u64)>, String> {
        let mut list = Vec::new();
        for (i, inst) in program.ops.iter().enumerate() {
            let code = self.op_microprogram(program, inst, i == 0)?;
            let name = format!("{:?}", inst.op);
            list.push((name.split([' ', '{', '(']).next().unwrap_or("").to_string(), code.cycles(&self.config, 0, token, position)));
        }
        Ok(list)
    }

    /// How deep the weight and factor memories of each cell must be.
    /// The widest code among the matrices in the cells, and the longest group.
    pub fn cell_shape(&self, weights: &Weights) -> Result<(u32, usize), String> {
        let mut shape = (2, 1);
        for e in &self.cell_plan {
            shape = (shape.0.max(weights.matrix(&e.name)?.cell_code_bits(e.cols, self.config.cell.activation_bits)), shape.1.max(e.group));
        }
        Ok(shape)
    }

    /// Whether the program also reads single rows of this matrix from the registers.
    pub fn is_lookup_table(&self, name: &str) -> bool {
        self.lookup_tables.contains(name)
    }

    /// Every matrix in the cells, in the order they hold them.
    pub fn cell_tensors(&self) -> Vec<CellTensor> {
        self.cell_plan.iter().map(|e| CellTensor { name: e.name.clone(), rows: e.rows, cols: e.cols, block: e.block, exponent: e.exponent, cells: self.config.cells as usize, bits: self.config.cell.activation_bits, first_row: e.first_row, pieces: e.pieces.clone() }).collect()
    }

    pub fn cell_depths(&self) -> (u64, u64) {
        let (_, weights, factors) = place_in_cells(&self.cell_plan, self.config.cells as usize, 1 << self.config.features.cluster_bits, 1 << self.config.features.lane_bits);
        (weights, factors)
    }

    /// What `cell` holds: one code per weight and the scale and bias of each group. Each
    /// vector unit's cells hold the columns of its own groups, padded with zeros.
    pub fn cell_contents(&self, weights: &Weights, cell: u32) -> Result<CellRow, String> {
        let units = 1usize << self.config.features.cluster_bits;
        let per_unit = (self.config.cells as usize / units).max(1);
        let (unit, place) = (cell as usize / per_unit, cell as usize % per_unit);
        let (mut codes, mut factors) = (Vec::new(), Vec::new());
        for (parts, starts, block) in cell_runs(&self.cell_plan, 1 << self.config.features.lane_bits) {
            let e = &parts[0];
            let groups = e.groups.div_ceil(units);
            let first = unit * groups;
            let blocks = if parts.len() == 1 { e.rows / e.block } else { 1 };
            // The scales and biases this unit's cells take in each pass, and those its
            // memory holds for the unit with most, which are left empty here.
            let quant_groups = if e.quant == e.group { first..first + groups } else { e.quant_groups_of(unit, units) };
            let spare = e.factors_per_pass(units) - quant_groups.len();
            for b in 0..blocks {
                for round in 0..block.div_ceil(per_unit) {
                    let row = round * per_unit + place;
                    // The matrix of the run that has this row, if any: runs pad between them.
                    let owner = parts.iter().zip(&starts).find(|(p, start)| row >= **start && row < *start + if parts.len() == 1 { p.block } else { p.rows });
                    match owner {
                        Some((p, start)) => {
                            let matrix = weights.matrix(&p.name)?;
                            let (w, f) = matrix.cell_row(whole_row(p.first_row, &p.pieces, b * p.block + row - start), p.cols, p.exponent, self.config.cell.activation_bits)?;
                            let count = p.cols.div_ceil(p.quant);
                            let shared = |q: usize| if f.len() == count { f.get(q).copied() } else { (q < count).then(|| f[0]) };
                            factors.extend(quant_groups.clone().map(|q| shared(q).unwrap_or((0, 0))));
                            // Each word keeps `1 << packing` codes of `chunk` bits, the first in the low bits.
                            let chunk = self.config.cell.weight_bits >> p.packing;
                            let mask = (1i64 << chunk) - 1;
                            for g in first..first + groups {
                                for j in (0..p.group).step_by(1 << p.packing) {
                                    let code = |slot: usize| w.get(g * p.group + j + slot).copied().unwrap_or(0);
                                    codes.push(if p.packing == 0 { code(0) } else { (0..1 << p.packing).fold(0, |word, slot| word | (code(slot) & mask) << (slot as u32 * chunk)) });
                                }
                            }
                        }
                        None => {
                            codes.extend(std::iter::repeat_n(0, (groups * e.group) >> e.packing));
                            factors.extend(std::iter::repeat_n((0, 0), quant_groups.len()));
                        }
                    }
                }
            }
            factors.extend(std::iter::repeat_n((0, 0), spare * blocks * block.div_ceil(per_unit)));
        }
        Ok((codes, factors))
    }

    pub fn keep_weights_of(&mut self, program: &Program, plan: &[Range<usize>], die: usize) {
        // The first chip makes the logits, as `die_microprogram` lays out.
        self.keep_weights_of_part(program, plan, die, die == 0);
    }

    /// Like `keep_weights_of` for the part `die` of the plan, with the logits' matrices
    /// only if `logits`: in groups, only the first member of the first group makes them.
    pub fn keep_weights_of_part(&mut self, program: &Program, plan: &[Range<usize>], die: usize, logits: bool) {
        // What goes from one chip to the next leaves whole: the link carries what the first
        // vector unit holds, so a vector left a slice per unit would arrive in part.
        for cut in plan.iter().skip(1).map(|part| part.start).chain([program.logits_from]) {
            if let Ok(buffer) = program.crossing(cut) {
                self.spread.remove(&buffer);
            }
        }
        let tail = program.logits_from..program.ops.len();
        let used: HashSet<String> = plan[die].clone().chain(if logits { tail } else { 0..0 }).flat_map(|i| matrices_of(program, &program.ops[i].op)).map(|(name, ..)| name).collect();
        let mut dropped: HashSet<u32> = self.weights.iter().filter(|(name, _)| !used.contains(*name)).map(|(_, reg)| *reg).collect();
        // So do the norms, convolution taps and decay rates of the layers other chips run.
        let kept: HashSet<String> = plan[die].clone().chain(if logits { program.logits_from..program.ops.len() } else { 0..0 }).map(|i| &program.ops[i].op).chain(all_ops(program).skip(program.ops.len()).map(|i| &i.op)).flat_map(small_of).collect();
        dropped.extend(program.ops.iter().flat_map(|i| small_of(&i.op)).filter(|name| !kept.contains(name)).filter_map(|name| self.small.get(&name).copied()));
        self.setup.initial.retain(|(register, _), _| !dropped.contains(register));
        self.cell_plan.retain(|e| used.contains(&e.name));
        self.in_cells = place_in_cells(&self.cell_plan, self.config.cells as usize, 1 << self.config.features.cluster_bits, 1 << self.config.features.lane_bits).0;
    }

    /// Makes this the layout of chip `shard.index` of a ring that shares every product by
    /// rows: each keeps an equal slice of each matrix that only plain products use, and
    /// whose rows split into whole blocks; the chips then pass the slices around. All the
    /// rest, from the attention to the states, every chip computes whole.
    pub fn share_rows(&mut self, program: &Program, shard: Shard) {
        let chips = shard.chips as usize;
        let lanes = 1usize << self.config.features.lane_bits;
        let mut plain: HashSet<String> = HashSet::new();
        let mut other: HashSet<String> = HashSet::new();
        for inst in all_ops(program) {
            for (name, ..) in matrices_of(program, &inst.op) {
                if matches!(inst.op, Op::Matvec { .. }) { &mut plain } else { &mut other }.insert(name);
            }
        }
        // In groups, the first group makes the logits, sharing their products as the layers'.
        let fits = |e: &CellEntry| plain.contains(&e.name) && !other.contains(&e.name) && e.block == e.rows && e.rows.is_multiple_of(chips * lanes);
        // Products that run in one pass of the cells are shared alike, or not at all: a pass
        // with some that cannot be shared, such as a few rows next to many, becomes two, the
        // shared ones and the rest, each in program order.
        let plan = std::mem::take(&mut self.cell_plan);
        let mut start = 0;
        while start < plan.len() {
            let end = (start + 1..plan.len()).find(|i| !plan[*i].chained).unwrap_or(plan.len());
            let (shared, whole): (Vec<CellEntry>, Vec<CellEntry>) = plan[start..end].iter().cloned().partition(|e| fits(e));
            if shared.is_empty() || whole.is_empty() {
                self.cell_plan.extend(plan[start..end].iter().cloned());
            } else {
                let parts = self.chains.remove(&plan[start].name).expect("a pass of several products is a chain");
                for mut run in [shared, whole] {
                    for (i, e) in run.iter_mut().enumerate() {
                        e.chained = i > 0;
                    }
                    if run.len() > 1 {
                        let chain: Vec<(String, usize)> = run.iter().map(|e| parts.iter().find(|(name, _)| *name == e.name).cloned().expect("in the chain")).collect();
                        self.chains.insert(run[0].name.clone(), chain);
                    }
                    self.cell_plan.extend(run);
                }
            }
            start = end;
        }
        self.followers = self.chains.values().flat_map(|parts| parts[1..].iter().map(|(name, _)| name.clone())).collect();
        let mut start = 0;
        while start < self.cell_plan.len() {
            let end = (start + 1..self.cell_plan.len()).find(|i| !self.cell_plan[*i].chained).unwrap_or(self.cell_plan.len());
            if self.cell_plan[start..end].iter().all(fits) {
                for e in &mut self.cell_plan[start..end] {
                    e.rows /= chips;
                    e.block = e.rows;
                    e.first_row = shard.index as usize * e.rows;
                    self.by_rows.insert(e.name.clone());
                }
            }
            start = end;
        }
        self.shard = Some(shard);
        self.in_cells = place_in_cells(&self.cell_plan, self.config.cells as usize, 1 << self.config.features.cluster_bits, 1 << self.config.features.lane_bits).0;
        self.share_heads(program);
    }

    /// Shares the attention's heads among the chips of the ring as well: the products that
    /// only heads read keep this chip's rows, which are its heads', and everything up to the
    /// attention's output works on those heads alone. What something else reads is gathered
    /// whole as soon as it is made. Only when every attention divides evenly among the chips.
    fn share_heads(&mut self, program: &Program) {
        let Some(Shard { index, chips, .. }) = self.shard else { return };
        let n = chips as usize;
        let ops: Vec<&Op> = all_ops(program).map(|i| &i.op).collect();
        let buffers = &program.buffers;
        let attention: Vec<(usize, usize)> = ops.iter().filter_map(|op| if let Op::AttnSum { heads, kv_heads, .. } = op { Some((*heads, *kv_heads)) } else { None }).collect();
        let divides = |count: usize| count.is_multiple_of(n) && count <= PARTS;
        if n < 2 || attention.is_empty() || attention.iter().any(|(heads, kv)| !divides(*heads) || !divides(*kv)) || program.vision.is_some() || ops.iter().any(|op| matches!(op, Op::Mla { .. })) {
            return;
        }
        // The delta rule's heads go with the chips too, when they divide evenly: each keeps
        // the state of its own heads only.
        let deltas: Vec<(usize, usize)> = ops.iter().filter_map(|op| if let Op::DeltaStep { key_heads, value_heads, .. } = op { Some((*key_heads, *value_heads)) } else { None }).collect();
        let delta_split = deltas.iter().all(|(keys, values)| divides(*keys) && divides(*values));
        let delta_out: HashSet<usize> = ops.iter().filter_map(|op| if let Op::DeltaStep { dst, .. } = op { Some(*dst) } else { None }).collect();
        // Each chip's slice must be whole blocks of the link.
        let block = if self.config.features.wide_link { 1usize << self.config.features.lane_bits } else { 1 };
        let fits = |buffer: usize| buffers[buffer].is_multiple_of(n) && (buffers[buffer] / n).is_multiple_of(block);
        let readers = |buffer: usize| ops.iter().filter(move |op| buffers_read(op).contains(&buffer));
        // The delta rule's projection makes queries, keys and values in one vector, which the
        // convolution and SiLU take channel by channel: a chip that runs some of the heads
        // makes only their channels, the queries', keys' and values' one after the other,
        // when nothing else reads them. The projection's rows that make them are its own.
        let mut sections: HashMap<usize, (String, usize, Vec<usize>)> = HashMap::new();
        let mut packed: HashSet<usize> = HashSet::new();
        for op in ops.iter().filter(|_| delta_split) {
            let Op::DeltaStep { mixed, key_dim, key_heads, value_dim, value_heads, .. } = op else { continue };
            let silu = ops.iter().find_map(|o| match o {
                Op::Silu { src, dst } if dst == mixed => Some(*src),
                _ => None,
            });
            let conv = silu.and_then(|c| ops.iter().find_map(|o| match o {
                Op::Dconv { src, dst, slot, .. } if *dst == c => Some((*src, *slot)),
                _ => None,
            }));
            let matrix = conv.and_then(|(q, _)| ops.iter().find_map(|o| match o {
                Op::Matvec { matrix, dst, .. } if *dst == q && self.by_rows.contains(matrix) => Some(matrix.clone()),
                _ => None,
            }));
            let (Some(c), Some((q, slot)), Some(matrix)) = (silu, conv, matrix) else { continue };
            let parts = vec![key_dim * key_heads, key_dim * key_heads, value_dim * value_heads];
            let divide = parts.iter().all(|part| part.is_multiple_of(n) && (part / n).is_multiple_of(block));
            if divide && buffers[q] == parts.iter().sum::<usize>() && readers(q).count() == 1 && readers(c).count() == 1 && readers(*mixed).all(|r| matches!(r, Op::DeltaStep { .. })) {
                packed.extend([q, c, *mixed]);
                sections.insert(q, (matrix, slot, parts));
            }
        }
        // What runs head by head: whether `op` reads `buffer` as heads.
        let by_heads = |op: &Op, buffer: usize| match op {
            Op::RmsGated { src, z, .. } => delta_split && delta_out.contains(src) && (*z == buffer || *src == buffer),
            Op::SplitQueryGate { src, query, block, .. } => *src == buffer && 2 * (buffers[*query] / block) <= PARTS && divides(buffers[*query] / block),
            Op::RmsNorm { src, block, dst, .. } => *src == buffer && buffers[*dst] > *block && divides(buffers[*dst] / block),
            Op::Rope { buf, block, table, .. } => *buf == buffer && *table < VISION_TABLE && divides(buffers[*buf] / block),
            Op::KvAppend { key, value, window: None, .. } => *key == buffer || *value == buffer,
            Op::AttnSum { query, window: None, .. } => *query == buffer,
            // Element by element: each chip takes its own slice, as long as it is whole
            // blocks of the link.
            Op::Silu { src, dst } | Op::Sigmoid { src, dst } => *src == buffer && fits(*dst),
            Op::MulElem { a, b, dst } => (*a == buffer || *b == buffer) && fits(*dst),
            Op::Dconv { src, .. } => *src == buffer && packed.contains(src),
            Op::DeltaStep { mixed, .. } => *mixed == buffer && packed.contains(mixed),
            _ => false,
        };
        let mut local: HashSet<usize> = HashSet::new();
        for op in &ops {
            match op {
                Op::Matvec { matrix, dst, .. } if self.by_rows.contains(matrix) && readers(*dst).count() > 0 && readers(*dst).all(|r| by_heads(r, *dst)) => {
                    local.insert(*dst);
                }
                Op::SplitQueryGate { src, query, gate, .. } if local.contains(src) => {
                    local.extend([*query, *gate]);
                }
                Op::RmsNorm { src, dst, .. } if local.contains(src) => {
                    local.insert(*dst);
                }
                Op::AttnSum { query, dst, .. } if local.contains(query) => {
                    local.insert(*dst);
                }
                Op::DeltaStep { dst, .. } if delta_split => {
                    local.insert(*dst);
                }
                Op::RmsGated { src, dst, .. } if local.contains(src) => {
                    local.insert(*dst);
                }
                Op::Silu { src, dst } | Op::Sigmoid { src, dst } if local.contains(src) => {
                    local.insert(*dst);
                }
                Op::MulElem { a, b, dst } if local.contains(a) || local.contains(b) => {
                    local.insert(*dst);
                }
                Op::Dconv { src, dst, .. } if local.contains(src) => {
                    local.insert(*dst);
                }
                _ => {}
            }
        }
        // A cache is written and read alike: by heads, or whole.
        for op in &ops {
            if let Op::KvAppend { slot, key, value, .. } = op {
                let query = ops.iter().find_map(|o| if let Op::AttnSum { slot: s, query, .. } = o { (s == slot).then_some(*query) } else { None });
                let sides = [local.contains(key), local.contains(value), query.is_some_and(|q| local.contains(&q))];
                if sides.iter().any(|s| *s != sides[0]) {
                    return;
                }
            }
        }
        if local.is_empty() {
            return;
        }
        if local.iter().any(|b| !fits(*b)) {
            return;
        }
        let gather: HashSet<usize> = local.iter().copied().filter(|b| readers(*b).any(|r| !by_heads(r, *b))).collect();
        // A packed projection takes its rows from each part, and the convolution's taps
        // follow its channels: this chip's, from the place its slice starts.
        for (q, (matrix, slot, parts)) in sections.into_iter().filter(|(q, _)| local.contains(q)) {
            let Some(entry) = self.cell_plan.iter_mut().find(|e| e.name == matrix) else { continue };
            let mut first = 0;
            entry.pieces = parts.iter().map(|part| {
                let piece = (first + index as usize * part / n, part / n);
                first += part;
                piece
            }).collect();
            let channels: Vec<usize> = entry.pieces.iter().flat_map(|(first, count)| *first..first + count).collect();
            let weight = ops.iter().find_map(|o| match o {
                Op::Dconv { src, weight, .. } if *src == q => Some(weight.clone()),
                _ => None,
            });
            let (Some(weight), (dim, kernel)) = (weight, program.conv_slots[slot]) else { continue };
            let taps = self.small[&format!("{weight}.weight")];
            let width = dim / n;
            let moved: Vec<((u32, u32), i64)> = (0..kernel)
                .flat_map(|k| channels.iter().enumerate().map(move |(j, c)| (k, j, *c)))
                .map(|(k, j, c)| ((taps, (k * width + j) as u32), self.setup.initial.get(&(taps, (k * dim + c) as u32)).copied().unwrap_or(0)))
                .collect();
            self.setup.initial.retain(|(register, _), _| *register != taps);
            self.setup.initial.extend(moved.into_iter().filter(|(_, value)| *value != 0));
        }
        (self.heads_shard, self.local, self.gather_local) = (Some((index, chips)), local, gather);
    }

    /// Whether the chips of the ring share the attention's heads.
    pub fn shares_heads(&self) -> bool {
        self.heads_shard.is_some()
    }

    /// Of `count` heads, the first this chip runs and how many: all of them unless the
    /// chips share the heads and `buffer` is a slice of them.
    fn heads_of(&self, buffer: usize, count: u32) -> (u32, u32) {
        match self.heads_shard {
            Some((index, chips)) if self.local.contains(&buffer) => (index * count / chips, count / chips),
            _ => (0, count),
        }
    }

    /// The elements of `buffer` this chip makes, `(first, count)`: all of them unless the
    /// chips share the heads and `buffer` is a slice of them.
    fn slice_of(&self, program: &Program, buffer: usize) -> (u32, u32) {
        let len = program.buffers[buffer] as u32;
        match self.heads_shard {
            Some((index, chips)) if self.local.contains(&buffer) => (index * len / chips, len / chips),
            _ => (0, len),
        }
    }

    /// Leaves the embedding table in the image instead of in the vector memory: before each
    /// token the firmware loads that token's row into the table's register, which then
    /// holds one row. Only one table can stay outside; with none, or with more than one,
    /// nothing changes. Returns whether the table went out.
    /// The rotation table goes too, when the model rotates by position alone.
    pub fn table_outside(&mut self, program: &Program) -> bool {
        let rotation = self.rotation_out(program, true);
        self.embedding_out(program) || rotation
    }

    /// Leaves the rotation table in the image: before each routine that rotates, the
    /// firmware brings the row of the token's position into the table's register, which
    /// then holds one row instead of one for every position of the context. Only a model
    /// that rotates by position alone, with one table, can: an image's tokens take rotations
    /// the host writes into the table. With `keep`, the table's rows stay with the layout
    /// for the image. Returns whether the table went out.
    fn rotation_out(&mut self, program: &Program, keep: bool) -> bool {
        let width = 3 * self.rope_half as u32 + 1;
        let register = self.rope_table[0];
        if self.rotation_outside.is_some() || self.rope_half == 0 || program.vision.is_some() || program.image.is_some() || program.local_rope_theta.is_some() || !self.setup.initial.contains_key(&(register, 0)) {
            return false;
        }
        let rows = self.positions;
        let mut mantissas = Vec::with_capacity(if keep { (rows * width) as usize } else { 0 });
        for i in 0..rows * width {
            let value = self.setup.initial.get(&(register, i)).copied().unwrap_or(0);
            if keep {
                mantissas.push(value);
            }
            if i >= width {
                self.setup.initial.remove(&(register, i));
            }
        }
        self.rotation_outside = Some(OutsideTable { name: "rotation".into(), register, cols: width, mantissas });
        true
    }

    /// The rotation table that stays in the image, if it does.
    pub fn rotation_table_outside(&self) -> Option<&OutsideTable> {
        self.rotation_outside.as_ref()
    }

    fn embedding_out(&mut self, program: &Program) -> bool {
        if self.lookup_tables.len() != 1 {
            return false;
        }
        // The text's embedding table, the largest: an image head's codebook stays inside.
        let text_table = program.ops.iter().find_map(|inst| match &inst.op {
            Op::EmbRead { table, .. } if self.lookup_tables.contains(table) => Some(table.clone()),
            _ => None,
        });
        let name = text_table.or_else(|| self.lookup_tables.iter().next().cloned()).expect("a table");
        let Some(cols) = all_ops(program).find_map(|inst| match &inst.op {
            Op::EmbRead { table, dst } if *table == name => Some(program.buffers[*dst] as u32),
            _ => None,
        }) else {
            return false;
        };
        let register = self.weights[&name];
        let rows = program.table_rows(&name) as u32;
        // A chip without this table's part of the model has nothing of it to leave out.
        if !self.setup.initial.contains_key(&(register, 0)) {
            return false;
        }
        let mantissas = (0..rows * cols).map(|i| self.setup.initial.remove(&(register, i)).unwrap_or(0)).collect();
        self.outside = Some(OutsideTable { name, register, cols, mantissas });
        // Nothing reads by the token past its row now: what remains taken by it, the image's
        // rows with an image part, reaches only as far as those.
        self.tokens = program.vision.as_ref().map_or(program.vocab, |v| v.patches) as u32;
        true
    }

    pub fn outside_table(&self) -> Option<&OutsideTable> {
        self.outside.as_ref()
    }

    /// Runs one routine of the microprogram directly, at software speed.
    pub fn run_routine<O: BlockOps>(&self, ops: &mut O, program: &Program, routine: usize, token: O::N, position: O::N) -> Result<(), String> {
        match routine {
            ENTRY_TOKEN => self.run_range(ops, program, 0..program.logits_from, token, position, true),
            ENTRY_LOGITS => self.run_range(ops, program, program.logits_from..program.ops.len(), token, position, false),
            ENTRY_FEED_FROM_MAIN => {
                self.feed_draft(ops, program, true);
                Ok(())
            }
            ENTRY_FEED_FROM_DRAFT => {
                self.feed_draft(ops, program, false);
                Ok(())
            }
            ENTRY_DRAFT | ENTRY_DRAFT_LOGITS => {
                let draft = program.draft.as_ref().ok_or("this model has no MTP head")?;
                let (list, rope) = if routine == ENTRY_DRAFT { (&draft.ops[..draft.logits_from], true) } else { (&draft.ops[draft.logits_from..], false) };
                self.run_list(ops, program, list, token, position, rope)
            }
            ENTRY_TOKEN_FROM_IMAGE if program.vision.is_some() => self.run_token_from_image(ops, program, position, false),
            // The encoder's routines, as `microprogram` lays them out.
            other if other > ENTRY_TOKEN_FROM_IMAGE && program.vision.is_some() => {
                let vision = program.vision.as_ref().unwrap();
                let index = other - ENTRY_VISION;
                let range = vision.routines().into_iter().nth(index).ok_or_else(|| format!("unknown routine: {other}"))?;
                let rope = index % 2 == 1 && index < 2 * vision.blocks.len();
                self.run_list(ops, program, &vision.ops[range], token, position, rope)
            }
            other => Err(format!("unknown routine: {other}")),
        }
    }

    /// The token's operations before `end`: with the last one left out, the hidden state that
    /// the logits' product would take.
    pub fn run_until<O: BlockOps>(&self, ops: &mut O, program: &Program, token: O::N, position: O::N, end: usize) -> Result<(), String> {
        self.run_range(ops, program, 0..end, token, position, true)
    }

    pub fn run_token<O: BlockOps>(&self, ops: &mut O, program: &Program, token: O::N, position: O::N, want_logits: bool) -> Result<(), String> {
        let limit = if want_logits { program.ops.len() } else { program.logits_from };
        self.run_range(ops, program, 0..limit, token, position, true)
    }

    /// One routine of a program made of several, such as an image encoder's: its operations,
    /// with `rope` loading the rotation of `position` first.
    pub fn run_ops<O: BlockOps>(&self, ops: &mut O, program: &Program, list: &[Inst], token: O::N, position: O::N, rope: bool) -> Result<(), String> {
        self.run_list(ops, program, list, token, position, rope)
    }

    /// A microprogram with one entry for each routine, `(operations, load the rotation)`,
    /// for a program made of several, such as an image encoder's.
    pub fn routines_microprogram(&self, program: &Program, routines: &[(&[Inst], bool)]) -> Result<Microprogram, String> {
        let mut code = DieProgram::new(self.config);
        for (list, rope) in routines {
            let mut asm = self.assembly();
            self.run_list(&mut asm, program, list, Expr::var(Var::Token), Expr::var(Var::Position), *rope)?;
            self.assemble(&mut code, &asm)?;
            code.next_entry();
        }
        Ok(self.finish(code))
    }

    /// What the host writes into the rotation table for the rows set with `set_rotation`, at
    /// the table's exponent: `(register, index)` and mantissa.
    pub fn rotation_writes(&self) -> Vec<((u32, u32), i64)> {
        self.rotations
            .borrow()
            .iter()
            .flat_map(|((table, row), values)| {
                let (register, half) = match (*table, self.vision_rope) {
                    (VISION_TABLE, Some((_, register, half))) => (register, half),
                    _ => (self.rope_table[*table], self.rope_half),
                };
                let width = 3 * half as u32 + 1;
                self.rotation_mantissas(register, values).into_iter().enumerate().map(move |(i, m)| ((register, row * width + i as u32), m)).collect::<Vec<_>>()
            })
            .collect()
    }

    /// A row of rotations as the table in `register` holds it, at that table's exponent.
    fn rotation_mantissas(&self, register: u32, values: &[f64]) -> Vec<i64> {
        let exponent = self.setup.initial.get(&(exponent_register(&self.config), register)).copied().unwrap_or(0);
        encode_at(values, self.config.cell.activation_bits, exponent)
    }

    /// The row of the text's rotation table for a token at `positions`, interleaved under
    /// M-RoPE with `section`, as the die's table holds it.
    pub fn text_rotation(&self, program: &Program, positions: [u64; 3], section: [usize; 3]) -> Vec<i64> {
        self.rotation_mantissas(self.rope_table[0], &mixed_rope_row(program.rope_theta, program.rope_scaling, self.rope_half, positions, section))
    }

    /// The row of the encoder's rotation table for a patch at `angles`, as the host writes it.
    pub fn patch_rotation(&self, angles: &[f64]) -> Vec<i64> {
        let Some((_, register, _)) = self.vision_rope else { return Vec::new() };
        self.rotation_mantissas(register, &patch_rotation_row(angles))
    }

    /// What the firmware needs to run the encoder this layout carries, for a die whose
    /// rotation table is in its registers.
    /// The text's rotation table, its row 0's address, and half the rotated elements of a
    /// row: only a table in registers, which the firmware can rewrite.
    pub fn rope_table(&self) -> Option<(u32, u32)> {
        (self.rotation_outside.is_none() && self.setup.initial.contains_key(&(self.rope_table[0], 0))).then(|| (self.rope_table[0] << self.config.index_bits, self.rope_half as u32))
    }

    pub fn vision_info(&self, program: &Program, config: &crate::vision::VisionConfig) -> Option<netlist::chip::VisionInfo> {
        let part = program.vision.as_ref()?;
        let (_, patch_table, patch_half) = self.vision_rope?;
        let bits = self.config.index_bits;
        let exponents = exponent_register(&self.config);
        let vector = |buffer: usize| (self.reg(buffer) << bits, exponents << bits | self.reg(buffer), program.buffers[buffer] as u32);
        // Only a table in registers can have its rows rewritten for an image's tokens.
        let rope = self.setup.initial.contains_key(&(self.rope_table[0], 0)).then(|| self.rope_table[0] << bits);
        Some(netlist::chip::VisionInfo {
            patches: part.patches as u32,
            image_token: config.image_token,
            merge: config.merge as u32,
            blocks: part.blocks.len() as u32,
            pixels: vector(part.pixels),
            position: vector(part.position),
            patch_rope: (patch_table << bits, 3 * patch_half as u32 + 1),
            rope: rope.unwrap_or(0),
            rope_half: if rope.is_some() { self.rope_half as u32 } else { 0 },
            mrope_rows: config.mrope_section[1] as u32,
            mrope_cols: config.mrope_section[2] as u32,
        })
    }

    /// The rotation of `slot` by its own angles, one for each pair of a head: an image patch's
    /// row and column. The host writes it in the table the die reads by position.
    pub fn set_rotation(&self, slot: u32, angles: &[f64]) {
        self.rotations.borrow_mut().insert((VISION_TABLE, slot), patch_rotation_row(angles));
    }

    fn run_range<O: BlockOps>(&self, ops: &mut O, program: &Program, range: Range<usize>, token: O::N, position: O::N, rope: bool) -> Result<(), String> {
        self.run_list(ops, program, &program.ops[range], token, position, rope)
    }

    fn run_list<O: BlockOps>(&self, ops: &mut O, program: &Program, list: &[Inst], token: O::N, position: O::N, rope: bool) -> Result<(), String> {
        if rope {
            match self.vision_rope.filter(|_| list.iter().any(|i| matches!(i.op, Op::Rope { table: VISION_TABLE, .. }))) {
                // The encoder's routines rotate by the patch's row, which the host wrote.
                Some((row, table, half)) => {
                    let rotations = self.rotations.borrow();
                    ops.load_row(table, position, 3 * half as u32 + 1, row, |r| rotations.get(&(VISION_TABLE, r)).cloned().unwrap_or_else(|| rope_row(1.0, None, half, 0)));
                }
                None => self.load_rope(ops, program, position),
            }
        }
        for inst in list {
            self.lower_op(ops, program, &inst.op, token, position)?;
        }
        Ok(())
    }

    pub fn run_draft<O: BlockOps>(&self, ops: &mut O, program: &Program, token: O::N, position: O::N, want_logits: bool) -> Result<(), String> {
        let draft = program.draft.as_ref().ok_or("this model has no MTP head")?;
        let limit = if want_logits { draft.ops.len() } else { draft.logits_from };
        self.run_list(ops, program, &draft.ops[..limit], token, position, true)
    }

    /// The rotation of the cache's position `slot`: that of `positions`, three under M-RoPE
    /// with `section`, or the table's own when they are all `slot`.
    pub fn place_rotation(&self, program: &Program, slot: u32, positions: [u64; 3], section: Option<[usize; 3]>) -> Result<(), String> {
        let mut rotations = self.rotations.borrow_mut();
        if positions == [slot as u64; 3] {
            rotations.remove(&(0, slot));
            return Ok(());
        }
        if program.local_rope_theta.is_some() {
            return Err("this model has two rotation tables and only the main one is rewritten".into());
        }
        let section = match section {
            Some(section) => section,
            None if positions[1] == positions[0] && positions[2] == positions[0] => [self.rope_half, 0, 0],
            None => return Err("three different positions without the M-RoPE sections".into()),
        };
        rotations.insert((0, slot), mixed_rope_row(program.rope_theta, program.rope_scaling, self.rope_half, positions, section));
        Ok(())
    }

    /// Back to the table's own rotations everywhere.
    pub fn clear_rotations(&self) {
        self.rotations.borrow_mut().clear();
    }

    /// An image head's scores of each code, from the final hidden state the last token left.
    pub fn run_image_logits<O: BlockOps>(&self, ops: &mut O, program: &Program) -> Result<(), String> {
        let image = program.image.as_ref().ok_or("this model does not draw")?;
        self.run_list(ops, program, &image.logits_ops, 0.into(), 0.into(), false)
    }

    /// The vector that goes in for a drawn code, in place of a token's embedding.
    pub fn run_image_input<O: BlockOps>(&self, ops: &mut O, program: &Program, code: O::N) -> Result<(), String> {
        let image = program.image.as_ref().ok_or("this model does not draw")?;
        self.run_list(ops, program, &image.input_ops, code, 0.into(), false)
    }

    pub fn feed_draft<O: BlockOps>(&self, ops: &mut O, program: &Program, from_main: bool) {
        let Some(draft) = &program.draft else { return };
        let source = if from_main { program.final_hidden } else { draft.hidden_out };
        ops.copy_part(self.reg(source), 0.into(), self.reg(draft.hidden_in), 0.into(), (program.hidden as u32).into());
    }

    fn load_rope<O: BlockOps>(&self, ops: &mut O, program: &Program, position: O::N) {
        if self.rope_half == 0 {
            return;
        }
        let (theta, scaling, half) = (program.rope_theta, program.rope_scaling, self.rope_half);
        let rotations = self.rotations.borrow();
        // A table left in the image holds just this position's row, put there by the firmware.
        let row = if self.rotation_outside.is_some() { 0.into() } else { position };
        ops.load_row(self.rope_table[0], row, 3 * half as u32 + 1, self.rope[0], |row| rotations.get(&(0, row)).cloned().unwrap_or_else(|| rope_row(theta, scaling, half, row)));
        if let Some(local) = program.local_rope_theta {
            ops.load_row(self.rope_table[1], position, 3 * half as u32 + 1, self.rope[1], |row| rope_row(local, None, half, row));
        }
        // Every rotation of the token uses the same factors, so they are laid out once.
        if let Some((block, factors)) = self.rope_factors {
            for (table, (cos, sin)) in factors.iter().enumerate().take(1 + program.local_rope_theta.is_some() as usize) {
                self.rope_factors_into(ops, self.rope[table], block, self.rope_half as u32, (*cos, *sin));
            }
        }
    }

    /// `cos` and `sin` for each element of a head of `d`: the rotated pairs take their
    /// angle's, and the elements past the rotated part are kept as they are.
    fn rope_factors_into<O: BlockOps>(&self, ops: &mut O, rope: u32, d: u32, h: u32, (cos, sin): (u32, u32)) {
        let zero: O::N = 0.into();
        let rotary = 2 * h;
        ops.copy_part(rope, zero, cos, zero, h.into());
        ops.copy_part(rope, zero, cos, h.into(), h.into());
        ops.copy_part(rope, h.into(), sin, zero, h.into());
        ops.copy_part(rope, rotary.into(), sin, h.into(), h.into());
        ops.each((d - rotary).into(), |ops, j| ops.move_element((rope, (3 * h).into()), (cos, j + rotary.into())));
    }

    fn reg(&self, buf: usize) -> u32 {
        self.buffers[buf]
    }

    /// `rms_blocks` for the `mine` blocks from `first` of `count`: those of a chip that shares
    /// the heads with others.
    fn rms_blocks_of<O: BlockOps>(&self, ops: &mut O, src: u32, weight: Option<u32>, block: usize, (first_block, mine, count): (usize, usize, usize), dst: u32) {
        let t = &self.temporaries;
        let d = block as u32;
        let first = self.parts[0];
        // One long block is sliced among the units, its sum of squares added across them.
        if count == 1 {
            let w = weight;
            return ops.in_slabs(d, &[dst], |ops, start, n| {
                ops.copy_part(src, start, t[40], 0.into(), n.into());
                ops.normalize(t[40], n.into());
                ops.rms_normalize_part(t[40], t[41], (n, d), (t[42], t[43]), self.inverse_len.get(&block).copied(), Some(self.epsilon));
                match w {
                    Some(w) => {
                        ops.copy_part(w, start, t[44], 0.into(), n.into());
                        ops.multiply(t[41], t[44], t[45], n.into());
                    }
                    None => ops.copy_part(t[41], 0.into(), t[45], 0.into(), n.into()),
                }
                ops.copy_part(t[45], 0.into(), dst, start, n.into());
            });
        }
        // Each block is normalised on its own, so several vector units can share them.
        let shared = mine < count;
        ops.heads_across_chips(shared);
        ops.each_head(mine as u32, 1, |ops, local, _| {
            let b = local + (first_block as u32).into();
            ops.with_window(b, &[first], |ops, offset| {
                let part = first + offset;
                ops.copy_part(src, b * d, t[40], 0.into(), d.into());
                ops.normalize(t[40], d.into());
                ops.rms_normalize(t[40], t[41], d, (t[42], t[43]), self.inverse_len.get(&block).copied(), Some(self.epsilon));
                match weight {
                    Some(w) => ops.multiply(t[41], w, part, d.into()),
                    None => ops.copy_part(t[41], 0.into(), part, 0.into(), d.into()),
                }
                ops.align_heads(part, d);
            });
        });
        let parts: Vec<(u32, u32)> = (first_block..first_block + mine).map(|b| (self.parts[b], d)).collect();
        ops.gather_heads_at(&parts, dst, first_block as u32 * d);
        ops.heads_across_chips(false);
    }

    /// `a + b` or `a · b`, a slice per unit where the die has several.
    fn pairwise<O: BlockOps>(&self, ops: &mut O, product: bool, (a, b): (u32, u32), dst: u32, (base, len): (u32, u32)) {
        let t = &self.temporaries;
        ops.in_slabs_at(base, len, &[dst], |ops, start, count| {
            let (x, y, z) = (t[58], t[59], t[60]);
            ops.copy_part(a, start, x, 0.into(), count.into());
            ops.copy_part(b, start, y, 0.into(), count.into());
            if product {
                ops.multiply(x, y, z, count.into());
            } else {
                ops.add(x, y, z, count.into());
            }
            ops.copy_part(z, 0.into(), dst, start, count.into());
        });
    }

    fn silu<O: BlockOps>(&self, ops: &mut O, src: u32, dst: u32, len: u32) {
        self.silu_at(ops, src, dst, (0, len));
    }

    /// SiLU of the `len` elements from `base`.
    fn silu_at<O: BlockOps>(&self, ops: &mut O, src: u32, dst: u32, (base, len): (u32, u32)) {
        let t = &self.temporaries;
        ops.in_slabs_at(base, len, &[dst], |ops, start, count| {
            let (x, sigmoid, y) = (t[58], t[39], t[59]);
            ops.copy_part(src, start, x, 0.into(), count.into());
            ops.function(FN_SIGMOID, x, sigmoid, count.into());
            ops.multiply(sigmoid, x, y, count.into());
            ops.copy_part(y, 0.into(), dst, start, count.into());
        });
    }

    fn lower_op<O: BlockOps>(&self, ops: &mut O, program: &Program, op: &Op, token: O::N, position: O::N) -> Result<(), String> {
        let done = self.lower_one(ops, program, op, token, position);
        ops.end_spread();
        // A slice of heads something else reads becomes whole again.
        if let Some((index, chips)) = self.heads_shard {
            let made: Vec<usize> = match op {
                Op::SplitQueryGate { query, gate, .. } => vec![*query, *gate],
                Op::RmsNorm { dst, .. } | Op::AttnSum { dst, .. } | Op::DeltaStep { dst, .. } | Op::RmsGated { dst, .. } => vec![*dst],
                Op::Silu { dst, .. } | Op::Sigmoid { dst, .. } | Op::MulElem { dst, .. } => vec![*dst],
                _ => Vec::new(),
            };
            for buffer in made.into_iter().filter(|b| self.gather_local.contains(b)) {
                let share = program.buffers[buffer] as u32 / chips;
                ops.all_gather(self.reg(buffer), share, index);
            }
        }
        done
    }

    fn lower_one<O: BlockOps>(&self, ops: &mut O, program: &Program, op: &Op, token: O::N, position: O::N) -> Result<(), String> {
        let made: Vec<usize> = match op {
            Op::Matvec { matrix, dst, .. } => self.chains.get(matrix).map_or_else(|| vec![*dst], |parts| parts.iter().map(|(_, dst)| *dst).collect()),
            Op::Silu { dst, .. } | Op::Sigmoid { dst, .. } | Op::MulElem { dst, .. } | Op::Add { dst, .. } | Op::Dconv { dst, .. } => vec![*dst],
            _ => Vec::new(),
        };
        for buffer in made.into_iter().filter(|b| self.spread.contains(b)) {
            ops.leave_spread(self.reg(buffer));
        }
        let t = &self.temporaries;
        let size = |buffer: usize| -> O::N { (program.buffers[buffer] as u32).into() };
        let zero: O::N = 0.into();
        match op {
            Op::EmbRead { dst, .. } if self.from_image.get() => {
                let output = program.vision.as_ref().ok_or("this model does not read images")?.output;
                let len = program.buffers[*dst] as u32;
                ops.copy_part(self.reg(output), zero, self.reg(*dst), zero, len.into());
                ops.normalize(self.reg(*dst), len.into());
            }
            Op::EmbRead { table, dst } => {
                let len = program.buffers[*dst] as u32;
                // A table left in the image holds just this token's row, put there by the firmware.
                let row = if self.outside.as_ref().is_some_and(|t| t.name == *table) { 0.into() } else { token };
                ops.copy_row(table, self.weights[table], row, len, self.reg(*dst));
                ops.normalize(self.reg(*dst), len.into());
            }
            Op::Matvec { matrix, src, dst } if self.by_rows.contains(matrix) => {
                let Shard { index, chips, .. } = self.shard.expect("split by rows");
                let share = |buffer: usize| program.buffers[buffer] as u32 / chips;
                if let Some(chain) = self.chains.get(matrix) {
                    let parts: Vec<(u32, u32, u32, u32)> = chain.iter().map(|(name, dst)| (self.weights[name], self.reg(*dst), share(*dst), index * share(*dst))).collect();
                    ops.matvecs_rows(matrix, &parts, self.reg(*src));
                    // A slice only its heads read stays this chip's.
                    for (_, dst) in chain.iter().filter(|(_, dst)| !self.local.contains(dst)) {
                        ops.all_gather(self.reg(*dst), share(*dst), index);
                    }
                } else if !self.followers.contains(matrix) {
                    ops.matvec_rows(matrix, self.weights[matrix], self.reg(*src), self.reg(*dst), share(*dst), index * share(*dst));
                    if !self.local.contains(dst) {
                        ops.all_gather(self.reg(*dst), share(*dst), index);
                    }
                }
            }
            Op::Matvec { matrix, src, dst } => {
                let (rows, cols) = (program.buffers[*dst] as u32, program.buffers[*src] as u32);
                if let Some(chain) = self.chains.get(matrix) {
                    let parts: Vec<MatvecPart> = chain.iter().map(|(name, dst)| MatvecPart { name: name.clone(), matrix: self.weights[name], dst: self.reg(*dst), rows: program.buffers[*dst] as u32 }).collect();
                    ops.matvecs(&parts, self.reg(*src), cols);
                } else if !self.followers.contains(matrix) {
                    ops.matvec(matrix, self.weights[matrix], 0.into(), self.reg(*src), self.reg(*dst), rows, cols);
                }
            }
            Op::Copy { src, dst } => ops.copy_part(self.reg(*src), zero, self.reg(*dst), zero, size(*dst)),
            Op::Add { a, b, dst } => self.pairwise(ops, false, (self.reg(*a), self.reg(*b)), self.reg(*dst), (0, program.buffers[*dst] as u32)),
            Op::MulElem { a, b, dst } => {
                ops.slices_across_chips(self.local.contains(dst));
                self.pairwise(ops, true, (self.reg(*a), self.reg(*b)), self.reg(*dst), self.slice_of(program, *dst));
                ops.slices_across_chips(false);
            }
            Op::Gelu { src, approximate, dst } => {
                let table = t_gelu(*approximate);
                let extra = self.temporaries[39];
                ops.function(table, self.reg(*src), extra, size(*dst));
                ops.multiply(extra, self.reg(*src), self.reg(*dst), size(*dst));
            }
            Op::Slice { src, from, dst } => ops.copy_part(self.reg(*src), (*from as u32).into(), self.reg(*dst), zero, size(*dst)),
            Op::Mla { slot, query, latent, rope_key, embed, unembed, heads, nope, rope, lora, value_dim, dst, .. } => {
                let (nope, rope, lora, value_dim, query) = (*nope as u32, *rope as u32, *lora as u32, *value_dim as u32, self.reg(*query));
                let (q_head, width) = (nope + rope, lora + rope);
                let store = self.keys[*slot];
                // Latent and rotary key sit side by side, so one dot product scores both.
                ops.copy_part(self.reg(*rope_key), zero, t[3], zero, rope.into());
                ops.normalize(t[3], rope.into());
                self.rotate_pairs(ops, t[3], rope, t[0]);
                ops.gather(&[(self.reg(*latent), lora), (t[0], rope)], t[1]);
                ops.align_to(t[1], zero, STATE_EXPONENT, store, position * width, width.into());
                let tokens = position + 1.into();
                ops.copy_part(store, zero, t[2], zero, tokens * width);
                ops.normalize(t[2], tokens * width);
                let regs = AttentionRegs { scores: t[10], top: t[11], shifted: t[12], weights: t[13], total: t[14], inverse: t[15], probabilities: t[16], softcap: None };
                let scale = self.inverse_root[&(q_head as usize)];
                let first = self.parts[0];
                ops.each((*heads as u32).into(), |ops, h| {
                    ops.with_window(h, &[first], |ops, offset| {
                        ops.copy_part(query, h * q_head, t[3], zero, nope.into());
                        ops.normalize(t[3], nope.into());
                        ops.matvec(embed, self.weights[embed], h * lora, t[3], t[4], lora, nope);
                        ops.copy_part(query, h * q_head + nope.into(), t[5], zero, rope.into());
                        ops.normalize(t[5], rope.into());
                        self.rotate_pairs(ops, t[5], rope, t[6]);
                        ops.gather(&[(t[4], lora), (t[6], rope)], t[7]);
                        ops.attention(t[7], KeyStore { keys: t[2], values: t[2], stride: width, offset: zero, local: false }, self.ones, t[8], width, tokens, Some(scale), regs);
                        ops.matvec(unembed, self.weights[unembed], h * value_dim, t[8], first + offset, value_dim, lora);
                    });
                });
                let parts: Vec<(u32, u32)> = (0..*heads).map(|h| (self.parts[h], value_dim)).collect();
                ops.gather(&parts, self.reg(*dst));
            }
            Op::SigmoidMoe { prefix, src, dst, experts, top_k, intermediate, shared_intermediate, scaling, normalize } => {
                self.lower_sigmoid_moe(ops, prefix, self.reg(*src), self.reg(*dst), program.hidden as u32, *experts as u32, *top_k as u32, *intermediate as u32, *shared_intermediate as u32, *scaling, *normalize);
            }
            Op::SoftCap { src, cap, dst } => {
                let (scaled, shaped) = (self.temporaries[38], self.temporaries[39]);
                ops.scale(self.small[&format!("escala:{}", 1.0 / cap)], self.reg(*src), scaled, size(*dst));
                ops.function(FN_TANH, scaled, shaped, size(*dst));
                ops.scale(self.small[&format!("escala:{cap}")], shaped, self.reg(*dst), size(*dst));
            }
            Op::Scale { src, factor, dst } => ops.scale(self.small[&format!("escala:{factor}")], self.reg(*src), self.reg(*dst), size(*dst)),
            Op::AddWeight { src, weight, dst } => ops.add(self.reg(*src), self.small[weight], self.reg(*dst), size(*dst)),
            Op::Silu { src, dst } => {
                ops.slices_across_chips(self.local.contains(dst));
                self.silu_at(ops, self.reg(*src), self.reg(*dst), self.slice_of(program, *dst));
                ops.slices_across_chips(false);
            }
            Op::Sigmoid { src, dst } => {
                let t = &self.temporaries;
                let (base, len) = self.slice_of(program, *dst);
                ops.slices_across_chips(self.local.contains(dst));
                ops.in_slabs_at(base, len, &[self.reg(*dst)], |ops, start, count| {
                    let (x, y) = (t[58], t[59]);
                    ops.copy_part(self.reg(*src), start, x, 0.into(), count.into());
                    ops.function(FN_SIGMOID, x, y, count.into());
                    ops.copy_part(y, 0.into(), self.reg(*dst), start, count.into());
                });
                ops.slices_across_chips(false);
            }
            Op::RmsNorm { src, weight, block, dst, .. } => {
                let count = program.buffers[*dst] / block;
                let w = self.small[&format!("{weight}.weight")];
                // A chip that shares the heads normalises its own.
                let (first, mine) = self.heads_of(*src, count as u32);
                self.rms_blocks_of(ops, self.reg(*src), Some(w), *block, (first as usize, mine as usize, count), self.reg(*dst));
            }
            Op::LayerNorm { src, weight, block, dst } => {
                let count = (program.buffers[*dst] / block) as u32;
                let d = *block as u32;
                let (w, bias) = (self.small[&format!("{weight}.weight")], self.small[&format!("{weight}.bias")]);
                let (src, inverse, first) = (self.reg(*src), self.inverse_len[block], self.parts[0]);
                // Each block less its mean, then the same norm as RMSNorm, scaled and moved
                // by its weight and bias, in a part of its own: blocks have their own exponents.
                ops.each_head(count, 1, |ops, k, _| {
                    ops.with_window(k, &[first], |ops, offset| {
                        let part = first + offset;
                        ops.copy_part(src, k * d, t[46], zero, d.into());
                        ops.dot(t[46], self.ones, t[47], d.into());
                        ops.scale(inverse, t[47], t[48], 1.into());
                        ops.sub_scalar(t[46], t[48], t[40], d.into());
                        ops.normalize(t[40], d.into());
                        ops.rms_normalize(t[40], t[41], d, (t[42], t[43]), Some(inverse), Some(self.epsilon));
                        ops.multiply(t[41], w, t[44], d.into());
                        ops.add(t[44], bias, part, d.into());
                        ops.align_heads(part, d);
                    });
                });
                let parts: Vec<(u32, u32)> = (0..count as usize).map(|k| (self.parts[k], d)).collect();
                ops.gather_heads(&parts, self.reg(*dst));
            }
            Op::LoadRow { store, dst } => {
                // Several rows side by side, such as the patches of one token, line up on
                // the largest of their exponents.
                let (exponents, row) = self.row_exponents[store];
                let rows = program.buffers[*dst] as u32 / row;
                let (store, dst) = (self.reg(*store), self.reg(*dst));
                if rows == 1 {
                    ops.copy_part(store, token * row, dst, zero, row.into());
                    ops.restore_exponent((exponents, token), dst);
                } else {
                    for k in 0..rows {
                        let at = token * rows + k.into();
                        ops.copy_part(store, at * row, self.parts[k as usize], zero, row.into());
                        ops.restore_exponent((exponents, at), self.parts[k as usize]);
                    }
                    let parts: Vec<(u32, u32)> = (0..rows as usize).map(|k| (self.parts[k], row)).collect();
                    ops.gather(&parts, dst);
                }
                ops.normalize(dst, (rows * row).into());
            }
            Op::LoadRows { store, dst, offsets } => {
                // A pixel's neighbours: rows at fixed distances in a map with a border of
                // zeros, side by side on the largest of their exponents.
                let (exponents, row) = self.row_exponents[store];
                let (store, dst) = (self.reg(*store), self.reg(*dst));
                for (k, offset) in offsets.iter().enumerate() {
                    let at = token + (*offset as u32).into();
                    ops.copy_part(store, at * row, self.parts[k], zero, row.into());
                    ops.restore_exponent((exponents, at), self.parts[k]);
                }
                let parts: Vec<(u32, u32)> = (0..offsets.len()).map(|k| (self.parts[k], row)).collect();
                ops.gather(&parts, dst);
                ops.normalize(dst, (offsets.len() as u32 * row).into());
            }
            Op::GroupNorm { src, sums, squares, count, weight, groups, dst } => {
                let d = (program.buffers[*dst] / groups) as u32;
                let (w, bias) = (self.small[&format!("{weight}.weight")], self.small[&format!("{weight}.bias")]);
                let per_value = self.small[&format!("escala:{}", 1.0 / *count as f32)];
                let (src, sums, squares, first) = (self.reg(*src), self.reg(*sums), self.reg(*squares), self.parts[0]);
                // Each group in a part of its own, as LayerNorm's blocks: its mean and inverse
                // deviation from the sums, then its channels moved, scaled and shifted.
                ops.each_head(*groups as u32, 1, |ops, k, _| {
                    ops.with_window(k, &[first], |ops, offset| {
                        let part = first + offset;
                        ops.copy_part(sums, k, t[47], zero, 1.into());
                        ops.scale(per_value, t[47], t[48], 1.into());
                        ops.copy_part(squares, k, t[49], zero, 1.into());
                        ops.scale(per_value, t[49], t[50], 1.into());
                        ops.multiply(t[48], t[48], t[51], 1.into());
                        ops.sub_scalar(t[50], t[51], t[52], 1.into());
                        ops.add(t[52], self.epsilon, t[53], 1.into());
                        ops.inverse_sqrt(t[53], t[54]);
                        ops.copy_part(src, k * d, t[46], zero, d.into());
                        ops.sub_scalar(t[46], t[48], t[40], d.into());
                        ops.scale(t[54], t[40], t[41], d.into());
                        ops.copy_part(w, k * d, t[42], zero, d.into());
                        ops.multiply(t[41], t[42], t[44], d.into());
                        ops.copy_part(bias, k * d, t[43], zero, d.into());
                        ops.add(t[44], t[43], part, d.into());
                        ops.align_heads(part, d);
                    });
                });
                let parts: Vec<(u32, u32)> = (0..*groups).map(|k| (self.parts[k], d)).collect();
                ops.gather_heads(&parts, self.reg(*dst));
            }
            Op::StoreRow { src, store } => {
                let (exponents, row) = self.row_exponents[store];
                ops.copy_part(self.reg(*src), zero, self.reg(*store), token * row, row.into());
                ops.save_exponent(self.reg(*src), (exponents, token));
            }
            Op::SplitQueryGate { src, query, gate, block } => {
                let heads = program.buffers[*query] / block;
                let d = *block as u32;
                let src = self.reg(*src);
                if 2 * heads <= PARTS {
                    // Each unit takes the two halves of its own head; a chip that shares the
                    // heads, its own heads only.
                    let (first_query, first_gate) = (self.parts[0], self.parts[heads]);
                    let (h0, count) = self.heads_of(*query, heads as u32);
                    let shared = count < heads as u32;
                    ops.heads_across_chips(shared);
                    ops.each_head(count, 1, |ops, local, _| {
                        let h = local + h0.into();
                        ops.with_window(h, &[first_query, first_gate], |ops, offset| {
                            ops.copy_part(src, h * (2 * d), first_query + offset, 0.into(), d.into());
                            ops.copy_part(src, h * (2 * d) + d.into(), first_gate + offset, 0.into(), d.into());
                            ops.align_heads(first_query + offset, d);
                            ops.align_heads(first_gate + offset, d);
                        });
                    });
                    let mine = h0 as usize..(h0 + count) as usize;
                    ops.gather_heads_at(&mine.clone().map(|h| (self.parts[h], d)).collect::<Vec<_>>(), self.reg(*query), h0 * d);
                    ops.gather_heads_at(&mine.map(|h| (self.parts[heads + h], d)).collect::<Vec<_>>(), self.reg(*gate), h0 * d);
                    ops.heads_across_chips(false);
                } else {
                    ops.each((heads as u32).into(), |ops, h| {
                        ops.copy_part(src, h * (2 * d), t[0], h * d, d.into());
                        ops.copy_part(src, h * (2 * d) + d.into(), t[1], h * d, d.into());
                    });
                    ops.gather(&[(t[0], (heads * block) as u32)], self.reg(*query));
                    ops.gather(&[(t[1], (heads * block) as u32)], self.reg(*gate));
                }
            }
            Op::Rope { buf, block, table, .. } => {
                let reg = self.reg(*buf);
                let total = program.buffers[*buf] as u32;
                let (d, h, rope) = match self.vision_rope {
                    Some((row, _, half)) if *table == VISION_TABLE => (*block as u32, half as u32, row),
                    _ => (*block as u32, self.rope_half as u32, self.rope[*table]),
                };
                // One table of factors for each element, so each head is one pair of sweeps.
                let first = self.parts[0];
                let rotary = 2 * h;
                let (cos, sin) = match self.rope_factors {
                    Some((block, factors)) if block == d && *table < VISION_TABLE => factors[*table],
                    _ => {
                        self.rope_factors_into(ops, rope, d, h, (t[50], t[51]));
                        (t[50], t[51])
                    }
                };
                let heads = total / d;
                // A chip that shares the heads rotates its own.
                let (h0, count) = self.heads_of(*buf, heads);
                ops.heads_across_chips(count < heads);
                ops.each_head(count, 1, |ops, local, _| {
                    let head = local + h0.into();
                    ops.with_window(head, &[first], |ops, offset| {
                        let (x, swapped) = (t[52], t[53]);
                        ops.copy_part(reg, head * d, x, zero, d.into());
                        ops.copy_part(x, h.into(), swapped, zero, h.into());
                        ops.copy_part(x, zero, swapped, h.into(), h.into());
                        ops.rotate(x, swapped, (cos, sin), first + offset, d, rotary);
                        ops.align_heads(first + offset, d);
                    });
                });
                let parts: Vec<(u32, u32)> = (h0..h0 + count).map(|part| (self.parts[part as usize], d)).collect();
                ops.gather_heads_at(&parts, reg, h0 * d);
                ops.heads_across_chips(false);
            }
            Op::KvAppend { slot, key, value, window } => {
                let width = program.buffers[*key] as u32;
                let slot_place = match window {
                    Some(limit) => ops.window_slot(position, *limit as u32),
                    None => position,
                };
                let reader = all_ops(program).find_map(|inst| match inst.op {
                    Op::AttnSum { slot: s, heads, kv_heads, head_dim, .. } if s == *slot => Some((heads, kv_heads, head_dim)),
                    _ => None,
                });
                let (keys, values) = (self.keys[*slot], self.values[*slot]);
                let key_buffer = *key;
                let (key, value) = (self.reg(*key), self.reg(*value));
                match reader {
                    // Units hold the columns they read: each writes just those, all at once.
                    Some((heads, kv_heads, head_dim)) if (kv_heads * head_dim) as u32 == width => {
                        let d = head_dim as u32;
                        // A chip that shares the heads keeps its own key heads.
                        let (k0, count) = self.heads_of(key_buffer, kv_heads as u32);
                        ops.each_head_split(count, (heads / kv_heads) as u32, |ops, local, _| {
                            let kv_head = local + k0.into();
                            ops.store_key_part(key, STATE_EXPONENT, keys, slot_place, width, kv_head * d, d);
                            ops.store_key_part(value, STATE_EXPONENT, values, slot_place, width, kv_head * d, d);
                        });
                    }
                    _ => {
                        ops.store_key(key, STATE_EXPONENT, keys, slot_place, width);
                        ops.store_key(value, STATE_EXPONENT, values, slot_place, width);
                    }
                }
            }
            Op::AttnSum { slot, query, head_dim, scale_dim, heads, kv_heads, window, softcap, dst } => {
                let group = heads / kv_heads;
                let tokens = match window {
                    Some(limit) => ops.window_length(position, *limit as u32),
                    None => position + 1.into(),
                };
                let width = (program.kv_slots[*slot].heads * head_dim) as u32;
                let d = *head_dim as u32;
                let (keys, values) = (self.keys[*slot], self.values[*slot]);
                // `KvAppend` gave each unit just its heads' columns when its key is exactly
                // the heads this reads.
                let local = all_ops(program).any(|inst| matches!(inst.op, Op::KvAppend { slot: s, key, .. } if s == *slot && program.buffers[key] == kv_heads * head_dim));
                let capped = softcap.map(|cap| (self.small[&format!("escala:{}", 1.0 / cap)], self.small[&format!("escala:{cap}")]));
                let regs = AttentionRegs { scores: t[10], top: t[11], shifted: t[12], weights: t[13], total: t[14], inverse: t[15], probabilities: t[16], softcap: capped };
                let first = self.parts[0];
                // A chip that shares the heads attends with its own.
                let (k0, kv_count) = self.heads_of(*query, *kv_heads as u32);
                let sharing = kv_count < *kv_heads as u32;
                ops.heads_across_chips(sharing);
                let query = self.reg(*query);
                ops.each_head_split(kv_count, group as u32, |ops, mine, shared| {
                    {
                        let kv_head = mine + k0.into();
                        let h = kv_head * group as u32 + shared;
                        ops.with_window(h, &[first], |ops, offset| {
                            let qh = t[20];
                            ops.copy_part(query, h * d, qh, zero, d.into());
                            ops.normalize(qh, d.into());
                            // A scale of 1/√d that is a power of two only moves the query's
                            // exponent, which costs nothing next to scaling every score.
                            let scale = if scale_dim.is_power_of_two() && scale_dim.trailing_zeros() % 2 == 0 {
                                ops.subtract_exponent(qh, scale_dim.trailing_zeros() as i64 / 2);
                                None
                            } else {
                                Some(self.inverse_root[scale_dim])
                            };
                            let store = KeyStore { keys, values, stride: width, offset: kv_head * d, local };
                            ops.attention(qh, store, self.ones, first + offset, d, tokens, scale, regs);
                            ops.align_heads(first + offset, d);
                        });
                    }
                });
                let h0 = (k0 as usize) * group;
                let parts: Vec<(u32, u32)> = (h0..h0 + kv_count as usize * group).map(|h| (self.parts[h], d)).collect();
                ops.gather_heads_at(&parts, self.reg(*dst), h0 as u32 * d);
                ops.heads_across_chips(false);
            }
            Op::Dconv { slot, weight, src, dst } => {
                let (dim, kernel) = program.conv_slots[*slot];
                let history = (kernel - 1) as u32;
                let taps = self.small[&format!("{weight}.weight")];
                let reg = self.conv[*slot];
                let (dim32, kernel32) = (dim as u32, kernel as u32);
                // A chip that runs some of the delta rule's heads convolves their channels,
                // and keeps their taps and history together, as rows of its own channels.
                let (base, len) = self.slice_of(program, *dst);
                let packed = self.local.contains(dst);
                let width = if packed { len } else { dim32 };
                ops.slices_across_chips(packed);
                let (src, dst) = (self.reg(*src), self.reg(*dst));
                // Each unit keeps the history of its own channels and convolves them.
                ops.in_slabs_at(base, len, &[dst], |ops, start, n| {
                    let channel = if packed { start + base.wrapping_neg().into() } else { start };
                    ops.align_to(src, start, STATE_EXPONENT, reg, channel + (history * width).into(), n.into());
                    ops.depthwise_part_at(taps, reg, dst, width, kernel32, channel, n, start);
                    ops.each(history.into(), |ops, k| ops.copy_elements(reg, (k + 1.into()) * width + channel, k * width + channel, n));
                });
                ops.slices_across_chips(false);
            }
            Op::DeltaStep { slot, mixed, a, b, a_log, dt_bias, key_dim, key_heads, value_dim, value_heads, dst } => {
                let ratio = value_heads / key_heads;
                let (kd, vd) = (*key_dim as u32, *value_dim as u32);
                let key_len = kd * *key_heads as u32;
                let root = self.inverse_root[key_dim];
                // A chip that makes only its heads' channels has their queries, keys and
                // values one after the other from where its slice starts.
                let packed = self.local.contains(mixed);
                let (base, share) = self.slice_of(program, *mixed);
                let chips = if packed { (program.buffers[*mixed] as u32) / share } else { 1 };
                let mixed = self.reg(*mixed);
                let (a_reg, b_reg) = (self.reg(*a), self.reg(*b));
                let (scale_a, bias) = (self.small[a_log], self.small[dt_bias]);
                let (first_state, first_out, first_part) = (self.delta[*slot][0], self.delta_parts[0], self.parts[0]);
                // A chip that shares the heads runs the delta rule of its own.
                let (k0, key_count) = self.heads_of(*dst, *key_heads as u32);
                ops.heads_across_chips(key_count < *key_heads as u32);
                ops.each_head_split(key_count, ratio as u32, |ops, mine, shared| {
                    {
                        let key_head = mine + k0.into();
                        let h = key_head * ratio as u32 + shared;
                        // A unit with a head of its own keeps its state in one register: the
                        // others, one per head, would only take room in every unit.
                        let own = ops.one_head_per_unit();
                        let windowed: &[u32] = if own { &[first_out, first_part] } else { &[first_state, first_out, first_part] };
                        ops.with_window(h, windowed, |ops, offset| {
                            let (qh, kh, vh, beta, decay) = (t[20], t[21], t[22], t[23], t[24]);
                            let (work, inv) = (t[25], t[26]);

                            let (query_at, key_at, value_at) = if packed {
                                let (keys, local) = (key_len / chips, mine * ratio as u32 + shared);
                                (mine * kd + base.into(), mine * kd + (base + keys).into(), local * vd + (base + 2 * keys).into())
                            } else {
                                (key_head * kd, key_head * kd + key_len.into(), h * vd + (2 * key_len).into())
                            };
                            ops.copy_part(mixed, query_at, work, 0.into(), kd.into());
                            ops.normalize(work, kd.into());
                            ops.l2_normalize(work, t[27], kd, (inv, t[28]), Some(self.epsilon));
                            ops.scale(root, t[27], qh, kd.into());
                            ops.copy_part(mixed, key_at, work, 0.into(), kd.into());
                            ops.normalize(work, kd.into());
                            ops.l2_normalize(work, kh, kd, (inv, t[28]), Some(self.epsilon));
                            ops.copy_part(mixed, value_at, vh, 0.into(), vd.into());
                            ops.normalize(vh, vd.into());

                            ops.copy_part(b_reg, h, work, 0.into(), 1.into());
                            ops.normalize(work, 1.into());
                            ops.function(FN_SIGMOID, work, beta, 1.into());

                            ops.copy_part(a_reg, h, t[29], 0.into(), 1.into());
                            ops.normalize(t[29], 1.into());
                            ops.copy_part(bias, h, t[30], 0.into(), 1.into());
                            ops.normalize(t[30], 1.into());
                            ops.add(t[29], t[30], work, 1.into());
                            ops.function(FN_SOFTPLUS, work, t[31], 1.into());
                            ops.copy_part(scale_a, h, t[32], 0.into(), 1.into());
                            ops.normalize(t[32], 1.into());
                            ops.multiply(t[31], t[32], t[33], 1.into());
                            ops.multiply(t[33], self.minus_one, work, 1.into());
                            ops.function(FN_EXP, work, decay, 1.into());

                            let regs = DeltaRegs { state: if own { first_state } else { first_state + offset }, key: kh, query: qh, value: vh, beta, decay, memory: t[10], error: t[11], delta: t[12], update: t[13], next: t[14], out: first_out + offset, key_rows: t[56], query_rows: t[57] };
                            ops.delta_step(regs, kd, vd);
                            // The gated norm reads each head's output where it was made, so a
                            // copy of it is the one brought to the shared exponent.
                            ops.copy_part(first_out + offset, 0.into(), first_part + offset, 0.into(), vd.into());
                            ops.align_heads(first_part + offset, vd);
                        });
                    }
                });
                let h0 = k0 as usize * ratio;
                let parts: Vec<(u32, u32)> = (h0..h0 + key_count as usize * ratio).map(|h| (self.parts[h], vd)).collect();
                ops.gather_heads_at(&parts, self.reg(*dst), h0 as u32 * vd);
                ops.heads_across_chips(false);
            }
            Op::RmsGated { src, z, weight, block, dst } => {
                let count = program.buffers[*src] / block;
                let w = self.small[&format!("{weight}.weight")];
                let by_head = self.delta_outputs.contains(src);
                let program_src = *src;
                let (src, z) = (self.reg(*src), self.reg(*z));
                let d = *block as u32;
                let (first_out, first_part) = (self.delta_parts[0], self.parts[0]);
                // Heads of a recurrent output stay on the units the delta rule gave them.
                // The same layout as the delta rule's, so each head is where it was made.
                let source = program.ops.iter().find_map(|i| match i.op {
                    Op::DeltaStep { dst, key_heads, value_heads, .. } if dst == program_src => Some((key_heads as u32, (value_heads / key_heads) as u32)),
                    _ => None,
                });
                // A chip that shares the heads normalises its own, where its delta rule left them.
                let (k0, key_count) = source.map_or((0, 0), |(key_heads, _)| self.heads_of(program_src, key_heads));
                let shared = source.is_some_and(|(key_heads, _)| key_count < key_heads);
                ops.heads_across_chips(shared);
                let split = |ops: &mut O, body: &mut dyn FnMut(&mut O, O::N)| match source.filter(|_| by_head) {
                    Some((_, ratio)) => ops.each_head_split(key_count, ratio, |ops, k, s| body(ops, (k + k0.into()) * ratio + s)),
                    None if by_head => ops.each_head_split(count as u32, 1, |ops, b, _| body(ops, b)),
                    None => ops.each_head(count as u32, 1, |ops, b, _| body(ops, b)),
                };
                split(ops, &mut |ops, b| {
                    ops.with_window(b, &[first_out, first_part], |ops, offset| {
                        if by_head {
                            ops.copy_part(first_out + offset, 0.into(), t[40], 0.into(), d.into());
                        } else {
                            ops.copy_part(src, b * d, t[40], 0.into(), d.into());
                        }
                        ops.normalize(t[40], d.into());
                        ops.rms_normalize(t[40], t[41], d, (t[42], t[43]), self.inverse_len.get(block).copied(), Some(self.epsilon));
                        ops.multiply(t[41], w, t[44], d.into());
                        ops.copy_part(z, b * d, t[45], 0.into(), d.into());
                        ops.normalize(t[45], d.into());
                        // SiLU's small values would be lost next to its peak, so z and its
                        // sigmoid are applied one after the other.
                        ops.function(FN_SIGMOID, t[45], t[46], d.into());
                        ops.multiply(t[44], t[45], t[38], d.into());
                        ops.multiply(t[38], t[46], first_part + offset, d.into());
                        ops.align_heads(first_part + offset, d);
                    });
                });
                let (h0, mine) = match source {
                    Some((_, ratio)) if shared => ((k0 * ratio) as usize, (key_count * ratio) as usize),
                    _ => (0, count),
                };
                let parts: Vec<(u32, u32)> = (h0..h0 + mine).map(|b| (self.parts[b], d)).collect();
                ops.gather_heads_at(&parts, self.reg(*dst), h0 as u32 * d);
                ops.heads_across_chips(false);
            }
            Op::Concat { a, b, dst } => {
                let (la, lb) = (program.buffers[*a] as u32, program.buffers[*b] as u32);
                let parts = [(self.reg(*a), la), (self.reg(*b), lb)];
                ops.gather(&parts, self.reg(*dst));
            }
            Op::Moe { prefix, src, dst, experts, top_k, intermediate, shared_intermediate } => {
                self.lower_moe(ops, prefix, self.reg(*src), self.reg(*dst), program.hidden as u32, *experts as u32, *top_k as u32, *intermediate as u32, *shared_intermediate as u32);
            }
        }
        Ok(())
    }

    /// Rotates consecutive pairs of `reg`, DeepSeek's form of RoPE.
    fn rotate_pairs<O: BlockOps>(&self, ops: &mut O, reg: u32, rope: u32, dst: u32) {
        let (half, table) = (self.rope_half as u32, self.rope[0]);
        ops.each(half.into(), |ops, j| {
            let (first, second) = (j * 2, j * 2 + 1.into());
            ops.sum_products(first, &[((reg, first), (table, j)), ((reg, second), (table, j + half.into()))]);
            ops.sum_products(second, &[((reg, second), (table, j)), ((reg, first), (table, j + (2 * half).into()))]);
        });
        ops.narrow_sums(reg, table, dst, rope.into());
    }

    /// DeepSeek's routing, in the same static form as `lower_moe`.
    #[allow(clippy::too_many_arguments)]
    fn lower_sigmoid_moe<O: BlockOps>(&self, ops: &mut O, prefix: &str, x: u32, dst: u32, hidden: u32, experts: u32, top_k: u32, intermediate: u32, shared: u32, scaling: f32, normalize: bool) {
        let t = &self.temporaries;
        let zero: O::N = 0.into();
        let (experts_n, hidden_n): (O::N, O::N) = (experts.into(), hidden.into());
        let matrix = |name: &str| format!("{prefix}.{name}");
        let reg_of = |name: &str| self.weights[&matrix(name)];
        let (raw, top, chances, total, inverse, work, mark, remaining, chosen, selected, scores) = (t[50], t[51], t[53], t[54], t[55], t[56], t[57], t[58], t[59], t[60], t[61]);
        ops.matvec(&matrix("gate"), reg_of("gate"), 0.into(), x, raw, experts, hidden);
        ops.function(FN_SIGMOID, raw, chances, experts_n);
        match self.small.get(&matrix("gate.e_score_correction_bias")) {
            Some(correction) => ops.add(chances, *correction, work, experts_n),
            None => ops.copy_part(chances, zero, work, zero, experts_n),
        }
        for pass in 0..top_k {
            ops.maximum(work, experts_n, top);
            ops.mark_equal(work, top, mark, experts_n);
            if pass == 0 {
                ops.copy_part(mark, zero, chosen, zero, experts_n);
            } else {
                ops.add(chosen, mark, t[62], experts_n);
                ops.copy_part(t[62], zero, chosen, zero, experts_n);
            }
            ops.add_sub(true, (self.ones, zero), (mark, zero), 1, remaining, experts_n);
            ops.multiply(work, remaining, t[62], experts_n);
            ops.copy_part(t[62], zero, work, zero, experts_n);
        }
        ops.multiply(chances, chosen, selected, experts_n);
        if top_k > 1 && normalize {
            let ones = self.ones;
            ops.dots(selected, ones, total, 1.into(), experts_n, |_, e| ((selected, e), (ones, e)));
            ops.reciprocal(total, inverse);
            ops.scale(inverse, selected, t[52], experts_n);
        } else {
            ops.copy_part(selected, zero, t[52], zero, experts_n);
        }
        ops.scale(self.small[&format!("escala:{scaling}")], t[52], scores, experts_n);

        let outputs = t[48];
        ops.set_exponent(outputs, EXPERT_EXPONENT);
        let (gate, up, product, down) = (t[34], t[35], t[36], t[37]);
        ops.each(experts_n, |ops, e| {
            if ops.known_zero(scores, e) {
                return;
            }
            ops.matvec(&matrix("switch_mlp.gate_proj"), reg_of("switch_mlp.gate_proj"), e * intermediate, x, gate, intermediate, hidden);
            ops.matvec(&matrix("switch_mlp.up_proj"), reg_of("switch_mlp.up_proj"), e * intermediate, x, up, intermediate, hidden);
            self.silu(ops, gate, t[38], intermediate);
            ops.multiply(t[38], up, product, intermediate.into());
            ops.matvec(&matrix("switch_mlp.down_proj"), reg_of("switch_mlp.down_proj"), e * hidden, product, down, hidden, intermediate);
            ops.align_to(down, zero, EXPERT_EXPONENT, outputs, e * hidden, hidden_n);
        });
        let mixed = t[47];
        ops.dots(scores, outputs, mixed, hidden_n, experts_n, |j, e| ((scores, e), (outputs, e * hidden + j)));
        if shared == 0 {
            ops.copy_part(mixed, zero, dst, zero, hidden_n);
            return;
        }
        ops.matvec(&matrix("shared_experts.gate_proj"), reg_of("shared_experts.gate_proj"), 0.into(), x, gate, shared, hidden);
        ops.matvec(&matrix("shared_experts.up_proj"), reg_of("shared_experts.up_proj"), 0.into(), x, up, shared, hidden);
        self.silu(ops, gate, t[38], shared);
        ops.multiply(t[38], up, product, shared.into());
        ops.matvec(&matrix("shared_experts.down_proj"), reg_of("shared_experts.down_proj"), 0.into(), product, down, hidden, shared);
        ops.add(mixed, down, dst, hidden_n);
    }

    /// Routes to the `top_k` experts. Selection uses only static operations, so the
    /// gates can evaluate every expert while unselected ones get a score of zero.
    #[allow(clippy::too_many_arguments)]
    fn lower_moe<O: BlockOps>(&self, ops: &mut O, prefix: &str, x: u32, dst: u32, hidden: u32, experts: u32, top_k: u32, intermediate: u32, shared: u32) {
        let t = &self.temporaries;
        let zero: O::N = 0.into();
        let (experts_n, hidden_n): (O::N, O::N) = (experts.into(), hidden.into());
        let matrix = |name: &str| format!("{prefix}.{name}");
        let reg_of = |name: &str| self.weights[&matrix(name)];
        let (raw, top, shifted, chances, total, inverse, work, mark, remaining, chosen, selected, scores) = (t[50], t[51], t[52], t[53], t[54], t[55], t[56], t[57], t[58], t[59], t[60], t[61]);

        ops.matvec(&matrix("gate"), reg_of("gate"), 0.into(), x, raw, experts, hidden);
        ops.maximum(raw, experts_n, top);
        ops.sub_scalar(raw, top, shifted, experts_n);
        ops.function(FN_EXP, shifted, chances, experts_n);

        // Repeatedly take the largest probability and blank it out.
        ops.copy_part(chances, zero, work, zero, experts_n);
        for pass in 0..top_k {
            ops.maximum(work, experts_n, top);
            ops.mark_equal(work, top, mark, experts_n);
            if pass == 0 {
                ops.copy_part(mark, zero, chosen, zero, experts_n);
            } else {
                ops.add(chosen, mark, t[62], experts_n);
                ops.copy_part(t[62], zero, chosen, zero, experts_n);
            }
            ops.add_sub(true, (self.ones, zero), (mark, zero), 1, remaining, experts_n);
            ops.multiply(work, remaining, t[62], experts_n);
            ops.copy_part(t[62], zero, work, zero, experts_n);
        }
        ops.multiply(chances, chosen, selected, experts_n);
        let ones = self.ones;
        ops.dots(selected, ones, total, 1.into(), experts_n, |_, e| ((selected, e), (ones, e)));
        ops.reciprocal(total, inverse);
        ops.scale(inverse, selected, scores, experts_n);

        // Every expert's output goes on one exponent, ready to be weighted.
        let outputs = t[48];
        ops.set_exponent(outputs, EXPERT_EXPONENT);
        let (gate, up, product, down) = (t[34], t[35], t[36], t[37]);
        ops.each(experts_n, |ops, e| {
            if ops.known_zero(scores, e) {
                return;
            }
            ops.matvec(&matrix("switch_mlp.gate_proj"), reg_of("switch_mlp.gate_proj"), e * intermediate, x, gate, intermediate, hidden);
            ops.matvec(&matrix("switch_mlp.up_proj"), reg_of("switch_mlp.up_proj"), e * intermediate, x, up, intermediate, hidden);
            self.silu(ops, gate, t[38], intermediate);
            ops.multiply(t[38], up, product, intermediate.into());
            ops.matvec(&matrix("switch_mlp.down_proj"), reg_of("switch_mlp.down_proj"), e * hidden, product, down, hidden, intermediate);
            ops.align_to(down, zero, EXPERT_EXPONENT, outputs, e * hidden, hidden_n);
        });
        let mixed = t[47];
        ops.dots(scores, outputs, mixed, hidden_n, experts_n, |j, e| ((scores, e), (outputs, e * hidden + j)));

        ops.matvec(&matrix("shared_expert.gate_proj"), reg_of("shared_expert.gate_proj"), 0.into(), x, gate, shared, hidden);
        ops.matvec(&matrix("shared_expert.up_proj"), reg_of("shared_expert.up_proj"), 0.into(), x, up, shared, hidden);
        self.silu(ops, gate, t[38], shared);
        ops.multiply(t[38], up, product, shared.into());
        ops.matvec(&matrix("shared_expert.down_proj"), reg_of("shared_expert.down_proj"), 0.into(), product, down, hidden, shared);
        ops.matvec(&matrix("shared_expert_gate"), reg_of("shared_expert_gate"), 0.into(), x, t[49], 1, hidden);
        ops.function(FN_SIGMOID, t[49], t[63], 1.into());
        ops.scale(t[63], down, t[62], hidden_n);
        ops.add(mixed, t[62], dst, hidden_n);
    }
}
