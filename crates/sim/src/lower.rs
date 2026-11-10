//! Expresses a compiled program in the die's vector arithmetic, token by token.
//! The same code drives the gates (through microcode) and the exact virtual machine.

use crate::weights::{CellRow, Weights};
use compiler::{Inst, Op, Program, RopeScaling};
use netlist::block::{FN_GELU, FN_GELU_TANH, FN_TANH};
use netlist::block::{chained_rows, encode_vector, Assembly, AttentionRegs, BlockOps, CellMatrix, DeltaRegs, Expr, KeyStore, MatvecPart, Var, FN_EXP, FN_SIGMOID, FN_SOFTPLUS};
use netlist::{DieConfig, DieProgram, Microprogram};
use std::cell::Cell;
use std::collections::{HashMap, HashSet};
use std::ops::Range;

/// KV entries, convolution history and expert outputs use this fixed exponent.
const STATE_EXPONENT: i64 = 6;
/// Expert outputs are small; a coarser exponent would leave them only a few bits.
const EXPERT_EXPONENT: i64 = 2;
/// Added to every squared norm, as the reference model does.
const ONES_LENGTH: usize = 512;
const TEMPORARIES: usize = 64;
const PARTS: usize = 128;

fn all_ops(program: &Program) -> impl Iterator<Item = &Inst> {
    program.ops.iter().chain(program.draft.iter().flat_map(|d| d.ops.iter()))
}

fn matrices_of(program: &Program, op: &Op) -> Vec<(String, usize, usize)> {
    match op {
        Op::Matvec { matrix, src, dst } => vec![(matrix.clone(), program.buffers[*dst], program.buffers[*src])],
        Op::EmbRead { table, dst } => vec![(table.clone(), program.vocab, program.buffers[*dst])],
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
        _ => unreachable!("solo los productos se encadenan"),
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
        let (codes, factors) = matrix.cell_row(row, self.cols, self.exponent, self.bits)?;
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
    exponent: i64,
    /// Its rows follow those of the entry before, in the same passes of the cells.
    chained: bool,
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
            places.insert(part.name.clone(), CellMatrix { weights: weights as u32, factors: factors as u32, group: e.group as u32, groups: e.groups as u32, quant_bits });
        }
        let blocks = if parts.len() == 1 { e.rows / e.block } else { 1 };
        let rounds = (blocks * block.div_ceil(per_unit)) as u64;
        let groups = e.groups.div_ceil(units) as u64;
        weights += rounds * groups * e.group as u64;
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

/// Where everything lives in the register file, stable across tokens.
pub struct Layout {
    config: DieConfig,
    buffers: Vec<u32>,
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
}

impl Layout {
    /// Allocates registers and stores constants. With `chip_tokens`, the matrix weights
    /// and a RoPE table for that many tokens go into registers for the gates; otherwise
    /// the weights stay in the checkpoint and RoPE is computed directly.
    pub fn new(program: &Program, weights: &Weights, config: DieConfig, chip_tokens: Option<usize>) -> Result<Self, String> {
        Self::build(program, weights, config, chip_tokens, true, false)
    }

    /// For the gates: matrices compute in the cells and only lookup tables stay in registers.
    pub fn with_cells(program: &Program, weights: &Weights, config: DieConfig, chip_tokens: usize) -> Result<Self, String> {
        Self::build(program, weights, config, Some(chip_tokens), true, true)
    }

    /// As a real chip is laid out: matrices live in the cells, not in vector registers.
    pub fn for_image(program: &Program, weights: &Weights, config: DieConfig, chip_tokens: usize) -> Result<Self, String> {
        Self::build(program, weights, config, Some(chip_tokens), false, true)
    }

    fn build(program: &Program, weights: &Weights, config: DieConfig, chip_tokens: Option<usize>, matrices_in_registers: bool, cells: bool) -> Result<Self, String> {
        let next = Cell::new(1u32);
        let alloc = |count: u32| next.replace(next.get() + count);
        let buffers: Vec<u32> = program.buffers.iter().map(|_| alloc(1)).collect();
        let mut setup = Assembly::new(&config);
        let mut matrices: HashMap<String, u32> = HashMap::new();
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
                let reg = *matrices.entry(name.clone()).or_insert_with(|| alloc(1));
                if chip_tokens.is_some() && matrices_in_registers && (!cells || table) && in_registers.insert(name.clone()) {
                    if rows * cols > 1 << config.index_bits {
                        return Err(format!("la matriz {name} tiene {} elementos y el direccionamiento solo llega a {}", rows * cols, 1usize << config.index_bits));
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
                    let cut = config.features.feeds_cells() && units > 1 && block == rows && !groups.is_multiple_of(units) && cols.is_multiple_of(units) && group.is_power_of_two() && group <= 128 && (2..group).contains(&part);
                    let (group, groups, quant) = if cut { (part, cols / part, group) } else { (group, groups, group) };
                    let exponent = matrix.exponent(cols);
                    setup.load_mantissas(reg, &[], exponent);
                    let chained = match (previous, &inst.op, cell_plan.last()) {
                        (Some(Op::Matvec { matrix: before, src: from, .. }), Op::Matvec { src, dst, .. }, Some(last)) => {
                            last.name == *before && from == src && dst != src && uses[before] == 1 && uses[&name] == 1 && block == rows && last.block == last.rows && (last.cols, last.group, last.groups, last.quant) == (cols, group, groups, quant)
                        }
                        _ => false,
                    };
                    if chained {
                        let leader = leader_of.get(&cell_plan.last().expect("anterior").name).cloned().expect("cadena");
                        chains.get_mut(&leader).expect("cadena").push((name.clone(), program_dst(&inst.op)));
                        leader_of.insert(name.clone(), leader);
                    } else if let Op::Matvec { dst, .. } = &inst.op {
                        chains.insert(name.clone(), vec![(name.clone(), *dst)]);
                        leader_of.insert(name.clone(), name.clone());
                    }
                    cell_plan.push(CellEntry { block, name, rows, cols, group, groups, quant, exponent, chained });
                }
            }
            previous = Some(&inst.op);
            if let Op::RmsNorm { block, .. } | Op::RmsGated { block, .. } = &inst.op {
                if !block.is_power_of_two() {
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

        let ones = alloc(1);
        setup.load_vector(ones, &[1.0; ONES_LENGTH]);
        let minus_one = alloc(1);
        setup.load_vector(minus_one, &[-1.0]);
        let epsilon = alloc(1);
        setup.load_vector(epsilon, &[program.rms_epsilon]);
        let rope = [alloc(1), alloc(1)];
        let rope_half = all_ops(program)
            .find_map(|i| match i.op {
                Op::Rope { rotary, .. } => Some(rotary / 2),
                Op::Mla { rope, .. } => Some(rope / 2),
                _ => None,
            })
            .unwrap_or(0);
        let rope_table = [alloc(1), alloc(1)];
        let mut rope_blocks = all_ops(program).filter_map(|i| if let Op::Rope { block, .. } = i.op { Some(block as u32) } else { None });
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
        let temporaries: Vec<u32> = (0..TEMPORARIES).map(|_| alloc(1)).collect();
        let parts: Vec<u32> = (0..PARTS).map(|_| alloc(1)).collect();
        let delta_parts: Vec<u32> = (0..PARTS).map(|_| alloc(1)).collect();
        let delta_outputs = all_ops(program).filter_map(|i| if let Op::DeltaStep { dst, .. } = i.op { Some(dst) } else { None }).collect();
        let spread = spread_buffers(program);
        let next = next.get();
        if next >= (1 << config.reg_bits) - 2 {
            return Err(format!("faltan registros: se necesitan {next}"));
        }
        let key_capacity = chip_tokens.filter(|_| cells && config.features.combines()).map(|tokens| match config.features.key_shift {
            0 => (tokens as u32).next_multiple_of(1 << config.features.lane_bits),
            shift => 1 << shift,
        });
        if let (Some(tokens), Some(capacity)) = (chip_tokens, key_capacity) {
            if tokens as u32 > capacity {
                return Err(format!("cada columna de claves tiene sitio para {capacity} tokens y el contexto es de {tokens}"));
            }
        }
        // A pass of the cells drains whole blocks: fewer cells than lanes on a unit would
        // leave lanes idle and its rows out of place.
        let (units, lanes) = (1u32 << config.features.cluster_bits, 1u32 << config.features.lane_bits);
        if cells && config.cells / units < lanes {
            return Err(format!("cada unidad necesita al menos {lanes} celdas, una por carril, y tiene {}", config.cells / units));
        }
        let (in_cells, cell_weights, cell_factors) = place_in_cells(&cell_plan, config.cells as usize, 1 << config.features.cluster_bits, 1 << config.features.lane_bits);
        chains.retain(|_, parts| parts.len() > 1);
        let followers: HashSet<String> = chains.values().flat_map(|parts| parts[1..].iter().map(|(name, _)| name.clone())).collect();
        let reach = 1u64 << config.index_bits.min(config.cell.addr_bits);
        if cell_weights > reach || 2 * cell_factors > 1 << config.cell.addr_bits {
            return Err(format!("las celdas necesitan {cell_weights} posiciones de pesos y {cell_factors} grupos, y solo llegan a {reach}"));
        }
        Ok(Self { config, buffers, weights: matrices, small, keys, values, conv, delta, next, ones, minus_one, epsilon, temporaries, parts, delta_parts, delta_outputs, spread, rope, rope_table, rope_half, rope_factors, inverse_root, inverse_len, setup, cell_plan, in_cells, lookup_tables, key_capacity, chains, followers })
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

    pub fn register_of(&self, buffer: usize) -> u32 {
        self.buffers[buffer]
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
            code.assemble(&asm.steps);
            Ok::<(), String>(())
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
        Ok(code.finish_program())
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
        if !last {
            let output = program.crossing(plan[die + 1].start)?;
            asm.send(self.reg(output), (program.buffers[output] as u32).into());
        } else if several {
            asm.send_signal();
        }
        if first && several {
            asm.receive_signal();
        }
        code.assemble(&asm.steps);
        code.next_entry();

        let mut asm = self.assembly();
        let logits_length: Expr = (program.vocab as u32).into();
        if last {
            self.run_list(&mut asm, program, &program.ops[program.logits_from..], token, position, false)?;
            if several {
                asm.send(self.reg(program.logits), logits_length);
            }
        } else if first {
            asm.receive(self.reg(program.logits), logits_length);
        }
        code.assemble(&asm.steps);
        Ok(code.finish_program())
    }

    /// Forgets the weights of every matrix that the given operations do not use.
    fn assembly(&self) -> Assembly {
        let mut asm = Assembly::new(&self.config);
        asm.in_cells = self.in_cells.clone();
        asm.key_capacity = self.key_capacity;
        asm
    }

    /// The microcode of one operation on its own, to see what it costs.
    /// One operation on its own; `first` also loads the token's rotary table, as the
    /// first operation of every token does.
    pub fn op_microprogram(&self, program: &Program, inst: &Inst, first: bool) -> Result<Microprogram, String> {
        let mut asm = self.assembly();
        self.run_list(&mut asm, program, std::slice::from_ref(inst), Expr::var(Var::Token), Expr::var(Var::Position), first)?;
        let mut code = DieProgram::new(self.config);
        code.assemble(&asm.steps);
        Ok(code.finish_program())
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
        self.cell_plan.iter().map(|e| CellTensor { name: e.name.clone(), rows: e.rows, cols: e.cols, block: e.block, exponent: e.exponent, cells: self.config.cells as usize, bits: self.config.cell.activation_bits }).collect()
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
                            let (w, f) = matrix.cell_row(b * p.block + row - start, p.cols, p.exponent, self.config.cell.activation_bits)?;
                            let count = p.cols.div_ceil(p.quant);
                            let shared = |q: usize| if f.len() == count { f.get(q).copied() } else { (q < count).then(|| f[0]) };
                            factors.extend(quant_groups.clone().map(|q| shared(q).unwrap_or((0, 0))));
                            for g in first..first + groups {
                                for j in 0..p.group {
                                    codes.push(w.get(g * p.group + j).copied().unwrap_or(0));
                                }
                            }
                        }
                        None => {
                            codes.extend(std::iter::repeat_n(0, groups * e.group));
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
        let last = die + 1 == plan.len();
        let tail = program.logits_from..program.ops.len();
        let used: HashSet<String> = plan[die].clone().chain(if last { tail } else { 0..0 }).flat_map(|i| matrices_of(program, &program.ops[i].op)).map(|(name, ..)| name).collect();
        let dropped: HashSet<u32> = self.weights.iter().filter(|(name, _)| !used.contains(*name)).map(|(_, reg)| *reg).collect();
        self.setup.initial.retain(|(register, _), _| !dropped.contains(register));
        self.cell_plan.retain(|e| used.contains(&e.name));
        self.in_cells = place_in_cells(&self.cell_plan, self.config.cells as usize, 1 << self.config.features.cluster_bits, 1 << self.config.features.lane_bits).0;
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
                let draft = program.draft.as_ref().ok_or("este modelo no tiene cabeza MTP")?;
                let (list, rope) = if routine == ENTRY_DRAFT { (&draft.ops[..draft.logits_from], true) } else { (&draft.ops[draft.logits_from..], false) };
                self.run_list(ops, program, list, token, position, rope)
            }
            other => Err(format!("rutina desconocida: {other}")),
        }
    }

    pub fn run_token<O: BlockOps>(&self, ops: &mut O, program: &Program, token: O::N, position: O::N, want_logits: bool) -> Result<(), String> {
        let limit = if want_logits { program.ops.len() } else { program.logits_from };
        self.run_range(ops, program, 0..limit, token, position, true)
    }

    fn run_range<O: BlockOps>(&self, ops: &mut O, program: &Program, range: Range<usize>, token: O::N, position: O::N, rope: bool) -> Result<(), String> {
        self.run_list(ops, program, &program.ops[range], token, position, rope)
    }

    fn run_list<O: BlockOps>(&self, ops: &mut O, program: &Program, list: &[Inst], token: O::N, position: O::N, rope: bool) -> Result<(), String> {
        if rope {
            self.load_rope(ops, program, position);
        }
        for inst in list {
            self.lower_op(ops, program, &inst.op, token, position)?;
        }
        Ok(())
    }

    pub fn run_draft<O: BlockOps>(&self, ops: &mut O, program: &Program, token: O::N, position: O::N, want_logits: bool) -> Result<(), String> {
        let draft = program.draft.as_ref().ok_or("este modelo no tiene cabeza MTP")?;
        let limit = if want_logits { draft.ops.len() } else { draft.logits_from };
        self.run_list(ops, program, &draft.ops[..limit], token, position, true)
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
        ops.load_row(self.rope_table[0], position, 3 * half as u32 + 1, self.rope[0], |row| rope_row(theta, scaling, half, row));
        if let Some(local) = program.local_rope_theta {
            ops.load_row(self.rope_table[1], position, 3 * half as u32 + 1, self.rope[1], |row| rope_row(local, None, half, row));
        }
        // Every rotation of the token uses the same factors, so they are laid out once.
        if let Some((block, factors)) = self.rope_factors {
            for (table, (cos, sin)) in factors.iter().enumerate().take(1 + program.local_rope_theta.is_some() as usize) {
                self.rope_factors_into(ops, self.rope[table], block, (*cos, *sin));
            }
        }
    }

    /// `cos` and `sin` for each element of a head of `d`: the rotated pairs take their
    /// angle's, and the elements past the rotated part are kept as they are.
    fn rope_factors_into<O: BlockOps>(&self, ops: &mut O, rope: u32, d: u32, (cos, sin): (u32, u32)) {
        let zero: O::N = 0.into();
        let h = self.rope_half as u32;
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

    fn rms_blocks<O: BlockOps>(&self, ops: &mut O, src: u32, weight: Option<u32>, block: usize, count: usize, dst: u32) {
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
        ops.each_head(count as u32, 1, |ops, b, _| {
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
        let parts: Vec<(u32, u32)> = (0..count).map(|b| (self.parts[b], d)).collect();
        ops.gather_heads(&parts, dst);
    }

    /// `a + b` or `a · b`, a slice per unit where the die has several.
    fn pairwise<O: BlockOps>(&self, ops: &mut O, product: bool, (a, b): (u32, u32), dst: u32, len: u32) {
        let t = &self.temporaries;
        ops.in_slabs(len, &[dst], |ops, start, count| {
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
        let t = &self.temporaries;
        ops.in_slabs(len, &[dst], |ops, start, count| {
            let (x, sigmoid, y) = (t[58], t[39], t[59]);
            ops.copy_part(src, start, x, 0.into(), count.into());
            ops.function(FN_SIGMOID, x, sigmoid, count.into());
            ops.multiply(sigmoid, x, y, count.into());
            ops.copy_part(y, 0.into(), dst, start, count.into());
        });
    }

    fn lower_op<O: BlockOps>(&self, ops: &mut O, program: &Program, op: &Op, token: O::N, position: O::N) -> Result<(), String> {
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
            Op::EmbRead { table, dst } => {
                let len = program.buffers[*dst] as u32;
                ops.copy_row(table, self.weights[table], token, len, self.reg(*dst));
                ops.normalize(self.reg(*dst), len.into());
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
            Op::Add { a, b, dst } => self.pairwise(ops, false, (self.reg(*a), self.reg(*b)), self.reg(*dst), program.buffers[*dst] as u32),
            Op::MulElem { a, b, dst } => self.pairwise(ops, true, (self.reg(*a), self.reg(*b)), self.reg(*dst), program.buffers[*dst] as u32),
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
                        ops.attention(t[7], KeyStore { keys: t[2], values: t[2], stride: width, offset: zero }, self.ones, t[8], width, tokens, Some(scale), regs);
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
            Op::Silu { src, dst } => self.silu(ops, self.reg(*src), self.reg(*dst), program.buffers[*dst] as u32),
            Op::Sigmoid { src, dst } => {
                let t = &self.temporaries;
                ops.in_slabs(program.buffers[*dst] as u32, &[self.reg(*dst)], |ops, start, count| {
                    let (x, y) = (t[58], t[59]);
                    ops.copy_part(self.reg(*src), start, x, 0.into(), count.into());
                    ops.function(FN_SIGMOID, x, y, count.into());
                    ops.copy_part(y, 0.into(), self.reg(*dst), start, count.into());
                });
            }
            Op::RmsNorm { src, weight, block, dst, .. } => {
                let count = program.buffers[*dst] / block;
                let w = self.small[&format!("{weight}.weight")];
                self.rms_blocks(ops, self.reg(*src), Some(w), *block, count, self.reg(*dst));
            }
            Op::SplitQueryGate { src, query, gate, block } => {
                let heads = program.buffers[*query] / block;
                let d = *block as u32;
                let src = self.reg(*src);
                if 2 * heads <= PARTS {
                    // Each unit takes the two halves of its own head.
                    let (first_query, first_gate) = (self.parts[0], self.parts[heads]);
                    ops.each_head(heads as u32, 1, |ops, h, _| {
                        ops.with_window(h, &[first_query, first_gate], |ops, offset| {
                            ops.copy_part(src, h * (2 * d), first_query + offset, 0.into(), d.into());
                            ops.copy_part(src, h * (2 * d) + d.into(), first_gate + offset, 0.into(), d.into());
                            ops.align_heads(first_query + offset, d);
                            ops.align_heads(first_gate + offset, d);
                        });
                    });
                    ops.gather_heads(&(0..heads).map(|h| (self.parts[h], d)).collect::<Vec<_>>(), self.reg(*query));
                    ops.gather_heads(&(0..heads).map(|h| (self.parts[heads + h], d)).collect::<Vec<_>>(), self.reg(*gate));
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
                let (d, h, rope) = (*block as u32, self.rope_half as u32, self.rope[*table]);
                // One table of factors for each element, so each head is one pair of sweeps.
                let first = self.parts[0];
                let rotary = 2 * h;
                let (cos, sin) = match self.rope_factors {
                    Some((block, factors)) if block == d => factors[*table],
                    _ => {
                        self.rope_factors_into(ops, rope, d, (t[50], t[51]));
                        (t[50], t[51])
                    }
                };
                let heads = total / d;
                ops.each_head(heads, 1, |ops, head, _| {
                    ops.with_window(head, &[first], |ops, offset| {
                        let (x, swapped) = (t[52], t[53]);
                        ops.copy_part(reg, head * d, x, zero, d.into());
                        ops.copy_part(x, h.into(), swapped, zero, h.into());
                        ops.copy_part(x, zero, swapped, h.into(), h.into());
                        ops.rotate(x, swapped, (cos, sin), first + offset, d, rotary);
                        ops.align_heads(first + offset, d);
                    });
                });
                let parts: Vec<(u32, u32)> = (0..heads as usize).map(|part| (self.parts[part], d)).collect();
                ops.gather_heads(&parts, reg);
            }
            Op::KvAppend { slot, key, value, window } => {
                let width = program.buffers[*key] as u32;
                let slot_place = match window {
                    Some(limit) => ops.window_slot(position, *limit as u32),
                    None => position,
                };
                let reader = program.ops.iter().chain(program.draft.iter().flat_map(|d| d.ops.iter())).find_map(|inst| match inst.op {
                    Op::AttnSum { slot: s, heads, kv_heads, head_dim, .. } if s == *slot => Some((heads, kv_heads, head_dim)),
                    _ => None,
                });
                let (keys, values) = (self.keys[*slot], self.values[*slot]);
                let (key, value) = (self.reg(*key), self.reg(*value));
                match reader {
                    // Units hold the columns they read: each writes just those, all at once.
                    Some((heads, kv_heads, head_dim)) if (kv_heads * head_dim) as u32 == width => {
                        let d = head_dim as u32;
                        ops.each_head_split(kv_heads as u32, (heads / kv_heads) as u32, |ops, kv_head, _| {
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
                let capped = softcap.map(|cap| (self.small[&format!("escala:{}", 1.0 / cap)], self.small[&format!("escala:{cap}")]));
                let regs = AttentionRegs { scores: t[10], top: t[11], shifted: t[12], weights: t[13], total: t[14], inverse: t[15], probabilities: t[16], softcap: capped };
                let first = self.parts[0];
                let query = self.reg(*query);
                ops.each_head_split(*kv_heads as u32, group as u32, |ops, kv_head, shared| {
                    {
                        let h = kv_head * group as u32 + shared;
                        ops.with_window(h, &[first], |ops, offset| {
                            let qh = t[20];
                            ops.copy_part(query, h * d, qh, zero, d.into());
                            ops.normalize(qh, d.into());
                            let store = KeyStore { keys, values, stride: width, offset: kv_head * d };
                            ops.attention(qh, store, self.ones, first + offset, d, tokens, Some(self.inverse_root[scale_dim]), regs);
                            ops.align_heads(first + offset, d);
                        });
                    }
                });
                let parts: Vec<(u32, u32)> = (0..*heads).map(|h| (self.parts[h], d)).collect();
                ops.gather_heads(&parts, self.reg(*dst));
            }
            Op::Dconv { slot, weight, src, dst } => {
                let (dim, kernel) = program.conv_slots[*slot];
                let history = (kernel - 1) as u32;
                let taps = self.small[&format!("{weight}.weight")];
                let reg = self.conv[*slot];
                let (dim32, kernel32) = (dim as u32, kernel as u32);
                let (src, dst) = (self.reg(*src), self.reg(*dst));
                // Each unit keeps the history of its own channels and convolves them.
                ops.in_slabs(dim32, &[dst], |ops, start, n| {
                    ops.align_to(src, start, STATE_EXPONENT, reg, start + (history * dim32).into(), n.into());
                    ops.depthwise_part(taps, reg, dst, dim32, kernel32, start, n);
                    ops.each(history.into(), |ops, k| ops.copy_elements(reg, (k + 1.into()) * dim32 + start, k * dim32 + start, n));
                });
            }
            Op::DeltaStep { slot, mixed, a, b, a_log, dt_bias, key_dim, key_heads, value_dim, value_heads, dst } => {
                let ratio = value_heads / key_heads;
                let (kd, vd) = (*key_dim as u32, *value_dim as u32);
                let key_len = kd * *key_heads as u32;
                let root = self.inverse_root[key_dim];
                let mixed = self.reg(*mixed);
                let (a_reg, b_reg) = (self.reg(*a), self.reg(*b));
                let (scale_a, bias) = (self.small[a_log], self.small[dt_bias]);
                let (first_state, first_out, first_part) = (self.delta[*slot][0], self.delta_parts[0], self.parts[0]);
                ops.each_head_split(*key_heads as u32, ratio as u32, |ops, key_head, shared| {
                    {
                        let h = key_head * ratio as u32 + shared;
                        ops.with_window(h, &[first_state, first_out, first_part], |ops, offset| {
                            let (qh, kh, vh, beta, decay) = (t[20], t[21], t[22], t[23], t[24]);
                            let (work, inv) = (t[25], t[26]);

                            ops.copy_part(mixed, key_head * kd, work, 0.into(), kd.into());
                            ops.normalize(work, kd.into());
                            ops.l2_normalize(work, t[27], kd, (inv, t[28]), Some(self.epsilon));
                            ops.scale(root, t[27], qh, kd.into());
                            ops.copy_part(mixed, key_head * kd + key_len.into(), work, 0.into(), kd.into());
                            ops.normalize(work, kd.into());
                            ops.l2_normalize(work, kh, kd, (inv, t[28]), Some(self.epsilon));
                            ops.copy_part(mixed, h * vd + (2 * key_len).into(), vh, 0.into(), vd.into());
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

                            let regs = DeltaRegs { state: first_state + offset, key: kh, query: qh, value: vh, beta, decay, memory: t[10], error: t[11], delta: t[12], update: t[13], next: t[14], out: first_out + offset, key_rows: t[56], query_rows: t[57] };
                            ops.delta_step(regs, kd, vd);
                            // The gated norm reads each head's output where it was made, so a
                            // copy of it is the one brought to the shared exponent.
                            ops.copy_part(first_out + offset, 0.into(), first_part + offset, 0.into(), vd.into());
                            ops.align_heads(first_part + offset, vd);
                        });
                    }
                });
                let parts: Vec<(u32, u32)> = (0..*value_heads).map(|h| (self.parts[h], vd)).collect();
                ops.gather_heads(&parts, self.reg(*dst));
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
                let split = |ops: &mut O, body: &mut dyn FnMut(&mut O, O::N)| match source.filter(|_| by_head) {
                    Some((key_heads, ratio)) => ops.each_head_split(key_heads, ratio, |ops, k, s| body(ops, k * ratio + s)),
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
                let parts: Vec<(u32, u32)> = (0..count).map(|b| (self.parts[b], d)).collect();
                ops.gather_heads(&parts, self.reg(*dst));
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
