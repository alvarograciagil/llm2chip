//! Expresses a compiled program in the die's vector arithmetic, token by token.
//! The same code drives the gates (through microcode) and the exact virtual machine.

use crate::weights::Weights;
use compiler::{Inst, Op, Program};
use netlist::block::{Assembly, AttentionRegs, BlockOps, DeltaRegs, FN_EXP, FN_SIGMOID, FN_SOFTPLUS};
use netlist::DieConfig;
use std::cell::Cell;
use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};

/// KV entries, convolution history and expert outputs use this fixed exponent.
const STATE_EXPONENT: i64 = 6;
/// Expert outputs are small; a coarser exponent would leave them only a few bits.
const EXPERT_EXPONENT: i64 = 2;
/// Added to every squared norm, as the reference model does.
const EPSILON: f64 = 1e-6;
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
        _ => Vec::new(),
    }
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
    rope: u32,
    rope_half: usize,
    inverse_root: HashMap<usize, u32>,
    inverse_len: HashMap<usize, u32>,
    setup: Assembly,
}

impl Layout {
    /// Allocates registers and stores constants. With `load_matrices`, matrix weights
    /// go into registers for the gates; otherwise they stay in the checkpoint.
    pub fn new(program: &Program, weights: &Weights, config: DieConfig, load_matrices: bool) -> Result<Self, String> {
        let next = Cell::new(1u32);
        let alloc = |count: u32| next.replace(next.get() + count);
        let buffers: Vec<u32> = program.buffers.iter().map(|_| alloc(1)).collect();
        let mut setup = Assembly::new(&config);
        let mut matrices: HashMap<String, u32> = HashMap::new();
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
                if let Entry::Vacant(slot) = matrices.entry(name.clone()) {
                    let reg = alloc(1);
                    slot.insert(reg);
                    if load_matrices {
                        let matrix = weights.matrix(&name)?;
                        let exponent = matrix.exponent(cols);
                        let mut flat = Vec::with_capacity(rows * cols);
                        for r in 0..rows {
                            flat.extend(matrix.fixed_row(r, cols, exponent)?);
                        }
                        setup.load_mantissas(reg, &flat, exponent);
                    }
                }
            }
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
                    let centred = raw.iter().sum::<f32>() / raw.len() as f32 > 0.5;
                    let values = raw.iter().map(|w| (*w + if centred { 0.0 } else { *offset }) as f64).collect();
                    store(name, values, &mut setup, &mut small);
                }
                Op::RmsGated { weight, .. } | Op::Dconv { weight, .. } => {
                    let name = format!("{weight}.weight");
                    let values = weights.floats(&name)?.iter().map(|w| *w as f64).collect();
                    store(name, values, &mut setup, &mut small);
                }
                Op::DeltaStep { a_log, dt_bias, key_dim, .. } => {
                    store(a_log.clone(), weights.floats(a_log)?.iter().map(|v| v.exp() as f64).collect(), &mut setup, &mut small);
                    store(dt_bias.clone(), weights.floats(dt_bias)?.iter().map(|v| *v as f64).collect(), &mut setup, &mut small);
                    root_of(*key_dim, &mut setup, &mut inverse_root);
                }
                Op::AttnSum { head_dim, .. } => root_of(*head_dim, &mut setup, &mut inverse_root),
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
        setup.load_vector(epsilon, &[EPSILON]);
        let rope = alloc(1);
        let rope_half = all_ops(program).find_map(|i| if let Op::Rope { rotary, .. } = i.op { Some(rotary / 2) } else { None }).unwrap_or(0);
        let temporaries: Vec<u32> = (0..TEMPORARIES).map(|_| alloc(1)).collect();
        let parts: Vec<u32> = (0..PARTS).map(|_| alloc(1)).collect();
        let delta_parts: Vec<u32> = (0..PARTS).map(|_| alloc(1)).collect();
        let delta_outputs = all_ops(program).filter_map(|i| if let Op::DeltaStep { dst, .. } = i.op { Some(dst) } else { None }).collect();
        let next = next.get();
        if next >= (1 << config.reg_bits) - 2 {
            return Err(format!("faltan registros: se necesitan {next}"));
        }
        Ok(Self { config, buffers, weights: matrices, small, keys, values, conv, delta, next, ones, minus_one, epsilon, temporaries, parts, delta_parts, delta_outputs, rope, rope_half, inverse_root, inverse_len, setup })
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

    /// The microcode and per-token constants for one token at `position`.
    pub fn token(&self, program: &Program, token: u32, position: usize, want_logits: bool) -> Result<Assembly, String> {
        let mut asm = Assembly::new(&self.config);
        self.run_token(&mut asm, program, token, position, want_logits)?;
        Ok(asm)
    }

    pub fn draft(&self, program: &Program, token: u32, position: usize, want_logits: bool) -> Result<Assembly, String> {
        let mut asm = Assembly::new(&self.config);
        self.run_draft(&mut asm, program, token, position, want_logits)?;
        Ok(asm)
    }

    pub fn run_token<O: BlockOps>(&self, ops: &mut O, program: &Program, token: u32, position: usize, want_logits: bool) -> Result<(), String> {
        let limit = if want_logits { program.ops.len() } else { program.logits_from };
        self.load_rope(ops, program, position);
        for inst in &program.ops[..limit] {
            self.lower_op(ops, program, &inst.op, token, position)?;
        }
        Ok(())
    }

    pub fn run_draft<O: BlockOps>(&self, ops: &mut O, program: &Program, token: u32, position: usize, want_logits: bool) -> Result<(), String> {
        let draft = program.draft.as_ref().ok_or("este modelo no tiene cabeza MTP")?;
        let limit = if want_logits { draft.ops.len() } else { draft.logits_from };
        self.load_rope(ops, program, position);
        for inst in &draft.ops[..limit] {
            self.lower_op(ops, program, &inst.op, token, position)?;
        }
        Ok(())
    }

    pub fn feed_draft<O: BlockOps>(&self, ops: &mut O, program: &Program, from_main: bool) {
        let Some(draft) = &program.draft else { return };
        let source = if from_main { program.final_hidden } else { draft.hidden_out };
        ops.copy_part(self.reg(source), 0, self.reg(draft.hidden_in), 0, program.hidden as u32);
    }

    fn load_rope<O: BlockOps>(&self, ops: &mut O, program: &Program, position: usize) {
        if self.rope_half == 0 {
            return;
        }
        let rotary = 2 * self.rope_half;
        let (mut cos, mut sin) = (Vec::new(), Vec::new());
        for i in 0..self.rope_half {
            let frequency = 1.0 / (program.rope_theta as f64).powf((2 * i) as f64 / rotary as f64);
            cos.push((position as f64 * frequency).cos());
            sin.push((position as f64 * frequency).sin());
        }
        let mut table = cos;
        table.extend(sin.iter().map(|s| -s));
        table.extend(sin.iter());
        table.push(1.0);
        ops.load_vector(self.rope, &table);
    }

    fn reg(&self, buf: usize) -> u32 {
        self.buffers[buf]
    }

    fn rms_blocks<O: BlockOps>(&self, ops: &mut O, src: u32, weight: Option<u32>, block: usize, count: usize, dst: u32) {
        let t = &self.temporaries;
        let mut parts = Vec::new();
        for b in 0..count {
            let part = self.parts[b];
            ops.copy_part(src, (b * block) as u32, t[40], 0, block as u32);
            ops.normalize(t[40], block as u32);
            ops.rms_normalize(t[40], t[41], block as u32, (t[42], t[43]), self.inverse_len.get(&block).copied(), Some(self.epsilon));
            match weight {
                Some(w) => ops.multiply(t[41], w, part, block as u32),
                None => ops.copy_part(t[41], 0, part, 0, block as u32),
            }
            parts.push((part, block as u32));
        }
        ops.gather(&parts, dst);
    }

    fn silu<O: BlockOps>(&self, ops: &mut O, src: u32, dst: u32, len: u32) {
        let t = self.temporaries[39];
        ops.function(FN_SIGMOID, src, t, len);
        ops.multiply(t, src, dst, len);
    }

    fn lower_op<O: BlockOps>(&self, ops: &mut O, program: &Program, op: &Op, token: u32, position: usize) -> Result<(), String> {
        let t = &self.temporaries;
        match op {
            Op::EmbRead { table, dst } => {
                let len = program.buffers[*dst] as u32;
                ops.copy_row(table, self.weights[table], token, len, self.reg(*dst));
                ops.normalize(self.reg(*dst), len);
            }
            Op::Matvec { matrix, src, dst } => {
                let (rows, cols) = (program.buffers[*dst] as u32, program.buffers[*src] as u32);
                ops.matvec(matrix, self.weights[matrix], 0, self.reg(*src), self.reg(*dst), rows, cols);
            }
            Op::Copy { src, dst } => ops.copy_part(self.reg(*src), 0, self.reg(*dst), 0, program.buffers[*dst] as u32),
            Op::Add { a, b, dst } => ops.add(self.reg(*a), self.reg(*b), self.reg(*dst), program.buffers[*dst] as u32),
            Op::MulElem { a, b, dst } => ops.multiply(self.reg(*a), self.reg(*b), self.reg(*dst), program.buffers[*dst] as u32),
            Op::Silu { src, dst } => self.silu(ops, self.reg(*src), self.reg(*dst), program.buffers[*dst] as u32),
            Op::Sigmoid { src, dst } => ops.function(FN_SIGMOID, self.reg(*src), self.reg(*dst), program.buffers[*dst] as u32),
            Op::RmsNorm { src, weight, block, dst, .. } => {
                let count = program.buffers[*dst] / block;
                let w = self.small[&format!("{weight}.weight")];
                self.rms_blocks(ops, self.reg(*src), Some(w), *block, count, self.reg(*dst));
            }
            Op::SplitQueryGate { src, query, gate, block } => {
                let heads = program.buffers[*query] / block;
                let d = *block as u32;
                for h in 0..heads as u32 {
                    ops.copy_part(self.reg(*src), h * 2 * d, t[0], h * d, d);
                    ops.copy_part(self.reg(*src), h * 2 * d + d, t[1], h * d, d);
                }
                ops.gather(&[(t[0], (heads * block) as u32)], self.reg(*query));
                ops.gather(&[(t[1], (heads * block) as u32)], self.reg(*gate));
            }
            Op::Rope { buf, block, .. } => {
                let reg = self.reg(*buf);
                let total = program.buffers[*buf] as u32;
                let (d, h, rope) = (*block as u32, self.rope_half as u32, self.rope);
                ops.dots(reg, rope, t[0], total, 2, |at, term| {
                    let head = at / d * d;
                    let j = at - head;
                    match (j < h, j < 2 * h, term) {
                        (true, _, 0) => Some(((reg, at), (rope, j))),
                        (true, _, _) => Some(((reg, head + h + j), (rope, h + j))),
                        (_, true, 0) => Some(((reg, at), (rope, j - h))),
                        (_, true, _) => Some(((reg, head + j - h), (rope, 2 * h + j - h))),
                        (_, _, 0) => Some(((reg, at), (rope, 3 * h))),
                        _ => None,
                    }
                });
                ops.copy_part(t[0], 0, reg, 0, total);
            }
            Op::KvAppend { slot, key, value } => {
                let width = program.buffers[*key] as u32;
                let at = position as u32 * width;
                ops.align_to(self.reg(*key), 0, STATE_EXPONENT, self.keys[*slot], at, width);
                ops.align_to(self.reg(*value), 0, STATE_EXPONENT, self.values[*slot], at, width);
            }
            Op::AttnSum { slot, query, head_dim, heads, kv_heads, dst } => {
                let group = heads / kv_heads;
                let tokens = position as u32 + 1;
                let width = (program.kv_slots[*slot].heads * head_dim) as u32;
                let d = *head_dim as u32;
                let regs = AttentionRegs { scores: t[10], top: t[11], shifted: t[12], weights: t[13], total: t[14], inverse: t[15], probabilities: t[16] };
                let mut parts = Vec::new();
                for h in 0..*heads {
                    let kv_head = (h / group) as u32;
                    let (qh, kt, vt) = (t[20], t[21], t[22]);
                    ops.copy_part(self.reg(*query), h as u32 * d, qh, 0, d);
                    ops.normalize(qh, d);
                    for token_index in 0..tokens {
                        let from = token_index * width + kv_head * d;
                        ops.copy_part(self.keys[*slot], from, kt, token_index * d, d);
                        ops.copy_part(self.values[*slot], from, vt, token_index * d, d);
                    }
                    ops.normalize(kt, tokens * d);
                    ops.normalize(vt, tokens * d);
                    ops.attention(qh, kt, vt, self.ones, self.parts[h], d, tokens, Some(self.inverse_root[head_dim]), regs);
                    parts.push((self.parts[h], d));
                }
                ops.gather(&parts, self.reg(*dst));
            }
            Op::Dconv { slot, weight, src, dst } => {
                let (dim, kernel) = program.conv_slots[*slot];
                let history = (kernel - 1) as u32;
                let taps = self.small[&format!("{weight}.weight")];
                let reg = self.conv[*slot];
                let (dim32, kernel32) = (dim as u32, kernel as u32);
                ops.align_to(self.reg(*src), 0, STATE_EXPONENT, reg, history * dim32, dim32);
                ops.dots(taps, reg, self.reg(*dst), dim32, kernel32, |c, k| Some(((taps, c * kernel32 + k), (reg, k * dim32 + c))));
                for k in 0..history {
                    for c in 0..dim32 {
                        ops.move_element((reg, (k + 1) * dim32 + c), (reg, k * dim32 + c));
                    }
                }
            }
            Op::DeltaStep { slot, mixed, a, b, a_log, dt_bias, key_dim, key_heads, value_dim, value_heads, dst } => {
                let ratio = value_heads / key_heads;
                let (kd, vd) = (*key_dim as u32, *value_dim as u32);
                let key_len = kd * *key_heads as u32;
                let root = self.inverse_root[key_dim];
                let mixed = self.reg(*mixed);
                let (a_reg, b_reg) = (self.reg(*a), self.reg(*b));
                let (scale_a, bias) = (self.small[a_log], self.small[dt_bias]);
                let mut parts = Vec::new();
                for h in 0..*value_heads {
                    let key_head = (h / ratio) as u32;
                    let h32 = h as u32;
                    let (qh, kh, vh, beta, decay) = (t[20], t[21], t[22], t[23], t[24]);
                    let (work, inv) = (t[25], t[26]);

                    ops.copy_part(mixed, key_head * kd, work, 0, kd);
                    ops.normalize(work, kd);
                    ops.l2_normalize(work, t[27], kd, (inv, t[28]), Some(self.epsilon));
                    ops.scale(root, t[27], qh, kd);
                    ops.copy_part(mixed, key_len + key_head * kd, work, 0, kd);
                    ops.normalize(work, kd);
                    ops.l2_normalize(work, kh, kd, (inv, t[28]), Some(self.epsilon));
                    ops.copy_part(mixed, 2 * key_len + h32 * vd, vh, 0, vd);
                    ops.normalize(vh, vd);

                    ops.copy_part(b_reg, h32, work, 0, 1);
                    ops.normalize(work, 1);
                    ops.function(FN_SIGMOID, work, beta, 1);

                    ops.copy_part(a_reg, h32, t[29], 0, 1);
                    ops.normalize(t[29], 1);
                    ops.copy_part(bias, h32, t[30], 0, 1);
                    ops.normalize(t[30], 1);
                    ops.add(t[29], t[30], work, 1);
                    ops.function(FN_SOFTPLUS, work, t[31], 1);
                    ops.copy_part(scale_a, h32, t[32], 0, 1);
                    ops.normalize(t[32], 1);
                    ops.multiply(t[31], t[32], t[33], 1);
                    ops.multiply(t[33], self.minus_one, work, 1);
                    ops.function(FN_EXP, work, decay, 1);

                    let regs = DeltaRegs { state: self.delta[*slot][h], key: kh, query: qh, value: vh, beta, decay, memory: t[10], error: t[11], delta: t[12], update: t[13], next: t[14], out: self.delta_parts[h] };
                    ops.delta_step(regs, kd, vd);
                    parts.push((self.delta_parts[h], vd));
                }
                ops.gather(&parts, self.reg(*dst));
            }
            Op::RmsGated { src, z, weight, block, dst } => {
                let count = program.buffers[*src] / block;
                let w = self.small[&format!("{weight}.weight")];
                let by_head = self.delta_outputs.contains(src);
                let (src, z) = (self.reg(*src), self.reg(*z));
                let d = *block as u32;
                let mut parts = Vec::new();
                for b in 0..count {
                    let b32 = b as u32;
                    if by_head {
                        ops.copy_part(self.delta_parts[b], 0, t[40], 0, d);
                    } else {
                        ops.copy_part(src, b32 * d, t[40], 0, d);
                    }
                    ops.normalize(t[40], d);
                    ops.rms_normalize(t[40], t[41], d, (t[42], t[43]), self.inverse_len.get(block).copied(), Some(self.epsilon));
                    ops.multiply(t[41], w, t[44], d);
                    ops.copy_part(z, b32 * d, t[45], 0, d);
                    ops.normalize(t[45], d);
                    // SiLU's small values would be lost next to its peak, so z and its
                    // sigmoid are applied one after the other.
                    ops.function(FN_SIGMOID, t[45], t[46], d);
                    ops.multiply(t[44], t[45], t[38], d);
                    ops.multiply(t[38], t[46], self.parts[b], d);
                    parts.push((self.parts[b], d));
                }
                ops.gather(&parts, self.reg(*dst));
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

    /// Routes to the `top_k` experts. Selection uses only static operations, so the
    /// gates can evaluate every expert while unselected ones get a score of zero.
    #[allow(clippy::too_many_arguments)]
    fn lower_moe<O: BlockOps>(&self, ops: &mut O, prefix: &str, x: u32, dst: u32, hidden: u32, experts: u32, top_k: u32, intermediate: u32, shared: u32) {
        let t = &self.temporaries;
        let matrix = |name: &str| format!("{prefix}.{name}");
        let reg_of = |name: &str| self.weights[&matrix(name)];
        let (raw, top, shifted, chances, total, inverse, work, mark, remaining, chosen, selected, scores) = (t[50], t[51], t[52], t[53], t[54], t[55], t[56], t[57], t[58], t[59], t[60], t[61]);

        ops.matvec(&matrix("gate"), reg_of("gate"), 0, x, raw, experts, hidden);
        ops.maximum(raw, experts, top);
        ops.sub_scalar(raw, top, shifted, experts);
        ops.function(FN_EXP, shifted, chances, experts);

        // Repeatedly take the largest probability and blank it out.
        ops.copy_part(chances, 0, work, 0, experts);
        for pass in 0..top_k {
            ops.maximum(work, experts, top);
            ops.mark_equal(work, top, mark, experts);
            if pass == 0 {
                ops.copy_part(mark, 0, chosen, 0, experts);
            } else {
                ops.add(chosen, mark, t[62], experts);
                ops.copy_part(t[62], 0, chosen, 0, experts);
            }
            ops.add_sub(true, (self.ones, 0), (mark, 0), 1, remaining, experts);
            ops.multiply(work, remaining, t[62], experts);
            ops.copy_part(t[62], 0, work, 0, experts);
        }
        ops.multiply(chances, chosen, selected, experts);
        let ones = self.ones;
        ops.dots(selected, ones, total, 1, experts, |_, e| Some(((selected, e), (ones, e))));
        ops.reciprocal(total, inverse);
        ops.scale(inverse, selected, scores, experts);

        // Every expert's output goes on one exponent, ready to be weighted.
        let outputs = t[48];
        ops.set_exponent(outputs, EXPERT_EXPONENT);
        let (gate, up, product, down) = (t[34], t[35], t[36], t[37]);
        for e in 0..experts {
            if ops.known_zero(scores, e) {
                continue;
            }
            ops.matvec(&matrix("switch_mlp.gate_proj"), reg_of("switch_mlp.gate_proj"), e * intermediate, x, gate, intermediate, hidden);
            ops.matvec(&matrix("switch_mlp.up_proj"), reg_of("switch_mlp.up_proj"), e * intermediate, x, up, intermediate, hidden);
            self.silu(ops, gate, t[38], intermediate);
            ops.multiply(t[38], up, product, intermediate);
            ops.matvec(&matrix("switch_mlp.down_proj"), reg_of("switch_mlp.down_proj"), e * hidden, product, down, hidden, intermediate);
            ops.align_to(down, 0, EXPERT_EXPONENT, outputs, e * hidden, hidden);
        }
        let mixed = t[47];
        ops.dots(scores, outputs, mixed, hidden, experts, |j, e| Some(((scores, e), (outputs, e * hidden + j))));

        ops.matvec(&matrix("shared_expert.gate_proj"), reg_of("shared_expert.gate_proj"), 0, x, gate, shared, hidden);
        ops.matvec(&matrix("shared_expert.up_proj"), reg_of("shared_expert.up_proj"), 0, x, up, shared, hidden);
        self.silu(ops, gate, t[38], shared);
        ops.multiply(t[38], up, product, shared);
        ops.matvec(&matrix("shared_expert.down_proj"), reg_of("shared_expert.down_proj"), 0, product, down, hidden, shared);
        ops.matvec(&matrix("shared_expert_gate"), reg_of("shared_expert_gate"), 0, x, t[49], 1, hidden);
        ops.function(FN_SIGMOID, t[49], t[63], 1);
        ops.scale(t[63], down, t[62], hidden);
        ops.add(mixed, t[62], dst, hidden);
    }
}
