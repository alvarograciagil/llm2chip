//! Expresses a compiled program in the die's vector arithmetic, token by token.
//! The same code drives the gates (through microcode) and the exact virtual machine.

use crate::weights::Weights;
use compiler::{Inst, Op, Program, RopeScaling};
use netlist::block::{FN_GELU, FN_GELU_TANH, FN_TANH};
use netlist::block::{encode_vector, Assembly, AttentionRegs, BlockOps, DeltaRegs, Expr, Var, FN_EXP, FN_SIGMOID, FN_SOFTPLUS};
use netlist::{DieConfig, DieProgram, Microprogram};
use std::cell::Cell;
use std::collections::hash_map::Entry;
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
        _ => Vec::new(),
    }
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
    rope: [u32; 2],
    rope_table: [u32; 2],
    rope_half: usize,
    inverse_root: HashMap<usize, u32>,
    inverse_len: HashMap<usize, u32>,
    setup: Assembly,
}

impl Layout {
    /// Allocates registers and stores constants. With `chip_tokens`, the matrix weights
    /// and a RoPE table for that many tokens go into registers for the gates; otherwise
    /// the weights stay in the checkpoint and RoPE is computed directly.
    pub fn new(program: &Program, weights: &Weights, config: DieConfig, chip_tokens: Option<usize>) -> Result<Self, String> {
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
                    if chip_tokens.is_some() {
                        let matrix = weights.matrix(&name)?;
                        let exponent = matrix.exponent(cols);
                        let mut flat = Vec::with_capacity(rows * cols);
                        for r in 0..rows {
                            flat.extend(matrix.fixed_row(r, cols, exponent, config.cell.activation_bits)?);
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
                    let centred = program.family == "qwen3_5" && raw.iter().sum::<f32>() / raw.len() as f32 > 0.5;
                    let values = raw.iter().map(|w| (*w + if centred { 0.0 } else { *offset }) as f64).collect();
                    store(name, values, &mut setup, &mut small);
                }
                Op::Scale { factor, .. } => store(format!("escala:{factor}"), vec![*factor as f64], &mut setup, &mut small),
                Op::AddWeight { weight, .. } => store(weight.clone(), weights.floats(weight)?.iter().map(|v| *v as f64).collect(), &mut setup, &mut small),
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
                Op::AttnSum { scale_dim, softcap, .. } => {
                    root_of(*scale_dim, &mut setup, &mut inverse_root);
                    if let Some(cap) = softcap {
                        store(format!("escala:{}", 1.0 / cap), vec![1.0 / *cap as f64], &mut setup, &mut small);
                        store(format!("escala:{cap}"), vec![*cap as f64], &mut setup, &mut small);
                    }
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
        let rope_half = all_ops(program).find_map(|i| if let Op::Rope { rotary, .. } = i.op { Some(rotary / 2) } else { None }).unwrap_or(0);
        let rope_table = [alloc(1), alloc(1)];
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
        let next = next.get();
        if next >= (1 << config.reg_bits) - 2 {
            return Err(format!("faltan registros: se necesitan {next}"));
        }
        Ok(Self { config, buffers, weights: matrices, small, keys, values, conv, delta, next, ones, minus_one, epsilon, temporaries, parts, delta_parts, delta_outputs, rope, rope_table, rope_half, inverse_root, inverse_len, setup })
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

    /// The whole microprogram: loaded once, then entered with the token and position
    /// held in its parameter registers.
    pub fn microprogram(&self, program: &Program) -> Result<Microprogram, String> {
        let token = Expr::var(Var::Token);
        let position = Expr::var(Var::Position);
        let mut code = DieProgram::new(self.config);
        let routine = |code: &mut DieProgram, build: &mut dyn FnMut(&mut Assembly) -> Result<(), String>| {
            let mut asm = Assembly::new(&self.config);
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

        let mut asm = Assembly::new(&self.config);
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

        let mut asm = Assembly::new(&self.config);
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
    pub fn keep_weights_of(&mut self, program: &Program, plan: &[Range<usize>], die: usize) {
        let last = die + 1 == plan.len();
        let tail = program.logits_from..program.ops.len();
        let used: HashSet<String> = plan[die].clone().chain(if last { tail } else { 0..0 }).flat_map(|i| matrices_of(program, &program.ops[i].op)).map(|(name, ..)| name).collect();
        let dropped: HashSet<u32> = self.weights.iter().filter(|(name, _)| !used.contains(*name)).map(|(_, reg)| *reg).collect();
        self.setup.initial.retain(|(register, _), _| !dropped.contains(register));
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
    }

    fn reg(&self, buf: usize) -> u32 {
        self.buffers[buf]
    }

    fn rms_blocks<O: BlockOps>(&self, ops: &mut O, src: u32, weight: Option<u32>, block: usize, count: usize, dst: u32) {
        let t = &self.temporaries;
        let d = block as u32;
        let first = self.parts[0];
        ops.each((count as u32).into(), |ops, b| {
            ops.with_window(b, &[first], |ops, offset| {
                let part = first + offset;
                ops.copy_part(src, b * d, t[40], 0.into(), d.into());
                ops.normalize(t[40], d.into());
                ops.rms_normalize(t[40], t[41], d, (t[42], t[43]), self.inverse_len.get(&block).copied(), Some(self.epsilon));
                match weight {
                    Some(w) => ops.multiply(t[41], w, part, d.into()),
                    None => ops.copy_part(t[41], 0.into(), part, 0.into(), d.into()),
                }
            });
        });
        let parts: Vec<(u32, u32)> = (0..count).map(|b| (self.parts[b], d)).collect();
        ops.gather(&parts, dst);
    }

    fn silu<O: BlockOps>(&self, ops: &mut O, src: u32, dst: u32, len: O::N) {
        let t = self.temporaries[39];
        ops.function(FN_SIGMOID, src, t, len);
        ops.multiply(t, src, dst, len);
    }

    fn lower_op<O: BlockOps>(&self, ops: &mut O, program: &Program, op: &Op, token: O::N, position: O::N) -> Result<(), String> {
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
                ops.matvec(matrix, self.weights[matrix], 0, self.reg(*src), self.reg(*dst), rows, cols);
            }
            Op::Copy { src, dst } => ops.copy_part(self.reg(*src), zero, self.reg(*dst), zero, size(*dst)),
            Op::Add { a, b, dst } => ops.add(self.reg(*a), self.reg(*b), self.reg(*dst), size(*dst)),
            Op::MulElem { a, b, dst } => ops.multiply(self.reg(*a), self.reg(*b), self.reg(*dst), size(*dst)),
            Op::Gelu { src, approximate, dst } => {
                let table = t_gelu(*approximate);
                let extra = self.temporaries[39];
                ops.function(table, self.reg(*src), extra, size(*dst));
                ops.multiply(extra, self.reg(*src), self.reg(*dst), size(*dst));
            }
            Op::SoftCap { src, cap, dst } => {
                let (scaled, shaped) = (self.temporaries[38], self.temporaries[39]);
                ops.scale(self.small[&format!("escala:{}", 1.0 / cap)], self.reg(*src), scaled, size(*dst));
                ops.function(FN_TANH, scaled, shaped, size(*dst));
                ops.scale(self.small[&format!("escala:{cap}")], shaped, self.reg(*dst), size(*dst));
            }
            Op::Scale { src, factor, dst } => ops.scale(self.small[&format!("escala:{factor}")], self.reg(*src), self.reg(*dst), size(*dst)),
            Op::AddWeight { src, weight, dst } => ops.add(self.reg(*src), self.small[weight], self.reg(*dst), size(*dst)),
            Op::Silu { src, dst } => self.silu(ops, self.reg(*src), self.reg(*dst), size(*dst)),
            Op::Sigmoid { src, dst } => ops.function(FN_SIGMOID, self.reg(*src), self.reg(*dst), size(*dst)),
            Op::RmsNorm { src, weight, block, dst, .. } => {
                let count = program.buffers[*dst] / block;
                let w = self.small[&format!("{weight}.weight")];
                self.rms_blocks(ops, self.reg(*src), Some(w), *block, count, self.reg(*dst));
            }
            Op::SplitQueryGate { src, query, gate, block } => {
                let heads = program.buffers[*query] / block;
                let d = *block as u32;
                ops.each((heads as u32).into(), |ops, h| {
                    ops.copy_part(self.reg(*src), h * (2 * d), t[0], h * d, d.into());
                    ops.copy_part(self.reg(*src), h * (2 * d) + d.into(), t[1], h * d, d.into());
                });
                ops.gather(&[(t[0], (heads * block) as u32)], self.reg(*query));
                ops.gather(&[(t[1], (heads * block) as u32)], self.reg(*gate));
            }
            Op::Rope { buf, block, table, .. } => {
                let reg = self.reg(*buf);
                let total = program.buffers[*buf] as u32;
                let (d, h, rope) = (*block as u32, self.rope_half as u32, self.rope[*table]);
                ops.each((total / d).into(), |ops, head| {
                    let base = head * d;
                    ops.each(h.into(), |ops, j| {
                        let at = base + j;
                        ops.sum_products(at, &[((reg, at), (rope, j)), ((reg, at + h.into()), (rope, j + h.into()))]);
                    });
                    ops.each(h.into(), |ops, j| {
                        let at = base + j + h.into();
                        ops.sum_products(at, &[((reg, at), (rope, j)), ((reg, base + j), (rope, j + (2 * h).into()))]);
                    });
                    ops.each((d - 2 * h).into(), |ops, j| {
                        let at = base + j + (2 * h).into();
                        ops.sum_products(at, &[((reg, at), (rope, (3 * h).into()))]);
                    });
                });
                ops.narrow_sums(reg, rope, t[0], total.into());
                ops.copy_part(t[0], zero, reg, zero, total.into());
            }
            Op::KvAppend { slot, key, value, window } => {
                let width = program.buffers[*key] as u32;
                let slot_place = match window {
                    Some(limit) => ops.window_slot(position, *limit as u32),
                    None => position,
                };
                let at = slot_place * width;
                ops.align_to(self.reg(*key), zero, STATE_EXPONENT, self.keys[*slot], at, width.into());
                ops.align_to(self.reg(*value), zero, STATE_EXPONENT, self.values[*slot], at, width.into());
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
                ops.each((*kv_heads as u32).into(), |ops, kv_head| {
                    ops.each((group as u32).into(), |ops, shared| {
                        let h = kv_head * group as u32 + shared;
                        ops.with_window(h, &[first], |ops, offset| {
                            let (qh, kt, vt) = (t[20], t[21], t[22]);
                            ops.copy_part(query, h * d, qh, zero, d.into());
                            ops.normalize(qh, d.into());
                            ops.each(tokens, |ops, token_index| {
                                let from = token_index * width + kv_head * d;
                                ops.copy_part(keys, from, kt, token_index * d, d.into());
                                ops.copy_part(values, from, vt, token_index * d, d.into());
                            });
                            ops.normalize(kt, tokens * d);
                            ops.normalize(vt, tokens * d);
                            ops.attention(qh, kt, vt, self.ones, first + offset, d, tokens, Some(self.inverse_root[scale_dim]), regs);
                        });
                    });
                });
                let parts: Vec<(u32, u32)> = (0..*heads).map(|h| (self.parts[h], d)).collect();
                ops.gather(&parts, self.reg(*dst));
            }
            Op::Dconv { slot, weight, src, dst } => {
                let (dim, kernel) = program.conv_slots[*slot];
                let history = (kernel - 1) as u32;
                let taps = self.small[&format!("{weight}.weight")];
                let reg = self.conv[*slot];
                let (dim32, kernel32) = (dim as u32, kernel as u32);
                ops.align_to(self.reg(*src), zero, STATE_EXPONENT, reg, (history * dim32).into(), dim32.into());
                ops.dots(taps, reg, self.reg(*dst), dim32.into(), kernel32.into(), |c, k| ((taps, c * kernel32 + k), (reg, k * dim32 + c)));
                ops.each(history.into(), |ops, k| {
                    ops.each(dim32.into(), |ops, c| {
                        ops.move_element((reg, (k + 1.into()) * dim32 + c), (reg, k * dim32 + c));
                    });
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
                let (first_state, first_out) = (self.delta[*slot][0], self.delta_parts[0]);
                ops.each((*key_heads as u32).into(), |ops, key_head| {
                    ops.each((ratio as u32).into(), |ops, shared| {
                        let h = key_head * ratio as u32 + shared;
                        ops.with_window(h, &[first_state, first_out], |ops, offset| {
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

                            let regs = DeltaRegs { state: first_state + offset, key: kh, query: qh, value: vh, beta, decay, memory: t[10], error: t[11], delta: t[12], update: t[13], next: t[14], out: first_out + offset };
                            ops.delta_step(regs, kd, vd);
                        });
                    });
                });
                let parts: Vec<(u32, u32)> = (0..*value_heads).map(|h| (self.delta_parts[h], vd)).collect();
                ops.gather(&parts, self.reg(*dst));
            }
            Op::RmsGated { src, z, weight, block, dst } => {
                let count = program.buffers[*src] / block;
                let w = self.small[&format!("{weight}.weight")];
                let by_head = self.delta_outputs.contains(src);
                let (src, z) = (self.reg(*src), self.reg(*z));
                let d = *block as u32;
                let (first_out, first_part) = (self.delta_parts[0], self.parts[0]);
                ops.each((count as u32).into(), |ops, b| {
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
                    });
                });
                let parts: Vec<(u32, u32)> = (0..count).map(|b| (self.parts[b], d)).collect();
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
        let zero: O::N = 0.into();
        let (experts_n, hidden_n): (O::N, O::N) = (experts.into(), hidden.into());
        let matrix = |name: &str| format!("{prefix}.{name}");
        let reg_of = |name: &str| self.weights[&matrix(name)];
        let (raw, top, shifted, chances, total, inverse, work, mark, remaining, chosen, selected, scores) = (t[50], t[51], t[52], t[53], t[54], t[55], t[56], t[57], t[58], t[59], t[60], t[61]);

        ops.matvec(&matrix("gate"), reg_of("gate"), 0, x, raw, experts, hidden);
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
        for e in 0..experts {
            if ops.known_zero(scores, e) {
                continue;
            }
            ops.matvec(&matrix("switch_mlp.gate_proj"), reg_of("switch_mlp.gate_proj"), e * intermediate, x, gate, intermediate, hidden);
            ops.matvec(&matrix("switch_mlp.up_proj"), reg_of("switch_mlp.up_proj"), e * intermediate, x, up, intermediate, hidden);
            self.silu(ops, gate, t[38], intermediate.into());
            ops.multiply(t[38], up, product, intermediate.into());
            ops.matvec(&matrix("switch_mlp.down_proj"), reg_of("switch_mlp.down_proj"), e * hidden, product, down, hidden, intermediate);
            ops.align_to(down, zero, EXPERT_EXPONENT, outputs, (e * hidden).into(), hidden_n);
        }
        let mixed = t[47];
        ops.dots(scores, outputs, mixed, hidden_n, experts_n, |j, e| ((scores, e), (outputs, e * hidden + j)));

        ops.matvec(&matrix("shared_expert.gate_proj"), reg_of("shared_expert.gate_proj"), 0, x, gate, shared, hidden);
        ops.matvec(&matrix("shared_expert.up_proj"), reg_of("shared_expert.up_proj"), 0, x, up, shared, hidden);
        self.silu(ops, gate, t[38], shared.into());
        ops.multiply(t[38], up, product, shared.into());
        ops.matvec(&matrix("shared_expert.down_proj"), reg_of("shared_expert.down_proj"), 0, product, down, hidden, shared);
        ops.matvec(&matrix("shared_expert_gate"), reg_of("shared_expert_gate"), 0, x, t[49], 1, hidden);
        ops.function(FN_SIGMOID, t[49], t[63], 1.into());
        ops.scale(t[63], down, t[62], hidden_n);
        ops.add(mixed, t[62], dst, hidden_n);
    }
}
