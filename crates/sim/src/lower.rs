//! Lowers a compiled program to die microcode, one token at a time, so a real
//! (if tiny) model runs on the gates exactly as the executor runs it in software.

use crate::weights::Weights;
use compiler::{Op, Program};
use netlist::block::{Assembly, AttentionRegs, DeltaRegs, FN_EXP, FN_SIGMOID, FN_SOFTPLUS};
use netlist::DieConfig;
use std::collections::HashMap;

/// KV entries and convolution history are stored with this fixed exponent.
const STATE_EXPONENT: i64 = 6;
const ONES_LENGTH: usize = 32;

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
    temporaries: Vec<u32>,
    rope: u32,
    constant_inverse: HashMap<usize, u32>,
    setup: Assembly,
}

/// For each output element, the register places whose products are summed.
type DotSets = Vec<Vec<((u32, u32), (u32, u32))>>;

fn sizes(program: &Program, inst: &Op) -> Option<(String, usize, usize)> {
    if let Op::Matvec { matrix, src, dst } = inst {
        return Some((matrix.clone(), program.buffers[*dst], program.buffers[*src]));
    }
    if let Op::EmbRead { table, dst } = inst {
        return Some((table.clone(), program.vocab, program.buffers[*dst]));
    }
    None
}

impl Layout {
    /// Allocates registers and stores all weights and constants for the host to load.
    pub fn new(program: &Program, weights: &Weights, config: DieConfig) -> Result<Self, String> {
        let next = std::cell::Cell::new(1u32);
        let alloc = |count: u32| next.replace(next.get() + count);
        let buffers: Vec<u32> = program.buffers.iter().map(|_| alloc(1)).collect();
        let mut setup = Assembly::new(&config);
        let mut matrices: HashMap<String, u32> = HashMap::new();
        let mut small: HashMap<String, u32> = HashMap::new();
        let mut constant_inverse = HashMap::new();

        for inst in &program.ops {
            if let Some((name, rows, cols)) = sizes(program, &inst.op) {
                if let std::collections::hash_map::Entry::Vacant(slot) = matrices.entry(name.clone()) {
                    let matrix = weights.matrix(&name)?;
                    let mut flat = Vec::with_capacity(rows * cols);
                    for r in 0..rows {
                        flat.extend(matrix.row(r, cols)?.iter().map(|v| *v as f64));
                    }
                    let reg = alloc(1);
                    setup.load_vector(reg, &flat);
                    slot.insert(reg);
                }
            }
            let mut store = |name: String, values: Vec<f64>, setup: &mut Assembly| {
                small.entry(name).or_insert_with(|| {
                    let reg = alloc(1);
                    setup.load_vector(reg, &values);
                    reg
                });
            };
            if let Op::AttnSum { head_dim, .. } = &inst.op {
                constant_inverse.entry(*head_dim).or_insert_with(|| {
                    let reg = alloc(1);
                    setup.load_vector(reg, &[1.0 / (*head_dim as f64).sqrt()]);
                    reg
                });
            }
            match &inst.op {
                Op::RmsNorm { weight, offset, .. } => {
                    let name = format!("{weight}.weight");
                    let raw = weights.floats(&name)?;
                    let centred = raw.iter().sum::<f32>() / raw.len() as f32 > 0.5;
                    let values = raw.iter().map(|w| (*w + if centred { 0.0 } else { *offset }) as f64).collect();
                    store(name, values, &mut setup);
                }
                Op::RmsGated { weight, .. } | Op::Dconv { weight, .. } => {
                    let name = format!("{weight}.weight");
                    let values = weights.floats(&name)?.iter().map(|w| *w as f64).collect();
                    store(name, values, &mut setup);
                }
                Op::DeltaStep { a_log, dt_bias, key_dim, .. } => {
                    let scales: Vec<f64> = weights.floats(a_log)?.iter().map(|v| v.exp() as f64).collect();
                    store(a_log.clone(), scales, &mut setup);
                    store(dt_bias.clone(), weights.floats(dt_bias)?.iter().map(|v| *v as f64).collect(), &mut setup);
                    if !constant_inverse.contains_key(key_dim) {
                        let reg = alloc(1);
                        setup.load_vector(reg, &[1.0 / (*key_dim as f64).sqrt()]);
                        constant_inverse.insert(*key_dim, reg);
                    }
                }
                _ => {}
            }
        }

        let state_register = |exponent: i64, setup: &mut Assembly| {
            let reg = alloc(1);
            let place = setup.exponent(reg);
            setup.initial.insert(place, exponent);
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
        let rope = alloc(1);
        let temporaries: Vec<u32> = (0..48).map(|_| alloc(1)).collect();
        let next = next.get();
        if next >= (1 << config.reg_bits) - 2 {
            return Err(format!("faltan registros: se necesitan {next}"));
        }
        Ok(Self { config, buffers, weights: matrices, small, keys, values, conv, delta, next, ones, minus_one, temporaries, rope, constant_inverse, setup })
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

    /// The microcode and per-token constants for one token at `position`.
    pub fn token(&self, program: &Program, token: u32, position: usize, want_logits: bool) -> Result<Assembly, String> {
        let mut asm = Assembly::new(&self.config);
        let rotary = program.ops.iter().find_map(|i| if let Op::Rope { rotary, .. } = i.op { Some(rotary) } else { None }).unwrap_or(0);
        if rotary > 0 {
            let half = rotary / 2;
            let (mut cos, mut sin) = (Vec::new(), Vec::new());
            for i in 0..half {
                let frequency = 1.0 / (program.rope_theta as f64).powf((2 * i) as f64 / rotary as f64);
                cos.push((position as f64 * frequency).cos());
                sin.push((position as f64 * frequency).sin());
            }
            let mut table = cos;
            table.extend(sin.iter().map(|s| -s));
            table.extend(sin.iter());
            table.push(1.0);
            asm.load_vector(self.rope, &table);
        }
        let limit = if want_logits { program.ops.len() } else { program.logits_from };
        for inst in &program.ops[..limit] {
            self.lower_op(&mut asm, program, &inst.op, token, position, rotary / 2)?;
        }
        Ok(asm)
    }

    fn reg(&self, buf: usize) -> u32 {
        self.buffers[buf]
    }

    fn matvec_sets(rows: usize, cols: usize, matrix: u32, src: u32) -> DotSets {
        (0..rows).map(|r| (0..cols).map(|j| ((matrix, (r * cols + j) as u32), (src, j as u32))).collect()).collect()
    }

    fn guard(len: usize) -> u32 {
        32 - (len.max(1) as u32 - 1).leading_zeros()
    }

    /// `x · rsqrt(mean(x²) + eps) · w`, block by block, written to `dst`.
    fn rms_blocks(&self, asm: &mut Assembly, src: u32, weight: Option<u32>, block: usize, count: usize, dst: u32) {
        let t = &self.temporaries;
        let mut parts = Vec::new();
        for b in 0..count {
            let part = t[b];
            asm.copy_part(src, (b * block) as u32, t[40], 0, block as u32);
            asm.normalize(t[40], block as u32);
            asm.rms_normalize(t[40], t[41], block as u32, (t[42], t[43]));
            match weight {
                Some(w) => asm.multiply(t[41], w, part, block as u32),
                None => asm.copy_part(t[41], 0, part, 0, block as u32),
            }
            parts.push((part, block as u32));
        }
        asm.gather(&parts, dst);
    }

    fn silu(&self, asm: &mut Assembly, src: u32, dst: u32, len: u32) {
        let t = self.temporaries[39];
        asm.function(FN_SIGMOID, src, t, len);
        asm.multiply(t, src, dst, len);
    }

    fn lower_op(&self, asm: &mut Assembly, program: &Program, op: &Op, token: u32, position: usize, half: usize) -> Result<(), String> {
        let t = &self.temporaries;
        match op {
            Op::EmbRead { table, dst } => {
                let len = program.buffers[*dst] as u32;
                asm.copy_part(self.weights[table], token * len, self.reg(*dst), 0, len);
                asm.normalize(self.reg(*dst), len);
            }
            Op::Matvec { matrix, src, dst } => {
                let (rows, cols) = (program.buffers[*dst], program.buffers[*src]);
                let sets = Self::matvec_sets(rows, cols, self.weights[matrix], self.reg(*src));
                asm.dots(&sets, self.reg(*dst), self.weights[matrix], self.reg(*src), Self::guard(cols));
            }
            Op::Copy { src, dst } => asm.copy_part(self.reg(*src), 0, self.reg(*dst), 0, program.buffers[*dst] as u32),
            Op::Add { a, b, dst } => asm.add(self.reg(*a), self.reg(*b), self.reg(*dst), program.buffers[*dst] as u32),
            Op::MulElem { a, b, dst } => asm.multiply(self.reg(*a), self.reg(*b), self.reg(*dst), program.buffers[*dst] as u32),
            Op::Silu { src, dst } => self.silu(asm, self.reg(*src), self.reg(*dst), program.buffers[*dst] as u32),
            Op::Sigmoid { src, dst } => asm.function(FN_SIGMOID, self.reg(*src), self.reg(*dst), program.buffers[*dst] as u32),
            Op::RmsNorm { src, weight, block, dst, .. } => {
                let count = program.buffers[*dst] / block;
                let w = self.small[&format!("{weight}.weight")];
                self.rms_blocks(asm, self.reg(*src), Some(w), *block, count, self.reg(*dst));
            }
            Op::SplitQueryGate { src, query, gate, block } => {
                let heads = program.buffers[*query] / block;
                let (q, g) = (t[0], t[1]);
                for h in 0..heads {
                    let d = *block as u32;
                    let h32 = h as u32;
                    asm.copy_part(self.reg(*src), h32 * 2 * d, q, h32 * d, d);
                    asm.copy_part(self.reg(*src), h32 * 2 * d + d, g, h32 * d, d);
                }
                asm.gather(&[(q, (heads * block) as u32)], self.reg(*query));
                asm.gather(&[(g, (heads * block) as u32)], self.reg(*gate));
            }
            Op::Rope { buf, block, .. } => {
                let reg = self.reg(*buf);
                let total = program.buffers[*buf];
                let mut sets = Vec::with_capacity(total);
                for head in 0..total / block {
                    for j in 0..*block {
                        let at = (head * block + j) as u32;
                        let base = (head * block) as u32;
                        let h = half as u32;
                        let j32 = j as u32;
                        if j < half {
                            sets.push(vec![((reg, at), (self.rope, j32)), ((reg, base + h + j32), (self.rope, h + j32))]);
                        } else if j < 2 * half {
                            let i = j32 - h;
                            sets.push(vec![((reg, at), (self.rope, i)), ((reg, base + i), (self.rope, 2 * h + i))]);
                        } else {
                            sets.push(vec![((reg, at), (self.rope, 3 * h))]);
                        }
                    }
                }
                asm.dots(&sets, t[0], reg, self.rope, 1);
                asm.copy_part(t[0], 0, reg, 0, total as u32);
            }
            Op::KvAppend { slot, key, value } => {
                let width = program.buffers[*key] as u32;
                let at = position as u32 * width;
                asm.align_to(self.reg(*key), 0, STATE_EXPONENT, self.keys[*slot], at, width);
                asm.align_to(self.reg(*value), 0, STATE_EXPONENT, self.values[*slot], at, width);
            }
            Op::AttnSum { slot, query, head_dim, heads, kv_heads, dst } => {
                let group = heads / kv_heads;
                let tokens = position + 1;
                let width = program.kv_slots[*slot].heads * head_dim;
                let d = *head_dim as u32;
                let regs = AttentionRegs { scores: t[10], top: t[11], shifted: t[12], weights: t[13], total: t[14], inverse: t[15], probabilities: t[16] };
                let mut parts = Vec::new();
                for h in 0..*heads {
                    let kv_head = (h / group) as u32;
                    let (qh, kt, vt) = (t[20], t[21], t[22]);
                    asm.copy_part(self.reg(*query), h as u32 * d, qh, 0, d);
                    asm.normalize(qh, d);
                    for token_index in 0..tokens as u32 {
                        let from = token_index * width as u32 + kv_head * d;
                        let to = token_index * d;
                        asm.copy_part(self.keys[*slot], from, kt, to, d);
                        asm.copy_part(self.values[*slot], from, vt, to, d);
                    }
                    asm.normalize(kt, tokens as u32 * d);
                    asm.normalize(vt, tokens as u32 * d);
                    asm.attention(qh, kt, vt, self.ones, t[2 + h], d, tokens as u32, Some(self.constant_inverse[head_dim]), regs);
                    parts.push((t[2 + h], d));
                }
                asm.gather(&parts, self.reg(*dst));
            }
            Op::Dconv { slot, weight, src, dst } => {
                let (dim, kernel) = program.conv_slots[*slot];
                let history = kernel - 1;
                let taps = self.small[&format!("{weight}.weight")];
                let reg = self.conv[*slot];
                let input = self.reg(*src);
                // New input joins the end of the history register, on the fixed exponent.
                asm.align_to(input, 0, STATE_EXPONENT, reg, (history * dim) as u32, dim as u32);
                let sets: DotSets = (0..dim)
                    .map(|c| (0..kernel).map(|k| ((taps, (c * kernel + k) as u32), (reg, (k * dim + c) as u32))).collect())
                    .collect();
                asm.dots(&sets, self.reg(*dst), taps, reg, Self::guard(kernel));
                // Slide the history window for the next token.
                for k in 0..history {
                    for c in 0..dim {
                        asm.binary(netlist::die::OP_MOV, (reg, ((k + 1) * dim + c) as u32), (0, 0), (reg, (k * dim + c) as u32));
                    }
                }
            }
            Op::DeltaStep { slot, mixed, a, b, a_log, dt_bias, key_dim, key_heads, value_dim, value_heads, dst } => {
                let ratio = value_heads / key_heads;
                let (kd, vd) = (*key_dim as u32, *value_dim as u32);
                let key_len = kd * *key_heads as u32;
                let root = self.constant_inverse[key_dim];
                let mixed = self.reg(*mixed);
                let (a_reg, b_reg) = (self.reg(*a), self.reg(*b));
                let (scale_a, bias) = (self.small[a_log], self.small[dt_bias]);
                let mut parts = Vec::new();
                for h in 0..*value_heads {
                    let key_head = (h / ratio) as u32;
                    let h32 = h as u32;
                    let (qh, kh, vh, beta, decay) = (t[20], t[21], t[22], t[23], t[24]);
                    let (work, inv) = (t[25], t[26]);

                    asm.copy_part(mixed, key_head * kd, work, 0, kd);
                    asm.normalize(work, kd);
                    asm.l2_normalize(work, t[27], kd, (inv, t[28]));
                    asm.scale(root, t[27], qh, kd);
                    asm.copy_part(mixed, key_len + key_head * kd, work, 0, kd);
                    asm.normalize(work, kd);
                    asm.l2_normalize(work, kh, kd, (inv, t[28]));
                    asm.copy_part(mixed, 2 * key_len + h32 * vd, vh, 0, vd);
                    asm.normalize(vh, vd);

                    asm.copy_part(b_reg, h32, work, 0, 1);
                    asm.normalize(work, 1);
                    asm.function(FN_SIGMOID, work, beta, 1);

                    asm.copy_part(a_reg, h32, t[29], 0, 1);
                    asm.normalize(t[29], 1);
                    asm.copy_part(bias, h32, t[30], 0, 1);
                    asm.normalize(t[30], 1);
                    asm.add(t[29], t[30], work, 1);
                    asm.function(FN_SOFTPLUS, work, t[31], 1);
                    asm.copy_part(scale_a, h32, t[32], 0, 1);
                    asm.normalize(t[32], 1);
                    asm.multiply(t[31], t[32], t[33], 1);
                    asm.multiply(t[33], self.minus_one, work, 1);
                    asm.function(FN_EXP, work, decay, 1);

                    let regs = DeltaRegs { state: self.delta[*slot][h], key: kh, query: qh, value: vh, beta, decay, memory: t[10], error: t[11], delta: t[12], update: t[13], next: t[14], out: t[2 + h] };
                    asm.delta_step(regs, kd, vd);
                    parts.push((t[2 + h], vd));
                }
                asm.gather(&parts, self.reg(*dst));
            }
            Op::RmsGated { src, z, weight, block, dst } => {
                let count = program.buffers[*src] / block;
                let w = self.small[&format!("{weight}.weight")];
                let (src, z) = (self.reg(*src), self.reg(*z));
                let mut parts = Vec::new();
                for b in 0..count {
                    let d = *block as u32;
                    let b32 = b as u32;
                    asm.copy_part(src, b32 * d, t[40], 0, d);
                    asm.normalize(t[40], d);
                    asm.rms_normalize(t[40], t[41], d, (t[42], t[43]));
                    asm.multiply(t[41], w, t[44], d);
                    asm.copy_part(z, b32 * d, t[45], 0, d);
                    asm.normalize(t[45], d);
                    self.silu(asm, t[45], t[46], d);
                    asm.multiply(t[44], t[46], t[b], d);
                    parts.push((t[b], d));
                }
                asm.gather(&parts, self.reg(*dst));
            }
            Op::Concat { .. } | Op::Moe { .. } => return Err("esta operación aún no se puede bajar a puertas".into()),
        }
        Ok(())
    }
}
