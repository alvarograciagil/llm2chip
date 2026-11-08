//! Block floating point on the die: each vector register has one exponent.
//! Macro operations lower to single-element steps that have a behavioural model,
//! so the gates can be checked against an implementation that shares no logic.

use crate::die::{OP_ACCUMULATOR, OP_ADD, OP_FUNCTION, OP_IMMEDIATE, OP_LZC, OP_MAX, OP_MOV, OP_MUL, OP_PARITY, OP_SHIFT_LEFT, OP_SUB};
use crate::DieConfig;
use std::collections::HashMap;

pub type Place = (u32, u32);

/// The register holding one exponent per vector register, by register number.
pub fn exponent_register(c: &DieConfig) -> u32 {
    (1 << c.reg_bits) - 1
}

fn scratch_register(c: &DieConfig) -> u32 {
    (1 << c.reg_bits) - 2
}

pub const FN_EXP: u32 = 0;
pub const FN_SIGMOID: u32 = 1;
pub const FN_RECIPROCAL: u32 = 2;
pub const FN_INV_SQRT: u32 = 3;
pub const FN_SOFTPLUS: u32 = 4;

/// Each table's input spans ±2^`INPUT_EXPONENT` unless it reads a raw mantissa.
const TABLE_INPUT_EXPONENT: i64 = 3;
const SOFTPLUS_OUTPUT_EXPONENT: i64 = 4;
const UNIT_OUTPUT_EXPONENT: i64 = 0;
const RECIPROCAL_OUTPUT_EXPONENT: i64 = 1;

#[derive(Clone, Copy, Debug)]
pub struct Alu {
    pub op: u32,
    pub a: Place,
    pub b: Place,
    pub dst: Place,
    pub shift: u32,
    pub function: u32,
    pub literal: Option<u32>,
    pub write: bool,
    pub load_shift_a: bool,
    pub load_shift_b: bool,
    pub load_shift_left: bool,
    pub magnitude_scan: bool,
    pub magnitude_clear: bool,
    pub accumulate: bool,
    pub accumulator_clear: bool,
    pub link_send: bool,
}

impl Alu {
    pub fn new(op: u32) -> Self {
        Alu {
            op,
            a: (0, 0),
            b: (0, 0),
            dst: (0, 0),
            shift: 0,
            function: 0,
            literal: None,
            write: true,
            load_shift_a: false,
            load_shift_b: false,
            load_shift_left: false,
            magnitude_scan: false,
            magnitude_clear: false,
            accumulate: false,
            accumulator_clear: false,
            link_send: false,
        }
    }
}

fn table_value(function: u32, index: usize, index_bits: u32, bits: u32) -> i64 {
    let top = (1i64 << (bits - 1)) - 1;
    // Tables hold each cell's centre, which halves the worst-case error.
    let top_entry = 1i64 << (index_bits - 1);
    let signed = (((index as i64) << (64 - index_bits)) >> (64 - index_bits)) as f64 + 0.5;
    let input = signed / top_entry as f64 * 2f64.powi(TABLE_INPUT_EXPONENT as i32);
    let mantissa = signed / top_entry as f64;
    let (value, exponent) = match function {
        FN_EXP => (input.exp(), UNIT_OUTPUT_EXPONENT),
        FN_SIGMOID => (1.0 / (1.0 + (-input).exp()), UNIT_OUTPUT_EXPONENT),
        FN_RECIPROCAL => (if mantissa > 0.0 { 1.0 / mantissa } else { 0.0 }, RECIPROCAL_OUTPUT_EXPONENT),
        FN_INV_SQRT => (if mantissa > 0.0 { 1.0 / mantissa.sqrt() } else { 0.0 }, RECIPROCAL_OUTPUT_EXPONENT),
        FN_SOFTPLUS => ((1.0 + input.exp()).ln(), SOFTPLUS_OUTPUT_EXPONENT),
        _ => (0.0, 0),
    };
    ((value / 2f64.powi(exponent as i32) * (top + 1) as f64).round() as i64).clamp(-top - 1, top)
}

/// Every word of the function tables, in host load order.
pub fn tables(c: &DieConfig) -> Vec<i64> {
    let bits = c.cell.activation_bits;
    (0..1usize << (c.function_bits + c.function_select_bits))
        .map(|address| {
            let index = address & ((1 << c.function_bits) - 1);
            let function = (address >> c.function_bits) as u32;
            table_value(function, index, c.function_bits, bits)
        })
        .collect()
}

/// A program of single-element steps plus the register contents it starts from.
pub struct Assembly {
    pub steps: Vec<Alu>,
    pub initial: HashMap<Place, i64>,
    exponents_used: u32,
    bits: u32,
    index_bits: u32,
    exponents: u32,
    scratch: u32,
}

impl Assembly {
    pub fn new(c: &DieConfig) -> Self {
        Self { steps: Vec::new(), initial: HashMap::new(), exponents_used: 0, bits: c.cell.activation_bits, index_bits: c.index_bits, exponents: exponent_register(c), scratch: scratch_register(c) }
    }

    fn bits(&self) -> u32 {
        self.bits
    }

    pub fn exponent(&self, reg: u32) -> Place {
        (self.exponents, reg)
    }

    /// Stores a real-valued vector as mantissas with one shared exponent.
    pub fn load_vector(&mut self, reg: u32, values: &[f64]) {
        let largest = values.iter().fold(0f64, |m, v| m.max(v.abs()));
        let exponent = if largest > 0.0 { largest.log2().floor() as i64 + 1 } else { 0 };
        let scale = 2f64.powi(self.bits() as i32 - 1 - exponent as i32);
        let top = (1i64 << (self.bits() - 1)) - 1;
        for (i, v) in values.iter().enumerate() {
            self.initial.insert((reg, i as u32), ((v * scale).round() as i64).clamp(-top, top));
        }
        self.initial.insert(self.exponent(reg), exponent);
        self.exponents_used = self.exponents_used.max(reg + 1);
    }

    pub fn immediate(&mut self, value: i64, dst: Place) {
        let mut s = Alu::new(OP_IMMEDIATE);
        s.literal = Some(value as u32 & ((1 << self.index_bits) - 1));
        s.dst = dst;
        self.steps.push(s);
    }

    pub fn binary(&mut self, op: u32, a: Place, b: Place, dst: Place) {
        let mut s = Alu::new(op);
        s.a = a;
        s.b = b;
        s.dst = dst;
        self.steps.push(s);
    }

    fn load_shifts(&mut self, a: Option<Place>, b: Option<Place>, left: Option<Place>) {
        // Sources are read through the A shifter, so A has to be loaded last.
        for (place, which) in [(left, 2), (b, 1), (a, 0)] {
            let mut s = Alu::new(OP_MOV);
            s.a = place.unwrap_or((self.scratch, 15));
            match which {
                0 => s.load_shift_a = true,
                1 => s.load_shift_b = true,
                _ => s.load_shift_left = true,
            }
            s.write = false;
            self.steps.push(s);
        }
    }

    fn clear_shifts(&mut self) {
        // Shift registers are read through the mover, so zero them with a zero source.
        self.immediate(0, (self.scratch, 15));
        self.load_shifts(None, None, None);
    }

    /// Shifts `reg` left until its largest element fills the mantissa.
    pub fn normalize(&mut self, reg: u32, len: u32) {
        self.clear_shifts();
        let mut first = Alu::new(OP_MOV);
        first.a = (reg, 0);
        first.write = false;
        first.magnitude_clear = true;
        first.magnitude_scan = false;
        self.steps.push(first);
        for i in 0..len {
            let mut s = Alu::new(OP_MOV);
            s.a = (reg, i);
            s.magnitude_scan = true;
            s.write = false;
            self.steps.push(s);
        }
        let mut count = Alu::new(OP_LZC);
        count.dst = (self.scratch, 0);
        self.steps.push(count);
        self.load_shifts(None, None, Some((self.scratch, 0)));
        for i in 0..len {
            let mut s = Alu::new(OP_SHIFT_LEFT);
            s.a = (reg, i);
            s.dst = (reg, i);
            self.steps.push(s);
        }
        self.binary(OP_SUB, self.exponent(reg), (self.scratch, 0), self.exponent(reg));
        self.clear_shifts();
    }

    fn add_sub(&mut self, op: u32, a: Place, b: Place, b_stride: u32, dst: u32, len: u32) {
        self.clear_shifts();
        let (ea, eb) = (self.exponent(a.0), self.exponent(b.0));
        self.binary(OP_MAX, ea, eb, (self.scratch, 1));
        self.immediate(1, (self.scratch, 2));
        self.binary(OP_ADD, (self.scratch, 1), (self.scratch, 2), (self.scratch, 3));
        self.binary(OP_SUB, (self.scratch, 3), ea, (self.scratch, 4));
        self.binary(OP_SUB, (self.scratch, 3), eb, (self.scratch, 5));
        self.load_shifts(Some((self.scratch, 4)), Some((self.scratch, 5)), None);
        for i in 0..len {
            self.binary(op, (a.0, a.1 + i), (b.0, b.1 + i * b_stride), (dst, i));
        }
        self.clear_shifts();
        self.binary(OP_MOV, (self.scratch, 3), (self.scratch, 15), self.exponent(dst));
        self.normalize(dst, len);
    }

    pub fn add(&mut self, a: u32, b: u32, dst: u32, len: u32) {
        self.add_sub(OP_ADD, (a, 0), (b, 0), 1, dst, len);
    }

    /// `dst[i] = a[i] - b[0]`
    pub fn sub_scalar(&mut self, a: u32, b: u32, dst: u32, len: u32) {
        self.add_sub(OP_SUB, (a, 0), (b, 0), 0, dst, len);
    }

    fn product_exponent(&mut self, a: u32, b: u32, dst: u32, extra: i64) {
        self.binary(OP_ADD, self.exponent(a), self.exponent(b), (self.scratch, 6));
        self.immediate(extra, (self.scratch, 7));
        self.binary(OP_ADD, (self.scratch, 6), (self.scratch, 7), self.exponent(dst));
    }

    pub fn multiply(&mut self, a: u32, b: u32, dst: u32, len: u32) {
        let bits = self.bits();
        for i in 0..len {
            let mut s = Alu::new(OP_MUL);
            s.a = (a, i);
            s.b = (b, i);
            s.dst = (dst, i);
            s.shift = bits - 1;
            self.steps.push(s);
        }
        self.product_exponent(a, b, dst, 0);
        self.normalize(dst, len);
    }

    /// `dst[i] = s[0] · a[i]`
    pub fn scale(&mut self, s_reg: u32, a: u32, dst: u32, len: u32) {
        let bits = self.bits();
        for i in 0..len {
            let mut s = Alu::new(OP_MUL);
            s.a = (s_reg, 0);
            s.b = (a, i);
            s.dst = (dst, i);
            s.shift = bits - 1;
            self.steps.push(s);
        }
        self.product_exponent(s_reg, a, dst, 0);
        self.normalize(dst, len);
    }

    /// Writes `Σ a·b` over `pairs` to `dst[index]`, scaled by the guard exponent.
    fn sum_of_products(&mut self, pairs: &[(Place, Place)], dst: Place, guard: u32) {
        let bits = self.bits();
        let mut clear = Alu::new(OP_MOV);
        clear.write = false;
        clear.accumulator_clear = true;
        self.steps.push(clear);
        for (a, b) in pairs {
            let mut s = Alu::new(OP_MOV);
            s.a = *a;
            s.b = *b;
            s.write = false;
            s.accumulate = true;
            self.steps.push(s);
        }
        let mut out = Alu::new(OP_ACCUMULATOR);
        out.shift = bits - 1 + guard;
        out.dst = dst;
        self.steps.push(out);
    }

    /// `dst[index] = Σ a[ia] · b[ib]` over `pairs`, sharing one guard.
    pub fn dots(&mut self, sets: &[Vec<(Place, Place)>], dst: u32, a: u32, b: u32, guard: u32) {
        for (i, pairs) in sets.iter().enumerate() {
            self.sum_of_products(pairs, (dst, i as u32), guard);
        }
        self.product_exponent(a, b, dst, guard as i64);
        self.normalize(dst, sets.len() as u32);
    }

    pub fn pairs(a: u32, b: u32, len: u32) -> Vec<(Place, Place)> {
        (0..len).map(|i| ((a, i), (b, i))).collect()
    }

    /// `dst[0] = Σ a[i] · b[i]`
    pub fn dot(&mut self, a: u32, b: u32, dst: u32, len: u32) {
        let guard = 32 - (len.max(1) - 1).leading_zeros();
        self.dots(&[Self::pairs(a, b, len)], dst, a, b, guard);
    }

    /// Puts the elements of `src` into the fixed format the tables read.
    fn align_for_table(&mut self, src: u32, len: u32) {
        self.align_to(src, 0, TABLE_INPUT_EXPONENT, self.scratch, 20, len);
    }

    /// Copies `src[from..]` into `dst[to..]`, re-expressed with a fixed exponent.
    pub fn align_to(&mut self, src: u32, from: u32, exponent: i64, dst: u32, to: u32, len: u32) {
        let e = self.exponent(src);
        self.immediate(exponent, (self.scratch, 8));
        self.binary(OP_SUB, e, (self.scratch, 8), (self.scratch, 9));
        self.immediate(0, (self.scratch, 10));
        self.binary(OP_MAX, (self.scratch, 9), (self.scratch, 10), (self.scratch, 11));
        self.binary(OP_SUB, (self.scratch, 8), e, (self.scratch, 12));
        self.binary(OP_MAX, (self.scratch, 12), (self.scratch, 10), (self.scratch, 13));
        self.load_shifts(Some((self.scratch, 13)), None, Some((self.scratch, 11)));
        for i in 0..len {
            let mut right = Alu::new(OP_MOV);
            right.a = (src, from + i);
            right.dst = (dst, to + i);
            self.steps.push(right);
            let mut left = Alu::new(OP_SHIFT_LEFT);
            left.a = (dst, to + i);
            left.dst = (dst, to + i);
            self.steps.push(left);
        }
        self.clear_shifts();
    }

    /// Applies a table to a vector, then normalizes the result.
    pub fn function(&mut self, function: u32, src: u32, dst: u32, len: u32) {
        self.align_for_table(src, len);
        for i in 0..len {
            let mut s = Alu::new(OP_FUNCTION);
            s.a = (self.scratch, 20 + i);
            s.function = function;
            s.dst = (dst, i);
            self.steps.push(s);
        }
        let exponent = if function == FN_SOFTPLUS { SOFTPLUS_OUTPUT_EXPONENT } else { UNIT_OUTPUT_EXPONENT };
        self.immediate(exponent, self.exponent(dst));
        self.normalize(dst, len);
    }

    /// `dst = 1 / src[0]`. The source must be normalized and positive.
    pub fn reciprocal(&mut self, src: u32, dst: u32) {
        let mut s = Alu::new(OP_FUNCTION);
        s.a = (src, 0);
        s.function = FN_RECIPROCAL;
        s.dst = (dst, 0);
        self.steps.push(s);
        self.immediate(RECIPROCAL_OUTPUT_EXPONENT, (self.scratch, 1));
        self.binary(OP_SUB, (self.scratch, 1), self.exponent(src), self.exponent(dst));
        self.normalize(dst, 1);
    }

    /// `dst = 1 / sqrt(src[0])`, making the exponent even first.
    pub fn inverse_sqrt(&mut self, src: u32, dst: u32) {
        let e = self.exponent(src);
        self.clear_shifts();
        let mut parity = Alu::new(OP_PARITY);
        parity.a = e;
        parity.dst = (self.scratch, 1);
        self.steps.push(parity);
        self.load_shifts(Some((self.scratch, 1)), None, None);
        self.binary(OP_MOV, (src, 0), (self.scratch, 15), (self.scratch, 2));
        self.clear_shifts();
        self.binary(OP_ADD, e, (self.scratch, 1), (self.scratch, 3));
        let mut s = Alu::new(OP_FUNCTION);
        s.a = (self.scratch, 2);
        s.function = FN_INV_SQRT;
        s.dst = (dst, 0);
        self.steps.push(s);
        // Exponent: table output minus half of the (now even) source exponent.
        self.immediate(1, (self.scratch, 4));
        self.load_shifts(Some((self.scratch, 4)), None, None);
        self.binary(OP_MOV, (self.scratch, 3), (self.scratch, 15), (self.scratch, 5));
        self.clear_shifts();
        self.immediate(RECIPROCAL_OUTPUT_EXPONENT, (self.scratch, 6));
        self.binary(OP_SUB, (self.scratch, 6), (self.scratch, 5), self.exponent(dst));
        self.normalize(dst, 1);
    }

    /// `dst = x · rsqrt(mean(x²))`. `n_log2` is log2 of the vector's length.
    pub fn rms_normalize(&mut self, x: u32, dst: u32, len: u32, temporary: (u32, u32)) {
        let n_log2 = 31 - len.leading_zeros() as i64;
        self.dot(x, x, temporary.0, len);
        self.immediate(n_log2, (self.scratch, 1));
        self.binary(OP_SUB, self.exponent(temporary.0), (self.scratch, 1), self.exponent(temporary.0));
        self.inverse_sqrt(temporary.0, temporary.1);
        self.scale(temporary.1, x, dst, len);
    }

    /// `dst = x / |x|`.
    pub fn l2_normalize(&mut self, x: u32, dst: u32, len: u32, temporary: (u32, u32)) {
        self.dot(x, x, temporary.0, len);
        self.inverse_sqrt(temporary.0, temporary.1);
        self.scale(temporary.1, x, dst, len);
    }

    #[allow(clippy::too_many_arguments)]
    /// Softmax attention: `query` against `tokens` keys, then the weighted sum of values.
    pub fn attention(&mut self, query: u32, keys: u32, values: u32, ones: u32, dst: u32, width: u32, tokens: u32, score_scale: Option<u32>, regs: AttentionRegs) {
        let guard = 32 - (width.max(1) - 1).leading_zeros();
        let sets: Vec<Vec<(Place, Place)>> = (0..tokens)
            .map(|t| (0..width).map(|j| ((query, j), (keys, t * width + j))).collect())
            .collect();
        self.dots(&sets, regs.scores, query, keys, guard);
        if let Some(scale) = score_scale {
            self.scale(scale, regs.scores, regs.top, tokens);
            self.copy_part(regs.top, 0, regs.scores, 0, tokens);
        }

        // The largest score, on the common exponent the vector already has.
        self.clear_shifts();
        self.binary(OP_MOV, (regs.scores, 0), (self.scratch, 15), (regs.top, 0));
        for t in 1..tokens {
            self.binary(OP_MAX, (regs.top, 0), (regs.scores, t), (regs.top, 0));
        }
        self.binary(OP_MOV, self.exponent(regs.scores), (self.scratch, 15), self.exponent(regs.top));

        self.sub_scalar(regs.scores, regs.top, regs.shifted, tokens);
        self.function(FN_EXP, regs.shifted, regs.weights, tokens);

        let guard_tokens = 32 - (tokens.max(1) - 1).leading_zeros();
        let sum_set = vec![(0..tokens).map(|t| ((regs.weights, t), (ones, t))).collect::<Vec<_>>()];
        self.dots(&sum_set, regs.total, regs.weights, ones, guard_tokens);
        self.reciprocal(regs.total, regs.inverse);
        self.scale(regs.inverse, regs.weights, regs.probabilities, tokens);

        let outputs: Vec<Vec<(Place, Place)>> = (0..width)
            .map(|j| (0..tokens).map(|t| ((regs.probabilities, t), (values, t * width + j))).collect())
            .collect();
        self.dots(&outputs, dst, regs.probabilities, values, guard_tokens);
    }

    /// One Gated DeltaNet head: the delta rule on a `key × value` state matrix.
    pub fn delta_step(&mut self, r: DeltaRegs, key: u32, value: u32) {
        let cells = key * value;
        self.scale(r.decay, r.state, r.state, cells);

        // memory[j] = Σ_i S[i, j] · k[i]
        let guard = 32 - (key.max(1) - 1).leading_zeros();
        let memory: Vec<Vec<(Place, Place)>> = (0..value)
            .map(|j| (0..key).map(|i| ((r.state, i * value + j), (r.key, i))).collect())
            .collect();
        self.dots(&memory, r.memory, r.state, r.key, guard);

        self.add_sub(OP_SUB, (r.value, 0), (r.memory, 0), 1, r.error, value);
        self.scale(r.beta, r.error, r.delta, value);

        // update[i, j] = k[i] · delta[j]
        let bits = self.bits();
        for i in 0..key {
            for j in 0..value {
                let mut s = Alu::new(OP_MUL);
                s.a = (r.key, i);
                s.b = (r.delta, j);
                s.dst = (r.update, i * value + j);
                s.shift = bits - 1;
                self.steps.push(s);
            }
        }
        self.product_exponent(r.key, r.delta, r.update, 0);
        self.normalize(r.update, cells);
        self.add(r.state, r.update, r.next, cells);
        for i in 0..cells {
            self.clear_copy((r.next, i), (r.state, i));
        }
        self.binary(OP_MOV, self.exponent(r.next), (self.scratch, 15), self.exponent(r.state));

        let out: Vec<Vec<(Place, Place)>> = (0..value)
            .map(|j| (0..key).map(|i| ((r.state, i * value + j), (r.query, i))).collect())
            .collect();
        self.dots(&out, r.out, r.state, r.query, guard);
    }

    /// Copies part of a register, exponent included. The shifters must be clear.
    pub fn copy_part(&mut self, src: u32, from: u32, dst: u32, to: u32, len: u32) {
        self.clear_shifts();
        let zero = (self.scratch, 15);
        for i in 0..len {
            self.binary(OP_MOV, (src, from + i), zero, (dst, to + i));
        }
        self.binary(OP_MOV, self.exponent(src), zero, self.exponent(dst));
    }

    /// `dst` becomes the concatenation of `parts`, on their largest exponent.
    pub fn gather(&mut self, parts: &[(u32, u32)], dst: u32) {
        self.clear_shifts();
        let zero = (self.scratch, 15);
        let (largest, shift) = ((self.scratch, 1), (self.scratch, 2));
        self.binary(OP_MOV, self.exponent(parts[0].0), zero, largest);
        for (reg, _) in &parts[1..] {
            self.binary(OP_MAX, largest, self.exponent(*reg), largest);
        }
        let mut offset = 0;
        for (reg, len) in parts {
            self.binary(OP_SUB, largest, self.exponent(*reg), shift);
            self.load_shifts(Some(shift), None, None);
            for i in 0..*len {
                self.binary(OP_MOV, (*reg, i), zero, (dst, offset + i));
            }
            self.clear_shifts();
            offset += len;
        }
        self.binary(OP_MOV, largest, zero, self.exponent(dst));
        self.normalize(dst, offset);
    }

    fn clear_copy(&mut self, from: Place, to: Place) {
        self.binary(OP_MOV, from, (self.scratch, 15), to);
    }
}

#[derive(Clone, Copy, Debug)]
pub struct AttentionRegs {
    pub scores: u32,
    pub top: u32,
    pub shifted: u32,
    pub weights: u32,
    pub total: u32,
    pub inverse: u32,
    pub probabilities: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct DeltaRegs {
    pub state: u32,
    pub key: u32,
    pub query: u32,
    pub value: u32,
    pub beta: u32,
    pub decay: u32,
    pub memory: u32,
    pub error: u32,
    pub delta: u32,
    pub update: u32,
    pub next: u32,
    pub out: u32,
}

/// Executes the steps directly on integers, with no reference to the circuit.
pub struct Behavior {
    exponents: u32,
    bits: u32,
    index_bits: u32,
    registers: HashMap<Place, i64>,
    table: Vec<i64>,
    function_bits: u32,
    shift_a: u32,
    shift_b: u32,
    shift_left: u32,
    magnitude: u64,
    accumulator: i128,
}

fn wrap(value: i128, bits: u32) -> i64 {
    let shift = 128 - bits;
    ((value << shift) >> shift) as i64
}

impl Behavior {
    pub fn new(c: &DieConfig, initial: &HashMap<Place, i64>) -> Self {
        Self {
            exponents: exponent_register(c),
            bits: c.cell.activation_bits,
            index_bits: c.index_bits,
            registers: initial.clone(),
            table: tables(c),
            function_bits: c.function_bits,
            shift_a: 0,
            shift_b: 0,
            shift_left: 0,
            magnitude: 0,
            accumulator: 0,
        }
    }

    fn get(&self, place: Place) -> i64 {
        self.registers.get(&place).copied().unwrap_or(0)
    }

    pub fn value(&self, reg: u32, index: u32) -> f64 {
        let exponent = self.get((self.exponents, reg));
        self.get((reg, index)) as f64 / 2f64.powi(self.bits as i32 - 1) * 2f64.powi(exponent as i32)
    }

    pub fn mantissa(&self, place: Place) -> i64 {
        self.get(place)
    }

    pub fn run(&mut self, steps: &[Alu]) {
        for s in steps {
            self.step(s);
        }
    }

    fn step(&mut self, s: &Alu) {
        let bits = self.bits;
        if s.magnitude_clear {
            self.magnitude = 0;
        }
        if s.accumulator_clear {
            self.accumulator = 0;
        }
        let a = self.get(s.a);
        let b = self.get(s.b);
        let aligned_a = a >> self.shift_a.min(63);
        let aligned_b = b >> self.shift_b.min(63);

        let result = match s.op {
            OP_ADD => wrap(aligned_a as i128 + aligned_b as i128, bits),
            OP_SUB => wrap(aligned_a as i128 - aligned_b as i128, bits),
            OP_MUL => wrap((a as i128 * b as i128) >> s.shift, bits),
            OP_MAX => aligned_a.max(aligned_b),
            OP_MOV => aligned_a,
            OP_IMMEDIATE => {
                let literal = s.literal.unwrap_or(s.b.1) as i64;
                let shift = 64 - self.index_bits;
                (literal << shift) >> shift
            }
            OP_PARITY => a & 1,
            OP_LZC => {
                let width = bits - 1;
                let value = self.magnitude & ((1 << width) - 1);
                let used = 64 - value.leading_zeros();
                (width - used) as i64
            }
            OP_SHIFT_LEFT => {
                let wide = wrap((a as i128) << self.shift_left, 2 * bits);
                let top = (1i64 << (bits - 1)) - 1;
                if wide > top {
                    top
                } else if wide < -top - 1 {
                    -top - 1
                } else {
                    wide
                }
            }
            OP_ACCUMULATOR => wrap(self.accumulator >> s.shift, bits),
            OP_FUNCTION => {
                let index = ((a as u64 & ((1 << bits) - 1)) >> (bits - self.function_bits)) as usize;
                self.table[index | (s.function as usize) << self.function_bits]
            }
            other => panic!("operación sin modelo: {other}"),
        };

        if s.magnitude_scan {
            let sign = (a >> (bits - 1)) & 1;
            let mask = (1u64 << (bits - 1)) - 1;
            let differs = if sign == 1 { !(a as u64) & mask } else { a as u64 & mask };
            self.magnitude |= differs;
        }
        if s.accumulate {
            self.accumulator = (self.accumulator + a as i128 * b as i128) << 40 >> 40;
        }
        if s.write {
            self.registers.insert(s.dst, result);
        }
        if s.load_shift_a {
            self.shift_a = (result & 63) as u32;
        }
        if s.load_shift_b {
            self.shift_b = (result & 63) as u32;
        }
        if s.load_shift_left {
            self.shift_left = (result & 63) as u32;
        }
    }
}
