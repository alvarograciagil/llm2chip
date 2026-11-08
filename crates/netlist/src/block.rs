//! Block floating point on the die: each vector register has one exponent.
//!
//! `BlockOps` describes the arithmetic once. `Assembly` turns it into microcode for
//! the gates, and a virtual machine elsewhere runs the same semantics at full speed.

use crate::die::{OP_ADD, OP_EQUAL, OP_FUNCTION, OP_IMMEDIATE, OP_LZC, OP_MAX, OP_MOV, OP_MUL, OP_PARITY, OP_SHIFT_LEFT, OP_SUB, OP_WIDE_LZC, OP_WIDE_OUT};
use crate::DieConfig;
use std::collections::HashMap;

pub type Place = (u32, u32);
pub type Pair = (Place, Place);

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

pub const TABLE_INPUT_EXPONENT: i64 = 3;
pub const SOFTPLUS_OUTPUT_EXPONENT: i64 = 4;
pub const UNIT_OUTPUT_EXPONENT: i64 = 0;
pub const RECIPROCAL_OUTPUT_EXPONENT: i64 = 1;
/// The accumulator's width; products are `2·16` bits, leaving 16 guard bits.
pub const ACCUMULATOR_BITS: u32 = 48;
/// Output mantissas keep this many magnitude bits.
const MANTISSA_MAGNITUDE_BITS: i64 = 15;

/// Rounds to nearest while narrowing; the largest positive value does not wrap.
pub fn round_right(value: i128, shift: u32, bits: u32) -> i64 {
    let truncated = wrap(value >> shift, bits);
    if shift == 0 || truncated == (1i64 << (bits - 1)) - 1 {
        return truncated;
    }
    wrap(truncated as i128 + ((value >> (shift - 1)) & 1), bits)
}

fn table_value(function: u32, index: usize, index_bits: u32, bits: u32) -> i64 {
    let top = (1i64 << (bits - 1)) - 1;
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
    // The most negative input stands for everything beyond the table's range, where
    // the sigmoid and the exponential are effectively zero.
    let lowest = 1usize << (c.function_bits - 1);
    (0..1usize << (c.function_bits + c.function_select_bits))
        .map(|address| {
            let (function, index) = ((address >> c.function_bits) as u32, address & ((1 << c.function_bits) - 1));
            if index == lowest && matches!(function, FN_SIGMOID | FN_EXP) {
                0
            } else {
                table_value(function, index, c.function_bits, bits)
            }
        })
        .collect()
}

/// A real-valued vector as mantissas sharing one exponent.
pub fn encode_vector(values: &[f64], bits: u32) -> (Vec<i64>, i64) {
    let largest = values.iter().fold(0f64, |m, v| m.max(v.abs()));
    let exponent = if largest > 0.0 { largest.log2().floor() as i64 + 1 } else { 0 };
    let scale = 2f64.powi(bits as i32 - 1 - exponent as i32);
    let top = (1i64 << (bits - 1)) - 1;
    (values.iter().map(|v| ((v * scale).round() as i64).clamp(-top, top)).collect(), exponent)
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

/// The vector arithmetic of the die. Implementations must agree bit for bit.
pub trait BlockOps {
    fn load_vector(&mut self, reg: u32, values: &[f64]);
    fn load_mantissas(&mut self, reg: u32, mantissas: &[i64], exponent: i64);
    fn normalize(&mut self, reg: u32, len: u32);
    fn add_sub(&mut self, subtract: bool, a: Place, b: Place, b_stride: u32, dst: u32, len: u32);
    fn multiply(&mut self, a: u32, b: u32, dst: u32, len: u32);
    fn scale(&mut self, s_reg: u32, a: u32, dst: u32, len: u32);
    /// `dst[set] = Σ a[·] · b[·]` for every set of index pairs.
    fn dots<F>(&mut self, a: u32, b: u32, dst: u32, sets: u32, terms: u32, indices: F)
    where
        F: Fn(u32, u32) -> Option<(Place, Place)>;
    fn function(&mut self, function: u32, src: u32, dst: u32, len: u32);
    fn reciprocal(&mut self, src: u32, dst: u32);
    fn inverse_sqrt(&mut self, src: u32, dst: u32);
    fn copy_part(&mut self, src: u32, from: u32, dst: u32, to: u32, len: u32);
    fn gather(&mut self, parts: &[(u32, u32)], dst: u32);
    fn align_to(&mut self, src: u32, from: u32, exponent: i64, dst: u32, to: u32, len: u32);
    fn move_element(&mut self, from: Place, to: Place);
    /// `dst[0]` becomes the largest of `src[..len]`, sharing `src`'s exponent.
    fn maximum(&mut self, src: u32, len: u32, dst: u32);
    /// `dst[i]` becomes 1 when `a[i] == b[0]`, otherwise 0.
    fn mark_equal(&mut self, a: u32, b: u32, dst: u32, len: u32);
    fn set_exponent(&mut self, reg: u32, exponent: i64);
    fn subtract_exponent(&mut self, reg: u32, amount: i64);
    /// `rows` rows of the matrix from `first`. It is held in register `matrix`, but
    /// `name` lets an implementation stream it from the checkpoint instead.
    #[allow(clippy::too_many_arguments)]
    fn matvec(&mut self, name: &str, matrix: u32, first: u32, src: u32, dst: u32, rows: u32, cols: u32);
    /// Copies one row of an embedding-like matrix into `dst`.
    fn copy_row(&mut self, name: &str, matrix: u32, row: u32, cols: u32, dst: u32);
    /// True only when an implementation can prove `reg[index]` is zero and may skip work.
    fn known_zero(&self, _reg: u32, _index: u32) -> bool {
        false
    }

    fn add(&mut self, a: u32, b: u32, dst: u32, len: u32) {
        self.add_sub(false, (a, 0), (b, 0), 1, dst, len);
    }

    fn sub_scalar(&mut self, a: u32, b: u32, dst: u32, len: u32) {
        self.add_sub(true, (a, 0), (b, 0), 0, dst, len);
    }

    fn dot(&mut self, a: u32, b: u32, dst: u32, len: u32) {
        self.dots(a, b, dst, 1, len, |_, i| Some(((a, i), (b, i))));
    }

    /// `inverse_len` holds `1 / len` when `len` is not a power of two; otherwise a shift does it.
    /// `epsilon`, when given, keeps an almost silent vector from being blown up.
    fn rms_normalize(&mut self, x: u32, dst: u32, len: u32, temporary: (u32, u32), inverse_len: Option<u32>, epsilon: Option<u32>) {
        self.dot(x, x, temporary.0, len);
        let (mean, spare) = match inverse_len {
            Some(factor) => {
                self.scale(factor, temporary.0, temporary.1, 1);
                (temporary.1, temporary.0)
            }
            None => {
                self.subtract_exponent(temporary.0, 31 - len.leading_zeros() as i64);
                temporary
            }
        };
        self.invert_root_and_scale(x, dst, len, (mean, spare), epsilon);
    }

    fn l2_normalize(&mut self, x: u32, dst: u32, len: u32, temporary: (u32, u32), epsilon: Option<u32>) {
        self.dot(x, x, temporary.0, len);
        self.invert_root_and_scale(x, dst, len, temporary, epsilon);
    }

    /// `dst = x / sqrt(value + epsilon)`, where `registers.0` holds `value`.
    fn invert_root_and_scale(&mut self, x: u32, dst: u32, len: u32, registers: (u32, u32), epsilon: Option<u32>) {
        let (value, spare) = match epsilon {
            Some(e) => {
                self.add(registers.0, e, registers.1, 1);
                (registers.1, registers.0)
            }
            None => registers,
        };
        self.inverse_sqrt(value, spare);
        self.scale(spare, x, dst, len);
    }

    /// Softmax attention: `query` against `tokens` keys, then the weighted sum of values.
    #[allow(clippy::too_many_arguments)]
    fn attention(&mut self, query: u32, keys: u32, values: u32, ones: u32, dst: u32, width: u32, tokens: u32, score_scale: Option<u32>, regs: AttentionRegs) {
        self.dots(query, keys, regs.scores, tokens, width, |t, j| Some(((query, j), (keys, t * width + j))));
        if let Some(scale) = score_scale {
            self.scale(scale, regs.scores, regs.top, tokens);
            self.copy_part(regs.top, 0, regs.scores, 0, tokens);
        }
        self.maximum(regs.scores, tokens, regs.top);
        self.sub_scalar(regs.scores, regs.top, regs.shifted, tokens);
        self.function(FN_EXP, regs.shifted, regs.weights, tokens);
        self.dots(regs.weights, ones, regs.total, 1, tokens, |_, t| Some(((regs.weights, t), (ones, t))));
        self.reciprocal(regs.total, regs.inverse);
        self.scale(regs.inverse, regs.weights, regs.probabilities, tokens);
        self.dots(regs.probabilities, values, dst, width, tokens, |j, t| Some(((regs.probabilities, t), (values, t * width + j))));
    }

    /// One Gated DeltaNet head: the delta rule on a `key × value` state matrix.
    fn delta_step(&mut self, r: DeltaRegs, key: u32, value: u32) {
        let cells = key * value;
        self.scale(r.decay, r.state, r.state, cells);
        self.dots(r.state, r.key, r.memory, value, key, |j, i| Some(((r.state, i * value + j), (r.key, i))));
        self.add_sub(true, (r.value, 0), (r.memory, 0), 1, r.error, value);
        self.scale(r.beta, r.error, r.delta, value);
        // The output reads the old state and the update apart, before rounding merges them:
        // the update's contribution is often a tiny difference that rounding would erase.
        self.dots(r.state, r.query, r.next, value, key, |j, i| Some(((r.state, i * value + j), (r.query, i))));
        self.dot(r.key, r.query, r.memory, key);
        self.scale(r.memory, r.delta, r.update, value);
        self.add(r.next, r.update, r.out, value);
        self.outer_product(r.key, r.delta, r.update, key, value);
        self.add(r.state, r.update, r.next, cells);
        self.copy_part(r.next, 0, r.state, 0, cells);
    }

    fn outer_product(&mut self, a: u32, b: u32, dst: u32, rows: u32, cols: u32);
}

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
    pub load_wide_shift: bool,
    pub magnitude_scan: bool,
    pub magnitude_clear: bool,
    pub accumulate: bool,
    pub accumulator_clear: bool,
    pub link_send: bool,
    pub wide_write: bool,
    pub wide_scan: bool,
    pub wide_clear: bool,
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
            load_wide_shift: false,
            magnitude_scan: false,
            magnitude_clear: false,
            accumulate: false,
            accumulator_clear: false,
            link_send: false,
            wide_write: false,
            wide_scan: false,
            wide_clear: false,
        }
    }
}

/// A program of single-element steps plus the register contents it starts from.
pub struct Assembly {
    pub steps: Vec<Alu>,
    pub initial: HashMap<Place, i64>,
    bits: u32,
    index_bits: u32,
    exponents: u32,
    scratch: u32,
}

impl Assembly {
    pub fn new(c: &DieConfig) -> Self {
        Self { steps: Vec::new(), initial: HashMap::new(), bits: c.cell.activation_bits, index_bits: c.index_bits, exponents: exponent_register(c), scratch: scratch_register(c) }
    }

    pub fn exponent(&self, reg: u32) -> Place {
        (self.exponents, reg)
    }

    /// Scratch element for values that have to be built up across several steps.
    fn sc(&self, index: u32) -> Place {
        (self.scratch, index)
    }

    /// Writes a small constant, composing it when it does not fit one literal.
    fn immediate(&mut self, value: i64, dst: Place) {
        let limit = (1i64 << (self.index_bits - 1)) - 1;
        if value.abs() <= limit {
            let mut s = Alu::new(OP_IMMEDIATE);
            s.literal = Some(value as u32 & ((1 << self.index_bits) - 1));
            s.dst = dst;
            self.steps.push(s);
        } else {
            let part = value / 2;
            self.immediate(part, self.sc(16));
            self.immediate(value - part, self.sc(17));
            self.binary(OP_ADD, self.sc(16), self.sc(17), dst);
        }
    }

    fn binary(&mut self, op: u32, a: Place, b: Place, dst: Place) {
        let mut s = Alu::new(op);
        s.a = a;
        s.b = b;
        s.dst = dst;
        self.steps.push(s);
    }

    fn load_shifts(&mut self, a: Option<Place>, b: Option<Place>, left: Option<Place>) {
        // Sources are read through the A shifter, so A has to be loaded last.
        let zero = self.sc(15);
        for (place, which) in [(left, 2), (b, 1), (a, 0)] {
            let mut s = Alu::new(OP_MOV);
            s.a = place.unwrap_or(zero);
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
        self.immediate(0, self.sc(15));
        self.load_shifts(None, None, None);
    }

    fn product_exponent(&mut self, a: u32, b: u32, dst: u32, extra: i64) {
        self.binary(OP_ADD, self.exponent(a), self.exponent(b), self.sc(6));
        self.immediate(extra, self.sc(7));
        self.binary(OP_ADD, self.sc(6), self.sc(7), self.exponent(dst));
    }

    fn accumulate_set(&mut self, pairs: impl Iterator<Item = Pair>, wide_index: u32) {
        let mut clear = Alu::new(OP_MOV);
        clear.write = false;
        clear.accumulator_clear = true;
        self.steps.push(clear);
        for (a, b) in pairs {
            let mut s = Alu::new(OP_MOV);
            s.a = a;
            s.b = b;
            s.write = false;
            s.accumulate = true;
            self.steps.push(s);
        }
        let mut write = Alu::new(OP_MOV);
        write.write = false;
        write.wide_write = true;
        write.dst = (0, wide_index);
        self.steps.push(write);
    }

    /// Narrows the wide sums to mantissas, shifting only as much as the largest needs.
    fn narrow_wide(&mut self, a: u32, b: u32, dst: u32, sets: u32) {
        let mut clear = Alu::new(OP_MOV);
        clear.write = false;
        clear.wide_clear = true;
        self.steps.push(clear);
        for set in 0..sets {
            let mut s = Alu::new(OP_MOV);
            s.a = (0, set);
            s.write = false;
            s.wide_scan = true;
            self.steps.push(s);
        }
        let mut count = Alu::new(OP_WIDE_LZC);
        count.dst = self.sc(0);
        self.steps.push(count);
        self.immediate(ACCUMULATOR_BITS as i64 - 1 - MANTISSA_MAGNITUDE_BITS, self.sc(1));
        self.binary(OP_SUB, self.sc(1), self.sc(0), self.sc(2));
        self.immediate(0, self.sc(3));
        self.binary(OP_MAX, self.sc(2), self.sc(3), self.sc(4));
        let mut load = Alu::new(OP_MOV);
        load.a = self.sc(4);
        load.write = false;
        load.load_wide_shift = true;
        self.steps.push(load);
        for set in 0..sets {
            let mut s = Alu::new(OP_WIDE_OUT);
            s.a = (0, set);
            s.dst = (dst, set);
            self.steps.push(s);
        }
        self.binary(OP_ADD, self.exponent(a), self.exponent(b), self.sc(5));
        self.binary(OP_ADD, self.sc(5), self.sc(4), self.sc(6));
        self.immediate(-MANTISSA_MAGNITUDE_BITS, self.sc(7));
        self.binary(OP_ADD, self.sc(6), self.sc(7), self.exponent(dst));
        self.normalize(dst, sets);
    }

    fn table_input(&mut self, function: u32, source: Place, dst: Place) {
        let mut s = Alu::new(OP_FUNCTION);
        s.a = source;
        s.function = function;
        s.dst = dst;
        self.steps.push(s);
    }
}

impl BlockOps for Assembly {
    fn load_vector(&mut self, reg: u32, values: &[f64]) {
        let (mantissas, exponent) = encode_vector(values, self.bits);
        for (i, m) in mantissas.iter().enumerate() {
            self.initial.insert((reg, i as u32), *m);
        }
        let place = self.exponent(reg);
        self.initial.insert(place, exponent);
    }

    fn load_mantissas(&mut self, reg: u32, mantissas: &[i64], exponent: i64) {
        for (i, m) in mantissas.iter().enumerate() {
            self.initial.insert((reg, i as u32), *m);
        }
        let place = self.exponent(reg);
        self.initial.insert(place, exponent);
    }

    fn normalize(&mut self, reg: u32, len: u32) {
        self.clear_shifts();
        let mut first = Alu::new(OP_MOV);
        first.a = (reg, 0);
        first.write = false;
        first.magnitude_clear = true;
        self.steps.push(first);
        for i in 0..len {
            let mut s = Alu::new(OP_MOV);
            s.a = (reg, i);
            s.magnitude_scan = true;
            s.write = false;
            self.steps.push(s);
        }
        let mut count = Alu::new(OP_LZC);
        count.dst = self.sc(0);
        self.steps.push(count);
        self.load_shifts(None, None, Some(self.sc(0)));
        for i in 0..len {
            let mut s = Alu::new(OP_SHIFT_LEFT);
            s.a = (reg, i);
            s.dst = (reg, i);
            self.steps.push(s);
        }
        self.binary(OP_SUB, self.exponent(reg), self.sc(0), self.exponent(reg));
        self.clear_shifts();
    }

    fn add_sub(&mut self, subtract: bool, a: Place, b: Place, b_stride: u32, dst: u32, len: u32) {
        self.clear_shifts();
        let (ea, eb) = (self.exponent(a.0), self.exponent(b.0));
        self.binary(OP_MAX, ea, eb, self.sc(1));
        self.immediate(1, self.sc(2));
        self.binary(OP_ADD, self.sc(1), self.sc(2), self.sc(3));
        self.binary(OP_SUB, self.sc(3), ea, self.sc(4));
        self.binary(OP_SUB, self.sc(3), eb, self.sc(5));
        self.load_shifts(Some(self.sc(4)), Some(self.sc(5)), None);
        let op = if subtract { OP_SUB } else { OP_ADD };
        for i in 0..len {
            self.binary(op, (a.0, a.1 + i), (b.0, b.1 + i * b_stride), (dst, i));
        }
        self.clear_shifts();
        self.binary(OP_MOV, self.sc(3), self.sc(15), self.exponent(dst));
        self.normalize(dst, len);
    }

    fn multiply(&mut self, a: u32, b: u32, dst: u32, len: u32) {
        for i in 0..len {
            let mut s = Alu::new(OP_MUL);
            s.a = (a, i);
            s.b = (b, i);
            s.dst = (dst, i);
            s.shift = self.bits - 1;
            self.steps.push(s);
        }
        self.product_exponent(a, b, dst, 0);
        self.normalize(dst, len);
    }

    fn scale(&mut self, s_reg: u32, a: u32, dst: u32, len: u32) {
        for i in 0..len {
            let mut s = Alu::new(OP_MUL);
            s.a = (s_reg, 0);
            s.b = (a, i);
            s.dst = (dst, i);
            s.shift = self.bits - 1;
            self.steps.push(s);
        }
        self.product_exponent(s_reg, a, dst, 0);
        self.normalize(dst, len);
    }

    fn outer_product(&mut self, a: u32, b: u32, dst: u32, rows: u32, cols: u32) {
        for i in 0..rows {
            for j in 0..cols {
                let mut s = Alu::new(OP_MUL);
                s.a = (a, i);
                s.b = (b, j);
                s.dst = (dst, i * cols + j);
                s.shift = self.bits - 1;
                self.steps.push(s);
            }
        }
        self.product_exponent(a, b, dst, 0);
        self.normalize(dst, rows * cols);
    }

    fn dots<F>(&mut self, a: u32, b: u32, dst: u32, sets: u32, terms: u32, indices: F)
    where
        F: Fn(u32, u32) -> Option<(Place, Place)>,
    {
        for set in 0..sets {
            let pairs: Vec<Pair> = (0..terms).filter_map(|term| indices(set, term)).collect();
            self.accumulate_set(pairs.into_iter(), set);
        }
        self.narrow_wide(a, b, dst, sets);
    }

    fn function(&mut self, function: u32, src: u32, dst: u32, len: u32) {
        let scratch = self.scratch;
        self.align_to(src, 0, TABLE_INPUT_EXPONENT, scratch, 32, len);
        for i in 0..len {
            self.table_input(function, (scratch, 32 + i), (dst, i));
        }
        let exponent = if function == FN_SOFTPLUS { SOFTPLUS_OUTPUT_EXPONENT } else { UNIT_OUTPUT_EXPONENT };
        self.immediate(exponent, self.exponent(dst));
        self.normalize(dst, len);
    }

    fn reciprocal(&mut self, src: u32, dst: u32) {
        self.table_input(FN_RECIPROCAL, (src, 0), (dst, 0));
        self.immediate(RECIPROCAL_OUTPUT_EXPONENT, self.sc(1));
        self.binary(OP_SUB, self.sc(1), self.exponent(src), self.exponent(dst));
        self.normalize(dst, 1);
    }

    fn inverse_sqrt(&mut self, src: u32, dst: u32) {
        let e = self.exponent(src);
        self.clear_shifts();
        let mut parity = Alu::new(OP_PARITY);
        parity.a = e;
        parity.dst = self.sc(1);
        self.steps.push(parity);
        self.load_shifts(Some(self.sc(1)), None, None);
        self.binary(OP_MOV, (src, 0), self.sc(15), self.sc(2));
        self.clear_shifts();
        self.binary(OP_ADD, e, self.sc(1), self.sc(3));
        self.table_input(FN_INV_SQRT, self.sc(2), (dst, 0));
        self.immediate(1, self.sc(4));
        self.load_shifts(Some(self.sc(4)), None, None);
        self.binary(OP_MOV, self.sc(3), self.sc(15), self.sc(5));
        self.clear_shifts();
        self.immediate(RECIPROCAL_OUTPUT_EXPONENT, self.sc(6));
        self.binary(OP_SUB, self.sc(6), self.sc(5), self.exponent(dst));
        self.normalize(dst, 1);
    }

    fn copy_part(&mut self, src: u32, from: u32, dst: u32, to: u32, len: u32) {
        self.clear_shifts();
        let zero = self.sc(15);
        for i in 0..len {
            self.binary(OP_MOV, (src, from + i), zero, (dst, to + i));
        }
        self.binary(OP_MOV, self.exponent(src), zero, self.exponent(dst));
    }

    fn gather(&mut self, parts: &[(u32, u32)], dst: u32) {
        self.clear_shifts();
        let zero = self.sc(15);
        let (largest, shift) = (self.sc(1), self.sc(2));
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

    fn align_to(&mut self, src: u32, from: u32, exponent: i64, dst: u32, to: u32, len: u32) {
        let e = self.exponent(src);
        self.immediate(exponent, self.sc(8));
        self.binary(OP_SUB, e, self.sc(8), self.sc(9));
        self.immediate(0, self.sc(10));
        self.binary(OP_MAX, self.sc(9), self.sc(10), self.sc(11));
        self.binary(OP_SUB, self.sc(8), e, self.sc(12));
        self.binary(OP_MAX, self.sc(12), self.sc(10), self.sc(13));
        self.load_shifts(Some(self.sc(13)), None, Some(self.sc(11)));
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

    fn move_element(&mut self, from: Place, to: Place) {
        self.binary(OP_MOV, from, self.sc(15), to);
    }

    fn maximum(&mut self, src: u32, len: u32, dst: u32) {
        self.clear_shifts();
        self.binary(OP_MOV, (src, 0), self.sc(15), (dst, 0));
        for t in 1..len {
            self.binary(OP_MAX, (dst, 0), (src, t), (dst, 0));
        }
        self.binary(OP_MOV, self.exponent(src), self.sc(15), self.exponent(dst));
    }

    fn mark_equal(&mut self, a: u32, b: u32, dst: u32, len: u32) {
        self.clear_shifts();
        for i in 0..len {
            self.binary(OP_EQUAL, (a, i), (b, 0), (dst, i));
        }
        self.immediate(0, self.exponent(dst));
    }

    fn set_exponent(&mut self, reg: u32, exponent: i64) {
        self.immediate(exponent, self.exponent(reg));
    }

    fn subtract_exponent(&mut self, reg: u32, amount: i64) {
        self.immediate(amount, self.sc(1));
        self.binary(OP_SUB, self.exponent(reg), self.sc(1), self.exponent(reg));
    }

    fn matvec(&mut self, _name: &str, matrix: u32, first: u32, src: u32, dst: u32, rows: u32, cols: u32) {
        self.dots(matrix, src, dst, rows, cols, |r, j| Some(((matrix, (first + r) * cols + j), (src, j))));
    }

    fn copy_row(&mut self, _name: &str, matrix: u32, row: u32, cols: u32, dst: u32) {
        self.copy_part(matrix, row * cols, dst, 0, cols);
    }
}

/// Executes the steps directly on integers, with no reference to the circuit.
pub struct Behavior {
    exponents: u32,
    bits: u32,
    index_bits: u32,
    registers: HashMap<Place, i64>,
    wide: HashMap<u32, i64>,
    table: Vec<i64>,
    function_bits: u32,
    shift_a: u32,
    shift_b: u32,
    shift_left: u32,
    wide_shift: u32,
    magnitude: u64,
    wide_magnitude: u64,
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
            wide: HashMap::new(),
            table: tables(c),
            function_bits: c.function_bits,
            shift_a: 0,
            shift_b: 0,
            shift_left: 0,
            wide_shift: 0,
            magnitude: 0,
            wide_magnitude: 0,
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
        if s.wide_clear {
            self.wide_magnitude = 0;
        }
        let a = self.get(s.a);
        let b = self.get(s.b);
        let wide_value = self.wide.get(&s.a.1).copied().unwrap_or(0);
        let aligned_a = a >> self.shift_a.min(63);
        let aligned_b = b >> self.shift_b.min(63);

        let result = match s.op {
            OP_ADD => wrap(aligned_a as i128 + aligned_b as i128, bits),
            OP_SUB => wrap(aligned_a as i128 - aligned_b as i128, bits),
            OP_MUL => round_right(a as i128 * b as i128, s.shift, bits),
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
                (width - (64 - (self.magnitude & ((1 << width) - 1)).leading_zeros())) as i64
            }
            OP_WIDE_LZC => {
                let width = ACCUMULATOR_BITS - 1;
                (width - (64 - (self.wide_magnitude & ((1 << width) - 1)).leading_zeros())) as i64
            }
            OP_WIDE_OUT => round_right(wide_value as i128, self.wide_shift.min(63), bits),
            OP_EQUAL => {
                if wrap(aligned_a as i128 - aligned_b as i128, bits) == 0 {
                    (1i64 << (bits - 1)) - 1
                } else {
                    0
                }
            }
            OP_SHIFT_LEFT => {
                let wide = wrap((a as i128) << self.shift_left, 2 * bits);
                let top = (1i64 << (bits - 1)) - 1;
                wide.clamp(-top - 1, top)
            }
            OP_FUNCTION => {
                let index = ((a as u64 & ((1 << bits) - 1)) >> (bits - self.function_bits)) as usize;
                self.table[index | (s.function as usize) << self.function_bits]
            }
            other => panic!("operación sin modelo: {other}"),
        };

        if s.magnitude_scan {
            let sign = (a >> (bits - 1)) & 1;
            let mask = (1u64 << (bits - 1)) - 1;
            self.magnitude |= if sign == 1 { !(a as u64) & mask } else { a as u64 & mask };
        }
        if s.wide_scan {
            let top = ACCUMULATOR_BITS - 1;
            let sign = (wide_value >> top) & 1;
            let mask = (1u64 << top) - 1;
            self.wide_magnitude |= if sign == 1 { !(wide_value as u64) & mask } else { wide_value as u64 & mask };
        }
        if s.accumulate {
            self.accumulator = wrap(self.accumulator + a as i128 * b as i128, ACCUMULATOR_BITS) as i128;
        }
        if s.wide_write {
            self.wide.insert(s.dst.1, self.accumulator as i64);
        }
        if s.write {
            self.registers.insert(s.dst, result);
        }
        let low = (result & 63) as u32;
        if s.load_shift_a {
            self.shift_a = low;
        }
        if s.load_shift_b {
            self.shift_b = low;
        }
        if s.load_shift_left {
            self.shift_left = low;
        }
        if s.load_wide_shift {
            self.wide_shift = low;
        }
    }
}
