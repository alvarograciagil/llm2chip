//! Block floating point on the die: each vector register has one exponent.
//!
//! `BlockOps` describes the arithmetic once. `Assembly` turns it into microcode for
//! the gates, and a virtual machine elsewhere runs the same semantics at full speed.

use crate::die::{OP_ADD, OP_EQUAL, OP_FUNCTION, OP_IMMEDIATE, OP_LINK, OP_LZC, OP_MAX, OP_MOV, OP_MUL, OP_PARITY, OP_SHIFT_LEFT, OP_SUB, OP_WIDE_LZC, OP_WIDE_OUT};
use crate::DieConfig;
use std::collections::HashMap;

pub type Place = (u32, u32);

pub fn exponent_register(c: &DieConfig) -> u32 {
    (1 << c.reg_bits) - 1
}

fn scratch_register(c: &DieConfig) -> u32 {
    (1 << c.reg_bits) - 2
}

pub const FN_GELU: u32 = 5;
pub const FN_GELU_TANH: u32 = 6;
pub const FN_TANH: u32 = 7;
pub const FN_EXP: u32 = 0;
pub const FN_SIGMOID: u32 = 1;
pub const FN_RECIPROCAL: u32 = 2;
pub const FN_INV_SQRT: u32 = 3;
pub const FN_SOFTPLUS: u32 = 4;

pub const TABLE_INPUT_EXPONENT: i64 = 3;
pub const SOFTPLUS_OUTPUT_EXPONENT: i64 = 4;
pub const UNIT_OUTPUT_EXPONENT: i64 = 0;
pub const RECIPROCAL_OUTPUT_EXPONENT: i64 = 1;
/// The accumulator's width: a full product plus 16 guard bits.
pub const fn accumulator_bits(bits: u32) -> u32 {
    2 * bits + 16
}


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
        FN_TANH => (input.tanh(), UNIT_OUTPUT_EXPONENT),
        FN_GELU => (0.5 * (1.0 + libm::erf(input / std::f64::consts::SQRT_2)), UNIT_OUTPUT_EXPONENT),
        FN_GELU_TANH => (0.5 * (1.0 + ((2.0 / std::f64::consts::PI).sqrt() * (input + 0.044715 * input.powi(3))).tanh()), UNIT_OUTPUT_EXPONENT),
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
            if index == lowest && matches!(function, FN_SIGMOID | FN_EXP | FN_GELU | FN_GELU_TANH) {
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
    /// Registers holding `1 / cap` and `cap`, when the scores are capped.
    pub softcap: Option<(u32, u32)>,
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

/// A quantity the fixed microprogram only knows while it runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Var {
    Token,
    Position,
    Loop(u32),
    /// How many of the last `n` tokens exist at this position.
    WindowLength(u32),
    /// Where this position's token goes in a ring of `n` entries.
    WindowSlot(u32),
}

const EXPR_TERMS: usize = 4;

/// `constant + Σ coefficient · variable`. Everything wraps, like the address registers.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Expr {
    constant: u32,
    terms: [Option<(Var, u32)>; EXPR_TERMS],
}

impl Expr {
    pub fn var(variable: Var) -> Self {
        let mut terms = [None; EXPR_TERMS];
        terms[0] = Some((variable, 1));
        Self { constant: 0, terms }
    }

    /// True when this index is always a fixed distance past `other`.
    pub fn is_after(&self, other: Expr) -> bool {
        self.constant > other.constant && self.variables().eq(other.variables())
    }

    /// This index without its part in `variable`.
    pub fn without(self, variable: Var) -> Self {
        self.variables().filter(|(v, _)| *v != variable).fold(Self::from(self.constant), |sum, (v, c)| sum + Self::var(v) * c)
    }

    /// True when `by` divides every value this index can take.
    pub fn is_multiple_of(&self, by: u32) -> bool {
        self.constant.is_multiple_of(by) && self.terms.iter().flatten().all(|(_, coefficient)| coefficient.is_multiple_of(by))
    }

    /// This index divided by `by`, which must divide every part of it.
    pub fn divided_by(&self, by: u32) -> Self {
        assert!(self.constant.is_multiple_of(by), "{} no es múltiplo de {by}", self.constant);
        let mut terms = self.terms;
        for (_, coefficient) in terms.iter_mut().flatten() {
            assert!(coefficient.is_multiple_of(by), "{coefficient} no es múltiplo de {by}");
            *coefficient /= by;
        }
        Self { constant: self.constant / by, terms }
    }

    pub fn constant(&self) -> u32 {
        self.constant
    }

    pub fn variables(&self) -> impl Iterator<Item = (Var, u32)> + '_ {
        self.terms.iter().flatten().copied()
    }

    pub fn is_constant(&self) -> bool {
        self.terms[0].is_none()
    }

    pub fn without_constant(mut self) -> Self {
        self.constant = 0;
        self
    }

    pub fn coefficient(&self, variable: Var) -> u32 {
        self.variables().find(|(v, _)| *v == variable).map_or(0, |(_, c)| c)
    }

    /// The same expression keeping only the variables `keep` accepts, and no constant.
    pub fn restricted(&self, keep: impl Fn(Var) -> bool) -> Self {
        self.variables().filter(|(v, _)| keep(*v)).fold(Self::default(), |sum, (v, c)| sum + Self::var(v) * c)
    }

    pub fn eval(&self, value: impl Fn(Var) -> u32) -> u32 {
        self.variables().fold(self.constant, |sum, (v, c)| sum.wrapping_add(c.wrapping_mul(value(v))))
    }
}

impl From<u32> for Expr {
    fn from(constant: u32) -> Self {
        Self { constant, terms: [None; EXPR_TERMS] }
    }
}

impl std::ops::Add for Expr {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        let mut merged: Vec<(Var, u32)> = self.variables().collect();
        for (variable, coefficient) in other.variables() {
            match merged.iter_mut().find(|(v, _)| *v == variable) {
                Some((_, c)) => *c = c.wrapping_add(coefficient),
                None => merged.push((variable, coefficient)),
            }
        }
        merged.retain(|(_, c)| *c != 0);
        merged.sort();
        assert!(merged.len() <= EXPR_TERMS, "índice demasiado complejo para el secuenciador");
        let mut terms = [None; EXPR_TERMS];
        for (slot, term) in terms.iter_mut().zip(merged) {
            *slot = Some(term);
        }
        Self { constant: self.constant.wrapping_add(other.constant), terms }
    }
}

impl std::ops::Mul<u32> for Expr {
    type Output = Self;

    fn mul(self, factor: u32) -> Self {
        if factor == 0 {
            return Self::default();
        }
        let mut result = self;
        result.constant = result.constant.wrapping_mul(factor);
        for (_, c) in result.terms.iter_mut().flatten() {
            *c = c.wrapping_mul(factor);
        }
        result
    }
}

/// What a length or index can be: a plain integer when executing, an `Expr` when compiling.
pub trait Number: Copy + From<u32> + std::ops::Add<Output = Self> + std::ops::Mul<u32, Output = Self> {}

impl Number for u32 {}
impl Number for Expr {}

pub type At<N> = (u32, N);
pub type Pair<N = u32> = (At<N>, At<N>);

/// The vector arithmetic of the die. Implementations must agree bit for bit.
pub trait BlockOps: Sized {
    type N: Number;

    /// Runs `body` for every value from 0 to `count`, which is never zero unless constant.
    fn each(&mut self, count: Self::N, body: impl FnMut(&mut Self, Self::N));
    fn load_vector(&mut self, reg: u32, values: &[f64]);
    fn load_mantissas(&mut self, reg: u32, mantissas: &[i64], exponent: i64);
    /// `dst` becomes row `row` of a table. A compiled program reads it from `table`,
    /// and an implementation that runs directly may use `values` instead.
    fn load_row(&mut self, table: u32, row: Self::N, width: u32, dst: u32, values: impl Fn(u32) -> Vec<f64>);
    fn normalize(&mut self, reg: u32, len: Self::N);
    fn add_sub(&mut self, subtract: bool, a: At<Self::N>, b: At<Self::N>, b_stride: u32, dst: u32, len: Self::N);
    fn multiply(&mut self, a: u32, b: u32, dst: u32, len: Self::N);
    fn scale(&mut self, s_reg: u32, a: u32, dst: u32, len: Self::N);
    /// `dst[set] = Σ a[·] · b[·]` for every set of index pairs.
    fn dots<F>(&mut self, a: u32, b: u32, dst: u32, sets: Self::N, terms: Self::N, indices: F)
    where
        F: Fn(Self::N, Self::N) -> Pair<Self::N>;
    /// One sum of products, kept at full precision until `narrow_sums`.
    fn sum_products(&mut self, set: Self::N, pairs: &[Pair<Self::N>]);
    /// Turns the first `sets` full-precision sums into `dst`, sharing one exponent.
    fn narrow_sums(&mut self, a: u32, b: u32, dst: u32, sets: Self::N);
    /// `dst[j] = Σ weights[t] · matrix[t · stride + offset + j]`, for hardware that can add whole rows.
    fn row_combination(&mut self, weights: u32, matrix: u32, dst: u32, (width, stride, offset): (u32, u32, Self::N), count: Self::N) {
        self.dots(weights, matrix, dst, width.into(), count, |j, t| ((weights, t), (matrix, t * stride + offset + j)));
    }
    fn function(&mut self, function: u32, src: u32, dst: u32, len: Self::N);
    fn reciprocal(&mut self, src: u32, dst: u32);
    fn inverse_sqrt(&mut self, src: u32, dst: u32);
    fn copy_part(&mut self, src: u32, from: Self::N, dst: u32, to: Self::N, len: Self::N);
    fn gather(&mut self, parts: &[(u32, u32)], dst: u32);
    fn align_to(&mut self, src: u32, from: Self::N, exponent: i64, dst: u32, to: Self::N, len: Self::N);
    fn move_element(&mut self, from: At<Self::N>, to: At<Self::N>);
    /// `dst[0]` becomes the largest of `src[..len]`, sharing `src`'s exponent.
    fn maximum(&mut self, src: u32, len: Self::N, dst: u32);
    /// `dst[i]` becomes 1 at the first element equal to `b[0]`, and 0 everywhere else.
    fn mark_equal(&mut self, a: u32, b: u32, dst: u32, len: Self::N);
    fn set_exponent(&mut self, reg: u32, exponent: i64);
    fn subtract_exponent(&mut self, reg: u32, amount: i64);
    /// Runs `body` for the registers that are `step` places after the ones in `windowed`.
    /// It receives the offset to add to them, which a compiled program applies at run time.
    fn with_window(&mut self, step: Self::N, windowed: &[u32], body: impl FnMut(&mut Self, u32));
    /// How many tokens a layer that sees the last `window` of them has at `position`.
    fn window_length(&self, position: Self::N, window: u32) -> Self::N;
    /// Where the token at `position` goes in a store of `window` entries.
    fn window_slot(&self, position: Self::N, window: u32) -> Self::N;
    /// `rows` rows of the matrix from `first`. It is held in register `matrix`, but
    /// `name` lets an implementation stream it from the checkpoint instead.
    #[allow(clippy::too_many_arguments)]
    fn matvec(&mut self, name: &str, matrix: u32, first: Self::N, src: u32, dst: u32, rows: u32, cols: u32);
    /// Copies one row of an embedding-like matrix into `dst`.
    fn copy_row(&mut self, name: &str, matrix: u32, row: Self::N, cols: u32, dst: u32);
    /// True only when an implementation can prove `reg[index]` is zero and may skip work.
    fn known_zero(&self, _reg: u32, _index: Self::N) -> bool {
        false
    }

    fn add(&mut self, a: u32, b: u32, dst: u32, len: Self::N) {
        self.add_sub(false, (a, 0.into()), (b, 0.into()), 1, dst, len);
    }

    fn sub_scalar(&mut self, a: u32, b: u32, dst: u32, len: Self::N) {
        self.add_sub(true, (a, 0.into()), (b, 0.into()), 0, dst, len);
    }

    fn dot(&mut self, a: u32, b: u32, dst: u32, len: Self::N) {
        self.dots(a, b, dst, 1.into(), len, |_, i| ((a, i), (b, i)));
    }

    /// `inverse_len` holds `1 / len` when `len` is not a power of two; otherwise a shift does it.
    /// `epsilon`, when given, keeps an almost silent vector from being blown up.
    fn rms_normalize(&mut self, x: u32, dst: u32, len: u32, temporary: (u32, u32), inverse_len: Option<u32>, epsilon: Option<u32>) {
        self.dot(x, x, temporary.0, len.into());
        let (mean, spare) = match inverse_len {
            Some(factor) => {
                self.scale(factor, temporary.0, temporary.1, 1.into());
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
        self.dot(x, x, temporary.0, len.into());
        self.invert_root_and_scale(x, dst, len, temporary, epsilon);
    }

    /// `dst = x / sqrt(value + epsilon)`, where `registers.0` holds `value`.
    fn invert_root_and_scale(&mut self, x: u32, dst: u32, len: u32, registers: (u32, u32), epsilon: Option<u32>) {
        let (value, spare) = match epsilon {
            Some(e) => {
                self.add(registers.0, e, registers.1, 1.into());
                (registers.1, registers.0)
            }
            None => registers,
        };
        self.inverse_sqrt(value, spare);
        self.scale(spare, x, dst, len.into());
    }

    /// Softmax attention: `query` against `tokens` keys, then the weighted sum of values.
    /// Runs `body` for every pair of indices, at once where the die has several vector units.
    fn each_head(&mut self, outer: u32, inner: u32, mut body: impl FnMut(&mut Self, Self::N, Self::N)) {
        self.each(outer.into(), |s, o| s.each(inner.into(), |s, i| body(s, o, i)));
    }

    /// Like `gather`, for parts that `each_head` left on the units that made them.
    fn gather_heads(&mut self, parts: &[(u32, u32)], dst: u32) {
        self.gather(parts, dst);
    }

    /// `dst[c] = Σ taps[k · width + c] · history[k · width + c]` over `kernel` rows.
    fn depthwise(&mut self, taps: u32, history: u32, dst: u32, width: u32, kernel: u32) {
        self.dots(taps, history, dst, width.into(), kernel.into(), |c, k| ((taps, k * width + c), (history, k * width + c)));
    }

    /// Moves `width` elements of `reg` to an earlier place in the same register.
    fn copy_elements(&mut self, reg: u32, from: Self::N, to: Self::N, width: u32) {
        self.each(width.into(), |s, c| s.move_element((reg, from + c), (reg, to + c)));
    }

    /// `dst[j] = Σ probabilities[t] · value of token t at j` for one head.
    fn combine_values(&mut self, probabilities: u32, store: KeyStore<Self::N>, dst: u32, width: u32, tokens: Self::N) {
        self.row_combination(probabilities, store.values, dst, (width, store.stride, store.offset), tokens);
    }

    /// Writes `cos · x + sin · swapped`, where `sin` only covers the first `rotary` elements.
    fn rotate(&mut self, x: u32, swapped: u32, (cos, sin): (u32, u32), dst: u32, width: u32, rotary: u32) {
        self.each(rotary.into(), |s, k| s.sum_products(k, &[((x, k), (cos, k)), ((swapped, k), (sin, k))]));
        self.each((width - rotary).into(), |s, k| {
            let at = k + rotary.into();
            s.sum_products(at, &[((x, at), (cos, at))]);
        });
        self.narrow_sums(x, cos, dst, width.into());
    }

    /// Keeps the key of one token at its place in the store.
    fn store_key(&mut self, src: u32, exponent: i64, keys: u32, slot: Self::N, width: u32) {
        self.align_to(src, 0.into(), exponent, keys, slot * width, width.into());
    }

    /// `dst[t]` becomes the product of a query with the key of every stored token.
    fn score_keys(&mut self, query: u32, store: KeyStore<Self::N>, dst: u32, width: u32, tokens: Self::N) {
        let KeyStore { keys, stride, offset, .. } = store;
        self.dots(query, keys, dst, tokens, width.into(), |t, j| ((query, j), (keys, t * stride + offset + j)));
    }

    #[allow(clippy::too_many_arguments)]
    fn attention(&mut self, query: u32, store: KeyStore<Self::N>, ones: u32, dst: u32, width: u32, tokens: Self::N, score_scale: Option<u32>, regs: AttentionRegs) {
        self.score_keys(query, store, regs.scores, width, tokens);
        if let Some(scale) = score_scale {
            self.scale(scale, regs.scores, regs.top, tokens);
            self.copy_part(regs.top, 0.into(), regs.scores, 0.into(), tokens);
        }
        if let Some((inverse_cap, cap)) = regs.softcap {
            self.scale(inverse_cap, regs.scores, regs.top, tokens);
            self.function(FN_TANH, regs.top, regs.shifted, tokens);
            self.scale(cap, regs.shifted, regs.scores, tokens);
        }
        self.maximum(regs.scores, tokens, regs.top);
        self.sub_scalar(regs.scores, regs.top, regs.shifted, tokens);
        self.function(FN_EXP, regs.shifted, regs.weights, tokens);
        self.dots(regs.weights, ones, regs.total, 1.into(), tokens, |_, t| ((regs.weights, t), (ones, t)));
        self.reciprocal(regs.total, regs.inverse);
        self.scale(regs.inverse, regs.weights, regs.probabilities, tokens);
        self.combine_values(regs.probabilities, store, dst, width, tokens);
    }

    /// One Gated DeltaNet head: the delta rule on a `key × value` state matrix.
    fn delta_step(&mut self, r: DeltaRegs, key: u32, value: u32) {
        let cells = key * value;
        self.scale(r.decay, r.state, r.state, cells.into());
        self.row_combination(r.key, r.state, r.memory, (value, value, 0.into()), key.into());
        self.add_sub(true, (r.value, 0.into()), (r.memory, 0.into()), 1, r.error, value.into());
        self.scale(r.beta, r.error, r.delta, value.into());
        // The output reads the old state and the update apart, before rounding merges them:
        // the update's contribution is often a tiny difference that rounding would erase.
        self.row_combination(r.query, r.state, r.next, (value, value, 0.into()), key.into());
        self.dot(r.key, r.query, r.memory, key.into());
        self.scale(r.memory, r.delta, r.update, value.into());
        self.add(r.next, r.update, r.out, value.into());
        self.outer_product(r.key, r.delta, r.update, key, value);
        self.add(r.state, r.update, r.next, cells.into());
        self.copy_part(r.next, 0.into(), r.state, 0.into(), cells.into());
    }

    fn outer_product(&mut self, a: u32, b: u32, dst: u32, rows: u32, cols: u32);
}

#[derive(Clone, Copy, Debug)]
pub struct Alu<N = u32> {
    pub op: u32,
    pub a: At<N>,
    pub b: At<N>,
    pub dst: At<N>,
    /// Index registers added to each operand's literal; 0 is always zero.
    pub offset_a: u32,
    pub offset_b: u32,
    pub offset_dst: u32,
    /// Added to the registers of the operands marked below, for loops over registers.
    pub window: N,
    pub window_x: u32,
    pub window_a: bool,
    pub window_b: bool,
    pub window_dst: bool,
    /// One element per lane at once; the indices are those of the first lane.
    pub vector: bool,
    /// Operand A comes from this vector unit, plus one; zero means its own.
    pub from_cluster: u32,
    pub broadcast_a: bool,
    pub broadcast_b: bool,
    pub shift: u32,
    pub function: u32,
    pub literal: Option<u32>,
    pub write: bool,
    pub load_shift_a: bool,
    pub load_shift_b: bool,
    pub load_shift_left: bool,
    pub load_wide_shift: bool,
    pub magnitude_scan: bool,
    /// Keeps the largest mantissa read, across every lane of a block.
    pub max_scan: bool,
    pub max_clear: bool,
    /// The result is that largest mantissa.
    pub max_out: bool,
    pub magnitude_clear: bool,
    pub accumulate: bool,
    pub accumulator_clear: bool,
    pub link_send: bool,
    pub wide_write: bool,
    pub wide_scan: bool,
    pub wide_clear: bool,
}

impl<N: Default> Alu<N> {
    fn with_op(op: u32) -> Self {
        Alu {
            op,
            a: (0, N::default()),
            b: (0, N::default()),
            dst: (0, N::default()),
            offset_a: 0,
            offset_b: 0,
            offset_dst: 0,
            window: N::default(),
            window_x: 0,
            window_a: false,
            window_b: false,
            window_dst: false,
            vector: false,
            from_cluster: 0,
            broadcast_a: false,
            broadcast_b: false,
            shift: 0,
            function: 0,
            literal: None,
            write: true,
            load_shift_a: false,
            load_shift_b: false,
            load_shift_left: false,
            load_wide_shift: false,
            magnitude_scan: false,
            max_scan: false,
            max_clear: false,
            max_out: false,
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

impl Alu<u32> {
    pub fn new(op: u32) -> Self {
        Self::with_op(op)
    }
}

impl Alu<Expr> {
    /// The same step with its indices replaced by what `index` makes of them.
    pub fn map_indices(&self, mut index: impl FnMut(Expr) -> u32) -> Alu<u32> {
        let (a, b, dst) = (index(self.a.1), index(self.b.1), index(self.dst.1));
        let window = index(self.window);
        Alu {
            op: self.op,
            a: (self.a.0, a),
            b: (self.b.0, b),
            dst: (self.dst.0, dst),
            offset_a: self.offset_a,
            offset_b: self.offset_b,
            offset_dst: self.offset_dst,
            window,
            window_x: self.window_x,
            window_a: self.window_a,
            window_b: self.window_b,
            window_dst: self.window_dst,
            vector: self.vector,
            from_cluster: self.from_cluster,
            broadcast_a: self.broadcast_a,
            broadcast_b: self.broadcast_b,
            shift: self.shift,
            function: self.function,
            literal: self.literal,
            write: self.write,
            load_shift_a: self.load_shift_a,
            load_shift_b: self.load_shift_b,
            load_shift_left: self.load_shift_left,
            load_wide_shift: self.load_wide_shift,
            magnitude_scan: self.magnitude_scan,
            max_scan: self.max_scan,
            max_clear: self.max_clear,
            max_out: self.max_out,
            magnitude_clear: self.magnitude_clear,
            accumulate: self.accumulate,
            accumulator_clear: self.accumulator_clear,
            link_send: self.link_send,
            wide_write: self.wide_write,
            wide_scan: self.wide_scan,
            wide_clear: self.wide_clear,
        }
    }
}

impl From<Alu<u32>> for Alu<Expr> {
    fn from(s: Alu<u32>) -> Self {
        let mut out = Alu::<Expr>::with_op(s.op);
        out.a = (s.a.0, s.a.1.into());
        out.b = (s.b.0, s.b.1.into());
        out.dst = (s.dst.0, s.dst.1.into());
        out.shift = s.shift;
        out.function = s.function;
        out.literal = s.literal;
        out.write = s.write;
        out.load_shift_a = s.load_shift_a;
        out.load_shift_b = s.load_shift_b;
        out.load_shift_left = s.load_shift_left;
        out.load_wide_shift = s.load_wide_shift;
        out.magnitude_scan = s.magnitude_scan;
        (out.max_scan, out.max_clear, out.max_out) = (s.max_scan, s.max_clear, s.max_out);
        out.magnitude_clear = s.magnitude_clear;
        out.accumulate = s.accumulate;
        out.accumulator_clear = s.accumulator_clear;
        out.link_send = s.link_send;
        out.wide_write = s.wide_write;
        out.wide_scan = s.wide_scan;
        out.wide_clear = s.wide_clear;
        out
    }
}

/// A microprogram before it is laid out: single-element steps and counted loops.
#[derive(Clone, Debug)]
pub enum Step {
    One(Alu<Expr>),
    Repeat { variable: Var, count: Expr, body: Vec<Step> },
    /// A single-operation loop that issues one element, or block, per cycle.
    Stream { variable: Var, count: Expr, in_elements: bool, body: Alu<Expr> },
    /// `wide[j] += weights · matrix[start + j]` for every `j` of a row, or from zero if `fresh`.
    Combine { weights: At<Expr>, matrix: u32, start: Expr, width: Expr, fresh: bool, window: Expr, window_a: bool, window_b: bool, moving: bool },
    /// One copy of `body` on several vector units, each with its own values of `vars`.
    Clustered { vars: Vec<Var>, assignments: Vec<Vec<u32>>, body: Vec<Step> },
    /// Moves every cell to where a matrix starts.
    CellSeek { weights: Expr, factors: Expr },
    /// Clears the row totals and restarts the input vector.
    CellPass,
    /// One group of columns through the cells.
    CellGroup { src: u32, length: u32, start: Expr },
    /// The totals of all the vector units are added together in their wide memories.
    WideReduce { rows: u32 },
    /// The totals of the first `rows` cells go to the wide memory.
    CellDrain { rows: u32 },
}

/// Where the attention of one head reads its keys and values: each token's row starts
/// `stride` further on, and this head's part of it `offset` into the row.
#[derive(Clone, Copy, Debug)]
pub struct KeyStore<N> {
    pub keys: u32,
    pub values: u32,
    pub stride: u32,
    pub offset: N,
}

/// Below this a loop is not worth the words it takes to set up a stream.
const MIN_STREAM: u32 = 4;

/// Fractional bits of the scale and bias the cells multiply by.
pub const FACTOR_FRACTION: u32 = 4;

/// Where the cells keep a matrix.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CellMatrix {
    pub weights: u32,
    pub factors: u32,
    pub group: u32,
    pub groups: u32,
}

impl From<Alu<u32>> for Step {
    fn from(s: Alu<u32>) -> Self {
        Step::One(s.into())
    }
}

fn alu(op: u32) -> Alu<Expr> {
    Alu::with_op(op)
}

/// A program of steps plus the register contents it starts from.
pub struct Assembly {
    pub steps: Vec<Step>,
    pub initial: HashMap<Place, i64>,
    bits: u32,
    index_bits: u32,
    exponents: u32,
    scratch: u32,
    loops: u32,
    lane_bits: u32,
    cells: u32,
    streams: bool,
    combines: bool,
    clusters: u32,
    /// With keys kept in columns, how many tokens each column has room for.
    pub key_capacity: Option<u32>,
    /// Matrices the cells hold. Any other is computed from the vector registers.
    pub in_cells: HashMap<String, CellMatrix>,
}

impl Assembly {
    pub fn new(c: &DieConfig) -> Self {
        Self { steps: Vec::new(), initial: HashMap::new(), bits: c.cell.activation_bits, index_bits: c.index_bits, exponents: exponent_register(c), scratch: scratch_register(c), loops: 0, lane_bits: c.features.lane_bits, cells: c.cells, streams: c.features.streams(), combines: c.features.combines(), clusters: 1 << c.features.cluster_bits, key_capacity: None, in_cells: HashMap::new() }
    }

    pub fn exponent(&self, reg: u32) -> At<Expr> {
        (self.exponents, reg.into())
    }

    fn push(&mut self, step: Alu<Expr>) {
        self.steps.push(Step::One(step));
    }

    /// Scratch element for values that have to be built up across several steps.
    fn sc(&self, index: u32) -> At<Expr> {
        (self.scratch, index.into())
    }

    /// Writes a small constant, composing it when it does not fit one literal.
    fn immediate(&mut self, value: i64, dst: At<Expr>) {
        let limit = (1i64 << (self.index_bits - 1)) - 1;
        if value.abs() <= limit {
            let mut s = alu(OP_IMMEDIATE);
            s.literal = Some(value as u32 & ((1 << self.index_bits) - 1));
            s.dst = dst;
            self.push(s);
        } else {
            let part = value / 2;
            self.immediate(part, self.sc(16));
            self.immediate(value - part, self.sc(17));
            self.binary(OP_ADD, self.sc(16), self.sc(17), dst);
        }
    }

    fn cell_round(&mut self, src: u32, place: CellMatrix, rows: u32, column: Expr) {
        self.steps.push(Step::CellPass);
        let groups = place.groups.div_ceil(self.clusters);
        self.each(groups.into(), |s, _| s.steps.push(Step::CellGroup { src, length: place.group, start: column }));
        self.steps.push(Step::CellDrain { rows });
    }

    /// Sets the shifters so that a vector of `src` lands on the fixed `exponent`.
    fn load_alignment(&mut self, src: u32, exponent: i64) {
        let e = self.exponent(src);
        self.immediate(exponent, self.sc(8));
        self.binary(OP_SUB, e, self.sc(8), self.sc(9));
        self.immediate(0, self.sc(10));
        self.binary(OP_MAX, self.sc(9), self.sc(10), self.sc(11));
        self.binary(OP_SUB, self.sc(8), e, self.sc(12));
        self.binary(OP_MAX, self.sc(12), self.sc(10), self.sc(13));
        self.load_shifts(Some(self.sc(13)), None, Some(self.sc(11)));
    }

    /// Runs `emit` for each element, whole blocks at a time where the die has lanes.
    /// A block needs a known length and operands that start on a block boundary, and
    /// none of its elements may change an operand another lane is taking whole.
    fn elements(&mut self, len: Expr, starts: &[Expr], independent: bool, mut emit: impl FnMut(&mut Self, Expr, bool)) {
        let lanes = 1u32 << self.lane_bits;
        let aligned = starts.iter().all(|s| s.is_multiple_of(lanes));
        let partial = len.is_constant() && !len.constant().is_multiple_of(lanes) && len.constant() >= MIN_STREAM;
        if lanes > 1 && self.streams && independent && aligned && (!len.is_constant() || partial) {
            // Blocks past the end only touch places nothing reads afterwards.
            let variable = Var::Loop(self.loops);
            self.loops += 1;
            let outer = std::mem::take(&mut self.steps);
            emit(self, Expr::var(variable) * lanes, true);
            let inner = std::mem::replace(&mut self.steps, outer);
            if let [Step::One(step)] = inner.as_slice() {
                if can_stream(variable, step, lanes) {
                    self.steps.push(Step::Stream { variable, count: len, in_elements: true, body: *step });
                    return;
                }
            }
        }
        if lanes == 1 || !independent || !aligned || !len.is_constant() || len.constant() < lanes {
            self.each(len, |s, i| emit(s, i, false));
            return self.stream_last(1);
        }
        let (blocks, rest) = (len.constant() / lanes, len.constant() % lanes);
        self.each(blocks.into(), |s, i| emit(s, i * lanes, true));
        self.stream_last(lanes);
        self.each(rest.into(), |s, i| emit(s, i + (blocks * lanes).into(), false));
        self.stream_last(1);
    }

    /// Turns the loop just built into a stream, if nothing in it needs the steps apart.
    fn stream_last(&mut self, unit: u32) {
        if !self.streams {
            return;
        }
        let Some(Step::Repeat { variable, count, body }) = self.steps.last() else {
            return;
        };
        let ([Step::One(step)], true) = (body.as_slice(), !count.is_constant() || count.constant() >= MIN_STREAM) else {
            return;
        };
        let (variable, count, step) = (*variable, *count, *step);
        if can_stream(variable, &step, unit) {
            *self.steps.last_mut().expect("bucle") = Step::Stream { variable, count, in_elements: false, body: step };
        }
    }

    fn binary(&mut self, op: u32, a: At<Expr>, b: At<Expr>, dst: At<Expr>) {
        let mut s = alu(op);
        s.a = a;
        s.b = b;
        s.dst = dst;
        self.push(s);
    }

    fn load_shifts(&mut self, a: Option<At<Expr>>, b: Option<At<Expr>>, left: Option<At<Expr>>) {
        // Sources are read through the A shifter, so A has to be loaded last.
        let zero = self.sc(15);
        for (place, which) in [(left, 2), (b, 1), (a, 0)] {
            let mut s = alu(OP_MOV);
            s.a = place.unwrap_or(zero);
            match which {
                0 => s.load_shift_a = true,
                1 => s.load_shift_b = true,
                _ => s.load_shift_left = true,
            }
            s.write = false;
            self.push(s);
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

    fn clear_accumulator(&mut self) {
        let mut clear = alu(OP_MOV);
        clear.write = false;
        clear.accumulator_clear = true;
        self.push(clear);
    }

    fn accumulate(&mut self, a: At<Expr>, b: At<Expr>) {
        let mut s = alu(OP_MOV);
        s.a = a;
        s.b = b;
        s.write = false;
        s.accumulate = true;
        self.push(s);
    }

    fn store_accumulator(&mut self, wide_index: Expr) {
        let mut write = alu(OP_MOV);
        write.write = false;
        write.wide_write = true;
        write.dst = (0, wide_index);
        self.push(write);
    }

    fn table_input(&mut self, function: u32, source: At<Expr>, dst: At<Expr>) {
        let mut s = alu(OP_FUNCTION);
        s.a = source;
        s.function = function;
        s.dst = dst;
        self.push(s);
    }

    /// Sends a register's exponent and its first `len` values to the next chip.
    pub fn send(&mut self, reg: u32, len: Expr) {
        self.clear_shifts();
        self.send_place(self.exponent(reg));
        self.each(len, |s, i| s.send_place((reg, i)));
    }

    /// Fills a register with what `send` delivered.
    pub fn receive(&mut self, reg: u32, len: Expr) {
        self.receive_place(self.exponent(reg));
        self.each(len, |s, i| s.receive_place((reg, i)));
    }

    /// Tells the next chip that this one has finished.
    pub fn send_signal(&mut self) {
        self.clear_shifts();
        self.immediate(1, self.sc(1));
        self.send_place(self.sc(1));
    }

    /// Waits for `send_signal`.
    pub fn receive_signal(&mut self) {
        self.receive_place(self.sc(1));
    }

    fn send_place(&mut self, place: At<Expr>) {
        let mut word = alu(OP_MOV);
        word.a = place;
        word.write = false;
        word.link_send = true;
        self.push(word);
    }

    fn receive_place(&mut self, place: At<Expr>) {
        let mut word = alu(OP_LINK);
        word.dst = place;
        self.push(word);
    }

    fn scan_and_shift(&mut self, reg: u32, len: Expr) {
        let mut first = alu(OP_MOV);
        first.a = (reg, 0.into());
        first.write = false;
        first.magnitude_clear = true;
        self.push(first);
        self.elements(len, &[], true, |s, i, vector| {
            let mut scan = alu(OP_MOV);
            scan.a = (reg, i);
            scan.magnitude_scan = true;
            scan.write = false;
            scan.vector = vector;
            s.push(scan);
        });
    }
}

fn mark_window(steps: &mut [Step], step: Expr, windowed: &[u32], exponents: u32) {
    for item in steps {
        match item {
            Step::One(alu) => {
                alu.window = step;
                alu.window_a = windowed.contains(&alu.a.0);
                alu.window_b = windowed.contains(&alu.b.0);
                alu.window_dst = windowed.contains(&alu.dst.0);
                // A register's exponent lives at its number, so that index moves as well.
                for at in [&mut alu.a, &mut alu.b, &mut alu.dst] {
                    if at.0 == exponents && at.1.is_constant() && windowed.contains(&at.1.constant()) {
                        at.1 = at.1 + step;
                    }
                }
            }
            Step::Repeat { body, .. } => mark_window(body, step, windowed, exponents),
            Step::Stream { body, .. } => {
                let mut one = [Step::One(*body)];
                mark_window(&mut one, step, windowed, exponents);
                if let [Step::One(marked)] = one {
                    *body = marked;
                }
            }
            Step::Combine { weights, matrix, window, window_a, window_b, .. } => {
                *window = step;
                *window_a = windowed.contains(&weights.0);
                *window_b = windowed.contains(matrix);
            }
            Step::Clustered { body, .. } => mark_window(body, step, windowed, exponents),
            Step::CellGroup { src, .. } => assert!(!windowed.contains(src), "las celdas leen de registros fijos"),
            Step::CellSeek { .. } | Step::CellPass | Step::CellDrain { .. } | Step::WideReduce { .. } => {}
        }
    }
}

/// Whether one loop can issue an element per cycle: results land two or three cycles
/// after the read, so nothing may read a place an earlier element is still to write.
fn can_stream(variable: Var, step: &Alu<Expr>, unit: u32) -> bool {
    let stride = |index: Expr| index.coefficient(variable);
    let moves_with = |index: Expr| stride(index) == 0 || stride(index) == unit;
    // Moving elements to earlier places is safe: the reads stay ahead of the writes.
    let safe = |operand: At<Expr>| operand.0 != step.dst.0 || (stride(operand.1) == unit && (operand.1 == step.dst.1 || operand.1.is_after(step.dst.1)));
    let plain = !(step.load_shift_a || step.load_shift_b || step.load_shift_left || step.load_wide_shift || step.link_send || step.magnitude_clear || step.max_clear || step.max_out || step.accumulator_clear || step.wide_clear || step.wide_write) && step.literal.is_none();
    let streamable = matches!(step.op, OP_ADD | OP_SUB | OP_MUL | OP_MOV | OP_FUNCTION | OP_SHIFT_LEFT | OP_WIDE_OUT);
    let scans = !step.write;
    plain && streamable && (scans || stride(step.dst.1) == unit) && moves_with(step.a.1) && moves_with(step.b.1) && (scans || (safe(step.a) && safe(step.b)))
}

impl BlockOps for Assembly {
    type N = Expr;

    fn window_length(&self, position: Expr, window: u32) -> Expr {
        assert_eq!(position, Expr::var(Var::Position), "la ventana se mide desde la posición");
        Expr::var(Var::WindowLength(window))
    }

    fn window_slot(&self, position: Expr, window: u32) -> Expr {
        assert_eq!(position, Expr::var(Var::Position), "la ventana se mide desde la posición");
        Expr::var(Var::WindowSlot(window))
    }

    fn with_window(&mut self, step: Expr, windowed: &[u32], mut body: impl FnMut(&mut Self, u32)) {
        if step.is_constant() {
            return body(self, step.constant());
        }
        let outer = std::mem::take(&mut self.steps);
        body(self, 0);
        let mut inner = std::mem::replace(&mut self.steps, outer);
        mark_window(&mut inner, step, windowed, self.exponents);
        self.steps.extend(inner);
    }

    fn each(&mut self, count: Expr, mut body: impl FnMut(&mut Self, Expr)) {
        if count.is_constant() {
            match count.constant() {
                0 => return,
                1 => return body(self, 0.into()),
                _ => {}
            }
        }
        let variable = Var::Loop(self.loops);
        self.loops += 1;
        let outer = std::mem::take(&mut self.steps);
        body(self, Expr::var(variable));
        let inner = std::mem::replace(&mut self.steps, outer);
        self.steps.push(Step::Repeat { variable, count, body: inner });
    }

    fn load_vector(&mut self, reg: u32, values: &[f64]) {
        let (mantissas, exponent) = encode_vector(values, self.bits);
        self.load_mantissas(reg, &mantissas, exponent);
    }

    fn load_mantissas(&mut self, reg: u32, mantissas: &[i64], exponent: i64) {
        for (i, m) in mantissas.iter().enumerate() {
            self.initial.insert((reg, i as u32), *m);
        }
        self.initial.insert((self.exponents, reg), exponent);
    }

    fn load_row(&mut self, table: u32, row: Expr, width: u32, dst: u32, _values: impl Fn(u32) -> Vec<f64>) {
        self.copy_part(table, row * width, dst, 0.into(), width.into());
    }

    fn normalize(&mut self, reg: u32, len: Expr) {
        self.clear_shifts();
        self.scan_and_shift(reg, len);
        let mut count = alu(OP_LZC);
        count.dst = self.sc(0);
        self.push(count);
        self.load_shifts(None, None, Some(self.sc(0)));
        self.elements(len, &[], true, |s, i, vector| {
            let mut shift = alu(OP_SHIFT_LEFT);
            shift.a = (reg, i);
            shift.dst = (reg, i);
            shift.vector = vector;
            s.push(shift);
        });
        self.binary(OP_SUB, self.exponent(reg), self.sc(0), self.exponent(reg));
        self.clear_shifts();
    }

    fn add_sub(&mut self, subtract: bool, a: At<Expr>, b: At<Expr>, b_stride: u32, dst: u32, len: Expr) {
        self.clear_shifts();
        let (ea, eb) = (self.exponent(a.0), self.exponent(b.0));
        self.binary(OP_MAX, ea, eb, self.sc(1));
        self.immediate(1, self.sc(2));
        self.binary(OP_ADD, self.sc(1), self.sc(2), self.sc(3));
        self.binary(OP_SUB, self.sc(3), ea, self.sc(4));
        self.binary(OP_SUB, self.sc(3), eb, self.sc(5));
        self.load_shifts(Some(self.sc(4)), Some(self.sc(5)), None);
        let op = if subtract { OP_SUB } else { OP_ADD };
        self.elements(len, &[a.1, b.1], b_stride != 0 || b.0 != dst, |s, i, vector| {
            let mut step = alu(op);
            (step.a, step.b, step.dst) = ((a.0, a.1 + i), (b.0, b.1 + i * b_stride), (dst, i));
            (step.vector, step.broadcast_b) = (vector, vector && b_stride == 0);
            s.push(step);
        });
        self.clear_shifts();
        self.binary(OP_MOV, self.sc(3), self.sc(15), self.exponent(dst));
        self.normalize(dst, len);
    }

    fn multiply(&mut self, a: u32, b: u32, dst: u32, len: Expr) {
        let shift = self.bits - 1;
        self.elements(len, &[], true, |s, i, vector| {
            let mut product = alu(OP_MUL);
            product.a = (a, i);
            product.b = (b, i);
            product.dst = (dst, i);
            product.shift = shift;
            product.vector = vector;
            s.push(product);
        });
        self.product_exponent(a, b, dst, 0);
        self.normalize(dst, len);
    }

    fn scale(&mut self, s_reg: u32, a: u32, dst: u32, len: Expr) {
        let shift = self.bits - 1;
        self.elements(len, &[], s_reg != dst, |s, i, vector| {
            let mut product = alu(OP_MUL);
            product.a = (s_reg, 0.into());
            product.b = (a, i);
            product.dst = (dst, i);
            product.shift = shift;
            (product.vector, product.broadcast_a) = (vector, vector);
            s.push(product);
        });
        self.product_exponent(s_reg, a, dst, 0);
        self.normalize(dst, len);
    }

    fn outer_product(&mut self, a: u32, b: u32, dst: u32, rows: u32, cols: u32) {
        let shift = self.bits - 1;
        self.each(rows.into(), |s, i| {
            s.elements(cols.into(), &[i * cols], a != dst && b != dst, |s, j, vector| {
                let mut product = alu(OP_MUL);
                product.a = (a, i);
                product.b = (b, j);
                product.dst = (dst, i * cols + j);
                product.shift = shift;
                (product.vector, product.broadcast_a) = (vector, vector);
                s.push(product);
            });
        });
        self.product_exponent(a, b, dst, 0);
        self.normalize(dst, (rows * cols).into());
    }

    fn dots<F>(&mut self, a: u32, b: u32, dst: u32, sets: Expr, terms: Expr, indices: F)
    where
        F: Fn(Expr, Expr) -> Pair<Expr>,
    {
        // Both operands must walk one element per term for a block to take several.
        let probe = Var::Loop(self.loops);
        self.loops += 1;
        let (x, y) = indices(0.into(), Expr::var(probe));
        let unit = |index: Expr| matches!(index.coefficient(probe), 0 | 1);
        let (fixed_x, fixed_y) = (x.1.coefficient(probe) == 0, y.1.coefficient(probe) == 0);
        let in_blocks = unit(x.1) && unit(y.1);
        self.each(sets, |s, set| {
            s.clear_accumulator();
            if in_blocks {
                let (first_x, first_y) = indices(set, 0.into());
                s.elements(terms, &[first_x.1, first_y.1], true, |s, term, vector| {
                    let (x, y) = indices(set, term);
                    let mut step = alu(OP_MOV);
                    (step.a, step.b, step.write, step.accumulate) = (x, y, false, true);
                    (step.vector, step.broadcast_a, step.broadcast_b) = (vector, vector && fixed_x, vector && fixed_y);
                    s.push(step);
                });
            } else {
                s.each(terms, |s, term| {
                    let (x, y) = indices(set, term);
                    s.accumulate(x, y);
                });
                s.stream_last(1);
            }
            s.store_accumulator(set);
        });
        self.narrow_sums(a, b, dst, sets);
    }

    fn sum_products(&mut self, set: Expr, pairs: &[Pair<Expr>]) {
        self.clear_accumulator();
        for (x, y) in pairs {
            self.accumulate(*x, *y);
        }
        self.store_accumulator(set);
    }

    /// Narrows the wide sums to mantissas, shifting only as much as the largest needs.
    fn row_combination(&mut self, weights: u32, matrix: u32, dst: u32, (width, stride, offset): (u32, u32, Expr), count: Expr) {
        if !self.combines || width < MIN_STREAM {
            return self.dots(weights, matrix, dst, width.into(), count, |j, t| ((weights, t), (matrix, t * stride + offset + j)));
        }
        // A zero weight clears every sum, whatever the matrix holds.
        let zero = self.sc(10);
        self.immediate(0, zero);
        self.steps.push(Step::Combine { weights: zero, matrix, start: 0.into(), width: width.into(), fresh: true, window: 0.into(), window_a: false, window_b: false, moving: false });
        self.each(count, |s, t| s.steps.push(Step::Combine { weights: (weights, t), matrix, start: t * stride + offset, width: width.into(), fresh: false, window: 0.into(), window_a: false, window_b: false, moving: false }));
        self.narrow_sums(weights, matrix, dst, width.into());
    }

    fn narrow_sums(&mut self, a: u32, b: u32, dst: u32, sets: Expr) {
        let mut clear = alu(OP_MOV);
        clear.write = false;
        clear.wide_clear = true;
        self.push(clear);
        self.elements(sets, &[], true, |s, set, vector| {
            let mut scan = alu(OP_MOV);
            scan.a = (0, set);
            scan.write = false;
            scan.wide_scan = true;
            scan.vector = vector;
            s.push(scan);
        });
        let mut count = alu(OP_WIDE_LZC);
        count.dst = self.sc(0);
        self.push(count);
        self.immediate(accumulator_bits(self.bits) as i64 - self.bits as i64, self.sc(1));
        self.binary(OP_SUB, self.sc(1), self.sc(0), self.sc(2));
        self.immediate(0, self.sc(3));
        self.binary(OP_MAX, self.sc(2), self.sc(3), self.sc(4));
        let mut load = alu(OP_MOV);
        load.a = self.sc(4);
        load.write = false;
        load.load_wide_shift = true;
        self.push(load);
        self.elements(sets, &[], true, |s, set, vector| {
            let mut out = alu(OP_WIDE_OUT);
            out.a = (0, set);
            out.dst = (dst, set);
            out.vector = vector;
            s.push(out);
        });
        self.binary(OP_ADD, self.exponent(a), self.exponent(b), self.sc(5));
        self.binary(OP_ADD, self.sc(5), self.sc(4), self.sc(6));
        self.immediate(1 - self.bits as i64, self.sc(7));
        self.binary(OP_ADD, self.sc(6), self.sc(7), self.exponent(dst));
        self.normalize(dst, sets);
    }

    fn function(&mut self, function: u32, src: u32, dst: u32, len: Expr) {
        let scratch = self.scratch;
        // The table input must start on a block boundary to go a block at a time.
        let start: Expr = 32u32.max(1 << self.lane_bits).into();
        self.align_to(src, 0.into(), TABLE_INPUT_EXPONENT, scratch, start, len);
        self.elements(len, &[start], true, |s, i, vector| {
            let mut step = alu(OP_FUNCTION);
            (step.a, step.function, step.dst, step.vector) = ((scratch, i + start), function, (dst, i), vector);
            s.push(step);
        });
        let exponent = if function == FN_SOFTPLUS { SOFTPLUS_OUTPUT_EXPONENT } else { UNIT_OUTPUT_EXPONENT };
        self.immediate(exponent, self.exponent(dst));
        self.normalize(dst, len);
    }

    fn reciprocal(&mut self, src: u32, dst: u32) {
        self.table_input(FN_RECIPROCAL, (src, 0.into()), (dst, 0.into()));
        self.immediate(RECIPROCAL_OUTPUT_EXPONENT, self.sc(1));
        self.binary(OP_SUB, self.sc(1), self.exponent(src), self.exponent(dst));
        self.normalize(dst, 1.into());
    }

    fn inverse_sqrt(&mut self, src: u32, dst: u32) {
        let e = self.exponent(src);
        self.clear_shifts();
        let mut parity = alu(OP_PARITY);
        parity.a = e;
        parity.dst = self.sc(1);
        self.push(parity);
        self.load_shifts(Some(self.sc(1)), None, None);
        self.binary(OP_MOV, (src, 0.into()), self.sc(15), self.sc(2));
        self.clear_shifts();
        self.binary(OP_ADD, e, self.sc(1), self.sc(3));
        self.table_input(FN_INV_SQRT, self.sc(2), (dst, 0.into()));
        self.immediate(1, self.sc(4));
        self.load_shifts(Some(self.sc(4)), None, None);
        self.binary(OP_MOV, self.sc(3), self.sc(15), self.sc(5));
        self.clear_shifts();
        self.immediate(RECIPROCAL_OUTPUT_EXPONENT, self.sc(6));
        self.binary(OP_SUB, self.sc(6), self.sc(5), self.exponent(dst));
        self.normalize(dst, 1.into());
    }

    fn copy_part(&mut self, src: u32, from: Expr, dst: u32, to: Expr, len: Expr) {
        self.clear_shifts();
        let zero = self.sc(15);
        self.elements(len, &[from, to], true, |s, i, vector| {
            let mut step = alu(OP_MOV);
            (step.a, step.b, step.dst) = ((src, from + i), zero, (dst, to + i));
            (step.vector, step.broadcast_b) = (vector, vector);
            s.push(step);
        });
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
            let at: Expr = offset.into();
            self.elements((*len).into(), &[at], true, |s, i, vector| {
                let mut step = alu(OP_MOV);
                (step.a, step.b, step.dst) = ((*reg, i), zero, (dst, at + i));
                (step.vector, step.broadcast_b) = (vector, vector);
                s.push(step);
            });
            self.clear_shifts();
            offset += len;
        }
        self.binary(OP_MOV, largest, zero, self.exponent(dst));
        self.normalize(dst, offset.into());
    }

    fn combine_values(&mut self, probabilities: u32, store: KeyStore<Expr>, dst: u32, width: u32, tokens: Expr) {
        let KeyStore { values, stride, offset, .. } = store;
        let Some(capacity) = self.key_capacity else {
            return self.row_combination(probabilities, values, dst, (width, stride, offset), tokens);
        };
        self.dots(probabilities, values, dst, width.into(), tokens, |j, t| ((probabilities, t), (values, (offset + j) * capacity + t)));
    }

    fn rotate(&mut self, x: u32, swapped: u32, (cos, sin): (u32, u32), dst: u32, width: u32, rotary: u32) {
        if !self.combines {
            self.each(rotary.into(), |s, k| s.sum_products(k, &[((x, k), (cos, k)), ((swapped, k), (sin, k))]));
            self.each((width - rotary).into(), |s, k| {
                let at = k + rotary.into();
                s.sum_products(at, &[((x, at), (cos, at))]);
            });
            return self.narrow_sums(x, cos, dst, width.into());
        }
        let zero = self.sc(10);
        self.immediate(0, zero);
        let combine = |weights: u32, matrix: u32, length: u32, fresh: bool| Step::Combine { weights: if fresh { zero } else { (weights, 0.into()) }, matrix, start: 0.into(), width: length.into(), fresh, window: 0.into(), window_a: false, window_b: false, moving: !fresh };
        self.steps.push(combine(cos, x, width, true));
        self.steps.push(combine(cos, x, width, false));
        self.steps.push(combine(sin, swapped, rotary, false));
        self.narrow_sums(x, cos, dst, width.into());
    }

    fn each_head(&mut self, outer: u32, inner: u32, mut body: impl FnMut(&mut Self, Expr, Expr)) {
        if self.clusters <= 1 {
            return self.each(outer.into(), |s, o| s.each(inner.into(), |s, i| body(s, o, i)));
        }
        let (first, second) = (Var::Loop(self.loops), Var::Loop(self.loops + 1));
        self.loops += 2;
        let before = std::mem::take(&mut self.steps);
        body(self, Expr::var(first), Expr::var(second));
        let inside = std::mem::replace(&mut self.steps, before);
        let total = outer * inner;
        for block in 0..total.div_ceil(self.clusters) {
            let assignments = (0..self.clusters.min(total - block * self.clusters)).map(|unit| {
                let head = block * self.clusters + unit;
                vec![head / inner, head % inner]
            });
            self.steps.push(Step::Clustered { vars: vec![first, second], assignments: assignments.collect(), body: inside.clone() });
        }
    }

    fn gather_heads(&mut self, parts: &[(u32, u32)], dst: u32) {
        if self.clusters <= 1 {
            return self.gather(parts, dst);
        }
        self.clear_shifts();
        let zero = self.sc(15);
        let (largest, shift, exponent) = (self.sc(1), self.sc(2), self.sc(3));
        let unit_of = |part: usize, units: u32| part as u32 % units + 1;
        let units = self.clusters;
        for (part, (reg, _)) in parts.iter().enumerate() {
            let mut read = alu(OP_MOV);
            (read.a, read.b, read.dst, read.from_cluster) = (self.exponent(*reg), zero, exponent, unit_of(part, units));
            self.push(read);
            if part == 0 {
                self.binary(OP_MOV, exponent, zero, largest);
            } else {
                self.binary(OP_MAX, largest, exponent, largest);
            }
        }
        let mut offset = 0;
        for (part, (reg, len)) in parts.iter().enumerate() {
            let mut read = alu(OP_MOV);
            (read.a, read.b, read.dst, read.from_cluster) = (self.exponent(*reg), zero, exponent, unit_of(part, units));
            self.push(read);
            self.binary(OP_SUB, largest, exponent, shift);
            self.load_shifts(Some(shift), None, None);
            let at: Expr = offset.into();
            self.elements((*len).into(), &[at], true, |s, i, vector| {
                let mut step = alu(OP_MOV);
                (step.a, step.b, step.dst, step.from_cluster) = ((*reg, i), zero, (dst, at + i), unit_of(part, units));
                (step.vector, step.broadcast_b) = (vector, vector);
                s.push(step);
            });
            self.clear_shifts();
            offset += len;
        }
        self.binary(OP_MOV, largest, zero, self.exponent(dst));
        self.normalize(dst, offset.into());
    }

    fn depthwise(&mut self, taps: u32, history: u32, dst: u32, width: u32, kernel: u32) {
        if !self.combines || width < MIN_STREAM {
            return self.dots(taps, history, dst, width.into(), kernel.into(), |c, k| ((taps, k * width + c), (history, k * width + c)));
        }
        let zero = self.sc(10);
        self.immediate(0, zero);
        self.steps.push(Step::Combine { weights: zero, matrix: history, start: 0.into(), width: width.into(), fresh: true, window: 0.into(), window_a: false, window_b: false, moving: false });
        self.each(kernel.into(), |s, k| s.steps.push(Step::Combine { weights: (taps, k * width), matrix: history, start: k * width, width: width.into(), fresh: false, window: 0.into(), window_a: false, window_b: false, moving: true }));
        self.narrow_sums(taps, history, dst, width.into());
    }

    fn copy_elements(&mut self, reg: u32, from: Expr, to: Expr, width: u32) {
        self.elements(width.into(), &[from, to], true, |s, c, vector| {
            let mut step = alu(OP_MOV);
            (step.a, step.dst, step.vector) = ((reg, from + c), (reg, to + c), vector);
            s.push(step);
        });
    }

    fn store_key(&mut self, src: u32, exponent: i64, keys: u32, slot: Expr, width: u32) {
        let Some(capacity) = self.key_capacity else {
            return self.align_to(src, 0.into(), exponent, keys, slot * width, width.into());
        };
        // The tokens of one column sit in consecutive places, one for each lane.
        self.load_alignment(src, exponent);
        self.each(width.into(), |s, j| {
            let mut right = alu(OP_MOV);
            right.a = (src, j);
            right.dst = (keys, j * capacity + slot);
            s.push(right);
            let mut left = alu(OP_SHIFT_LEFT);
            left.a = right.dst;
            left.dst = right.dst;
            s.push(left);
        });
        self.clear_shifts();
    }

    fn score_keys(&mut self, query: u32, store: KeyStore<Expr>, dst: u32, width: u32, tokens: Expr) {
        let KeyStore { keys, stride, offset, .. } = store;
        let Some(capacity) = self.key_capacity else {
            return self.dots(query, keys, dst, tokens, width.into(), |t, j| ((query, j), (keys, t * stride + offset + j)));
        };
        let zero = self.sc(10);
        self.immediate(0, zero);
        self.steps.push(Step::Combine { weights: zero, matrix: keys, start: 0.into(), width: tokens, fresh: true, window: 0.into(), window_a: false, window_b: false, moving: false });
        self.each(width.into(), |s, j| s.steps.push(Step::Combine { weights: (query, j), matrix: keys, start: (offset + j) * capacity, width: tokens, fresh: false, window: 0.into(), window_a: false, window_b: false, moving: false }));
        self.narrow_sums(query, keys, dst, tokens);
    }

    fn align_to(&mut self, src: u32, from: Expr, exponent: i64, dst: u32, to: Expr, len: Expr) {
        self.load_alignment(src, exponent);
        let apart = src != dst || from == to;
        self.elements(len, &[from, to], true, |s, i, vector| {
            let mut right = alu(OP_MOV);
            right.a = (src, from + i);
            right.dst = (dst, to + i);
            right.vector = vector;
            s.push(right);
            if !apart {
                let mut left = alu(OP_SHIFT_LEFT);
                left.a = (dst, to + i);
                left.dst = (dst, to + i);
                left.vector = vector;
                s.push(left);
            }
        });
        if apart {
            self.elements(len, &[to], true, |s, i, vector| {
                let mut left = alu(OP_SHIFT_LEFT);
                left.a = (dst, to + i);
                left.dst = (dst, to + i);
                left.vector = vector;
                s.push(left);
            });
        }
        self.clear_shifts();
    }

    fn move_element(&mut self, from: At<Expr>, to: At<Expr>) {
        self.binary(OP_MOV, from, self.sc(15), to);
    }

    fn maximum(&mut self, src: u32, len: Expr, dst: u32) {
        self.clear_shifts();
        if self.streams {
            let mut reset = alu(OP_MOV);
            (reset.write, reset.max_clear) = (false, true);
            self.push(reset);
            self.elements(len, &[], true, |s, t, vector| {
                let mut step = alu(OP_MOV);
                (step.a, step.write, step.max_scan, step.vector) = ((src, t), false, true, vector);
                s.push(step);
            });
            let mut out = alu(OP_MOV);
            (out.dst, out.max_out) = ((dst, 0.into()), true);
            self.push(out);
            return self.binary(OP_MOV, self.exponent(src), self.sc(15), self.exponent(dst));
        }
        let first: At<Expr> = (dst, 0.into());
        self.binary(OP_MOV, (src, 0.into()), self.sc(15), first);
        // Starting from the first element again keeps the count from ever being zero.
        self.each(len, |s, t| s.binary(OP_MAX, first, (src, t), first));
        self.binary(OP_MOV, self.exponent(src), self.sc(15), self.exponent(dst));
    }

    fn mark_equal(&mut self, a: u32, b: u32, dst: u32, len: Expr) {
        self.clear_shifts();
        let mut clear = alu(OP_MOV);
        clear.magnitude_clear = true;
        clear.write = false;
        self.push(clear);
        self.each(len, |s, i| s.binary(OP_EQUAL, (a, i), (b, 0.into()), (dst, i)));
        self.immediate(0, self.exponent(dst));
    }

    fn set_exponent(&mut self, reg: u32, exponent: i64) {
        self.immediate(exponent, self.exponent(reg));
    }

    fn subtract_exponent(&mut self, reg: u32, amount: i64) {
        self.immediate(amount, self.sc(1));
        self.binary(OP_SUB, self.exponent(reg), self.sc(1), self.exponent(reg));
    }

    fn matvec(&mut self, name: &str, matrix: u32, first: Expr, src: u32, dst: u32, rows: u32, cols: u32) {
        let Some(place) = self.in_cells.get(name).copied() else {
            return self.dots(matrix, src, dst, rows.into(), cols.into(), |r, j| ((matrix, (r + first) * cols + j), (src, j)));
        };
        // A matrix used a block at a time keeps each block's rows from cell 0 on. Each
        // vector unit has its own cells, and takes its own slice of the columns.
        let units = self.clusters;
        let per_unit = self.cells / units;
        let groups = place.groups.div_ceil(units);
        let columns = groups * place.group;
        let passes = rows.div_ceil(per_unit);
        let block = first.divided_by(rows);
        self.steps.push(Step::CellSeek { weights: Expr::from(place.weights) + block * (passes * columns), factors: Expr::from(place.factors) + block * (passes * groups) });
        let column = Var::Loop(self.loops);
        self.loops += 1;
        let before = std::mem::take(&mut self.steps);
        self.each((rows / per_unit).into(), |s, _| s.cell_round(src, place, per_unit, Expr::var(column)));
        if !rows.is_multiple_of(per_unit) {
            self.cell_round(src, place, rows % per_unit, Expr::var(column));
        }
        let inside = std::mem::replace(&mut self.steps, before);
        self.steps.push(Step::Clustered { vars: vec![column], assignments: (0..units).map(|unit| vec![unit * columns]).collect(), body: inside });
        if units > 1 {
            self.steps.push(Step::WideReduce { rows });
        }
        self.narrow_sums(matrix, src, dst, rows.into());
    }

    fn copy_row(&mut self, _name: &str, matrix: u32, row: Expr, cols: u32, dst: u32) {
        self.copy_part(matrix, row * cols, dst, 0.into(), cols.into());
    }
}

/// Executes the steps directly on integers, with no reference to the circuit.
pub struct Behavior {
    lanes: u32,
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
    equal_seen: bool,
    accumulator: i128,
}

fn wrap(value: i128, bits: u32) -> i64 {
    let shift = 128 - bits;
    ((value << shift) >> shift) as i64
}

impl Behavior {
    pub fn new(c: &DieConfig, initial: &HashMap<Place, i64>) -> Self {
        Self {
            lanes: 1 << c.features.lane_bits,
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
            equal_seen: false,
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

    pub fn run(&mut self, steps: &[Step]) {
        self.run_with(steps, 0, 0);
    }

    /// Runs a program for one token at one position of the context.
    pub fn run_with(&mut self, steps: &[Step], token: u32, position: u32) {
        let mut loops = HashMap::new();
        self.walk(steps, token, position, &mut loops);
    }

    fn walk(&mut self, steps: &[Step], token: u32, position: u32, loops: &mut HashMap<Var, u32>) {
        for step in steps {
            let value = |loops: &HashMap<Var, u32>, e: Expr| e.eval(|v| match v {
                Var::Token => token,
                Var::Position => position,
                Var::WindowLength(n) => (position + 1).min(n),
                Var::WindowSlot(n) => position % n,
                loop_variable => loops[&loop_variable],
            });
            match step {
                Step::One(alu) => {
                    let bound = alu.map_indices(|e| value(loops, e));
                    // A block is the same thing as its elements one after another.
                    let (lanes, moved) = (if bound.vector { self.lanes } else { 1 }, |at: At<u32>, on: bool, j: u32| (at.0, at.1 + if on { j } else { 0 }));
                    for j in 0..lanes {
                        self.step(&Alu { a: moved(bound.a, !bound.broadcast_a, j), b: moved(bound.b, !bound.broadcast_b, j), dst: moved(bound.dst, true, j), ..bound });
                    }
                }
                Step::Repeat { variable, count, body } => {
                    for i in 0..value(loops, *count) {
                        loops.insert(*variable, i);
                        self.walk(body, token, position, loops);
                    }
                }
                Step::Stream { variable, count, body, .. } => {
                    for i in 0..value(loops, *count) {
                        loops.insert(*variable, i);
                        self.walk(&[Step::One(*body)], token, position, loops);
                    }
                }
                Step::Clustered { .. } | Step::Combine { .. } | Step::CellSeek { .. } | Step::CellPass | Step::CellGroup { .. } | Step::CellDrain { .. } | Step::WideReduce { .. } => panic!("este modelo no guarda los pesos de las celdas"),
            }
        }
    }

    fn step(&mut self, s: &Alu) {
        let stepped = |at: At<u32>, on: bool| if on { (at.0 + s.window, at.1) } else { at };
        let moved = Alu { a: stepped(s.a, s.window_a), b: stepped(s.b, s.window_b), dst: stepped(s.dst, s.window_dst), ..*s };
        let s = &moved;
        let bits = self.bits;
        if s.magnitude_clear {
            self.magnitude = 0;
            self.equal_seen = false;
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
        let aligned_a = round_right(a as i128, self.shift_a.min(63), bits);
        let aligned_b = round_right(b as i128, self.shift_b.min(63), bits);

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
                let width = accumulator_bits(bits) - 1;
                (width - (64 - (self.wide_magnitude & ((1 << width) - 1)).leading_zeros())) as i64
            }
            OP_WIDE_OUT => round_right(wide_value as i128, self.wide_shift.min(63), bits),
            OP_EQUAL => {
                if wrap(aligned_a as i128 - aligned_b as i128, bits) == 0 && !self.equal_seen {
                    self.equal_seen = true;
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
            let top = accumulator_bits(bits) - 1;
            let sign = (wide_value >> top) & 1;
            let mask = (1u64 << top) - 1;
            self.wide_magnitude |= if sign == 1 { !(wide_value as u64) & mask } else { wide_value as u64 & mask };
        }
        if s.accumulate {
            self.accumulator = wrap(self.accumulator + a as i128 * b as i128, accumulator_bits(bits)) as i128;
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
