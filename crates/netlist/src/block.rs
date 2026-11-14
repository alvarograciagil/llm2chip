//! Block floating point on the die: each vector register has one exponent.
//!
//! `BlockOps` describes the arithmetic once. `Assembly` turns it into microcode for
//! the gates, and a virtual machine elsewhere runs the same semantics at full speed.

use crate::die::{OP_ADD, OP_EQUAL, OP_FUNCTION, OP_IMMEDIATE, OP_LINK, OP_LZC, OP_MAX, OP_MOV, OP_MUL, OP_PARITY, OP_SHIFT_LEFT, OP_SUB, OP_WIDE_LZC, OP_WIDE_OUT};
use crate::DieConfig;
use std::collections::{HashMap, HashSet};

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
    (0..1usize << crate::die::function_address_bits(c))
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
    /// Room for the rows of the key and the query that one unit keeps, when a head is shared.
    pub key_rows: u32,
    pub query_rows: u32,
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
        assert!(self.constant.is_multiple_of(by), "{} is not a multiple of {by}", self.constant);
        let mut terms = self.terms;
        for (_, coefficient) in terms.iter_mut().flatten() {
            assert!(coefficient.is_multiple_of(by), "{coefficient} is not a multiple of {by}");
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
        assert!(merged.len() <= EXPR_TERMS, "index too complex for the sequencer");
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
    /// `to` becomes the exponent of `reg`, to keep it beside a row stored elsewhere.
    fn save_exponent(&mut self, reg: u32, to: At<Self::N>);
    /// `reg` takes the exponent kept at `from`.
    fn restore_exponent(&mut self, from: At<Self::N>, reg: u32);
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
    /// Several whole matrices applied to the same `src`, each into its own `dst`. A die
    /// with cells runs them in one pass when they lie one after another there.
    fn matvecs(&mut self, parts: &[MatvecPart], src: u32, cols: u32) {
        for part in parts {
            self.matvec(&part.name, part.matrix, 0.into(), src, part.dst, part.rows, cols);
        }
    }
    /// Copies one row of an embedding-like matrix into `dst`.
    fn copy_row(&mut self, name: &str, matrix: u32, row: Self::N, cols: u32, dst: u32);
    /// The `rows` rows of a product that this chip of a ring holds, into `dst` from
    /// `offset`. Only a die with cells shares products among chips.
    fn matvec_rows(&mut self, name: &str, matrix: u32, src: u32, dst: u32, rows: u32, offset: u32) {
        let _ = (matrix, src, dst, rows, offset);
        unimplemented!("{name}: only a die with cells splits the products among chips")
    }
    /// Like `matvec_rows` for products run in one pass: `(matrix, dst, rows, offset)`.
    fn matvecs_rows(&mut self, first: &str, parts: &[(u32, u32, u32, u32)], src: u32) {
        let _ = (parts, src);
        unimplemented!("{first}: only a die with cells splits the products among chips")
    }
    /// Gives every chip of the ring the blocks of `reg` the others made.
    fn all_gather(&mut self, reg: u32, block: u32, index: u32) {
        let _ = (reg, block, index);
        unimplemented!("only a die with cells splits the products among chips")
    }
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
        self.rms_normalize_part(x, dst, (len, len), temporary, inverse_len, epsilon);
    }

    /// `rms_normalize` on a slice of `part` elements of a vector of `len`.
    fn rms_normalize_part(&mut self, x: u32, dst: u32, (part, len): (u32, u32), temporary: (u32, u32), inverse_len: Option<u32>, epsilon: Option<u32>) {
        self.dot(x, x, temporary.0, part.into());
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
        self.invert_root_and_scale(x, dst, part, (mean, spare), epsilon);
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

    /// Inside `each_head` or `each_head_split`: whether every head has a unit of its own in
    /// a single round, so a state kept from one token to the next can sit in the same
    /// register on each unit rather than one register per head.
    fn one_head_per_unit(&self) -> bool {
        false
    }

    /// Like `each_head`, for a body that can share each head's work among several units.
    fn each_head_split(&mut self, outer: u32, inner: u32, body: impl FnMut(&mut Self, Self::N, Self::N)) {
        self.each_head(outer, inner, body);
    }

    /// Runs `body` on slices of a vector of `len` elements, one per unit where the die has
    /// several, and collects `outputs` whole in every unit afterwards. The body gets the
    /// start of its slice and its length.
    fn in_slabs(&mut self, len: u32, outputs: &[u32], mut body: impl FnMut(&mut Self, Self::N, u32)) {
        let _ = outputs;
        body(self, 0.into(), len);
    }

    /// Like `gather`, for parts that `each_head` left on the units that made them.
    fn gather_heads(&mut self, parts: &[(u32, u32)], dst: u32) {
        self.gather(parts, dst);
    }

    /// The next operation that writes `reg` a slice per unit may leave it that way: what
    /// reads it next takes the same slices. Elsewhere the whole vector is there anyway.
    fn leave_spread(&mut self, reg: u32) {
        let _ = reg;
    }

    /// At the end of a head, inside `each_head`: a die whose units can agree on the
    /// largest exponent brings the head to it there, all heads at once, so that
    /// `gather_heads` only has to copy. Elsewhere `gather_heads` does it, to the same bits.
    fn align_heads(&mut self, reg: u32, len: u32) {
        let _ = (reg, len);
    }

    /// `dst[c] = Σ taps[k · width + c] · history[k · width + c]` over `kernel` rows.
    fn depthwise(&mut self, taps: u32, history: u32, dst: u32, width: u32, kernel: u32) {
        self.dots(taps, history, dst, width.into(), kernel.into(), |c, k| ((taps, k * width + c), (history, k * width + c)));
    }

    /// `depthwise` over the `len` channels from `start` of a convolution `width` wide.
    /// Only a die that splits the channels among its units passes a start other than 0.
    #[allow(clippy::too_many_arguments)]
    fn depthwise_part(&mut self, taps: u32, history: u32, dst: u32, width: u32, kernel: u32, start: Self::N, len: u32) {
        let _ = (start, len);
        self.depthwise(taps, history, dst, width, kernel);
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

    /// `store_key` for the `len` elements from `from` of a key `width` wide. A die that
    /// splits a head among units writes in each only the columns that unit reads.
    #[allow(clippy::too_many_arguments)]
    fn store_key_part(&mut self, src: u32, exponent: i64, keys: u32, slot: Self::N, width: u32, from: Self::N, len: u32) {
        self.align_to(src, from, exponent, keys, slot * width + from, len.into());
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
        whole_delta_step(self, r, key, value);
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
    /// The shifters load straight from the operands read: A and left from A, B from B.
    pub shift_now: bool,
    pub load_wide_shift: bool,
    pub magnitude_scan: bool,
    /// A count of leading zeros looks at the magnitude of `1 << reduce_group` neighbouring units.
    pub reduce_group: u32,
    /// Keeps the largest mantissa read, across every lane of a block.
    pub max_scan: bool,
    pub max_clear: bool,
    /// The result is that largest mantissa.
    pub max_out: bool,
    /// The destination index counts in whole columns of the key store.
    pub write_spread: bool,
    /// Operand A comes from the unit that owns each slice of `1 << gather_shift` elements.
    pub gather_shift: u32,
    pub magnitude_clear: bool,
    pub accumulate: bool,
    pub accumulator_clear: bool,
    pub link_send: bool,
    /// The word sends or takes over the loop back, not the link to the next chip: the last
    /// and first chips of a group close their ring with it.
    pub link_loop: bool,
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
            shift_now: false,
            load_wide_shift: false,
            magnitude_scan: false,
            reduce_group: 0,
            max_scan: false,
            max_clear: false,
            max_out: false,
            write_spread: false,
            gather_shift: 0,
            magnitude_clear: false,
            accumulate: false,
            accumulator_clear: false,
            link_send: false,
            link_loop: false,
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
            shift_now: self.shift_now,
            load_wide_shift: self.load_wide_shift,
            magnitude_scan: self.magnitude_scan,
            reduce_group: self.reduce_group,
            max_scan: self.max_scan,
            max_clear: self.max_clear,
            max_out: self.max_out,
            write_spread: self.write_spread,
            gather_shift: self.gather_shift,
            magnitude_clear: self.magnitude_clear,
            accumulate: self.accumulate,
            accumulator_clear: self.accumulator_clear,
            link_send: self.link_send,
            link_loop: self.link_loop,
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
        out.shift_now = s.shift_now;
        out.load_wide_shift = s.load_wide_shift;
        out.magnitude_scan = s.magnitude_scan;
        out.reduce_group = s.reduce_group;
        (out.max_scan, out.max_clear, out.max_out, out.write_spread, out.gather_shift) = (s.max_scan, s.max_clear, s.max_out, s.write_spread, s.gather_shift);
        out.magnitude_clear = s.magnitude_clear;
        out.accumulate = s.accumulate;
        out.accumulator_clear = s.accumulator_clear;
        out.link_send = s.link_send;
        out.link_loop = s.link_loop;
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
    /// `wide[j] = Σ_i weights[i] · matrix[start + i · lanes + j]` over `rows` rows of one
    /// block each, one row a cycle, from zero.
    CombineRows { weights: At<Expr>, matrix: u32, start: Expr, rows: u32, window: Expr, window_a: bool, window_b: bool },
    /// `wide[j] += weights · matrix[start + j]` for every `j` of a row, or from zero if `fresh`.
    Combine { weights: At<Expr>, matrix: u32, start: Expr, width: Expr, fresh: bool, window: Expr, window_a: bool, window_b: bool, moving: bool },
    /// One copy of `body` on several vector units, each with its own values of `vars`.
    /// With `slots = (shift, bits, limit)`, only the units whose number has
    /// `(k >> shift) mod 2^bits < limit` take part: the others fill out a power of two.
    Clustered { vars: Vec<Var>, assignments: Vec<Vec<u32>>, slots: Option<(u32, u32, u32)>, body: Vec<Step> },
    /// Moves every cell to where a matrix starts.
    /// `packing` is log2 of the codes the matrix keeps in each word.
    CellSeek { weights: Expr, factors: Expr, packing: u32 },
    /// Clears the row totals and restarts the input vector.
    CellPass,
    /// One group of columns through the cells.
    /// With `quant_bits`, the group is part of a quantisation group of that many bits of
    /// columns; `last` marks a unit's last group of the pass.
    CellGroup { src: u32, length: u32, start: Expr, quant_bits: u32, last: bool },
    /// The totals of all the vector units are added together in their wide memories.
    WideReduce { rows: Expr, group: u32 },
    /// The totals of the first `rows` cells go to the wide memory.
    CellDrain { rows: u32 },
}

/// The largest value each variable of a microprogram can take while it runs.
#[derive(Clone, Debug)]
pub struct VarRanges {
    pub token: u32,
    pub position: u32,
    loops: HashMap<Var, u32>,
}

impl VarRanges {
    /// Tokens up to `tokens - 1` at positions up to `positions - 1`.
    pub fn new(tokens: u32, positions: u32) -> Self {
        Self { token: tokens.saturating_sub(1), position: positions.saturating_sub(1), loops: HashMap::new() }
    }

    fn largest(&self, variable: Var) -> u32 {
        match variable {
            Var::Token => self.token,
            Var::Position => self.position,
            Var::WindowLength(window) => window,
            Var::WindowSlot(window) => window.saturating_sub(1),
            // A loop the steps do not enclose cannot run: none appears outside its own.
            Var::Loop(_) => self.loops.get(&variable).copied().unwrap_or(0),
        }
    }

    /// The largest value an index takes. Coefficients wrap like the address registers, so
    /// one past half the range is a subtraction, largest when its variable is zero.
    pub fn largest_of(&self, index: Expr) -> u32 {
        let total = index.variables().fold(index.constant() as i32 as i64, |sum, (variable, coefficient)| {
            let coefficient = coefficient as i32 as i64;
            sum + if coefficient > 0 { coefficient * self.largest(variable) as i64 } else { 0 }
        });
        total.clamp(0, u32::MAX as i64) as u32
    }
}

/// How far into each register `steps` reach: one past the largest index any of them can
/// touch, whole blocks included, for every value their variables can take. It never
/// falls short; it may reach a little further than a run ever does.
/// Registers a loop reaches by adding to a register's number go in `windows`: the first,
/// and how far past it the loop goes.
pub fn register_extents(steps: &[Step], ranges: &mut VarRanges, lanes: u32, key_shift: u32, extents: &mut HashMap<u32, u32>, windows: &mut HashMap<u32, u32>) {
    let mut reach = Reach { lanes, key_shift, extents, windows };
    reach.steps(steps, ranges);
}

/// What `register_extents` gathers while it walks the steps.
struct Reach<'a> {
    lanes: u32,
    key_shift: u32,
    extents: &'a mut HashMap<u32, u32>,
    windows: &'a mut HashMap<u32, u32>,
}

impl Reach<'_> {
    fn touch(&mut self, ranges: &VarRanges, (reg, index): At<Expr>, span: u32, window: Option<Expr>) {
        let end = ranges.largest_of(index).saturating_add(span);
        let registers = window.map_or(0, |w| ranges.largest_of(w));
        if window.is_some() {
            let reach = self.windows.entry(reg).or_default();
            *reach = (*reach).max(registers);
        }
        for r in reg..=reg.saturating_add(registers) {
            let entry = self.extents.entry(r).or_default();
            *entry = (*entry).max(end);
        }
    }

    /// `columns` is how many columns a write to the key store steps through: the die
    /// moves on a column, `1 << key_shift` places, for each element it writes.
    fn alu(&mut self, ranges: &VarRanges, s: &Alu<Expr>, columns: u32) {
        let span = |whole: bool| if s.vector && !whole { self.lanes } else { 1 };
        let (span_a, span_b, span_dst) = (span(s.broadcast_a), span(s.broadcast_b), span(false));
        let window = |on: bool| on.then_some(s.window);
        self.touch(ranges, s.a, span_a, window(s.window_a));
        self.touch(ranges, s.b, span_b, window(s.window_b));
        let reach = if s.write_spread { span_dst.max(1) + (columns.saturating_sub(1) << self.key_shift) } else { span_dst };
        self.touch(ranges, s.dst, reach, window(s.window_dst));
    }

    fn steps(&mut self, steps: &[Step], ranges: &mut VarRanges) {
        let lanes = self.lanes;
        for step in steps {
            match step {
                Step::One(s) => self.alu(ranges, s, 1),
                Step::Repeat { variable, count, body } => {
                    let last = ranges.largest_of(*count).saturating_sub(1);
                    ranges.loops.insert(*variable, last);
                    self.steps(body, ranges);
                }
                Step::Stream { variable, count, in_elements, body } => {
                    // A stream of blocks steps once a block over `count` elements.
                    let count = ranges.largest_of(*count);
                    let last = if *in_elements { count.div_ceil(lanes) } else { count }.saturating_sub(1);
                    ranges.loops.insert(*variable, last);
                    self.alu(ranges, body, count);
                }
                Step::Clustered { vars, assignments, body, .. } => {
                    for (i, variable) in vars.iter().enumerate() {
                        let largest = assignments.iter().filter_map(|unit| unit.get(i)).copied().max().unwrap_or(0);
                        ranges.loops.insert(*variable, largest);
                    }
                    self.steps(body, ranges);
                }
                Step::CombineRows { weights, matrix, start, rows, window, window_a, window_b } => {
                    self.touch(ranges, *weights, *rows, window_a.then_some(*window));
                    self.touch(ranges, (*matrix, *start), rows * lanes, window_b.then_some(*window));
                }
                Step::Combine { weights, matrix, start, width, window, window_a, window_b, .. } => {
                    self.touch(ranges, *weights, 1, window_a.then_some(*window));
                    let width = ranges.largest_of(*width);
                    self.touch(ranges, (*matrix, *start), width.next_multiple_of(lanes), window_b.then_some(*window));
                }
                Step::CellGroup { src, length, start, .. } => self.touch(ranges, (*src, *start), length.next_multiple_of(lanes), None),
                Step::CellSeek { .. } | Step::CellPass | Step::WideReduce { .. } | Step::CellDrain { .. } => {}
            }
        }
    }
}

/// Where the attention of one head reads its keys and values: each token's row starts
/// `stride` further on, and this head's part of it `offset` into the row.
#[derive(Clone, Copy, Debug)]
pub struct KeyStore<N> {
    pub keys: u32,
    pub values: u32,
    pub stride: u32,
    pub offset: N,
    /// The keys were written by `store_key_part` inside the same loop over heads: a die that
    /// gives each unit its own heads keeps there only those heads' columns.
    pub local: bool,
}

/// Below this a loop is not worth the words it takes to set up a stream.
const MIN_STREAM: u32 = 4;

/// Fractional bits of the scale and bias the cells multiply by.
pub const FACTOR_FRACTION: u32 = 4;

/// One matrix of those `BlockOps::matvecs` applies together.
#[derive(Clone, Debug)]
pub struct MatvecPart {
    pub name: String,
    pub matrix: u32,
    pub dst: u32,
    pub rows: u32,
}

/// Where each of several matrices starts when their rows follow one another in the cells,
/// each on a block boundary, and how many rows they take together.
pub fn chained_rows(rows: &[u32], lanes: u32) -> (Vec<u32>, u32) {
    let mut starts = Vec::with_capacity(rows.len());
    let mut end = 0u32;
    for r in rows {
        let start = end.next_multiple_of(lanes);
        starts.push(start);
        end = start + r;
    }
    (starts, end)
}

/// Where the cells keep a matrix.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CellMatrix {
    pub weights: u32,
    pub factors: u32,
    pub group: u32,
    pub groups: u32,
    /// Non-zero when each group is part of a quantisation group of `1 << quant_bits`
    /// columns, so the columns divide evenly among the units.
    pub quant_bits: u32,
    /// log2 of the codes each word of the cells holds for this matrix.
    pub packing: u32,
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
    /// The link to the next chip carries a block of lanes a cycle.
    wide_link: bool,
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
    key_shift: u32,
    /// Inside a loop over heads split among units: this unit's part and how many there are.
    split: Option<(Expr, u32)>,
    /// Inside a loop over heads run by several units: which round of heads this is. Each
    /// unit keeps the key columns of its own heads only, one round after another.
    head_round: Option<Expr>,
    /// Inside slices of one vector: the units counted together when normalising.
    slab_group: u32,
    /// True inside a body that every unit runs with its own indices.
    in_units: bool,
    /// How many neighbouring units the last loop over heads gave each head.
    /// The unit that holds each head `each_head` laid out, for `gather_heads`.
    head_units: Vec<u32>,
    /// Which of the A, B and left shifters are known to hold zero.
    shift_zero: [bool; 3],
    /// The die loads its shifters from the read ports, as described in `load_shifts_now`.
    shifts_now: bool,
    /// The die adds the units' totals as their chains drain, as `DieProgram::drain_into` does.
    drains_add: bool,
    /// A stream of blocks can take one element of a scalar operand for each block, and pile
    /// rows of one block up in the wide memory.
    rows_stream: bool,
    /// Every head of the `each_head` being built has a unit of its own at once.
    heads_at_once: bool,
    /// Bits of an immediate, which rides in the word's jump target.
    immediate_bits: u32,
    /// Registers the next slices among the units may leave where they were made.
    spread: HashSet<u32>,
    /// Sets of heads brought to one exponent where they were made, which the next as many
    /// `gather_heads` only copy: see `align_heads`.
    heads_aligned: u32,
    zero_ready: bool,
    /// With keys kept in columns, how many tokens each column has room for.
    pub key_capacity: Option<u32>,
    /// Matrices the cells hold. Any other is computed from the vector registers.
    pub in_cells: HashMap<String, CellMatrix>,
    /// How many chips around the ring share each product by rows: their narrowing has to
    /// agree on the shifts, which each would otherwise take from its own rows alone.
    pub ring: u32,
    /// The ends of a group's ring when the chips are in several groups: the last sends its
    /// ring's steps over the loop back, and the first takes them from it.
    pub ring_first: bool,
    pub ring_last: bool,
    across_chips: bool,
}

/// Words a chip sends before it takes as many: each link holds twice as many in its queue,
/// so chips that all send at once never wait on one another.
const EXCHANGE_CHUNK: u32 = 4;

impl Assembly {
    /// The units whose values count together inside the current loop.
    /// Heads to a group of `inner` laid out `width` apart: a power of two when that costs no
    /// units, so every index the units need is linear in the bits of their number, and the
    /// units in between sit out.
    fn head_slots(&self, outer: u32, inner: u32, parts: u32) -> (u32, Option<(u32, u32, u32)>) {
        let padded = inner.next_power_of_two();
        if padded != inner && outer > 1 && outer * padded * parts <= self.clusters {
            (padded, Some((parts.trailing_zeros(), padded.trailing_zeros(), inner)))
        } else {
            (inner, None)
        }
    }

    /// The round of heads whose columns `store` keeps from the first word of each unit, when
    /// `store_key_part` wrote them that way: keys in columns, written whole columns at a time,
    /// inside a loop over heads that the units share.
    fn local_columns(&self, store: &KeyStore<Expr>) -> Option<Expr> {
        self.head_round.filter(|_| store.local && self.key_shift > 0 && self.key_capacity.is_some())
    }

    fn group_bits(&self) -> u32 {
        self.split.map_or(0, |(_, parts)| parts.trailing_zeros()).max(self.slab_group)
    }

    /// Normalises `len` elements of `reg` from `from`, sharing its one exponent.
    fn normalize_from(&mut self, reg: u32, from: Expr, len: Expr) {
        self.clear_shifts();
        self.scan_and_shift(reg, from, len);

        let mut count = alu(OP_LZC);
        count.dst = self.sc(0);
        // Units sharing a head look at all their parts, so they all shift alike.
        count.reduce_group = self.group_bits();
        if self.across_chips {
            // The chips agree on the smallest count, as the maximum of its negation.
            self.push(count);
            self.immediate(0, self.sc(1));
            self.binary(OP_SUB, self.sc(1), self.sc(0), self.sc(2));
            self.ring_max(self.sc(2), self.sc(3));
            self.binary(OP_SUB, self.sc(1), self.sc(2), self.sc(0));
            self.load_shifts(None, None, Some(self.sc(0)));
        } else if self.shifts_now {
            // The count goes into the left shifter as it is written.
            (count.load_shift_left, count.shift_now) = (true, true);
            self.shift_zero[2] = false;
            self.push(count);
        } else {
            self.push(count);
            self.load_shifts(None, None, Some(self.sc(0)));
        }
        self.elements(len, &[from], true, |s, i, vector| {
            let mut shift = alu(OP_SHIFT_LEFT);
            shift.a = (reg, from + i);
            shift.dst = (reg, from + i);
            shift.vector = vector;
            s.push(shift);
        });
        self.binary(OP_SUB, self.exponent(reg), self.sc(0), self.exponent(reg));
        self.clear_shifts();
    }

    /// Narrows `sets` wide sums from `from` into the same places of `dst`.
    fn narrow_part(&mut self, a: u32, b: u32, dst: u32, from: Expr, sets: Expr) {
        self.narrow_part_at(a, b, dst, from, from, sets);
    }

    /// Like `narrow_part`, with the sums taken from `wide` on in the wide memory.
    fn narrow_part_at(&mut self, a: u32, b: u32, dst: u32, wide: Expr, from: Expr, sets: Expr) {
        let mut clear = alu(OP_MOV);
        clear.write = false;
        clear.wide_clear = true;
        self.push(clear);
        self.elements(sets, &[wide, from], true, |s, set, vector| {
            let mut scan = alu(OP_MOV);
            scan.a = (0, wide + set);
            scan.write = false;
            scan.wide_scan = true;
            scan.vector = vector;
            s.push(scan);
        });
        let mut count = alu(OP_WIDE_LZC);
        count.dst = self.sc(0);
        count.reduce_group = self.group_bits();
        self.push(count);
        self.immediate(accumulator_bits(self.bits) as i64 - self.bits as i64, self.sc(1));
        self.binary(OP_SUB, self.sc(1), self.sc(0), self.sc(2));
        if self.across_chips {
            self.ring_max(self.sc(2), self.sc(3));
        }
        self.immediate(0, self.sc(3));
        if self.shifts_now {
            // The shift goes into the wide shifter as it is written.
            let mut most = alu(OP_MAX);
            (most.a, most.b, most.dst, most.load_wide_shift, most.shift_now) = (self.sc(2), self.sc(3), self.sc(4), true, true);
            self.push(most);
        } else {
            self.binary(OP_MAX, self.sc(2), self.sc(3), self.sc(4));
            let mut load = alu(OP_MOV);
            load.a = self.sc(4);
            load.write = false;
            load.load_wide_shift = true;
            self.push(load);
        }
        self.elements(sets, &[wide, from], true, |s, set, vector| {
            let mut out = alu(OP_WIDE_OUT);
            out.a = (0, wide + set);
            out.dst = (dst, from + set);
            out.vector = vector;
            s.push(out);
        });
        self.binary(OP_ADD, self.exponent(a), self.exponent(b), self.sc(5));
        self.binary(OP_ADD, self.sc(5), self.sc(4), self.sc(6));
        self.immediate(1 - self.bits as i64, self.sc(7));
        self.binary(OP_ADD, self.sc(6), self.sc(7), self.exponent(dst));
        self.normalize_from(dst, from, sets);
    }

    /// `dst[j] = Σ weights[i] · matrix[i · width + j]` over this unit's `rows`, added
    /// across a group of `1 << group` units.
    fn combine_rows(&mut self, weights: u32, matrix: u32, dst: u32, (width, rows): (u32, u32), group: u32) {
        if self.rows_stream && width == 1 << self.lane_bits && rows > 1 {
            // Rows of one block go through one a cycle, adding up in a single block.
            self.steps.push(Step::CombineRows { weights: (weights, 0.into()), matrix, start: 0.into(), rows, window: 0.into(), window_a: false, window_b: false });
            self.steps.push(Step::WideReduce { rows: width.into(), group });
            return self.narrow_sums(weights, matrix, dst, width.into());
        }
        let zero = self.sc(10);
        self.immediate(0, zero);
        self.steps.push(Step::Combine { weights: zero, matrix, start: 0.into(), width: width.into(), fresh: true, window: 0.into(), window_a: false, window_b: false, moving: false });
        self.each(rows.into(), |s, i| s.steps.push(Step::Combine { weights: (weights, i), matrix, start: i * width, width: width.into(), fresh: false, window: 0.into(), window_a: false, window_b: false, moving: false }));
        self.steps.push(Step::WideReduce { rows: width.into(), group });
        self.narrow_sums(weights, matrix, dst, width.into());
    }

    pub fn new(c: &DieConfig) -> Self {
        Self { steps: Vec::new(), initial: HashMap::new(), wide_link: c.features.wide_link && c.features.lane_bits > 0, bits: c.cell.activation_bits, index_bits: c.index_bits, exponents: exponent_register(c), scratch: scratch_register(c), loops: 0, lane_bits: c.features.lane_bits, cells: c.cells, streams: c.features.streams(), combines: c.features.combines(), clusters: 1 << c.features.cluster_bits, key_shift: c.features.key_shift, split: None, head_round: None, slab_group: 0, in_units: false, head_units: Vec::new(), shift_zero: [false; 3], shifts_now: c.features.shifts_now(), drains_add: c.features.feeds_cells(), rows_stream: c.features.streams_rows() && c.features.combines(), heads_at_once: false, immediate_bits: c.index_bits.min(c.microcode_bits), spread: HashSet::new(), heads_aligned: 0, zero_ready: false, key_capacity: None, in_cells: HashMap::new(), ring: 1, ring_first: false, ring_last: false, across_chips: false }
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
        let bits = self.immediate_bits;
        let limit = (1i64 << (bits - 1)) - 1;
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
        let group = |last: bool| Step::CellGroup { src, length: place.group, start: column, quant_bits: place.quant_bits, last };
        if groups <= 16 {
            for g in 0..groups {
                self.steps.push(group(g + 1 == groups));
            }
        } else {
            self.each((groups - 1).into(), |s, _| s.steps.push(group(false)));
            self.steps.push(group(true));
        }
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
                if can_stream(variable, step, lanes, false) {
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
        if can_stream(variable, &step, unit, self.rows_stream) {
            *self.steps.last_mut().expect("loop") = Step::Stream { variable, count, in_elements: false, body: step };
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
        if self.shifts_now {
            return self.load_shifts_now([a, b, left]);
        }
        // A loaded value passes through the A shifter, so A must be zero while the others
        // load. A register already known to be zero is not loaded again.
        let wanted = [a, b, left];
        let needed: Vec<bool> = (0..3).map(|i| wanted[i].is_some() || !self.shift_zero[i]).collect();
        if !needed.iter().any(|n| *n) {
            return;
        }
        let zero = self.zero_place();
        if !self.shift_zero[0] && (needed[1] || needed[2] || wanted[0].is_some()) {
            self.load_shift(0, zero);
        }
        for which in [2, 1, 0] {
            if needed[which] && !(wanted[which].is_none() && self.shift_zero[which]) {
                self.load_shift(which, wanted[which].unwrap_or(zero));
                self.shift_zero[which] = wanted[which].is_none();
            }
        }
    }

    /// Loads straight from the read ports: B beside A or left in the same step, and
    /// nothing has to be cleared first.
    fn load_shifts_now(&mut self, wanted: [Option<At<Expr>>; 3]) {
        let needed: Vec<bool> = (0..3).map(|i| wanted[i].is_some() || !self.shift_zero[i]).collect();
        if !needed.iter().any(|n| *n) {
            return;
        }
        let mut places = [None; 3];
        for which in 0..3 {
            if needed[which] {
                places[which] = Some(wanted[which].unwrap_or_else(|| self.zero_place()));
            }
        }
        // Port A carries A and left, together when they load the same element.
        let mut port_a: Vec<(At<Expr>, bool, bool)> = Vec::new();
        match (places[0], places[2]) {
            (Some(x), Some(y)) if x == y => port_a.push((x, true, true)),
            (x, y) => {
                port_a.extend(x.map(|x| (x, true, false)));
                port_a.extend(y.map(|y| (y, false, true)));
            }
        }
        let mut port_b = places[1];
        for (place, load_a, load_left) in port_a {
            let mut s = alu(OP_MOV);
            (s.a, s.load_shift_a, s.load_shift_left) = (place, load_a, load_left);
            if let Some(b) = port_b.take() {
                (s.b, s.load_shift_b) = (b, true);
            }
            (s.write, s.shift_now) = (false, true);
            self.push(s);
        }
        if let Some(b) = port_b {
            let mut s = alu(OP_MOV);
            (s.b, s.load_shift_b, s.write, s.shift_now) = (b, true, false, true);
            self.push(s);
        }
        for which in 0..3 {
            if needed[which] {
                self.shift_zero[which] = wanted[which].is_none();
            }
        }
    }

    fn load_shift(&mut self, which: usize, place: At<Expr>) {
        let mut s = alu(OP_MOV);
        s.a = place;
        match which {
            0 => s.load_shift_a = true,
            1 => s.load_shift_b = true,
            _ => s.load_shift_left = true,
        }
        s.write = false;
        self.push(s);
        self.shift_zero[which] = true;
    }

    /// An element holding zero, written the first time it is needed.
    fn zero_place(&mut self) -> At<Expr> {
        if !self.zero_ready {
            self.immediate(0, self.sc(15));
            self.zero_ready = true;
        }
        self.sc(15)
    }

    /// Loops repeat their body, so what it leaves in the shifters is not known before it.
    fn forget_shifts(&mut self) {
        self.shift_zero = [false; 3];
    }

    fn clear_shifts(&mut self) {
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
        // Units that each summed a slice store the total of all of them.
        write.reduce_group = self.slab_group;
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

    /// Sends a register's exponent and its first `len` values to the next chip: a block
    /// of lanes a cycle over a wide link.
    pub fn send(&mut self, reg: u32, len: Expr) {
        self.clear_shifts();
        self.send_place(self.exponent(reg));
        if self.wide_link {
            self.elements(len, &[0.into()], true, |s, i, vector| {
                s.send_place((reg, i));
                if let Some(Step::One(word)) = s.steps.last_mut() {
                    word.vector = vector;
                }
            });
        } else {
            self.each(len, |s, i| s.send_place((reg, i)));
        }
    }

    /// Fills a register with what `send` delivered, in the same pieces.
    pub fn receive(&mut self, reg: u32, len: Expr) {
        self.receive_place(self.exponent(reg));
        if self.wide_link {
            self.elements(len, &[0.into()], true, |s, i, vector| {
                s.receive_place((reg, i));
                if let Some(Step::One(word)) = s.steps.last_mut() {
                    word.vector = vector;
                }
            });
        } else {
            self.each(len, |s, i| s.receive_place((reg, i)));
        }
    }

    /// The largest `value` among the chips of the ring, in every one of them: each passes
    /// on the largest it has seen, and after one turn less than there are chips, all have it.
    fn ring_max(&mut self, value: At<Expr>, spare: At<Expr>) {
        for _ in 1..self.ring {
            self.send_place(value);
            self.on_ring_send();
            self.receive_place(spare);
            self.on_ring_receive();
            self.binary(OP_MAX, value, spare, value);
        }
    }

    /// The last word sends to the next chip of the ring: over the loop back if this chip
    /// closes its group's ring.
    fn on_ring_send(&mut self) {
        if let (true, Some(Step::One(word))) = (self.ring_last, self.steps.last_mut()) {
            word.link_loop = true;
        }
    }

    /// The last word takes from the chip before in the ring: over the loop back if this
    /// chip opens its group's ring.
    fn on_ring_receive(&mut self) {
        if let (true, Some(Step::One(word))) = (self.ring_first, self.steps.last_mut()) {
            word.link_loop = true;
        }
    }

    /// The rows of a product this chip holds, `rows` of them written into `dst` from
    /// `offset`, narrowed with the same shifts as the whole product on one chip.
    fn ring_matvec_rows(&mut self, name: &str, matrix: u32, src: u32, dst: u32, rows: u32, offset: u32) {
        let place = *self.in_cells.get(name).unwrap_or_else(|| panic!("{name} is split by rows and must be in the cells"));
        self.cell_matvec(place, 0.into(), src, rows);
        self.across_chips = true;
        self.narrow_part_at(matrix, src, dst, 0.into(), offset.into(), rows.into());
        self.across_chips = false;
    }

    /// Several products of one vector that run in a single pass of the cells, each part
    /// with the rows this chip holds: `(matrix, dst, rows, offset in dst)`.
    fn ring_matvecs_rows(&mut self, first: &str, parts: &[(u32, u32, u32, u32)], src: u32) {
        let place = *self.in_cells.get(first).unwrap_or_else(|| panic!("{first} is split by rows and must be in the cells"));
        let rows: Vec<u32> = parts.iter().map(|p| p.2).collect();
        let (starts, total) = chained_rows(&rows, 1 << self.lane_bits);
        self.cell_matvec(place, 0.into(), src, total);
        self.across_chips = true;
        for (&(matrix, dst, rows, offset), start) in parts.iter().zip(starts) {
            self.narrow_part_at(matrix, src, dst, start.into(), offset.into(), rows.into());
        }
        self.across_chips = false;
    }

    /// Every chip of the ring made `block` elements of `reg`, the one at `index` those from
    /// `index * block`: after a turn less than there are chips, each has them all. In each
    /// turn a chip passes on the block it received in the turn before.
    fn ring_all_gather(&mut self, reg: u32, block: u32, index: u32) {
        let chips = self.ring;
        for turn in 0..chips - 1 {
            let sent = (index + chips - turn) % chips;
            let received = (index + 2 * chips - turn - 1) % chips;
            self.exchange(reg, sent * block, received * block, block);
        }
    }

    /// Sends `len` elements of `reg` from `sent` to the next chip while it takes as many
    /// from the chip before into `received`, a few at a time each way.
    fn exchange(&mut self, reg: u32, sent: u32, received: u32, len: u32) {
        self.clear_shifts();
        let unit = if self.wide_link { 1 << self.lane_bits } else { 1 };
        assert!(len.is_multiple_of(unit), "the blocks the chips share take whole blocks of the link");
        let words = len / unit;
        let chunk = EXCHANGE_CHUNK.min(words);
        let pass = |s: &mut Self, first: Expr, count: u32| {
            for k in 0..count {
                let at = first + Expr::from(k * unit);
                let mut word = alu(OP_MOV);
                (word.a, word.write, word.link_send, word.vector) = ((reg, at + Expr::from(sent)), false, true, unit > 1);
                word.link_loop = s.ring_last;
                s.push(word);
            }
            for k in 0..count {
                let at = first + Expr::from(k * unit);
                let mut word = alu(OP_LINK);
                (word.dst, word.vector) = ((reg, at + Expr::from(received)), unit > 1);
                word.link_loop = s.ring_first;
                s.push(word);
            }
        };
        self.each((words / chunk).into(), |s, c| pass(s, c * (chunk * unit), chunk));
        pass(self, Expr::from(words / chunk * chunk * unit), words % chunk);
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

    fn scan_and_shift(&mut self, reg: u32, from: Expr, len: Expr) {
        // A step clears in its first cycle and scans in its second, so a scan that is a
        // single step needs no clearing step of its own.
        let lanes = 1u32 << self.lane_bits;
        if len.is_constant() && (len.constant() == 1 || (len.constant() == lanes && from.is_multiple_of(lanes))) {
            let mut scan = alu(OP_MOV);
            (scan.a, scan.write, scan.magnitude_clear, scan.magnitude_scan) = ((reg, from), false, true, true);
            scan.vector = len.constant() > 1;
            return self.push(scan);
        }
        let mut first = alu(OP_MOV);
        first.a = (reg, from);
        first.write = false;
        first.magnitude_clear = true;
        self.push(first);
        self.elements(len, &[from], true, |s, i, vector| {
            let mut scan = alu(OP_MOV);
            scan.a = (reg, from + i);
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
            Step::Combine { weights, matrix, window, window_a, window_b, .. } | Step::CombineRows { weights, matrix, window, window_a, window_b, .. } => {
                *window = step;
                *window_a = windowed.contains(&weights.0);
                *window_b = windowed.contains(matrix);
            }
            Step::Clustered { body, .. } => mark_window(body, step, windowed, exponents),
            Step::CellGroup { src, .. } => assert!(!windowed.contains(src), "the cells read from fixed registers"),
            Step::CellSeek { .. } | Step::CellPass | Step::CellDrain { .. } | Step::WideReduce { .. } => {}
        }
    }
}

/// The delta rule on one unit, with the whole state.
fn whole_delta_step<O: BlockOps>(ops: &mut O, r: DeltaRegs, key: u32, value: u32) {
    let cells = key * value;
    ops.scale(r.decay, r.state, r.state, cells.into());
    ops.row_combination(r.key, r.state, r.memory, (value, value, 0.into()), key.into());
    ops.add_sub(true, (r.value, 0.into()), (r.memory, 0.into()), 1, r.error, value.into());
    ops.scale(r.beta, r.error, r.delta, value.into());
    // The output reads the old state and the update apart, before rounding merges them:
    // the update's contribution is often a tiny difference that rounding would erase.
    ops.row_combination(r.query, r.state, r.next, (value, value, 0.into()), key.into());
    ops.dot(r.key, r.query, r.memory, key.into());
    ops.scale(r.memory, r.delta, r.update, value.into());
    ops.add(r.next, r.update, r.out, value.into());
    ops.outer_product(r.key, r.delta, r.update, key, value);
    ops.add(r.state, r.update, r.next, cells.into());
    ops.copy_part(r.next, 0.into(), r.state, 0.into(), cells.into());
}

/// Whether one loop can issue an element per cycle: results land two or three cycles
/// after the read, so nothing may read a place an earlier element is still to write.
/// With `per_block`, a broadcast operand A may also take one element for each block.
fn can_stream(variable: Var, step: &Alu<Expr>, unit: u32, per_block: bool) -> bool {
    let stride = |index: Expr| index.coefficient(variable);
    let moves_with = |index: Expr| stride(index) == 0 || stride(index) == unit;
    let a_per_block = per_block && unit > 1 && step.vector && step.broadcast_a && stride(step.a.1) == 1;
    // Moving elements to earlier places is safe: the reads stay ahead of the writes.
    let safe = |operand: At<Expr>| operand.0 != step.dst.0 || (stride(operand.1) == unit && (operand.1 == step.dst.1 || operand.1.is_after(step.dst.1)));
    let plain = !(step.load_shift_a || step.load_shift_b || step.load_shift_left || step.load_wide_shift || step.link_send || step.magnitude_clear || step.max_clear || step.max_out || step.accumulator_clear || step.wide_clear || step.wide_write) && step.literal.is_none();
    let streamable = matches!(step.op, OP_ADD | OP_SUB | OP_MUL | OP_MOV | OP_FUNCTION | OP_SHIFT_LEFT | OP_WIDE_OUT);
    let scans = !step.write;
    plain && streamable && (scans || stride(step.dst.1) == unit) && (moves_with(step.a.1) || a_per_block) && moves_with(step.b.1) && (scans || (safe(step.a) && safe(step.b)))
}

impl BlockOps for Assembly {
    type N = Expr;

    fn window_length(&self, position: Expr, window: u32) -> Expr {
        assert_eq!(position, Expr::var(Var::Position), "the window is measured from the position");
        Expr::var(Var::WindowLength(window))
    }

    fn window_slot(&self, position: Expr, window: u32) -> Expr {
        assert_eq!(position, Expr::var(Var::Position), "the window is measured from the position");
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
        self.zero_place();
        self.forget_shifts();
        let outer = std::mem::take(&mut self.steps);
        body(self, Expr::var(variable));
        let inner = std::mem::replace(&mut self.steps, outer);
        self.steps.push(Step::Repeat { variable, count, body: inner });
        self.forget_shifts();
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
        self.normalize_from(reg, 0.into(), len);
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
        self.stream_last(1 << self.lane_bits);
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
        self.narrow_part(a, b, dst, 0.into(), sets);
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
        let Some((part, parts)) = self.split.filter(|(_, parts)| width.is_multiple_of(*parts)) else {
            let column = self.local_columns(&store).map_or(offset, |round| round * width);
            return self.dots(probabilities, values, dst, width.into(), tokens, |j, t| ((probabilities, t), (values, (column + j) * capacity + t)));
        };
        // Each unit sums its share of the columns over every token; the rest of its sums
        // stay zero, so adding the units' wide memories puts every column in all of them.
        let share = width / parts;
        let zero = self.sc(10);
        self.immediate(0, zero);
        self.steps.push(Step::Combine { weights: zero, matrix: values, start: 0.into(), width: width.into(), fresh: true, window: 0.into(), window_a: false, window_b: false, moving: false });
        let first = part * share;
        let column = self.local_columns(&store).map_or(offset + first, |round| round * share);
        self.each(share.into(), |s, k| {
            let j = first + k;
            let at = (column + k) * capacity;
            s.clear_accumulator();
            s.elements(tokens, &[0.into(), at], true, |s, t, vector| {
                let mut step = alu(OP_MOV);
                (step.a, step.b, step.write, step.accumulate, step.vector) = ((probabilities, t), (values, at + t), false, true, vector);
                s.push(step);
            });
            s.store_accumulator(j);
        });
        self.steps.push(Step::WideReduce { rows: width.into(), group: parts.trailing_zeros() });
        self.narrow_sums(probabilities, values, dst, width.into());
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

    fn delta_step(&mut self, r: DeltaRegs, key: u32, value: u32) {
        let Some((part, parts)) = self.split.filter(|(_, parts)| key.is_multiple_of(*parts)) else {
            return whole_delta_step(self, r, key, value);
        };
        // Each unit keeps its own rows of the state: element-wise work divides among them,
        // and the sums over rows add up across the group, exact, before anything rounds.
        let (rows, group) = (key / parts, parts.trailing_zeros());
        let cells = rows * value;
        self.copy_part(r.key, part * rows, r.key_rows, 0.into(), rows.into());
        self.copy_part(r.query, part * rows, r.query_rows, 0.into(), rows.into());
        self.scale(r.decay, r.state, r.state, cells.into());
        self.combine_rows(r.key_rows, r.state, r.memory, (value, rows), group);
        self.add_sub(true, (r.value, 0.into()), (r.memory, 0.into()), 1, r.error, value.into());
        self.scale(r.beta, r.error, r.delta, value.into());
        self.combine_rows(r.query_rows, r.state, r.next, (value, rows), group);
        self.dot(r.key, r.query, r.memory, key.into());
        self.scale(r.memory, r.delta, r.update, value.into());
        self.add(r.next, r.update, r.out, value.into());
        self.outer_product(r.key_rows, r.delta, r.update, rows, value);
        self.add(r.state, r.update, r.next, cells.into());
        self.copy_part(r.next, 0.into(), r.state, 0.into(), cells.into());
    }

    fn in_slabs(&mut self, len: u32, outputs: &[u32], mut body: impl FnMut(&mut Self, Expr, u32)) {
        // Slices of a power of two, whole blocks, that divide the vector among the units.
        let lanes = 1u32 << self.lane_bits;
        let slab = (1..=len.max(1).ilog2()).map(|bits| 1u32 << bits).find(|slab| *slab >= lanes && len.is_multiple_of(*slab) && len / slab <= self.clusters);
        // Units already running separate copies of a loop cannot split again.
        let Some(slab) = slab.filter(|slab| self.clusters > 1 && self.streams && len / slab > 1 && !self.in_units) else {
            return body(self, 0.into(), len);
        };
        let units = len / slab;
        let start = Var::Loop(self.loops);
        self.loops += 1;
        self.zero_place();
        self.forget_shifts();
        let before = std::mem::take(&mut self.steps);
        let group = units.next_power_of_two().trailing_zeros();
        let outer_group = std::mem::replace(&mut self.slab_group, group);
        // The unit number times the slice length, so the start is known to sit on a block.
        self.in_units = true;
        body(self, Expr::var(start) * slab, slab);
        self.in_units = false;
        self.slab_group = outer_group;
        let inside = std::mem::replace(&mut self.steps, before);
        self.forget_shifts();
        self.steps.push(Step::Clustered { vars: vec![start], assignments: (0..units).map(|unit| vec![unit]).collect(), slots: None, body: inside });
        self.clear_shifts();
        // The units are told apart by the stream's counter, so the copy is a stream however
        // short it is.
        let (count, unit) = if lanes > 1 { (len / lanes, lanes) } else { (len, 1) };
        for output in outputs {
            if self.spread.remove(output) {
                continue;
            }
            let variable = Var::Loop(self.loops);
            self.loops += 1;
            let mut step = alu(OP_MOV);
            let at = Expr::var(variable) * unit;
            (step.a, step.dst, step.vector, step.gather_shift) = ((*output, at), (*output, at), lanes > 1, slab.trailing_zeros());
            self.steps.push(Step::Stream { variable, count: count.into(), in_elements: false, body: step });
            // Units that held no slice never wrote the exponent: they take unit 0's.
            if units < self.clusters {
                let mut exponent = alu(OP_MOV);
                (exponent.a, exponent.dst, exponent.from_cluster) = (self.exponent(*output), self.exponent(*output), 1);
                self.push(exponent);
            }
        }
    }

    fn one_head_per_unit(&self) -> bool {
        self.head_round.is_some_and(|round| round == Expr::from(0))
    }

    fn each_head_split(&mut self, outer: u32, inner: u32, mut body: impl FnMut(&mut Self, Expr, Expr)) {
        let heads = outer * inner;
        let mut parts = 1;
        while heads * parts * 2 <= self.clusters {
            parts *= 2;
        }
        if parts == 1 {
            return self.each_head(outer, inner, body);
        }
        let (first, second, part) = (Var::Loop(self.loops), Var::Loop(self.loops + 1), Var::Loop(self.loops + 2));
        self.loops += 3;
        self.zero_place();
        self.forget_shifts();
        let before = std::mem::take(&mut self.steps);
        let (width, slots) = self.head_slots(outer, inner, parts);
        self.split = Some((Expr::var(part), parts));
        self.in_units = true;
        self.heads_at_once = outer * width * parts <= self.clusters;
        // Heads split into parts always fit the units in one round.
        self.head_round = Some(0.into());
        body(self, Expr::var(first), Expr::var(second));
        self.head_round = None;
        self.heads_at_once = false;
        self.in_units = false;
        self.split = None;
        let inside = std::mem::replace(&mut self.steps, before);
        self.forget_shifts();
        let total = outer * width * parts;
        for block in 0..total.div_ceil(self.clusters) {
            let assignments = (0..self.clusters.min(total - block * self.clusters)).map(|unit| {
                let index = block * self.clusters + unit;
                let head = index / parts;
                vec![head / width, head % width, index % parts]
            });
            self.steps.push(Step::Clustered { vars: vec![first, second, part], assignments: assignments.collect(), slots, body: inside.clone() });
        }
        self.head_units = (0..heads).map(|h| ((h / inner) * width + h % inner) * parts).collect();
    }

    fn each_head(&mut self, outer: u32, inner: u32, mut body: impl FnMut(&mut Self, Expr, Expr)) {
        self.head_units = (0..outer * inner).collect();
        if self.clusters <= 1 {
            return self.each(outer.into(), |s, o| s.each(inner.into(), |s, i| body(s, o, i)));
        }
        let (first, second) = (Var::Loop(self.loops), Var::Loop(self.loops + 1));
        self.loops += 2;
        self.zero_place();
        self.forget_shifts();
        let before = std::mem::take(&mut self.steps);
        let (width, slots) = self.head_slots(outer, inner, 1);
        let total = outer * width;
        let rounds = total.div_ceil(self.clusters);
        // More heads than units: the round is a number of its own, the same in every unit.
        let round = (rounds > 1).then(|| {
            self.loops += 1;
            Var::Loop(self.loops - 1)
        });
        self.in_units = true;
        self.heads_at_once = outer * width <= self.clusters;
        self.head_round = Some(round.map_or(0.into(), Expr::var));
        body(self, Expr::var(first), Expr::var(second));
        self.head_round = None;
        self.heads_at_once = false;
        self.in_units = false;
        let inside = std::mem::replace(&mut self.steps, before);
        self.forget_shifts();
        let vars: Vec<Var> = [first, second].into_iter().chain(round).collect();
        for block in 0..rounds {
            let assignments = (0..self.clusters.min(total - block * self.clusters)).map(|unit| {
                let head = block * self.clusters + unit;
                let mut values = vec![head / width, head % width];
                if round.is_some() {
                    values.push(block);
                }
                values
            });
            self.steps.push(Step::Clustered { vars: vars.clone(), assignments: assignments.collect(), slots, body: inside.clone() });
        }
        self.head_units = (0..outer * inner).map(|h| (h / inner) * width + h % inner).collect();
    }

    fn leave_spread(&mut self, reg: u32) {
        self.spread.insert(reg);
    }

    fn align_heads(&mut self, reg: u32, len: u32) {
        if !(self.in_units && self.heads_at_once) {
            return;
        }
        self.clear_shifts();
        let (largest, shift) = (self.sc(1), self.sc(2));
        let exponent = self.exponent(reg);
        // Each unit offers its exponent and all of them get the largest.
        let mut most = alu(OP_MAX);
        (most.a, most.b, most.dst, most.reduce_group) = (exponent, exponent, largest, self.clusters.trailing_zeros());
        self.push(most);
        self.binary(OP_SUB, largest, exponent, shift);
        self.load_shifts(Some(shift), None, None);
        self.elements(len.into(), &[0.into()], true, |s, i, vector| {
            let mut step = alu(OP_MOV);
            (step.a, step.dst, step.vector) = ((reg, i), (reg, i), vector);
            s.push(step);
        });
        self.clear_shifts();
        self.binary(OP_MOV, largest, largest, exponent);
        // Normalised together, as the gathered vector would be: the units' magnitudes
        // combine, so every head moves by the same amount.
        let outer = std::mem::replace(&mut self.slab_group, self.clusters.trailing_zeros());
        self.normalize(reg, len.into());
        self.slab_group = outer;
        self.heads_aligned += 1;
    }

    fn gather_heads(&mut self, parts: &[(u32, u32)], dst: u32) {
        if self.clusters <= 1 {
            return self.gather(parts, dst);
        }
        self.clear_shifts();
        let zero = self.sc(15);
        let (largest, shift, exponent) = (self.sc(1), self.sc(2), self.sc(3));
        let head_units = self.head_units.clone();
        let unit_of = |part: usize, units: u32| head_units.get(part).copied().unwrap_or(part as u32) % units + 1;
        let units = self.clusters;
        if self.heads_aligned > 0 {
            self.heads_aligned -= 1;
            // Every part already sits at the one exponent: copy them, and that exponent.
            let mut read = alu(OP_MOV);
            (read.a, read.b, read.dst, read.from_cluster) = (self.exponent(parts[0].0), zero, self.exponent(dst), unit_of(0, units));
            self.push(read);
            let mut offset = 0;
            for (part, (reg, len)) in parts.iter().enumerate() {
                let at: Expr = offset.into();
                self.elements((*len).into(), &[at], true, |s, i, vector| {
                    let mut step = alu(OP_MOV);
                    (step.a, step.b, step.dst, step.from_cluster) = ((*reg, i), zero, (dst, at + i), unit_of(part, units));
                    (step.vector, step.broadcast_b) = (vector, vector);
                    s.push(step);
                });
                offset += len;
            }
            return;
        }
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

    fn depthwise_part(&mut self, taps: u32, history: u32, dst: u32, width: u32, kernel: u32, start: Expr, len: u32) {
        if start == Expr::from(0) && len == width {
            return self.depthwise(taps, history, dst, width, kernel);
        }
        assert!(self.combines, "only a die that sums rows splits the convolution");
        let zero = self.sc(10);
        self.immediate(0, zero);
        self.steps.push(Step::Combine { weights: zero, matrix: history, start, width: len.into(), fresh: true, window: 0.into(), window_a: false, window_b: false, moving: false });
        self.each(kernel.into(), |s, k| s.steps.push(Step::Combine { weights: (taps, k * width + start), matrix: history, start: k * width + start, width: len.into(), fresh: false, window: 0.into(), window_a: false, window_b: false, moving: true }));
        self.narrow_part_at(taps, history, dst, 0.into(), start, len.into());
    }

    fn copy_elements(&mut self, reg: u32, from: Expr, to: Expr, width: u32) {
        self.elements(width.into(), &[from, to], true, |s, c, vector| {
            let mut step = alu(OP_MOV);
            (step.a, step.dst, step.vector) = ((reg, from + c), (reg, to + c), vector);
            s.push(step);
        });
    }

    fn store_key_part(&mut self, src: u32, exponent: i64, keys: u32, slot: Expr, width: u32, from: Expr, len: u32) {
        // The columns this unit's share of the head reads, as `score_keys` splits them.
        let (from, len) = match self.split.filter(|(_, parts)| len.is_multiple_of(*parts)) {
            Some((part, parts)) => (from + part * (len / parts), len / parts),
            None => (from, len),
        };
        let Some(capacity) = self.key_capacity else {
            return self.align_to(src, from, exponent, keys, slot * width + from, len.into());
        };
        if self.key_shift == 0 {
            // Columns written one element at a time: the whole key, as `store_key` does.
            return self.store_key(src, exponent, keys, slot, width);
        }
        let start: Expr = 32u32.max(1 << self.lane_bits).into();
        let scratch = self.scratch;
        self.align_to(src, from, exponent, scratch, start, len.into());
        let variable = Var::Loop(self.loops);
        self.loops += 1;
        // A unit with heads of its own keeps their columns from its first word, a round of
        // heads after another, not at their places in the whole key.
        let column = self.head_round.map_or(from, |round| round * len);
        let mut step = alu(OP_MOV);
        (step.a, step.dst, step.write_spread) = ((scratch, start + Expr::var(variable)), (keys, slot + column * capacity), true);
        self.steps.push(Step::Stream { variable, count: len.into(), in_elements: false, body: step });
    }

    fn store_key(&mut self, src: u32, exponent: i64, keys: u32, slot: Expr, width: u32) {
        let Some(capacity) = self.key_capacity else {
            return self.align_to(src, 0.into(), exponent, keys, slot * width, width.into());
        };
        // With a die that counts in whole columns, the key goes out one element per cycle
        // from an aligned copy.
        if self.key_shift > 0 {
            let start: Expr = 32u32.max(1 << self.lane_bits).into();
            let scratch = self.scratch;
            self.align_to(src, 0.into(), exponent, scratch, start, width.into());
            let variable = Var::Loop(self.loops);
            self.loops += 1;
            let mut step = alu(OP_MOV);
            (step.a, step.dst, step.write_spread) = ((scratch, start + Expr::var(variable)), (keys, slot), true);
            return self.steps.push(Step::Stream { variable, count: width.into(), in_elements: false, body: step });
        }
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
        // Split among units, each adds its share of the columns and the units add up.
        let (first, share, parts) = match self.split.filter(|(_, parts)| width.is_multiple_of(*parts)) {
            Some((part, parts)) => (part * (width / parts), width / parts, parts),
            None => (0.into(), width, 1),
        };
        let column = self.local_columns(&store).map_or(offset + first, |round| round * share);
        self.each(share.into(), |s, j| s.steps.push(Step::Combine { weights: (query, first + j), matrix: keys, start: (column + j) * capacity, width: tokens, fresh: false, window: 0.into(), window_a: false, window_b: false, moving: false }));
        if parts > 1 {
            self.steps.push(Step::WideReduce { rows: tokens, group: parts.trailing_zeros() });
        }
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

    fn save_exponent(&mut self, reg: u32, to: At<Expr>) {
        self.binary(OP_MOV, self.exponent(reg), self.sc(15), to);
    }

    fn restore_exponent(&mut self, from: At<Expr>, reg: u32) {
        self.binary(OP_MOV, from, self.sc(15), self.exponent(reg));
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
        self.cell_matvec(place, first, src, rows);
        // Every unit holds every total; each narrows a slice and one stream collects them.
        self.in_slabs(rows, &[dst], |s, start, count| s.narrow_part(matrix, src, dst, start, count.into()));
    }

    fn matvecs(&mut self, parts: &[MatvecPart], src: u32, cols: u32) {
        let place = self.in_cells.get(&parts[0].name).copied();
        let (Some(place), true) = (place, parts.len() > 1) else {
            for part in parts {
                self.matvec(&part.name, part.matrix, 0.into(), src, part.dst, part.rows, cols);
            }
            return;
        };
        // One pass of the cells computes every matrix; each is then narrowed on its own,
        // with its own exponent, exactly as if it had been applied alone.
        let rows: Vec<u32> = parts.iter().map(|p| p.rows).collect();
        let (starts, total) = chained_rows(&rows, 1 << self.lane_bits);
        self.cell_matvec(place, 0.into(), src, total);
        for (part, start) in parts.iter().zip(starts) {
            let (matrix, dst) = (part.matrix, part.dst);
            self.in_slabs(part.rows, &[dst], |s, from, count| s.narrow_part_at(matrix, src, dst, from + Expr::from(start), from, count.into()));
        }
    }

    fn copy_row(&mut self, _name: &str, matrix: u32, row: Expr, cols: u32, dst: u32) {
        self.copy_part(matrix, row * cols, dst, 0.into(), cols.into());
    }

    fn matvec_rows(&mut self, name: &str, matrix: u32, src: u32, dst: u32, rows: u32, offset: u32) {
        self.ring_matvec_rows(name, matrix, src, dst, rows, offset);
    }

    fn matvecs_rows(&mut self, first: &str, parts: &[(u32, u32, u32, u32)], src: u32) {
        self.ring_matvecs_rows(first, parts, src);
    }

    fn all_gather(&mut self, reg: u32, block: u32, index: u32) {
        self.ring_all_gather(reg, block, index);
    }
}

impl Assembly {
    /// The cells' part of a matrix-vector product: `rows` sums from `first`, in the wide
    /// memory of every unit.
    fn cell_matvec(&mut self, place: CellMatrix, first: Expr, src: u32, rows: u32) {
        // A matrix used a block at a time keeps each block's rows from cell 0 on. Each
        // vector unit has its own cells, and takes its own slice of the columns.
        let units = self.clusters;
        let per_unit = self.cells / units;
        let groups = place.groups.div_ceil(units);
        let columns = groups * place.group;
        let passes = rows.div_ceil(per_unit);
        let block = first.divided_by(rows);
        self.steps.push(Step::CellSeek { weights: Expr::from(place.weights) + block * ((passes * columns) >> place.packing), factors: Expr::from(place.factors) + block * (passes * groups), packing: place.packing });
        let column = Var::Loop(self.loops);
        self.loops += 1;
        let before = std::mem::take(&mut self.steps);
        // The first pass stands apart, so every turn of the loop drains the pass before it
        // while it computes its own. Nothing may come between them, not even the zero a
        // loop wants ready.
        self.zero_place();
        let full = rows / per_unit;
        if full > 0 {
            self.cell_round(src, place, per_unit, Expr::var(column));
        }
        if full > 1 {
            self.each((full - 1).into(), |s, _| s.cell_round(src, place, per_unit, Expr::var(column)));
        }
        if !rows.is_multiple_of(per_unit) {
            self.cell_round(src, place, rows % per_unit, Expr::var(column));
        }
        let inside = std::mem::replace(&mut self.steps, before);
        self.steps.push(Step::Clustered { vars: vec![column], assignments: (0..units).map(|unit| vec![unit * columns]).collect(), slots: None, body: inside });
        // A die that feeds its cells in a stream adds the units' rows as they drain.
        if units > 1 && !self.drains_add {
            self.steps.push(Step::WideReduce { rows: rows.into(), group: self.clusters.trailing_zeros() });
        }
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
                Step::Clustered { .. } | Step::Combine { .. } | Step::CombineRows { .. } | Step::CellSeek { .. } | Step::CellPass | Step::CellGroup { .. } | Step::CellDrain { .. } | Step::WideReduce { .. } => panic!("this model does not keep the cells' weights"),
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
            other => panic!("operation without a model: {other}"),
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
