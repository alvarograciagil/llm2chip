//! Runs the die's block floating point at software speed. Every operation matches
//! the microcode bit for bit, which the tiny-model test proves against the gates.

use crate::weights::Weights;
use netlist::block::{accumulator_bits, round_right, tables, BlockOps, Place, FN_RECIPROCAL, FN_INV_SQRT, FN_SOFTPLUS, RECIPROCAL_OUTPUT_EXPONENT, SOFTPLUS_OUTPUT_EXPONENT, TABLE_INPUT_EXPONENT, UNIT_OUTPUT_EXPONENT};
use netlist::DieConfig;
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

#[inline]
fn shift_amount(x: i64) -> u32 {
    (x & 63) as u32
}

#[inline]
fn saturate_left(x: i64, amount: u32, bits: u32) -> i64 {
    let unused = 128 - 2 * bits;
    let widened = (((x as i128) << amount) << unused) >> unused;
    let top = (1i128 << (bits - 1)) - 1;
    widened.clamp(-top - 1, top) as i64
}

pub struct BlockVm {
    bits: u32,
    registers: Vec<Vec<i32>>,
    exponents: Vec<i64>,
    table: Vec<i32>,
    function_bits: u32,
    weights: Option<Arc<Weights>>,
    matrix_exponents: RefCell<HashMap<String, i64>>,
}

impl BlockVm {
    pub fn new(config: &DieConfig, registers: usize, weights: Option<Arc<Weights>>) -> Self {
        Self {
            bits: config.cell.activation_bits,
            registers: vec![Vec::new(); registers],
            exponents: vec![0; registers],
            table: tables(config).iter().map(|v| *v as i32).collect(),
            function_bits: config.function_bits,
            weights,
            matrix_exponents: RefCell::new(HashMap::new()),
        }
    }

    #[inline]
    fn top(&self) -> i64 {
        (1i64 << (self.bits - 1)) - 1
    }

    #[inline]
    fn fraction(&self) -> u32 {
        self.bits - 1
    }

    #[inline]
    fn wrap(&self, x: i64) -> i64 {
        let shift = 64 - self.bits;
        (x << shift) >> shift
    }

    #[inline]
    fn wrap_wide(&self, x: i64) -> i64 {
        let shift = 64 - accumulator_bits(self.bits);
        (x << shift) >> shift
    }

    #[inline]
    fn saturating_left(&self, x: i64, amount: u32) -> i64 {
        saturate_left(x, amount, self.bits)
    }

    #[inline]
    fn narrow_shift(&self, value: i64, amount: u32) -> i64 {
        round_right(value as i128, amount, self.bits)
    }

    pub fn clear(&mut self) {
        self.registers.iter_mut().for_each(Vec::clear);
        self.exponents.fill(0);
    }

    pub fn mantissa(&self, reg: u32, index: u32) -> i64 {
        self.get(reg, index)
    }

    pub fn register_exponent(&self, reg: u32) -> i64 {
        self.exponents[reg as usize]
    }

    pub fn floats(&self, reg: u32, len: usize) -> Vec<f32> {
        let scale = 2f32.powi(self.exponents[reg as usize] as i32 - self.fraction() as i32);
        (0..len).map(|i| self.get(reg, i as u32) as f32 * scale).collect()
    }

    #[inline]
    fn get(&self, reg: u32, index: u32) -> i64 {
        self.registers[reg as usize].get(index as usize).copied().unwrap_or(0) as i64
    }

    fn set(&mut self, reg: u32, index: u32, value: i64) {
        let r = &mut self.registers[reg as usize];
        if r.len() <= index as usize {
            r.resize(index as usize + 1, 0);
        }
        r[index as usize] = value as i32;
    }

    fn exp(&self, reg: u32) -> i64 {
        self.exponents[reg as usize]
    }

    fn set_exp(&mut self, reg: u32, value: i64) {
        self.exponents[reg as usize] = self.wrap(value);
    }

    fn table_at(&self, function: u32, value: i64) -> i64 {
        let index = ((value as u64 & ((1u64 << self.bits) - 1)) >> (self.bits - self.function_bits)) as usize;
        self.table[index | (function as usize) << self.function_bits] as i64
    }

    fn narrow(&mut self, sums: &[i64], ea: i64, eb: i64, dst: u32) {
        let wide = accumulator_bits(self.bits);
        let mask = (1u64 << (wide - 1)) - 1;
        let magnitude = sums.iter().fold(0u64, |m, w| m | ((*w ^ (*w >> 63)) as u64 & mask));
        let leading = (wide - 1) as i64 - (64 - magnitude.leading_zeros()) as i64;
        let keep = wide as i64 - self.bits as i64;
        let shift = self.wrap(keep - leading).max(0);
        for (i, w) in sums.iter().enumerate() {
            self.set(dst, i as u32, self.narrow_shift(*w, shift_amount(shift).min(63)));
        }
        let exponent = self.wrap(self.wrap(self.wrap(ea + eb) + shift) - self.fraction() as i64);
        self.set_exp(dst, exponent);
        self.normalize(dst, sums.len() as u32);
    }

    fn aligned(&self, src: u32, from: u32, exponent: i64, len: u32) -> Vec<i64> {
        let e = self.exp(src);
        let left = self.wrap(e - exponent).max(0);
        let right = self.wrap(exponent - e).max(0);
        (0..len)
            .map(|i| self.saturating_left(self.narrow_shift(self.get(src, from + i), shift_amount(right).min(63)), shift_amount(left)))
            .collect()
    }

    fn matrix_exponent(&self, name: &str, cols: u32) -> i64 {
        let weights = self.weights.as_ref().expect("la máquina necesita los pesos");
        *self.matrix_exponents.borrow_mut().entry(name.to_string()).or_insert_with(|| {
            weights.matrix(name).expect("matriz del programa").exponent(cols as usize)
        })
    }
}

impl BlockOps for BlockVm {
    fn load_vector(&mut self, reg: u32, values: &[f64]) {
        let (mantissas, exponent) = netlist::block::encode_vector(values, self.bits);
        self.load_mantissas(reg, &mantissas, exponent);
    }

    fn load_mantissas(&mut self, reg: u32, mantissas: &[i64], exponent: i64) {
        self.registers[reg as usize] = mantissas.iter().map(|m| *m as i32).collect();
        self.set_exp(reg, exponent);
    }

    fn normalize(&mut self, reg: u32, len: u32) {
        let (mask, fraction, bits) = (self.top() as u64, self.fraction(), self.bits);
        let values = &mut self.registers[reg as usize];
        if values.len() < len as usize {
            values.resize(len as usize, 0);
        }
        let magnitude = values[..len as usize].iter().fold(0u64, |m, v| m | ((*v ^ (*v >> 31)) as u64 & mask));
        let count = fraction - (64 - magnitude.leading_zeros());
        for v in &mut values[..len as usize] {
            *v = saturate_left(*v as i64, count, bits) as i32;
        }
        self.set_exp(reg, self.exp(reg) - count as i64);
    }

    fn add_sub(&mut self, subtract: bool, a: Place, b: Place, b_stride: u32, dst: u32, len: u32) {
        let (ea, eb) = (self.exp(a.0), self.exp(b.0));
        let common = self.wrap(ea.max(eb) + 1);
        let shift_a = shift_amount(self.wrap(common - ea)).min(63);
        let shift_b = shift_amount(self.wrap(common - eb)).min(63);
        let results: Vec<i64> = (0..len)
            .map(|i| {
                let (x, y) = (round_right(self.get(a.0, a.1 + i) as i128, shift_a, self.bits), round_right(self.get(b.0, b.1 + i * b_stride) as i128, shift_b, self.bits));
                self.wrap(if subtract { x - y } else { x + y })
            })
            .collect();
        for (i, r) in results.iter().enumerate() {
            self.set(dst, i as u32, *r);
        }
        self.set_exp(dst, common);
        self.normalize(dst, len);
    }

    fn multiply(&mut self, a: u32, b: u32, dst: u32, len: u32) {
        let results: Vec<i64> = (0..len).map(|i| round_right((self.get(a, i) * self.get(b, i)) as i128, self.fraction(), self.bits)).collect();
        for (i, r) in results.iter().enumerate() {
            self.set(dst, i as u32, *r);
        }
        self.set_exp(dst, self.exp(a) + self.exp(b));
        self.normalize(dst, len);
    }

    fn scale(&mut self, s_reg: u32, a: u32, dst: u32, len: u32) {
        let s = self.get(s_reg, 0);
        let results: Vec<i64> = (0..len).map(|i| round_right((s * self.get(a, i)) as i128, self.fraction(), self.bits)).collect();
        for (i, r) in results.iter().enumerate() {
            self.set(dst, i as u32, *r);
        }
        self.set_exp(dst, self.exp(s_reg) + self.exp(a));
        self.normalize(dst, len);
    }

    fn outer_product(&mut self, a: u32, b: u32, dst: u32, rows: u32, cols: u32) {
        for i in 0..rows {
            for j in 0..cols {
                let value = round_right((self.get(a, i) * self.get(b, j)) as i128, self.fraction(), self.bits);
                self.set(dst, i * cols + j, value);
            }
        }
        self.set_exp(dst, self.exp(a) + self.exp(b));
        self.normalize(dst, rows * cols);
    }

    fn dots<F>(&mut self, a: u32, b: u32, dst: u32, sets: u32, terms: u32, indices: F)
    where
        F: Fn(u32, u32) -> Option<(Place, Place)>,
    {
        let sums: Vec<i64> = (0..sets)
            .map(|set| {
                let mut total = 0i64;
                for term in 0..terms {
                    if let Some((x, y)) = indices(set, term) {
                        total = self.wrap_wide(total + self.get(x.0, x.1) * self.get(y.0, y.1));
                    }
                }
                total
            })
            .collect();
        self.narrow(&sums, self.exp(a), self.exp(b), dst);
    }

    fn function(&mut self, function: u32, src: u32, dst: u32, len: u32) {
        let inputs = self.aligned(src, 0, TABLE_INPUT_EXPONENT, len);
        for (i, x) in inputs.iter().enumerate() {
            self.set(dst, i as u32, self.table_at(function, *x));
        }
        self.set_exp(dst, if function == FN_SOFTPLUS { SOFTPLUS_OUTPUT_EXPONENT } else { UNIT_OUTPUT_EXPONENT });
        self.normalize(dst, len);
    }

    fn reciprocal(&mut self, src: u32, dst: u32) {
        self.set(dst, 0, self.table_at(FN_RECIPROCAL, self.get(src, 0)));
        self.set_exp(dst, RECIPROCAL_OUTPUT_EXPONENT - self.exp(src));
        self.normalize(dst, 1);
    }

    fn inverse_sqrt(&mut self, src: u32, dst: u32) {
        let e = self.exp(src);
        let parity = e & 1;
        let even = self.wrap(e + parity);
        self.set(dst, 0, self.table_at(FN_INV_SQRT, self.get(src, 0) >> parity));
        self.set_exp(dst, RECIPROCAL_OUTPUT_EXPONENT - (even >> 1));
        self.normalize(dst, 1);
    }

    fn copy_part(&mut self, src: u32, from: u32, dst: u32, to: u32, len: u32) {
        for i in 0..len {
            self.set(dst, to + i, self.get(src, from + i));
        }
        self.set_exp(dst, self.exp(src));
    }

    fn gather(&mut self, parts: &[(u32, u32)], dst: u32) {
        let largest = parts.iter().map(|(r, _)| self.exp(*r)).max().unwrap_or(0);
        let mut offset = 0;
        for (reg, len) in parts {
            let shift = shift_amount(self.wrap(largest - self.exp(*reg))).min(63);
            for i in 0..*len {
                self.set(dst, offset + i, round_right(self.get(*reg, i) as i128, shift, self.bits));
            }
            offset += len;
        }
        self.set_exp(dst, largest);
        self.normalize(dst, offset);
    }

    fn align_to(&mut self, src: u32, from: u32, exponent: i64, dst: u32, to: u32, len: u32) {
        for (i, v) in self.aligned(src, from, exponent, len).iter().enumerate() {
            self.set(dst, to + i as u32, *v);
        }
    }

    fn move_element(&mut self, from: Place, to: Place) {
        self.set(to.0, to.1, self.get(from.0, from.1));
    }

    fn maximum(&mut self, src: u32, len: u32, dst: u32) {
        let best = (0..len).map(|i| self.get(src, i)).max().unwrap_or(0);
        self.set(dst, 0, best);
        self.set_exp(dst, self.exp(src));
    }

    fn mark_equal(&mut self, a: u32, b: u32, dst: u32, len: u32) {
        let target = self.get(b, 0);
        let mut seen = false;
        for i in 0..len {
            let first = !seen && self.wrap(self.get(a, i) - target) == 0;
            seen |= first;
            self.set(dst, i, if first { self.top() } else { 0 });
        }
        self.set_exp(dst, 0);
    }

    fn set_exponent(&mut self, reg: u32, exponent: i64) {
        self.set_exp(reg, exponent);
    }

    fn subtract_exponent(&mut self, reg: u32, amount: i64) {
        self.set_exp(reg, self.exp(reg) - amount);
    }

    fn matvec(&mut self, name: &str, _matrix: u32, first: u32, src: u32, dst: u32, rows: u32, cols: u32) {
        let weights = self.weights.clone().expect("la máquina necesita los pesos");
        let exponent = self.matrix_exponent(name, cols);
        let mut x = self.registers[src as usize].clone();
        x.resize(cols as usize, 0);
        let matrix = weights.matrix(name).expect("matriz del programa");
        let sums: Vec<i64> = matrix.exact_sums(first as usize, rows as usize, &x, exponent, self.bits).iter().map(|v| self.wrap_wide(*v)).collect();
        self.narrow(&sums, exponent, self.exp(src), dst);
    }

    fn copy_row(&mut self, name: &str, _matrix: u32, row: u32, cols: u32, dst: u32) {
        let weights = self.weights.clone().expect("la máquina necesita los pesos");
        let exponent = self.matrix_exponent(name, cols);
        let matrix = weights.matrix(name).expect("matriz del programa");
        let mantissas = matrix.fixed_row(row as usize, cols as usize, exponent, self.bits).expect("fila de la matriz");
        self.load_mantissas(dst, &mantissas, exponent);
    }

    fn known_zero(&self, reg: u32, index: u32) -> bool {
        self.get(reg, index) == 0
    }
}
