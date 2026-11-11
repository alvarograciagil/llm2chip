//! The control CPU: an RV32IM core in gates and an instruction-set simulator that
//! matches it cycle for cycle. Each instruction takes fetch, decode and execute;
//! loads add one cycle, and multiplication and division 33 more.

use crate::cell::Gen;
use crate::{Bit, Builder, Design};
use std::collections::BTreeMap;

const LUI: u32 = 0b0110111;
const AUIPC: u32 = 0b0010111;
const JAL: u32 = 0b1101111;
const JALR: u32 = 0b1100111;
const BRANCH: u32 = 0b1100011;
const LOAD: u32 = 0b0000011;
const STORE: u32 = 0b0100011;
const OP_IMM: u32 = 0b0010011;
const OP: u32 = 0b0110011;
const SYSTEM: u32 = 0b1110011;

/// Cycle at which a plain instruction writes its result.
pub const EXECUTE_CYCLE: u64 = 2;
/// Cycles spent inside the multiply and divide unit, including writing the result.
pub const MULDIV_CYCLES: u64 = 33;

/// Small combinational helpers over one module.
/// Adders this wide or wider take their carries from a prefix tree.
pub const PREFIX_FROM: usize = 8;

pub(crate) struct Logic<'a> {
    pub b: &'a mut Builder,
    pub next: usize,
}

impl Logic<'_> {
    pub fn fresh(&mut self) -> Bit {
        self.next += 1;
        self.b.wire(&format!("n{}", self.next), 1)[0]
    }

    pub fn fresh_bus(&mut self, width: usize) -> Vec<Bit> {
        (0..width).map(|_| self.fresh()).collect()
    }

    pub fn and(&mut self, a: Bit, b: Bit) -> Bit {
        let y = self.fresh();
        self.b.and_to(a, b, y);
        y
    }

    pub fn or(&mut self, a: Bit, b: Bit) -> Bit {
        let y = self.fresh();
        self.b.or_to(a, b, y);
        y
    }

    pub fn xor(&mut self, a: Bit, b: Bit) -> Bit {
        let y = self.fresh();
        self.b.xor_to(a, b, y);
        y
    }

    pub fn not(&mut self, a: Bit) -> Bit {
        let y = self.fresh();
        self.b.not_to(a, y);
        y
    }

    /// `a` when `s` is 0 and `b` when it is 1.
    pub fn mux(&mut self, s: Bit, a: Bit, b: Bit) -> Bit {
        let y = self.fresh();
        self.b.mux_to(s, a, b, y);
        y
    }

    pub fn mux_bus(&mut self, s: Bit, a: &[Bit], b: &[Bit]) -> Vec<Bit> {
        a.iter().zip(b).map(|(x, y)| self.mux(s, *x, *y)).collect()
    }

    /// In a tree, so the depth grows with the logarithm of the bits.
    pub fn all(&mut self, bits: &[Bit]) -> Bit {
        self.tree(bits, Bit::One, Self::and)
    }

    pub fn any(&mut self, bits: &[Bit]) -> Bit {
        self.tree(bits, Bit::Zero, Self::or)
    }

    /// The parity of `bits`, as a balanced tree of XORs.
    pub fn parity(&mut self, bits: &[Bit]) -> Bit {
        self.tree(bits, Bit::Zero, Self::xor)
    }

    fn tree(&mut self, bits: &[Bit], empty: Bit, join: fn(&mut Self, Bit, Bit) -> Bit) -> Bit {
        let mut level = bits.to_vec();
        if level.is_empty() {
            return empty;
        }
        while level.len() > 1 {
            level = level.chunks(2).map(|pair| if let [x, y] = pair { join(self, *x, *y) } else { pair[0] }).collect();
        }
        level[0]
    }

    pub fn equals(&mut self, bits: &[Bit], value: u32) -> Bit {
        let literals: Vec<Bit> = bits.iter().enumerate().map(|(i, b)| if value >> i & 1 == 1 { *b } else { self.not(*b) }).collect();
        self.all(&literals)
    }

    pub fn connect_flops(&mut self, d: &[Bit], q: &[Bit]) {
        for (d, q) in d.iter().zip(q) {
            self.b.dff_to(*d, *q);
        }
    }

    pub fn select(&mut self, select: &[Bit], leaves: &[Vec<Bit>]) -> Vec<Bit> {
        if select.is_empty() {
            return leaves[0].clone();
        }
        let top = *select.last().unwrap();
        let (low, high) = leaves.split_at(leaves.len() / 2);
        let (low, high) = (self.select(&select[..select.len() - 1], low), self.select(&select[..select.len() - 1], high));
        self.mux_bus(top, &low, &high)
    }

    /// `value >> amount`, filling with `fill` from the top.
    fn shift_right(&mut self, value: &[Bit], amount: &[Bit], fill: Bit) -> Vec<Bit> {
        let mut current = value.to_vec();
        for (stage, select) in amount.iter().enumerate() {
            let step = 1usize << stage;
            let shifted: Vec<Bit> = (0..current.len()).map(|i| current.get(i + step).copied().unwrap_or(fill)).collect();
            current = self.mux_bus(*select, &current, &shifted);
        }
        current
    }

    /// A Brent-Kung adder: the carries come from a tree of generate and propagate pairs,
    /// about twice as deep as the bits' logarithm instead of as deep as the bits.
    pub fn prefix_adder(&mut self, a: &[Bit], b: &[Bit], carry_in: Bit) -> (Vec<Bit>, Bit) {
        let n = a.len().min(b.len());
        let p: Vec<Bit> = (0..n).map(|i| self.xor(a[i], b[i])).collect();
        let mut g: Vec<Bit> = (0..n).map(|i| self.and(a[i], b[i])).collect();
        // The carry in folds into the first bit's generate.
        let first = self.and(p[0], carry_in);
        g[0] = self.or(g[0], first);
        let mut pp = p.clone();
        let combine = |l: &mut Self, g: &mut [Bit], pp: &mut [Bit], i: usize, j: usize| {
            let t = l.and(pp[i], g[j]);
            g[i] = l.or(g[i], t);
            pp[i] = l.and(pp[i], pp[j]);
        };
        let mut d = 1;
        while d < n {
            let mut i = 2 * d - 1;
            while i < n {
                combine(self, &mut g, &mut pp, i, i - d);
                i += 2 * d;
            }
            d *= 2;
        }
        d /= 2;
        while d >= 1 {
            let mut i = 3 * d - 1;
            while i < n {
                combine(self, &mut g, &mut pp, i, i - d);
                i += 2 * d;
            }
            d /= 2;
        }
        let sum: Vec<Bit> = (0..n).map(|i| {
            let carry = if i == 0 { carry_in } else { g[i - 1] };
            self.xor(p[i], carry)
        }).collect();
        (sum, g[n - 1])
    }

    /// The sum of many operands of one width, modulo that width: full adders take three
    /// at a time down to two without carrying along, and one fast adder finishes.
    pub fn sum_many(&mut self, mut operands: Vec<Vec<Bit>>) -> Vec<Bit> {
        let width = operands.first().map_or(0, Vec::len);
        while operands.len() > 2 {
            let mut next = Vec::with_capacity(operands.len() * 2 / 3 + 2);
            for group in operands.chunks(3) {
                if let [x, y, z] = group {
                    let mut sum = Vec::with_capacity(width);
                    let mut carry = vec![Bit::Zero];
                    for i in 0..width {
                        let p = self.xor(x[i], y[i]);
                        sum.push(self.xor(p, z[i]));
                        if i + 1 < width {
                            let g = self.and(x[i], y[i]);
                            let t = self.and(p, z[i]);
                            carry.push(self.or(g, t));
                        }
                    }
                    next.push(sum);
                    next.push(carry);
                } else {
                    next.extend(group.iter().cloned());
                }
            }
            operands = next;
        }
        match operands.as_slice() {
            [x, y] => self.adder(x, y, Bit::Zero).0,
            [x] => x.clone(),
            _ => vec![Bit::Zero; width],
        }
    }

    pub fn adder(&mut self, a: &[Bit], b: &[Bit], carry_in: Bit) -> (Vec<Bit>, Bit) {
        if a.len().min(b.len()) >= PREFIX_FROM {
            return self.prefix_adder(a, b, carry_in);
        }
        let mut carry = carry_in;
        let mut sum = Vec::with_capacity(a.len());
        for (a, b) in a.iter().zip(b) {
            let p = self.xor(*a, *b);
            sum.push(self.xor(p, carry));
            let g = self.and(*a, *b);
            let t = self.and(p, carry);
            carry = self.or(g, t);
        }
        (sum, carry)
    }

    fn negate_if(&mut self, enable: Bit, value: &[Bit]) -> Vec<Bit> {
        let mut carry = enable;
        value
            .iter()
            .map(|bit| {
                let inverted = self.xor(*bit, enable);
                let out = self.xor(inverted, carry);
                carry = self.and(inverted, carry);
                out
            })
            .collect()
    }
}

pub(crate) fn constant(value: u32, width: usize) -> Vec<Bit> {
    (0..width).map(|i| if value.checked_shr(i as u32).unwrap_or(0) & 1 == 1 { Bit::One } else { Bit::Zero }).collect()
}

fn muldiv(gen: &mut Gen) -> String {
    let mut b = Builder::new("rv32_muldiv", "Multiplicador y divisor", "Unidad iterativa de 32 ciclos: multiplica sumando y desplazando, y divide restando y desplazando, sobre las magnitudes de los operandos.");
    let start = b.input("start", 1)[0];
    let run = b.input("run", 1)[0];
    let funct3 = b.input("funct3", 3);
    let a = b.input("a", 32);
    let d = b.input("b", 32);
    let result = b.output("result", 32);
    let last = b.output("last", 1)[0];
    let mut l = Logic { b: &mut b, next: 0 };

    l.b.group("Signo de los operandos");
    let signed_a = {
        let mulh = l.equals(&funct3, 1);
        let mulhsu = l.equals(&funct3, 2);
        let div = l.equals(&funct3, 4);
        let rem = l.equals(&funct3, 6);
        l.any(&[mulh, mulhsu, div, rem])
    };
    let signed_b = {
        let mulh = l.equals(&funct3, 1);
        let div = l.equals(&funct3, 4);
        let rem = l.equals(&funct3, 6);
        l.any(&[mulh, div, rem])
    };
    let is_div = funct3[2];
    let sign_a = l.and(signed_a, a[31]);
    let sign_b = l.and(signed_b, d[31]);
    let magnitude_a = l.negate_if(sign_a, &a);
    let magnitude_b = l.negate_if(sign_b, &d);
    let b_is_zero = {
        let any = l.any(&d);
        l.not(any)
    };
    let quotient_negative = {
        let differ = l.xor(sign_a, sign_b);
        let nonzero = l.not(b_is_zero);
        l.and(differ, nonzero)
    };
    let divide_negative = l.mux(funct3[1], quotient_negative, sign_a);
    let multiply_negative = l.xor(sign_a, sign_b);
    let negative_now = l.mux(is_div, multiply_negative, divide_negative);

    l.b.group("Estado");
    let operand = l.fresh_bus(32);
    let hi = l.fresh_bus(32);
    let lo = l.fresh_bus(32);
    let count = l.fresh_bus(5);
    let negative = l.fresh_bus(1)[0];

    l.b.group("Paso de cálculo");
    let zero33 = constant(0, 33);
    let hi_wide: Vec<Bit> = hi.iter().copied().chain([Bit::Zero]).collect();
    let shifted: Vec<Bit> = std::iter::once(lo[31]).chain(hi.iter().copied()).collect();
    let operand_wide: Vec<Bit> = operand.iter().copied().chain([Bit::Zero]).collect();
    let inverted: Vec<Bit> = operand_wide.iter().map(|v| l.not(*v)).collect();
    let add_operand = l.mux_bus(lo[0], &zero33, &operand_wide);
    let left = l.mux_bus(is_div, &hi_wide, &shifted);
    let right = l.mux_bus(is_div, &add_operand, &inverted);
    let (sum, carry) = l.adder(&left, &right, is_div);

    let divide_hi = l.mux_bus(carry, &shifted[..32], &sum[..32]);
    let divide_lo: Vec<Bit> = std::iter::once(carry).chain(lo[..31].iter().copied()).collect();
    let multiply_hi = sum[1..33].to_vec();
    let multiply_lo: Vec<Bit> = lo[1..].iter().copied().chain([sum[0]]).collect();
    let step_hi = l.mux_bus(is_div, &multiply_hi, &divide_hi);
    let step_lo = l.mux_bus(is_div, &multiply_lo, &divide_lo);

    l.b.group("Registros");
    let first_lo = l.mux_bus(is_div, &magnitude_b, &magnitude_a);
    let first_operand = l.mux_bus(is_div, &magnitude_a, &magnitude_b);
    let zero32 = constant(0, 32);
    let held_hi = l.mux_bus(run, &hi, &step_hi);
    let held_lo = l.mux_bus(run, &lo, &step_lo);
    let next_hi = l.mux_bus(start, &held_hi, &zero32);
    let next_lo = l.mux_bus(start, &held_lo, &first_lo);
    let next_operand = l.mux_bus(start, &operand, &first_operand);
    let next_negative = l.mux(start, negative, negative_now);
    l.connect_flops(&next_hi, &hi);
    l.connect_flops(&next_lo, &lo);
    l.connect_flops(&next_operand, &operand);
    l.connect_flops(&[next_negative], &[negative]);

    let mut carry = run;
    let mut advanced = Vec::new();
    for bit in &count {
        advanced.push(l.xor(*bit, carry));
        carry = l.and(*bit, carry);
    }
    let zero5 = constant(0, 5);
    let next_count = l.mux_bus(start, &advanced, &zero5);
    l.connect_flops(&next_count, &count);
    let done = l.all(&count);
    let done_running = l.and(done, run);
    l.b.or_to(done_running, Bit::Zero, last);

    l.b.group("Resultado");
    let wide: Vec<Bit> = lo.iter().chain(&hi).copied().collect();
    let product = l.negate_if(negative, &wide);
    let high_part = product[32..].to_vec();
    let is_mul_low = l.equals(&funct3, 0);
    let multiply_result = l.mux_bus(is_mul_low, &high_part, &lo);
    let chosen = l.mux_bus(funct3[1], &lo, &hi);
    let divide_result = l.negate_if(negative, &chosen);
    let output = l.mux_bus(is_div, &multiply_result, &divide_result);
    for (from, to) in output.iter().zip(&result) {
        l.b.or_to(*from, Bit::Zero, *to);
    }
    gen.add(b.finish())
}

pub(crate) fn core(gen: &mut Gen) -> String {
    let unit = muldiv(gen);
    let mut b = Builder::new("rv32_core", "CPU de control RV32IM", "Núcleo RISC-V de 32 bits que ejecuta el firmware: arranque, tokenizador y bucle principal.");
    let reset = b.input("reset", 1)[0];
    let imem_rdata = b.input("imem_rdata", 32);
    let dmem_rdata = b.input("dmem_rdata", 32);
    let imem_addr = b.output("imem_addr", 32);
    let dmem_addr = b.output("dmem_addr", 32);
    let dmem_wdata = b.output("dmem_wdata", 32);
    let dmem_we = b.output("dmem_we", 4);
    let dmem_re = b.output("dmem_re", 1)[0];
    let rf_we = b.output("rf_we", 1)[0];
    let rf_addr = b.output("rf_addr", 5);
    let rf_wdata = b.output("rf_wdata", 32);
    let halted = b.output("halted", 1)[0];

    let rs1_data = b.wire("rs1_data", 32);
    let rs2_data = b.wire("rs2_data", 32);
    let md_result = b.wire("md_result", 32);
    let md_last = b.wire("md_last", 1)[0];
    let mut l = Logic { b: &mut b, next: 0 };

    l.b.group("Control");
    let fetch = l.fresh();
    let decode = l.fresh();
    let execute = l.fresh();
    let load_wait = l.fresh();
    let md_run = l.fresh();
    let md_done = l.fresh();

    l.b.group("Instrucción");
    let enable_ir = decode;
    let ir = l.fresh_bus(32);
    let next_ir = l.mux_bus(enable_ir, &ir, &imem_rdata);
    l.connect_flops(&next_ir, &ir);

    l.b.group("Decodificación");
    let opcode = ir[0..7].to_vec();
    let funct3 = ir[12..15].to_vec();
    let bit30 = ir[30];
    let is_mext = ir[25];
    let (is_lui, is_auipc, is_jal, is_jalr, is_branch, is_load, is_store, is_opimm, is_op, is_system) = (
        l.equals(&opcode, LUI),
        l.equals(&opcode, AUIPC),
        l.equals(&opcode, JAL),
        l.equals(&opcode, JALR),
        l.equals(&opcode, BRANCH),
        l.equals(&opcode, LOAD),
        l.equals(&opcode, STORE),
        l.equals(&opcode, OP_IMM),
        l.equals(&opcode, OP),
        l.equals(&opcode, SYSTEM),
    );
    let is_jump = l.or(is_jal, is_jalr);
    let is_muldiv = l.and(is_op, is_mext);
    let is_alu = {
        let plain = l.not(is_mext);
        let reg = l.and(is_op, plain);
        l.or(reg, is_opimm)
    };
    let writes_now = l.any(&[is_lui, is_auipc, is_jump, is_alu]);

    l.b.group("Inmediato");
    let sign = ir[31];
    let imm_i: Vec<Bit> = ir[20..32].iter().copied().chain(std::iter::repeat_n(sign, 20)).collect();
    let imm_s: Vec<Bit> = ir[7..12].iter().chain(&ir[25..32]).copied().chain(std::iter::repeat_n(sign, 20)).collect();
    let imm_b: Vec<Bit> = [Bit::Zero].iter().chain(&ir[8..12]).chain(&ir[25..31]).chain(&[ir[7]]).copied().chain(std::iter::repeat_n(sign, 20)).collect();
    let imm_u: Vec<Bit> = std::iter::repeat_n(Bit::Zero, 12).chain(ir[12..32].iter().copied()).collect();
    let imm_j: Vec<Bit> = [Bit::Zero].iter().chain(&ir[21..31]).chain(&[ir[20]]).chain(&ir[12..20]).copied().chain(std::iter::repeat_n(sign, 12)).collect();
    let upper = l.or(is_lui, is_auipc);
    let mut imm = l.mux_bus(is_store, &imm_i, &imm_s);
    imm = l.mux_bus(is_branch, &imm, &imm_b);
    imm = l.mux_bus(upper, &imm, &imm_u);
    imm = l.mux_bus(is_jal, &imm, &imm_j);

    l.b.group("Contador de programa");
    let pc = l.fresh_bus(32);
    let four = constant(4, 32);
    let (pc_plus4, _) = l.adder(&pc, &four, Bit::Zero);
    for (from, to) in pc.iter().zip(&imem_addr) {
        l.b.or_to(*from, Bit::Zero, *to);
    }
    let target_from_reg = l.any(&[is_jalr, is_load, is_store]);
    let base = l.mux_bus(target_from_reg, &pc, &rs1_data);
    let (target, _) = l.adder(&base, &imm, Bit::Zero);

    l.b.group("ALU");
    let use_rs2 = l.or(is_op, is_branch);
    let right = l.mux_bus(use_rs2, &imm, &rs2_data);
    let is_add_sub = l.equals(&funct3, 0);
    let is_slt = l.equals(&funct3, 2);
    let is_sltu = l.equals(&funct3, 3);
    let wants_sub = {
        let reg_sub = {
            let t = l.and(is_op, bit30);
            l.and(t, is_add_sub)
        };
        let compare = l.or(is_slt, is_sltu);
        l.any(&[reg_sub, is_branch, compare])
    };
    let inverted: Vec<Bit> = right.iter().map(|b| l.xor(*b, wants_sub)).collect();
    let (difference, carry) = l.adder(&rs1_data, &inverted, wants_sub);
    let is_zero = {
        let any = l.any(&difference);
        l.not(any)
    };
    let less_unsigned = l.not(carry);
    let less_signed = {
        let differ = l.xor(rs1_data[31], right[31]);
        l.mux(differ, difference[31], rs1_data[31])
    };

    let amount = right[0..5].to_vec();
    let is_left = l.equals(&funct3, 1);
    let reversed: Vec<Bit> = rs1_data.iter().rev().copied().collect();
    let shift_in = l.mux_bus(is_left, &rs1_data, &reversed);
    let fill = {
        let right_shift = l.not(is_left);
        let arithmetic = l.and(bit30, right_shift);
        l.and(arithmetic, rs1_data[31])
    };
    let shifted = l.shift_right(&shift_in, &amount, fill);
    let unreversed: Vec<Bit> = shifted.iter().rev().copied().collect();
    let shift_out = l.mux_bus(is_left, &shifted, &unreversed);

    let xor_out: Vec<Bit> = rs1_data.iter().zip(&right).map(|(a, b)| l.xor(*a, *b)).collect();
    let or_out: Vec<Bit> = rs1_data.iter().zip(&right).map(|(a, b)| l.or(*a, *b)).collect();
    let and_out: Vec<Bit> = rs1_data.iter().zip(&right).map(|(a, b)| l.and(*a, *b)).collect();
    let slt_out: Vec<Bit> = std::iter::once(less_signed).chain(std::iter::repeat_n(Bit::Zero, 31)).collect();
    let sltu_out: Vec<Bit> = std::iter::once(less_unsigned).chain(std::iter::repeat_n(Bit::Zero, 31)).collect();
    let alu_out = l.select(&funct3, &[difference.clone(), shift_out.clone(), slt_out, sltu_out, xor_out, shift_out, or_out, and_out]);

    l.b.group("Saltos");
    let not_zero = l.not(is_zero);
    let not_less = l.not(less_signed);
    let not_less_u = l.not(less_unsigned);
    let taken_condition = l.select(&funct3, &[vec![is_zero], vec![not_zero], vec![Bit::Zero], vec![Bit::Zero], vec![less_signed], vec![not_less], vec![less_unsigned], vec![not_less_u]])[0];
    let taken = l.and(is_branch, taken_condition);
    let redirect = l.or(taken, is_jump);
    let target_aligned: Vec<Bit> = {
        let not_jalr = l.not(is_jalr);
        let low = l.and(target[0], not_jalr);
        std::iter::once(low).chain(target[1..].iter().copied()).collect()
    };
    let pc_next = l.mux_bus(redirect, &pc_plus4, &target_aligned);

    l.b.group("Memoria de datos");
    let byte_offset = target[0..2].to_vec();
    for (from, to) in target.iter().zip(&dmem_addr) {
        l.b.or_to(*from, Bit::Zero, *to);
    }
    let is_byte = l.equals(&funct3[0..2], 0);
    let is_half = l.equals(&funct3[0..2], 1);
    let is_word = l.equals(&funct3[0..2], 2);
    let store_active = l.and(execute, is_store);
    let offsets: Vec<Bit> = (0..4).map(|i| l.equals(&byte_offset, i)).collect();
    let upper_half = byte_offset[1];
    let lower_half = l.not(upper_half);
    let strobes = [
        (offsets[0], lower_half),
        (offsets[1], lower_half),
        (offsets[2], upper_half),
        (offsets[3], upper_half),
    ];
    for (index, (byte, half)) in strobes.iter().enumerate() {
        let single = l.and(is_byte, *byte);
        let double = l.and(is_half, *half);
        let enabled = l.any(&[single, double, is_word]);
        let strobe = l.and(enabled, store_active);
        l.b.or_to(strobe, Bit::Zero, dmem_we[index]);
    }
    let byte_lane: Vec<Bit> = rs2_data[0..8].to_vec();
    let half_lane: Vec<Bit> = rs2_data[0..16].to_vec();
    let bytes: Vec<Bit> = byte_lane.iter().cycle().take(32).copied().collect();
    let halves: Vec<Bit> = half_lane.iter().cycle().take(32).copied().collect();
    let mut store_data = l.mux_bus(is_half, &rs2_data, &halves);
    store_data = l.mux_bus(is_byte, &store_data, &bytes);
    for (from, to) in store_data.iter().zip(&dmem_wdata) {
        l.b.or_to(*from, Bit::Zero, *to);
    }
    let load_active = l.and(execute, is_load);
    l.b.or_to(load_active, Bit::Zero, dmem_re);

    let amount_bits: Vec<Bit> = vec![Bit::Zero, Bit::Zero, Bit::Zero, byte_offset[0], byte_offset[1]];
    let moved = l.shift_right(&dmem_rdata, &amount_bits, Bit::Zero);
    let unsigned_load = funct3[2];
    let signed_load = l.not(unsigned_load);
    let byte_fill = l.and(moved[7], signed_load);
    let half_fill = l.and(moved[15], signed_load);
    let byte_value: Vec<Bit> = moved[0..8].iter().copied().chain(std::iter::repeat_n(byte_fill, 24)).collect();
    let half_value: Vec<Bit> = moved[0..16].iter().copied().chain(std::iter::repeat_n(half_fill, 16)).collect();
    let mut load_value = l.mux_bus(is_half, &moved, &half_value);
    load_value = l.mux_bus(is_byte, &load_value, &byte_value);

    l.b.group("Escritura en registros");
    let mut write_value = alu_out;
    write_value = l.mux_bus(is_lui, &write_value, &imm);
    write_value = l.mux_bus(is_auipc, &write_value, &target);
    write_value = l.mux_bus(is_jump, &write_value, &pc_plus4);
    write_value = l.mux_bus(load_wait, &write_value, &load_value);
    write_value = l.mux_bus(md_done, &write_value, &md_result);
    let write_plain = l.and(execute, writes_now);
    let write_any = l.any(&[write_plain, load_wait, md_done]);
    let rd = ir[7..12].to_vec();
    let rd_nonzero = l.any(&rd);
    let write_enable = l.and(write_any, rd_nonzero);
    l.b.or_to(write_enable, Bit::Zero, rf_we);
    for (from, to) in rd.iter().zip(&rf_addr) {
        l.b.or_to(*from, Bit::Zero, *to);
    }
    for (from, to) in write_value.iter().zip(&rf_wdata) {
        l.b.or_to(*from, Bit::Zero, *to);
    }

    l.b.group("Control");
    let stays_in_execute = l.and(execute, is_system);
    let starts_long = l.or(is_load, is_muldiv);
    let finishes_now = {
        let simple = l.not(starts_long);
        let not_system = l.not(is_system);
        let t = l.and(simple, not_system);
        l.and(execute, t)
    };
    let finish = l.any(&[finishes_now, load_wait, md_done]);
    let not_reset = l.not(reset);
    let next_fetch = l.or(finish, reset);
    let next_decode = l.and(fetch, not_reset);
    let next_execute = {
        let t = l.or(decode, stays_in_execute);
        l.and(t, not_reset)
    };
    let next_load = {
        let t = l.and(execute, is_load);
        l.and(t, not_reset)
    };
    let start_md = l.and(execute, is_muldiv);
    let next_md_run = {
        let still = {
            let more = l.not(md_last);
            l.and(md_run, more)
        };
        let t = l.or(start_md, still);
        l.and(t, not_reset)
    };
    let next_md_done = {
        let t = l.and(md_run, md_last);
        l.and(t, not_reset)
    };
    l.connect_flops(&[next_fetch, next_decode, next_execute, next_load, next_md_run, next_md_done], &[fetch, decode, execute, load_wait, md_run, md_done]);
    l.b.or_to(stays_in_execute, Bit::Zero, halted);

    let zero32 = constant(0, 32);
    let updated = l.mux_bus(finish, &pc, &pc_next);
    let next_pc = l.mux_bus(reset, &updated, &zero32);
    l.connect_flops(&next_pc, &pc);

    l.b.group("Banco de registros");
    let read_rs1 = l.mux_bus(decode, &ir[15..20], &imem_rdata[15..20]);
    let read_rs2 = l.mux_bus(decode, &ir[20..25], &imem_rdata[20..25]);
    b.ram2p("registers_a", 5, 32, &[("raddr", read_rs1), ("waddr", rf_addr.clone()), ("we", vec![rf_we]), ("wdata", rf_wdata.clone()), ("rdata", rs1_data.clone())]);
    b.ram2p("registers_b", 5, 32, &[("raddr", read_rs2), ("waddr", rf_addr.clone()), ("we", vec![rf_we]), ("wdata", rf_wdata.clone()), ("rdata", rs2_data.clone())]);

    b.group("Multiplicación y división");
    b.inst("muldiv", &unit, &[("start", vec![start_md]), ("run", vec![md_run]), ("funct3", funct3), ("a", rs1_data), ("b", rs2_data), ("result", md_result), ("last", vec![md_last])]);
    gen.add(b.finish())
}

/// The control CPU on its own, with its memory buses as ports.
pub fn core_design() -> Design {
    let mut gen = Gen { modules: BTreeMap::new() };
    let top = core(&mut gen);
    Design { top, modules: gen.modules }
}

/// What the CPU reads and writes, in words.
pub trait Rv32Bus {
    /// Called before each data access with the cycle it happens in.
    fn at_cycle(&mut self, _cycle: u64) {}
    fn fetch(&mut self, address: u32) -> u32;
    fn read(&mut self, address: u32) -> u32;
    /// `strobe` has one bit per byte of `data`.
    fn write(&mut self, address: u32, data: u32, strobe: u8);
}

/// A register write: the cycle it happens in, the register and the value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Retired {
    pub cycle: u64,
    pub rd: u32,
    pub value: u32,
}

/// Instruction-set simulator. Times every instruction exactly as the gates do.
#[derive(Clone, Debug, Default)]
pub struct Rv32Iss {
    pub pc: u32,
    pub regs: [u32; 32],
    pub cycle: u64,
    pub halted: bool,
}

fn muldiv_result(funct3: u32, a: u32, b: u32) -> u32 {
    let (sa, sb) = (a as i32 as i64, b as i32 as i64);
    match funct3 {
        0 => a.wrapping_mul(b),
        1 => ((sa * sb) >> 32) as u32,
        2 => (((sa as i128) * (b as i128)) >> 32) as u32,
        3 => (((a as u64) * (b as u64)) >> 32) as u32,
        4 if b == 0 => u32::MAX,
        4 => (a as i32).wrapping_div(b as i32) as u32,
        5 => a.checked_div(b).unwrap_or(u32::MAX),
        6 if b == 0 => a,
        6 => (a as i32).wrapping_rem(b as i32) as u32,
        _ => a.checked_rem(b).unwrap_or(a),
    }
}

impl Rv32Iss {
    /// Runs one instruction and returns the register it wrote, if any.
    pub fn step(&mut self, bus: &mut impl Rv32Bus) -> Option<Retired> {
        if self.halted {
            self.cycle += 1;
            return None;
        }
        let ir = bus.fetch(self.pc);
        let opcode = ir & 0x7F;
        let funct3 = ir >> 12 & 7;
        let (rs1, rs2, rd) = (self.regs[(ir >> 15 & 31) as usize], self.regs[(ir >> 20 & 31) as usize], ir >> 7 & 31);
        let imm_i = (ir as i32 >> 20) as u32;
        let imm_s = ((ir as i32 >> 20) as u32 & !31) | (ir >> 7 & 31);
        let imm_b = ((ir as i32 >> 19) as u32 & !0xFFF) | (ir << 4 & 0x800) | (ir >> 20 & 0x7E0) | (ir >> 7 & 0x1E);
        let imm_u = ir & 0xFFFF_F000;
        let imm_j = ((ir as i32 >> 11) as u32 & !0xF_FFFF) | (ir & 0xF_F000) | (ir >> 9 & 0x800) | (ir >> 20 & 0x7FE);
        let pc4 = self.pc.wrapping_add(4);
        let mut next = pc4;
        let mut write = None;
        let mut extra = 0;
        match opcode {
            LUI => write = Some(imm_u),
            AUIPC => write = Some(self.pc.wrapping_add(imm_u)),
            JAL => {
                write = Some(pc4);
                next = self.pc.wrapping_add(imm_j);
            }
            JALR => {
                write = Some(pc4);
                next = rs1.wrapping_add(imm_i) & !1;
            }
            BRANCH => {
                let taken = match funct3 {
                    0 => rs1 == rs2,
                    1 => rs1 != rs2,
                    4 => (rs1 as i32) < rs2 as i32,
                    5 => rs1 as i32 >= rs2 as i32,
                    6 => rs1 < rs2,
                    7 => rs1 >= rs2,
                    _ => false,
                };
                if taken {
                    next = self.pc.wrapping_add(imm_b);
                }
            }
            LOAD => {
                let address = rs1.wrapping_add(imm_i);
                bus.at_cycle(self.cycle + EXECUTE_CYCLE);
                let moved = bus.read(address & !3) >> (8 * (address & 3));
                write = Some(match funct3 {
                    0 => moved as u8 as i8 as i32 as u32,
                    1 => moved as u16 as i16 as i32 as u32,
                    4 => moved & 0xFF,
                    5 => moved & 0xFFFF,
                    _ => moved,
                });
                extra = 1;
            }
            STORE => {
                let address = rs1.wrapping_add(imm_s);
                bus.at_cycle(self.cycle + EXECUTE_CYCLE);
                let offset = address & 3;
                let (data, strobe) = match funct3 & 3 {
                    0 => ((rs2 & 0xFF) * 0x0101_0101, 1u8 << offset),
                    1 => ((rs2 & 0xFFFF) * 0x0001_0001, 3u8 << (offset & 2)),
                    _ => (rs2, 15),
                };
                bus.write(address & !3, data, strobe);
            }
            OP if ir >> 25 & 1 == 1 => {
                write = Some(muldiv_result(funct3, rs1, rs2));
                extra = MULDIV_CYCLES;
            }
            OP | OP_IMM => {
                let right = if opcode == OP { rs2 } else { imm_i };
                let bit30 = ir >> 30 & 1 == 1;
                let amount = right & 31;
                write = Some(match funct3 {
                    0 if opcode == OP && bit30 => rs1.wrapping_sub(right),
                    0 => rs1.wrapping_add(right),
                    1 => rs1 << amount,
                    2 => ((rs1 as i32) < right as i32) as u32,
                    3 => (rs1 < right) as u32,
                    4 => rs1 ^ right,
                    5 if bit30 => (rs1 as i32 >> amount) as u32,
                    5 => rs1 >> amount,
                    6 => rs1 | right,
                    _ => rs1 & right,
                });
            }
            SYSTEM => {
                self.halted = true;
                self.cycle += EXECUTE_CYCLE + 1;
                return None;
            }
            _ => {}
        }
        let at = self.cycle + EXECUTE_CYCLE + extra;
        self.pc = next;
        self.cycle = at + 1;
        match write {
            Some(value) if rd != 0 => {
                self.regs[rd as usize] = value;
                Some(Retired { cycle: at, rd, value })
            }
            _ => None,
        }
    }
}
