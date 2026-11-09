//! A worked example the die executes: one gated MLP layer with a residual.
//! Used by the tests and shown live in the app.

use crate::block::{exponent_register, tables, Assembly, AttentionRegs, BlockOps, DeltaRegs};
use crate::verilog::TestStep;
use crate::{CellConfig, ControlWord, DieConfig, DieProgram, Simulator};

fn mask(v: i64, bits: u32) -> u64 {
    (v as u64) & ((1u64 << bits) - 1)
}

fn wrap(v: i128, bits: u32) -> i64 {
    let shift = 128 - bits;
    ((v << shift) >> shift) as i64
}

pub struct Example {
    pub x: Vec<i64>,
    /// `[matrix][row][column]` of 8-bit weights.
    pub matrices: Vec<Vec<Vec<i64>>>,
    /// `[matrix][row][group]` of (scale, bias).
    pub factors: Vec<Vec<Vec<(i64, i64)>>>,
    pub groups: Vec<usize>,
}

pub fn example() -> Example {
    Example {
        x: vec![1200, -700, 350, 90],
        matrices: vec![
            vec![vec![12, -7, 3, 25], vec![-30, 4, 11, -2], vec![5, 5, -9, 17]],
            vec![vec![-1, 8, 22, -13], vec![6, -6, 2, 9], vec![31, -14, 1, 3]],
            vec![vec![9, -4, 2], vec![-5, 13, 7], vec![1, 3, -11]],
        ],
        factors: vec![
            vec![vec![(3, 1), (2, -1)], vec![(1, 2), (4, 0)], vec![(2, 0), (1, 1)]],
            vec![vec![(1, 0), (2, 3)], vec![(5, -2), (1, 1)], vec![(1, 1), (3, 0)]],
            vec![vec![(2, 1)], vec![(1, -1)], vec![(3, 0)]],
        ],
        groups: vec![2, 2, 3],
    }
}

/// The table holds `silu`-like values: here simply `index · 32`, signed.
fn table_value(index: usize) -> i64 {
    (index as i8 as i64) * 32
}

pub fn matvec_ref(m: &[Vec<i64>], factors: &[Vec<(i64, i64)>], group: usize, x: &[i64], shift: u32) -> Vec<i64> {
    m.iter()
        .zip(factors)
        .map(|(row, f)| {
            let mut total: i128 = 0;
            for (g, (scale, bias)) in f.iter().enumerate() {
                let (mut dot, mut sum) = (0i128, 0i128);
                for j in 0..group {
                    dot += row[g * group + j] as i128 * x[g * group + j] as i128;
                    sum += x[g * group + j] as i128;
                }
                total += dot * *scale as i128 + sum * *bias as i128;
            }
            wrap(total >> shift, 16)
        })
        .collect()
}

pub fn reference(e: &Example) -> Vec<i64> {
    let gate = matvec_ref(&e.matrices[0], &e.factors[0], e.groups[0], &e.x, 2);
    let up = matvec_ref(&e.matrices[1], &e.factors[1], e.groups[1], &e.x, 2);
    let silu: Vec<i64> = gate.iter().map(|v| table_value(((mask(*v, 16) >> 8) & 0xFF) as usize)).collect();
    let product: Vec<i64> = silu.iter().zip(&up).map(|(a, b)| crate::block::round_right(*a as i128 * *b as i128, 8, 16)).collect();
    let down = matvec_ref(&e.matrices[2], &e.factors[2], e.groups[2], &product, 3);
    down.iter().zip(&e.x).map(|(a, b)| wrap(*a as i128 + *b as i128, 16)).collect()
}

pub fn program(c: DieConfig) -> Vec<ControlWord> {
    let mut p = DieProgram::new(c);
    p.restart_cells();
    p.matvec(0, 1, 2, 2, 2);
    p.matvec(0, 2, 2, 2, 2);
    p.function(1, 3, 3);
    p.mul(3, 2, 8, 4, 3);
    p.matvec(4, 5, 3, 1, 3);
    p.add(5, 0, 6, 3);
    p.finish()
}

pub fn host_steps(c: DieConfig, e: &Example) -> Vec<TestStep> {
    let step = |pairs: Vec<(&str, u128)>| TestStep { inputs: pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect() };
    let mut steps = vec![step(vec![("start", 1)])];
    for (i, word) in program(c).iter().enumerate() {
        steps.push(step(vec![("load_code", 1), ("code_addr", i as u128), ("code_data", word.encode(&c))]));
    }
    steps.push(step(vec![("load_code", 0)]));
    for (i, x) in e.x.iter().enumerate() {
        steps.push(step(vec![("load_vec", 1), ("vec_addr", i as u128), ("vec_data", mask(*x, 16) as u128)]));
    }
    steps.push(step(vec![("load_vec", 0)]));
    for index in 0..256usize {
        steps.push(step(vec![("load_fn", 1), ("fn_addr", index as u128), ("cell_data", mask(table_value(index), 16) as u128)]));
    }
    steps.push(step(vec![("load_fn", 0)]));
    if !c.cell.fixed_weights {
        for row in 0..3usize {
            let mut address = 0u64;
            let mut group_address = 0u64;
            for (m, matrix) in e.matrices.iter().enumerate() {
                for w in &matrix[row] {
                    steps.push(step(vec![("load_cell", 1 << row), ("load_scale", 0), ("cell_addr", address as u128), ("cell_data", mask(*w, 8) as u128)]));
                    address += 1;
                }
                for (scale, bias) in &e.factors[m][row] {
                    steps.push(step(vec![("load_cell", 1 << row), ("load_scale", 1), ("cell_addr", group_address as u128 * 2), ("cell_data", mask(*scale, 16) as u128)]));
                    steps.push(step(vec![("load_cell", 1 << row), ("load_scale", 1), ("cell_addr", group_address as u128 * 2 + 1), ("cell_data", mask(*bias, 16) as u128)]));
                    group_address += 1;
                }
            }
        }
        steps.push(step(vec![("load_cell", 0), ("load_scale", 0)]));
    }
    steps.push(step(vec![("start", 1)]));
    steps
}

/// Register the result is left in, and its length.
pub const RESULT_REG: u64 = 6;
pub const RESULT_LEN: u64 = 3;

/// What a mask would fix in each cell: weights in `weight_rom`, factors in their memories.
pub fn rom_words(e: &Example) -> Vec<(String, usize, u128)> {
    let mut words = Vec::new();
    for row in 0..3usize {
        let (mut address, mut group) = (0, 0);
        for (m, matrix) in e.matrices.iter().enumerate() {
            for w in &matrix[row] {
                words.push((format!("/cell{row}/compute/weight_rom"), address, mask(*w, 8) as u128));
                address += 1;
            }
            for (scale, bias) in &e.factors[m][row] {
                words.push((format!("/cell{row}/scale_mem"), group, mask(*scale, 16) as u128));
                words.push((format!("/cell{row}/bias_mem"), group, mask(*bias, 16) as u128));
                group += 1;
            }
        }
    }
    words
}

/// Applies the loading steps, leaving the die ready to run once `start` falls.
pub fn load(sim: &mut Simulator, config: DieConfig, example: &Example) {
    if config.cell.fixed_weights {
        for (path, address, value) in rom_words(example) {
            sim.poke_memory(&path, address, value).expect("memoria del die");
        }
    }
    for step in host_steps(config, example) {
        for (name, value) in &step.inputs {
            sim.set_wide(name, *value).expect("puerto del die");
        }
        sim.tick();
    }
}

pub fn read_register(sim: &mut Simulator, config: DieConfig, register: u64, index: u64) -> i64 {
    sim.set("vec_addr", (register << config.index_bits) | index).expect("puerto del die");
    sim.tick();
    let value = sim.get("vec_out").expect("puerto del die");
    let unused = 64 - config.cell.activation_bits;
    ((value << unused) as i64) >> unused
}

/// One of the block floating point programs, ready to run on the die.
pub struct BlockDemo {
    pub assembly: Assembly,
    pub output_reg: u32,
    pub output_len: u32,
    pub expected: Vec<f64>,
    pub tolerance: f64,
}

/// The die configuration the block floating point programs need.
pub fn block_config() -> DieConfig {
    DieConfig {
        cells: 1,
        cell: CellConfig { weight_bits: 8, activation_bits: 16, accumulator_bits: 32, mac_stages: 1, adder_stages: 1, addr_bits: 3, coding: Default::default(), exponent_bits: 0, multiplier: crate::multiplier::MultiplierKind::Array, fixed_weights: false, ecc: false },
        scale_bits: 16,
        reg_bits: 4,
        index_bits: 6,
        microcode_bits: 14,
        broadcast_stages: 1,
        function_bits: 8,
        function_select_bits: 3,
    }
}

/// Softmax attention over three cached tokens.
pub fn attention_demo(c: &DieConfig) -> BlockDemo {
    let (width, tokens) = (4usize, 3usize);
    let q = [0.4, -0.2, 0.7, 0.1];
    let k = [0.3, 0.1, 0.9, -0.2, -0.5, 0.4, 0.2, 0.6, 0.8, -0.3, -0.1, 0.25];
    let v = [1.2, -0.7, 0.3, 2.0, -1.1, 0.4, 0.9, -0.6, 0.5, 1.5, -2.1, 0.2];
    let mut assembly = Assembly::new(c);
    assembly.load_vector(0, &q);
    assembly.load_vector(1, &k);
    assembly.load_vector(2, &v);
    assembly.load_vector(3, &vec![1.0; tokens]);
    let regs = AttentionRegs { scores: 5, top: 6, shifted: 7, weights: 8, total: 9, inverse: 10, probabilities: 11 };
    assembly.attention(0, 1, 2, 3, 4, width as u32, (tokens as u32).into(), None, regs);

    let scores: Vec<f64> = (0..tokens).map(|t| (0..width).map(|j| q[j] * k[t * width + j]).sum()).collect();
    let top = scores.iter().copied().fold(f64::MIN, f64::max);
    let weights: Vec<f64> = scores.iter().map(|s| (s - top).exp()).collect();
    let total: f64 = weights.iter().sum();
    let expected = (0..width).map(|j| (0..tokens).map(|t| weights[t] / total * v[t * width + j]).sum()).collect();
    BlockDemo { assembly, output_reg: 4, output_len: width as u32, expected, tolerance: 0.10 }
}

/// One Gated DeltaNet head, updating its state and producing its output.
pub fn delta_demo(c: &DieConfig) -> BlockDemo {
    let (key, value) = (3usize, 2usize);
    let start = [0.5, -0.25, 0.1, 0.4, -0.3, 0.2];
    let k = [0.6, -0.48, 0.64];
    let q = [0.2, 0.5, -0.3];
    let v = [1.1, -0.6];
    let (beta, decay) = (0.75f64, 0.9f64);
    let mut assembly = Assembly::new(c);
    assembly.load_vector(0, &start);
    assembly.load_vector(1, &k);
    assembly.load_vector(2, &q);
    assembly.load_vector(3, &v);
    assembly.load_vector(4, &[beta]);
    assembly.load_vector(5, &[decay]);
    let regs = DeltaRegs { state: 0, key: 1, query: 2, value: 3, beta: 4, decay: 5, memory: 6, error: 7, delta: 8, update: 9, next: 10, out: 11 };
    assembly.delta_step(regs, key as u32, value as u32);

    let mut s: Vec<f64> = start.iter().map(|x| x * decay).collect();
    let memory: Vec<f64> = (0..value).map(|j| (0..key).map(|i| s[i * value + j] * k[i]).sum()).collect();
    let delta: Vec<f64> = (0..value).map(|j| (v[j] - memory[j]) * beta).collect();
    for i in 0..key {
        for j in 0..value {
            s[i * value + j] += k[i] * delta[j];
        }
    }
    let expected = (0..value).map(|j| (0..key).map(|i| s[i * value + j] * q[i]).sum()).collect();
    BlockDemo { assembly, output_reg: 11, output_len: value as u32, expected, tolerance: 0.01 }
}

/// The host-port steps that load a block program, its registers and its tables.
pub fn block_host_steps(c: DieConfig, assembly: &Assembly) -> Vec<TestStep> {
    let step = |pairs: Vec<(&str, u128)>| TestStep { inputs: pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect() };
    let mut program = DieProgram::new(c);
    program.assemble(&assembly.steps);
    let mut steps = vec![step(vec![("start", 1)])];
    for (i, word) in program.finish().iter().enumerate() {
        steps.push(step(vec![("load_code", 1), ("code_addr", i as u128), ("code_data", word.encode(&c))]));
    }
    steps.push(step(vec![("load_code", 0)]));
    let mut places: Vec<_> = assembly.initial.iter().collect();
    places.sort();
    for ((reg, index), value) in places {
        let address = ((*reg as u128) << c.index_bits) | *index as u128;
        steps.push(step(vec![("load_vec", 1), ("vec_addr", address), ("vec_data", mask(*value, 16) as u128)]));
    }
    steps.push(step(vec![("load_vec", 0)]));
    for (address, value) in tables(&c).iter().enumerate() {
        steps.push(step(vec![("load_fn", 1), ("fn_addr", address as u128), ("cell_data", mask(*value, 16) as u128)]));
    }
    steps.push(step(vec![("load_fn", 0)]));
    steps.push(step(vec![("start", 1)]));
    steps
}

/// A result mantissa read back through the host port, and its real value.
pub fn read_block_value(sim: &mut Simulator, c: DieConfig, reg: u32, index: u32) -> (i64, f64) {
    let mantissa = read_register(sim, c, reg as u64, index as u64);
    let exponent = read_register(sim, c, exponent_register(&c) as u64, reg as u64);
    (mantissa, mantissa as f64 / 2f64.powi(c.cell.activation_bits as i32 - 1) * 2f64.powi(exponent as i32))
}
