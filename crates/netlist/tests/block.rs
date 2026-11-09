//! Attention and the delta rule, run as microprograms on the gates. Each result is
//! checked against the integer behavioural model and against ordinary floating point.

use netlist::block::{exponent_register, tables, Alu, Assembly, AttentionRegs, Behavior, BlockOps, DeltaRegs, FN_SIGMOID, FN_SOFTPLUS};
use netlist::die::{OP_ADD, OP_FUNCTION, OP_IMMEDIATE, OP_LZC, OP_MAX, OP_MOV, OP_MUL, OP_PARITY, OP_SHIFT_LEFT, OP_SUB};
use netlist::demo::{attention_demo, block_config, block_host_steps, delta_demo, read_block_value};
use netlist::{to_verilog, verilog_testbench, TestStep, die_design, CellConfig, DieConfig, DieProgram, Simulator};

fn config() -> DieConfig {
    DieConfig {
        cells: 1,
        cell: CellConfig { weight_bits: 8, activation_bits: 16, accumulator_bits: 32, mac_stages: 1, adder_stages: 1, addr_bits: 3, coding: Default::default(), exponent_bits: 0, multiplier: netlist::multiplier::MultiplierKind::Array, fixed_weights: false },
        scale_bits: 16,
        reg_bits: 4,
        index_bits: 6,
        microcode_bits: 14,
        broadcast_stages: 1,
        function_bits: 8,
        function_select_bits: 3,
    }
}

fn mask(v: i64, bits: u32) -> u128 {
    (v as u64 as u128) & ((1u128 << bits) - 1)
}

/// Loads the program through the host ports, runs it, and returns the simulator.
fn run_gates(c: DieConfig, asm: &Assembly) -> Simulator {
    let mut program = DieProgram::new(c);
    program.assemble(&asm.steps);
    let words = program.finish();
    assert!(words.len() < 1 << c.microcode_bits, "microprograma demasiado largo: {}", words.len());

    let mut sim = Simulator::new(&die_design(c)).unwrap();
    sim.set("start", 1).unwrap();
    sim.tick();
    sim.set("load_code", 1).unwrap();
    for (i, word) in words.iter().enumerate() {
        sim.set("code_addr", i as u64).unwrap();
        sim.set_wide("code_data", word.encode(&c)).unwrap();
        sim.tick();
    }
    sim.set("load_code", 0).unwrap();
    sim.set("load_vec", 1).unwrap();
    for ((reg, index), value) in &asm.initial {
        sim.set("vec_addr", ((*reg as u64) << c.index_bits) | *index as u64).unwrap();
        sim.set_wide("vec_data", mask(*value, 16)).unwrap();
        sim.tick();
    }
    sim.set("load_vec", 0).unwrap();
    sim.set("load_fn", 1).unwrap();
    for (address, value) in tables(&c).iter().enumerate() {
        sim.set("fn_addr", address as u64).unwrap();
        sim.set_wide("cell_data", mask(*value, 16)).unwrap();
        sim.tick();
    }
    sim.set("load_fn", 0).unwrap();
    sim.tick();
    sim.set("start", 0).unwrap();
    let mut cycles = 0;
    while sim.get("halted").unwrap() == 0 {
        sim.tick();
        cycles += 1;
        assert!(cycles < 2_000_000, "el secuenciador no se detiene");
    }
    sim
}

fn gate_value(sim: &mut Simulator, c: DieConfig, reg: u32, index: u32) -> i64 {
    sim.set("vec_addr", ((reg as u64) << c.index_bits) | index as u64).unwrap();
    sim.tick();
    ((sim.get("vec_out").unwrap() << 48) as i64) >> 48
}

/// Gates and behavioural model must agree on every result mantissa and exponent.
fn assert_same(sim: &mut Simulator, model: &Behavior, c: DieConfig, reg: u32, len: u32) {
    for i in 0..len {
        assert_eq!(gate_value(sim, c, reg, i), model.mantissa((reg, i)), "registro {reg}, posición {i}");
    }
    let exponents = exponent_register(&c);
    assert_eq!(gate_value(sim, c, exponents, reg), model.mantissa((exponents, reg)), "exponente de {reg}");
}

fn close(a: f64, b: f64, tolerance: f64, what: &str) {
    assert!((a - b).abs() <= tolerance * b.abs().max(0.05), "{what}: {a} frente a {b}");
}

#[test]
fn block_multiply_and_add_track_real_numbers() {
    let c = config();
    let mut asm = Assembly::new(&c);
    let a = [0.75, -1.5, 3.25, 0.125];
    let b = [100.0, 0.5, -2.0, 8.0];
    asm.load_vector(0, &a);
    asm.load_vector(1, &b);
    asm.multiply(0, 1, 2, 4.into());
    asm.add(0, 1, 3, 4.into());

    let mut sim = run_gates(c, &asm);
    let mut model = Behavior::new(&c, &asm.initial);
    model.run(&asm.steps);
    assert_same(&mut sim, &model, c, 2, 4);
    assert_same(&mut sim, &model, c, 3, 4);
    for i in 0..4 {
        close(model.value(2, i as u32), a[i] * b[i], 0.002, "producto");
        close(model.value(3, i as u32), a[i] + b[i], 0.002, "suma");
    }
}

#[test]
fn norms_and_function_tables_work_in_gates() {
    let c = config();
    let mut asm = Assembly::new(&c);
    let x = [3.0, -4.0, 1.5, 0.5];
    asm.load_vector(0, &x);
    asm.rms_normalize(0, 1, 4, (2, 3), None, None);
    asm.l2_normalize(0, 4, 4, (5, 6), None);
    asm.function(FN_SIGMOID, 0, 7, 4.into());
    asm.function(FN_SOFTPLUS, 0, 8, 4.into());

    let mut sim = run_gates(c, &asm);
    let mut model = Behavior::new(&c, &asm.initial);
    model.run(&asm.steps);
    for (reg, len) in [(1, 4), (4, 4), (7, 4), (8, 4)] {
        assert_same(&mut sim, &model, c, reg, len);
    }
    let mean = x.iter().map(|v| v * v).sum::<f64>() / 4.0;
    let norm = x.iter().map(|v| v * v).sum::<f64>().sqrt();
    for (i, v) in x.iter().enumerate() {
        close(model.value(1, i as u32), v / mean.sqrt(), 0.03, "RMS");
        close(model.value(4, i as u32), v / norm, 0.03, "L2");
        close(model.value(7, i as u32), 1.0 / (1.0 + (-v).exp()), 0.05, "sigmoide");
        close(model.value(8, i as u32), (1.0 + v.exp()).ln(), 0.08, "softplus");
    }
}

#[test]
fn softmax_attention_matches_floating_point() {
    let c = config();
    let width = 4usize;
    let tokens = 3usize;
    let q = [0.4, -0.2, 0.7, 0.1];
    let k = [0.3, 0.1, 0.9, -0.2, -0.5, 0.4, 0.2, 0.6, 0.8, -0.3, -0.1, 0.25];
    let v = [1.2, -0.7, 0.3, 2.0, -1.1, 0.4, 0.9, -0.6, 0.5, 1.5, -2.1, 0.2];
    let ones = vec![1.0; tokens];

    let mut asm = Assembly::new(&c);
    asm.load_vector(0, &q);
    asm.load_vector(1, &k);
    asm.load_vector(2, &v);
    asm.load_vector(3, &ones);
    let regs = AttentionRegs { scores: 5, top: 6, shifted: 7, weights: 8, total: 9, inverse: 10, probabilities: 11 };
    asm.attention(0, 1, 2, 3, 4, width as u32, (tokens as u32).into(), None, regs);

    let mut sim = run_gates(c, &asm);
    let mut model = Behavior::new(&c, &asm.initial);
    model.run(&asm.steps);
    assert_same(&mut sim, &model, c, 4, width as u32);
    assert_same(&mut sim, &model, c, 11, tokens as u32);

    let scores: Vec<f64> = (0..tokens).map(|t| (0..width).map(|j| q[j] * k[t * width + j]).sum()).collect();
    let top = scores.iter().copied().fold(f64::MIN, f64::max);
    let weights: Vec<f64> = scores.iter().map(|s| (s - top).exp()).collect();
    let total: f64 = weights.iter().sum();
    let p: Vec<f64> = weights.iter().map(|w| w / total).collect();
    for (t, probability) in p.iter().enumerate() {
        close(model.value(11, t as u32), *probability, 0.08, "probabilidad");
    }
    for j in 0..width {
        let expected: f64 = (0..tokens).map(|t| p[t] * v[t * width + j]).sum();
        close(model.value(4, j as u32), expected, 0.10, "atención");
    }
}

#[test]
fn delta_rule_matches_floating_point() {
    let c = config();
    let (key, value) = (3usize, 2usize);
    let start = [0.5, -0.25, 0.1, 0.4, -0.3, 0.2];
    let k = [0.6, -0.48, 0.64];
    let q = [0.2, 0.5, -0.3];
    let v = [1.1, -0.6];
    let (beta, decay) = (0.75f64, 0.9f64);

    let mut asm = Assembly::new(&c);
    asm.load_vector(0, &start);
    asm.load_vector(1, &k);
    asm.load_vector(2, &q);
    asm.load_vector(3, &v);
    asm.load_vector(4, &[beta]);
    asm.load_vector(5, &[decay]);
    let regs = DeltaRegs { state: 0, key: 1, query: 2, value: 3, beta: 4, decay: 5, memory: 6, error: 7, delta: 8, update: 9, next: 10, out: 11 };
    asm.delta_step(regs, key as u32, value as u32);

    let mut sim = run_gates(c, &asm);
    let mut model = Behavior::new(&c, &asm.initial);
    model.run(&asm.steps);
    assert_same(&mut sim, &model, c, 0, 6);
    assert_same(&mut sim, &model, c, 11, 2);

    let mut s: Vec<f64> = start.iter().map(|x| x * decay).collect();
    let memory: Vec<f64> = (0..value).map(|j| (0..key).map(|i| s[i * value + j] * k[i]).sum()).collect();
    let delta: Vec<f64> = (0..value).map(|j| (v[j] - memory[j]) * beta).collect();
    for i in 0..key {
        for j in 0..value {
            s[i * value + j] += k[i] * delta[j];
        }
    }
    let out: Vec<f64> = (0..value).map(|j| (0..key).map(|i| s[i * value + j] * q[i]).sum()).collect();
    for (i, expected) in s.iter().enumerate() {
        close(model.value(0, i as u32), *expected, 0.01, "estado");
    }
    for (j, expected) in out.iter().enumerate() {
        close(model.value(11, j as u32), *expected, 0.01, "salida");
    }
}

#[test]
fn every_single_operation_matches_the_model() {
    let c = config();
    let mut asm = Assembly::new(&c);
    asm.initial.insert((0, 0), 6144);
    asm.initial.insert((0, 1), -1234);
    asm.initial.insert((1, 0), 25600);
    asm.initial.insert((1, 1), 77);
    asm.initial.insert((2, 0), 3);
    asm.initial.insert((2, 1), 2);

    let mk = |op: u32, a: (u32, u32), b: (u32, u32), d: u32| {
        let mut s = Alu::new(op);
        s.a = a;
        s.b = b;
        s.dst = (3, d);
        s
    };
    let mut steps = vec![
        mk(OP_ADD, (0, 0), (1, 1), 0),
        mk(OP_SUB, (0, 1), (1, 0), 1),
        { let mut s = mk(OP_MUL, (0, 0), (1, 0), 2); s.shift = 15; s },
        mk(OP_MAX, (0, 1), (1, 1), 3),
        mk(OP_MOV, (0, 1), (0, 0), 4),
        { let mut s = mk(OP_IMMEDIATE, (0, 0), (0, 0), 5); s.literal = Some(0x3D); s },
        mk(OP_PARITY, (1, 1), (0, 0), 6),
        { let mut s = mk(OP_FUNCTION, (0, 1), (0, 0), 7); s.function = 1; s },
        { let mut s = mk(OP_MOV, (2, 0), (0, 0), 8); s.load_shift_a = true; s },
        mk(OP_MOV, (0, 0), (0, 0), 9),
        { let mut s = mk(OP_MOV, (2, 1), (0, 0), 10); s.load_shift_left = true; s },
        mk(OP_SHIFT_LEFT, (0, 0), (0, 0), 11),
        mk(OP_SHIFT_LEFT, (1, 0), (0, 0), 12),
        { let mut s = mk(OP_MOV, (0, 1), (0, 0), 13); s.magnitude_clear = true; s.magnitude_scan = true; s.write = false; s },
        mk(OP_LZC, (0, 0), (0, 0), 14),
        { let mut s = mk(OP_MOV, (0, 0), (1, 0), 15); s.accumulator_clear = true; s.accumulate = true; s.write = false; s },
        { let mut s = mk(OP_MOV, (0, 1), (1, 1), 16); s.accumulate = true; s.write = false; s },
        { let mut s = mk(OP_MOV, (0, 0), (0, 0), 17); s.wide_write = true; s.write = false; s.dst = (0, 4); s },
        { let mut s = mk(OP_MOV, (0, 4), (0, 0), 17); s.wide_clear = true; s.wide_scan = true; s.write = false; s },
        mk(netlist::die::OP_WIDE_LZC, (0, 0), (0, 0), 17),
    ];
    steps.retain(|_| true);
    asm.steps = steps.into_iter().map(Into::into).collect();

    let mut sim = run_gates(c, &asm);
    let mut model = Behavior::new(&c, &asm.initial);
    model.run(&asm.steps);
    let mut wrong = Vec::new();
    for d in 0..18 {
        let (gate, expected) = (gate_value(&mut sim, c, 3, d), model.mantissa((3, d)));
        if gate != expected {
            wrong.push((d, gate, expected));
        }
    }
    assert!(wrong.is_empty(), "posición, puertas, modelo: {wrong:?}");
}

fn execute_demo(c: DieConfig, steps: &[TestStep]) -> Simulator {
    let mut sim = Simulator::new(&die_design(c)).unwrap();
    for step in steps {
        for (name, value) in &step.inputs {
            sim.set_wide(name, *value).unwrap();
        }
        sim.tick();
    }
    sim.set("start", 0).unwrap();
    let mut cycles = 0;
    while sim.get("halted").unwrap() == 0 {
        sim.tick();
        cycles += 1;
        assert!(cycles < 2_000_000);
    }
    sim
}

#[test]
fn app_demos_match_model_and_floating_point() {
    let c = block_config();
    for demo in [attention_demo(&c), delta_demo(&c)] {
        let mut sim = execute_demo(c, &block_host_steps(c, &demo.assembly));
        let mut model = Behavior::new(&c, &demo.assembly.initial);
        model.run(&demo.assembly.steps);
        for i in 0..demo.output_len {
            let (mantissa, value) = read_block_value(&mut sim, c, demo.output_reg, i);
            assert_eq!(mantissa, model.mantissa((demo.output_reg, i)));
            close(value, demo.expected[i as usize], demo.tolerance, "demostración");
        }
    }
}

#[test]
fn attention_in_verilog_matches_the_gate_simulator() {
    if std::process::Command::new("iverilog").arg("-V").output().is_err() {
        return;
    }
    let c = block_config();
    let demo = attention_demo(&c);
    let design = die_design(c);
    let mut steps = block_host_steps(c, &demo.assembly);
    let mut sim = execute_demo(c, &steps);
    let running = sim.cycle() as usize - steps.len();
    steps.push(TestStep { inputs: vec![("start".into(), 0)] });
    for _ in 1..running {
        steps.push(TestStep { inputs: vec![] });
    }
    let mut ours = Vec::new();
    let skip = steps.len();
    for i in 0..demo.output_len {
        let address = ((demo.output_reg as u64) << c.index_bits) | i as u64;
        steps.push(TestStep { inputs: vec![("vec_addr".into(), address as u128)] });
        sim.set("vec_addr", address).unwrap();
        sim.tick();
        ours.push(sim.get("vec_out").unwrap().to_string());
    }
    let dir = std::env::temp_dir().join(format!("llm2chip-block-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let source = dir.join("attention.v");
    let binary = dir.join("attention.vvp");
    std::fs::write(&source, format!("{}\n{}", to_verilog(&design), verilog_testbench(&design, &steps, "vec_out"))).unwrap();
    let compile = std::process::Command::new("iverilog").args(["-g2005", "-s", "testbench", "-o"]).arg(&binary).arg(&source).output().unwrap();
    assert!(compile.status.success(), "{}", String::from_utf8_lossy(&compile.stderr));
    let out = std::process::Command::new("vvp").arg(&binary).output().unwrap();
    let theirs: Vec<String> = String::from_utf8_lossy(&out.stdout).lines().filter(|l| l.parse::<u64>().is_ok()).map(str::to_string).collect();
    assert_eq!(theirs[skip..], ours[..]);
    std::fs::remove_dir_all(&dir).ok();
}
