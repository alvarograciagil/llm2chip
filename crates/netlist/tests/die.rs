//! A whole die, sequencer included, must run a real network layer bit for bit.

use netlist::demo::{example, host_steps, matvec_ref, reference, Example};
use netlist::{die_design, to_verilog, verilog_testbench, CellConfig, DieConfig, Simulator, TestStep};
use std::process::Command;

fn config() -> DieConfig {
    DieConfig {
        cells: 3,
        cell: CellConfig { weight_bits: 8, activation_bits: 16, accumulator_bits: 32, mac_stages: 2, adder_stages: 2, addr_bits: 4, coding: Default::default(), exponent_bits: 0, multiplier: netlist::multiplier::MultiplierKind::Array, fixed_weights: false, ecc: false, skip_zeros: false },
        scale_bits: 16,
        reg_bits: 3,
        index_bits: 4,
        microcode_bits: 9,
        broadcast_stages: 2,
        function_bits: 8,
        function_select_bits: 3, features: netlist::DieFeatures::without_register_windows(),
    }
}

fn run(c: DieConfig, e: &Example) -> (Simulator, Vec<TestStep>) {
    let design = die_design(c);
    let mut sim = Simulator::new(&design).unwrap();
    let mut steps = host_steps(c, e);
    for s in &steps {
        for (k, v) in &s.inputs {
            sim.set_wide(k, *v).unwrap();
        }
        sim.tick();
    }
    steps.push(TestStep { inputs: vec![("start".into(), 0)] });
    sim.set("start", 0).unwrap();
    let mut guard = 0;
    while sim.get("halted").unwrap() == 0 {
        sim.tick();
        steps.push(TestStep { inputs: vec![] });
        guard += 1;
        assert!(guard < 20_000, "the sequencer does not stop");
    }
    (sim, steps)
}

fn read_vec(sim: &mut Simulator, reg: u64, index: u64, c: DieConfig) -> i64 {
    sim.set("vec_addr", (reg << c.index_bits) | index).unwrap();
    sim.tick();
    let v = sim.get("vec_out").unwrap();
    ((v << 48) as i64) >> 48
}

#[test]
fn die_runs_a_full_mlp_layer_bit_exact() {
    let c = config();
    let e = example();
    let (mut sim, _) = run(c, &e);
    let got: Vec<i64> = (0..3).map(|i| read_vec(&mut sim, 6, i, c)).collect();
    assert_eq!(got, reference(&e));
}

#[test]
fn intermediate_registers_match_every_step() {
    let c = config();
    let e = example();
    let (mut sim, _) = run(c, &e);
    let gate = matvec_ref(&e.matrices[0], &e.factors[0], e.groups[0], &e.x, 2);
    let up = matvec_ref(&e.matrices[1], &e.factors[1], e.groups[1], &e.x, 2);
    assert_eq!((0..3).map(|i| read_vec(&mut sim, 1, i, c)).collect::<Vec<_>>(), gate);
    assert_eq!((0..3).map(|i| read_vec(&mut sim, 2, i, c)).collect::<Vec<_>>(), up);
}

#[test]
fn verilog_die_matches_the_gate_simulator() {
    if Command::new("iverilog").arg("-V").output().is_err() {
        return;
    }
    let c = config();
    let e = example();
    let design = die_design(c);
    let (mut sim, mut steps) = run(c, &e);
    let mut read_steps = Vec::new();
    for i in 0..3u64 {
        read_steps.push(TestStep { inputs: vec![("vec_addr".into(), ((6u64 << c.index_bits) | i) as u128)] });
    }
    let mut ours = Vec::new();
    for s in &read_steps {
        for (k, v) in &s.inputs {
            sim.set_wide(k, *v).unwrap();
        }
        sim.tick();
        ours.push(sim.get("vec_out").unwrap().to_string());
    }
    let skip = steps.len();
    steps.extend(read_steps);

    let dir = std::env::temp_dir().join(format!("llm2chip-die-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let source = dir.join("die.v");
    let binary = dir.join("die.vvp");
    std::fs::write(&source, format!("{}\n{}", to_verilog(&design), verilog_testbench(&design, &steps, "vec_out"))).unwrap();
    let compile = Command::new("iverilog").args(["-g2005", "-s", "testbench", "-o"]).arg(&binary).arg(&source).output().unwrap();
    assert!(compile.status.success(), "{}", String::from_utf8_lossy(&compile.stderr));
    let out = Command::new("vvp").arg(&binary).output().unwrap();
    let theirs: Vec<String> = String::from_utf8_lossy(&out.stdout).lines().filter(|l| l.parse::<u64>().is_ok()).map(str::to_string).collect();
    assert_eq!(theirs[skip..], ours[..]);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn an_fpga_gets_the_register_table_as_a_memory_that_does_what_the_gates_do() {
    use netlist::die::register_table_design;
    let design = register_table_design(4, 6, 2);
    let fpga = netlist::to_verilog_for_fpga(&design);
    let body = fpga.split("module register_table_4x6_r2 (").nth(1).unwrap().split("endmodule").next().unwrap();
    assert!(body.contains("assign entry1 = entries[key1];"), "{body}");
    if Command::new("iverilog").arg("-V").output().is_err() {
        return;
    }
    // Writes and reads mixed, also reading the entry being written, which still gives
    // its old value until the clock.
    let mut seed = 0x2545_F491_4F6C_DD1Du64;
    let mut next = |bits: u32| {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed & ((1 << bits) - 1)) as u128
    };
    let steps: Vec<TestStep> = (0..120)
        .map(|_| TestStep { inputs: vec![("we".into(), next(1)), ("waddr".into(), next(4)), ("wdata".into(), next(6)), ("key0".into(), next(4)), ("key1".into(), next(4))] })
        .collect();
    let mut sim = netlist::Simulator::new(&design).unwrap();
    let mut ours = Vec::new();
    for step in &steps {
        for (name, value) in &step.inputs {
            sim.set_wide(name, *value).unwrap();
        }
        sim.tick();
        ours.push(format!("{}", (sim.get("entry0").unwrap() << 6) | sim.get("entry1").unwrap()));
    }
    let dir = std::env::temp_dir().join(format!("llm2chip-table-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let (source, binary) = (dir.join("table.v"), dir.join("table.vvp"));
    std::fs::write(&source, format!("{fpga}\n{}", verilog_testbench(&design, &steps, "{entry0, entry1}"))).unwrap();
    let compile = Command::new("iverilog").args(["-g2005", "-s", "testbench", "-o"]).arg(&binary).arg(&source).output().unwrap();
    assert!(compile.status.success(), "{}", String::from_utf8_lossy(&compile.stderr));
    let out = Command::new("vvp").arg(&binary).output().unwrap();
    let theirs: Vec<String> = String::from_utf8_lossy(&out.stdout).lines().filter(|l| l.parse::<u64>().is_ok()).map(str::to_string).collect();
    assert_eq!(theirs.len(), steps.len());
    assert_eq!(theirs, ours);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn die_contains_every_block() {
    let design = die_design(config());
    let groups: Vec<&str> = design.modules["llm2chip_die"].instances.iter().map(|i| i.group.as_str()).collect();
    for name in ["Sequencer", "Vector bank", "Vector unit", "Function table", "Activation network", "Cells", "Result chain"] {
        assert!(groups.contains(&name), "{name} is missing");
    }
}

#[test]
fn two_dies_pass_a_value_over_the_link() {
    use netlist::block::Alu;
    use netlist::die::{OP_ADD, OP_LINK, OP_MOV, OP_MUL};
    use netlist::{package_design, DieProgram};

    let c = config();
    const LINK_STAGES: u32 = 2;
    let design = package_design(c, 2, LINK_STAGES);
    let mut sim = Simulator::new(&design).unwrap();

    // Die 0 multiplies and sends; die 1 receives and adds a local value.
    let mut sender = DieProgram::new(c);
    let mut product = Alu::new(OP_MUL);
    product.a = (0, 0);
    product.b = (0, 1);
    product.dst = (1, 0);
    sender.alu(&product);
    let mut send = Alu::new(OP_MOV);
    send.a = (1, 0);
    send.link_send = true;
    send.write = false;
    sender.alu(&send);
    let code0 = sender.finish();

    let mut receiver = DieProgram::new(c);
    let mut receive = Alu::new(OP_LINK);
    receive.dst = (1, 0);
    receiver.alu(&receive);
    let mut add = Alu::new(OP_ADD);
    add.a = (1, 0);
    add.b = (0, 0);
    add.dst = (2, 0);
    receiver.alu(&add);
    let code1 = receiver.finish();

    for die in 0..2 {
        sim.set(&format!("d{die}_start"), 1).unwrap();
    }
    sim.tick();
    for (die, (code, values)) in [(&code0, [12i64, 34]), (&code1, [500, 0])].iter().enumerate() {
        let name = |pin: &str| format!("d{die}_{pin}");
        sim.set(&name("load_code"), 1).unwrap();
        for (i, word) in code.iter().enumerate() {
            sim.set(&name("code_addr"), i as u64).unwrap();
            sim.set_wide(&name("code_data"), word.encode(&c)).unwrap();
            sim.tick();
        }
        sim.set(&name("load_code"), 0).unwrap();
        sim.set(&name("load_vec"), 1).unwrap();
        for (i, v) in values.iter().enumerate() {
            sim.set(&name("vec_addr"), i as u64).unwrap();
            sim.set(&name("vec_data"), *v as u64).unwrap();
            sim.tick();
        }
        sim.set(&name("load_vec"), 0).unwrap();
    }
    sim.tick();
    sim.set("d0_start", 0).unwrap();
    sim.set("d1_start", 0).unwrap();
    let mut guard = 0;
    while sim.get("d0_halted").unwrap() == 0 || sim.get("d1_halted").unwrap() == 0 {
        sim.tick();
        guard += 1;
        assert!(guard < 2000);
    }
    sim.set("d1_vec_addr", 2u64 << c.index_bits).unwrap();
    sim.tick();
    assert_eq!(sim.get("d1_vec_out").unwrap(), 12 * 34 + 500);
}

#[test]
fn a_microcode_word_reads_back_as_what_it_does() {
    use netlist::die::{decode_word, describe_word, OP_ADD};
    let c = config();
    let word = netlist::ControlWord { op: OP_ADD, reg_a: 2, reg_b: 3, reg_dst: 5, write_en: true, ..Default::default() };
    let bits: String = word.encode_words(&c).iter().rev().map(|w| format!("{w:032b}")).collect::<String>();
    let bits = &bits[bits.len() - netlist::word_bits(&c) as usize..];
    let fields = decode_word(bits, &c);
    assert_eq!((fields["op"], fields["reg_a"], fields["reg_b"], fields["reg_dst"]), (OP_ADD as u128, 2, 3, 5));
    assert_eq!(describe_word(&fields), "Adds r2 + r3 and stores it in r5.");
    let halt = netlist::ControlWord { halt: true, ..Default::default() };
    let bits: String = halt.encode_words(&c).iter().rev().map(|w| format!("{w:032b}")).collect();
    assert!(describe_word(&decode_word(&bits[bits.len() - netlist::word_bits(&c) as usize..], &c)).starts_with("End of the routine"));
}

#[test]
fn a_wide_link_carries_a_whole_block_in_one_word() {
    use netlist::block::Alu;
    use netlist::die::{OP_LINK, OP_MOV};
    use netlist::{package_design, DieProgram};

    // Four lanes and a link lane for each: one send moves four values. The die keeps only
    // the fields its two programs use, as a real one would.
    let general = DieConfig { cells: 4, features: netlist::DieFeatures::default().with_lane_bits(2), ..config() };
    let programs = |c: DieConfig| {
        let mut sender = DieProgram::new(c);
        let mut send = Alu::new(OP_MOV);
        (send.a, send.link_send, send.write, send.vector) = ((1, 0), true, false, true);
        sender.alu(&send);
        let mut receiver = DieProgram::new(c);
        let mut receive = Alu::new(OP_LINK);
        (receive.dst, receive.vector) = ((2, 0), true);
        receiver.alu(&receive);
        (sender.finish_program(), receiver.finish_program())
    };
    let (p0, p1) = programs(general);
    let features = netlist::DieFeatures::of(&p0).union(netlist::DieFeatures::of(&p1)).with_lane_bits(2);
    assert!(features.wide_link, "a block over the link takes one link lane per lane");
    let c = DieConfig { features, ..general };
    let (code0, code1) = (p0.words, p1.words);
    let design = package_design(c, 2, 1);
    let mut sim = Simulator::new(&design).unwrap();

    for die in 0..2 {
        sim.set(&format!("d{die}_start"), 1).unwrap();
    }
    sim.tick();
    let values = [111i64, -222, 333, -444];
    for (die, code) in [&code0, &code1].iter().enumerate() {
        let name = |pin: &str| format!("d{die}_{pin}");
        sim.set(&name("load_code"), 1).unwrap();
        for (i, word) in code.iter().enumerate() {
            sim.set(&name("code_addr"), i as u64).unwrap();
            sim.set_wide(&name("code_data"), word.encode(&c)).unwrap();
            sim.tick();
        }
        sim.set(&name("load_code"), 0).unwrap();
    }
    sim.set("d0_load_vec", 1).unwrap();
    for (i, v) in values.iter().enumerate() {
        sim.set("d0_vec_addr", 1u64 << c.index_bits | i as u64).unwrap();
        sim.set("d0_vec_data", *v as u64 & 0xFFFF).unwrap();
        sim.tick();
    }
    sim.set("d0_load_vec", 0).unwrap();
    sim.tick();
    sim.set("d0_start", 0).unwrap();
    sim.set("d1_start", 0).unwrap();
    let mut cycles = 0;
    while sim.get("d0_halted").unwrap() == 0 || sim.get("d1_halted").unwrap() == 0 {
        sim.tick();
        cycles += 1;
        assert!(cycles < 2000);
    }
    let got: Vec<i64> = (0..4)
        .map(|i| {
            sim.set("d1_vec_addr", 2u64 << c.index_bits | i).unwrap();
            sim.tick();
            sim.get("d1_vec_out").unwrap() as i16 as i64
        })
        .collect();
    assert_eq!(got, values);
}
