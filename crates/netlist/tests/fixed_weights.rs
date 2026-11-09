//! Weights fixed at manufacture: the same results as a loadable chip, with less circuit.

use netlist::demo::{example, host_steps, load, matvec_ref, reference};
use netlist::format::WeightCoding;
use netlist::multiplier::MultiplierKind;
use netlist::{cell_design, die_design, to_verilog, verilog_testbench_with_rom, CellConfig, DieConfig, RomWord, Simulator, TestStep};
use std::process::Command;

fn cell(fixed_weights: bool) -> CellConfig {
    CellConfig { weight_bits: 8, activation_bits: 16, accumulator_bits: 32, mac_stages: 2, adder_stages: 2, addr_bits: 3, coding: WeightCoding::default(), exponent_bits: 0, multiplier: MultiplierKind::Array, fixed_weights }
}

fn die(fixed_weights: bool) -> DieConfig {
    DieConfig { cells: 3, cell: CellConfig { addr_bits: 4, ..cell(fixed_weights) }, scale_bits: 16, reg_bits: 3, index_bits: 4, microcode_bits: 9, broadcast_stages: 2, function_bits: 8, function_select_bits: 3 }
}

const WEIGHTS: [u64; 4] = [3, 0xFE, 5, 0x81];
const ACTIVATIONS: [u64; 4] = [100, 0xFFCE, 25, 8];

fn run_cell(sim: &mut Simulator) -> u64 {
    sim.set("run", 1).unwrap();
    for x in ACTIVATIONS {
        sim.set("x", x).unwrap();
        sim.tick();
    }
    sim.set("run", 0).unwrap();
    for _ in 0..cell(false).latency() {
        sim.tick();
    }
    sim.get("total").unwrap()
}

fn clear(sim: &mut Simulator) {
    sim.set("clear", 1).unwrap();
    sim.tick();
    sim.set("clear", 0).unwrap();
}

#[test]
fn a_fixed_cell_computes_the_same_with_fewer_gates() {
    let loadable = cell_design(cell(false));
    let mut sim = Simulator::new(&loadable).unwrap();
    sim.set("load_en", 1).unwrap();
    for (i, w) in WEIGHTS.iter().enumerate() {
        sim.set("load_addr", i as u64).unwrap();
        sim.set("load_data", *w).unwrap();
        sim.tick();
    }
    sim.set("load_en", 0).unwrap();
    clear(&mut sim);
    let expected = run_cell(&mut sim);

    let fixed = cell_design(cell(true));
    let mut sim = Simulator::new(&fixed).unwrap();
    for (i, w) in WEIGHTS.iter().enumerate() {
        sim.poke_memory("/weight_rom", i, *w as u128).unwrap();
    }
    clear(&mut sim);
    assert_eq!(run_cell(&mut sim), expected);

    let (a, b) = (loadable.flat_counts(), fixed.flat_counts());
    assert!(b["mux"] < a["mux"], "{a:?} {b:?}");
    assert_eq!(b["rom"], 1);
    assert!(!b.contains_key("sram"));
    let top = &fixed.modules[&fixed.top];
    assert!(top.ports.iter().all(|p| !p.name.starts_with("load_")));
}

fn run_die(c: DieConfig) -> Simulator {
    let e = example();
    let mut sim = Simulator::new(&die_design(c)).unwrap();
    load(&mut sim, c, &e);
    sim.set("start", 0).unwrap();
    let mut guard = 0;
    while sim.get("halted").unwrap() == 0 {
        sim.tick();
        guard += 1;
        assert!(guard < 20_000, "el secuenciador no se detiene");
    }
    sim
}

fn read(sim: &mut Simulator, c: DieConfig, register: u64, index: u64) -> i64 {
    sim.set("vec_addr", (register << c.index_bits) | index).unwrap();
    sim.tick();
    ((sim.get("vec_out").unwrap() << 48) as i64) >> 48
}

#[test]
fn a_die_with_fixed_weights_runs_the_layer_bit_exact() {
    let c = die(true);
    let e = example();
    let mut sim = run_die(c);
    assert_eq!((0..3).map(|i| read(&mut sim, c, 6, i)).collect::<Vec<_>>(), reference(&e));
    let gate = matvec_ref(&e.matrices[0], &e.factors[0], e.groups[0], &e.x, 2);
    assert_eq!((0..3).map(|i| read(&mut sim, c, 1, i)).collect::<Vec<_>>(), gate);

    let (loadable, fixed) = (die_design(die(false)), die_design(c));
    let total = |d: &netlist::Design| d.flat_counts().iter().filter(|(k, _)| !["sram", "rom"].contains(&k.as_str())).map(|(_, v)| *v).sum::<u64>();
    assert!(total(&fixed) < total(&loadable));
    let top = &fixed.modules[&fixed.top];
    assert!(top.ports.iter().all(|p| !["load_cell", "load_scale", "cell_addr"].contains(&p.name.as_str())));
    assert!(host_steps(c, &e).len() < host_steps(die(false), &e).len(), "no hay pasos de carga de celdas");
}

#[test]
fn the_fixed_cell_in_verilog_matches_the_gate_simulator() {
    if Command::new("iverilog").arg("-V").output().is_err() {
        return;
    }
    let design = cell_design(cell(true));
    let mut sim = Simulator::new(&design).unwrap();
    let rom: Vec<RomWord> = WEIGHTS.iter().enumerate().map(|(address, w)| RomWord { path: "weight_rom".into(), address, value: *w as u128 }).collect();
    for word in &rom {
        sim.poke_memory("/weight_rom", word.address, word.value).unwrap();
    }
    let mut steps = vec![TestStep { inputs: vec![("clear".into(), 1)] }];
    for x in ACTIVATIONS {
        steps.push(TestStep { inputs: vec![("clear".into(), 0), ("run".into(), 1), ("x".into(), x as u128)] });
    }
    for _ in 0..cell(true).latency() {
        steps.push(TestStep { inputs: vec![("run".into(), 0)] });
    }
    let mut ours = Vec::new();
    for step in &steps {
        for (name, value) in &step.inputs {
            sim.set_wide(name, *value).unwrap();
        }
        sim.tick();
        ours.push(sim.get("total").unwrap().to_string());
    }
    let dir = std::env::temp_dir().join(format!("llm2chip-rom-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let (source, binary) = (dir.join("cell.v"), dir.join("cell.vvp"));
    std::fs::write(&source, format!("{}\n{}", to_verilog(&design), verilog_testbench_with_rom(&design, &rom, &steps, "total"))).unwrap();
    let compile = Command::new("iverilog").args(["-g2005", "-s", "testbench", "-o"]).arg(&binary).arg(&source).output().unwrap();
    assert!(compile.status.success(), "{}", String::from_utf8_lossy(&compile.stderr));
    let out = Command::new("vvp").arg(&binary).output().unwrap();
    let theirs: Vec<String> = String::from_utf8_lossy(&out.stdout).lines().filter(|l| l.parse::<u64>().is_ok()).map(str::to_string).collect();
    assert_eq!(theirs, ours);
    std::fs::remove_dir_all(&dir).ok();
}
