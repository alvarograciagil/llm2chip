//! The generated gates must compute exactly what the arithmetic says they should.

use netlist::format::WeightCoding;
use netlist::{cell_design, to_verilog, to_vhdl, verilog_testbench, vhdl_testbench, CellConfig, Simulator, TestStep};
use std::process::Command;

const CONFIGS: [CellConfig; 6] = [
    CellConfig { weight_bits: 4, activation_bits: 8, accumulator_bits: 24, mac_stages: 1, adder_stages: 1, addr_bits: 4, coding: WeightCoding::Affine, exponent_bits: 0, multiplier: netlist::multiplier::MultiplierKind::Array, fixed_weights: false, ecc: false },
    CellConfig { weight_bits: 4, activation_bits: 16, accumulator_bits: 32, mac_stages: 2, adder_stages: 2, addr_bits: 5, coding: WeightCoding::Affine, exponent_bits: 0, multiplier: netlist::multiplier::MultiplierKind::Array, fixed_weights: false, ecc: false },
    CellConfig { weight_bits: 8, activation_bits: 16, accumulator_bits: 32, mac_stages: 2, adder_stages: 2, addr_bits: 6, coding: WeightCoding::Affine, exponent_bits: 0, multiplier: netlist::multiplier::MultiplierKind::Array, fixed_weights: false, ecc: false },
    CellConfig { weight_bits: 8, activation_bits: 16, accumulator_bits: 32, mac_stages: 4, adder_stages: 3, addr_bits: 6, coding: WeightCoding::Affine, exponent_bits: 0, multiplier: netlist::multiplier::MultiplierKind::Array, fixed_weights: false, ecc: false },
    CellConfig { weight_bits: 3, activation_bits: 12, accumulator_bits: 20, mac_stages: 3, adder_stages: 1, addr_bits: 3, coding: WeightCoding::Affine, exponent_bits: 0, multiplier: netlist::multiplier::MultiplierKind::Array, fixed_weights: false, ecc: false },
    CellConfig { weight_bits: 6, activation_bits: 16, accumulator_bits: 12, mac_stages: 1, adder_stages: 2, addr_bits: 4, coding: WeightCoding::Affine, exponent_bits: 0, multiplier: netlist::multiplier::MultiplierKind::Array, fixed_weights: false, ecc: false },
];

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn signed(&mut self, bits: u32) -> i64 {
        let raw = self.next() >> (64 - bits);
        (raw as i64) << (64 - bits) >> (64 - bits)
    }
}

fn mask(value: i64, bits: u32) -> u64 {
    if bits >= 64 {
        value as u64
    } else {
        (value as u64) & ((1u64 << bits) - 1)
    }
}

fn sign_extend(value: u64, bits: u32) -> i64 {
    ((value << (64 - bits)) as i64) >> (64 - bits)
}

fn clear(sim: &mut Simulator) {
    sim.set("clear", 1).unwrap();
    sim.set("run", 0).unwrap();
    sim.set("load_en", 0).unwrap();
    sim.tick();
    sim.set("clear", 0).unwrap();
}

fn load(sim: &mut Simulator, c: CellConfig, weights: &[i64]) {
    sim.set("load_en", 1).unwrap();
    for (i, w) in weights.iter().enumerate() {
        sim.set("load_addr", i as u64).unwrap();
        sim.set("load_data", mask(*w, c.weight_bits)).unwrap();
        sim.tick();
    }
    sim.set("load_en", 0).unwrap();
}

/// Streams activations through the cell and returns the accumulator's value.
fn run(sim: &mut Simulator, c: CellConfig, activations: &[i64]) -> i64 {
    sim.set("run", 1).unwrap();
    for x in activations {
        sim.set("x", mask(*x, c.activation_bits)).unwrap();
        sim.tick();
    }
    sim.set("run", 0).unwrap();
    for _ in 0..c.latency() {
        sim.tick();
    }
    sign_extend(sim.get("total").unwrap(), c.accumulator_bits)
}

fn expected(c: CellConfig, weights: &[i64], activations: &[i64]) -> i64 {
    let sum: i64 = weights.iter().zip(activations).map(|(w, x)| w * x).sum();
    sign_extend(mask(sum, c.accumulator_bits), c.accumulator_bits)
}

#[test]
fn every_4_bit_by_8_bit_product_is_exact() {
    let c = CONFIGS[0];
    let mut sim = Simulator::new(&cell_design(c)).unwrap();
    for w in -8i64..8 {
        for x in -128i64..128 {
            clear(&mut sim);
            load(&mut sim, c, &[w]);
            clear(&mut sim);
            assert_eq!(run(&mut sim, c, &[x]), w * x, "{w} × {x}");
        }
    }
}

#[test]
fn random_dot_products_match_reference_in_every_configuration() {
    let mut rng = Rng(0x9E3779B97F4A7C15);
    for c in CONFIGS {
        let depth = 1usize << c.addr_bits;
        let mut sim = Simulator::new(&cell_design(c)).unwrap();
        for _ in 0..80 {
            let n = 1 + (rng.next() as usize % depth);
            let weights: Vec<i64> = (0..n).map(|_| rng.signed(c.weight_bits)).collect();
            let activations: Vec<i64> = (0..n).map(|_| rng.signed(c.activation_bits)).collect();
            clear(&mut sim);
            load(&mut sim, c, &weights);
            clear(&mut sim);
            assert_eq!(run(&mut sim, c, &activations), expected(c, &weights, &activations), "{c:?}");
        }
    }
}

#[test]
fn extreme_values_do_not_overflow_the_product() {
    let c = CONFIGS[2];
    let mut sim = Simulator::new(&cell_design(c)).unwrap();
    for (w, x) in [(-128, -32768), (-128, 32767), (127, -32768), (127, 32767), (-1, -1), (0, -32768)] {
        clear(&mut sim);
        load(&mut sim, c, &[w]);
        clear(&mut sim);
        assert_eq!(run(&mut sim, c, &[x]), w * x, "{w} × {x}");
    }
}

#[test]
fn stored_weights_survive_repeated_runs() {
    let c = CONFIGS[1];
    let mut sim = Simulator::new(&cell_design(c)).unwrap();
    let weights = [3, -2, 7, -8];
    clear(&mut sim);
    load(&mut sim, c, &weights);
    for x in [[1, 1, 1, 1], [100, -100, 50, 25], [-3, 9, 2, -1]] {
        clear(&mut sim);
        assert_eq!(run(&mut sim, c, &x), expected(c, &weights, &x));
    }
}

#[test]
fn pipeline_stages_change_latency_but_not_the_answer() {
    let weights = [5, -3, 2];
    let activations = [1000, 250, -77];
    let mut results = Vec::new();
    for stages in 1..=4 {
        let c = CellConfig { weight_bits: 8, activation_bits: 16, accumulator_bits: 32, mac_stages: stages, adder_stages: stages, addr_bits: 3, coding: WeightCoding::Affine, exponent_bits: 0, multiplier: netlist::multiplier::MultiplierKind::Array, fixed_weights: false, ecc: false };
        let mut sim = Simulator::new(&cell_design(c)).unwrap();
        clear(&mut sim);
        load(&mut sim, c, &weights);
        clear(&mut sim);
        results.push(run(&mut sim, c, &activations));
    }
    assert!(results.iter().all(|r| *r == results[0] && *r == 5000 - 750 - 154));
}

#[test]
fn hierarchy_expands_into_real_gates() {
    let design = cell_design(CONFIGS[2]);
    let counts = design.flat_counts();
    assert!(counts["and"] > 100 && counts["xor"] > 100 && counts["dff"] > 40);
    assert_eq!(counts["sram"], 1);
    assert!(design.modules.contains_key("full_adder"));
    let sim = Simulator::new(&design).unwrap();
    assert!(sim.snapshot("/multiplier/adder0/fa0").is_some(), "se puede entrar hasta el sumador completo");
}

#[test]
fn verilog_matches_the_gate_simulator_cycle_by_cycle() {
    if Command::new("iverilog").arg("-V").output().is_err() {
        eprintln!("iverilog no está instalado: se omite");
        return;
    }
    let mut rng = Rng(0xDEADBEEFCAFE);
    let dir = std::env::temp_dir().join(format!("llm2chip-verilog-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();

    for (index, c) in CONFIGS.iter().copied().enumerate() {
        let design = cell_design(c);
        let mut steps = vec![TestStep { inputs: vec![("clear".into(), 1)] }];
        let n = 1usize << c.addr_bits;
        for i in 0..n {
            let inputs = vec![
                ("clear".to_string(), 0),
                ("load_en".to_string(), 1),
                ("load_addr".to_string(), i as u128),
                ("load_data".to_string(), mask(rng.signed(c.weight_bits), c.weight_bits) as u128),
            ];
            steps.push(TestStep { inputs });
        }
        steps.push(TestStep { inputs: vec![("load_en".into(), 0), ("clear".into(), 1)] });
        for _ in 0..n {
            steps.push(TestStep {
                inputs: vec![
                    ("clear".into(), 0),
                    ("run".into(), 1),
                    ("x".into(), mask(rng.signed(c.activation_bits), c.activation_bits) as u128),
                ],
            });
        }
        for _ in 0..c.latency() {
            steps.push(TestStep { inputs: vec![("run".into(), 0)] });
        }

        let mut sim = Simulator::new(&design).unwrap();
        let mut ours = Vec::new();
        for step in &steps {
            for (name, value) in &step.inputs {
                sim.set_wide(name, *value).unwrap();
            }
            sim.tick();
            ours.push(sim.get("total").unwrap().to_string());
        }

        let source = dir.join(format!("cell{index}.v"));
        let binary = dir.join(format!("cell{index}.vvp"));
        std::fs::write(&source, format!("{}\n{}", to_verilog(&design), verilog_testbench(&design, &steps, "total"))).unwrap();
        let compile = Command::new("iverilog").args(["-g2005", "-s", "testbench", "-o"]).arg(&binary).arg(&source).output().unwrap();
        assert!(compile.status.success(), "{}", String::from_utf8_lossy(&compile.stderr));
        let run = Command::new("vvp").arg(&binary).output().unwrap();
        let theirs: Vec<String> = String::from_utf8_lossy(&run.stdout).lines().filter(|l| l.parse::<u64>().is_ok()).map(str::to_string).collect();
        assert_eq!(ours, theirs, "configuración {index}");
    }
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn vhdl_export_computes_the_same_result() {
    if Command::new("nvc").arg("--version").output().is_err() {
        eprintln!("nvc no está instalado: se omite");
        return;
    }
    let c = CONFIGS[1];
    let design = cell_design(c);
    let weights = [3i64, -2, 7, -8, 1, 0, 5, -6];
    let activations = [100i64, -250, 77, 13, 900, 4, -6, 2];
    let mut steps = vec![TestStep { inputs: vec![("clear".into(), 1)] }];
    for (i, w) in weights.iter().enumerate() {
        steps.push(TestStep { inputs: vec![("clear".into(), 0), ("load_en".into(), 1), ("load_addr".into(), i as u128), ("load_data".into(), mask(*w, c.weight_bits) as u128)] });
    }
    steps.push(TestStep { inputs: vec![("load_en".into(), 0), ("clear".into(), 1)] });
    for x in activations {
        steps.push(TestStep { inputs: vec![("clear".into(), 0), ("run".into(), 1), ("x".into(), mask(x, c.activation_bits) as u128)] });
    }
    for _ in 0..c.latency() {
        steps.push(TestStep { inputs: vec![("run".into(), 0)] });
    }
    let expected = mask(expected(c, &weights, &activations), c.accumulator_bits);

    let dir = std::env::temp_dir().join(format!("llm2chip-vhdl-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let source = dir.join("cell.vhd");
    std::fs::write(&source, format!("{}\n{}", to_vhdl(&design), vhdl_testbench(&design, &steps, "total", expected))).unwrap();
    let run = |args: &[&str]| {
        let output = Command::new("nvc").current_dir(&dir).args(["--std=2008"]).args(args).output().unwrap();
        assert!(output.status.success(), "{}\n{}", String::from_utf8_lossy(&output.stdout), String::from_utf8_lossy(&output.stderr));
    };
    run(&["-a", "cell.vhd"]);
    run(&["-e", "testbench"]);
    run(&["-r", "testbench"]);
    std::fs::remove_dir_all(&dir).ok();
}
