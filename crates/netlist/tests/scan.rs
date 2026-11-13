//! The scan chain and the patterns for the factory test, checked against the gate simulator.

use netlist::atpg::{self, Outcome, Site, TestPattern};
use netlist::format::WeightCoding;
use netlist::scan::insert_scan;
use netlist::{cell_design, CellConfig, Design, Dir, Simulator};

fn small_cell() -> Design {
    cell_design(CellConfig { weight_bits: 4, activation_bits: 8, accumulator_bits: 24, mac_stages: 1, adder_stages: 1, addr_bits: 4, coding: WeightCoding::Affine, exponent_bits: 0, multiplier: netlist::multiplier::MultiplierKind::Array, fixed_weights: false, ecc: false, skip_zeros: false })
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

fn ports(design: &Design, dir: Dir) -> Vec<(String, u32)> {
    let top = &design.modules[&design.top];
    top.ports.iter().filter(|p| p.dir == dir).map(|p| (p.name.clone(), top.nets[p.net].width)).collect()
}

fn words(bits: &[bool]) -> Vec<u32> {
    bits.chunks(32).map(|chunk| chunk.iter().enumerate().fold(0, |w, (i, b)| w | (*b as u32) << i)).collect()
}

#[test]
fn with_the_chain_idle_the_design_works_as_before() {
    let plain = small_cell();
    let scanned = insert_scan(plain.clone());
    assert!(scanned.length > 0);
    let (mut a, mut b) = (Simulator::new(&plain).unwrap(), Simulator::new(&scanned.design).unwrap());
    let mut rng = Rng(7);
    for cycle in 0..400 {
        for (name, width) in ports(&plain, Dir::Input) {
            let value: Vec<u32> = (0..width.div_ceil(32)).map(|_| rng.next() as u32).collect();
            a.set_words(&name, &value).unwrap();
            b.set_words(&name, &value).unwrap();
        }
        a.tick();
        b.tick();
        for (name, _) in ports(&plain, Dir::Output) {
            assert_eq!(a.get_words(&name), b.get_words(&name), "{name} en el ciclo {cycle}");
        }
    }
}

#[test]
fn the_chain_shifts_a_pattern_from_scan_in_to_scan_out() {
    let scanned = insert_scan(small_cell());
    let length = scanned.length;
    let mut sim = Simulator::new(&scanned.design).unwrap();
    sim.set("test_mode", 1).unwrap();
    sim.set("scan_enable", 1).unwrap();
    // 0011 repeated: each flip-flop passes on both values, and both changes.
    let bit = |i: usize| (i / 2) % 2 == 1;
    let mut out = Vec::new();
    for i in 0..length + 40 {
        sim.set("scan_in", bit(i) as u64).unwrap();
        out.push(sim.get("scan_out").unwrap() == 1);
        sim.tick();
    }
    for i in 0..40 {
        assert_eq!(out[length + i], bit(i), "el bit {i} sale {length} ciclos después");
    }
}

/// Shifts `0011…` through the chain; whether everything came out as it went in.
fn flush(sim: &mut Simulator, length: usize) -> bool {
    sim.set("test_mode", 1).unwrap();
    sim.set("scan_enable", 1).unwrap();
    let bit = |i: usize| (i / 2) % 2 == 1;
    let mut good = true;
    for i in 0..length + 8 {
        sim.set("scan_in", bit(i) as u64).unwrap();
        if i >= length {
            good &= (sim.get("scan_out").unwrap() == 1) == bit(i - length);
        }
        sim.tick();
    }
    good
}

#[test]
fn the_chain_test_finds_the_faults_of_the_shift_path() {
    let scanned = insert_scan(small_cell());
    let faults = atpg::chain_faults(&scanned.design).unwrap();
    assert!(flush(&mut Simulator::new(&scanned.design).unwrap(), scanned.length));
    // Those at a signal's source can be injected; those at a multiplexer's input are the
    // chain's by construction: they stop that flip-flop shifting.
    let stems: Vec<_> = faults.iter().filter(|f| matches!(f.site, Site::Stem(_))).collect();
    assert!(stems.len() >= 4, "{} fallos de origen en la cadena", stems.len());
    for fault in stems {
        let Site::Stem(signal) = fault.site else { unreachable!() };
        let mut sim = Simulator::new(&scanned.design).unwrap();
        sim.stick_signal(signal, fault.stuck);
        assert!(!flush(&mut sim, scanned.length), "{fault:?} {:?}", atpg::describe(&scanned.design, &[*fault]).unwrap());
    }
}

#[test]
fn the_event_driven_fault_simulation_agrees_with_simulating_each_faulty_circuit() {
    let scanned = insert_scan(small_cell());
    let report = atpg::generate(&scanned.design, 0).unwrap();
    let mut rng = Rng(11);
    let controlled = report.patterns[0].inputs.iter().map(|(_, b)| b.len()).sum::<usize>() + report.chain_length;
    for _ in 0..3 {
        let pattern: Vec<bool> = (0..controlled).map(|_| rng.next() & 1 == 1).collect();
        let full = atpg::faults_found_by_full_simulation(&scanned.design, &pattern).unwrap();
        let fast = atpg::faults_found_by_fault_simulation(&scanned.design, &pattern).unwrap();
        assert_eq!(full.len(), fast.len());
        let found = full.iter().filter(|(_, d)| *d).count();
        assert!(found > 100, "un patrón al azar encuentra {found} de {}", full.len());
        for ((fault, a), (_, b)) in full.iter().zip(&fast) {
            assert_eq!(a, b, "{fault:?}");
        }
    }
}

/// Runs a pattern as a tester would: shifts its state in, sets the pins, reads the outputs,
/// captures with one clock, and shifts the result out. Whether everything read matched.
fn replay(sim: &mut Simulator, p: &TestPattern) -> bool {
    let length = p.load.len();
    sim.set("test_mode", 1).unwrap();
    sim.set("scan_enable", 1).unwrap();
    for i in (0..length).rev() {
        sim.set("scan_in", p.load[i] as u64).unwrap();
        sim.tick();
    }
    for (name, bits) in &p.inputs {
        sim.set_words(name, &words(bits)).unwrap();
    }
    sim.set("scan_enable", 0).unwrap();
    sim.settle();
    let mut matched = true;
    for (name, expected) in &p.outputs {
        let got = sim.get_words(name).unwrap();
        for (i, e) in expected.iter().enumerate() {
            if let Some(e) = e {
                matched &= (got[i / 32] >> (i % 32) & 1 == 1) == *e;
            }
        }
    }
    sim.tick();
    sim.set("scan_enable", 1).unwrap();
    for i in (0..length).rev() {
        if let Some(e) = p.unload[i] {
            matched &= (sim.get("scan_out").unwrap() == 1) == e;
        }
        sim.tick();
    }
    matched
}

#[test]
fn the_patterns_pass_on_the_gate_simulator_and_fail_with_a_fault_they_find() {
    let scanned = insert_scan(small_cell());
    let report = atpg::generate(&scanned.design, 200).unwrap();
    eprintln!(
        "celda: {} biestables en la cadena, {} fallos, {} detectados, {} por la cadena, {} imposibles, {} abandonados, {} + {} patrones, cobertura {:.2} % ({:.2} % de los posibles)",
        report.chain_length, report.faults, report.detected, report.chain, report.untestable, report.aborted, report.random_patterns, report.deterministic_patterns,
        report.fault_coverage() * 100.0, report.test_coverage() * 100.0
    );
    assert!(report.test_coverage() > 0.99, "cobertura {}", report.test_coverage());
    assert!(report.aborted * 100 < report.faults, "{} abandonados", report.aborted);
    // What PODEM calls impossible, thousands of random patterns do not find either.
    let impossible: Vec<_> = report.outcomes.iter().filter(|(_, o)| *o == Outcome::Untestable).map(|(f, _)| *f).collect();
    assert_eq!(atpg::found_at_random(&scanned.design, &impossible, 200).unwrap(), 0);

    let mut sim = Simulator::new(&scanned.design).unwrap();
    for (i, p) in report.patterns.iter().enumerate() {
        assert!(replay(&mut sim, p), "el patrón {i} no da lo previsto en el circuito sin fallos");
    }
    // Faults at a signal's source, a sample of those found and of those no pattern can find.
    let stems: Vec<_> = report.outcomes.iter().filter(|(f, _)| matches!(f.site, Site::Stem(_))).collect();
    let mut checked = [0, 0];
    for (fault, outcome) in stems.iter().step_by(stems.len() / 60 + 1).chain(stems.iter().filter(|(_, o)| *o == Outcome::Untestable).take(10)) {
        let Site::Stem(signal) = fault.site else { unreachable!() };
        let mut faulty = Simulator::new(&scanned.design).unwrap();
        faulty.stick_signal(signal, fault.stuck);
        let caught = !report.patterns.iter().all(|p| replay(&mut faulty, p));
        match outcome {
            Outcome::Detected => {
                assert!(caught, "{fault:?} debía detectarse");
                checked[0] += 1;
            }
            Outcome::Untestable => {
                assert!(!caught, "{fault:?} era imposible de detectar");
                checked[1] += 1;
            }
            Outcome::Aborted => {}
        }
    }
    assert!(checked[0] >= 30, "{checked:?}");
}

#[test]
fn the_scanned_verilog_runs_the_patterns_as_the_gate_simulator() {
    use netlist::{to_verilog, verilog_testbench, TestStep};
    use std::process::Command;
    if Command::new("iverilog").arg("-V").output().is_err() {
        eprintln!("iverilog no está instalado: se omite");
        return;
    }
    let scanned = insert_scan(small_cell());
    let report = atpg::generate(&scanned.design, 200).unwrap();
    // Three patterns, each shifted in while the one before shifts out, as a tester does.
    let mut steps = Vec::new();
    for p in report.patterns.iter().take(3) {
        for i in (0..p.load.len()).rev() {
            steps.push(TestStep { inputs: vec![("test_mode".into(), 1), ("scan_enable".into(), 1), ("scan_in".into(), p.load[i] as u128)] });
        }
        let mut capture: Vec<(String, u128)> = p.inputs.iter().map(|(name, bits)| (name.clone(), bits.iter().enumerate().fold(0u128, |v, (i, b)| v | (*b as u128) << i))).collect();
        capture.push(("scan_enable".into(), 0));
        steps.push(TestStep { inputs: capture });
    }
    steps.extend((0..report.chain_length).map(|_| TestStep { inputs: vec![("scan_enable".into(), 1)] }));

    let mut sim = Simulator::new(&scanned.design).unwrap();
    let mut ours = Vec::new();
    for step in &steps {
        for (name, value) in &step.inputs {
            sim.set_wide(name, *value).unwrap();
        }
        sim.tick();
        ours.push(sim.get("scan_out").unwrap().to_string());
    }
    let dir = std::env::temp_dir().join(format!("llm2chip-scan-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let (source, binary) = (dir.join("scan.v"), dir.join("scan.vvp"));
    std::fs::write(&source, format!("{}\n{}", to_verilog(&scanned.design), verilog_testbench(&scanned.design, &steps, "scan_out"))).unwrap();
    let compile = Command::new("iverilog").args(["-g2005", "-s", "testbench", "-o"]).arg(&binary).arg(&source).output().unwrap();
    assert!(compile.status.success(), "{}", String::from_utf8_lossy(&compile.stderr));
    let run = Command::new("vvp").arg(&binary).output().unwrap();
    let theirs: Vec<String> = String::from_utf8_lossy(&run.stdout).lines().filter(|l| ["0", "1", "x"].contains(l)).map(str::to_string).collect();
    std::fs::remove_dir_all(&dir).ok();
    assert_eq!(ours.len(), theirs.len());
    // Until the first load is in, what comes out is the unknown state of the power-up.
    let length = report.chain_length;
    assert_eq!(ours[length..], theirs[length..]);
}
