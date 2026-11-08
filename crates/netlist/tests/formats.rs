//! Every weight format's gates must agree with its reference arithmetic.

use netlist::format::{log_table, WeightCoding};
use netlist::{cell_design, to_verilog, verilog_testbench, CellConfig, Simulator, TestStep};
use std::process::Command;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

fn config(coding: WeightCoding, weight_bits: u32, exponent_bits: u32, mac_stages: u32) -> CellConfig {
    let mut c = CellConfig { weight_bits, activation_bits: 16, accumulator_bits: 32, mac_stages, adder_stages: 1, addr_bits: 4, coding, exponent_bits };
    c.accumulator_bits = c.required_accumulator().max(32);
    c
}

fn wrap(value: i128, bits: u32) -> u128 {
    if bits >= 128 {
        value as u128
    } else {
        (value as u128) & ((1u128 << bits) - 1)
    }
}

fn shift_of(c: &CellConfig, exponent: u64, base: u64) -> Option<u32> {
    if !c.shifts() {
        return Some(0);
    }
    if exponent < base {
        return None;
    }
    let bits = 32 - (c.accumulator_bits - 1).leading_zeros();
    let top = (1u64 << bits) - 1;
    Some((exponent - base).min(top) as u32)
}

fn check(c: CellConfig, codes: &[u64], base: u64, rng: &mut Rng) {
    let mut sim = Simulator::new(&cell_design(c)).unwrap();
    sim.set("clear", 1).unwrap();
    sim.tick();
    sim.set("clear", 0).unwrap();
    if c.shifts() {
        sim.set("exp_base", base).unwrap();
    }
    sim.set("load_en", 1).unwrap();
    for (i, code) in codes.iter().enumerate() {
        sim.set("load_addr", i as u64).unwrap();
        sim.set("load_data", *code).unwrap();
        sim.tick();
    }
    sim.set("load_en", 0).unwrap();
    sim.set("clear", 1).unwrap();
    sim.tick();
    sim.set("clear", 0).unwrap();

    let mut expected = 0i128;
    sim.set("run", 1).unwrap();
    for code in codes {
        let x = (rng.next() >> 48) as i16 as i64;
        sim.set("x", x as u64 & 0xFFFF).unwrap();
        sim.tick();
        let (significand, exponent) = c.decode_code(*code);
        if let Some(shift) = shift_of(&c, exponent, base) {
            let product = wrap(significand as i128 * x as i128, c.accumulator_bits);
            let moved = wrap((product << shift) as i128, c.accumulator_bits);
            expected = wrap(expected + moved as i128, c.accumulator_bits) as i128;
        }
    }
    sim.set("run", 0).unwrap();
    for _ in 0..c.latency() {
        sim.tick();
    }
    assert_eq!(sim.get_wide("total").unwrap(), expected as u128, "{c:?} códigos {codes:?}");
}

fn codes(rng: &mut Rng, bits: u32) -> Vec<u64> {
    (0..16).map(|_| rng.next() >> (64 - bits)).collect()
}

#[test]
fn every_q_width_from_2_to_32_multiplies_correctly() {
    let mut rng = Rng(0x1357_9BDF_0246_8ACE);
    for bits in 2..=32 {
        let c = config(WeightCoding::Fixed, bits, 0, 1 + bits % 3);
        for _ in 0..4 {
            let list = codes(&mut rng, bits);
            check(c, &list, 0, &mut rng);
        }
    }
}

#[test]
fn floating_formats_match_reference_arithmetic() {
    let mut rng = Rng(0xABCD_EF01_2345_6789);
    for (bits, exponent, stages) in [(4, 2, 1), (6, 2, 1), (6, 3, 2), (8, 4, 1), (8, 5, 2), (16, 5, 2), (16, 8, 1), (24, 7, 3), (32, 8, 2)] {
        let c = config(WeightCoding::Float, bits, exponent, stages);
        for _ in 0..12 {
            let list = codes(&mut rng, bits);
            let base = (1u64 << exponent) / 2;
            check(c, &list, base, &mut rng);
        }
    }
}

#[test]
fn subnormal_and_zero_floats_are_decoded() {
    let c = config(WeightCoding::Float, 16, 5, 1);
    assert_eq!(c.decode_code(0), (0, 1));
    assert_eq!(c.decode_code(0x0001), (1, 1));
    assert_eq!(c.decode_code(0x0400), (1024, 1));
    assert_eq!(c.decode_code(0x8400), (-1024, 1));
    assert_eq!(c.decode_code(0x3C00), (1024, 15));
}

#[test]
fn posits_match_reference_arithmetic() {
    let mut rng = Rng(0x0F1E_2D3C_4B5A_6978);
    for (bits, es) in [(6, 0), (8, 0), (8, 1), (8, 2), (12, 1), (16, 1), (16, 2), (24, 2), (32, 2)] {
        let c = config(WeightCoding::Posit, bits, es, 1 + bits % 2);
        for _ in 0..12 {
            let list = codes(&mut rng, bits);
            check(c, &list, (bits as u64 - 1) << es, &mut rng);
        }
    }
}

#[test]
fn known_posits_decode_to_their_published_values() {
    let c = config(WeightCoding::Posit, 8, 0, 1);
    // 0b01000000 is 1.0, whose biased regime is 7 for an 8-bit posit.
    assert_eq!(c.decode_code(0b0100_0000), (32, 7));
    assert_eq!(c.decode_code(0b0110_0000), (32, 8));
    assert_eq!(c.decode_code(0b1100_0000), (-32, 7));
    assert_eq!(c.decode_code(0).0, 0);
    assert_eq!(c.decode_code(0x80).0, 0, "NaR se trata como cero");
}

#[test]
fn logarithmic_formats_match_reference_arithmetic() {
    let mut rng = Rng(0x7777_1234_ABCD_0001);
    for (bits, fraction) in [(4, 1), (8, 3), (8, 4), (12, 6), (16, 8)] {
        let c = config(WeightCoding::Log, bits, fraction, 2);
        for _ in 0..12 {
            let list = codes(&mut rng, bits);
            check(c, &list, 2, &mut rng);
        }
    }
    assert_eq!(log_table(1), vec![32, 45]);
}

#[test]
fn codebook_reads_the_loaded_table() {
    let c = config(WeightCoding::Codebook, 4, 0, 2);
    let table: Vec<i64> = (0..16).map(|i| (i * i * 37 - 2500) as i64).collect();
    let mut sim = Simulator::new(&cell_design(c)).unwrap();
    sim.set("clear", 1).unwrap();
    sim.tick();
    sim.set("clear", 0).unwrap();
    sim.set("table_en", 1).unwrap();
    for (i, v) in table.iter().enumerate() {
        sim.set("table_addr", i as u64).unwrap();
        sim.set("table_data", *v as u64 & 0xFFFF).unwrap();
        sim.tick();
    }
    sim.set("table_en", 0).unwrap();
    let codes = [3u64, 15, 0, 7, 9];
    let xs = [100i64, -3, 250, 77, -1200];
    sim.set("load_en", 1).unwrap();
    for (i, code) in codes.iter().enumerate() {
        sim.set("load_addr", i as u64).unwrap();
        sim.set("load_data", *code).unwrap();
        sim.tick();
    }
    sim.set("load_en", 0).unwrap();
    sim.set("clear", 1).unwrap();
    sim.tick();
    sim.set("clear", 0).unwrap();
    sim.set("run", 1).unwrap();
    for x in xs {
        sim.set("x", x as u64 & 0xFFFF).unwrap();
        sim.tick();
    }
    sim.set("run", 0).unwrap();
    for _ in 0..c.latency() {
        sim.tick();
    }
    let expected: i64 = codes.iter().zip(xs).map(|(q, x)| table[*q as usize] * x).sum();
    assert_eq!(sim.get_wide("total").unwrap(), wrap(expected as i128, c.accumulator_bits));
}

#[test]
fn the_hardware_grows_with_the_format() {
    let area = |c: CellConfig| cell_design(c).flat_counts().values().sum::<u64>();
    let q4 = area(config(WeightCoding::Fixed, 4, 0, 1));
    let q8 = area(config(WeightCoding::Fixed, 8, 0, 1));
    let q16 = area(config(WeightCoding::Fixed, 16, 0, 1));
    let f16 = area(config(WeightCoding::Float, 16, 5, 1));
    let f32 = area(config(WeightCoding::Float, 32, 8, 1));
    assert!(q4 < q8 && q8 < q16, "{q4} {q8} {q16}");
    assert!(f32 > f16, "{f16} {f32}");
    let design = cell_design(config(WeightCoding::Float, 16, 5, 1));
    assert!(design.modules["compute_cell"].instances.iter().any(|i| i.name == "signed_float_adder"), "la celda incluye el decodificador");
}

#[test]
fn exported_float_cell_matches_the_simulator() {
    if Command::new("iverilog").arg("-V").output().is_err() {
        eprintln!("iverilog no está instalado: se omite");
        return;
    }
    let c = config(WeightCoding::Float, 16, 5, 2);
    let design = cell_design(c);
    let mut rng = Rng(0xFEED_0001_2345_6789);
    let mut steps = vec![TestStep { inputs: vec![("clear".into(), 1), ("exp_base".into(), 12)] }];
    for i in 0..16u128 {
        steps.push(TestStep { inputs: vec![("clear".into(), 0), ("load_en".into(), 1), ("load_addr".into(), i), ("load_data".into(), (rng.next() >> 48) as u128)] });
    }
    steps.push(TestStep { inputs: vec![("load_en".into(), 0), ("clear".into(), 1)] });
    for _ in 0..16 {
        steps.push(TestStep { inputs: vec![("clear".into(), 0), ("run".into(), 1), ("x".into(), (rng.next() >> 48) as u128)] });
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
    let dir = std::env::temp_dir().join(format!("llm2chip-float-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let source = dir.join("cell.v");
    let binary = dir.join("cell.vvp");
    std::fs::write(&source, format!("{}\n{}", to_verilog(&design), verilog_testbench(&design, &steps, "total"))).unwrap();
    let compile = Command::new("iverilog").args(["-g2005", "-s", "testbench", "-o"]).arg(&binary).arg(&source).output().unwrap();
    assert!(compile.status.success(), "{}", String::from_utf8_lossy(&compile.stderr));
    let run = Command::new("vvp").arg(&binary).output().unwrap();
    let theirs: Vec<String> = String::from_utf8_lossy(&run.stdout).lines().filter(|l| l.parse::<u64>().is_ok()).map(str::to_string).collect();
    assert_eq!(ours, theirs);
    std::fs::remove_dir_all(&dir).ok();
}
