//! Error correction, the march test and row repair, with faults injected into the memories.

use netlist::format::WeightCoding;
use netlist::multiplier::MultiplierKind;
use netlist::reliability::{check_bits, decode, encode, lane_check_bits, lane_encode, repairable_memory_design, secded_design};
use netlist::{cell_design, CellConfig, Simulator};

struct Noise(u64);

impl Noise {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

#[test]
fn the_code_corrects_one_error_and_detects_two() {
    let mut noise = Noise(0x1234_5678_9ABC_DEF1);
    for data_bits in [8u32, 32, 64] {
        let r = check_bits(data_bits);
        for _ in 0..200 {
            let data = (noise.next() as u128) & ((1u128 << data_bits) - 1);
            let check = encode(data, data_bits);
            assert_eq!(decode(data, check, data_bits), netlist::reliability::Decoded { data, corrected: false, failed: false });
            for bit in 0..data_bits + r {
                let (d, c) = if bit < data_bits { (data ^ (1 << bit), check) } else { (data, check ^ (1 << (bit - data_bits))) };
                let got = decode(d, c, data_bits);
                assert!(got.corrected && !got.failed && got.data == data, "{data_bits} bits, error en {bit}: {got:?}");
            }
            let (a, b) = (noise.next() as u32 % data_bits, noise.next() as u32 % data_bits);
            if a != b {
                assert!(decode(data ^ (1 << a) ^ (1 << b), check, data_bits).failed, "two errors not detected");
            }
        }
    }
    assert_eq!(check_bits(64), 8, "64 bits carry 8 check bits, 12.5 %");
}

#[test]
fn the_gates_match_the_code() {
    let data_bits = 32;
    let mut encoder = Simulator::new(&secded_design(data_bits, false)).unwrap();
    let mut decoder = Simulator::new(&secded_design(data_bits, true)).unwrap();
    let mut noise = Noise(77);
    for _ in 0..500 {
        let data = noise.next() & 0xFFFF_FFFF;
        encoder.set("data", data).unwrap();
        encoder.settle();
        let check = encoder.get("check").unwrap() as u32;
        assert_eq!(check, encode(data as u128, data_bits));
        let flips = noise.next() as u32 % 3;
        let total = data_bits + check_bits(data_bits);
        let (mut d, mut c) = (data, check);
        let (first, second) = (noise.next() as u32 % total, noise.next() as u32 % total);
        for bit in [first, second].iter().take(flips as usize) {
            if *bit < data_bits { d ^= 1 << bit } else { c ^= 1 << (bit - data_bits) }
        }
        decoder.set("data", d).unwrap();
        decoder.set("check", c as u64).unwrap();
        decoder.settle();
        let model = decode(d as u128, c, data_bits);
        assert_eq!((decoder.get("corrected").unwrap() as u128, decoder.get("fixed").unwrap() == 1, decoder.get("failed").unwrap() == 1), (model.data, model.corrected, model.failed), "data {d:#x} check {c:#x}");
    }
}

#[test]
fn every_single_error_in_a_lane_word_has_its_own_syndrome() {
    let total = 64 + lane_check_bits(8, 8);
    let word = 0x0123_4567_89AB_CDEFu128;
    let good = lane_encode(word, 8, 8);
    let mut seen = std::collections::HashSet::new();
    for bit in 0..total {
        let syndrome = if bit < 64 { lane_encode(word ^ 1 << bit, 8, 8) ^ good } else { 1 << (bit - 64) };
        assert_eq!(syndrome.count_ones() & 1, 1, "un error deja paridad impar");
        assert!(seen.insert(syndrome), "bit {bit}");
    }
    assert_eq!(lane_check_bits(8, 8), 9);
}

fn protected_cell() -> CellConfig {
    CellConfig { weight_bits: 8, activation_bits: 16, accumulator_bits: 32, mac_stages: 2, adder_stages: 2, addr_bits: 5, coding: WeightCoding::default(), exponent_bits: 0, multiplier: MultiplierKind::Array, fixed_weights: false, ecc: true, skip_zeros: false }
}

const ACTIVATIONS: [u64; 8] = [100, 0xFFCE, 25, 8, 3, 0xFFF0, 77, 1];

fn loaded_cell(weights: &[u64]) -> Simulator {
    let mut sim = Simulator::new(&cell_design(protected_cell())).unwrap();
    sim.set("load_en", 1).unwrap();
    for (i, word) in weights.chunks(8).enumerate() {
        let data = word.iter().rev().fold(0u128, |all, w| all << 8 | *w as u128);
        sim.set("load_addr", i as u64).unwrap();
        sim.set_wide("load_data", data | (lane_encode(data, 8, 8) as u128) << 64).unwrap();
        sim.tick();
    }
    sim.set("load_en", 0).unwrap();
    sim.set("clear", 1).unwrap();
    sim.tick();
    sim.set("clear", 0).unwrap();
    sim
}

/// Runs the cell and returns the total and whether each flag was ever raised.
fn run(sim: &mut Simulator) -> (u64, bool, bool) {
    let (mut corrected, mut failed) = (false, false);
    sim.set("run", 1).unwrap();
    for x in ACTIVATIONS {
        sim.set("x", x).unwrap();
        sim.tick();
        corrected |= sim.get("ecc_corrected").unwrap() == 1;
        failed |= sim.get("ecc_failed").unwrap() == 1;
    }
    sim.set("run", 0).unwrap();
    for _ in 0..protected_cell().latency() {
        sim.tick();
        corrected |= sim.get("ecc_corrected").unwrap() == 1;
        failed |= sim.get("ecc_failed").unwrap() == 1;
    }
    (sim.get("total").unwrap(), corrected, failed)
}

#[test]
fn a_flipped_weight_bit_does_not_change_the_result() {
    let weights = [3, 0xFE, 5, 0x81, 1, 2, 0x7F, 9];
    let (clean, corrected, failed) = run(&mut loaded_cell(&weights));
    assert!(!corrected && !failed);

    let mut hit = loaded_cell(&weights);
    hit.flip_memory_bit("/weight_sram", 0, 13).unwrap();
    let (total, corrected, failed) = run(&mut hit);
    assert_eq!(total, clean, "the ECC must correct the flipped bit");
    assert!(corrected && !failed);

    let mut twice = loaded_cell(&weights);
    twice.flip_memory_bit("/weight_sram", 0, 13).unwrap();
    twice.flip_memory_bit("/weight_sram", 0, 40).unwrap();
    assert!(run(&mut twice).2, "dos bits deben avisar");
}

fn start_test(sim: &mut Simulator) -> u64 {
    sim.set("test_start", 1).unwrap();
    sim.tick();
    sim.set("test_start", 0).unwrap();
    let mut cycles = 1;
    while sim.get("test_done").unwrap() == 0 {
        sim.tick();
        cycles += 1;
        assert!(cycles < 200_000, "the self-test does not finish");
    }
    cycles
}

fn write(sim: &mut Simulator, address: u64, value: u64) {
    sim.set("we", 1).unwrap();
    sim.set("addr", address).unwrap();
    sim.set("wdata", value).unwrap();
    sim.tick();
    sim.set("we", 0).unwrap();
}

fn read(sim: &mut Simulator, address: u64) -> u64 {
    sim.set("addr", address).unwrap();
    sim.tick();
    sim.get("rdata").unwrap()
}

const WORDS: u64 = 64;

#[test]
fn a_good_memory_passes_in_eleven_cycles_per_word() {
    let mut sim = Simulator::new(&repairable_memory_design(6, 16, 2)).unwrap();
    let cycles = start_test(&mut sim);
    assert_eq!(sim.get("test_failed"), Some(0));
    assert_eq!(sim.get("spares_used"), Some(0));
    assert!((11 * WORDS..12 * WORDS).contains(&cycles), "{cycles} ciclos");
}

#[test]
fn bad_rows_are_found_and_replaced() {
    let mut sim = Simulator::new(&repairable_memory_design(6, 16, 2)).unwrap();
    sim.stick_memory_bit("/cells", 17, 3, true).unwrap();
    sim.stick_memory_bit("/cells", 40, 9, false).unwrap();
    start_test(&mut sim);
    assert_eq!(sim.get("test_failed"), Some(0));
    assert_eq!(sim.get("spares_used"), Some(2));
    for address in 0..WORDS {
        write(&mut sim, address, address * 3 + 1);
    }
    for address in 0..WORDS {
        assert_eq!(read(&mut sim, address), address * 3 + 1, "address {address}");
    }
}

#[test]
fn too_many_bad_rows_are_reported() {
    let mut sim = Simulator::new(&repairable_memory_design(6, 16, 2)).unwrap();
    for address in [5, 20, 33] {
        sim.stick_memory_bit("/cells", address, 0, true).unwrap();
    }
    start_test(&mut sim);
    assert_eq!(sim.get("test_failed"), Some(1));
}
