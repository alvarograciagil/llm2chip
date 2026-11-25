//! The test port and the memories' self-test, on a whole small chip.

use netlist::chip::{chip_design, ChipConfig};
use netlist::demo::block_config;
use netlist::jtag::{IDCODE, IR_BYPASS, IR_DEBUG_SELECT, IR_DEBUG_VALUE, IR_SELF_TEST};
use netlist::sampler::SamplerConfig;
use netlist::soc::SocConfig;
use netlist::Simulator;

fn chip() -> Simulator {
    let config = ChipConfig { soc: SocConfig { program_words: 1 << 8, data_bits: 8, fifo_bits: 4, links: Default::default() }, die: netlist::DieConfig { reg_bits: 3, index_bits: 4, microcode_bits: 6, cell: netlist::CellConfig { addr_bits: 4, ..block_config().cell }, ..block_config() }, sampler: SamplerConfig { logit_bits: 16, index_bits: 12, entries: 4, lanes: 1 } };
    Simulator::new(&chip_design(config)).unwrap()
}

/// Cycles of the chip's clock in each half of TCK: the port's pins pass through two
/// flip-flops, so TCK must be several times slower than the chip's clock.
const HALF_TCK: usize = 4;

fn pulse(sim: &mut Simulator, tms: u64, tdi: u64) -> u64 {
    sim.set("jtag_tms", tms).unwrap();
    sim.set("jtag_tdi", tdi).unwrap();
    for level in [0, 1, 0] {
        sim.set("jtag_tck", level).unwrap();
        for _ in 0..HALF_TCK {
            sim.tick();
        }
    }
    sim.settle();
    sim.get("jtag_tdo").unwrap()
}

fn reset(sim: &mut Simulator) {
    for _ in 0..6 {
        pulse(sim, 1, 0);
    }
    pulse(sim, 0, 0);
}

/// Shifts `bits` through a register, least significant first, and returns what came out.
fn scan(sim: &mut Simulator, instruction: bool, value: u64, bits: u32) -> u64 {
    pulse(sim, 1, 0);
    if instruction {
        pulse(sim, 1, 0);
    }
    pulse(sim, 0, 0);
    pulse(sim, 0, 0);
    let mut out = 0;
    for bit in 0..bits {
        let last = bit + 1 == bits;
        out |= sim.get("jtag_tdo").unwrap() << bit;
        pulse(sim, last as u64, value >> bit & 1);
    }
    pulse(sim, 1, 0);
    pulse(sim, 0, 0);
    out
}

#[test]
fn the_port_identifies_the_chip_after_reset() {
    let mut sim = chip();
    reset(&mut sim);
    assert_eq!(scan(&mut sim, false, 0, 32), IDCODE as u64);
}

#[test]
fn bypass_delays_data_by_one_bit() {
    let mut sim = chip();
    reset(&mut sim);
    scan(&mut sim, true, IR_BYPASS as u64, 4);
    assert_eq!(scan(&mut sim, false, 0b1011_0110, 9), 0b1011_0110 << 1);
}

#[test]
fn counters_can_be_read_through_the_port() {
    let mut sim = chip();
    reset(&mut sim);
    scan(&mut sim, true, IR_DEBUG_SELECT as u64, 4);
    scan(&mut sim, false, 0, 3);
    scan(&mut sim, true, IR_DEBUG_VALUE as u64, 4);
    let first = scan(&mut sim, false, 0, 32);
    for _ in 0..64 {
        sim.tick();
    }
    let second = scan(&mut sim, false, 0, 32);
    assert!(second > first, "the cycle counter moves on: {first} and {second}");
}

fn run_self_test(sim: &mut Simulator, cycles: u64) -> (u64, u64) {
    reset(sim);
    scan(sim, true, IR_SELF_TEST as u64, 4);
    scan(sim, false, 1, 3);
    for _ in 0..cycles {
        sim.tick();
    }
    let status = scan(sim, false, 1, 3);
    (status >> 1 & 1, status >> 2 & 1)
}

#[test]
fn every_memory_passes_its_self_test_and_a_stuck_bit_is_found() {
    let cycles = 1 << 16;
    let (done, failed) = run_self_test(&mut chip(), cycles);
    assert_eq!((done, failed), (1, 0), "un chip sano pasa");

    let mut faulty = chip();
    faulty.stick_memory_bit("/control/data_2", 5, 3, true).unwrap();
    let (done, failed) = run_self_test(&mut faulty, cycles);
    assert_eq!((done, failed), (1, 1), "a stuck bit in the data memory is detected");
}
