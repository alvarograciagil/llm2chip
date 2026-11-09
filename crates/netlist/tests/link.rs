//! The chip-to-chip link: order, flow control and damaged words.

use netlist::link::{link_endpoint_design, wire_bits};
use netlist::reliability::encode;
use netlist::Simulator;

const WIDTH: u32 = 16;

fn wire_word(value: u64) -> u64 {
    value | (encode(value as u128, WIDTH) as u64) << WIDTH
}

fn sim() -> Simulator {
    Simulator::new(&link_endpoint_design(WIDTH)).unwrap()
}

fn receive(sim: &mut Simulator, bits: u64) {
    sim.set("pin_rx_valid", 1).unwrap();
    sim.set("pin_rx_bits", bits).unwrap();
    sim.tick();
    sim.set("pin_rx_valid", 0).unwrap();
}

fn pop(sim: &mut Simulator) -> u64 {
    sim.settle();
    assert_eq!(sim.get("rx_has_data"), Some(1));
    let value = sim.get("rx_data").unwrap();
    sim.set("rx_pop", 1).unwrap();
    sim.tick();
    sim.set("rx_pop", 0).unwrap();
    value
}

#[test]
fn words_arrive_in_order_through_the_queue() {
    let mut link = sim();
    for value in [11, 22, 33, 44] {
        receive(&mut link, wire_word(value));
    }
    assert_eq!([pop(&mut link), pop(&mut link), pop(&mut link), pop(&mut link)], [11, 22, 33, 44]);
    link.settle();
    assert_eq!(link.get("rx_has_data"), Some(0));
}

#[test]
fn the_receiver_asks_the_sender_to_wait_before_it_overflows() {
    let mut link = sim();
    let mut accepted = 0;
    while link.get("pin_rx_ready") == Some(1) {
        receive(&mut link, wire_word(accepted));
        accepted += 1;
        link.settle();
        assert!(accepted < 16, "la cola debe avisar antes de llenarse");
    }
    // What was already travelling still fits.
    for extra in 0..8 {
        receive(&mut link, wire_word(1000 + extra));
    }
    for value in 0..accepted {
        assert_eq!(pop(&mut link), value);
    }
    for extra in 0..8 {
        assert_eq!(pop(&mut link), 1000 + extra);
    }
}

#[test]
fn single_bit_errors_are_corrected_and_double_ones_reported() {
    let total = wire_bits(WIDTH);
    for bit in 0..total {
        let mut link = sim();
        link.set("pin_rx_valid", 1).unwrap();
        link.set("pin_rx_bits", wire_word(0xBEEF) ^ (1 << bit)).unwrap();
        link.settle();
        assert_eq!(link.get("corrected"), Some(1), "bit {bit}");
        assert_eq!(link.get("failed"), Some(0));
        link.tick();
        link.set("pin_rx_valid", 0).unwrap();
        assert_eq!(pop(&mut link), 0xBEEF, "bit {bit}");
    }
    let mut link = sim();
    link.set("pin_rx_valid", 1).unwrap();
    link.set("pin_rx_bits", wire_word(0xBEEF) ^ 0b101).unwrap();
    link.settle();
    assert_eq!(link.get("failed"), Some(1));
}
