//! The chip-to-chip link: order, flow control and damaged words.

use netlist::link::{link_endpoint_design, wire_bits};
use netlist::reliability::encode;
use netlist::Simulator;

const WIDTH: u32 = 16;

const SEQUENCE: u32 = 4;

/// A word as it travels: its data, its number and their check bits.
fn numbered(value: u64, sequence: u64) -> u64 {
    let frame = value | sequence << WIDTH;
    frame | (encode(frame as u128, WIDTH + SEQUENCE) as u64) << (WIDTH + SEQUENCE)
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

fn settle_queue(sim: &mut Simulator) {
    for _ in 0..4 {
        sim.tick();
    }
}

#[test]
fn words_arrive_in_order_through_the_queue() {
    let mut link = sim();
    for (sequence, value) in [11, 22, 33, 44].into_iter().enumerate() {
        receive(&mut link, numbered(value, sequence as u64));
    }
    settle_queue(&mut link);
    assert_eq!([pop(&mut link), pop(&mut link), pop(&mut link), pop(&mut link)], [11, 22, 33, 44]);
    settle_queue(&mut link);
    link.settle();
    assert_eq!(link.get("rx_has_data"), Some(0));
}

#[test]
fn the_receiver_asks_the_sender_to_wait_before_it_overflows() {
    let mut link = sim();
    let mut accepted = 0;
    link.settle();
    while link.get("pin_rx_ready") == Some(1) {
        receive(&mut link, numbered(accepted, accepted % 16));
        accepted += 1;
        link.settle();
        assert!(accepted < 16, "la cola debe avisar antes de llenarse");
    }
    settle_queue(&mut link);
    for value in 0..accepted {
        assert_eq!(pop(&mut link), value);
    }
}

#[test]
fn single_bit_errors_are_corrected_and_double_ones_reported() {
    let total = wire_bits(WIDTH);
    for bit in 0..total {
        let mut link = sim();
        link.set("pin_rx_valid", 1).unwrap();
        link.set("pin_rx_bits", numbered(0xBEEF, 0) ^ (1 << bit)).unwrap();
        link.settle();
        assert_eq!(link.get("corrected"), Some(1), "bit {bit}");
        assert_eq!(link.get("failed"), Some(0));
        link.tick();
        link.set("pin_rx_valid", 0).unwrap();
        settle_queue(&mut link);
        assert_eq!(pop(&mut link), 0xBEEF, "bit {bit}");
    }
    let mut link = sim();
    link.set("pin_rx_valid", 1).unwrap();
    link.set("pin_rx_bits", numbered(0xBEEF, 0) ^ 0b101).unwrap();
    link.settle();
    assert_eq!(link.get("failed"), Some(1));
}

struct Noise(u64);

impl Noise {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0 >> 16
    }
}

/// Two ends joined by wires, sending from A to B, with the given odds of trouble.
/// Each end runs on its own clock, whose edges the two `every` values space out.
fn deliver(words: u64, double_every: u64, a_every: u64, b_every: u64) -> (Vec<u64>, u32) {
    let (mut a, mut b) = (sim(), sim());
    let mut noise = Noise(0x1234_5678_9ABC_DEF1);
    let mut delayed = std::collections::VecDeque::from(vec![(0u64, 0u64); 3]);
    let mut acks = std::collections::VecDeque::from(vec![(0u64, 1u64); 3]);
    let (mut sent, mut received, mut doubles) = (0, Vec::new(), 0);
    for _ in 0..200_000 {
        if received.len() as u64 == words {
            break;
        }
        let a_edge = noise.next() % a_every == 0;
        let b_edge = noise.next() % b_every == 0;
        a.set("core_stall", (!a_edge) as u64).unwrap();
        b.set("core_stall", (!b_edge) as u64).unwrap();
        // The receiving half of B runs on the edges of A, whose clock comes with the data.
        b.set("pin_rx_stall", (!a_edge) as u64).unwrap();
        a.settle();
        b.settle();

        let (ack, ready) = acks.pop_front().unwrap();
        a.set("pin_tx_ack", ack).unwrap();
        a.set("pin_tx_ready", ready).unwrap();
        a.settle();
        let wants = sent < words && a.get("tx_can_send") == Some(1);
        a.set("tx_send", wants as u64).unwrap();
        a.set("tx_data", 0x4000 + sent).unwrap();

        let (valid, mut bits) = if a_edge { delayed.pop_front().unwrap() } else { (0, 0) };
        if valid == 1 && noise.next() % double_every == 0 {
            bits ^= 0b11 << (noise.next() % 8);
            doubles += 1;
        }
        b.set("pin_rx_valid", valid).unwrap();
        b.set("pin_rx_bits", bits).unwrap();
        b.settle();
        let popping = b_edge && b.get("rx_has_data") == Some(1);
        b.set("rx_pop", popping as u64).unwrap();
        b.settle();
        if popping {
            received.push(b.get("rx_data").unwrap());
        }
        sent += wants as u64;
        if a_edge {
            delayed.push_back((a.get("pin_tx_valid").unwrap(), a.get("pin_tx_bits").unwrap()));
            acks.push_back((b.get("pin_rx_ack").unwrap(), b.get("pin_rx_ready").unwrap()));
        } else {
            acks.push_front(acks.front().copied().unwrap_or((ack, ready)));
            let _ = (ack, ready);
        }
        a.tick();
        b.tick();
    }
    (received, doubles)
}

#[test]
fn damaged_words_are_sent_again_and_arrive_in_order() {
    let (received, doubles) = deliver(300, 12, 1, 1);
    assert!(doubles > 5, "la prueba debe dañar palabras: {doubles}");
    assert_eq!(received, (0..300).map(|i| 0x4000 + i).collect::<Vec<_>>());
}

#[test]
fn words_cross_between_two_unrelated_clocks() {
    for (a_every, b_every) in [(2, 3), (3, 2), (1, 5), (5, 1)] {
        let (received, _) = deliver(120, u64::MAX, a_every, b_every);
        assert_eq!(received, (0..120).map(|i| 0x4000 + i).collect::<Vec<_>>(), "relojes {a_every} y {b_every}");
    }
}
