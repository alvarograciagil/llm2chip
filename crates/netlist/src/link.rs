//! One end of a chip-to-chip link. Words travel with an error-correcting code and the
//! receiver holds a queue, so the board may delay them or insert gaps at will.

use crate::cell::Gen;
use crate::reliability::check_bits;
use crate::rv32::Logic;
use crate::{Bit, Builder, Design};
use std::collections::BTreeMap;

/// log2 of the words the receiver's queue holds: it asks the sender to wait at half.
pub const QUEUE_BITS: usize = 6;
/// log2 of the words a sender keeps for repeats: it sends up to half as many before it
/// hears that they arrived, at least a chunk of an exchange between chips.
pub const SEQUENCE_BITS: usize = 5;
/// Cycles to wait, as a power of two, before asking again for a repeat that never came.
const RETRY_BITS: usize = 7;
/// The next word the receiver expects, and whether it is asking for a repeat.
pub const ACK_BITS: usize = SEQUENCE_BITS + 1;

/// Bits on the wire for a word of `width` bits.
pub fn wire_bits(width: u32) -> u32 {
    let framed = width + SEQUENCE_BITS as u32;
    framed + check_bits(framed)
}

/// Registers `d` into `q` only on the cycles when its clock domain ticks.
fn clocked(l: &mut Logic, stall: Bit, d: &[Bit], q: &[Bit]) {
    let held = l.mux_bus(stall, d, q);
    l.connect_flops(&held, q);
}

fn increment(l: &mut Logic, value: &[Bit], enable: Bit) -> Vec<Bit> {
    let mut carry = enable;
    value
        .iter()
        .map(|bit| {
            let out = l.xor(*bit, carry);
            carry = l.and(*bit, carry);
            out
        })
        .collect()
}

fn to_gray(l: &mut Logic, value: &[Bit]) -> Vec<Bit> {
    (0..value.len()).map(|i| if i + 1 < value.len() { l.xor(value[i], value[i + 1]) } else { value[i] }).collect()
}

fn from_gray(l: &mut Logic, gray: &[Bit]) -> Vec<Bit> {
    let mut out = vec![Bit::Zero; gray.len()];
    let mut above = Bit::Zero;
    for i in (0..gray.len()).rev() {
        above = l.xor(above, gray[i]);
        out[i] = above;
    }
    out
}

fn same(l: &mut Logic, a: &[Bit], b: &[Bit]) -> Bit {
    let differs: Vec<Bit> = a.iter().zip(b).map(|(x, y)| l.xor(*x, *y)).collect();
    let any = l.any(&differs);
    l.not(any)
}

impl Gen {
    /// One end of a link. The receiving half runs on the sender's clock, which comes with
    /// the data, and hands words to this chip's clock through a queue with Gray pointers.
    pub(crate) fn link_endpoint(&mut self, width: u32) -> String {
        let name = format!("link_endpoint_{width}");
        if self.modules.contains_key(&name) {
            return name;
        }
        let framed = width + SEQUENCE_BITS as u32;
        let encoder = self.secded_encoder(framed);
        let decoder = self.secded_decoder(framed);
        let (data_width, frame_width, check) = (width as usize, framed as usize, check_bits(framed));
        let mut b = Builder::new(&name, "Chip-to-chip link", "Sends numbered, protected words, repeats those that arrive damaged and crosses between the two chips' clocks.");
        let tx_send = b.input("tx_send", 1)[0];
        let tx_data = b.input("tx_data", width);
        let tx_can_send = b.output("tx_can_send", 1)[0];
        let pin_tx_valid = b.output("pin_tx_valid", 1)[0];
        let pin_tx_bits = b.output("pin_tx_bits", framed + check);
        let pin_tx_ready = b.input("pin_tx_ready", 1)[0];
        let pin_tx_ack = b.input("pin_tx_ack", ACK_BITS as u32);
        let pin_rx_valid = b.input("pin_rx_valid", 1)[0];
        let pin_rx_bits = b.input("pin_rx_bits", framed + check);
        let pin_rx_ready = b.output("pin_rx_ready", 1)[0];
        let pin_rx_ack = b.output("pin_rx_ack", ACK_BITS as u32);
        // High on the cycles when the sender's clock has no edge.
        let rx_stall = b.input("pin_rx_stall", 1)[0];
        let core_stall = b.input("core_stall", 1)[0];
        let rx_pop = b.input("rx_pop", 1)[0];
        let rx_data = b.output("rx_data", width);
        let rx_has_data = b.output("rx_has_data", 1)[0];
        let corrected = b.output("corrected", 1)[0];
        let failed = b.output("failed", 1)[0];

        b.group("Sending");
        let frame_out = b.wire("frame_out", framed);
        let check_out = b.wire("check_out", check);
        b.inst("encoder", &encoder, &[("data", frame_out.clone()), ("check", check_out.clone())]);
        b.group("Receiving");
        let frame_in = b.wire("received", framed);
        let (fixed, bad) = (b.wire("fixed", 1)[0], b.wire("bad", 1)[0]);
        b.inst("decoder", &decoder, &[("data", pin_rx_bits[..frame_width].to_vec()), ("check", pin_rx_bits[frame_width..].to_vec()), ("corrected", frame_in.clone()), ("fixed", vec![fixed]), ("failed", vec![bad])]);

        let mut l = Logic { b: &mut b, next: 0 };
        l.b.group("Retry");
        let (next, base) = (l.fresh_bus(SEQUENCE_BITS), l.fresh_bus(SEQUENCE_BITS));
        let replay = l.fresh_bus(SEQUENCE_BITS);
        let (replaying, rewound) = (l.fresh(), l.fresh());
        let (nak, expected_in) = (pin_tx_ack[SEQUENCE_BITS], pin_tx_ack[..SEQUENCE_BITS].to_vec());
        let not_rewound = l.not(rewound);
        let first_request = l.and(nak, not_rewound);
        let caught_up = same(&mut l, &replay, &next);
        let still = l.not(caught_up);
        // A repeat can be damaged too: if the request outlives a whole replay, it is repeated.
        let retry = l.fresh_bus(RETRY_BITS);
        let waiting_again = {
            let idle = l.not(replaying);
            let t = l.and(nak, rewound);
            l.and(t, idle)
        };
        let timed_out = retry.iter().fold(waiting_again, |all, bit| l.and(all, *bit));
        let rewind = l.or(first_request, timed_out);
        let retry_counted = increment(&mut l, &retry, waiting_again);
        let retry_clear = {
            let idle = l.not(waiting_again);
            l.or(idle, timed_out)
        };
        let retry_d: Vec<Bit> = retry_counted.iter().map(|bit| {
            let keep = l.not(retry_clear);
            l.and(*bit, keep)
        }).collect();
        clocked(&mut l, core_stall, &retry_d, &retry);
        let replay_on = l.and(replaying, still);
        let replay_sends = l.and(replay_on, pin_tx_ready);
        let active_core = l.not(core_stall);
        let accepted_new = l.and(tx_send, active_core);
        // Waiting words are counted so the numbers never wrap past what is unacknowledged.
        let inverted: Vec<Bit> = base.iter().map(|bit| l.not(*bit)).collect();
        let waiting = l.adder(&next, &inverted, Bit::One).0;
        let window_full = waiting[SEQUENCE_BITS - 1];
        let open = l.not(window_full);
        let quiet = {
            let replay_off = l.not(replaying);
            let rewind_off = l.not(rewind);
            l.and(replay_off, rewind_off)
        };
        let can_send = {
            let t = l.and(pin_tx_ready, open);
            let t = l.and(t, quiet);
            l.and(t, active_core)
        };
        l.b.or_to(can_send, Bit::Zero, tx_can_send);

        let next_after = increment(&mut l, &next, accepted_new);
        clocked(&mut l, core_stall, &next_after, &next);
        clocked(&mut l, core_stall, &expected_in, &base);
        let stepped = increment(&mut l, &replay, replay_sends);
        let replay_d = l.mux_bus(rewind, &stepped, &expected_in);
        clocked(&mut l, core_stall, &replay_d, &replay);
        let replay_keep = l.and(replaying, still);
        let replay_set = l.or(rewind, replay_keep);
        clocked(&mut l, core_stall, &[replay_set], &[replaying]);
        clocked(&mut l, core_stall, &[nak], &[rewound]);

        let stored: Vec<Vec<Bit>> = (0..1usize << SEQUENCE_BITS)
            .map(|slot| {
                let q = l.fresh_bus(data_width);
                let here = l.equals(&next, slot as u32);
                let write = l.and(here, accepted_new);
                let d = l.mux_bus(write, &q, &tx_data);
                l.connect_flops(&d, &q);
                q
            })
            .collect();
        let replayed = l.select(&replay, &stored);
        let data = l.mux_bus(replay_sends, &tx_data, &replayed);
        let sequence = l.mux_bus(replay_sends, &next, &replay);
        for (from, to) in data.iter().chain(&sequence).zip(&frame_out) {
            l.b.or_to(*from, Bit::Zero, *to);
        }
        let send_any = l.or(accepted_new, replay_sends);
        let wire_out: Vec<Bit> = frame_out.iter().chain(&check_out).copied().collect();
        clocked(&mut l, core_stall, &[send_any], &[pin_tx_valid]);
        clocked(&mut l, core_stall, &wire_out, &pin_tx_bits);

        l.b.group("Input queue");
        let (write, read) = (l.fresh_bus(QUEUE_BITS + 1), l.fresh_bus(QUEUE_BITS + 1));
        let read_gray = l.fresh_bus(QUEUE_BITS + 1);
        let write_gray = l.fresh_bus(QUEUE_BITS + 1);
        // Two flip-flops in the receiving domain settle a pointer that changed mid-clock.
        let (read_sync_a, read_sync_b) = (l.fresh_bus(QUEUE_BITS + 1), l.fresh_bus(QUEUE_BITS + 1));
        clocked(&mut l, rx_stall, &read_gray, &read_sync_a);
        clocked(&mut l, rx_stall, &read_sync_a, &read_sync_b);
        let (write_sync_a, write_sync_b) = (l.fresh_bus(QUEUE_BITS + 1), l.fresh_bus(QUEUE_BITS + 1));
        clocked(&mut l, core_stall, &write_gray, &write_sync_a);
        clocked(&mut l, core_stall, &write_sync_a, &write_sync_b);

        let read_seen = from_gray(&mut l, &read_sync_b);
        let inverted: Vec<Bit> = read_seen.iter().map(|bit| l.not(*bit)).collect();
        let count = l.adder(&write, &inverted, Bit::One).0;
        let full = count[QUEUE_BITS];
        let room = l.not(full);

        let expected = l.fresh_bus(SEQUENCE_BITS);
        let matches_sequence = same(&mut l, &frame_in[data_width..], &expected);
        let intact = l.not(bad);
        let in_order = l.and(matches_sequence, intact);
        let valid = pin_rx_valid;
        let wanted = l.and(valid, in_order);
        let accepted = l.and(wanted, room);
        let expected_next = increment(&mut l, &expected, accepted);
        clocked(&mut l, rx_stall, &expected_next, &expected);
        // Once anything is refused, the sender is told until the missing word arrives.
        let nak_q = l.fresh();
        // A word older than the one expected is a harmless repeat, not a loss.
        let distance = {
            let inverted: Vec<Bit> = expected.iter().map(|bit| l.not(*bit)).collect();
            l.adder(&frame_in[data_width..], &inverted, Bit::One).0
        };
        let behind = distance[SEQUENCE_BITS - 1];
        let harmless = l.and(behind, intact);
        let refused = {
            let not_accepted = l.not(accepted);
            let not_harmless = l.not(harmless);
            let t = l.and(valid, not_accepted);
            l.and(t, not_harmless)
        };
        let nak_kept = {
            let not_accepted = l.not(accepted);
            l.and(nak_q, not_accepted)
        };
        let nak_d = l.or(refused, nak_kept);
        clocked(&mut l, rx_stall, &[nak_d], &[nak_q]);
        let ack_out: Vec<Bit> = expected.iter().copied().chain([nak_q]).collect();
        for (from, to) in ack_out.iter().zip(&pin_rx_ack) {
            l.b.or_to(*from, Bit::Zero, *to);
        }

        let write_next = increment(&mut l, &write, accepted);
        clocked(&mut l, rx_stall, &write_next, &write);
        let write_next_gray = to_gray(&mut l, &write_next);
        clocked(&mut l, rx_stall, &write_next_gray, &write_gray);
        let entries: Vec<Vec<Bit>> = (0..1usize << QUEUE_BITS)
            .map(|slot| {
                let q = l.fresh_bus(data_width);
                let selected = l.equals(&write[..QUEUE_BITS], slot as u32);
                let store = l.and(selected, accepted);
                let d = l.mux_bus(store, &q, &frame_in[..data_width]);
                l.connect_flops(&d, &q);
                q
            })
            .collect();

        let write_seen = from_gray(&mut l, &write_sync_b);
        let empty = same(&mut l, &write_seen, &read);
        let some = l.not(empty);
        let taken = {
            let t = l.and(rx_pop, some);
            l.and(t, active_core)
        };
        let read_next = increment(&mut l, &read, taken);
        clocked(&mut l, core_stall, &read_next, &read);
        let read_next_gray = to_gray(&mut l, &read_next);
        clocked(&mut l, core_stall, &read_next_gray, &read_gray);
        let head = l.select(&read[..QUEUE_BITS], &entries);
        for (from, to) in head.iter().zip(&rx_data) {
            l.b.or_to(*from, Bit::Zero, *to);
        }
        l.b.or_to(some, Bit::Zero, rx_has_data);
        // Half the queue stays free for what is already on its way.
        let half_full = l.or(count[QUEUE_BITS - 1], full);
        let ready = l.not(half_full);
        l.b.or_to(ready, Bit::Zero, pin_rx_ready);
        let was_fixed = l.and(fixed, pin_rx_valid);
        let was_bad = l.and(bad, pin_rx_valid);
        l.b.or_to(was_fixed, Bit::Zero, corrected);
        l.b.or_to(was_bad, Bit::Zero, failed);
        self.add(b.finish())
    }
}

pub fn link_endpoint_design(width: u32) -> Design {
    let mut gen = Gen { modules: BTreeMap::new() };
    let top = gen.link_endpoint(width);
    Design { top, modules: gen.modules }
}
