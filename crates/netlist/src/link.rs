//! One end of a chip-to-chip link. Words travel with an error-correcting code and the
//! receiver holds a queue, so the board may delay them or insert gaps at will.

use crate::cell::Gen;
use crate::reliability::check_bits;
use crate::rv32::Logic;
use crate::{Bit, Builder, Design};
use std::collections::BTreeMap;

const QUEUE_BITS: usize = 4;

/// Bits on the wire for a word of `width` bits.
pub fn wire_bits(width: u32) -> u32 {
    width + check_bits(width)
}

impl Gen {
    pub(crate) fn link_endpoint(&mut self, width: u32) -> String {
        let name = format!("link_endpoint_{width}");
        if self.modules.contains_key(&name) {
            return name;
        }
        let encoder = self.secded_encoder(width);
        let decoder = self.secded_decoder(width);
        let (data_width, check) = (width as usize, check_bits(width));
        let mut b = Builder::new(&name, "Enlace entre chips", "Envía y recibe palabras protegidas por un código corrector, con cola de entrada y control de flujo.");
        let tx_send = b.input("tx_send", 1)[0];
        let tx_data = b.input("tx_data", width);
        let tx_can_send = b.output("tx_can_send", 1)[0];
        let pin_tx_valid = b.output("pin_tx_valid", 1)[0];
        let pin_tx_bits = b.output("pin_tx_bits", width + check);
        let pin_tx_ready = b.input("pin_tx_ready", 1)[0];
        let pin_rx_valid = b.input("pin_rx_valid", 1)[0];
        let pin_rx_bits = b.input("pin_rx_bits", width + check);
        let pin_rx_ready = b.output("pin_rx_ready", 1)[0];
        let rx_pop = b.input("rx_pop", 1)[0];
        let rx_data = b.output("rx_data", width);
        let rx_has_data = b.output("rx_has_data", 1)[0];
        let corrected = b.output("corrected", 1)[0];
        let failed = b.output("failed", 1)[0];

        b.group("Envío");
        let check_out = b.wire("check_out", check);
        b.inst("encoder", &encoder, &[("data", tx_data.clone()), ("check", check_out.clone())]);
        b.group("Recepción");
        let word = b.wire("received", width);
        let (fixed, bad) = (b.wire("fixed", 1)[0], b.wire("bad", 1)[0]);
        b.inst("decoder", &decoder, &[("data", pin_rx_bits[..data_width].to_vec()), ("check", pin_rx_bits[data_width..].to_vec()), ("corrected", word.clone()), ("fixed", vec![fixed]), ("failed", vec![bad])]);

        let mut l = Logic { b: &mut b, next: 0 };
        l.b.or_to(pin_tx_ready, Bit::Zero, tx_can_send);
        let wire_out: Vec<Bit> = tx_data.iter().chain(&check_out).copied().collect();
        l.connect_flops(&[tx_send], &[pin_tx_valid]);
        l.connect_flops(&wire_out, &pin_tx_bits);

        l.b.group("Cola de entrada");
        let (write, read) = (l.fresh_bus(QUEUE_BITS + 1), l.fresh_bus(QUEUE_BITS + 1));
        let inverted: Vec<Bit> = read.iter().map(|bit| l.not(*bit)).collect();
        let count = l.adder(&write, &inverted, Bit::One).0;
        let some = l.any(&count);
        let full = count[QUEUE_BITS];
        let room = l.not(full);
        let accepted = l.and(pin_rx_valid, room);
        let taken = l.and(rx_pop, some);
        let step = |l: &mut Logic, value: &[Bit], enable: Bit| -> Vec<Bit> {
            let mut carry = enable;
            value.iter().map(|bit| { let out = l.xor(*bit, carry); carry = l.and(*bit, carry); out }).collect()
        };
        let next_write = step(&mut l, &write, accepted);
        let next_read = step(&mut l, &read, taken);
        l.connect_flops(&next_write, &write);
        l.connect_flops(&next_read, &read);
        let entries: Vec<Vec<Bit>> = (0..1usize << QUEUE_BITS)
            .map(|slot| {
                let q = l.fresh_bus(data_width);
                let selected = l.equals(&write[..QUEUE_BITS], slot as u32);
                let store = l.and(selected, accepted);
                let d = l.mux_bus(store, &q, &word);
                l.connect_flops(&d, &q);
                q
            })
            .collect();
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
