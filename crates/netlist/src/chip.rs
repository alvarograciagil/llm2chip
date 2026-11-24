//! The complete chip: control system, die and sampling unit, joined the way silicon
//! would join them. Everything outside is the host port and the image port.

use crate::block::exponent_register;
use crate::cell::Gen;
use crate::die::{word_bits, x_bits};
use crate::rv32::{constant, Logic};
use crate::sampler::{sampler, SamplerConfig, SamplerModel, TABLE_BITS};
use crate::soc::{soc, LoadWrite, Peripheral, SocConfig, PERIPHERAL_BITS};
use crate::{Bit, Builder, Design, DieConfig, Microprogram};
use serde::Serialize;
use std::collections::BTreeMap;

/// The die's parameter registers occupy the first words of the peripheral bus.
pub const REG_ENTRY: u32 = 32;
pub const REG_STATUS: u32 = 33;
pub const REG_VECTOR_ADDRESS: u32 = 34;
pub const REG_VECTOR_DATA: u32 = 35;
pub const REG_SAMPLE_BASE: u32 = 36;
pub const REG_SAMPLE_COUNT: u32 = 37;
pub const REG_SAMPLE_INVERSE: u32 = 38;
pub const REG_SAMPLE_LIMIT: u32 = 39;
pub const REG_SAMPLE_GREEDY: u32 = 40;
pub const REG_SAMPLE_SEED: u32 = 41;
pub const REG_SAMPLE_START: u32 = 42;
pub const REG_SAMPLE_DONE: u32 = 43;
pub const REG_SAMPLE_TOKEN: u32 = 44;
pub const REG_FOLLOWER: u32 = 45;
pub const REG_CHAIN_DATA: u32 = 46;
pub const REG_CHAIN_STATUS: u32 = 47;
/// How many more orders the die's command queue takes, where the die has one.
pub const REG_QUEUE_ROOM: u32 = 49;

/// What the debug port shows for each value of `debug_select`.
pub const DEBUG_CYCLES: u64 = 0;
pub const DEBUG_DIE_BUSY: u64 = 1;
pub const DEBUG_SAMPLER_BUSY: u64 = 2;
pub const DEBUG_CPU_PC: u64 = 3;
pub const DEBUG_LINK_CORRECTED: u64 = 4;
pub const DEBUG_LINK_FAILED: u64 = 5;
/// Cycles the chip keeps itself in reset after the external signal is released.
pub const RESET_STRETCH: u64 = 15;
/// Cycles from the release of the reset pin until the chip is out of reset: the two
/// flip-flops that synchronize the pin, then the stretch.
pub const RESET_RELEASE: u64 = RESET_STRETCH + 2;

#[derive(Clone, Copy, Debug, Serialize)]
pub struct ChipConfig {
    pub soc: SocConfig,
    pub die: DieConfig,
    pub sampler: SamplerConfig,
}

/// `alone`: a chip that will always work by itself, as on one FPGA board, without the
/// manufacturing tests (JTAG and the memories' march tests) or the chain to other chips.
/// Its pins are the same, held at zero.
fn chip(gen: &mut Gen, c: ChipConfig, alone: bool) -> String {
    let control = soc(gen, c.soc);
    let die = gen.die(&c.die);
    let sample = sampler(gen, c.sampler);
    let activation = c.die.cell.activation_bits as usize;
    let vector_bits = (c.die.reg_bits + c.die.index_bits) as usize;
    let cell_bits = c.die.scale_bits.max(c.die.cell.weight_bits).max(c.die.cell.activation_bits) as usize;
    let function_address = crate::die::function_address_bits(&c.die) as usize;
    let limit_bits = (c.sampler.entries as usize + 1).next_power_of_two().trailing_zeros() as usize;

    assert_eq!(c.sampler.lanes, 1, "the reduced die delivers only one logit per cycle");
    let mut b = Builder::new("llm2chip_top", "Complete chip", "Control CPU, memories, load controller, inference die and sampling unit, with the host and image ports as the only contacts with the outside.");
    let reset = b.input("reset", 1)[0];
    let cpu_hold = b.input("cpu_hold", 1)[0];
    let host_rx_valid = b.input("host_rx_valid", 1);
    let host_rx_data = b.input("host_rx_data", 8);
    let host_tx_ready = b.input("host_tx_ready", 1);
    let image_resp_valid = b.input("image_resp_valid", 1);
    let image_resp_data = b.input("image_resp_data", 32);
    let host_rx_ready = b.output("host_rx_ready", 1);
    let host_tx_valid = b.output("host_tx_valid", 1);
    let host_tx_data = b.output("host_tx_data", 8);
    let halted = b.output("halted", 1);
    let booting = b.output("booting", 1);
    let image_req_valid = b.output("image_req_valid", 1);
    let image_req_addr = b.output("image_req_addr", 32);
    // The only thing a board decides: which chip leads. Numbers are handed out at start-up.
    let follower = b.input("follower", 1);
    let control_bits = crate::link::wire_bits(32);
    let data_bits = crate::link::wire_bits(c.die.cell.activation_bits);
    let link_tx_valid = b.output("link_tx_valid", 1);
    let link_tx_bits = b.output("link_tx_bits", data_bits);
    let link_tx_ready = b.input("link_tx_ready", 1);
    let link_rx_valid = b.input("link_rx_valid", 1);
    let link_rx_bits = b.input("link_rx_bits", data_bits);
    let link_rx_ready = b.output("link_rx_ready", 1);
    let link_tx_ack = b.input("link_tx_ack", crate::link::ACK_BITS as u32);
    let link_rx_ack = b.output("link_rx_ack", crate::link::ACK_BITS as u32);
    let link_rx_stall = b.input("link_rx_stall", 1);
    // A wide link's further lanes, and the loop back of a group's ring, go straight from
    // the die to pins of the same names.
    let ack = crate::link::ACK_BITS as u32;
    let loops = if c.die.features.uses_loop() { c.die.features.link_lanes() } else { 0 };
    let prefixes: Vec<String> = (1..c.die.features.link_lanes()).map(|k| format!("link{k}")).chain((0..loops).map(|k| if k == 0 { "loop".to_string() } else { format!("loop{k}") })).collect();
    let extra_link: Vec<(String, Vec<Bit>)> = prefixes
        .iter()
        .flat_map(|p| {
            [
                (format!("{p}_tx_valid"), b.output(&format!("{p}_tx_valid"), 1)),
                (format!("{p}_tx_bits"), b.output(&format!("{p}_tx_bits"), data_bits)),
                (format!("{p}_tx_ready"), b.input(&format!("{p}_tx_ready"), 1)),
                (format!("{p}_rx_valid"), b.input(&format!("{p}_rx_valid"), 1)),
                (format!("{p}_rx_bits"), b.input(&format!("{p}_rx_bits"), data_bits)),
                (format!("{p}_rx_ready"), b.output(&format!("{p}_rx_ready"), 1)),
                (format!("{p}_tx_ack"), b.input(&format!("{p}_tx_ack"), ack)),
                (format!("{p}_rx_ack"), b.output(&format!("{p}_rx_ack"), ack)),
            ]
        })
        .collect();
    let chain_tx_valid = b.output("chain_tx_valid", 1);
    let chain_tx_bits = b.output("chain_tx_bits", control_bits);
    let chain_tx_ready = b.input("chain_tx_ready", 1);
    let chain_rx_valid = b.input("chain_rx_valid", 1);
    let chain_rx_bits = b.input("chain_rx_bits", control_bits);
    let chain_rx_ready = b.output("chain_rx_ready", 1);
    let chain_tx_ack = b.input("chain_tx_ack", crate::link::ACK_BITS as u32);
    let chain_rx_ack = b.output("chain_rx_ack", crate::link::ACK_BITS as u32);
    let chain_rx_stall = b.input("chain_rx_stall", 1);
    let debug_select = b.input("debug_select", 3);
    let debug_value = b.output("debug_value", 32);
    let jtag_tck = b.input("jtag_tck", 1);
    let jtag_tms = b.input("jtag_tms", 1);
    let jtag_tdi = b.input("jtag_tdi", 1);
    let jtag_tdo = b.output("jtag_tdo", 1);
    let self_test_pin = b.input("self_test", 1)[0];
    let self_test_done = b.output("self_test_done", 1);
    let self_test_fail = b.output("self_test_fail", 1);
    let bist_enable = b.wire("bist_enable", 1)[0];
    let (bist_done, bist_fail) = (b.wire("bist_done", 1)[0], b.wire("bist_fail", 1)[0]);
    let (tap_select, tap_override, tap_test) = (b.wire("tap_select", 3), b.wire("tap_override", 1)[0], b.wire("tap_test", 1)[0]);
    let debug_bus = b.wire("debug_bus", 32);
    b.group("Test port");
    if alone {
        for bit in tap_select.iter().chain(&[tap_override, tap_test, bist_done, bist_fail]).chain(&jtag_tdo) {
            b.or_to(Bit::Zero, Bit::Zero, *bit);
        }
    } else {
        let tap = gen.jtag_tap();
        b.inst("jtag", &tap, &[("tck", jtag_tck), ("tms", jtag_tms), ("tdi", jtag_tdi), ("tdo", jtag_tdo), ("debug_value", debug_bus.clone()), ("test_status", vec![bist_done, bist_fail]), ("debug_select", tap_select.clone()), ("debug_override", vec![tap_override]), ("test_enable", vec![tap_test])]);
    }
    b.or_to(self_test_pin, tap_test, bist_enable);
    b.or_to(bist_done, Bit::Zero, self_test_done[0]);
    b.or_to(bist_fail, Bit::Zero, self_test_fail[0]);

    let load_valid = b.wire("load_valid", 1)[0];
    let load_target = b.wire("load_target", crate::soc::TARGET_BITS);
    let load_unit = b.wire("load_unit", 16);
    let load_addr = b.wire("load_addr", 32);
    let load_data = b.wire("load_data", 32 * crate::soc::LOAD_LANES as u32);
    let bus_write = b.wire("bus_write", 1)[0];
    let bus_read = b.wire("bus_read", 1)[0];
    let bus_address = b.wire("bus_address", PERIPHERAL_BITS);
    let bus_wdata = b.wire("bus_wdata", 32);
    let bus_rdata = b.wire("bus_rdata", 32);
    let internal_reset = b.wire("internal_reset", 1)[0];
    let cpu_pc = b.wire("cpu_pc", 32);
    b.group("Control system");
    b.inst("control", &control, &[
        ("reset", vec![internal_reset]),
        ("debug_pc", cpu_pc.clone()),
        ("cpu_hold", vec![cpu_hold]),
        ("host_rx_valid", host_rx_valid),
        ("host_rx_data", host_rx_data),
        ("host_tx_ready", host_tx_ready),
        ("host_rx_ready", host_rx_ready),
        ("host_tx_valid", host_tx_valid),
        ("host_tx_data", host_tx_data),
        ("halted", halted),
        ("image_resp_valid", image_resp_valid),
        ("image_resp_data", image_resp_data),
        ("image_req_valid", image_req_valid),
        ("image_req_addr", image_req_addr),
        ("load_valid", vec![load_valid]),
        ("load_target", load_target.clone()),
        ("load_unit", load_unit.clone()),
        ("load_addr", load_addr.clone()),
        ("load_data", load_data.clone()),
        ("booting", booting),
        ("peripheral_write", vec![bus_write]),
        ("peripheral_read", vec![bus_read]),
        ("peripheral_address", bus_address.clone()),
        ("peripheral_wdata", bus_wdata.clone()),
        ("peripheral_rdata", bus_rdata.clone()),
    ]);

    let die_halted = b.wire("die_halted", 1);
    let vector_out = b.wire("vector_out", activation as u32);
    let sampler_done = b.wire("sampler_done", 1);
    let sampler_token = b.wire("sampler_token", c.sampler.index_bits);
    let link_lanes = c.die.features.link_lanes() as u32;
    let (link_corrected, link_failed) = (b.wire("link_corrected", link_lanes), b.wire("link_failed", link_lanes));
    let (chain_corrected, chain_failed) = (b.wire("chain_corrected", 1)[0], b.wire("chain_failed", 1)[0]);
    let (chain_can_send, chain_has_data) = (b.wire("chain_can_send", 1), b.wire("chain_has_data", 1));
    let chain_word = b.wire("chain_word", 32);
    let (chain_send, chain_pop) = (b.wire("chain_send", 1)[0], b.wire("chain_pop", 1)[0]);
    let external_reset = reset;
    let reset = internal_reset;
    let mut die_pins: Vec<(&str, Vec<Bit>)> = vec![("halted", die_halted.clone()), ("vec_out", vector_out.clone())];
    let mut sampler_pins: Vec<(&str, Vec<Bit>)> = vec![("done", sampler_done.clone()), ("token", sampler_token.clone())];
    {
        let mut l = Logic { b: &mut b, next: 0 };
        l.b.group("Reset");
        // The reset pin changes whenever the board likes: two flip-flops in a row give a
        // value that may have caught it halfway a cycle to settle before anything reads it.
        let (first, synced) = (l.fresh(), l.fresh());
        l.connect_flops(&[external_reset], &[first]);
        l.connect_flops(&[first], &[synced]);
        let external_reset = synced;
        // The chip leaves reset only after a fixed delay, so that every part starts together.
        let stretch = l.fresh_bus(4);
        let released = l.fresh();
        let release_now = l.all(&stretch);
        let counting = l.not(released);
        let mut carry = counting;
        let next_stretch: Vec<Bit> = stretch.iter().map(|bit| { let out = l.xor(*bit, carry); carry = l.and(*bit, carry); out }).collect();
        let cleared_stretch = constant(0, 4);
        let held_stretch = l.mux_bus(external_reset, &next_stretch, &cleared_stretch);
        l.connect_flops(&held_stretch, &stretch);
        let next_released = {
            let stays = l.or(released, release_now);
            let free = l.not(external_reset);
            l.and(stays, free)
        };
        l.connect_flops(&[next_released], &[released]);
        let holding = l.not(released);
        let in_reset = l.or(external_reset, holding);
        l.b.or_to(in_reset, Bit::Zero, internal_reset);

        l.b.group("Die registers");
        let hit = |l: &mut Logic, register: u32| -> Bit {
            let selected = l.equals(&bus_address, register);
            l.and(selected, bus_write)
        };
        let stored = |l: &mut Logic, register: u32, width: usize| -> Vec<Bit> {
            let q = l.fresh_bus(width);
            let write = hit(l, register);
            let d = l.mux_bus(write, &q, &bus_wdata[..width]);
            l.connect_flops(&d, &q);
            q
        };

        let upper = l.any(&bus_address[5..]);
        let in_parameters = l.not(upper);
        let param_write = l.and(in_parameters, bus_write);
        let entry_write = hit(&mut l, REG_ENTRY);

        // The die stays parked from reset until the firmware first starts it.
        let parked = l.fresh();
        // The order that starts the die: the CPU's own, or the queue's.
        let entry_order = l.fresh();
        let keep_parked = {
            let held = l.or(parked, reset);
            let released = l.not(entry_order);
            l.and(held, released)
        };
        l.connect_flops(&[keep_parked], &[parked]);
        // The bus's orders reach the die a cycle later, from registers: otherwise a store
        // of the CPU would decode, start the die and address its memories in one cycle.
        let delayed = |l: &mut Logic, bits: &[Bit]| -> Vec<Bit> {
            let q = l.fresh_bus(bits.len());
            l.connect_flops(bits, &q);
            q
        };
        let start = l.fresh();
        let x = x_bits(&c.die) as usize;
        let (order, param_write, param_index, queue_empty, queue_room) = if c.die.features.queue_bits == 0 {
            l.b.or_to(entry_write, Bit::Zero, entry_order);
            let order = delayed(&mut l, &bus_wdata);
            let param_write = delayed(&mut l, &[param_write])[0];
            let param_index = delayed(&mut l, &bus_address[..x]);
            (order, param_write, param_index, Bit::One, constant(0, 1))
        } else {
            l.b.group("Command queue");
            // The CPU's writes to the parameters and the entry wait here; the die takes them
            // one a cycle whenever it is free, an entry starting it. The CPU can order the
            // next routine while the die runs this one.
            let bits = c.die.features.queue_bits as usize;
            let item: Vec<Bit> = [entry_write].iter().chain(&bus_address[..x]).chain(&bus_wdata).copied().collect();
            let push = l.or(param_write, entry_write);
            let (head, tail, count) = (l.fresh_bus(bits), l.fresh_bus(bits), l.fresh_bus(bits + 1));
            let slots: Vec<Vec<Bit>> = (0..1u32 << bits)
                .map(|k| {
                    let q = l.fresh_bus(item.len());
                    let here = l.equals(&tail, k);
                    let write = l.and(push, here);
                    let d = l.mux_bus(write, &q, &item);
                    l.connect_flops(&d, &q);
                    q
                })
                .collect();
            let empty = {
                let any = l.any(&count);
                l.not(any)
            };
            let die_free = {
                let waiting = l.not(start);
                l.and(die_halted[0], waiting)
            };
            let can_take = l.or(die_free, parked);
            let waiting_items = l.not(empty);
            let pop = l.and(waiting_items, can_take);
            let front = l.select(&head, &slots);
            let pop_entry = l.and(pop, front[0]);
            let not_entry = l.not(front[0]);
            let pop_param = l.and(pop, not_entry);
            l.b.or_to(pop_entry, Bit::Zero, entry_order);
            let step = |l: &mut Logic, bus: &[Bit], by: bool| -> Vec<Bit> {
                let addend = if by { constant(u32::MAX, bus.len()) } else { constant(1, bus.len()) };
                l.adder(bus, &addend, Bit::Zero).0
            };
            let cleared = |l: &mut Logic, bus: &[Bit]| -> Vec<Bit> {
                let zero = constant(0, bus.len());
                l.mux_bus(reset, bus, &zero)
            };
            let head_next = step(&mut l, &head, false);
            let head_d = l.mux_bus(pop, &head, &head_next);
            let head_d = cleared(&mut l, &head_d);
            l.connect_flops(&head_d, &head);
            let tail_next = step(&mut l, &tail, false);
            let tail_d = l.mux_bus(push, &tail, &tail_next);
            let tail_d = cleared(&mut l, &tail_d);
            l.connect_flops(&tail_d, &tail);
            let (more, fewer) = (step(&mut l, &count, false), step(&mut l, &count, true));
            let (no_pop, no_push) = (l.not(pop), l.not(push));
            let (push_only, pop_only) = (l.and(push, no_pop), l.and(pop, no_push));
            let count_d = l.mux_bus(push_only, &count, &more);
            let count_d = l.mux_bus(pop_only, &count_d, &fewer);
            let count_d = cleared(&mut l, &count_d);
            l.connect_flops(&count_d, &count);
            let room = {
                let inverted: Vec<Bit> = count.iter().map(|b| l.not(*b)).collect();
                l.adder(&constant(1 << bits, bits + 1), &inverted, Bit::One).0
            };
            let order = delayed(&mut l, &front[1 + x..]);
            let param_write = delayed(&mut l, &[pop_param])[0];
            let param_index = delayed(&mut l, &front[1..1 + x]);
            (order, param_write, param_index, empty, room)
        };
        let start_d = l.or(parked, entry_order);
        l.connect_flops(&[start_d], &[start]);

        let cpu_vector_address = stored(&mut l, REG_VECTOR_ADDRESS, vector_bits);
        let sample_base = stored(&mut l, REG_SAMPLE_BASE, vector_bits);
        let sample_count = stored(&mut l, REG_SAMPLE_COUNT, c.die.index_bits as usize + 1);
        let inverse = stored(&mut l, REG_SAMPLE_INVERSE, 16);
        let limit = stored(&mut l, REG_SAMPLE_LIMIT, limit_bits);
        let greedy = stored(&mut l, REG_SAMPLE_GREEDY, 1);
        let seed_write = hit(&mut l, REG_SAMPLE_SEED);
        let sample_start = hit(&mut l, REG_SAMPLE_START);

        l.b.group("Logit reading");
        // The logits go from the vector memory to the sampling unit without the CPU.
        let streaming = l.fresh();
        let count_bits = sample_count.len();
        let stream_index = l.fresh_bus(count_bits);
        let stepped = {
            let mut carry = streaming;
            stream_index.iter().map(|bit| { let out = l.xor(*bit, carry); carry = l.and(*bit, carry); out }).collect::<Vec<Bit>>()
        };
        let cleared = constant(0, count_bits);
        let next_index = l.mux_bus(sample_start, &stepped, &cleared);
        l.connect_flops(&next_index, &stream_index);
        let last_index = {
            let one = constant(1, count_bits);
            let plus_one = l.adder(&stream_index, &one, Bit::Zero).0;
            let same: Vec<Bit> = plus_one.iter().zip(&sample_count).map(|(a, b)| { let d = l.xor(*a, *b); l.not(d) }).collect();
            l.all(&same)
        };
        let stream_ends = l.and(streaming, last_index);
        let next_streaming = {
            let stay = {
                let more = l.not(stream_ends);
                l.and(streaming, more)
            };
            l.or(sample_start, stay)
        };
        l.connect_flops(&[next_streaming], &[streaming]);
        let (logit_valid, logit_last) = (l.fresh(), l.fresh());
        l.connect_flops(&[streaming, stream_ends], &[logit_valid, logit_last]);
        let wide_index: Vec<Bit> = stream_index.iter().copied().chain(std::iter::repeat_n(Bit::Zero, vector_bits - count_bits.min(vector_bits))).take(vector_bits).collect();
        let stream_address = l.adder(&sample_base, &wide_index, Bit::Zero).0;

        l.b.group("Loading into the die");
        let load_to = |l: &mut Logic, target: crate::image::Target| -> Bit {
            let selected = l.equals(&load_target, target as u32);
            l.and(selected, load_valid)
        };
        let load_code = load_to(&mut l, crate::image::Target::Microcode);
        let load_vec = load_to(&mut l, crate::image::Target::Vector);
        let load_fn = load_to(&mut l, crate::image::Target::Function);
        let table_write = load_to(&mut l, crate::image::Target::SamplerTable);
        // A microcode kept as a dictionary has its index memory, and each entry its number.
        let dictionary = c.die.features.dictionary_bits as usize;
        if dictionary > 0 {
            let load_index = load_to(&mut l, crate::image::Target::MicrocodeIndex);
            let shift = crate::die::ENTRY_INDEX_SHIFT as usize;
            die_pins.extend([("load_index", vec![load_index]), ("entry_index", order[shift..shift + dictionary].to_vec())]);
        }
        // Where each register starts in a compact vector memory, one entry a register.
        if c.die.features.vector_bits > 0 {
            let load_base = load_to(&mut l, crate::image::Target::RegisterBase);
            let entry = (c.die.features.vector_bits + crate::die::BASE_SHIFT_BITS) as usize;
            die_pins.extend([("load_base", vec![load_base]), ("base_addr", load_addr[..c.die.reg_bits as usize].to_vec()), ("base_data", load_data[..entry].to_vec())]);
        }
        // Weights fixed at manufacture need no network to reach their cells.
        if !c.die.cell.fixed_weights {
            let weight_load = load_to(&mut l, crate::image::Target::CellWeight);
            let scale_load = load_to(&mut l, crate::image::Target::CellScale);
            let cell_load = l.or(weight_load, scale_load);
            let load_cell: Vec<Bit> = (0..c.die.cells)
                .map(|i| {
                    let selected = l.equals(&load_unit, i);
                    l.and(selected, cell_load)
                })
                .collect();
            die_pins.extend([("load_cell", load_cell), ("load_scale", vec![scale_load]), ("cell_addr", load_addr[..c.die.cell.addr_bits as usize].to_vec())]);
        }
        let idle_address = l.mux_bus(streaming, &cpu_vector_address, &stream_address);
        let vector_address = l.mux_bus(load_vec, &idle_address, &load_addr[..vector_bits]);
        let (vector_write, vector_data) = if c.die.features.host_writes {
            // The CPU writes an element, or a register's exponent, at the address it set,
            // through the load port a cycle later: what the host sends, such as a picture,
            // goes in this way while the die is idle and no logits are on their way out.
            let strobe = hit(&mut l, REG_VECTOR_DATA);
            let strobe = delayed(&mut l, &[strobe])[0];
            let free = l.not(streaming);
            let cpu_write = l.and(strobe, free);
            let data = delayed(&mut l, &bus_wdata[..activation]);
            (l.or(load_vec, cpu_write), l.mux_bus(cpu_write, &load_data[..activation], &data))
        } else {
            (load_vec, load_data[..activation].to_vec())
        };

        die_pins.extend([
            ("start", vec![start]),
            ("entry", order[..c.die.microcode_bits as usize].to_vec()),
            ("param_write", vec![param_write]),
            ("param_index", param_index),
            ("param_data", order[..c.die.index_bits as usize].to_vec()),
            ("load_code", vec![load_code]),
            ("code_addr", load_addr[..c.die.microcode_bits as usize].to_vec()),
            ("code_data", load_data[..word_bits(&c.die) as usize].to_vec()),
            ("load_vec", vec![vector_write]),
            ("vec_addr", vector_address),
            ("vec_data", vector_data),
            ("load_fn", vec![load_fn]),
            ("fn_addr", load_addr[..function_address].to_vec()),
            ("cell_data", load_data[..cell_bits].to_vec()),
            ("link_tx_valid", link_tx_valid),
            ("link_tx_bits", link_tx_bits),
            ("link_tx_ready", link_tx_ready),
            ("link_rx_valid", link_rx_valid),
            ("link_rx_bits", link_rx_bits),
            ("link_rx_ready", link_rx_ready),
            ("link_tx_ack", link_tx_ack),
            ("link_rx_ack", link_rx_ack),
            ("link_rx_stall", link_rx_stall),
            ("link_corrected", link_corrected.clone()),
            ("link_failed", link_failed.clone()),
        ]);
        sampler_pins.extend([
            ("start", vec![sample_start]),
            ("greedy", greedy.clone()),
            ("seed_write", vec![seed_write]),
            ("seed", bus_wdata.clone()),
            ("inverse", inverse),
            ("limit", limit),
            ("logit_valid", vec![logit_valid]),
            ("logit_data", vector_out[..c.sampler.logit_bits as usize].to_vec()),
            ("logit_last", vec![logit_last]),
            ("table_write", vec![table_write]),
            ("table_address", load_addr[..TABLE_BITS as usize].to_vec()),
            ("table_data", load_data[..16].to_vec()),
        ]);

        l.b.group("Counters and debug");
        let not_halted = l.not(die_halted[0]);
        // Neither the cycle of the order nor the one it reaches the die in is the die's work.
        let idle_flag = l.any(&[parked, entry_order, start]);
        let not_idle = l.not(idle_flag);
        let die_busy = l.and(not_halted, not_idle);
        let counters: Vec<Vec<Bit>> = [Bit::One, die_busy, streaming]
            .into_iter()
            .map(|event| {
                let q = l.fresh_bus(32);
                let mut carry = event;
                let stepped: Vec<Bit> = q.iter().map(|bit| { let out = l.xor(*bit, carry); carry = l.and(*bit, carry); out }).collect();
                let cleared = constant(0, 32);
                let next = l.mux_bus(external_reset, &stepped, &cleared);
                l.connect_flops(&next, &q);
                q
            })
            .collect();
        // Every lane of the link and the command chain may report in the same cycle: each
        // counter adds them all.
        let link_counters: Vec<Vec<Bit>> = [(link_corrected, chain_corrected), (link_failed, chain_failed)]
            .into_iter()
            .map(|(lanes, chain)| {
                let q = l.fresh_bus(32);
                // Wide enough for every lane at once: the sum wraps at its operands' width.
                let events: Vec<Vec<Bit>> = lanes.iter().chain([&chain]).map(|bit| (0..8).map(|i| if i == 0 { *bit } else { Bit::Zero }).collect()).collect();
                let count = l.sum_many(events);
                let addend: Vec<Bit> = (0..32).map(|i| count.get(i).copied().unwrap_or(Bit::Zero)).collect();
                let stepped = l.adder(&q, &addend, Bit::Zero).0;
                let cleared = constant(0, 32);
                let next = l.mux_bus(external_reset, &stepped, &cleared);
                l.connect_flops(&next, &q);
                q
            })
            .collect();
        let mut leaves: Vec<Vec<Bit>> = counters.into_iter().chain([cpu_pc.clone()]).chain(link_counters).collect();
        leaves.resize(8, constant(0, 32));
        let chosen_view = l.mux_bus(tap_override, &debug_select, &tap_select);
        let shown = l.select(&chosen_view, &leaves);
        for ((from, to), bus) in shown.iter().zip(&debug_value).zip(&debug_bus) {
            l.b.or_to(*from, Bit::Zero, *to);
            l.b.or_to(*from, Bit::Zero, *bus);
        }

        l.b.group("Control chain");
        let send_word = hit(&mut l, REG_CHAIN_DATA);
        l.b.or_to(send_word, Bit::Zero, chain_send);
        let chain_read = {
            let selected = l.equals(&bus_address, REG_CHAIN_DATA);
            l.and(selected, bus_read)
        };
        l.b.or_to(chain_read, Bit::Zero, chain_pop);
        // The bus answers a cycle late, by when the queue has moved on.
        let chain_held = l.fresh_bus(32);
        let next_chain = l.mux_bus(chain_read, &chain_held, &chain_word);
        l.connect_flops(&next_chain, &chain_held);

        l.b.group("Register read");
        let selected = l.fresh_bus(PERIPHERAL_BITS as usize);
        let held = l.mux_bus(bus_read, &selected, &bus_address);
        l.connect_flops(&held, &selected);
        let widen = |bits: &[Bit]| -> Vec<Bit> { bits.iter().copied().chain(std::iter::repeat_n(Bit::Zero, 32 - bits.len())).collect() };
        let sign = vector_out[activation - 1];
        let vector_word: Vec<Bit> = vector_out.iter().copied().chain(std::iter::repeat_n(sign, 32 - activation)).collect();
        let mut value = constant(0, 32);
        // A die whose order is on its way already counts as busy.
        let free = {
            let waiting = l.not(start);
            let halted = l.and(die_halted[0], waiting);
            l.and(halted, queue_empty)
        };
        for (register, word) in [(REG_STATUS, widen(&[free])), (REG_QUEUE_ROOM, widen(&queue_room)), (REG_VECTOR_DATA, vector_word), (REG_SAMPLE_DONE, widen(&sampler_done)), (REG_SAMPLE_TOKEN, widen(&sampler_token)), (REG_FOLLOWER, widen(&follower)), (REG_CHAIN_DATA, chain_held.clone()), (REG_CHAIN_STATUS, widen(&[chain_has_data[0], chain_can_send[0]]))] {
            let chosen = l.equals(&selected, register);
            value = l.mux_bus(chosen, &value, &word);
        }
        for (from, to) in value.iter().zip(&bus_rdata) {
            l.b.or_to(*from, Bit::Zero, *to);
        }
    }
    b.group("Control chain");
    if alone {
        // Nothing ever arrives and nothing can leave, so the firmware finds itself alone.
        let held: Vec<Bit> = chain_can_send.iter().chain(&chain_has_data).chain(&chain_word).chain(&[chain_corrected, chain_failed]).chain(&chain_tx_valid).chain(&chain_tx_bits).chain(&chain_rx_ready).chain(&chain_rx_ack).copied().collect();
        for bit in held {
            b.or_to(Bit::Zero, Bit::Zero, bit);
        }
    } else {
        let chain = gen.link_endpoint(32);
        b.inst("chain", &chain, &[
        ("tx_send", vec![chain_send]),
        ("tx_data", bus_wdata.clone()),
        ("tx_can_send", chain_can_send),
        ("pin_tx_valid", chain_tx_valid),
        ("pin_tx_bits", chain_tx_bits),
        ("pin_tx_ready", chain_tx_ready),
        ("pin_tx_ack", chain_tx_ack),
        ("pin_rx_ack", chain_rx_ack),
        ("pin_rx_stall", chain_rx_stall),
        ("core_stall", vec![Bit::Zero]),
        ("pin_rx_valid", chain_rx_valid),
        ("pin_rx_bits", chain_rx_bits),
        ("pin_rx_ready", chain_rx_ready),
        ("rx_pop", vec![chain_pop]),
        ("rx_data", chain_word),
        ("rx_has_data", chain_has_data),
        ("corrected", vec![chain_corrected]),
        ("failed", vec![chain_failed]),
        ]);
    }
    b.group("Inference die");
    die_pins.extend(extra_link.iter().map(|(name, bits)| (name.as_str(), bits.clone())));
    b.inst("die", &die, &die_pins);
    b.group("Sampling unit");
    b.inst("sampler", &sample, &sampler_pins);
    gen.add(b.finish())
}

pub fn chip_design(config: ChipConfig) -> Design {
    let mut gen = Gen { modules: BTreeMap::new() };
    let top = chip(&mut gen, config, false);
    crate::self_test::insert(Design { top, modules: gen.modules })
}

/// The chip for manufacturing: `chip_design` with the factory test's scan chains
/// (`scan::insert_scan_around`). The die's cells are tested side by side from
/// `scan_cells_in`; one chain from `scan_in` to `scan_out` goes through everything else,
/// the die's sequencer and vector units, the CPU, the memories' shadows and the sampler.
/// With the test pins at zero it works as `chip_design` does. FPGA builds go without it.
pub fn chip_design_for_test(config: ChipConfig) -> Result<crate::scan::ParallelScan, String> {
    crate::scan::insert_scan_around(chip_design(config), "die", &crate::die::cell_groups(config.die))
}

/// The chip for one FPGA board on its own: the same firmware runs the same way, without
/// the JTAG port, the memories' march tests or the chain to other chips, which the board
/// does not need and which take logic the FPGA can use for the model.
pub fn chip_design_alone(config: ChipConfig) -> Design {
    let mut gen = Gen { modules: BTreeMap::new() };
    let top = chip(&mut gen, config, true);
    let mut design = Design { top, modules: gen.modules };
    crate::optimize::trim_top(&mut design);
    design
}

/// Runs the die's routines at software speed, on the same registers as the gates.
pub trait DieEngine {
    fn run(&mut self, routine: usize, token: u32, position: u32);
    /// How long the die is busy with that routine.
    fn cycles(&self, routine: usize, token: u32, position: u32) -> u64;
    /// Reads a vector register element or exponent, by its full address.
    fn read(&mut self, address: u32) -> i64;
    /// Writes one, as the CPU does while the die is idle.
    fn write(&mut self, _address: u32, _value: i64) {}
}

/// Orders waiting in a command queue: the cycle each came in, whether it starts a routine,
/// the register and the value.
type Orders = std::collections::VecDeque<(u64, bool, u32, u32)>;

/// The chip's peripherals at instruction level, with the same timing as the gates.
/// A status read answers one cycle after its address, so the times here are one early.
pub struct ChipModel<E> {
    pub engine: E,
    pub code: Microprogram,
    pub sampler: SamplerModel,
    /// Cycles the die has been busy, as the chip's own counter would read.
    pub die_busy: u64,
    /// For each routine of the microprogram, how many times it ran and its cycles.
    pub routines: Vec<(u64, u64)>,
    /// Tokens the sampling unit has picked, and the cycles it took.
    pub samples: (u64, u64),
    parameters: [u32; 32],
    /// What the firmware reports of the draft head, by how far ahead it guessed.
    pub draft_hits: [u32; 4],
    pub draft_tries: [u32; 4],
    vector_address: u32,
    sample_base: u32,
    sample_count: u32,
    inverse: u32,
    limit: u32,
    greedy: bool,
    die_free_at: u64,
    sampler_done_at: Option<u64>,
    sampler_token: u32,
    /// The die's command queue, if it has one: how many orders it holds, and the orders
    /// waiting, each with the cycle it came in, whether it starts a routine, the register
    /// and the value.
    queue: Option<(usize, Orders)>,
    last_pop: u64,
}

impl<E: DieEngine> ChipModel<E> {
    pub fn new(engine: E, code: Microprogram, sampler: SamplerModel) -> Self {
        Self { engine, routines: vec![(0, 0); code.entries.len()], code, sampler, die_busy: 0, samples: (0, 0), parameters: [0; 32], draft_hits: [0; 4], draft_tries: [0; 4], vector_address: 0, sample_base: 0, sample_count: 0, inverse: 0, limit: 0, greedy: false, die_free_at: 0, sampler_done_at: None, sampler_token: 0, queue: None, last_pop: 0 }
    }

    /// The same chip with a command queue of `1 << bits` orders before the die, as
    /// `DieFeatures::queue_bits` builds it.
    pub fn with_queue(mut self, bits: u32) -> Self {
        self.queue = (bits > 0).then(|| (1 << bits, std::collections::VecDeque::new()));
        self
    }

    /// Starts routine `value` (an entry, as the firmware writes it) in `cycle`.
    fn start(&mut self, value: u32, cycle: u64) {
        // Above the PC may come the entry's number in a microcode dictionary.
        let pc = value & ((1 << crate::die::ENTRY_INDEX_SHIFT) - 1);
        let routine = self.code.entries.iter().position(|e| *e == value || *e == pc).expect("unknown routine");
        let (token, position) = self.token_and_position();
        self.engine.run(routine, token, position);
        let cycles = self.engine.cycles(routine, token, position);
        self.die_busy += cycles;
        self.routines[routine].0 += 1;
        self.routines[routine].1 += cycles;
        // The order reaches the die a cycle after the store.
        self.die_free_at = cycle + 1 + cycles;
    }

    /// The orders the queue gives the die up to `now`: one a cycle, from the cycle after
    /// each came in, once the die is free.
    fn advance(&mut self, now: u64) {
        while let Some(&(came, entry, register, value)) = self.queue.as_ref().and_then(|(_, q)| q.front()) {
            let at = (came + 1).max(self.last_pop + 1).max(self.die_free_at + 1);
            if at > now {
                break;
            }
            self.queue.as_mut().expect("the queue").1.pop_front();
            self.last_pop = at;
            if entry {
                self.start(value, at);
            } else {
                self.parameters[register as usize] = value;
            }
        }
    }

    /// Recovers the token and position from the parameter registers.
    fn token_and_position(&self) -> (u32, u32) {
        let (mut token, mut position) = (0, 0);
        for p in self.code.parameters.iter().filter(|p| p.window == 0) {
            let value = self.parameters[p.register as usize].wrapping_sub(p.constant);
            token = value.checked_div(p.token).unwrap_or(token);
            position = value.checked_div(p.position).unwrap_or(position);
        }
        (token, position)
    }
}

impl<E: DieEngine> Peripheral for ChipModel<E> {
    fn read(&mut self, register: u32, cycle: u64) -> u32 {
        self.advance(cycle);
        let waiting = self.queue.as_ref().map_or(0, |(_, q)| q.len());
        match register {
            REG_STATUS => (cycle >= self.die_free_at && waiting == 0) as u32,
            REG_QUEUE_ROOM => self.queue.as_ref().map_or(0, |(depth, q)| depth - q.len()) as u32,
            REG_VECTOR_DATA => self.engine.read(self.vector_address) as u32,
            REG_SAMPLE_DONE => self.sampler_done_at.is_some_and(|at| cycle >= at) as u32,
            REG_SAMPLE_TOKEN => self.sampler_token,
            _ => 0,
        }
    }

    fn write(&mut self, register: u32, value: u32, cycle: u64) {
        self.advance(cycle);
        if let Some((depth, queue)) = &mut self.queue {
            if register < 32 || register == REG_ENTRY {
                assert!(queue.len() < *depth, "the die's command queue overflows");
                queue.push_back((cycle, register == REG_ENTRY, register, value));
                return;
            }
        }
        match register {
            0..=31 => self.parameters[register as usize] = value,
            REG_ENTRY => self.start(value, cycle),
            REG_VECTOR_ADDRESS => self.vector_address = value,
            REG_VECTOR_DATA => self.engine.write(self.vector_address, value as i32 as i64),
            REG_SAMPLE_BASE => self.sample_base = value,
            REG_SAMPLE_COUNT => self.sample_count = value,
            REG_SAMPLE_INVERSE => self.inverse = value & 0xFFFF,
            REG_SAMPLE_LIMIT => self.limit = value,
            REG_SAMPLE_GREEDY => self.greedy = value & 1 == 1,
            REG_SAMPLE_SEED => self.sampler.rng = value,
            REG_DRAFT_RESULT => {
                let depth = (value >> 1) as usize & 3;
                self.draft_tries[depth] += 1;
                self.draft_hits[depth] += value & 1;
            }
            REG_SAMPLE_START => {
                let logits: Vec<i64> = (0..self.sample_count).map(|i| self.engine.read(self.sample_base + i)).collect();
                let sampled = self.sampler.sample(&logits, self.inverse, self.limit as usize, self.greedy);
                self.sampler_token = sampled.token;
                self.sampler_done_at = Some(cycle + sampled.stream_cycles + sampled.finish_cycles);
                self.samples.0 += 1;
                self.samples.1 += sampled.stream_cycles + sampled.finish_cycles;
            }
            _ => {}
        }
    }

    fn load(&mut self, write: &LoadWrite) {
        if write.target == crate::image::Target::SamplerTable as u32 {
            self.sampler.table[write.address as usize] = write.words[0] & 0xFFFF;
        } else if write.target == crate::image::Target::Vector as u32 {
            self.engine.write(write.address, write.words[0] as i64);
        }
    }
}

/// Where the firmware finds what it needs to drive this chip, in the description words.
pub const INFO_ENTRY_TOKEN: usize = 0;
pub const INFO_ENTRY_LOGITS: usize = 1;
pub const INFO_LOGITS: usize = 2;
pub const INFO_LOGITS_EXPONENT: usize = 3;
pub const INFO_VOCABULARY: usize = 4;
pub const INFO_MAX_NEW: usize = 5;
pub const INFO_GREEDY: usize = 6;
pub const INFO_LIMIT: usize = 7;
pub const INFO_TEMPERATURE: usize = 8;
pub const INFO_SEED: usize = 9;
pub const INFO_ACTIVATION_BITS: usize = 10;
pub const INFO_PARAMETER_COUNT: usize = 11;
pub const INFO_RAW_TOKENS: usize = 12;
pub const INFO_STOP_COUNT: usize = 13;
pub const INFO_STOP_TABLE: usize = 14;
pub const INFO_DRAFT_DEPTH: usize = 15;
pub const INFO_DRAFT_LOGITS: usize = 16;
pub const INFO_ENTRIES: usize = 17;
pub const INFO_THINK_OPEN: usize = 18;
pub const INFO_THINK_CLOSE: usize = 19;
pub const INFO_THINK_BUDGET: usize = 20;
/// A lookup table the firmware brings in from the image one row at a time, before each
/// routine that reads a token's row: where it is in the image, the register its row goes
/// to, and how many elements a row has. With no such table the count is zero.
pub const INFO_TABLE_SOURCE: usize = 21;
pub const INFO_TABLE_ADDRESS: usize = 22;
pub const INFO_TABLE_COLS: usize = 23;
/// Where the description of the chip's image encoder starts, if it carries one; zero if not.
pub const INFO_VISION: usize = 24;
pub const INFO_SHOWO: usize = 25;
/// The rotation table, when it stays in the image: the firmware brings the row of each
/// token's position before each routine that rotates, the same way as the lookup table's.
pub const INFO_ROTATION_SOURCE: usize = 26;
pub const INFO_ROTATION_ADDRESS: usize = 27;
pub const INFO_ROTATION_COLS: usize = 28;
/// The word of data memory where the schedule of a die run by routines starts (see
/// `schedule`); zero if it has none.
pub const INFO_SCHEDULE: usize = 29;
/// How many orders the die's command queue holds; zero without one.
pub const INFO_QUEUE: usize = 30;
pub const INFO_PARAMETERS: usize = 31;

/// What the firmware needs to run the image encoder a chip carries, in the order of its
/// words: see `VisionInfo`.
pub const VISION_WORDS: usize = 16;

/// The image encoder a chip carries, for the firmware. Addresses are of vector memory.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VisionInfo {
    /// The most patches it takes.
    pub patches: u32,
    /// The placeholder token whose place the image's tokens take.
    pub image_token: u32,
    /// Patches per side of a token.
    pub merge: u32,
    pub blocks: u32,
    /// The patch's pixels, its learned position and the row of the encoder's rotation
    /// table for patch 0: each vector's first element and its exponent's address, and size.
    pub pixels: (u32, u32, u32),
    pub position: (u32, u32, u32),
    pub patch_rope: (u32, u32),
    /// The text's rotation table, row 0, and half its rotated elements: zero when the
    /// firmware need not rotate image tokens, as a die run by software.
    pub rope: u32,
    pub rope_half: u32,
    /// How many of the text's frequencies follow the image's row and its column (M-RoPE).
    pub mrope_rows: u32,
    pub mrope_cols: u32,
}

impl VisionInfo {
    fn words(&self) -> [u32; VISION_WORDS] {
        [self.patches, self.image_token, self.merge, self.blocks, self.pixels.0, self.pixels.1, self.pixels.2, self.position.0, self.position.1, self.position.2, self.patch_rope.0, self.patch_rope.1, self.rope, self.rope_half, self.mrope_rows, self.mrope_cols]
    }
}

/// What the firmware needs to run Show-o's pass over an image: its rows (start, codes and
/// end), its layers, the codes' scores it reads back, and the text's rotation table, whose
/// rows it moves to each token's real position.
#[derive(Clone, Copy, Debug, Default)]
pub struct ShowOInfo {
    pub rows: u32,
    pub layers: u32,
    pub codebook: u32,
    /// The scores' first element and their exponent's address.
    pub scores: (u32, u32),
    /// The rotation table's row 0, and half its rotated elements.
    pub rope: u32,
    pub rope_half: u32,
    /// Whether the die guides the scores: after the scores' routine, one that keeps them
    /// with the text and one that combines them with those without it.
    pub guided: u32,
}

/// Adds Show-o's description to a chip's.
pub fn add_showo(info: &mut Vec<u32>, showo: &ShowOInfo) {
    info[INFO_SHOWO] = info.len() as u32;
    info.extend([showo.rows, showo.layers, showo.codebook, showo.scores.0, showo.scores.1, showo.rope, showo.rope_half, showo.guided]);
}

/// Adds a schedule of routines to a chip's description: the firmware follows it in place of
/// reading a prompt. A schedule can be longer than the description has room for, so it goes
/// where a tokenizer's tables would (`TOKENIZER_WORD`): a chip run by a schedule exchanges
/// words, not text, and has none. Gives the words to place there (`build_image`'s
/// `tokenizer`).
pub fn add_schedule(info: &mut [u32], records: &[crate::schedule::Record]) -> Vec<u32> {
    info[INFO_SCHEDULE] = crate::soc::TOKENIZER_WORD as u32;
    crate::schedule::words(records)
}

/// Adds the image encoder's description to a chip's.
pub fn add_vision(info: &mut Vec<u32>, vision: &VisionInfo) {
    info[INFO_VISION] = info.len() as u32;
    info.extend(vision.words());
}
pub const REG_DRAFT_RESULT: u32 = 48;
pub const INFO_PARAMETER_SLOTS: usize = 16;

/// Settings for generating text.
#[derive(Clone, Debug)]
pub struct GenerationSetup {
    pub max_new: u32,
    pub greedy: bool,
    pub limit: u32,
    /// Temperature times 256.
    pub temperature: u32,
    pub seed: u32,
    /// The host exchanges 32-bit token numbers instead of text.
    pub raw_tokens: bool,
    /// Tokens that end the generation, as many as the model has.
    pub stop: Vec<u32>,
    /// How many tokens ahead the draft head guesses; zero leaves it unused.
    pub draft_depth: u32,
    /// Where the draft head leaves its logits.
    pub draft_logits: u32,
    /// The tokens that open and close the model's reasoning, if it has them.
    pub think: Option<(u32, u32)>,
    /// The most tokens the model may reason before the chip closes the reasoning itself;
    /// `u32::MAX` leaves it unbounded.
    pub think_budget: u32,
}

/// The chip description the firmware reads from its data memory.
pub fn chip_info(c: &DieConfig, code: &Microprogram, logits_register: u32, vocabulary: u32, setup: &GenerationSetup) -> Vec<u32> {
    let mut info = vec![0u32; INFO_PARAMETERS];
    info[INFO_ENTRY_TOKEN] = code.entries[0];
    info[INFO_ENTRY_LOGITS] = code.entries[1];
    info[INFO_LOGITS] = logits_register << c.index_bits;
    info[INFO_LOGITS_EXPONENT] = exponent_register(c) << c.index_bits | logits_register;
    info[INFO_VOCABULARY] = vocabulary;
    info[INFO_MAX_NEW] = setup.max_new;
    info[INFO_GREEDY] = setup.greedy as u32;
    info[INFO_LIMIT] = setup.limit;
    info[INFO_TEMPERATURE] = setup.temperature;
    info[INFO_SEED] = setup.seed;
    info[INFO_ACTIVATION_BITS] = c.cell.activation_bits;
    info[INFO_QUEUE] = if c.features.queue_bits > 0 { 1 << c.features.queue_bits } else { 0 };
    info[INFO_RAW_TOKENS] = setup.raw_tokens as u32;
    info[INFO_STOP_COUNT] = setup.stop.len() as u32;
    // A model without reasoning tokens never opens it: no token is u32::MAX.
    let (open, close) = setup.think.unwrap_or((u32::MAX, u32::MAX));
    (info[INFO_THINK_OPEN], info[INFO_THINK_CLOSE], info[INFO_THINK_BUDGET]) = (open, close, setup.think_budget);
    assert!(code.parameters.len() <= INFO_PARAMETER_SLOTS, "too many parameters");
    info[INFO_PARAMETER_COUNT] = code.parameters.len() as u32;
    for p in &code.parameters {
        info.extend([p.register, p.token, p.position, p.constant, p.window_length, p.window_slot, p.window]);
    }
    info[INFO_STOP_TABLE] = info.len() as u32;
    info.extend_from_slice(&setup.stop);
    info[INFO_DRAFT_DEPTH] = setup.draft_depth;
    info[INFO_DRAFT_LOGITS] = setup.draft_logits;
    info[INFO_ENTRIES] = info.len() as u32;
    if c.features.dictionary_bits > 0 {
        assert!(c.microcode_bits <= crate::die::ENTRY_INDEX_SHIFT, "with a dictionary, the microcode cannot go past 2^16 words");
        let (_, index) = crate::die::code_dictionary(code, c);
        info.extend(code.entries.iter().map(|pc| pc | index.get(*pc as usize).copied().unwrap_or(0) << crate::die::ENTRY_INDEX_SHIFT));
    } else {
        info.extend_from_slice(&code.entries);
    }
    info
}
