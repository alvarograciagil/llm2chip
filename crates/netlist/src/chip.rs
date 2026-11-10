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

/// What the debug port shows for each value of `debug_select`.
pub const DEBUG_CYCLES: u64 = 0;
pub const DEBUG_DIE_BUSY: u64 = 1;
pub const DEBUG_SAMPLER_BUSY: u64 = 2;
pub const DEBUG_CPU_PC: u64 = 3;
pub const DEBUG_LINK_CORRECTED: u64 = 4;
pub const DEBUG_LINK_FAILED: u64 = 5;
/// Cycles the chip keeps itself in reset after the external signal is released.
pub const RESET_STRETCH: u64 = 15;

#[derive(Clone, Copy, Debug, Serialize)]
pub struct ChipConfig {
    pub soc: SocConfig,
    pub die: DieConfig,
    pub sampler: SamplerConfig,
}

fn chip(gen: &mut Gen, c: ChipConfig) -> String {
    let control = soc(gen, c.soc);
    let die = gen.die(&c.die);
    let sample = sampler(gen, c.sampler);
    let activation = c.die.cell.activation_bits as usize;
    let vector_bits = (c.die.reg_bits + c.die.index_bits) as usize;
    let cell_bits = c.die.scale_bits.max(c.die.cell.weight_bits).max(c.die.cell.activation_bits) as usize;
    let function_address = (c.die.function_bits + c.die.function_select_bits) as usize;
    let limit_bits = (c.sampler.entries as usize + 1).next_power_of_two().trailing_zeros() as usize;

    assert_eq!(c.sampler.lanes, 1, "el die reducido solo entrega un logit por ciclo");
    let mut b = Builder::new("llm2chip_top", "Chip completo", "CPU de control, memorias, controlador de carga, die de inferencia y unidad de muestreo, con el puerto de host y el de imagen como únicos contactos con el exterior.");
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
    let tap = gen.jtag_tap();
    let (tap_select, tap_override, tap_test) = (b.wire("tap_select", 3), b.wire("tap_override", 1)[0], b.wire("tap_test", 1)[0]);
    let debug_bus = b.wire("debug_bus", 32);
    b.group("Puerto de pruebas");
    b.inst("jtag", &tap, &[("tck", jtag_tck), ("tms", jtag_tms), ("tdi", jtag_tdi), ("tdo", jtag_tdo), ("debug_value", debug_bus.clone()), ("test_status", vec![bist_done, bist_fail]), ("debug_select", tap_select.clone()), ("debug_override", vec![tap_override]), ("test_enable", vec![tap_test])]);
    b.or_to(self_test_pin, tap_test, bist_enable);
    b.or_to(bist_done, Bit::Zero, self_test_done[0]);
    b.or_to(bist_fail, Bit::Zero, self_test_fail[0]);

    let load_valid = b.wire("load_valid", 1)[0];
    let load_target = b.wire("load_target", 3);
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
    b.group("Sistema de control");
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
    let (link_corrected, link_failed) = (b.wire("link_corrected", 1)[0], b.wire("link_failed", 1)[0]);
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

        l.b.group("Registros del die");
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
        let keep_parked = {
            let held = l.or(parked, reset);
            let released = l.not(entry_write);
            l.and(held, released)
        };
        l.connect_flops(&[keep_parked], &[parked]);
        let start = l.or(parked, entry_write);

        let cpu_vector_address = stored(&mut l, REG_VECTOR_ADDRESS, vector_bits);
        let sample_base = stored(&mut l, REG_SAMPLE_BASE, vector_bits);
        let sample_count = stored(&mut l, REG_SAMPLE_COUNT, c.die.index_bits as usize + 1);
        let inverse = stored(&mut l, REG_SAMPLE_INVERSE, 16);
        let limit = stored(&mut l, REG_SAMPLE_LIMIT, limit_bits);
        let greedy = stored(&mut l, REG_SAMPLE_GREEDY, 1);
        let seed_write = hit(&mut l, REG_SAMPLE_SEED);
        let sample_start = hit(&mut l, REG_SAMPLE_START);

        l.b.group("Lectura de logits");
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

        l.b.group("Carga en el die");
        let load_to = |l: &mut Logic, target: crate::image::Target| -> Bit {
            let selected = l.equals(&load_target, target as u32);
            l.and(selected, load_valid)
        };
        let load_code = load_to(&mut l, crate::image::Target::Microcode);
        let load_vec = load_to(&mut l, crate::image::Target::Vector);
        let load_fn = load_to(&mut l, crate::image::Target::Function);
        let table_write = load_to(&mut l, crate::image::Target::SamplerTable);
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

        die_pins.extend([
            ("start", vec![start]),
            ("entry", bus_wdata[..c.die.microcode_bits as usize].to_vec()),
            ("param_write", vec![param_write]),
            ("param_index", bus_address[..x_bits(&c.die) as usize].to_vec()),
            ("param_data", bus_wdata[..c.die.index_bits as usize].to_vec()),
            ("load_code", vec![load_code]),
            ("code_addr", load_addr[..c.die.microcode_bits as usize].to_vec()),
            ("code_data", load_data[..word_bits(&c.die) as usize].to_vec()),
            ("load_vec", vec![load_vec]),
            ("vec_addr", vector_address),
            ("vec_data", load_data[..activation].to_vec()),
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
            ("link_corrected", vec![link_corrected]),
            ("link_failed", vec![link_failed]),
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

        l.b.group("Contadores y depuración");
        let not_halted = l.not(die_halted[0]);
        let idle_flag = l.or(parked, entry_write);
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
        let any_corrected = l.or(link_corrected, chain_corrected);
        let any_failed = l.or(link_failed, chain_failed);
        let link_counters: Vec<Vec<Bit>> = [any_corrected, any_failed]
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
        let mut leaves: Vec<Vec<Bit>> = counters.into_iter().chain([cpu_pc.clone()]).chain(link_counters).collect();
        leaves.resize(8, constant(0, 32));
        let chosen_view = l.mux_bus(tap_override, &debug_select, &tap_select);
        let shown = l.select(&chosen_view, &leaves);
        for ((from, to), bus) in shown.iter().zip(&debug_value).zip(&debug_bus) {
            l.b.or_to(*from, Bit::Zero, *to);
            l.b.or_to(*from, Bit::Zero, *bus);
        }

        l.b.group("Cadena de control");
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

        l.b.group("Lectura de registros");
        let selected = l.fresh_bus(PERIPHERAL_BITS as usize);
        let held = l.mux_bus(bus_read, &selected, &bus_address);
        l.connect_flops(&held, &selected);
        let widen = |bits: &[Bit]| -> Vec<Bit> { bits.iter().copied().chain(std::iter::repeat_n(Bit::Zero, 32 - bits.len())).collect() };
        let sign = vector_out[activation - 1];
        let vector_word: Vec<Bit> = vector_out.iter().copied().chain(std::iter::repeat_n(sign, 32 - activation)).collect();
        let mut value = constant(0, 32);
        for (register, word) in [(REG_STATUS, widen(&die_halted)), (REG_VECTOR_DATA, vector_word), (REG_SAMPLE_DONE, widen(&sampler_done)), (REG_SAMPLE_TOKEN, widen(&sampler_token)), (REG_FOLLOWER, widen(&follower)), (REG_CHAIN_DATA, chain_held.clone()), (REG_CHAIN_STATUS, widen(&[chain_has_data[0], chain_can_send[0]]))] {
            let chosen = l.equals(&selected, register);
            value = l.mux_bus(chosen, &value, &word);
        }
        for (from, to) in value.iter().zip(&bus_rdata) {
            l.b.or_to(*from, Bit::Zero, *to);
        }
    }
    b.group("Cadena de control");
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
    b.group("Die de inferencia");
    b.inst("die", &die, &die_pins);
    b.group("Unidad de muestreo");
    b.inst("sampler", &sample, &sampler_pins);
    gen.add(b.finish())
}

pub fn chip_design(config: ChipConfig) -> Design {
    let mut gen = Gen { modules: BTreeMap::new() };
    let top = chip(&mut gen, config);
    crate::self_test::insert(Design { top, modules: gen.modules })
}

/// Runs the die's routines at software speed, on the same registers as the gates.
pub trait DieEngine {
    fn run(&mut self, routine: usize, token: u32, position: u32);
    /// How long the die is busy with that routine.
    fn cycles(&self, routine: usize, token: u32, position: u32) -> u64;
    /// Reads a vector register element or exponent, by its full address.
    fn read(&mut self, address: u32) -> i64;
}

/// The chip's peripherals at instruction level, with the same timing as the gates.
/// A status read answers one cycle after its address, so the times here are one early.
pub struct ChipModel<E> {
    pub engine: E,
    pub code: Microprogram,
    pub sampler: SamplerModel,
    /// Cycles the die has been busy, as the chip's own counter would read.
    pub die_busy: u64,
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
}

impl<E: DieEngine> ChipModel<E> {
    pub fn new(engine: E, code: Microprogram, sampler: SamplerModel) -> Self {
        Self { engine, code, sampler, die_busy: 0, parameters: [0; 32], draft_hits: [0; 4], draft_tries: [0; 4], vector_address: 0, sample_base: 0, sample_count: 0, inverse: 0, limit: 0, greedy: false, die_free_at: 0, sampler_done_at: None, sampler_token: 0 }
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
        match register {
            REG_STATUS => (cycle >= self.die_free_at) as u32,
            REG_VECTOR_DATA => self.engine.read(self.vector_address) as u32,
            REG_SAMPLE_DONE => self.sampler_done_at.is_some_and(|at| cycle >= at) as u32,
            REG_SAMPLE_TOKEN => self.sampler_token,
            _ => 0,
        }
    }

    fn write(&mut self, register: u32, value: u32, cycle: u64) {
        match register {
            0..=31 => self.parameters[register as usize] = value,
            REG_ENTRY => {
                let routine = self.code.entries.iter().position(|e| *e == value).expect("rutina desconocida");
                let (token, position) = self.token_and_position();
                self.engine.run(routine, token, position);
                let cycles = self.engine.cycles(routine, token, position);
                self.die_busy += cycles;
                self.die_free_at = cycle + cycles;
            }
            REG_VECTOR_ADDRESS => self.vector_address = value,
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
            }
            _ => {}
        }
    }

    fn load(&mut self, write: &LoadWrite) {
        if write.target == crate::image::Target::SamplerTable as u32 {
            self.sampler.table[write.address as usize] = write.words[0] & 0xFFFF;
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
pub const INFO_PARAMETERS: usize = 21;
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
    info[INFO_RAW_TOKENS] = setup.raw_tokens as u32;
    info[INFO_STOP_COUNT] = setup.stop.len() as u32;
    // A model without reasoning tokens never opens it: no token is u32::MAX.
    let (open, close) = setup.think.unwrap_or((u32::MAX, u32::MAX));
    (info[INFO_THINK_OPEN], info[INFO_THINK_CLOSE], info[INFO_THINK_BUDGET]) = (open, close, setup.think_budget);
    assert!(code.parameters.len() <= INFO_PARAMETER_SLOTS, "demasiados parámetros");
    info[INFO_PARAMETER_COUNT] = code.parameters.len() as u32;
    for p in &code.parameters {
        info.extend([p.register, p.token, p.position, p.constant, p.window_length, p.window_slot, p.window]);
    }
    info[INFO_STOP_TABLE] = info.len() as u32;
    info.extend_from_slice(&setup.stop);
    info[INFO_DRAFT_DEPTH] = setup.draft_depth;
    info[INFO_DRAFT_LOGITS] = setup.draft_logits;
    info[INFO_ENTRIES] = info.len() as u32;
    info.extend_from_slice(&code.entries);
    info
}
