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
    b.group("Sistema de control");
    b.inst("control", &control, &[
        ("reset", vec![reset]),
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
    let mut die_pins: Vec<(&str, Vec<Bit>)> = vec![("halted", die_halted.clone()), ("vec_out", vector_out.clone())];
    let mut sampler_pins: Vec<(&str, Vec<Bit>)> = vec![("done", sampler_done.clone()), ("token", sampler_token.clone())];
    {
        let mut l = Logic { b: &mut b, next: 0 };
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
            ("link_in", vec![Bit::Zero; activation]),
        ]);
        let link_out = l.fresh_bus(activation);
        die_pins.push(("link_out", link_out));
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

        l.b.group("Lectura de registros");
        let selected = l.fresh_bus(PERIPHERAL_BITS as usize);
        let held = l.mux_bus(bus_read, &selected, &bus_address);
        l.connect_flops(&held, &selected);
        let widen = |bits: &[Bit]| -> Vec<Bit> { bits.iter().copied().chain(std::iter::repeat_n(Bit::Zero, 32 - bits.len())).collect() };
        let sign = vector_out[activation - 1];
        let vector_word: Vec<Bit> = vector_out.iter().copied().chain(std::iter::repeat_n(sign, 32 - activation)).collect();
        let mut value = constant(0, 32);
        for (register, word) in [(REG_STATUS, widen(&die_halted)), (REG_VECTOR_DATA, vector_word), (REG_SAMPLE_DONE, widen(&sampler_done)), (REG_SAMPLE_TOKEN, widen(&sampler_token))] {
            let chosen = l.equals(&selected, register);
            value = l.mux_bus(chosen, &value, &word);
        }
        for (from, to) in value.iter().zip(&bus_rdata) {
            l.b.or_to(*from, Bit::Zero, *to);
        }
    }
    b.group("Die de inferencia");
    b.inst("die", &die, &die_pins);
    b.group("Unidad de muestreo");
    b.inst("sampler", &sample, &sampler_pins);
    gen.add(b.finish())
}

pub fn chip_design(config: ChipConfig) -> Design {
    let mut gen = Gen { modules: BTreeMap::new() };
    let top = chip(&mut gen, config);
    Design { top, modules: gen.modules }
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
    parameters: [u32; 32],
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
        Self { engine, code, sampler, parameters: [0; 32], vector_address: 0, sample_base: 0, sample_count: 0, inverse: 0, limit: 0, greedy: false, die_free_at: 0, sampler_done_at: None, sampler_token: 0 }
    }

    /// Recovers the token and position from the parameter registers.
    fn token_and_position(&self) -> (u32, u32) {
        let (mut token, mut position) = (0, 0);
        for p in &self.code.parameters {
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
                self.die_free_at = cycle + self.engine.cycles(routine, token, position);
            }
            REG_VECTOR_ADDRESS => self.vector_address = value,
            REG_SAMPLE_BASE => self.sample_base = value,
            REG_SAMPLE_COUNT => self.sample_count = value,
            REG_SAMPLE_INVERSE => self.inverse = value & 0xFFFF,
            REG_SAMPLE_LIMIT => self.limit = value,
            REG_SAMPLE_GREEDY => self.greedy = value & 1 == 1,
            REG_SAMPLE_SEED => self.sampler.rng = value,
            REG_SAMPLE_START => {
                let logits: Vec<i64> = (0..self.sample_count).map(|i| self.engine.read(self.sample_base + i)).collect();
                let (token, cycles) = self.sampler.sample(&logits, self.inverse, self.limit as usize, self.greedy);
                self.sampler_token = token;
                self.sampler_done_at = Some(cycle + logits.len() as u64 + cycles);
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
pub const INFO_STOP_FIRST: usize = 13;
pub const INFO_STOP_SECOND: usize = 14;
pub const INFO_PARAMETERS: usize = 15;
pub const INFO_PARAMETER_SLOTS: usize = 16;
pub const INFO_BYTE_TO_TOKEN: usize = INFO_PARAMETERS + 4 * INFO_PARAMETER_SLOTS;
pub const INFO_TOKEN_TO_BYTE: usize = INFO_BYTE_TO_TOKEN + 256;

/// Settings for generating text, and the mapping between bytes and tokens.
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
    /// Tokens that end the generation; `u32::MAX` means none.
    pub stop: [u32; 2],
    pub byte_to_token: Vec<u32>,
    pub token_to_byte: Vec<u32>,
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
    info[INFO_STOP_FIRST] = setup.stop[0];
    info[INFO_STOP_SECOND] = setup.stop[1];
    assert!(code.parameters.len() <= INFO_PARAMETER_SLOTS, "demasiados parámetros");
    info[INFO_PARAMETER_COUNT] = code.parameters.len() as u32;
    for p in &code.parameters {
        info.extend([p.register, p.token, p.position, p.constant]);
    }
    info.resize(INFO_BYTE_TO_TOKEN, 0);
    info.extend_from_slice(&setup.byte_to_token);
    info.resize(INFO_TOKEN_TO_BYTE, 0);
    info.extend_from_slice(&setup.token_to_byte);
    info
}
