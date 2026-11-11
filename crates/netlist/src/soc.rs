//! The control part of the chip: CPU, program and data memories and the host port.
//! Simulation only replaces what drives the host port's pins.

use crate::cell::Gen;
use crate::rv32::{constant, core, Logic, Rv32Bus};
use crate::{Bit, Builder, Design};
use serde::Serialize;
use std::collections::{BTreeMap, VecDeque};

pub const RAM_BASE: u32 = 0x1000_0000;
pub const HOST_BASE: u32 = 0x2000_0000;
pub const HOST_RX_DATA: u32 = HOST_BASE;
pub const HOST_RX_STATUS: u32 = HOST_BASE + 4;
pub const HOST_TX_DATA: u32 = HOST_BASE + 8;
pub const HOST_TX_STATUS: u32 = HOST_BASE + 12;

pub const LOADER_BASE: u32 = 0x3000_0000;
pub const LOADER_SOURCE: u32 = LOADER_BASE;
pub const LOADER_TARGET: u32 = LOADER_BASE + 4;
pub const LOADER_UNIT: u32 = LOADER_BASE + 8;
pub const LOADER_ADDRESS: u32 = LOADER_BASE + 12;
pub const LOADER_COUNT: u32 = LOADER_BASE + 16;
pub const LOADER_BEATS: u32 = LOADER_BASE + 20;
pub const LOADER_START: u32 = LOADER_BASE + 24;
pub const LOADER_BUSY: u32 = LOADER_BASE + 28;
pub const LOADER_CRC: u32 = LOADER_BASE + 32;

pub const PERIPHERAL_BASE: u32 = 0x4000_0000;
/// Word of data memory where the image builder places the chip description.
pub const CONFIG_WORD: usize = 0x0C00;
/// Word of data memory where the image builder places the segment table.
pub const SEGMENT_TABLE_WORD: usize = 0x1000;
/// Word of data memory with the directory of parts for each number of chips.
pub const PLAN_DIRECTORY_WORD: usize = 0x0FF0;
/// Word of data memory where the image builder places the tokenizer's tables.
pub const TOKENIZER_WORD: usize = 0x2000;
/// The widest thing the load controller writes, in 32-bit image words.
pub const LOAD_LANES: usize = 7;
/// Word-address bits of the peripheral bus.
pub const PERIPHERAL_BITS: u32 = 10;

/// Bits that say which memory a block of the image goes to.
pub const TARGET_BITS: u32 = 4;

#[derive(Clone, Copy, Debug, Serialize)]
pub struct SocConfig {
    /// log2 of the words of program memory.
    pub program_bits: u32,
    /// log2 of the words of data memory.
    pub data_bits: u32,
    /// log2 of the bytes each host queue holds.
    pub fifo_bits: u32,
}

fn fifo(gen: &mut Gen, bits: u32) -> String {
    let name = format!("byte_fifo_{bits}");
    if gen.modules.contains_key(&name) {
        return name;
    }
    let mut b = Builder::new(&name, "Cola de bytes", "Memoria circular entre el puerto de host y la CPU: guarda los bytes hasta que el otro lado los recoge.");
    let reset = b.input("reset", 1)[0];
    let push = b.input("push", 1)[0];
    let data_in = b.input("data_in", 8);
    let pop = b.input("pop", 1)[0];
    let data_out = b.output("data_out", 8);
    let not_empty = b.output("not_empty", 1)[0];
    let not_full = b.output("not_full", 1)[0];
    let valid = b.output("valid", 1)[0];
    let (write_address, read_address, enable);
    {
        let mut l = Logic { b: &mut b, next: 0 };
        l.b.group("Punteros");
        let width = bits as usize + 1;
        let write = l.fresh_bus(width);
        let read = l.fresh_bus(width);
        let same_low: Vec<Bit> = write[..bits as usize].iter().zip(&read).map(|(a, b)| { let d = l.xor(*a, *b); l.not(d) }).collect();
        let low_equal = l.all(&same_low);
        let top_differs = l.xor(write[bits as usize], read[bits as usize]);
        let top_equal = l.not(top_differs);
        let empty = l.and(low_equal, top_equal);
        let full = l.and(low_equal, top_differs);
        let some = l.not(empty);
        let room = l.not(full);
        l.b.or_to(some, Bit::Zero, not_empty);
        l.b.or_to(room, Bit::Zero, not_full);
        let accepted = l.and(push, room);
        let taken = l.and(pop, some);
        // The memory answers a cycle late, so the head is only readable once it has settled.
        let idle = l.not(taken);
        let ready_next = l.and(some, idle);
        l.connect_flops(&[ready_next], &[valid]);

        let increment = |l: &mut Logic, value: &[Bit], enable: Bit| -> Vec<Bit> {
            let mut carry = enable;
            value.iter().map(|bit| { let out = l.xor(*bit, carry); carry = l.and(*bit, carry); out }).collect()
        };
        let keep = l.not(reset);
        let next_write: Vec<Bit> = increment(&mut l, &write, accepted).iter().map(|b| l.and(*b, keep)).collect();
        let next_read: Vec<Bit> = increment(&mut l, &read, taken).iter().map(|b| l.and(*b, keep)).collect();
        l.connect_flops(&next_write, &write);
        l.connect_flops(&next_read, &read);
        write_address = write[..bits as usize].to_vec();
        read_address = read[..bits as usize].to_vec();
        enable = accepted;
    }
    b.group("Memoria");
    b.ram2p("bytes", bits, 8, &[("raddr", read_address), ("waddr", write_address), ("we", vec![enable]), ("wdata", data_in), ("rdata", data_out)]);
    gen.add(b.finish())
}

/// Which input bits feed each output of the parallel CRC-32.
fn crc_terms() -> Vec<u64> {
    // Each bit is tracked as a linear combination of the 32 state and 32 data inputs.
    let mut value: Vec<u64> = (0..32).map(|i| (1u64 << i) | (1u64 << (32 + i))).collect();
    for _ in 0..32 {
        let feedback = value[0];
        for i in 0..31 {
            value[i] = value[i + 1];
        }
        value[31] = 0;
        for (i, bit) in value.iter_mut().enumerate() {
            if 0xEDB8_8320u32 >> i & 1 == 1 {
                *bit ^= feedback;
            }
        }
    }
    value
}

fn counter_step(l: &mut Logic, value: &[Bit], enable: Bit) -> Vec<Bit> {
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

fn loader(gen: &mut Gen) -> String {
    let mut b = Builder::new("load_controller", "Controlador de carga", "Lee la imagen del chip, comprueba su CRC y escribe cada palabra en la memoria de destino. Al arrancar copia el firmware por sí solo.");
    let reset = b.input("reset", 1)[0];
    let reg_write = b.input("reg_write", 1)[0];
    let reg_index = b.input("reg_index", 4);
    let reg_data = b.input("reg_data", 32);
    let image_valid = b.input("image_resp_valid", 1)[0];
    let image_data = b.input("image_resp_data", 32);
    let image_req = b.output("image_req_valid", 1)[0];
    let image_addr = b.output("image_req_addr", 32);
    let write_valid = b.output("write_valid", 1)[0];
    let write_target = b.output("write_target", TARGET_BITS);
    let write_unit = b.output("write_unit", 16);
    let write_addr = b.output("write_addr", 32);
    let write_data = b.output("write_data", 32 * LOAD_LANES as u32);
    let busy_out = b.output("busy", 1)[0];
    let cpu_hold = b.output("cpu_hold", 1)[0];
    let crc_out = b.output("crc", 32);
    let mut l = Logic { b: &mut b, next: 0 };

    l.b.group("Registros");
    let source = l.fresh_bus(32);
    let target = l.fresh_bus(TARGET_BITS as usize);
    let unit = l.fresh_bus(16);
    let address = l.fresh_bus(32);
    let count = l.fresh_bus(32);
    let beats = l.fresh_bus(3);
    let beat = l.fresh_bus(3);
    let crc = l.fresh_bus(32);
    let lanes: Vec<Vec<Bit>> = (0..LOAD_LANES).map(|_| l.fresh_bus(32)).collect();

    l.b.group("Control");
    let request = l.fresh();
    let wait = l.fresh();
    let emit = l.fresh();
    let boot = l.fresh_bus(7);
    let running = l.fresh();

    let write_reg = |l: &mut Logic, index: u32| {
        let hit = l.equals(&reg_index, index);
        l.and(hit, reg_write)
    };
    let start = write_reg(&mut l, 6);
    let in_boot = l.not(running);

    l.b.group("Arranque");
    // Boot steps: read the program size, copy, read the data size, copy, and release the CPU.
    let steps: Vec<Bit> = boot.clone();
    let [ask_program, wait_program, copy_program, ask_data, wait_data, copy_data, begin] = [steps[0], steps[1], steps[2], steps[3], steps[4], steps[5], steps[6]];
    let header_request = l.or(ask_program, ask_data);
    let got_program = l.and(wait_program, image_valid);
    let got_data = l.and(wait_data, image_valid);
    let got_header = l.or(got_program, got_data);

    l.b.group("Transferencia");
    let dma_active = l.any(&[request, wait, emit]);
    let response = l.and(wait, image_valid);
    let last_beat = {
        let next_beat: Vec<Bit> = counter_step(&mut l, &beat, Bit::One);
        let same: Vec<Bit> = next_beat.iter().zip(&beats).map(|(a, b)| { let d = l.xor(*a, *b); l.not(d) }).collect();
        l.all(&same)
    };
    let word_complete = l.and(response, last_beat);
    let more_beats = {
        let more = l.not(last_beat);
        l.and(response, more)
    };
    let last_element = {
        let one = constant(1, 32);
        let same: Vec<Bit> = count.iter().zip(&one).map(|(a, b)| { let d = l.xor(*a, *b); l.not(d) }).collect();
        l.all(&same)
    };
    let dma_done = l.and(emit, last_element);
    let continue_after_emit = {
        let more = l.not(last_element);
        l.and(emit, more)
    };
    let header_start = got_header;
    let dma_start = l.or(start, header_start);

    let next_request = {
        let t = l.any(&[dma_start, more_beats, continue_after_emit]);
        let keep = l.not(reset);
        l.and(t, keep)
    };
    let hold_wait = {
        let no = l.not(image_valid);
        l.and(wait, no)
    };
    let next_wait = {
        let t = l.or(request, hold_wait);
        let keep = l.not(reset);
        l.and(t, keep)
    };
    let next_emit = {
        let keep = l.not(reset);
        l.and(word_complete, keep)
    };
    l.connect_flops(&[next_request, next_wait, next_emit], &[request, wait, emit]);

    l.b.group("Arranque");
    let keep = l.not(reset);
    let keep_wait_program = {
        let no = l.not(image_valid);
        l.and(wait_program, no)
    };
    let keep_wait_data = {
        let no = l.not(image_valid);
        l.and(wait_data, no)
    };
    let copy_running = l.not(dma_done);
    let keep_copy_program = {
        let t = l.and(copy_program, copy_running);
        l.and(t, keep)
    };
    let keep_copy_data = {
        let t = l.and(copy_data, copy_running);
        l.and(t, keep)
    };
    let finish_program = l.and(copy_program, dma_done);
    let finish_data = l.and(copy_data, dma_done);
    let step_ask_program = reset;
    let step_wait_program = {
        let t = l.or(ask_program, keep_wait_program);
        l.and(t, keep)
    };
    let step_copy_program = {
        let t = l.and(got_program, keep);
        l.or(t, keep_copy_program)
    };
    let step_ask_data = l.and(finish_program, keep);
    let step_wait_data = {
        let t = l.or(ask_data, keep_wait_data);
        l.and(t, keep)
    };
    let step_copy_data = {
        let t = l.and(got_data, keep);
        l.or(t, keep_copy_data)
    };
    let step_begin = l.and(finish_data, keep);
    l.connect_flops(&[step_ask_program, step_wait_program, step_copy_program, step_ask_data, step_wait_data, step_copy_data, step_begin], &steps);
    let next_running = {
        let t = l.or(running, begin);
        l.and(t, keep)
    };
    l.connect_flops(&[next_running], &[running]);
    let not_running = l.not(running);
    l.b.or_to(not_running, Bit::Zero, cpu_hold);

    l.b.group("Registros");
    let advance_source = response;
    let incremented_source = counter_step(&mut l, &source, advance_source);
    let mut next_source = incremented_source;
    let header_size = constant(crate::image::HEADER_WORDS as u32, 32);
    let hit = write_reg(&mut l, 0);
    next_source = l.mux_bus(hit, &next_source, &reg_data);
    next_source = l.mux_bus(reset, &next_source, &header_size);
    l.connect_flops(&next_source, &source);

    let store = |l: &mut Logic, index: u32, q: &[Bit], data: &[Bit]| -> Vec<Bit> {
        let hit = write_reg(l, index);
        l.mux_bus(hit, q, data)
    };
    let mut next_target = store(&mut l, 1, &target, &reg_data[..TARGET_BITS as usize]);
    let program_target = constant(0, TARGET_BITS as usize);
    let data_target = constant(1, TARGET_BITS as usize);
    next_target = l.mux_bus(got_program, &next_target, &program_target);
    next_target = l.mux_bus(got_data, &next_target, &data_target);
    l.connect_flops(&next_target, &target);

    let next_unit = store(&mut l, 2, &unit, &reg_data[..16]);
    l.connect_flops(&next_unit, &unit);

    let stepped_address = counter_step(&mut l, &address, emit);
    let mut next_address = store(&mut l, 3, &stepped_address, &reg_data);
    let zero32 = constant(0, 32);
    next_address = l.mux_bus(got_header, &next_address, &zero32);
    l.connect_flops(&next_address, &address);

    let down: Vec<Bit> = {
        let mut borrow = emit;
        count
            .iter()
            .map(|bit| {
                let out = l.xor(*bit, borrow);
                let inverse = l.not(*bit);
                borrow = l.and(inverse, borrow);
                out
            })
            .collect()
    };
    let mut next_count = store(&mut l, 4, &down, &reg_data);
    next_count = l.mux_bus(got_header, &next_count, &image_data);
    l.connect_flops(&next_count, &count);

    let mut next_beats = store(&mut l, 5, &beats, &reg_data[..3]);
    let one_beat = constant(1, 3);
    next_beats = l.mux_bus(got_header, &next_beats, &one_beat);
    l.connect_flops(&next_beats, &beats);

    let stepped_beat = counter_step(&mut l, &beat, response);
    let zero3 = constant(0, 3);
    let clear_beat = l.or(dma_start, emit);
    let next_beat = l.mux_bus(clear_beat, &stepped_beat, &zero3);
    l.connect_flops(&next_beat, &beat);

    l.b.group("CRC");
    let terms = crc_terms();
    let inputs: Vec<Bit> = crc.iter().chain(&image_data).copied().collect();
    let crc_step: Vec<Bit> = terms
        .iter()
        .map(|mask| {
            let used: Vec<Bit> = (0..64).filter(|i| mask >> i & 1 == 1).map(|i| inputs[i]).collect();
            used.into_iter().reduce(|a, b| l.xor(a, b)).unwrap_or(Bit::Zero)
        })
        .collect();
    let running_crc = l.mux_bus(response, &crc, &crc_step);
    let ones = constant(u32::MAX, 32);
    let next_crc = l.mux_bus(dma_start, &running_crc, &ones);
    l.connect_flops(&next_crc, &crc);
    for (from, to) in crc.iter().zip(&crc_out) {
        l.b.or_to(*from, Bit::Zero, *to);
    }

    l.b.group("Datos de salida");
    for (index, lane) in lanes.iter().enumerate() {
        let selected = l.equals(&beat, index as u32);
        let enable = l.and(response, selected);
        let next = l.mux_bus(enable, lane, &image_data);
        l.connect_flops(&next, lane);
    }
    for (index, to) in write_data.iter().enumerate() {
        l.b.or_to(lanes[index / 32][index % 32], Bit::Zero, *to);
    }
    l.b.or_to(emit, Bit::Zero, write_valid);
    for (from, to) in target.iter().zip(&write_target) {
        l.b.or_to(*from, Bit::Zero, *to);
    }
    for (from, to) in unit.iter().zip(&write_unit) {
        l.b.or_to(*from, Bit::Zero, *to);
    }
    for (from, to) in address.iter().zip(&write_addr) {
        l.b.or_to(*from, Bit::Zero, *to);
    }

    l.b.group("Petición a la imagen");
    let any_request = l.or(request, header_request);
    l.b.or_to(any_request, Bit::Zero, image_req);
    let one = constant(1, 32);
    let two = constant(2, 32);
    let header_address = l.mux_bus(ask_data, &one, &two);
    let requested = l.mux_bus(header_request, &source, &header_address);
    for (from, to) in requested.iter().zip(&image_addr) {
        l.b.or_to(*from, Bit::Zero, *to);
    }
    let busy = l.or(dma_active, in_boot);
    l.b.or_to(busy, Bit::Zero, busy_out);
    gen.add(b.finish())
}

pub(crate) fn soc(gen: &mut Gen, config: SocConfig) -> String {
    let cpu = core(gen);
    let queue = fifo(gen, config.fifo_bits);
    let load = loader(gen);
    let mut b = Builder::new("soc_control", "Sistema de control", "CPU de control, memorias de programa y datos y puerto de host, unidos por el bus interno.");
    let reset = b.input("reset", 1)[0];
    let cpu_hold = b.input("cpu_hold", 1)[0];
    let host_rx_valid = b.input("host_rx_valid", 1)[0];
    let host_rx_data = b.input("host_rx_data", 8);
    let host_tx_ready = b.input("host_tx_ready", 1)[0];
    let host_rx_ready = b.output("host_rx_ready", 1)[0];
    let host_tx_valid = b.output("host_tx_valid", 1)[0];
    let host_tx_data = b.output("host_tx_data", 8);
    let halted = b.output("halted", 1)[0];
    let image_resp_valid = b.input("image_resp_valid", 1)[0];
    let image_resp_data = b.input("image_resp_data", 32);
    let image_req_valid = b.output("image_req_valid", 1)[0];
    let image_req_addr = b.output("image_req_addr", 32);
    let load_valid = b.output("load_valid", 1)[0];
    let load_target = b.output("load_target", TARGET_BITS);
    let load_unit = b.output("load_unit", 16);
    let load_addr = b.output("load_addr", 32);
    let load_data = b.output("load_data", 32 * LOAD_LANES as u32);
    let booting = b.output("booting", 1)[0];
    let peripheral_write = b.output("peripheral_write", 1)[0];
    let peripheral_read = b.output("peripheral_read", 1)[0];
    let peripheral_address = b.output("peripheral_address", PERIPHERAL_BITS);
    let peripheral_wdata = b.output("peripheral_wdata", 32);
    let peripheral_rdata = b.input("peripheral_rdata", 32);
    let debug_pc = b.output("debug_pc", 32);

    let loader_busy = b.wire("loader_busy", 1)[0];
    let loader_crc = b.wire("loader_crc", 32);
    let loader_reg_write = b.wire("loader_reg_write", 1)[0];
    let imem_addr = b.wire("imem_addr", 32);
    let imem_rdata = b.wire("imem_rdata", 32);
    let dmem_addr = b.wire("dmem_addr", 32);
    let dmem_wdata = b.wire("dmem_wdata", 32);
    let dmem_we = b.wire("dmem_we", 4);
    let dmem_re = b.wire("dmem_re", 1)[0];
    let dmem_rdata = b.wire("dmem_rdata", 32);
    let rf_we = b.wire("rf_we", 1);
    let rf_addr = b.wire("rf_addr", 5);
    let rf_wdata = b.wire("rf_wdata", 32);
    b.group("CPU");
    let cpu_reset = b.wire("cpu_reset", 1)[0];
    let held = b.wire("held", 1)[0];
    b.or_to(reset, cpu_hold, held);
    b.or_to(held, booting, cpu_reset);
    b.inst("cpu", &cpu, &[("reset", vec![cpu_reset]), ("imem_rdata", imem_rdata.clone()), ("dmem_rdata", dmem_rdata.clone()), ("imem_addr", imem_addr.clone()), ("dmem_addr", dmem_addr.clone()), ("dmem_wdata", dmem_wdata.clone()), ("dmem_we", dmem_we.clone()), ("dmem_re", vec![dmem_re]), ("rf_we", rf_we), ("rf_addr", rf_addr), ("rf_wdata", rf_wdata), ("halted", vec![halted])]);

    b.group("Memoria de programa");
    let program_address = b.wire("program_address", config.program_bits);
    let program_write = b.wire("program_write", 1)[0];
    b.sram("program", config.program_bits, 32, &[("addr", program_address.clone()), ("we", vec![program_write]), ("wdata", load_data[..32].to_vec()), ("rdata", imem_rdata.clone())]);

    let ram_data = b.wire("ram_data", 32);
    let rx_out = b.wire("rx_out", 8);
    let rx_not_empty = b.wire("rx_not_empty", 1)[0];
    let tx_not_full = b.wire("tx_not_full", 1)[0];
    let tx_out = b.wire("tx_out", 8);
    let tx_not_empty = b.wire("tx_not_empty", 1);
    let rx_not_full = b.wire("rx_not_full", 1)[0];
    let tx_valid = b.wire("tx_valid", 1)[0];
    let tx_taken = b.wire("tx_taken", 1)[0];
    b.group("Puerto de host");
    b.and_to(host_tx_ready, tx_valid, tx_taken);

    let (ram_we, rx_pop, tx_push, read_data, data_address, data_wdata);
    {
        let mut l = Logic { b: &mut b, next: 0 };
        l.b.group("Bus interno");
        let top = dmem_addr[28..32].to_vec();
        let is_ram = l.equals(&top, RAM_BASE >> 28);
        let is_host = l.equals(&top, HOST_BASE >> 28);
        let is_loader = l.equals(&top, LOADER_BASE >> 28);
        let register = &dmem_addr[2..4];
        let (reg_rx_data, reg_rx_status, reg_tx_data, reg_tx_status) = (l.equals(register, 0), l.equals(register, 1), l.equals(register, 2), l.equals(register, 3));
        let any_write = l.any(&dmem_we);
        let host_read = l.and(is_host, dmem_re);
        let host_write = l.and(is_host, any_write);
        let loader_write = l.and(is_loader, any_write);
        l.b.or_to(loader_write, Bit::Zero, loader_reg_write);

        for (from, to) in imem_addr.iter().zip(&debug_pc) {
            l.b.or_to(*from, Bit::Zero, *to);
        }
        let is_peripheral = l.equals(&top, PERIPHERAL_BASE >> 28);
        let write_peripheral = l.and(is_peripheral, any_write);
        let read_peripheral = l.and(is_peripheral, dmem_re);
        l.b.or_to(write_peripheral, Bit::Zero, peripheral_write);
        l.b.or_to(read_peripheral, Bit::Zero, peripheral_read);
        for (from, to) in dmem_addr[2..].iter().zip(&peripheral_address) {
            l.b.or_to(*from, Bit::Zero, *to);
        }
        for (from, to) in dmem_wdata.iter().zip(&peripheral_wdata) {
            l.b.or_to(*from, Bit::Zero, *to);
        }

        l.b.group("Carga en las memorias");
        let boot_write = l.and(load_valid, booting);
        let to_program = l.equals(&load_target, 0);
        let to_data = l.equals(&load_target, 1);
        let program_load = l.and(boot_write, to_program);
        let data_load = l.and(boot_write, to_data);
        l.b.or_to(program_load, Bit::Zero, program_write);
        let pc_word = imem_addr[2..2 + config.program_bits as usize].to_vec();
        let load_word = load_addr[..config.program_bits as usize].to_vec();
        let chosen_program = l.mux_bus(booting, &pc_word, &load_word);
        for (from, to) in chosen_program.iter().zip(&program_address) {
            l.b.or_to(*from, Bit::Zero, *to);
        }
        let cpu_word = dmem_addr[2..2 + config.data_bits as usize].to_vec();
        let load_data_word = load_addr[..config.data_bits as usize].to_vec();
        data_address = l.mux_bus(booting, &cpu_word, &load_data_word);
        data_wdata = l.mux_bus(booting, &dmem_wdata, &load_data[..32]);

        l.b.group("Bus interno");
        ram_we = dmem_we
            .iter()
            .map(|w| {
                let cpu = l.and(*w, is_ram);
                l.or(cpu, data_load)
            })
            .collect::<Vec<Bit>>();
        rx_pop = l.and(host_read, reg_rx_data);
        tx_push = l.and(host_write, reg_tx_data);

        // Reads answer one cycle after the address, like the memories do.
        l.b.group("Respuesta del bus");
        let pick_ram = l.fresh();
        let pick_rx_data = l.fresh();
        let pick_rx_status = l.fresh();
        let pick_tx_status = l.fresh();
        let status_rx = l.fresh();
        let status_tx = l.fresh();
        let (status_rx_now, status_tx_now) = (rx_not_empty, tx_not_full);
        let want_rx_data = l.and(host_read, reg_rx_data);
        let want_rx_status = l.and(host_read, reg_rx_status);
        let want_tx_status = l.and(host_read, reg_tx_status);
        let loader_read = l.and(is_loader, dmem_re);
        let loader_register = &dmem_addr[2..6];
        let reads_crc = l.equals(loader_register, 8);
        let want_busy = {
            let is_busy = l.equals(loader_register, 7);
            l.and(loader_read, is_busy)
        };
        let want_crc = l.and(loader_read, reads_crc);
        let (pick_busy, pick_crc, busy_seen) = (l.fresh(), l.fresh(), l.fresh());
        let pick_peripheral = l.fresh();
        l.connect_flops(&[read_peripheral], &[pick_peripheral]);
        l.connect_flops(&[is_ram, want_rx_data, want_rx_status, want_tx_status, status_rx_now, status_tx_now, want_busy, want_crc, loader_busy], &[pick_ram, pick_rx_data, pick_rx_status, pick_tx_status, status_rx, status_tx, pick_busy, pick_crc, busy_seen]);
        let rx_word: Vec<Bit> = rx_out.iter().copied().chain(std::iter::repeat_n(Bit::Zero, 24)).collect();
        let rx_status_word: Vec<Bit> = std::iter::once(status_rx).chain(std::iter::repeat_n(Bit::Zero, 31)).collect();
        let tx_status_word: Vec<Bit> = std::iter::once(status_tx).chain(std::iter::repeat_n(Bit::Zero, 31)).collect();
        let zero = constant(0, 32);
        let mut value = l.mux_bus(pick_ram, &zero, &ram_data);
        value = l.mux_bus(pick_rx_data, &value, &rx_word);
        value = l.mux_bus(pick_rx_status, &value, &rx_status_word);
        value = l.mux_bus(pick_tx_status, &value, &tx_status_word);
        let busy_word: Vec<Bit> = std::iter::once(busy_seen).chain(std::iter::repeat_n(Bit::Zero, 31)).collect();
        value = l.mux_bus(pick_busy, &value, &busy_word);
        value = l.mux_bus(pick_crc, &value, &loader_crc);
        value = l.mux_bus(pick_peripheral, &value, &peripheral_rdata);
        read_data = value;
        for (from, to) in read_data.iter().zip(&dmem_rdata) {
            l.b.or_to(*from, Bit::Zero, *to);
        }
        let not_full = rx_not_full;
        l.b.or_to(not_full, Bit::Zero, host_rx_ready);
        l.b.or_to(tx_valid, Bit::Zero, host_tx_valid);
        for (from, to) in tx_out.iter().zip(&host_tx_data) {
            l.b.or_to(*from, Bit::Zero, *to);
        }
    }

    b.group("Memoria de datos");
    for byte in 0..4usize {
        b.sram(&format!("data_{byte}"), config.data_bits, 8, &[("addr", data_address.clone()), ("we", vec![ram_we[byte]]), ("wdata", data_wdata[8 * byte..8 * byte + 8].to_vec()), ("rdata", ram_data[8 * byte..8 * byte + 8].to_vec())]);
    }

    b.group("Controlador de carga");
    b.inst("loader", &load, &[("reset", vec![reset]), ("reg_write", vec![loader_reg_write]), ("reg_index", dmem_addr[2..6].to_vec()), ("reg_data", dmem_wdata.clone()), ("image_resp_valid", vec![image_resp_valid]), ("image_resp_data", image_resp_data), ("image_req_valid", vec![image_req_valid]), ("image_req_addr", image_req_addr), ("write_valid", vec![load_valid]), ("write_target", load_target.clone()), ("write_unit", load_unit), ("write_addr", load_addr.clone()), ("write_data", load_data.clone()), ("busy", vec![loader_busy]), ("cpu_hold", vec![booting]), ("crc", loader_crc)]);

    b.group("Puerto de host");
    let rx_head_valid = b.wire("rx_head_valid", 1);
    b.inst("rx_queue", &queue, &[("reset", vec![reset]), ("push", vec![host_rx_valid]), ("data_in", host_rx_data), ("pop", vec![rx_pop]), ("data_out", rx_out), ("not_empty", vec![rx_not_empty]), ("not_full", vec![rx_not_full]), ("valid", rx_head_valid)]);
    b.inst("tx_queue", &queue, &[("reset", vec![reset]), ("push", vec![tx_push]), ("data_in", dmem_wdata[..8].to_vec()), ("pop", vec![tx_taken]), ("data_out", tx_out), ("not_empty", tx_not_empty), ("not_full", vec![tx_not_full]), ("valid", vec![tx_valid])]);
    gen.add(b.finish())
}

pub fn soc_design(config: SocConfig) -> Design {
    let mut gen = Gen { modules: BTreeMap::new() };
    let top = soc(&mut gen, config);
    Design { top, modules: gen.modules }
}

/// A program split into what each memory must hold at start-up.
#[derive(Clone, Debug, Default)]
pub struct FirmwareImage {
    pub program: Vec<u32>,
    pub data: Vec<u32>,
}

impl FirmwareImage {
    /// Reads the loadable segments of a 32-bit little-endian ELF file.
    pub fn from_elf(elf: &[u8]) -> Result<Self, String> {
        if elf.len() < 52 || elf[..4] != [0x7F, b'E', b'L', b'F'] || elf[4] != 1 || elf[5] != 1 {
            return Err("no es un ELF de 32 bits".into());
        }
        let half = |at: usize| u16::from_le_bytes([elf[at], elf[at + 1]]) as usize;
        let word = |at: usize| u32::from_le_bytes([elf[at], elf[at + 1], elf[at + 2], elf[at + 3]]) as usize;
        let mut image = Self::default();
        for index in 0..half(44) {
            let header = word(28) + index * half(42);
            if word(header) != 1 || word(header + 16) == 0 {
                continue;
            }
            let (offset, address, length) = (word(header + 4), word(header + 8), word(header + 16));
            let (target, base) = if address as u32 >= RAM_BASE { (&mut image.data, RAM_BASE as usize) } else { (&mut image.program, 0) };
            let end = (address + length - base).div_ceil(4);
            if target.len() < end {
                target.resize(end, 0);
            }
            for (i, byte) in elf[offset..offset + length].iter().enumerate() {
                let at = address + i - base;
                target[at / 4] |= (*byte as u32) << (8 * (at % 4));
            }
        }
        Ok(image)
    }
}

/// One write the load controller makes outside the firmware's memories.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LoadWrite {
    pub target: u32,
    pub unit: u32,
    pub address: u32,
    pub words: Vec<u32>,
}

/// What sits on the peripheral bus when the CPU runs at instruction level.
pub trait Peripheral {
    fn read(&mut self, register: u32, cycle: u64) -> u32;
    fn write(&mut self, register: u32, value: u32, cycle: u64);
    /// One write of the load controller outside the firmware's memories.
    fn load(&mut self, write: &LoadWrite);
}

/// An empty peripheral bus.
#[derive(Default)]
pub struct NoPeripheral;

impl Peripheral for NoPeripheral {
    fn read(&mut self, _register: u32, _cycle: u64) -> u32 {
        0
    }

    fn write(&mut self, _register: u32, _value: u32, _cycle: u64) {}

    fn load(&mut self, _write: &LoadWrite) {}
}

/// The same machine at instruction level, for firmware runs too long for gates.
pub struct SocBus<P = NoPeripheral> {
    pub program: Vec<u32>,
    pub ram: Vec<u32>,
    pub rx: VecDeque<u8>,
    pub tx: Vec<u8>,
    pub image: Vec<u32>,
    pub writes: Vec<LoadWrite>,
    registers: [u32; 6],
    now: u64,
    busy_until: Option<u64>,
    crc: u32,
    pub peripheral: P,
}

impl SocBus<NoPeripheral> {
    pub fn booted(config: SocConfig, image: &[u32], input: &[u8]) -> Result<Self, String> {
        Self::booted_with(config, image, input, NoPeripheral)
    }
}

impl<P: Peripheral> SocBus<P> {
    pub fn new(config: SocConfig, program: &[u32], data: &[u32], input: &[u8], peripheral: P) -> Self {
        let mut image = vec![0; 1 << config.program_bits];
        image[..program.len()].copy_from_slice(program);
        let mut ram = vec![0; 1 << config.data_bits];
        ram[..data.len()].copy_from_slice(data);
        Self { program: image, ram, rx: input.iter().copied().collect(), tx: Vec::new(), image: Vec::new(), writes: Vec::new(), registers: [0; 6], now: 0, busy_until: None, crc: 0, peripheral }
    }

    /// The state right after the load controller has copied the firmware from `image`.
    pub fn booted_with(config: SocConfig, image: &[u32], input: &[u8], peripheral: P) -> Result<Self, String> {
        if image.len() < crate::image::HEADER_WORDS || image[0] != crate::image::MAGIC {
            return Err("la imagen no empieza por la cabecera del chip".into());
        }
        let (program_words, data_words) = (image[1] as usize, image[2] as usize);
        let program_start = crate::image::HEADER_WORDS;
        let data_start = program_start + program_words;
        let mut bus = Self::new(config, &image[program_start..data_start], &image[data_start..data_start + data_words], input, peripheral);
        bus.image = image.to_vec();
        Ok(bus)
    }

    /// Copies a segment and notes when the controller becomes free again.
    fn start_transfer(&mut self) {
        let [source, target, unit, address, count, beats] = self.registers;
        self.crc = u32::MAX;
        for element in 0..count {
            let first = (source + element * beats) as usize;
            let words = self.image[first..first + beats as usize].to_vec();
            for word in &words {
                self.crc = crate::image::crc32_word(self.crc, *word);
            }
            let write = LoadWrite { target, unit, address: address + element, words };
            self.peripheral.load(&write);
            self.writes.push(write);
        }
        self.busy_until = Some(self.now + count as u64 * (2 * beats as u64 + 1));
    }
}

impl<P: Peripheral> Rv32Bus for SocBus<P> {
    fn at_cycle(&mut self, cycle: u64) {
        self.now = cycle;
    }

    fn fetch(&mut self, address: u32) -> u32 {
        self.program[(address as usize / 4) & (self.program.len() - 1)]
    }

    fn read(&mut self, address: u32) -> u32 {
        match address >> 28 {
            1 => self.ram[(address as usize / 4) & (self.ram.len() - 1)],
            2 => match address & 0xC {
                0 => self.rx.pop_front().unwrap_or(0) as u32,
                4 => !self.rx.is_empty() as u32,
                12 => 1,
                _ => 0,
            },
            3 => match (address >> 2) & 15 {
                7 => self.busy_until.is_some_and(|end| self.now <= end) as u32,
                8 => self.crc,
                _ => 0,
            },
            4 => self.peripheral.read((address >> 2) & ((1 << PERIPHERAL_BITS) - 1), self.now),
            _ => 0,
        }
    }

    fn write(&mut self, address: u32, data: u32, strobe: u8) {
        match address >> 28 {
            1 => {
                let slot = (address as usize / 4) & (self.ram.len() - 1);
                for byte in 0..4 {
                    if strobe >> byte & 1 == 1 {
                        self.ram[slot] = (self.ram[slot] & !(0xFF << (8 * byte))) | (data & (0xFF << (8 * byte)));
                    }
                }
            }
            2 if address & 0xC == 8 => self.tx.push(data as u8),
            3 => match (address >> 2) & 15 {
                index @ 0..=5 => self.registers[index as usize] = data,
                6 => self.start_transfer(),
                _ => {}
            },
            4 => self.peripheral.write((address >> 2) & ((1 << PERIPHERAL_BITS) - 1), data, self.now),
            _ => {}
        }
    }
}
