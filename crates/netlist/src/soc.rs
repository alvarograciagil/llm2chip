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
/// Read: the head of the receive queue without taking it. Write: starts a message of that
/// kind on the way out.
pub const HOST_RX_PEEK: u32 = HOST_BASE + 16;
pub const HOST_TX_START: u32 = HOST_BASE + 16;
/// Write: ends the message on the way out.
pub const HOST_TX_END: u32 = HOST_BASE + 20;
/// The receive status's bits: a payload byte at the head, the clear pin, a message's start
/// (its kind) at the head, and a message's end at the head. Only the first is the usual case.
pub const RX_BYTE: u32 = 1;
pub const RX_START: u32 = 4;
pub const RX_END: u32 = 8;
/// The host port's framing, as SLIP's: a message is its kind, its bytes and this end, and
/// either mark inside a message goes as the escape and then its code.
pub const FRAME_END: u8 = 0xC0;
pub const FRAME_ESCAPE: u8 = 0xDB;
pub const FRAME_ESCAPED_END: u8 = 0xDC;
pub const FRAME_ESCAPED_ESCAPE: u8 = 0xDD;
/// The marks the port keeps beside each byte in its queues, above the byte.
pub const MARK_START: u32 = 1;
pub const MARK_END: u32 = 2;

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
/// While it holds 1, the load controller writes zeros instead of reading the image.
pub const LOADER_FILL: u32 = LOADER_BASE + 36;
/// The host port's status bit that says the clear pin was raised; writing the status
/// register forgets it.
pub const CLEAR_REQUESTED: u32 = 2;

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
    /// Words of program memory, in banks of powers of two: 12,288 (48 KB) are a bank of
    /// 8,192 and one of 4,096.
    pub program_words: u32,
    /// log2 of the words of data memory.
    pub data_bits: u32,
    /// log2 of the bytes each host queue holds.
    pub fifo_bits: u32,
}

/// The program memory the firmware takes: 48 KB, for its 35.6 KB and room to grow.
pub const PROGRAM_WORDS: u32 = 12 << 10;

impl SocConfig {
    /// Bits of a program memory word address.
    pub fn program_bits(&self) -> u32 {
        self.program_words.next_power_of_two().ilog2()
    }

    /// The program memory's banks, largest first: each one's first word and its address bits.
    pub fn program_banks(&self) -> Vec<(u32, u32)> {
        let mut first = 0;
        (0..=self.program_bits()).rev().filter(|k| self.program_words >> k & 1 == 1).map(|k| {
            let bank = (first, k);
            first += 1 << k;
            bank
        }).collect()
    }
}

/// Bits of an entry of the host port's queues: a byte and its mark.
const ENTRY_BITS: u32 = 10;

fn fifo(gen: &mut Gen, bits: u32, width: u32) -> String {
    let name = format!("byte_fifo_{bits}_{width}");
    if gen.modules.contains_key(&name) {
        return name;
    }
    let mut b = Builder::new(&name, "Byte queue", "Circular memory between the host port and the CPU: keeps the bytes, each with the mark of a message's start or end, until the other side collects them.");
    let reset = b.input("reset", 1)[0];
    let push = b.input("push", 1)[0];
    let data_in = b.input("data_in", width);
    let pop = b.input("pop", 1)[0];
    let data_out = b.output("data_out", width);
    let not_empty = b.output("not_empty", 1)[0];
    let not_full = b.output("not_full", 1)[0];
    let valid = b.output("valid", 1)[0];
    let (write_address, read_address, enable);
    {
        let mut l = Logic { b: &mut b, next: 0 };
        l.b.group("Pointers");
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
    b.group("Memory");
    b.ram2p("bytes", bits, width, &[("raddr", read_address), ("waddr", write_address), ("we", vec![enable]), ("wdata", data_in), ("rdata", data_out)]);
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
    let mut b = Builder::new("load_controller", "Load controller", "Reads the chip's image, checks its CRC and writes each word into its destination memory. At boot it copies the firmware by itself.");
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

    l.b.group("Registers");
    let source = l.fresh_bus(32);
    let target = l.fresh_bus(TARGET_BITS as usize);
    let unit = l.fresh_bus(16);
    let address = l.fresh_bus(32);
    let count = l.fresh_bus(32);
    let beats = l.fresh_bus(3);
    let beat = l.fresh_bus(3);
    let crc = l.fresh_bus(32);
    // While set, each element's words are zeros and the image is not read.
    let fill = l.fresh();
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

    l.b.group("Boot");
    // Boot steps: read the program size, copy, read the data size, copy, and release the CPU.
    let steps: Vec<Bit> = boot.clone();
    let [ask_program, wait_program, copy_program, ask_data, wait_data, copy_data, begin] = [steps[0], steps[1], steps[2], steps[3], steps[4], steps[5], steps[6]];
    let header_request = l.or(ask_program, ask_data);
    let got_program = l.and(wait_program, image_valid);
    let got_data = l.and(wait_data, image_valid);
    let got_header = l.or(got_program, got_data);

    l.b.group("Transfer");
    let dma_active = l.any(&[request, wait, emit]);
    // A fill answers itself, in the cycle the image would.
    let answered = l.or(image_valid, fill);
    let response = l.and(wait, answered);
    let zero_word = constant(0, 32);
    let word_in = l.mux_bus(fill, &image_data, &zero_word);
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
        let no = l.not(answered);
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

    l.b.group("Boot");
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

    l.b.group("Registers");
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

    let next_fill = {
        let stored = store(&mut l, 9, &[fill], &reg_data[..1])[0];
        let keep = l.not(reset);
        l.and(stored, keep)
    };
    l.connect_flops(&[next_fill], &[fill]);

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
    let inputs: Vec<Bit> = crc.iter().chain(&word_in).copied().collect();
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

    l.b.group("Output data");
    for (index, lane) in lanes.iter().enumerate() {
        let selected = l.equals(&beat, index as u32);
        let enable = l.and(response, selected);
        let next = l.mux_bus(enable, lane, &word_in);
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

    l.b.group("Image request");
    let not_fill = l.not(fill);
    let reads = l.and(request, not_fill);
    let any_request = l.or(reads, header_request);
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
    let queue = fifo(gen, config.fifo_bits, ENTRY_BITS);
    let load = loader(gen);
    let mut b = Builder::new("soc_control", "Control system", "Control CPU, program and data memories and host port, joined by the internal bus.");
    let reset = b.input("reset", 1)[0];
    let cpu_hold = b.input("cpu_hold", 1)[0];
    // Clears the conversation: the firmware sees it in the host port's status, and the
    // bytes the host had sent are dropped.
    let clear = b.input("clear", 1)[0];
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

    b.group("Program memory");
    let program_bits = config.program_bits();
    let program_address = b.wire("program_address", program_bits);
    let program_write = b.wire("program_write", 1)[0];
    {
        // Banks of powers of two, each answering for the addresses whose top bits name it; a
        // read answers a cycle later, from the bank the address chose then.
        let banks = config.program_banks();
        let mut l = Logic { b: &mut b, next: 0 };
        let mut read = constant(0, 32);
        for (index, &(first, bits)) in banks.iter().enumerate() {
            let chosen = l.equals(&program_address[bits as usize..], first >> bits);
            let write = l.and(program_write, chosen);
            let data = l.b.wire(&format!("program_{index}_data"), 32);
            let name = if banks.len() == 1 { "program".to_string() } else { format!("program_{index}") };
            l.b.sram(&name, bits, 32, &[("addr", program_address[..bits as usize].to_vec()), ("we", vec![write]), ("wdata", load_data[..32].to_vec()), ("rdata", data.clone())]);
            let was = l.fresh();
            l.connect_flops(&[chosen], &[was]);
            read = l.mux_bus(was, &read, &data);
        }
        for (from, to) in read.iter().zip(&imem_rdata) {
            l.b.or_to(*from, Bit::Zero, *to);
        }
    }

    let ram_data = b.wire("ram_data", 32);
    let rx_out = b.wire("rx_out", ENTRY_BITS);
    let rx_not_empty = b.wire("rx_not_empty", 1)[0];
    let tx_not_full = b.wire("tx_not_full", 1)[0];
    let tx_out = b.wire("tx_out", ENTRY_BITS);
    let tx_not_empty = b.wire("tx_not_empty", 1);
    let rx_not_full = b.wire("rx_not_full", 1)[0];
    let tx_valid = b.wire("tx_valid", 1)[0];
    let tx_taken = b.wire("tx_taken", 1)[0];
    let rx_head_valid = b.wire("rx_head_valid", 1)[0];
    let rx_entry = b.wire("rx_entry", ENTRY_BITS);
    let rx_push = b.wire("rx_push", 1)[0];
    let tx_entry = b.wire("tx_entry", ENTRY_BITS);

    let (ram_we, rx_pop, tx_push, read_data, data_address, data_wdata);
    {
        let mut l = Logic { b: &mut b, next: 0 };
        l.b.group("Internal bus");
        let top = dmem_addr[28..32].to_vec();
        let is_ram = l.equals(&top, RAM_BASE >> 28);
        let is_host = l.equals(&top, HOST_BASE >> 28);
        let is_loader = l.equals(&top, LOADER_BASE >> 28);
        let register = &dmem_addr[2..5];
        let (reg_rx_data, reg_rx_status, reg_tx_data, reg_tx_status) = (l.equals(register, 0), l.equals(register, 1), l.equals(register, 2), l.equals(register, 3));
        let (reg_peek_start, reg_tx_end) = (l.equals(register, 4), l.equals(register, 5));
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

        l.b.group("Loading into the memories");
        let boot_write = l.and(load_valid, booting);
        let to_program = l.equals(&load_target, 0);
        let to_data = l.equals(&load_target, 1);
        let program_load = l.and(boot_write, to_program);
        let data_load = l.and(boot_write, to_data);
        l.b.or_to(program_load, Bit::Zero, program_write);
        let pc_word = imem_addr[2..2 + program_bits as usize].to_vec();
        let load_word = load_addr[..program_bits as usize].to_vec();
        let chosen_program = l.mux_bus(booting, &pc_word, &load_word);
        for (from, to) in chosen_program.iter().zip(&program_address) {
            l.b.or_to(*from, Bit::Zero, *to);
        }
        let cpu_word = dmem_addr[2..2 + config.data_bits as usize].to_vec();
        let load_data_word = load_addr[..config.data_bits as usize].to_vec();
        data_address = l.mux_bus(booting, &cpu_word, &load_data_word);
        data_wdata = l.mux_bus(booting, &dmem_wdata, &load_data[..32]);

        l.b.group("Internal bus");
        ram_we = dmem_we
            .iter()
            .map(|w| {
                let cpu = l.and(*w, is_ram);
                l.or(cpu, data_load)
            })
            .collect::<Vec<Bit>>();
        rx_pop = l.and(host_read, reg_rx_data);
        // On the way out the CPU writes a message's start (its kind), its bytes and its end;
        // each goes in the queue with its mark.
        let tx_byte = l.and(host_write, reg_tx_data);
        let tx_start = l.and(host_write, reg_peek_start);
        let tx_end = l.and(host_write, reg_tx_end);
        let some_tx = l.or(tx_byte, tx_start);
        tx_push = l.or(some_tx, tx_end);
        let not_end = l.not(tx_end);
        for (bit, to) in dmem_wdata[..8].iter().zip(&tx_entry) {
            let kept = l.and(*bit, not_end);
            l.b.or_to(kept, Bit::Zero, *to);
        }
        l.b.or_to(tx_start, Bit::Zero, tx_entry[8]);
        l.b.or_to(tx_end, Bit::Zero, tx_entry[9]);

        // Reads answer one cycle after the address, like the memories do.
        l.b.group("Bus response");
        let pick_ram = l.fresh();
        let pick_rx_data = l.fresh();
        let pick_rx_status = l.fresh();
        let pick_tx_status = l.fresh();
        let status_rx = l.fresh();
        let status_tx = l.fresh();
        // What is at the head of the receive queue, once it has settled: a payload byte, a
        // message's start or its end.
        let (rx_start_mark, rx_end_mark) = (rx_out[8], rx_out[9]);
        let is_mark = l.or(rx_start_mark, rx_end_mark);
        let plain = l.not(is_mark);
        let head_byte = l.and(rx_head_valid, plain);
        let head_start = l.and(rx_head_valid, rx_start_mark);
        let head_end = l.and(rx_head_valid, rx_end_mark);
        let _ = rx_not_empty;
        let (status_rx_now, status_tx_now) = (head_byte, tx_not_full);
        // The clear pin stays requested until the firmware writes the status register.
        let requested = l.fresh();
        let next_requested = {
            let forget = l.and(host_write, reg_rx_status);
            let kept = l.not(forget);
            let held = l.and(requested, kept);
            let raised = l.or(held, clear);
            let keep = l.not(reset);
            l.and(raised, keep)
        };
        l.connect_flops(&[next_requested], &[requested]);
        let status_clear = l.fresh();
        let want_rx_data = {
            let peek = l.and(host_read, reg_peek_start);
            let take = l.and(host_read, reg_rx_data);
            l.or(take, peek)
        };
        let (status_start, status_end) = (l.fresh(), l.fresh());
        l.connect_flops(&[head_start, head_end], &[status_start, status_end]);
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
        l.connect_flops(&[is_ram, want_rx_data, want_rx_status, want_tx_status, status_rx_now, status_tx_now, want_busy, want_crc, loader_busy, requested], &[pick_ram, pick_rx_data, pick_rx_status, pick_tx_status, status_rx, status_tx, pick_busy, pick_crc, busy_seen, status_clear]);
        let rx_word: Vec<Bit> = rx_out.iter().copied().chain(std::iter::repeat_n(Bit::Zero, 32 - ENTRY_BITS as usize)).collect();
        let rx_status_word: Vec<Bit> = [status_rx, status_clear, status_start, status_end].into_iter().chain(std::iter::repeat_n(Bit::Zero, 28)).collect();
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
        l.b.group("Host port: messages");
        // In: a byte after a message's end starts the next one, as its kind; 0xC0 ends it;
        // 0xDB says the next byte is 0xC0 (sent as 0xDC) or 0xDB (sent as 0xDD). Each byte
        // goes in the queue with its mark, so the CPU sees where messages begin and end, and
        // a byte lost on the way spoils one message only: the next end starts afresh.
        let not_full = rx_not_full;
        l.b.or_to(not_full, Bit::Zero, host_rx_ready);
        let accepted = l.and(host_rx_valid, not_full);
        let (inside, escaping) = (l.fresh(), l.fresh());
        let is_end = l.equals(&host_rx_data, FRAME_END as u32);
        let is_escape = l.equals(&host_rx_data, FRAME_ESCAPE as u32);
        let outside = l.not(inside);
        let not_escaping = l.not(escaping);
        let not_end_byte = l.not(is_end);
        let not_escape_byte = l.not(is_escape);
        // A message's kind: the first byte that is neither mark.
        let plain_byte = l.and(not_end_byte, not_escape_byte);
        let starts = l.and(outside, plain_byte);
        let ends = l.and(inside, is_end);
        let begins_escape = {
            let a = l.and(inside, not_escaping);
            l.and(a, is_escape)
        };
        let payload = {
            let a = l.and(inside, not_end_byte);
            let skip = l.not(begins_escape);
            l.and(a, skip)
        };
        let pushes = {
            let a = l.or(starts, ends);
            l.or(a, payload)
        };
        let push_now = l.and(accepted, pushes);
        l.b.or_to(push_now, Bit::Zero, rx_push);
        // An escaped byte comes back as it was.
        let restored = {
            let was_end = l.equals(&host_rx_data, FRAME_ESCAPED_END as u32);
            let was_escape = l.equals(&host_rx_data, FRAME_ESCAPED_ESCAPE as u32);
            let to_end = l.and(escaping, was_end);
            let to_escape = l.and(escaping, was_escape);
            let end_byte = constant(FRAME_END as u32, 8);
            let escape_byte = constant(FRAME_ESCAPE as u32, 8);
            let a = l.mux_bus(to_end, &host_rx_data, &end_byte);
            l.mux_bus(to_escape, &a, &escape_byte)
        };
        let not_ends = l.not(ends);
        for (bit, to) in restored.iter().zip(&rx_entry) {
            let kept = l.and(*bit, not_ends);
            l.b.or_to(kept, Bit::Zero, *to);
        }
        l.b.or_to(starts, Bit::Zero, rx_entry[8]);
        l.b.or_to(ends, Bit::Zero, rx_entry[9]);
        let keep = {
            let a = l.not(reset);
            let b = l.not(clear);
            l.and(a, b)
        };
        let next_inside = {
            let now = l.or(inside, starts);
            let left = l.and(now, not_ends);
            let stays = l.mux(accepted, inside, left);
            l.and(stays, keep)
        };
        let next_escaping = {
            let now = l.mux(accepted, escaping, begins_escape);
            l.and(now, keep)
        };
        l.connect_flops(&[next_inside, next_escaping], &[inside, escaping]);

        // Out: a start sends its kind, an end sends 0xC0, and a byte that is either mark goes
        // as 0xDB and then its code, the entry staying at the head for that second byte.
        let second = l.fresh();
        let byte = &tx_out[..8];
        let (start_mark, end_mark) = (tx_out[8], tx_out[9]);
        let marked = l.or(start_mark, end_mark);
        let plain = l.not(marked);
        let out_is_end = l.equals(byte, FRAME_END as u32);
        let out_is_escape = l.equals(byte, FRAME_ESCAPE as u32);
        let special = l.or(out_is_end, out_is_escape);
        let needs_escape = l.and(plain, special);
        let first_of_two = {
            let not_second = l.not(second);
            l.and(needs_escape, not_second)
        };
        let code = {
            let escaped_end = constant(FRAME_ESCAPED_END as u32, 8);
            let escaped_escape = constant(FRAME_ESCAPED_ESCAPE as u32, 8);
            l.mux_bus(out_is_end, &escaped_escape, &escaped_end)
        };
        let shown = {
            let end_byte = constant(FRAME_END as u32, 8);
            let escape_byte = constant(FRAME_ESCAPE as u32, 8);
            let a = l.mux_bus(end_mark, byte, &end_byte);
            let b = l.mux_bus(first_of_two, &a, &escape_byte);
            l.mux_bus(second, &b, &code)
        };
        l.b.or_to(tx_valid, Bit::Zero, host_tx_valid);
        for (from, to) in shown.iter().zip(&host_tx_data) {
            l.b.or_to(*from, Bit::Zero, *to);
        }
        let sent = l.and(host_tx_ready, tx_valid);
        let holds = l.and(sent, first_of_two);
        let not_holds = l.not(holds);
        let taken = l.and(sent, not_holds);
        l.b.or_to(taken, Bit::Zero, tx_taken);
        let next_second = {
            let started = l.or(second, holds);
            let not_taken = l.not(taken);
            let stays = l.and(started, not_taken);
            let alive = l.not(reset);
            l.and(stays, alive)
        };
        l.connect_flops(&[next_second], &[second]);
    }

    b.group("Data memory");
    for byte in 0..4usize {
        b.sram(&format!("data_{byte}"), config.data_bits, 8, &[("addr", data_address.clone()), ("we", vec![ram_we[byte]]), ("wdata", data_wdata[8 * byte..8 * byte + 8].to_vec()), ("rdata", ram_data[8 * byte..8 * byte + 8].to_vec())]);
    }

    b.group("Load controller");
    b.inst("loader", &load, &[("reset", vec![reset]), ("reg_write", vec![loader_reg_write]), ("reg_index", dmem_addr[2..6].to_vec()), ("reg_data", dmem_wdata.clone()), ("image_resp_valid", vec![image_resp_valid]), ("image_resp_data", image_resp_data), ("image_req_valid", vec![image_req_valid]), ("image_req_addr", image_req_addr), ("write_valid", vec![load_valid]), ("write_target", load_target.clone()), ("write_unit", load_unit), ("write_addr", load_addr.clone()), ("write_data", load_data.clone()), ("busy", vec![loader_busy]), ("cpu_hold", vec![booting]), ("crc", loader_crc)]);

    b.group("Host port");
    let rx_reset = b.wire("rx_reset", 1)[0];
    b.or_to(reset, clear, rx_reset);
    b.inst("rx_queue", &queue, &[("reset", vec![rx_reset]), ("push", vec![rx_push]), ("data_in", rx_entry), ("pop", vec![rx_pop]), ("data_out", rx_out), ("not_empty", vec![rx_not_empty]), ("not_full", vec![rx_not_full]), ("valid", vec![rx_head_valid])]);
    b.inst("tx_queue", &queue, &[("reset", vec![reset]), ("push", vec![tx_push]), ("data_in", tx_entry), ("pop", vec![tx_taken]), ("data_out", tx_out), ("not_empty", tx_not_empty), ("not_full", vec![tx_not_full]), ("valid", vec![tx_valid])]);
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
            return Err("not a 32-bit ELF".into());
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
    /// The receive queue: each byte with its mark above it (`MARK_START`, `MARK_END`).
    pub rx: VecDeque<u16>,
    /// The bytes the chip has sent, framed as they leave its pins.
    pub tx: Vec<u8>,
    /// Where the port's reading of the host's bytes stands: inside a message, and after an
    /// escape.
    inside: bool,
    escaping: bool,
    pub image: Vec<u32>,
    pub writes: Vec<LoadWrite>,
    registers: [u32; 6],
    /// The load controller writes zeros instead of reading the image.
    fill: bool,
    /// The clear pin was raised and the firmware has not yet forgotten it.
    clear_requested: bool,
    now: u64,
    busy_until: Option<u64>,
    crc: u32,
    pub peripheral: P,
    /// The CPU's last access outside its memories, for `activity::CpuProfile`.
    pub last_access: Option<crate::activity::BusAccess>,
}

impl SocBus<NoPeripheral> {
    pub fn booted(config: SocConfig, image: &[u32], input: &[u8]) -> Result<Self, String> {
        Self::booted_with(config, image, input, NoPeripheral)
    }
}

impl<P: Peripheral> SocBus<P> {
    pub fn new(config: SocConfig, program: &[u32], data: &[u32], input: &[u8], peripheral: P) -> Self {
        if program.len() > config.program_words as usize {
            panic!("the firmware takes {} words and the program memory holds {}", program.len(), config.program_words);
        }
        let mut image = vec![0; config.program_words as usize];
        image[..program.len()].copy_from_slice(program);
        let mut ram = vec![0; 1 << config.data_bits];
        ram[..data.len()].copy_from_slice(data);
        let mut bus = Self { program: image, ram, rx: VecDeque::new(), tx: Vec::new(), inside: false, escaping: false, image: Vec::new(), writes: Vec::new(), registers: [0; 6], fill: false, clear_requested: false, now: 0, busy_until: None, crc: 0, peripheral, last_access: None };
        bus.send(input);
        bus
    }

    /// The state right after the load controller has copied the firmware from `image`.
    pub fn booted_with(config: SocConfig, image: &[u32], input: &[u8], peripheral: P) -> Result<Self, String> {
        if image.len() < crate::image::HEADER_WORDS || image[0] != crate::image::MAGIC {
            return Err("the image does not start with the chip's header".into());
        }
        let (program_words, data_words) = (image[1] as usize, image[2] as usize);
        let program_start = crate::image::HEADER_WORDS;
        let data_start = program_start + program_words;
        let mut bus = Self::new(config, &image[program_start..data_start], &image[data_start..data_start + data_words], input, peripheral);
        bus.image = image.to_vec();
        Ok(bus)
    }

    /// Raises the clear pin: the bytes the host had sent and the chip had not read are
    /// dropped, and the firmware sees the request in the port's status.
    pub fn clear(&mut self) {
        self.rx.clear();
        (self.inside, self.escaping) = (false, false);
        self.clear_requested = true;
    }

    /// The clear pin was raised and the firmware has not yet ended its clearing.
    pub fn clearing(&self) -> bool {
        self.clear_requested
    }

    /// The host sends more bytes, framed: the port reads them into its queue as the gates do.
    pub fn send(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            if !self.inside {
                if byte != FRAME_END && byte != FRAME_ESCAPE {
                    self.rx.push_back((MARK_START << 8) as u16 | byte as u16);
                    self.inside = true;
                }
            } else if byte == FRAME_END {
                self.rx.push_back((MARK_END << 8) as u16);
                (self.inside, self.escaping) = (false, false);
            } else if self.escaping {
                let restored = match byte {
                    FRAME_ESCAPED_END => FRAME_END,
                    FRAME_ESCAPED_ESCAPE => FRAME_ESCAPE,
                    other => other,
                };
                self.rx.push_back(restored as u16);
                self.escaping = false;
            } else if byte == FRAME_ESCAPE {
                self.escaping = true;
            } else {
                self.rx.push_back(byte as u16);
            }
        }
    }

    /// Copies a segment, or fills it with zeros, and notes when the controller becomes free
    /// again.
    fn start_transfer(&mut self) {
        let [source, target, unit, address, count, beats] = self.registers;
        self.crc = u32::MAX;
        for element in 0..count {
            let first = (source.wrapping_add(element * beats)) as usize;
            let words = if self.fill { vec![0; beats as usize] } else { self.image[first..first + beats as usize].to_vec() };
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
        // Past the last bank no bank answers, and the gates read zeros.
        self.program.get((address as usize / 4) & ((1 << self.program.len().next_power_of_two().ilog2()) - 1)).copied().unwrap_or(0)
    }

    fn read(&mut self, address: u32) -> u32 {
        let value = self.read_from(address);
        if address >> 28 >= 2 {
            self.last_access = Some(crate::activity::BusAccess { address, write: false, value });
        }
        value
    }

    fn write(&mut self, address: u32, data: u32, strobe: u8) {
        if address >> 28 >= 2 {
            self.last_access = Some(crate::activity::BusAccess { address, write: true, value: data });
        }
        self.write_to(address, data, strobe);
    }
}

impl<P: Peripheral> SocBus<P> {
    fn read_from(&mut self, address: u32) -> u32 {
        match address >> 28 {
            1 => self.ram[(address as usize / 4) & (self.ram.len() - 1)],
            2 => match address & 0x1C {
                0 => self.rx.pop_front().unwrap_or(0) as u32,
                4 => {
                    let head = match self.rx.front().map(|e| (*e >> 8) as u32) {
                        Some(0) => RX_BYTE,
                        Some(MARK_START) => RX_START,
                        Some(MARK_END) => RX_END,
                        _ => 0,
                    };
                    head | if self.clear_requested { CLEAR_REQUESTED } else { 0 }
                }
                12 => 1,
                16 => self.rx.front().copied().unwrap_or(0) as u32,
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

    fn write_to(&mut self, address: u32, data: u32, strobe: u8) {
        match address >> 28 {
            1 => {
                let slot = (address as usize / 4) & (self.ram.len() - 1);
                for byte in 0..4 {
                    if strobe >> byte & 1 == 1 {
                        self.ram[slot] = (self.ram[slot] & !(0xFF << (8 * byte))) | (data & (0xFF << (8 * byte)));
                    }
                }
            }
            2 => match address & 0x1C {
                // A byte inside a message: either mark goes as the escape and its code.
                8 => match data as u8 {
                    FRAME_END => self.tx.extend([FRAME_ESCAPE, FRAME_ESCAPED_END]),
                    FRAME_ESCAPE => self.tx.extend([FRAME_ESCAPE, FRAME_ESCAPED_ESCAPE]),
                    byte => self.tx.push(byte),
                },
                4 => self.clear_requested = false,
                16 => self.tx.push(data as u8),
                20 => self.tx.push(FRAME_END),
                _ => {}
            },
            3 => match (address >> 2) & 15 {
                index @ 0..=5 => self.registers[index as usize] = data,
                6 => self.start_transfer(),
                9 => self.fill = data & 1 == 1,
                _ => {}
            },
            4 => self.peripheral.write((address >> 2) & ((1 << PERIPHERAL_BITS) - 1), data, self.now),
            _ => {}
        }
    }
}
