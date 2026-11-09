//! Firmware of the control CPU: loads the model, reads text, runs the die and writes text.
//! It talks to the outside world only through the host port.
#![no_std]
#![no_main]

use core::arch::global_asm;
use core::ptr::{read_volatile, write_volatile};
use tokenizer_rt::{Decoding, Tokenizer, F_ARENA_WORDS, F_TEXT, F_TEXT_BYTES};

const HOST_RX_DATA: *const u32 = 0x2000_0000 as *const u32;
const HOST_RX_STATUS: *const u32 = 0x2000_0004 as *const u32;
const HOST_TX_DATA: *mut u32 = 0x2000_0008 as *mut u32;
const HOST_TX_STATUS: *const u32 = 0x2000_000C as *const u32;

const LOADER: *mut u32 = 0x3000_0000 as *mut u32;
const LOADER_START: usize = 6;
const LOADER_BUSY: usize = 7;
const LOADER_CRC: usize = 8;
const DATA_MEMORY: *const u32 = 0x1000_0000 as *const u32;
const DEFAULT_SEGMENT_TABLE: usize = 0x1000;
const PLAN_DIRECTORY: usize = 0x0FF0;
const SEGMENT_FIELDS: usize = 7;

const PERIPHERALS: *mut u32 = 0x4000_0000 as *mut u32;
const REG_ENTRY: usize = 32;
const REG_STATUS: usize = 33;
const REG_VECTOR_ADDRESS: usize = 34;
const REG_VECTOR_DATA: usize = 35;
const REG_SAMPLE_BASE: usize = 36;
const REG_SAMPLE_COUNT: usize = 37;
const REG_SAMPLE_INVERSE: usize = 38;
const REG_SAMPLE_LIMIT: usize = 39;
const REG_SAMPLE_GREEDY: usize = 40;
const REG_SAMPLE_SEED: usize = 41;
const REG_SAMPLE_START: usize = 42;
const REG_SAMPLE_DONE: usize = 43;
const REG_SAMPLE_TOKEN: usize = 44;
const REG_FOLLOWER: usize = 45;
const REG_CHAIN_DATA: usize = 46;
const REG_CHAIN_STATUS: usize = 47;

/// The description the image builder leaves in data memory.
const DEFAULT_INFO: usize = 0x0C00;
const INFO_ENTRY_TOKEN: usize = 0;
const INFO_LOGITS: usize = 2;
const INFO_LOGITS_EXPONENT: usize = 3;
const INFO_VOCABULARY: usize = 4;
const INFO_MAX_NEW: usize = 5;
const INFO_GREEDY: usize = 6;
const INFO_LIMIT: usize = 7;
const INFO_TEMPERATURE: usize = 8;
const INFO_SEED: usize = 9;
const INFO_ACTIVATION_BITS: usize = 10;
const INFO_PARAMETER_COUNT: usize = 11;
const INFO_RAW_TOKENS: usize = 12;
const INFO_STOP_COUNT: usize = 13;
const INFO_STOP_TABLE: usize = 14;
const INFO_PARAMETERS: usize = 15;

/// The tokenizer's tables, which the image builder leaves after the segment table.
const TOKENIZER: *const u32 = 0x1000_8000 as *const u32;

const END_OF_TOKENS: u32 = u32::MAX;
const ROUTINE_TOKEN: u32 = 0;
const ROUTINE_LOGITS: u32 = 1;

global_asm!(
    r#"
    .section .text.entry
    .global _start
_start:
    la sp, __stack_top
    la a0, __bss_start
    la a1, __bss_end
1:  bgeu a0, a1, 2f
    sw zero, 0(a0)
    addi a0, a0, 4
    j 1b
2:  call main
    ebreak
"#
);

/// Words of data memory of this chip's description and segment table, and chip count.
static mut INFO_WORD: usize = DEFAULT_INFO;
static mut TABLE_WORD: usize = DEFAULT_SEGMENT_TABLE;
static mut CHIPS: u32 = 1;

fn data_word(word: usize) -> u32 {
    unsafe { read_volatile(DATA_MEMORY.add(word)) }
}

fn info(field: usize) -> u32 {
    data_word(unsafe { INFO_WORD } + field)
}

fn chips() -> u32 {
    unsafe { CHIPS }
}

fn peripheral_write(register: usize, value: u32) {
    unsafe { write_volatile(PERIPHERALS.add(register), value) }
}

fn peripheral_read(register: usize) -> u32 {
    unsafe { read_volatile(PERIPHERALS.add(register)) }
}

fn receive() -> u8 {
    unsafe {
        while read_volatile(HOST_RX_STATUS) == 0 {}
        read_volatile(HOST_RX_DATA) as u8
    }
}

fn send(byte: u8) {
    unsafe {
        while read_volatile(HOST_TX_STATUS) == 0 {}
        write_volatile(HOST_TX_DATA, byte as u32);
    }
}

/// Loads every segment of the image; each CRC mismatch is reported to the host.
fn load_segments() {
    unsafe {
        let table = TABLE_WORD;
        let count = data_word(table) as usize;
        for index in 0..count {
            let field = |i: usize| data_word(table + 1 + index * SEGMENT_FIELDS + i);
            for register in 0..6 {
                write_volatile(LOADER.add(register), field(register));
            }
            write_volatile(LOADER.add(LOADER_START), 1);
            while read_volatile(LOADER.add(LOADER_BUSY)) != 0 {}
            let good = read_volatile(LOADER.add(LOADER_CRC)) == field(6);
            send(if good { b'.' } else { b'E' });
        }
    }
}

const COUNT_WORD: u32 = 0xC000_0000;
const TOTAL_WORD: u32 = 0xD000_0000;
const WORD_VALUE: u32 = 0x0FFF_FFFF;
/// How long a chip waits for a neighbour's ready signal before concluding it is alone.
const LONELY_POLLS: u32 = 32;

fn chain_connected() -> bool {
    (0..LONELY_POLLS).any(|_| peripheral_read(REG_CHAIN_STATUS) & 2 != 0)
}

/// Finds out how many chips there are and which one this is, around the chain.
fn count_chips() -> (u32, u32) {
    if peripheral_read(REG_FOLLOWER) != 0 {
        let id = chain_receive() & WORD_VALUE;
        chain_send(COUNT_WORD | (id + 1));
        let total = chain_receive();
        chain_send(total);
        return (id, total & WORD_VALUE);
    }
    if !chain_connected() {
        return (0, 1);
    }
    chain_send(COUNT_WORD | 1);
    let total = chain_receive() & WORD_VALUE;
    chain_send(TOTAL_WORD | total);
    chain_receive();
    (0, total)
}

/// Takes this chip's part of the image, if the image has a plan for this many chips.
fn select_part(id: u32, total: u32) -> bool {
    let plans = data_word(PLAN_DIRECTORY) as usize;
    if plans == 0 {
        return total == 1;
    }
    let lists = data_word(PLAN_DIRECTORY + 1) as usize;
    for plan in 0..plans {
        if data_word(lists + 2 * plan) == total {
            let parts = data_word(lists + 2 * plan + 1) as usize;
            unsafe {
                INFO_WORD = data_word(parts + 2 * id as usize) as usize;
                TABLE_WORD = data_word(parts + 2 * id as usize + 1) as usize;
                CHIPS = total;
            }
            return true;
        }
    }
    false
}

fn chain_send(word: u32) {
    while peripheral_read(REG_CHAIN_STATUS) & 2 == 0 {}
    peripheral_write(REG_CHAIN_DATA, word);
}

fn chain_receive() -> u32 {
    while peripheral_read(REG_CHAIN_STATUS) & 1 == 0 {}
    peripheral_read(REG_CHAIN_DATA)
}

/// Starts one routine of this chip's die, without waiting for it.
fn start_die(routine: u32, token: u32, position: u32) {
    for p in 0..info(INFO_PARAMETER_COUNT) as usize {
        let at = INFO_PARAMETERS + 7 * p;
        let window = info(at + 6);
        let windowed = if window > 0 { info(at + 4).wrapping_mul((position + 1).min(window)).wrapping_add(info(at + 5).wrapping_mul(position % window)) } else { 0 };
        let value = info(at + 1).wrapping_mul(token).wrapping_add(info(at + 2).wrapping_mul(position)).wrapping_add(info(at + 3)).wrapping_add(windowed);
        peripheral_write(info(at) as usize, value);
    }
    peripheral_write(REG_ENTRY, info(INFO_ENTRY_TOKEN + routine as usize));
}

fn wait_for_die() {
    while peripheral_read(REG_STATUS) == 0 {}
}

/// Orders a routine on every chip: the command goes down the chain before this die starts.
fn run_die(routine: u32, token: u32, position: u32) {
    if chips() > 1 {
        chain_send(routine);
        chain_send(token);
        chain_send(position);
    }
    start_die(routine, token, position);
    wait_for_die();
}

/// What every chip but the first does: obey the chain and pass the commands on.
fn serve_chain(id: u32) -> ! {
    let forward = id + 1 < chips();
    loop {
        let (routine, token, position) = (chain_receive(), chain_receive(), chain_receive());
        if forward {
            chain_send(routine);
            chain_send(token);
            chain_send(position);
        }
        start_die(routine, token, position);
        wait_for_die();
    }
}

/// Inverse temperature for the sampling unit: mantissa steps per table step.
fn sampling_inverse() -> u32 {
    peripheral_write(REG_VECTOR_ADDRESS, info(INFO_LOGITS_EXPONENT));
    let raw = peripheral_read(REG_VECTOR_DATA) as i32;
    let bits = info(INFO_ACTIVATION_BITS);
    let unused = 32 - bits;
    let exponent = (raw << unused) >> unused;
    let doubled = 17 + exponent - bits as i32;
    let scaled = if doubled >= 0 { (256u32 << doubled.min(16)) / info(INFO_TEMPERATURE).max(1) } else { (256u32 >> (-doubled).min(16)) / info(INFO_TEMPERATURE).max(1) };
    scaled.min(0xFFFF)
}

fn sample() -> u32 {
    peripheral_write(REG_SAMPLE_BASE, info(INFO_LOGITS));
    peripheral_write(REG_SAMPLE_COUNT, info(INFO_VOCABULARY));
    peripheral_write(REG_SAMPLE_INVERSE, sampling_inverse());
    peripheral_write(REG_SAMPLE_LIMIT, info(INFO_LIMIT));
    peripheral_write(REG_SAMPLE_GREEDY, info(INFO_GREEDY));
    peripheral_write(REG_SAMPLE_START, 1);
    while peripheral_read(REG_SAMPLE_DONE) == 0 {}
    peripheral_read(REG_SAMPLE_TOKEN)
}

fn tok(field: usize) -> u32 {
    unsafe { read_volatile(TOKENIZER.add(field)) }
}

fn tokenizer() -> Tokenizer<'static> {
    Tokenizer::new(unsafe { core::slice::from_raw_parts(TOKENIZER, tok(F_TEXT) as usize) })
}

/// Tokenizes the text in its buffer. Nothing is produced if it does not fit.
fn encode_text(length: usize, sink: &mut impl FnMut(u32)) {
    let text_at = tok(F_TEXT) as usize;
    let text = unsafe { core::slice::from_raw_parts(TOKENIZER.add(text_at) as *const u8, length) };
    let arena_at = text_at + (tok(F_TEXT_BYTES) as usize).div_ceil(4);
    let arena = unsafe { core::slice::from_raw_parts_mut(TOKENIZER.add(arena_at) as *mut u32, tok(F_ARENA_WORDS) as usize) };
    let _ = tokenizer().encode(text, arena, sink);
}

/// Reads the whole prompt from the host and hands over its tokens.
fn read_prompt(sink: &mut impl FnMut(u32)) {
    if info(INFO_RAW_TOKENS) != 0 {
        loop {
            let mut token = 0u32;
            for shift in 0..4 {
                token |= (receive() as u32) << (8 * shift);
            }
            if token == END_OF_TOKENS {
                return;
            }
            sink(token);
        }
    }
    let buffer = unsafe { core::slice::from_raw_parts_mut(TOKENIZER.add(tok(F_TEXT) as usize) as *mut u8, tok(F_TEXT_BYTES) as usize) };
    let (mut length, mut fits) = (0, true);
    loop {
        let byte = receive();
        if byte == 0 {
            break;
        }
        match buffer.get_mut(length) {
            Some(slot) => {
                *slot = byte;
                length += 1;
            }
            None => fits = false,
        }
    }
    if fits {
        encode_text(length, sink);
    }
}

fn emit(token: u32, decoding: &mut Decoding) {
    if info(INFO_RAW_TOKENS) != 0 {
        for shift in 0..4 {
            send((token >> (8 * shift)) as u8);
        }
        return;
    }
    tokenizer().write_token(token, decoding, &mut send);
}

#[unsafe(no_mangle)]
extern "C" fn main() {
    let (id, total) = count_chips();
    if !select_part(id, total) {
        if id == 0 {
            send(b'N');
        }
        return;
    }
    load_segments();
    if id != 0 {
        serve_chain(id);
    }
    peripheral_write(REG_SAMPLE_SEED, info(INFO_SEED));
    let (entry_token, entry_logits) = (ROUTINE_TOKEN, ROUTINE_LOGITS);
    let (mut last, mut position) = (0, 0);
    read_prompt(&mut |token| {
        run_die(entry_token, token, position);
        last = token;
        position += 1;
    });
    if position == 0 {
        return;
    }
    let stops = info(INFO_STOP_TABLE) as usize..(info(INFO_STOP_TABLE) + info(INFO_STOP_COUNT)) as usize;
    // Raw mode has no tokenizer in memory to ask.
    let mut decoding = if info(INFO_RAW_TOKENS) != 0 { Decoding::default() } else { tokenizer().decoding() };
    for _ in 0..info(INFO_MAX_NEW) {
        run_die(entry_logits, last, position - 1);
        last = sample();
        if stops.clone().any(|at| info(at) == last) {
            return;
        }
        emit(last, &mut decoding);
        run_die(entry_token, last, position);
        position += 1;
    }
}

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {
        unsafe { core::arch::asm!("ebreak") };
    }
}
