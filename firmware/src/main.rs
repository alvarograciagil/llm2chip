//! Firmware of the control CPU: loads the model, reads text, runs the die and writes text.
//! It talks to the outside world only through the host port.
#![no_std]
#![no_main]

use core::arch::global_asm;
use core::ptr::{read_volatile, write_volatile};

const HOST_RX_DATA: *const u32 = 0x2000_0000 as *const u32;
const HOST_RX_STATUS: *const u32 = 0x2000_0004 as *const u32;
const HOST_TX_DATA: *mut u32 = 0x2000_0008 as *mut u32;
const HOST_TX_STATUS: *const u32 = 0x2000_000C as *const u32;

const LOADER: *mut u32 = 0x3000_0000 as *mut u32;
const LOADER_START: usize = 6;
const LOADER_BUSY: usize = 7;
const LOADER_CRC: usize = 8;
const SEGMENT_TABLE: *const u32 = 0x1000_4000 as *const u32;
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

/// The description the image builder leaves in data memory.
const INFO: *const u32 = 0x1000_3000 as *const u32;
const INFO_ENTRY_TOKEN: usize = 0;
const INFO_ENTRY_LOGITS: usize = 1;
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
const INFO_STOP_FIRST: usize = 13;
const INFO_STOP_SECOND: usize = 14;
const INFO_PARAMETERS: usize = 15;
const INFO_BYTE_TO_TOKEN: usize = INFO_PARAMETERS + 4 * 16;
const INFO_TOKEN_TO_BYTE: usize = INFO_BYTE_TO_TOKEN + 256;

const END_OF_TOKENS: u32 = u32::MAX;

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

fn info(field: usize) -> u32 {
    unsafe { read_volatile(INFO.add(field)) }
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
        let count = read_volatile(SEGMENT_TABLE) as usize;
        for index in 0..count {
            let field = |i: usize| read_volatile(SEGMENT_TABLE.add(1 + index * SEGMENT_FIELDS + i));
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

/// Runs one routine of the die for this token and position.
fn run_die(entry: u32, token: u32, position: u32) {
    for p in 0..info(INFO_PARAMETER_COUNT) as usize {
        let at = INFO_PARAMETERS + 4 * p;
        let value = info(at + 1).wrapping_mul(token).wrapping_add(info(at + 2).wrapping_mul(position)).wrapping_add(info(at + 3));
        peripheral_write(info(at) as usize, value);
    }
    peripheral_write(REG_ENTRY, entry);
    while peripheral_read(REG_STATUS) == 0 {}
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

/// The next token from the host, or `None` once the prompt is over.
fn next_token() -> Option<u32> {
    if info(INFO_RAW_TOKENS) != 0 {
        let mut token = 0u32;
        for shift in 0..4 {
            token |= (receive() as u32) << (8 * shift);
        }
        return (token != END_OF_TOKENS).then_some(token);
    }
    let byte = receive();
    (byte != 0).then(|| info(INFO_BYTE_TO_TOKEN + byte as usize))
}

fn emit(token: u32) {
    if info(INFO_RAW_TOKENS) != 0 {
        for shift in 0..4 {
            send((token >> (8 * shift)) as u8);
        }
    } else {
        send(info(INFO_TOKEN_TO_BYTE + token as usize) as u8);
    }
}

#[unsafe(no_mangle)]
extern "C" fn main() {
    load_segments();
    peripheral_write(REG_SAMPLE_SEED, info(INFO_SEED));
    let (entry_token, entry_logits) = (info(INFO_ENTRY_TOKEN), info(INFO_ENTRY_LOGITS));
    let (mut last, mut position) = (0, 0);
    while let Some(token) = next_token() {
        run_die(entry_token, token, position);
        last = token;
        position += 1;
    }
    if position == 0 {
        return;
    }
    let (first_stop, second_stop) = (info(INFO_STOP_FIRST), info(INFO_STOP_SECOND));
    for _ in 0..info(INFO_MAX_NEW) {
        run_die(entry_logits, last, position - 1);
        last = sample();
        if last == first_stop || last == second_stop {
            return;
        }
        emit(last);
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
