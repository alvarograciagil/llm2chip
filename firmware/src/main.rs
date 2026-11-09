//! Firmware of the control CPU. Talks to the outside world only through the host port.
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

#[unsafe(no_mangle)]
extern "C" fn main() {
    load_segments();
    loop {
        let byte = receive();
        if byte == 0 {
            return;
        }
        send(byte.to_ascii_uppercase());
    }
}

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {
        unsafe { core::arch::asm!("ebreak") };
    }
}
