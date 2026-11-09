//! Firmware of the control CPU. Talks to the outside world only through the host port.
#![no_std]
#![no_main]

use core::arch::global_asm;
use core::ptr::{read_volatile, write_volatile};

const HOST_RX_DATA: *const u32 = 0x2000_0000 as *const u32;
const HOST_RX_STATUS: *const u32 = 0x2000_0004 as *const u32;
const HOST_TX_DATA: *mut u32 = 0x2000_0008 as *mut u32;
const HOST_TX_STATUS: *const u32 = 0x2000_000C as *const u32;

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

#[unsafe(no_mangle)]
extern "C" fn main() {
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
