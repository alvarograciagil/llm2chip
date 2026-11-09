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

fn soc(gen: &mut Gen, config: SocConfig) -> String {
    let cpu = core(gen);
    let queue = fifo(gen, config.fifo_bits);
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
    b.or_to(reset, cpu_hold, cpu_reset);
    b.inst("cpu", &cpu, &[("reset", vec![cpu_reset]), ("imem_rdata", imem_rdata.clone()), ("dmem_rdata", dmem_rdata.clone()), ("imem_addr", imem_addr.clone()), ("dmem_addr", dmem_addr.clone()), ("dmem_wdata", dmem_wdata.clone()), ("dmem_we", dmem_we.clone()), ("dmem_re", vec![dmem_re]), ("rf_we", rf_we), ("rf_addr", rf_addr), ("rf_wdata", rf_wdata), ("halted", vec![halted])]);

    b.group("Memoria de programa");
    b.sram("program", config.program_bits, 32, &[("addr", imem_addr[2..2 + config.program_bits as usize].to_vec()), ("we", vec![Bit::Zero]), ("wdata", constant(0, 32)), ("rdata", imem_rdata.clone())]);

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

    let (ram_we, rx_pop, tx_push, read_data);
    {
        let mut l = Logic { b: &mut b, next: 0 };
        l.b.group("Bus interno");
        let top = dmem_addr[28..32].to_vec();
        let is_ram = l.equals(&top, RAM_BASE >> 28);
        let is_host = l.equals(&top, HOST_BASE >> 28);
        let register = &dmem_addr[2..4];
        let (reg_rx_data, reg_rx_status, reg_tx_data, reg_tx_status) = (l.equals(register, 0), l.equals(register, 1), l.equals(register, 2), l.equals(register, 3));
        let any_write = l.any(&dmem_we);
        let host_read = l.and(is_host, dmem_re);
        let host_write = l.and(is_host, any_write);
        ram_we = dmem_we.iter().map(|w| l.and(*w, is_ram)).collect::<Vec<Bit>>();
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
        l.connect_flops(&[is_ram, want_rx_data, want_rx_status, want_tx_status, status_rx_now, status_tx_now], &[pick_ram, pick_rx_data, pick_rx_status, pick_tx_status, status_rx, status_tx]);
        let rx_word: Vec<Bit> = rx_out.iter().copied().chain(std::iter::repeat_n(Bit::Zero, 24)).collect();
        let rx_status_word: Vec<Bit> = std::iter::once(status_rx).chain(std::iter::repeat_n(Bit::Zero, 31)).collect();
        let tx_status_word: Vec<Bit> = std::iter::once(status_tx).chain(std::iter::repeat_n(Bit::Zero, 31)).collect();
        let zero = constant(0, 32);
        let mut value = l.mux_bus(pick_ram, &zero, &ram_data);
        value = l.mux_bus(pick_rx_data, &value, &rx_word);
        value = l.mux_bus(pick_rx_status, &value, &rx_status_word);
        value = l.mux_bus(pick_tx_status, &value, &tx_status_word);
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
    let address = dmem_addr[2..2 + config.data_bits as usize].to_vec();
    for byte in 0..4usize {
        b.sram(&format!("data_{byte}"), config.data_bits, 8, &[("addr", address.clone()), ("we", vec![ram_we[byte]]), ("wdata", dmem_wdata[8 * byte..8 * byte + 8].to_vec()), ("rdata", ram_data[8 * byte..8 * byte + 8].to_vec())]);
    }

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

/// The same machine at instruction level, for firmware runs too long for gates.
pub struct SocBus {
    pub program: Vec<u32>,
    pub ram: Vec<u32>,
    pub rx: VecDeque<u8>,
    pub tx: Vec<u8>,
}

impl SocBus {
    pub fn new(config: SocConfig, program: &[u32], data: &[u32], input: &[u8]) -> Self {
        let mut image = vec![0; 1 << config.program_bits];
        image[..program.len()].copy_from_slice(program);
        let mut ram = vec![0; 1 << config.data_bits];
        ram[..data.len()].copy_from_slice(data);
        Self { program: image, ram, rx: input.iter().copied().collect(), tx: Vec::new() }
    }
}

impl Rv32Bus for SocBus {
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
            _ => {}
        }
    }
}
