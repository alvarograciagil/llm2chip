//! The host port's physical links in gates: the byte port, a serial port, a quad SPI
//! target and an I2C target, each answered on its own, with the real firmware.

use netlist::image::build_image;
use netlist::port::{self, Event};
use netlist::soc::{soc_design, FirmwareImage, HostLinks, SocConfig, QSPI_READ, QSPI_STATUS, QSPI_WRITE};
use netlist::Simulator;
use std::collections::VecDeque;
use std::path::Path;
use std::process::Command;

/// Clocks a bit on the serial port, and the quad SPI clock's period in the chip's clocks.
const DIVIDER: u32 = 8;
const SCK: u64 = 16;
/// The I2C target's address, and half its clock's period in the chip's clocks.
const ADDRESS: u8 = 0x42;
const HALF: u64 = 16;

fn firmware() -> Option<FirmwareImage> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../firmware");
    let built = Command::new("cargo").current_dir(&dir).env_remove("RUSTFLAGS").env_remove("CARGO_ENCODED_RUSTFLAGS").args(["build", "--release"]).output().ok()?;
    if !built.status.success() {
        eprintln!("firmware skipped: {}", String::from_utf8_lossy(&built.stderr));
        return None;
    }
    FirmwareImage::from_elf(&std::fs::read(dir.join("target/riscv32im-unknown-none-elf/release/firmware")).ok()?).ok()
}

/// The chip and the three hosts around it, a cycle at a time.
struct Board {
    sim: Simulator,
    image: Vec<u32>,
    pending: Option<u32>,
    cycle: u64,
    /// The byte port: what is to go in, and what came out.
    bytes_in: VecDeque<u8>,
    bytes_out: Vec<u8>,
    /// The serial port: the line's levels to come, one a clock, and what it heard.
    uart_in: VecDeque<bool>,
    uart_out: Vec<u8>,
    listening: Option<(u64, u8, u32)>,
    last_line: bool,
    /// The I2C bus: the host's clock, whether it pulls the data line down, and whether the
    /// chip did, both pulling it as an open drain does.
    scl: bool,
    host_low: bool,
    chip_low: bool,
}

impl Board {
    fn new(config: SocConfig, image: Vec<u32>) -> Self {
        let mut sim = Simulator::new(&soc_design(config)).unwrap();
        sim.set("reset", 1).unwrap();
        sim.set("cpu_hold", 1).unwrap();
        sim.set("uart_rx", 1).unwrap();
        sim.set("qspi_cs_n", 1).unwrap();
        sim.tick();
        sim.set("reset", 0).unwrap();
        let mut board = Self { sim, image, pending: None, cycle: 0, bytes_in: VecDeque::new(), bytes_out: Vec::new(), uart_in: VecDeque::new(), uart_out: Vec::new(), listening: None, last_line: true, scl: true, host_low: false, chip_low: false };
        while board.sim.get("booting").unwrap_or(1) == 1 || board.cycle == 0 {
            board.step();
        }
        board.sim.set("cpu_hold", 0).unwrap();
        board
    }

    fn step(&mut self) {
        let sim = &mut self.sim;
        sim.set("image_resp_valid", self.pending.is_some() as u64).unwrap();
        sim.set("image_resp_data", self.pending.unwrap_or(0) as u64).unwrap();
        sim.set("host_tx_ready", 1).unwrap();
        sim.set("host_rx_valid", !self.bytes_in.is_empty() as u64).unwrap();
        sim.set("host_rx_data", self.bytes_in.front().copied().unwrap_or(0) as u64).unwrap();
        sim.set("uart_rx", self.uart_in.pop_front().unwrap_or(true) as u64).unwrap();
        sim.set("i2c_scl", self.scl as u64).unwrap();
        sim.set("i2c_sda", !(self.host_low || self.chip_low) as u64).unwrap();
        sim.settle();
        self.chip_low = sim.get("i2c_sda_low").unwrap() == 1;
        self.pending = (sim.get("image_req_valid").unwrap() == 1).then(|| self.image[sim.get("image_req_addr").unwrap() as usize]);
        if !self.bytes_in.is_empty() && sim.get("host_rx_ready").unwrap() == 1 {
            self.bytes_in.pop_front();
        }
        if sim.get("host_tx_valid").unwrap() == 1 {
            self.bytes_out.push(sim.get("host_tx_data").unwrap() as u8);
        }
        // The serial line: a start bit's fall, then each bit at its middle.
        let line = sim.get("uart_tx").unwrap() == 1;
        match self.listening {
            None if self.last_line && !line => self.listening = Some((self.cycle, 0, 0)),
            Some((start, byte, bits)) => {
                let at = self.cycle - start;
                let middle = DIVIDER as u64 / 2 + (bits as u64 + 1) * DIVIDER as u64;
                if at == middle {
                    if bits < 8 {
                        self.listening = Some((start, byte | (line as u8) << bits, bits + 1));
                    } else {
                        assert!(line, "a stop bit");
                        self.uart_out.push(byte);
                        self.listening = None;
                    }
                }
            }
            None => {}
        }
        self.last_line = line;
        sim.tick();
        self.cycle += 1;
    }

    fn run(&mut self, cycles: u64) {
        for _ in 0..cycles {
            self.step();
        }
    }

    /// An output pin, settled.
    fn pin(&mut self, name: &str) -> u64 {
        self.sim.settle();
        self.sim.get(name).unwrap()
    }

    /// Sends bytes on the serial port: a start bit, the byte least significant first, a stop
    /// bit.
    fn uart_send(&mut self, bytes: &[u8]) {
        for byte in bytes {
            let bits = [false].into_iter().chain((0..8).map(|i| byte >> i & 1 == 1)).chain([true]);
            for bit in bits {
                self.uart_in.extend(std::iter::repeat_n(bit, DIVIDER as usize));
            }
        }
    }

    /// The I2C data line as the host sees it.
    fn sda(&self) -> bool {
        !(self.host_low || self.chip_low)
    }

    /// An I2C start and the address byte, `read` or not; whether the chip acknowledged.
    fn i2c_start(&mut self, read: bool) -> bool {
        self.host_low = true;
        self.run(HALF);
        self.scl = false;
        self.i2c_write(ADDRESS << 1 | read as u8)
    }

    /// One byte to the chip, the clock low on entry and on leaving; whether it acknowledged.
    fn i2c_write(&mut self, byte: u8) -> bool {
        for bit in (0..8).rev() {
            self.host_low = byte >> bit & 1 == 0;
            self.clock_pulse();
        }
        self.host_low = false;
        self.run(HALF);
        self.scl = true;
        self.run(HALF / 2);
        let acknowledged = !self.sda();
        self.run(HALF / 2);
        self.scl = false;
        acknowledged
    }

    /// One byte from the chip, then the host's acknowledgement, or not for the last.
    fn i2c_read(&mut self, more: bool) -> u8 {
        self.host_low = false;
        let mut byte = 0;
        for _ in 0..8 {
            self.run(HALF);
            self.scl = true;
            self.run(HALF / 2);
            byte = byte << 1 | self.sda() as u8;
            self.run(HALF / 2);
            self.scl = false;
        }
        self.host_low = more;
        self.clock_pulse();
        self.host_low = false;
        byte
    }

    /// Half a period low with the data set, then half high.
    fn clock_pulse(&mut self) {
        self.run(HALF);
        self.scl = true;
        self.run(HALF);
        self.scl = false;
    }

    fn i2c_stop(&mut self) {
        self.host_low = true;
        self.run(HALF);
        self.scl = true;
        self.run(HALF);
        self.host_low = false;
        self.run(HALF);
    }

    /// Writes bytes to the chip in one transaction; whether it took each.
    fn i2c_send(&mut self, bytes: &[u8]) -> Vec<bool> {
        assert!(self.i2c_start(false), "the chip answers to its address");
        let taken = bytes.iter().map(|b| self.i2c_write(*b)).collect();
        self.i2c_stop();
        taken
    }

    /// Reads `count` bytes from the chip in one transaction: the room, the waiting, then
    /// what it sends.
    fn i2c_receive(&mut self, count: usize) -> Vec<u8> {
        assert!(self.i2c_start(true));
        let got = (0..count).map(|i| self.i2c_read(i + 1 < count)).collect();
        self.i2c_stop();
        got
    }

    /// Reads what waits over I2C, no more than the waiting count each time, until nothing
    /// does: an escaped byte takes one entry and two bytes, so it can take more than one read.
    fn i2c_all(&mut self) -> Vec<u8> {
        let mut got = Vec::new();
        loop {
            let [room, waiting] = self.i2c_receive(2)[..] else { unreachable!() };
            assert_eq!(room, 64, "the receive queue is empty");
            if waiting == 0 {
                return got;
            }
            let read = self.i2c_receive(2 + waiting as usize);
            got.extend(&read[2..]);
        }
    }

    /// The same over quad SPI: the status, then a read of what waits.
    fn qspi_receive(&mut self) -> Vec<u8> {
        let mut got = Vec::new();
        loop {
            let waiting = self.qspi(QSPI_STATUS, &[], 2)[1] as usize;
            if waiting == 0 {
                return got;
            }
            got.extend(self.qspi(QSPI_READ, &[], waiting));
        }
    }

    /// One quad SPI transaction: the command, the bytes to write, and for a read or the
    /// status the dummy byte and `read` bytes back.
    fn qspi(&mut self, command: u8, write: &[u8], read: usize) -> Vec<u8> {
        self.sim.set("qspi_cs_n", 0).unwrap();
        self.run(SCK);
        let mut out = Vec::new();
        let reading = command == QSPI_READ || command == QSPI_STATUS;
        let sent: Vec<u8> = [command].into_iter().chain(write.iter().copied()).chain(reading.then_some(0)).collect();
        for byte in sent.iter().copied().chain(std::iter::repeat_n(0, read)) {
            let mut got = 0u8;
            for nibble in [byte >> 4, byte & 15] {
                self.sim.set("qspi_sck", 0).unwrap();
                self.sim.set("qspi_in", nibble as u64).unwrap();
                self.run(SCK / 2);
                self.sim.set("qspi_sck", 1).unwrap();
                self.sim.settle();
                got = got << 4 | self.sim.get("qspi_out").unwrap() as u8;
                self.run(SCK / 2);
            }
            out.push(got);
        }
        self.sim.set("qspi_sck", 0).unwrap();
        self.run(SCK);
        self.sim.set("qspi_cs_n", 1).unwrap();
        self.run(SCK);
        out.split_off(sent.len())
    }
}

#[test]
fn each_link_is_answered_on_its_own() {
    let Some(firmware) = firmware() else { return };
    let config = SocConfig { program_words: netlist::soc::PROGRAM_WORDS, data_bits: 13, fifo_bits: 6, links: HostLinks { uart: Some(DIVIDER), qspi: true, i2c: Some(ADDRESS) } };
    let image = build_image(&firmware.program, &firmware.data, &[], &[], &[]).unwrap();
    let mut board = Board::new(config, image);
    // The CPU clears its memory first, then reports the boot.
    board.run(80_000);

    // Before any message the chip speaks on its first serial link, the quad SPI one: its
    // interrupt line says it has something, and a read brings the boot and the ready.
    assert_eq!(board.pin("qspi_irq"), 1, "bytes {:?}, serial {:?}", board.bytes_out, board.uart_out);
    let status = board.qspi(QSPI_STATUS, &[], 2);
    assert!(status[1] > 0, "bytes waiting: {status:?}");
    let first = board.qspi_receive();
    let events = port::events(&first);
    assert!(matches!(events[..], [Event::Boot { .. }, Event::Ready { capacity: 0 }]), "{events:?}");
    assert_eq!(board.pin("qspi_irq"), 0, "nothing left to send");

    // A question on the serial port is answered there.
    board.uart_send(&port::status());
    board.run(40_000);
    assert_eq!(port::events(&board.uart_out), [Event::Status { answering: false, position: 0, capacity: 0 }]);

    // On the byte port, there.
    board.bytes_in.extend(port::options());
    board.run(20_000);
    assert!(matches!(port::events(&board.bytes_out)[..], [Event::Options(_)]), "{:?}", port::events(&board.bytes_out));

    // On the quad SPI link, written and then read, there.
    let room = board.qspi(QSPI_STATUS, &[], 2)[0];
    assert_eq!(room, 64, "the receive queue is empty");
    board.qspi(QSPI_WRITE, &port::set(9, 0x1234_C0DB), 0);
    board.run(20_000);
    assert_eq!(board.pin("qspi_irq"), 1, "bytes {:?}, serial {:?}", board.bytes_out, board.uart_out);
    let answer = board.qspi_receive();
    assert_eq!(port::events(&answer), [Event::Options(vec![(9, 0x1234_C0DB)])], "escapes both ways: {answer:?}");
    assert_eq!(port::events(&board.uart_out).len(), 1, "the serial port heard nothing more");

    // A serial byte that comes while the byte port is inside a message is dropped and
    // counted; the byte port's message goes on.
    assert_eq!(board.pin("uart_cts_n"), 0, "the serial port may send");
    board.bytes_in.extend([port::kind::STATUS]);
    board.run(100);
    assert_eq!(board.pin("uart_cts_n"), 1, "another link holds the port");
    board.uart_send(&[0x55]);
    let mut dropped = 0;
    for _ in 0..DIVIDER * 12 {
        board.step();
        dropped += board.sim.get("host_dropped").unwrap();
    }
    assert_eq!(dropped, 1);
    board.bytes_in.extend([netlist::soc::FRAME_END]);
    board.run(20_000);
    assert!(matches!(port::events(&board.bytes_out)[..], [Event::Options(_), Event::Status { .. }]));

    // On the I2C target, there. A read starts with the room and what waits; a byte to
    // another address is not acknowledged.
    board.i2c_stop();
    assert_eq!(board.i2c_receive(2), [64, 0]);
    board.host_low = true;
    board.run(HALF);
    board.scl = false;
    assert!(!board.i2c_write((ADDRESS + 1) << 1), "another chip's address");
    board.i2c_stop();
    let message = port::set(5, 0xDB_C0_00_07);
    assert!(board.i2c_send(&message).iter().all(|t| *t));
    board.run(20_000);
    assert_eq!(board.pin("i2c_irq"), 1);
    let read = board.i2c_all();
    assert_eq!(port::events(&read), [Event::Options(vec![(5, 0xDB_C0_00_07)])], "{read:?}");
    assert_eq!(board.pin("i2c_irq"), 0);
    assert_eq!(port::events(&board.uart_out).len(), 1, "the serial port heard nothing more");

    // While the byte port is inside a message, the I2C target refuses, and the host can
    // send the byte again afterwards.
    board.bytes_in.extend([port::kind::STATUS]);
    board.run(100);
    assert_eq!(board.i2c_send(&[port::kind::OPTIONS]), [false]);
    board.bytes_in.extend([netlist::soc::FRAME_END]);
    board.run(20_000);
    assert_eq!(port::events(&board.bytes_out).len(), 3);
    assert_eq!(board.i2c_send(&port::options()), [true, true]);
    board.run(20_000);
    let read = board.i2c_all();
    assert!(matches!(port::events(&read)[..], [Event::Options(_)]), "{read:?} bytes {:?}", port::events(&board.bytes_out));

    // The serial port's flow control: with the CPU held, the receive queue fills, and the
    // clear-to-send line rises once less than a quarter of it is free, at 48 entries of 64.
    board.sim.set("cpu_hold", 1).unwrap();
    let stream = port::stream(&[0x0101_0101; 12]);
    board.uart_send(&stream[..47]);
    board.run(47 * 10 * DIVIDER as u64 + 4 * DIVIDER as u64);
    assert_eq!(board.pin("uart_cts_n"), 0, "47 entries in");
    board.uart_send(&stream[47..48]);
    board.run(12 * DIVIDER as u64);
    assert_eq!(board.pin("uart_cts_n"), 1, "48 entries in: wait");
    // Freed, the CPU drains it, and the host may send again.
    board.sim.set("cpu_hold", 0).unwrap();
    board.uart_send(&stream[48..]);
    board.run(40_000);
    assert_eq!(board.pin("uart_cts_n"), 0);
}
