//! The host port's physical links in gates: the byte port, a serial port and a quad SPI
//! target, each answered on its own, with the real firmware.

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
        let mut board = Self { sim, image, pending: None, cycle: 0, bytes_in: VecDeque::new(), bytes_out: Vec::new(), uart_in: VecDeque::new(), uart_out: Vec::new(), listening: None, last_line: true };
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
        sim.settle();
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
    let config = SocConfig { program_words: netlist::soc::PROGRAM_WORDS, data_bits: 13, fifo_bits: 6, links: HostLinks { uart: Some(DIVIDER), qspi: true } };
    let image = build_image(&firmware.program, &firmware.data, &[], &[], &[]).unwrap();
    let mut board = Board::new(config, image);
    // The CPU clears its memory first, then reports the boot.
    board.run(80_000);

    // Before any message the chip speaks on its first serial link, the quad SPI one: its
    // interrupt line says it has something, and a read brings the boot and the ready.
    assert_eq!(board.pin("qspi_irq"), 1, "bytes {:?}, serial {:?}", board.bytes_out, board.uart_out);
    let status = board.qspi(QSPI_STATUS, &[], 2);
    assert!(status[1] > 0, "bytes waiting: {status:?}");
    let first = board.qspi(QSPI_READ, &[], status[1] as usize + 4);
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
    let waiting = board.qspi(QSPI_STATUS, &[], 2)[1] as usize;
    let answer = board.qspi(QSPI_READ, &[], waiting + 2);
    assert_eq!(port::events(&answer), [Event::Options(vec![(9, 0x1234_C0DB)])], "escapes both ways: {answer:?}");
    assert_eq!(port::events(&board.uart_out).len(), 1, "the serial port heard nothing more");

    // A serial byte that comes while the byte port is inside a message is dropped and
    // counted; the byte port's message goes on.
    board.bytes_in.extend([port::kind::STATUS]);
    board.run(100);
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
}
