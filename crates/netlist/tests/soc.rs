//! Real firmware, compiled for the chip's CPU, on the gates and on the instruction simulator.

use netlist::rv32::Rv32Iss;
use netlist::image::{build_image, Segment, Target};
use netlist::soc::{soc_design, FirmwareImage, LoadWrite, SocBus, SocConfig};
use netlist::Simulator;
use std::path::Path;
use std::process::Command;

const CONFIG: SocConfig = SocConfig { program_words: netlist::soc::PROGRAM_WORDS, data_bits: 13, fifo_bits: 6, links: netlist::soc::HostLinks { uart: None, qspi: false, i2c: None } };

/// Built once per test run, so that tests running together do not race on the binary.
fn build_firmware() -> Option<FirmwareImage> {
    static FIRMWARE: std::sync::OnceLock<Option<FirmwareImage>> = std::sync::OnceLock::new();
    FIRMWARE
        .get_or_init(|| {
            let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../firmware");
            let built = Command::new("cargo").current_dir(&dir).env_remove("RUSTFLAGS").env_remove("CARGO_ENCODED_RUSTFLAGS").args(["build", "--release"]).output().ok()?;
            if !built.status.success() {
                eprintln!("firmware skipped: {}", String::from_utf8_lossy(&built.stderr));
                return None;
            }
            let elf = std::fs::read(dir.join("target/riscv32im-unknown-none-elf/release/firmware")).ok()?;
            Some(FirmwareImage::from_elf(&elf).unwrap())
        })
        .clone()
}

struct GateRun {
    output: Vec<u8>,
    writes: Vec<LoadWrite>,
    cycles: u64,
}

/// Runs from power-up, answering the image port the cycle after each request.
fn run_gates(image: &[u32], input: &[u8], limit: u64) -> GateRun {
    let mut sim = Simulator::new(&soc_design(CONFIG)).unwrap();
    sim.set("reset", 1).unwrap();
    sim.set("cpu_hold", 1).unwrap();
    sim.tick();
    sim.set("reset", 0).unwrap();
    let mut pending = None;
    let serve = |sim: &mut Simulator, pending: &mut Option<u32>| {
        sim.set("image_resp_valid", pending.is_some() as u64).unwrap();
        sim.set("image_resp_data", pending.unwrap_or(0) as u64).unwrap();
        sim.settle();
        *pending = (sim.get("image_req_valid").unwrap() == 1).then(|| image[sim.get("image_req_addr").unwrap() as usize]);
    };
    for byte in input {
        sim.set("host_rx_valid", 1).unwrap();
        sim.set("host_rx_data", *byte as u64).unwrap();
        serve(&mut sim, &mut pending);
        sim.tick();
    }
    sim.set("host_rx_valid", 0).unwrap();
    let mut boot = 0;
    loop {
        serve(&mut sim, &mut pending);
        if sim.get("booting").unwrap() == 0 {
            break;
        }
        sim.tick();
        boot += 1;
        assert!(boot < 1_000_000, "boot does not finish");
    }
    sim.set("cpu_hold", 0).unwrap();
    sim.set("host_tx_ready", 1).unwrap();

    let mut run = GateRun { output: Vec::new(), writes: Vec::new(), cycles: 0 };
    for cycle in 0..limit {
        serve(&mut sim, &mut pending);
        if sim.get("host_tx_valid").unwrap() == 1 {
            run.output.push(sim.get("host_tx_data").unwrap() as u8);
        }
        if sim.get("load_valid").unwrap() == 1 {
            let words = sim.get_words("load_data").unwrap();
            run.writes.push(LoadWrite { target: sim.get("load_target").unwrap() as u32, unit: sim.get("load_unit").unwrap() as u32, address: sim.get("load_addr").unwrap() as u32, words });
        }
        if sim.get("halted").unwrap() == 1 {
            run.cycles = cycle + 1;
            return run;
        }
        sim.tick();
    }
    panic!("the firmware did not finish within {limit} cycles");
}

fn run_iss(image: &[u32], input: &[u8]) -> (SocBus, u64) {
    let mut bus = SocBus::booted(CONFIG, image, input).unwrap();
    let mut cpu = Rv32Iss::default();
    while !cpu.halted {
        cpu.step(&mut bus);
        assert!(cpu.cycle < 5_000_000, "the firmware does not finish");
    }
    (bus, cpu.cycle)
}

fn segments() -> Vec<Segment> {
    vec![
        Segment { target: Target::Vector, unit: 0, address: 16, beats: 1, words: vec![0x11, 0x22, 0x33] },
        Segment { target: Target::Microcode, unit: 0, address: 0, beats: 3, words: (0..12).map(|i| 0xA000_0000 | i).collect() },
        Segment { target: Target::CellWeight, unit: 7, address: 4, beats: 1, words: (0..9).map(|i| i * i).collect() },
    ]
}

/// Boots, reports the segments it loaded and the ones that failed, and stops at an empty
/// question.
fn compare(image: &[u32], loaded: usize, failed: usize) {
    let input = netlist::port::tokens(&[]);
    let (bus, cycles) = run_iss(image, &input);
    assert_eq!(netlist::port::boot(&bus.tx), Some((loaded, failed)), "{:?}", netlist::port::events(&bus.tx));
    assert_eq!(netlist::port::events(&bus.tx).last(), Some(&netlist::port::Event::Ready { capacity: 0 }));
    let run = run_gates(image, &input, cycles + 64);
    assert_eq!(run.output, bus.tx, "the gates' output");
    assert_eq!(run.writes.len(), bus.writes.len());
    for (gate, model) in run.writes.iter().zip(&bus.writes) {
        assert_eq!((gate.target, gate.unit, gate.address), (model.target, model.unit, model.address));
        assert_eq!(gate.words[..model.words.len()], model.words[..]);
    }
    assert_eq!(run.cycles, cycles, "cycles");
    eprintln!("{cycles} cycles in gates and in the instruction-level simulator");
}

#[test]
fn the_chip_boots_from_its_image_and_loads_every_segment() {
    let Some(firmware) = build_firmware() else { return };
    let image = build_image(&firmware.program, &firmware.data, &[], &[], &segments()).unwrap();
    compare(&image, 3, 0);
}

#[test]
fn a_corrupted_segment_is_reported() {
    let Some(firmware) = build_firmware() else { return };
    let mut image = build_image(&firmware.program, &firmware.data, &[], &[], &segments()).unwrap();
    let last = image.len() - 1;
    image[last] ^= 0x10;
    compare(&image, 3, 1);
}
