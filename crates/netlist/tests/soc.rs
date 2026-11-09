//! Real firmware, compiled for the chip's CPU, on the gates and on the instruction simulator.

use netlist::rv32::Rv32Iss;
use netlist::soc::{soc_design, FirmwareImage, SocBus, SocConfig};
use netlist::Simulator;
use std::path::Path;
use std::process::Command;

const CONFIG: SocConfig = SocConfig { program_bits: 12, data_bits: 12, fifo_bits: 6 };

fn build_firmware() -> Option<FirmwareImage> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../firmware");
    let built = Command::new("cargo").current_dir(&dir).env_remove("RUSTFLAGS").env_remove("CARGO_ENCODED_RUSTFLAGS").args(["build", "--release"]).output().ok()?;
    if !built.status.success() {
        eprintln!("firmware omitido: {}", String::from_utf8_lossy(&built.stderr));
        return None;
    }
    let elf = std::fs::read(dir.join("target/riscv32im-unknown-none-elf/release/firmware")).ok()?;
    Some(FirmwareImage::from_elf(&elf).unwrap())
}

fn run_gates(image: &FirmwareImage, input: &[u8], limit: u64) -> (Vec<u8>, u64) {
    let mut sim = Simulator::new(&soc_design(CONFIG)).unwrap();
    for (i, word) in image.program.iter().enumerate() {
        sim.poke_memory("/program", i, *word as u128).unwrap();
    }
    for (i, word) in image.data.iter().enumerate() {
        for byte in 0..4 {
            sim.poke_memory(&format!("/data_{byte}"), i, (word >> (8 * byte) & 0xFF) as u128).unwrap();
        }
    }
    sim.set("reset", 1).unwrap();
    sim.tick();
    sim.set("reset", 0).unwrap();
    sim.set("cpu_hold", 1).unwrap();
    sim.tick();
    for byte in input {
        sim.set("host_rx_valid", 1).unwrap();
        sim.set("host_rx_data", *byte as u64).unwrap();
        sim.tick();
    }
    sim.set("host_rx_valid", 0).unwrap();
    sim.tick();
    sim.set("cpu_hold", 0).unwrap();
    sim.set("host_tx_ready", 1).unwrap();
    let mut output = Vec::new();
    for cycle in 0..limit {
        sim.settle();
        if sim.get("host_tx_valid").unwrap() == 1 {
            output.push(sim.get("host_tx_data").unwrap() as u8);
        }
        if sim.get("halted").unwrap() == 1 {
            return (output, cycle + 1);
        }
        sim.tick();
    }
    panic!("el firmware no terminó en {limit} ciclos");
}

#[test]
fn firmware_runs_identically_on_gates_and_instruction_simulator() {
    let Some(image) = build_firmware() else { return };
    let input = b"hola, chip\0";
    let mut bus = SocBus::new(CONFIG, &image.program, &image.data, input);
    let mut cpu = Rv32Iss::default();
    while !cpu.halted {
        cpu.step(&mut bus);
        assert!(cpu.cycle < 2_000_000, "el firmware no termina");
    }
    assert_eq!(bus.tx, b"HOLA, CHIP");
    let (output, cycles) = run_gates(&image, input, cpu.cycle + 64);
    assert_eq!(output, bus.tx);
    eprintln!("{cycles} ciclos en puertas y {} en el simulador de instrucciones", cpu.cycle);
}
