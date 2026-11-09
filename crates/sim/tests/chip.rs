//! The whole chip, from bytes in to bytes out: real firmware, the load controller,
//! a fixed microprogram, the die and the sampling unit.

use compiler::compile_qwen3_5;
use netlist::chip::{chip_design, chip_info, ChipConfig, ChipModel, GenerationSetup};
use netlist::image::{build_image, Segment, Target};
use netlist::rv32::Rv32Iss;
use netlist::sampler::{exponential_table, SamplerConfig, SamplerModel};
use netlist::soc::{FirmwareImage, SocBus, SocConfig};
use netlist::Simulator;
use sim::chip::{image_segments, VmEngine};
use sim::lower::Layout;
use sim::tokenizer::TokenizerTables;
use sim::Weights;
use std::path::Path;
use std::process::Command;

mod common;
use common::*;

const CHIP_TOKENS: usize = 8;

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

fn run_gates(design: &netlist::Design, image: &[u32], input: &[u8], limit: u64) -> (Vec<u8>, u64, Simulator) {
    let mut sim = Simulator::new(design).unwrap();
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
        assert!(boot < 1_000_000, "el arranque no termina");
    }
    sim.set("cpu_hold", 0).unwrap();
    sim.set("host_tx_ready", 1).unwrap();
    let mut output = Vec::new();
    for cycle in 0..limit {
        serve(&mut sim, &mut pending);
        if sim.get("host_tx_valid").unwrap() == 1 {
            output.push(sim.get("host_tx_data").unwrap() as u8);
        }
        if sim.get("halted").unwrap() == 1 {
            return (output, cycle + 1, sim);
        }
        sim.tick();
    }
    panic!("el firmware no terminó en {limit} ciclos");
}

#[test]
#[cfg_attr(debug_assertions, ignore = "solo tiene sentido con optimizaciones: son cientos de miles de ciclos")]
fn text_goes_through_the_whole_chip_in_gates() {
    let Some(firmware) = build_firmware() else { return };
    let dir = std::env::temp_dir().join(format!("llm2chip-chip-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    make_checkpoint(&dir);
    let program = compile_qwen3_5(CONFIG).unwrap();

    let die = netlist::DieConfig { microcode_bits: 15, ..die_config() };
    let config = ChipConfig { soc: SocConfig { program_bits: 12, data_bits: 14, fifo_bits: 6 }, die, sampler: SamplerConfig { logit_bits: 16, index_bits: 12, entries: 4, lanes: 1 } };
    let layout = Layout::new(&program, &Weights::open(&dir).unwrap(), die, Some(CHIP_TOKENS)).unwrap();
    let code = layout.microprogram(&program).unwrap();
    let setup = GenerationSetup { max_new: 2, greedy: false, limit: 4, temperature: 256, seed: 0x1234_5678, raw_tokens: false, stop: [u32::MAX, u32::MAX] };
    let info = chip_info(&die, &code, layout.logits_register(&program), 8, &setup);
    let mut segments = image_segments(&die, &code, &layout);
    // A sample of the model's own weights goes through the real controller into a cell.
    let sample = sim::image::sample_codes(&Weights::open(&dir).unwrap(), "model.embed_tokens.weight", 8).unwrap();
    segments.push(Segment { target: Target::CellWeight, unit: 0, address: 0, beats: 1, words: sample.clone() });
    eprintln!("{} segmentos, microprograma de {} palabras", segments.len(), code.words.len());
    let tables = TokenizerTables { byte_symbols: (0..256).map(|b| b % 8).collect(), letters: vec![(97, 122)], token_bytes: (0..8).map(|t| vec![b'a' + t]).collect(), text_capacity: 64, ..Default::default() };
    let image = build_image(&firmware.program, &firmware.data, &info, &tables.to_words(), &segments).unwrap();
    eprintln!("imagen de {} palabras", image.len());

    let input = b"hoy\0";
    let engine = VmEngine::new(compile_qwen3_5(CONFIG).unwrap(), Weights::open(&dir).unwrap(), die, CHIP_TOKENS).unwrap();
    let model = ChipModel::new(engine, code.clone(), SamplerModel { config: config.sampler, rng: 0, table: exponential_table() });
    let mut bus = SocBus::booted_with(config.soc, &image, input, model).unwrap();
    let mut cpu = Rv32Iss::default();
    while !cpu.halted {
        cpu.step(&mut bus);
        assert!(cpu.cycle < 100_000_000, "el firmware no termina");
    }
    let expected = bus.tx.clone();
    eprintln!("simulador de instrucciones: {} ciclos, salida {:?}", cpu.cycle, String::from_utf8_lossy(&expected));
    assert_eq!(expected.iter().filter(|b| **b == b'.').count(), segments.len(), "todos los segmentos deben cargar bien");

    let design = chip_design(config);
    let start = std::time::Instant::now();
    let (output, cycles, sim) = run_gates(&design, &image, input, cpu.cycle + 4096);
    eprintln!("puertas: {cycles} ciclos en {:.1} s, salida {:?}", start.elapsed().as_secs_f64(), String::from_utf8_lossy(&output));
    assert_eq!(String::from_utf8_lossy(&output), String::from_utf8_lossy(&expected));
    assert_eq!(cycles, cpu.cycle, "ciclos");
    for (address, code) in sample.iter().enumerate() {
        assert_eq!(sim.peek_memory("/die/cell0/compute/weight_sram", address), Some(*code as u128), "peso {address}");
    }
    std::fs::remove_dir_all(&dir).ok();
}
