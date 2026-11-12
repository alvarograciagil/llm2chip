//! The whole chip, from bytes in to bytes out: real firmware, the load controller,
//! a fixed microprogram, the die and the sampling unit.

use compiler::compile_qwen3_5;
use netlist::chip::{chip_design, chip_info, ChipConfig, ChipModel, GenerationSetup, DEBUG_CPU_PC, DEBUG_CYCLES, DEBUG_DIE_BUSY, DEBUG_SAMPLER_BUSY};
use netlist::image::build_image;
use netlist::rv32::Rv32Iss;
use netlist::sampler::{exponential_table, SamplerConfig, SamplerModel};
use netlist::soc::{FirmwareImage, SocBus, SocConfig};
use netlist::Simulator;
use sim::chip::{cell_segments, image_segments, VmEngine};
use sim::lower::Layout;
use sim::Weights;
use std::path::PathBuf;

mod common;
use common::*;

const CHIP_TOKENS: usize = 8;

fn build_firmware() -> Option<FirmwareImage> {
    common::firmware()
}

fn run_gates(design: &netlist::Design, image: &[u32], input: &[u8], limit: u64) -> (Vec<u8>, u64, Simulator) {
    let mut sim = Simulator::new(design).unwrap();
    sim.set("reset", 1).unwrap();
    sim.set("cpu_hold", 1).unwrap();
    sim.tick();
    sim.set("reset", 0).unwrap();
    for _ in 0..=netlist::chip::RESET_STRETCH {
        sim.tick();
    }
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

/// Runs the whole chip in gates on the test model and checks it says what the
/// instruction-level simulation says, in as many cycles.
fn whole_chip_in_gates(die: netlist::DieConfig, tag: &str) -> (Simulator, u64, Layout, PathBuf) {
    chip_in_gates(die, tag, false)
}

/// The same, choosing whether the embedding table stays in the image.
fn chip_in_gates(die: netlist::DieConfig, tag: &str, table_outside: bool) -> (Simulator, u64, Layout, PathBuf) {
    let firmware = build_firmware().expect("firmware");
    let dir = std::env::temp_dir().join(format!("llm2chip-chip-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    make_checkpoint(&dir);
    let program = compile_qwen3_5(CONFIG).unwrap();

    let mut layout = Layout::with_cells(&program, &Weights::open(&dir).unwrap(), die, CHIP_TOKENS).unwrap();
    if table_outside {
        assert!(layout.table_outside(&program), "el modelo de prueba tiene una tabla de embedding");
    }
    let code = layout.microprogram(&program).unwrap();
    // Each register takes only the vector memory its routines reach, and the microcode
    // keeps each distinct word once.
    let die = sim::chip::with_code_dictionary(sim::chip::compact_die(die, &layout), &code);
    eprintln!("diccionario de microcódigo: 2^{} palabras", die.features.dictionary_bits);
    let config = ChipConfig { soc: SocConfig { program_bits: 13, data_bits: 14, fifo_bits: 6 }, die, sampler: SamplerConfig { logit_bits: 16, index_bits: 12, entries: 4, lanes: 1 } };
    let setup = GenerationSetup { max_new: 2, greedy: false, limit: 4, temperature: 256, seed: 0x1234_5678, raw_tokens: false, stop: Vec::new(), draft_depth: 0, draft_logits: 0, think: None, think_budget: u32::MAX };
    let info = chip_info(&die, &code, layout.logits_register(&program), 8, &setup);
    let mut segments = image_segments(&die, &code, &layout);
    // The model's own weights go through the real controller into the cells.
    let checkpoint = Weights::open(&dir).unwrap();
    segments.extend(cell_segments(&die, &layout, &checkpoint).unwrap());
    eprintln!("{} segmentos, microprograma de {} palabras", segments.len(), code.words.len());
    let tables = tiny_tokenizer();
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
    // A table that stays in the image is not loaded at boot.
    let at_boot = segments.iter().filter(|s| s.target != netlist::image::Target::RuntimeTable).count();
    assert_eq!(expected.iter().filter(|b| **b == b'.').count(), at_boot, "todos los segmentos deben cargar bien");

    let design = chip_design(config);
    let start = std::time::Instant::now();
    let (output, cycles, sim) = run_gates(&design, &image, input, cpu.cycle + 4096);
    eprintln!("puertas: {cycles} ciclos en {:.1} s, salida {:?}", start.elapsed().as_secs_f64(), String::from_utf8_lossy(&output));
    assert_eq!(String::from_utf8_lossy(&output), String::from_utf8_lossy(&expected));
    assert_eq!(cycles, cpu.cycle, "ciclos");
    (sim, bus.peripheral.die_busy, layout, dir)
}

#[test]
#[ignore = "lenta: solo tiene sentido con optimizaciones: son cientos de miles de ciclos"]
fn the_chip_sized_for_an_fpga_says_the_same_in_gates() {
    if build_firmware().is_none() {
        return;
    }
    let (_, _, _, dir) = whole_chip_in_gates(fpga_die_config(), "fpga");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
#[ignore = "lenta: solo tiene sentido con optimizaciones: son cientos de miles de ciclos"]
fn the_embedding_can_stay_in_the_image_and_come_a_row_at_a_time() {
    if build_firmware().is_none() {
        return;
    }
    let (_, _, layout, dir) = chip_in_gates(fpga_die_config(), "fila", true);
    // The register keeps one row: the vector memory no longer holds the whole table.
    let table = layout.outside_table().expect("tabla fuera");
    assert_eq!(table.mantissas.len(), 8 * table.cols as usize);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
#[ignore = "lenta: solo tiene sentido con optimizaciones: son cientos de miles de ciclos"]
fn text_goes_through_the_whole_chip_in_gates() {
    if build_firmware().is_none() {
        return;
    }
    let die = netlist::DieConfig { microcode_bits: 15, ..cell_die_config() };
    let (mut sim, die_busy, layout, dir) = whole_chip_in_gates(die, "full");
    let checkpoint = Weights::open(&dir).unwrap();
    let debug = |sim: &mut Simulator, what: u64| {
        sim.set("debug_select", what).unwrap();
        sim.settle();
        sim.get("debug_value").unwrap()
    };
    assert_eq!(debug(&mut sim, DEBUG_CYCLES), sim.cycle() - 1, "contador de ciclos");
    assert_eq!(debug(&mut sim, DEBUG_DIE_BUSY), die_busy, "ciclos del die");
    assert!(debug(&mut sim, DEBUG_SAMPLER_BUSY) > 0);
    assert!(debug(&mut sim, DEBUG_CPU_PC) > 0);
    let (codes, _) = layout.cell_contents(&checkpoint, 0).unwrap();
    for (address, code) in codes.iter().enumerate() {
        assert_eq!(sim.peek_memory("/die/cell0/compute/weight_sram", address), Some((*code as u8) as u128), "peso {address}");
    }
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
#[ignore = "lenta: millones de instrucciones: solo con optimizaciones"]
fn any_token_on_the_stop_list_ends_the_answer() {
    let Some(firmware) = build_firmware() else { return };
    let dir = std::env::temp_dir().join(format!("llm2chip-stop-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    make_checkpoint(&dir);
    let program = compile_qwen3_5(CONFIG).unwrap();
    let die = netlist::DieConfig { microcode_bits: 15, ..cell_die_config() };
    let sampler = SamplerConfig { logit_bits: 16, index_bits: 12, entries: 4, lanes: 1 };
    let soc = SocConfig { program_bits: 13, data_bits: 14, fifo_bits: 6 };
    let layout = Layout::with_cells(&program, &Weights::open(&dir).unwrap(), die, CHIP_TOKENS).unwrap();
    let code = layout.microprogram(&program).unwrap();
    let generate = |stop: Vec<u32>| {
        let setup = GenerationSetup { max_new: 1, greedy: true, limit: 4, temperature: 256, seed: 1, raw_tokens: false, stop, draft_depth: 0, draft_logits: 0, think: None, think_budget: u32::MAX };
        let info = chip_info(&die, &code, layout.logits_register(&program), 8, &setup);
        let image = build_image(&firmware.program, &firmware.data, &info, &tiny_tokenizer().to_words(), &[image_segments(&die, &code, &layout), cell_segments(&die, &layout, &Weights::open(&dir).unwrap()).unwrap()].concat()).unwrap();
        let engine = VmEngine::new(compile_qwen3_5(CONFIG).unwrap(), Weights::open(&dir).unwrap(), die, CHIP_TOKENS).unwrap();
        let model = ChipModel::new(engine, code.clone(), SamplerModel { config: sampler, rng: 0, table: exponential_table() });
        let mut bus = SocBus::booted_with(soc, &image, b"hoy\0", model).unwrap();
        let mut cpu = Rv32Iss::default();
        while !cpu.halted {
            cpu.step(&mut bus);
        }
        bus.tx.iter().filter(|b| **b != b'.').copied().collect::<Vec<u8>>()
    };
    let free = generate(Vec::new());
    assert_eq!(free.len(), 1, "un token de una letra");
    let token = match free[0] { b'h' => 0, b'y' => 1, b'o' => 7, letter => (letter - b'a') as u32 };
    assert!(generate(vec![100, 200, 300, token]).is_empty(), "el cuarto de la lista también detiene");
    assert_eq!(generate(vec![100, 200, 300]), free);
    std::fs::remove_dir_all(dir).unwrap();
}
