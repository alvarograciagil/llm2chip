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
    let (output, cycles, halted) = run_gates_until(&mut sim, image, input, limit);
    assert!(halted, "el firmware no terminó en {limit} ciclos");
    (output, cycles, sim)
}

/// Boots the chip and runs it for at most `limit` cycles: its output, the cycles it ran and
/// whether it halted.
fn run_gates_until(sim: &mut Simulator, image: &[u32], input: &[u8], limit: u64) -> (Vec<u8>, u64, bool) {
    sim.set("reset", 1).unwrap();
    sim.set("cpu_hold", 1).unwrap();
    sim.tick();
    sim.set("reset", 0).unwrap();
    for _ in 0..=netlist::chip::RESET_RELEASE {
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
        serve(sim, &mut pending);
        sim.tick();
    }
    sim.set("host_rx_valid", 0).unwrap();
    let mut boot = 0;
    loop {
        serve(sim, &mut pending);
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
        serve(sim, &mut pending);
        if sim.get("host_tx_valid").unwrap() == 1 {
            output.push(sim.get("host_tx_data").unwrap() as u8);
        }
        if sim.get("halted").unwrap() == 1 {
            return (output, cycle + 1, true);
        }
        sim.tick();
    }
    (output, limit, false)
}

/// Runs the whole chip in gates on the test model and checks it says what the
/// instruction-level simulation says, in as many cycles.
fn whole_chip_in_gates(die: netlist::DieConfig, tag: &str) -> (Simulator, u64, Layout, PathBuf) {
    chip_in_gates(die, tag, false, chip_design)
}

/// The same, choosing whether the embedding table stays in the image.
fn chip_in_gates(die: netlist::DieConfig, tag: &str, table_outside: bool, design: fn(ChipConfig) -> netlist::Design) -> (Simulator, u64, Layout, PathBuf) {
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
    let design = design(config);
    let (sim, die_busy) = says_the_same_in_gates(config, &design, &layout, &code, CONFIG, &dir);
    (sim, die_busy, layout, dir)
}

/// The image of one model for a chip of `config`, and what the host sends it.
fn image_and_input(config: ChipConfig, layout: &Layout, code: &netlist::Microprogram, model: &str, dir: &std::path::Path) -> (Vec<u32>, Vec<u8>) {
    let firmware = build_firmware().expect("firmware");
    let die = config.die;
    let setup = GenerationSetup { max_new: 2, greedy: false, limit: 4, temperature: 256, seed: 0x1234_5678, raw_tokens: false, stop: Vec::new(), draft_depth: 0, draft_logits: 0, think: None, think_budget: u32::MAX };
    let info = chip_info(&die, code, layout.logits_register(&compile_qwen3_5(model).unwrap()), 8, &setup);
    let mut segments = image_segments(&die, code, layout);
    segments.extend(cell_segments(&die, layout, &Weights::open(dir).unwrap()).unwrap());
    let image = build_image(&firmware.program, &firmware.data, &info, &tiny_tokenizer().to_words(), &segments).unwrap();
    (image, b"hoy\0".to_vec())
}

/// Boots `design` with the image of one model and checks it writes what the instruction-level
/// simulation writes, in as many cycles. Gives the simulator and the cycles the die was busy.
fn says_the_same_in_gates(config: ChipConfig, design: &netlist::Design, layout: &Layout, code: &netlist::Microprogram, model: &str, dir: &std::path::Path) -> (Simulator, u64) {
    let firmware = build_firmware().expect("firmware");
    let die = config.die;
    let setup = GenerationSetup { max_new: 2, greedy: false, limit: 4, temperature: 256, seed: 0x1234_5678, raw_tokens: false, stop: Vec::new(), draft_depth: 0, draft_logits: 0, think: None, think_budget: u32::MAX };
    let info = chip_info(&die, code, layout.logits_register(&compile_qwen3_5(model).unwrap()), 8, &setup);
    let mut segments = image_segments(&die, code, layout);
    // The model's own weights go through the real controller into the cells.
    let checkpoint = Weights::open(dir).unwrap();
    segments.extend(cell_segments(&die, layout, &checkpoint).unwrap());
    eprintln!("{} segmentos, microprograma de {} palabras", segments.len(), code.words.len());
    let tables = tiny_tokenizer();
    let image = build_image(&firmware.program, &firmware.data, &info, &tables.to_words(), &segments).unwrap();
    eprintln!("imagen de {} palabras", image.len());

    let input = b"hoy\0";
    let mut engine = VmEngine::new(compile_qwen3_5(model).unwrap(), Weights::open(dir).unwrap(), die, CHIP_TOKENS).unwrap();
    if layout.outside_table().is_some() {
        engine = engine.with_table_outside().unwrap();
    }
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

    let start = std::time::Instant::now();
    let (output, cycles, sim) = run_gates(design, &image, input, cpu.cycle + 4096);
    eprintln!("puertas: {cycles} ciclos en {:.1} s, salida {:?}", start.elapsed().as_secs_f64(), String::from_utf8_lossy(&output));
    assert_eq!(String::from_utf8_lossy(&output), String::from_utf8_lossy(&expected));
    assert_eq!(cycles, cpu.cycle, "ciclos");
    (sim, bus.peripheral.die_busy)
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
    let (_, _, layout, dir) = chip_in_gates(fpga_die_config(), "fila", true, chip_design);
    // The register keeps one row: the vector memory no longer holds the whole table.
    let table = layout.outside_table().expect("tabla fuera");
    assert_eq!(table.mantissas.len(), 8 * table.cols as usize);
    std::fs::remove_dir_all(&dir).ok();
}

/// The same chip and text in Icarus Verilog, a simulator that is not this project's, from
/// the exported netlist: the same bytes in the same cycles.
#[test]
#[ignore = "lenta: Icarus Verilog simula el chip entero puerta a puerta, cerca de una hora; con LLM2CHIP_ICARUS=1"]
fn text_goes_through_the_whole_chip_in_icarus_verilog() {
    // Hours of simulation: only when asked for, `LLM2CHIP_ICARUS=1`.
    if std::env::var_os("LLM2CHIP_ICARUS").is_none() {
        eprintln!("omitida: LLM2CHIP_ICARUS=1 para simular el chip entero en Icarus Verilog");
        return;
    }
    if build_firmware().is_none() {
        return;
    }
    let die = netlist::DieConfig { microcode_bits: 15, ..cell_die_config() };
    let dir = std::env::temp_dir().join(format!("llm2chip-chip-icarus-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    make_checkpoint(&dir);
    let program = compile_qwen3_5(CONFIG).unwrap();
    let layout = Layout::with_cells(&program, &Weights::open(&dir).unwrap(), die, CHIP_TOKENS).unwrap();
    let code = layout.microprogram(&program).unwrap();
    let die = sim::chip::with_code_dictionary(sim::chip::compact_die(die, &layout), &code);
    let config = ChipConfig { soc: SocConfig { program_bits: 13, data_bits: 14, fifo_bits: 6 }, die, sampler: SamplerConfig { logit_bits: 16, index_bits: 12, entries: 4, lanes: 1 } };
    let (image, input) = image_and_input(config, &layout, &code, CONFIG, &dir);
    let engine = VmEngine::new(program, Weights::open(&dir).unwrap(), die, CHIP_TOKENS).unwrap();
    let model = ChipModel::new(engine, code.clone(), SamplerModel { config: config.sampler, rng: 0, table: exponential_table() });
    let mut bus = SocBus::booted_with(config.soc, &image, &input, model).unwrap();
    let mut cpu = Rv32Iss::default();
    while !cpu.halted {
        cpu.step(&mut bus);
    }
    let started = std::time::Instant::now();
    let Some((output, cycles)) = chip_in_iverilog(&chip_design(config), &image, &input, cpu.cycle + 4096, "texto") else { return };
    eprintln!("Icarus Verilog: {cycles} ciclos en {:.0} s, salida {:?}", started.elapsed().as_secs_f64(), String::from_utf8_lossy(&output));
    assert_eq!(String::from_utf8_lossy(&output), String::from_utf8_lossy(&bus.tx));
    assert_eq!(cycles, cpu.cycle, "ciclos");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
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
    // The counters start once the synchronized reset is let go, two cycles after the pin.
    let synchronizer = netlist::chip::RESET_RELEASE - netlist::chip::RESET_STRETCH;
    assert_eq!(debug(&mut sim, DEBUG_CYCLES), sim.cycle() - 1 - synchronizer, "contador de ciclos");
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
fn a_chip_alone_on_one_board_says_the_same_without_tests_or_chain() {
    if build_firmware().is_none() {
        return;
    }
    let die = netlist::DieConfig { microcode_bits: 15, ..cell_die_config() };
    let (sim, _, _, dir) = chip_in_gates(die, "sola", false, netlist::chip::chip_design_alone);
    // Its pins are those of every chip, the ones it does not use held at zero.
    assert_eq!(sim.get("chain_tx_valid").unwrap(), 0);
    assert_eq!(sim.get("jtag_tdo").unwrap(), 0);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn one_chip_runs_either_of_two_models_from_its_image() {
    if build_firmware().is_none() {
        return;
    }
    // The test model twice, with codes of 4 and of 2 bits: different weights, different
    // packing in the cells, different microcode.
    let dirs: Vec<PathBuf> = [(2, 4), (4, 2)]
        .iter()
        .enumerate()
        .map(|(i, (group, bits))| {
            GROUP.with(|g| g.set(*group));
            BITS.with(|b| b.set(*bits));
            let dir = std::env::temp_dir().join(format!("llm2chip-dos-{i}-{}", std::process::id()));
            std::fs::create_dir_all(&dir).unwrap();
            make_checkpoint(&dir);
            dir
        })
        .collect();
    let programs: Vec<_> = dirs.iter().map(|_| compile_qwen3_5(CONFIG).unwrap()).collect();
    let weights: Vec<_> = dirs.iter().map(|d| Weights::open(d).unwrap()).collect();
    let models: Vec<_> = programs.iter().zip(&weights).collect();
    let trial = netlist::DieConfig { microcode_bits: 15, ..cell_die_config() };
    let chip = sim::chip::fpga_chip_for(&models, trial, CHIP_TOKENS, sim::chip::FpgaOptions { table_inside: true, tokenizer_words: tiny_tokenizer().to_words().len(), ..Default::default() }).unwrap();
    for ((layout, code), dir) in chip.models.iter().zip(&dirs) {
        says_the_same_in_gates(chip.config, &chip.design, layout, code, CONFIG, dir);
        std::fs::remove_dir_all(dir).ok();
    }
}

#[test]
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

/// A real model, stories260K, through the whole chip as the FPGA example builds it: the
/// firmware tokenizes the text, and each token's embedding row comes from the image. The
/// instruction-level chip writes what the die's own machine writes driven by the host, and
/// what floating point writes. The model is not in the repository: it is taken from
/// `LLM2CHIP_STORIES` or `~/.llm2chip/models`.
/// The gates write the same, in as many cycles.
#[test]
#[ignore = "lenta: un modelo real por el chip entero en puertas, millones de ciclos"]
fn a_real_model_writes_the_same_text_through_the_whole_chip() {
    use netlist::chip::DieEngine;
    use sim::chip::{fpga_chip_for, FpgaChip, FpgaOptions};
    let Some(firmware) = build_firmware() else { return };
    let dir = std::env::var_os("LLM2CHIP_STORIES").map(PathBuf::from).unwrap_or_else(|| PathBuf::from(std::env::var_os("HOME").unwrap_or_default()).join(".llm2chip/models/stories260K"));
    if !dir.join("model.safetensors").exists() {
        eprintln!("omitida: no está stories260K en {}", dir.display());
        return;
    }
    let context = 32;
    let bits_for = |count: u64| (u64::BITS - count.max(2).saturating_sub(1).leading_zeros()).max(1);
    let program = compiler::compile_model(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
    let weights = Weights::open(&dir).unwrap();
    // Room for everything first, then just the bits the layout needs.
    let base = fpga_die_config();
    let roomy = netlist::DieConfig { reg_bits: 12, index_bits: 20, cell: netlist::CellConfig { addr_bits: 20, weight_bits: base.cell.activation_bits, ..base.cell }, features: base.features.with_key_shift(bits_for(context as u64)), ..base };
    let (code_bits, _) = Layout::with_cells(&program, &weights, roomy, context).unwrap().cell_shape(&weights).unwrap();
    // The cells' depth with words as wide as the codes.
    let roomy = netlist::DieConfig { cell: netlist::CellConfig { weight_bits: code_bits, ..roomy.cell }, ..roomy };
    let (depth, groups) = Layout::with_cells(&program, &weights, roomy, context).unwrap().cell_depths();
    let addr_bits = bits_for(depth.max(2 * groups));
    let index_bits = bits_for(program.buffers.iter().map(|b| *b as u64).max().unwrap_or(1).max(context as u64 * 512).max((program.vocab * program.hidden) as u64)).max(addr_bits);
    let trial = netlist::DieConfig { index_bits, cell: netlist::CellConfig { addr_bits, ..roomy.cell }, ..roomy };
    // The models of llama2.c start every text with their first token.
    let mut tables = sim::tokenizer::TokenizerTables::from_file(&dir.join("tokenizer.json"), context as u64).unwrap();
    tables.set_add_special(true);
    let tables = tables.to_words();
    let options = FpgaOptions { factor_steps: trial.features.factor_steps, table_inside: false, dictionary: true, tokenizer_words: tables.len(), chain: false };
    let FpgaChip { design, config, models } = fpga_chip_for(&[(&program, &weights)], trial, context, options).unwrap();
    let (layout, code) = &models[0];

    let setup = GenerationSetup { max_new: 3, greedy: true, limit: 1, temperature: 256, seed: 1, raw_tokens: false, stop: Vec::new(), draft_depth: 0, draft_logits: 0, think: None, think_budget: u32::MAX };
    let info = chip_info(&config.die, code, layout.logits_register(&program), program.vocab as u32, &setup);
    let mut segments = image_segments(&config.die, code, layout);
    segments.extend(cell_segments(&config.die, layout, &weights).unwrap());
    let image = build_image(&firmware.program, &firmware.data, &info, &tables, &segments).unwrap();
    let input = b"Once upon a time\0".to_vec();

    let engine = VmEngine::new(program.clone(), Weights::open(&dir).unwrap(), config.die, context).unwrap().with_table_outside().unwrap();
    let model = ChipModel::new(engine, code.clone(), SamplerModel { config: config.sampler, rng: 0, table: exponential_table() });
    let mut bus = SocBus::booted_with(config.soc, &image, &input, model).unwrap();
    let mut cpu = Rv32Iss::default();
    while !cpu.halted {
        cpu.step(&mut bus);
        assert!(cpu.cycle < 100_000_000, "el firmware no termina");
    }
    let text = String::from_utf8_lossy(&bus.tx).trim_start_matches('.').to_string();
    eprintln!("simulador de instrucciones: {} ciclos, {text:?}", cpu.cycle);

    // The same die's machine driven by the host, and floating point, from the same tokens.
    let tokenizer = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).unwrap();
    let prompt = tokenizer.encode("Once upon a time", true).unwrap().get_ids().to_vec();
    let mut vm = VmEngine::new(program.clone(), Weights::open(&dir).unwrap(), config.die, context).unwrap();
    let mut written = Vec::new();
    for (position, id) in prompt.iter().enumerate() {
        vm.run(sim::lower::ENTRY_TOKEN, *id, position as u32);
    }
    let base = layout.logits_register(&program) << config.die.index_bits;
    for step in 0..3 {
        let position = (prompt.len() + step) as u32;
        vm.run(sim::lower::ENTRY_LOGITS, *written.last().unwrap_or(prompt.last().unwrap()), position - 1);
        let next = (0..program.vocab as u32).max_by_key(|i| (vm.read(base + i), std::cmp::Reverse(*i))).unwrap();
        written.push(next);
        vm.run(sim::lower::ENTRY_TOKEN, next, position);
    }
    let by_host = tokenizer.decode(&written, true).unwrap();
    let mut engine = sim::Engine::new(program.clone(), Weights::open(&dir).unwrap(), sim::Numeric::Float).unwrap();
    let mut logits = Vec::new();
    for (i, id) in prompt.iter().enumerate() {
        logits = engine.step(*id, i + 1 == prompt.len()).unwrap().0.map(<[f32]>::to_vec).unwrap_or_default();
    }
    let mut float = Vec::new();
    for _ in 0..3 {
        let next = logits.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0 as u32;
        float.push(next);
        logits = engine.step(next, true).unwrap().0.unwrap().to_vec();
    }
    let float = tokenizer.decode(&float, true).unwrap();
    eprintln!("el anfitrión con la máquina del die: {by_host:?}; coma flotante: {float:?}");
    assert_eq!(text.trim_start(), by_host.trim_start(), "el firmware escribe lo que la máquina del die");
    assert_eq!(by_host, float, "la aritmética del die elige los tokens de la coma flotante");

    let started = std::time::Instant::now();
    let (output, cycles, _) = run_gates(&design, &image, &input, cpu.cycle + 4096);
    eprintln!("puertas: {cycles} ciclos en {:.0} s", started.elapsed().as_secs_f64());
    assert_eq!(String::from_utf8_lossy(&output), String::from_utf8_lossy(&bus.tx));
    assert_eq!(cycles, cpu.cycle, "ciclos");
}
