//! The whole chip, from bytes in to bytes out: real firmware, the load controller,
//! a fixed microprogram, the die and the sampling unit.

use compiler::compile_qwen3_5;
use netlist::chip::{chip_design, chip_info, ChipConfig, ChipModel, DieEngine, GenerationSetup, DEBUG_CPU_PC, DEBUG_CYCLES, DEBUG_DIE_BUSY, DEBUG_SAMPLER_BUSY};
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
    assert!(halted, "the firmware did not finish within {limit} cycles");
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
        assert!(boot < 1_000_000, "boot does not finish");
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
        assert!(layout.table_outside(&program), "the test model has an embedding table");
    }
    let code = layout.microprogram(&program).unwrap();
    // Each register takes only the vector memory its routines reach, and the microcode
    // keeps each distinct word once.
    let queue = die.features.queue_bits;
    let die = sim::chip::with_code_dictionary(sim::chip::compact_die(die, &layout), &code);
    let die = netlist::DieConfig { features: die.features.with_queue_bits(queue), ..die };
    eprintln!("microcode dictionary: 2^{} words", die.features.dictionary_bits);
    let config = ChipConfig { soc: SocConfig { program_words: netlist::soc::PROGRAM_WORDS, data_bits: 14, fifo_bits: 6, links: Default::default() }, die, sampler: SamplerConfig { logit_bits: 16, index_bits: 12, entries: 4, lanes: 1 } };
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
    (image, netlist::port::text("hoy"))
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
    eprintln!("{} segments, a microprogram of {} words", segments.len(), code.words.len());
    let tables = tiny_tokenizer();
    let image = build_image(&firmware.program, &firmware.data, &info, &tables.to_words(), &segments).unwrap();
    eprintln!("image of {} words", image.len());

    let input = &netlist::port::text("hoy");
    let mut engine = VmEngine::new(compile_qwen3_5(model).unwrap(), Weights::open(dir).unwrap(), die, CHIP_TOKENS).unwrap();
    if layout.outside_table().is_some() || layout.rotation_table_outside().is_some() {
        engine = engine.with_table_outside().unwrap();
    }
    let model = ChipModel::new(engine, code.clone(), SamplerModel { config: config.sampler, rng: 0, table: exponential_table() }).with_queue(config.die.features.queue_bits);
    let mut bus = SocBus::booted_with(config.soc, &image, input, model).unwrap();
    let mut cpu = Rv32Iss::default();
    while !cpu.halted {
        cpu.step(&mut bus);
        assert!(cpu.cycle < 100_000_000, "the firmware does not finish");
    }
    let expected = bus.tx.clone();
    eprintln!("instruction-level simulator: {} cycles, output {:?}", cpu.cycle, String::from_utf8_lossy(&expected));
    // A table that stays in the image is not loaded at boot.
    let at_boot = segments.iter().filter(|s| !matches!(s.target, netlist::image::Target::RuntimeTable | netlist::image::Target::RotationTable)).count();
    assert_eq!(netlist::port::boot(&expected), Some((at_boot, 0)), "every segment must load correctly");

    let start = std::time::Instant::now();
    let (output, cycles, sim) = run_gates(design, &image, input, cpu.cycle + 4096);
    eprintln!("gates: {cycles} cycles in {:.1} s, output {:?}", start.elapsed().as_secs_f64(), String::from_utf8_lossy(&output));
    assert_eq!(String::from_utf8_lossy(&output), String::from_utf8_lossy(&expected));
    assert_eq!(cycles, cpu.cycle, "cycles");
    (sim, bus.peripheral.die_busy)
}

#[test]
#[ignore = "slow: only makes sense with optimizations: hundreds of thousands of cycles"]
fn the_chip_sized_for_an_fpga_says_the_same_in_gates() {
    if build_firmware().is_none() {
        return;
    }
    let (_, _, _, dir) = whole_chip_in_gates(fpga_die_config(), "fpga");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
#[ignore = "slow: only makes sense with optimizations: hundreds of thousands of cycles"]
fn the_embedding_can_stay_in_the_image_and_come_a_row_at_a_time() {
    if build_firmware().is_none() {
        return;
    }
    let (_, _, layout, dir) = chip_in_gates(fpga_die_config(), "row", true, chip_design);
    // The register keeps one row: the vector memory no longer holds the whole table.
    let table = layout.outside_table().expect("table outside");
    assert_eq!(table.mantissas.len(), 8 * table.cols as usize);
    // So does the rotation table: the firmware brings each position's row.
    let rotation = layout.rotation_table_outside().expect("rotation table outside");
    assert_eq!(rotation.mantissas.len(), CHIP_TOKENS * rotation.cols as usize);
    assert!(layout.longest_registers(64).iter().all(|(register, reach)| *register != rotation.register || *reach <= rotation.cols), "the rotation table's register holds one row");
    std::fs::remove_dir_all(&dir).ok();
}

/// The same chip and text in Icarus Verilog, a simulator that is not this project's, from
/// the exported netlist: the same bytes in the same cycles.
#[test]
#[ignore = "slow: Icarus Verilog simulates the whole chip gate by gate, close to an hour; with LLM2CHIP_ICARUS=1"]
fn text_goes_through_the_whole_chip_in_icarus_verilog() {
    // Hours of simulation: only when asked for, `LLM2CHIP_ICARUS=1`.
    if std::env::var_os("LLM2CHIP_ICARUS").is_none() {
        eprintln!("skipped: LLM2CHIP_ICARUS=1 to simulate the whole chip in Icarus Verilog");
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
    let config = ChipConfig { soc: SocConfig { program_words: netlist::soc::PROGRAM_WORDS, data_bits: 14, fifo_bits: 6, links: Default::default() }, die, sampler: SamplerConfig { logit_bits: 16, index_bits: 12, entries: 4, lanes: 1 } };
    let (image, input) = image_and_input(config, &layout, &code, CONFIG, &dir);
    let engine = VmEngine::new(program, Weights::open(&dir).unwrap(), die, CHIP_TOKENS).unwrap();
    let model = ChipModel::new(engine, code.clone(), SamplerModel { config: config.sampler, rng: 0, table: exponential_table() });
    let mut bus = SocBus::booted_with(config.soc, &image, &input, model).unwrap();
    let mut cpu = Rv32Iss::default();
    while !cpu.halted {
        cpu.step(&mut bus);
    }
    let started = std::time::Instant::now();
    let Some((output, cycles)) = chip_in_iverilog(&chip_design(config), &image, &input, cpu.cycle + 4096, "text") else { return };
    eprintln!("Icarus Verilog: {cycles} cycles in {:.0} s, output {:?}", started.elapsed().as_secs_f64(), String::from_utf8_lossy(&output));
    assert_eq!(String::from_utf8_lossy(&output), String::from_utf8_lossy(&bus.tx));
    assert_eq!(cycles, cpu.cycle, "cycles");
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
    assert_eq!(debug(&mut sim, DEBUG_CYCLES), sim.cycle() - 1 - synchronizer, "cycle counter");
    assert_eq!(debug(&mut sim, DEBUG_DIE_BUSY), die_busy, "die cycles");
    assert!(debug(&mut sim, DEBUG_SAMPLER_BUSY) > 0);
    assert!(debug(&mut sim, DEBUG_CPU_PC) > 0);
    let (codes, _) = layout.cell_contents(&checkpoint, 0).unwrap();
    for (address, code) in codes.iter().enumerate() {
        assert_eq!(sim.peek_memory("/die/cell0/compute/weight_sram", address), Some((*code as u8) as u128), "weight {address}");
    }
    std::fs::remove_dir_all(&dir).ok();
}

/// The same chip with a command queue before its die: the firmware's orders wait there
/// until the die is free, and the gates take them as the instruction-level model does,
/// cycle for cycle.
#[test]
fn text_goes_through_a_chip_with_a_command_queue_in_gates() {
    if build_firmware().is_none() {
        return;
    }
    let base = netlist::DieConfig { microcode_bits: 15, ..cell_die_config() };
    let die = netlist::DieConfig { features: base.features.with_queue_bits(3), ..base };
    let (_, _, _, dir) = whole_chip_in_gates(die, "queue");
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
    let soc = SocConfig { program_words: netlist::soc::PROGRAM_WORDS, data_bits: 14, fifo_bits: 6, links: Default::default() };
    let layout = Layout::with_cells(&program, &Weights::open(&dir).unwrap(), die, CHIP_TOKENS).unwrap();
    let code = layout.microprogram(&program).unwrap();
    let generate = |stop: Vec<u32>| {
        let setup = GenerationSetup { max_new: 1, greedy: true, limit: 4, temperature: 256, seed: 1, raw_tokens: false, stop, draft_depth: 0, draft_logits: 0, think: None, think_budget: u32::MAX };
        let info = chip_info(&die, &code, layout.logits_register(&program), 8, &setup);
        let image = build_image(&firmware.program, &firmware.data, &info, &tiny_tokenizer().to_words(), &[image_segments(&die, &code, &layout), cell_segments(&die, &layout, &Weights::open(&dir).unwrap()).unwrap()].concat()).unwrap();
        let engine = VmEngine::new(compile_qwen3_5(CONFIG).unwrap(), Weights::open(&dir).unwrap(), die, CHIP_TOKENS).unwrap();
        let model = ChipModel::new(engine, code.clone(), SamplerModel { config: sampler, rng: 0, table: exponential_table() });
        let mut bus = SocBus::booted_with(soc, &image, &netlist::port::text("hoy"), model).unwrap();
        let mut cpu = Rv32Iss::default();
        while !cpu.halted {
            cpu.step(&mut bus);
        }
        netlist::port::answer_texts(&bus.tx)[0].clone().into_bytes()
    };
    let free = generate(Vec::new());
    assert_eq!(free.len(), 1, "a one-letter token");
    let token = match free[0] { b'h' => 0, b'y' => 1, b'o' => 7, letter => (letter - b'a') as u32 };
    assert!(generate(vec![100, 200, 300, token]).is_empty(), "the fourth on the list also stops");
    assert_eq!(generate(vec![100, 200, 300]), free);
    std::fs::remove_dir_all(dir).unwrap();
}

/// A chip that holds a conversation, at instruction level on the die's exact machine.
struct Talk {
    bus: SocBus<ChipModel<VmEngine>>,
    cpu: Rv32Iss,
}

impl Talk {
    /// Runs until the chip has ended `answers` answers in all: the tokens of each.
    fn until(&mut self, answers: usize) -> Vec<Vec<u32>> {
        while netlist::port::answers(&self.bus.tx).len() < answers {
            self.cpu.step(&mut self.bus);
            assert!(!self.cpu.halted && self.cpu.cycle < 200_000_000, "the chip stopped or never answered");
        }
        netlist::port::answers(&self.bus.tx)[..answers].to_vec()
    }

    fn say(&mut self, tokens: &[u32]) {
        self.bus.send(&netlist::port::tokens(tokens));
    }
}

/// The chip answers a turn after another over the same cache, as if it had been given the
/// whole conversation at once, and the clear pin empties the cache and the recurrent state,
/// whether it waits or answers, so the next conversation starts as the first did.
#[test]
fn a_conversation_goes_on_over_its_cache_until_the_clear_pin() {
    // A die that keeps each column of the cache's tokens together clears it whole.
    let state = conversation_on(netlist::DieConfig { microcode_bits: 15, ..cell_die_config() }, "columns");
    assert!(state.iter().all(|e| e.row == 0));
}

/// The same on a die that keeps each token's row of the cache together: it clears only the
/// rows the conversation used.
#[test]
fn a_die_with_the_cache_by_rows_clears_only_the_rows_used() {
    // Without the wide sums a die keeps the cache a token's row after another.
    let die = netlist::DieConfig { microcode_bits: 15, features: netlist::DieFeatures::without_register_windows(), ..cell_die_config() };
    let state = conversation_on(die, "rows");
    assert!(state.iter().any(|e| e.row > 0), "{state:?}");
}

fn conversation_on(die: netlist::DieConfig, tag: &str) -> Vec<netlist::chip::ClearEntry> {
    let Some(firmware) = build_firmware() else { return Vec::new() };
    let dir = std::env::temp_dir().join(format!("llm2chip-talk-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    make_checkpoint(&dir);
    let program = compile_qwen3_5(CONFIG).unwrap();
    let sampler = SamplerConfig { logit_bits: 16, index_bits: 12, entries: 4, lanes: 1 };
    let soc = SocConfig { program_words: netlist::soc::PROGRAM_WORDS, data_bits: 14, fifo_bits: 6, links: Default::default() };
    let layout = Layout::with_cells(&program, &Weights::open(&dir).unwrap(), die, 16).unwrap();
    let code = layout.microprogram(&program).unwrap();
    let state = layout.conversation_state(&program);
    assert!(!state.is_empty());
    let segments = [image_segments(&die, &code, &layout), cell_segments(&die, &layout, &Weights::open(&dir).unwrap()).unwrap()].concat();
    let start = |conversation: Option<&[netlist::chip::ClearEntry]>, first: &[u32]| {
        let setup = GenerationSetup { max_new: 3, greedy: true, limit: 1, temperature: 256, seed: 1, raw_tokens: true, stop: Vec::new(), draft_depth: 0, draft_logits: 0, think: None, think_budget: u32::MAX };
        let mut info = chip_info(&die, &code, layout.logits_register(&program), 8, &setup);
        if let Some(entries) = conversation {
            netlist::chip::add_conversation(&mut info, 16, entries).unwrap();
        }
        let image = build_image(&firmware.program, &firmware.data, &info, &[], &segments).unwrap();
        let engine = VmEngine::new(compile_qwen3_5(CONFIG).unwrap(), Weights::open(&dir).unwrap(), die, 16).unwrap();
        let model = ChipModel::new(engine, code.clone(), SamplerModel { config: sampler, rng: 0, table: exponential_table() });
        let mut talk = Talk { bus: SocBus::booted_with(soc, &image, &[], model).unwrap(), cpu: Rv32Iss::default() };
        talk.say(first);
        talk
    };
    // One question with the whole conversation, on a chip that answers once.
    let once = |tokens: &[u32]| {
        let mut chip = start(None, tokens);
        while !chip.cpu.halted {
            chip.cpu.step(&mut chip.bus);
        }
        netlist::port::answers(&chip.bus.tx).concat()
    };
    let (first, second) = ([3u32, 5, 1], [2u32, 6]);
    let mut talk = start(Some(&state), &first);
    let answers = talk.until(1);
    assert_eq!(answers[0], once(&first));
    talk.say(&second);
    let answers = talk.until(2);
    let whole: Vec<u32> = first.iter().chain(&answers[0]).chain(&second).copied().collect();
    assert_eq!(answers[1], once(&whole), "the second turn goes on from the first");

    // The clear pin while the chip waits: the cache and the state are zeros again.
    talk.bus.clear();
    for _ in 0..200_000 {
        talk.cpu.step(&mut talk.bus);
    }
    let engine = &mut talk.bus.peripheral.engine;
    for entry in &state {
        for register in 0..entry.registers {
            let base = entry.address + register * entry.step;
            assert!((0..entry.extent).all(|i| engine.read(base + i) == 0), "register at {base:#x} is cleared");
        }
    }
    talk.say(&first);
    assert_eq!(talk.until(3)[2], answers[0], "a new conversation starts as the first did");

    // And in the middle of an answer.
    talk.say(&second);
    let before = talk.bus.tx.len();
    while talk.bus.tx.len() < before + 4 {
        talk.cpu.step(&mut talk.bus);
    }
    talk.bus.clear();
    talk.say(&first);
    let after = talk.until(5);
    assert_eq!(after[4], answers[0], "the cut answer ends, and the next conversation starts afresh");

    // Without clearing the recurrent state the next conversation would start from the old.
    let mut dirty = start(Some(&[]), &first);
    dirty.until(1);
    dirty.bus.clear();
    for _ in 0..200_000 {
        dirty.cpu.step(&mut dirty.bus);
    }
    let left = state.iter().filter(|e| e.row == 0).any(|e| (0..e.extent).any(|i| dirty.bus.peripheral.engine.read(e.address + i) != 0));
    assert!(left, "the model's recurrent state needs the clear");
    std::fs::remove_dir_all(dir).unwrap();
    state
}

/// The host asks and steers a chip in messages: the options and their values, a new value
/// for one, where the conversation stands, even in the middle of an answer, a stop, a clear
/// asked as a message, and the errors of what the chip does not take.
#[test]
fn the_host_asks_and_steers_the_chip_in_messages() {
    use netlist::port::{self, Ending, Event};
    let Some(firmware) = build_firmware() else { return };
    let dir = std::env::temp_dir().join(format!("llm2chip-messages-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    make_checkpoint(&dir);
    let program = compile_qwen3_5(CONFIG).unwrap();
    let die = netlist::DieConfig { microcode_bits: 15, ..cell_die_config() };
    let layout = Layout::with_cells(&program, &Weights::open(&dir).unwrap(), die, 16).unwrap();
    let code = layout.microprogram(&program).unwrap();
    let segments = [image_segments(&die, &code, &layout), cell_segments(&die, &layout, &Weights::open(&dir).unwrap()).unwrap()].concat();
    let setup = GenerationSetup { max_new: 3, greedy: true, limit: 1, temperature: 256, seed: 1, raw_tokens: true, stop: Vec::new(), draft_depth: 0, draft_logits: 0, think: None, think_budget: 77 };
    let mut info = chip_info(&die, &code, layout.logits_register(&program), 8, &setup);
    netlist::chip::add_conversation(&mut info, 16, &layout.conversation_state(&program)).unwrap();
    let image = build_image(&firmware.program, &firmware.data, &info, &[], &segments).unwrap();
    let engine = VmEngine::new(compile_qwen3_5(CONFIG).unwrap(), Weights::open(&dir).unwrap(), die, 16).unwrap();
    let model = ChipModel::new(engine, code.clone(), SamplerModel { config: SamplerConfig { logit_bits: 16, index_bits: 12, entries: 4, lanes: 1 }, rng: 0, table: exponential_table() });
    let soc = SocConfig { program_words: netlist::soc::PROGRAM_WORDS, data_bits: 14, fifo_bits: 6, links: Default::default() };
    let mut talk = Talk { bus: SocBus::booted_with(soc, &image, &[], model).unwrap(), cpu: Rv32Iss::default() };
    // Runs until the chip has sent `count` messages of its own after the boot, and gives them.
    let mut seen = 0;
    let mut next = |talk: &mut Talk, count: usize| -> Vec<Event> {
        while port::events(&talk.bus.tx).len() < seen + count {
            talk.cpu.step(&mut talk.bus);
            assert!(!talk.cpu.halted && talk.cpu.cycle < 200_000_000, "the chip stopped or never answered: {:?}", port::events(&talk.bus.tx));
        }
        let events = port::events(&talk.bus.tx);
        let got = events[seen..seen + count].to_vec();
        seen += count;
        got
    };
    assert!(matches!(next(&mut talk, 2)[..], [Event::Boot { failed: 0, .. }, Event::Ready { capacity: 16 }]));

    // The options, then one set anew, which the next answer follows.
    talk.bus.send(&port::options());
    let options = vec![(5, 3), (6, 1), (7, 1), (8, 256), (9, 1), (20, 77)];
    assert_eq!(next(&mut talk, 1), [Event::Options(options)]);
    talk.bus.send(&port::set(port::option("max_new_tokens").unwrap(), 1));
    assert_eq!(next(&mut talk, 1), [Event::Options(vec![(5, 1)])]);
    talk.bus.send(&port::tokens(&[3, 5, 1]));
    let answer = next(&mut talk, 2);
    assert!(matches!(answer[..], [Event::Tokens(_), Event::End { ending: Ending::Limit, position: 4, capacity: 16 }]), "{answer:?}");

    // Where it stands while it waits.
    talk.bus.send(&port::status());
    assert_eq!(next(&mut talk, 1), [Event::Status { answering: false, position: 4, capacity: 16 }]);

    // In the middle of an answer: asked, it says it is answering; stopped, it ends there.
    talk.bus.send(&port::set(5, 3));
    next(&mut talk, 1);
    talk.bus.send(&[port::tokens(&[2]), port::status()].concat());
    let during = next(&mut talk, 1);
    assert!(matches!(during[..], [Event::Status { answering: true, position: 5, .. }]), "{during:?}");
    let rest = next(&mut talk, 4);
    assert!(matches!(rest[3], Event::End { ending: Ending::Limit, position: 8, .. }), "{rest:?}");
    talk.bus.send(&[port::tokens(&[6]), port::stop()].concat());
    assert!(matches!(next(&mut talk, 1)[..], [Event::End { ending: Ending::Stopped, position: 9, .. }]));

    // What it does not take.
    talk.bus.send(&[port::set(3, 0), port::frame(0x7E, &[1, 2]), port::text("hola")].concat());
    let refused = next(&mut talk, 3);
    assert_eq!(refused, [Event::Error { code: port::error::NO_OPTION, detail: 3 }, Event::Error { code: port::error::UNKNOWN, detail: 0x7E }, Event::Error { code: port::error::UNEXPECTED, detail: port::kind::TEXT as u32 }]);

    // A clear asked in a message: the conversation starts again, its cache empty.
    talk.bus.send(&[port::clear(), port::status()].concat());
    assert_eq!(next(&mut talk, 2), [Event::Ready { capacity: 16 }, Event::Status { answering: false, position: 0, capacity: 16 }]);
    std::fs::remove_dir_all(dir).unwrap();
}

/// The clear pin and the load controller's fills in gates: a chip holding a conversation
/// answers, is cleared while it waits, and answers the same question again, writing what the
/// instruction-level chip writes, byte for byte and cycle for cycle.
#[test]
fn the_clear_pin_works_in_gates_as_at_instruction_level() {
    let Some(firmware) = build_firmware() else { return };
    let dir = std::env::temp_dir().join(format!("llm2chip-clear-gates-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    make_checkpoint(&dir);
    let program = compile_qwen3_5(CONFIG).unwrap();
    let base = netlist::DieConfig { microcode_bits: 15, ..cell_die_config() };
    let layout = Layout::with_cells(&program, &Weights::open(&dir).unwrap(), base, CHIP_TOKENS).unwrap();
    let code = layout.microprogram(&program).unwrap();
    let die = sim::chip::with_code_dictionary(sim::chip::compact_die(base, &layout), &code);
    let config = ChipConfig { soc: SocConfig { program_words: netlist::soc::PROGRAM_WORDS, data_bits: 14, fifo_bits: 6, links: Default::default() }, die, sampler: SamplerConfig { logit_bits: 16, index_bits: 12, entries: 4, lanes: 1 } };
    let setup = GenerationSetup { max_new: 2, greedy: true, limit: 1, temperature: 256, seed: 1, raw_tokens: true, stop: Vec::new(), draft_depth: 0, draft_logits: 0, think: None, think_budget: u32::MAX };
    let mut info = chip_info(&die, &code, layout.logits_register(&program), 8, &setup);
    netlist::chip::add_conversation(&mut info, CHIP_TOKENS as u32, &layout.conversation_state(&program)).unwrap();
    let segments = [image_segments(&die, &code, &layout), cell_segments(&die, &layout, &Weights::open(&dir).unwrap()).unwrap()].concat();
    let image = build_image(&firmware.program, &firmware.data, &info, &[], &segments).unwrap();
    let question = netlist::port::tokens(&[3, 5]);
    let answers = |output: &[u8]| netlist::port::answers(output).len();

    // At instruction level: the cycle each answer ends, and the cycle the pin is raised.
    let engine = VmEngine::new(compile_qwen3_5(CONFIG).unwrap(), Weights::open(&dir).unwrap(), die, CHIP_TOKENS).unwrap();
    let model = ChipModel::new(engine, code.clone(), SamplerModel { config: config.sampler, rng: 0, table: exponential_table() });
    let mut bus = SocBus::booted_with(config.soc, &image, &question, model).unwrap();
    let mut cpu = Rv32Iss::default();
    while answers(&bus.tx) < 1 {
        cpu.step(&mut bus);
        assert!(cpu.cycle < 50_000_000);
    }
    let first = cpu.cycle;
    // The pin a while after the answer, while the chip waits.
    let clear_at = first + 500;
    while cpu.cycle < clear_at {
        cpu.step(&mut bus);
    }
    let raised = cpu.cycle;
    bus.clear();
    bus.send(&question);
    while answers(&bus.tx) < 2 {
        cpu.step(&mut bus);
        assert!(cpu.cycle < 100_000_000);
    }
    let second = cpu.cycle;
    let expected = bus.tx.clone();
    let both = netlist::port::answers(&expected);
    assert_eq!(both[0], both[1], "the same question again after the clear");

    // In gates, with the pin raised at the same cycle and the question sent after it.
    let mut sim = Simulator::new(&chip_design(config)).unwrap();
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
    let mut to_send: std::collections::VecDeque<u8> = question.iter().copied().collect();
    loop {
        sim.set("host_rx_valid", !to_send.is_empty() as u64).unwrap();
        sim.set("host_rx_data", to_send.front().copied().unwrap_or(0) as u64).unwrap();
        serve(&mut sim, &mut pending);
        if !to_send.is_empty() && sim.get("host_rx_ready").unwrap() == 1 {
            to_send.pop_front();
        }
        if sim.get("booting").unwrap() == 0 {
            break;
        }
        sim.tick();
    }
    sim.set("cpu_hold", 0).unwrap();
    sim.set("host_tx_ready", 1).unwrap();
    let (mut output, mut ends) = (Vec::new(), Vec::new());
    for cycle in 0..second + 4096 {
        sim.set("clear", (cycle + 1 == raised) as u64).unwrap();
        if cycle == raised {
            to_send.extend(&question);
        }
        sim.set("host_rx_valid", !to_send.is_empty() as u64).unwrap();
        sim.set("host_rx_data", to_send.front().copied().unwrap_or(0) as u64).unwrap();
        serve(&mut sim, &mut pending);
        if !to_send.is_empty() && sim.get("host_rx_ready").unwrap() == 1 {
            to_send.pop_front();
        }
        if sim.get("host_tx_valid").unwrap() == 1 {
            output.push(sim.get("host_tx_data").unwrap() as u8);
            if answers(&output) > ends.len() {
                ends.push(cycle + 1);
            }
        }
        if ends.len() == 2 {
            break;
        }
        sim.tick();
    }
    assert_eq!(output, expected);
    // A byte leaves the pins two cycles after the CPU writes it, through the output queue;
    // between the answers, the clear and its fills take as long as at instruction level.
    assert_eq!(ends, [first + 2, second + 2], "cycles");
    std::fs::remove_dir_all(dir).ok();
}

/// A real model, stories260K, through the whole chip as the FPGA example builds it: the
/// firmware tokenizes the text, and each token's embedding row comes from the image. The
/// instruction-level chip writes what the die's own machine writes driven by the host, and
/// what floating point writes. The model is not in the repository: it is taken from
/// `LLM2CHIP_STORIES` or `~/.llm2chip/models`.
/// The gates write the same, in as many cycles.
#[test]
#[ignore = "slow: a real model through the whole chip in gates, millions of cycles"]
fn a_real_model_writes_the_same_text_through_the_whole_chip() {
    use netlist::chip::DieEngine;
    use sim::chip::{fpga_chip_for, FpgaChip, FpgaOptions};
    let Some(firmware) = build_firmware() else { return };
    let dir = std::env::var_os("LLM2CHIP_STORIES").map(PathBuf::from).unwrap_or_else(|| PathBuf::from(std::env::var_os("HOME").unwrap_or_default()).join(".llm2chip/models/stories260K"));
    if !dir.join("model.safetensors").exists() {
        eprintln!("skipped: stories260K is not in {}", dir.display());
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
    let options = FpgaOptions { factor_steps: trial.features.factor_steps, table_inside: false, dictionary: true, tokenizer_words: tables.len(), chain: false, links: Default::default() };
    let FpgaChip { design, config, models } = fpga_chip_for(&[(&program, &weights)], trial, context, options).unwrap();
    let (layout, code) = &models[0];

    let setup = GenerationSetup { max_new: 3, greedy: true, limit: 1, temperature: 256, seed: 1, raw_tokens: false, stop: Vec::new(), draft_depth: 0, draft_logits: 0, think: None, think_budget: u32::MAX };
    let info = chip_info(&config.die, code, layout.logits_register(&program), program.vocab as u32, &setup);
    let mut segments = image_segments(&config.die, code, layout);
    segments.extend(cell_segments(&config.die, layout, &weights).unwrap());
    let image = build_image(&firmware.program, &firmware.data, &info, &tables, &segments).unwrap();
    let input = netlist::port::text("Once upon a time");

    let engine = VmEngine::new(program.clone(), Weights::open(&dir).unwrap(), config.die, context).unwrap().with_table_outside().unwrap();
    let model = ChipModel::new(engine, code.clone(), SamplerModel { config: config.sampler, rng: 0, table: exponential_table() });
    let mut bus = SocBus::booted_with(config.soc, &image, &input, model).unwrap();
    let mut cpu = Rv32Iss::default();
    while !cpu.halted {
        cpu.step(&mut bus);
        assert!(cpu.cycle < 100_000_000, "the firmware does not finish");
    }
    let text = netlist::port::answer_texts(&bus.tx)[0].clone();
    eprintln!("instruction-level simulator: {} cycles, {text:?}", cpu.cycle);

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
    eprintln!("the host with the die's machine: {by_host:?}; floating point: {float:?}");
    assert_eq!(text.trim_start(), by_host.trim_start(), "the firmware writes what the die's machine writes");
    assert_eq!(by_host, float, "the die's arithmetic picks the floating point's tokens");

    let started = std::time::Instant::now();
    let (output, cycles, _) = run_gates(&design, &image, &input, cpu.cycle + 4096);
    eprintln!("gates: {cycles} cycles in {:.0} s", started.elapsed().as_secs_f64());
    assert_eq!(String::from_utf8_lossy(&output), String::from_utf8_lossy(&bus.tx));
    assert_eq!(cycles, cpu.cycle, "cycles");
}

/// Boots a chip in gates with `input` sent while it boots, leaving it ready to run.
fn boot(sim: &mut Simulator, image: &[u32], input: &[u8], pending: &mut Option<u32>) {
    let serve = |sim: &mut Simulator, pending: &mut Option<u32>| {
        sim.set("image_resp_valid", pending.is_some() as u64).unwrap();
        sim.set("image_resp_data", pending.unwrap_or(0) as u64).unwrap();
        sim.settle();
        *pending = (sim.get("image_req_valid").unwrap() == 1).then(|| image[sim.get("image_req_addr").unwrap() as usize]);
    };
    sim.set("reset", 1).unwrap();
    sim.set("cpu_hold", 1).unwrap();
    sim.tick();
    sim.set("reset", 0).unwrap();
    for _ in 0..=netlist::chip::RESET_RELEASE {
        sim.tick();
    }
    for byte in input {
        sim.set("host_rx_valid", 1).unwrap();
        sim.set("host_rx_data", *byte as u64).unwrap();
        serve(sim, pending);
        sim.tick();
    }
    sim.set("host_rx_valid", 0).unwrap();
    while {
        serve(sim, pending);
        sim.get("booting").unwrap() == 1
    } {
        sim.tick();
    }
    sim.set("cpu_hold", 0).unwrap();
}

/// Two chips in a chain on their pins, as stages of a system are joined: the first's
/// output (`host_tx_valid`, `host_tx_data`) drives the second's input (`host_rx_valid`,
/// `host_rx_data`), and the second's `host_rx_ready` holds the first back when it is full.
/// Between them a little logic reads the first's messages as they go by: its text goes on as
/// the second's question, framing and all, the rest is dropped, and the first's answer's end
/// closes the question. The second writes what it writes when a host gives it the first's
/// answer.
#[test]
#[ignore = "slow: two whole chips in gates, one after the other and then in a chain"]
fn two_chips_in_a_chain_pass_their_text_on_their_pins() {
    if build_firmware().is_none() {
        return;
    }
    let dir = std::env::temp_dir().join(format!("llm2chip-chip-chain-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    make_checkpoint(&dir);
    let program = compile_qwen3_5(CONFIG).unwrap();
    let base = netlist::DieConfig { microcode_bits: 15, ..cell_die_config() };
    let layout = Layout::with_cells(&program, &Weights::open(&dir).unwrap(), base, CHIP_TOKENS).unwrap();
    let code = layout.microprogram(&program).unwrap();
    let die = sim::chip::with_code_dictionary(sim::chip::compact_die(base, &layout), &code);
    let config = ChipConfig { soc: SocConfig { program_words: netlist::soc::PROGRAM_WORDS, data_bits: 14, fifo_bits: 6, links: Default::default() }, die, sampler: SamplerConfig { logit_bits: 16, index_bits: 12, entries: 4, lanes: 1 } };
    let design = chip_design(config);
    let (image, input) = image_and_input(config, &layout, &code, CONFIG, &dir);
    const LIMIT: u64 = 20_000_000;

    // Each alone, the second given the first's answer by a host.
    let (first, _, _) = run_gates(&design, &image, &input, LIMIT);
    let answer = netlist::port::answer_texts(&first)[0].clone().into_bytes();
    assert!(!answer.is_empty(), "the first chip answers");
    let prompt = netlist::port::frame(netlist::port::kind::TEXT, &answer);
    let (expected, _, _) = run_gates(&design, &image, &prompt, LIMIT);
    eprintln!("alone: the first answers {:?}, the second {:?}", String::from_utf8_lossy(&answer), netlist::port::all_text(&expected));

    // In a chain: between them, the first's messages are read on the way, its text goes on
    // as the second's question and its answer's end closes it.
    let (mut a, mut b) = (Simulator::new(&design).unwrap(), Simulator::new(&design).unwrap());
    let (mut pending_a, mut pending_b) = (None, None);
    boot(&mut a, &image, &input, &mut pending_a);
    boot(&mut b, &image, &[], &mut pending_b);
    b.set("host_tx_ready", 1).unwrap();
    use netlist::port::kind;
    use netlist::soc::FRAME_END;
    // The kind of the first's message going by, the question opened on the second, and the
    // end that closes it, waiting to go.
    let (mut going, mut opened, mut closing, mut passed, mut output) = (None::<u8>, false, false, Vec::new(), Vec::new());
    for cycle in 0.. {
        assert!(cycle < LIMIT, "the chain does not finish");
        for (sim, pending) in [(&mut a, &mut pending_a), (&mut b, &mut pending_b)] {
            sim.set("image_resp_valid", pending.is_some() as u64).unwrap();
            sim.set("image_resp_data", pending.unwrap_or(0) as u64).unwrap();
        }
        // The second's room decides whether the first's text may go this cycle.
        b.set("host_rx_valid", 0).unwrap();
        b.settle();
        let room = b.get("host_rx_ready").unwrap() == 1;
        a.settle();
        let (valid, byte) = (a.get("host_tx_valid").unwrap() == 1, a.get("host_tx_data").unwrap() as u8);
        let text_byte = valid && going == Some(kind::TEXT) && byte != FRAME_END;
        let (take, send) = if closing {
            (false, room.then(|| { closing = false; FRAME_END }))
        } else if text_byte && !opened {
            (false, room.then(|| { opened = true; kind::TEXT }))
        } else if text_byte {
            (room, room.then_some(byte))
        } else {
            (valid, None)
        };
        a.set("host_tx_ready", take as u64).unwrap();
        a.settle();
        if take {
            match going {
                None => going = Some(byte),
                Some(k) if byte == FRAME_END => {
                    going = None;
                    closing = k == kind::END;
                }
                _ => {}
            }
        }
        if let Some(byte) = send.filter(|_| text_byte && take) {
            passed.push(byte);
        }
        b.set("host_rx_valid", send.is_some() as u64).unwrap();
        b.set("host_rx_data", send.unwrap_or(0) as u64).unwrap();
        b.settle();
        if b.get("host_tx_valid").unwrap() == 1 {
            output.push(b.get("host_tx_data").unwrap() as u8);
        }
        if b.get("halted").unwrap() == 1 {
            break;
        }
        for (sim, pending) in [(&mut a, &mut pending_a), (&mut b, &mut pending_b)] {
            *pending = (sim.get("image_req_valid").unwrap() == 1).then(|| image[sim.get("image_req_addr").unwrap() as usize]);
            sim.tick();
        }
    }
    eprintln!("in a chain: {:?} passed on, the second writes {:?}", String::from_utf8_lossy(&passed), netlist::port::all_text(&output));
    assert_eq!(passed, answer, "what went from one chip to the other");
    assert_eq!(output, expected, "the second chip's answer");
    std::fs::remove_dir_all(&dir).ok();
}
