//! The image encoder run by the chip's own firmware: the host sends the picture's patches
//! before the prompt, and the firmware orders the encoder's routines, puts the image's
//! tokens in the placeholder's place with their M-RoPE rotations and shifts the rotations of
//! the text after it. Checked against the host ordering the same routines on the same
//! machine, and the instruction-level chip against the gates.

mod common;
use common::*;
use compiler::{attach_vision, compile_qwen3_5, Program};
use netlist::chip::{add_vision, chip_design, chip_info, ChipConfig, ChipModel, DieEngine, GenerationSetup, VisionInfo};
use netlist::image::build_image;
use netlist::rv32::Rv32Iss;
use netlist::sampler::{exponential_table, SamplerConfig, SamplerModel};
use netlist::soc::{SocBus, SocConfig};
use netlist::{DieConfig, Microprogram, Simulator};
use sim::chip::{cell_segments, image_segments, VmEngine};
use sim::lower::{Layout, ENTRY_LOGITS, ENTRY_TOKEN, ENTRY_TOKEN_FROM_IMAGE, ENTRY_VISION};
use sim::vision::{patches, Patches, VisionTower};
use sim::vision_chip::picture_bytes;
use sim::Weights;
use std::path::PathBuf;

/// Enough positions for every patch's keys, and for the prompt and the answer.
const CHIP_TOKENS: usize = 32;
const NEW_TOKENS: usize = 3;
/// Text, the image's placeholder, more text.
const PROMPT: [u32; 5] = [1, 2, 7, 3, 4];

struct Chip {
    dir: PathBuf,
    program: Program,
    die: DieConfig,
    layout: Layout,
    code: Microprogram,
    vision: VisionInfo,
    tower: VisionTower,
    image: Patches,
}

fn chip(tag: &str) -> Chip {
    let dir = std::env::temp_dir().join(format!("llm2chip-vision-chip-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    make_checkpoint(&dir);
    make_vision_checkpoint(&dir);
    let tower = VisionTower::open(&dir).unwrap().unwrap();
    // 16 × 8 pixels: 8 × 4 patches, a grid of 4 × 2 tokens.
    let image = patches(&tower.config, &test_picture(16, 8), 16, 8).unwrap();
    let config = std::fs::read_to_string(dir.join("config.json")).unwrap();
    let program = attach_vision(compile_qwen3_5(CONFIG).unwrap(), &config, image.rows * image.cols).unwrap();
    let weights = Weights::open_with_vision(&dir).unwrap();
    // The toy encoder's weights are plain floats, as the real one's are BF16: with no format
    // of their own they go into the cells whole, a mantissa of the vectors' width each.
    let base = cell_die_config();
    let roomy = DieConfig { microcode_bits: 15, reg_bits: 12, cell: netlist::CellConfig { weight_bits: base.cell.activation_bits, ..base.cell }, features: base.features.with_host_writes(true).with_key_shift(5), ..base };
    // As many registers as the routines reach, once assembled.
    let wide = Layout::with_cells(&program, &weights, roomy, CHIP_TOKENS).unwrap();
    wide.microprogram(&program).unwrap();
    let die = sim::chip::with_registers_needed(roomy, &wide);
    let layout = Layout::with_cells(&program, &weights, die, CHIP_TOKENS).unwrap();
    let code = layout.microprogram(&program).unwrap();
    // Each register takes only the vector memory its routines reach, and the microcode keeps
    // each distinct word once: the firmware orders routines by their number in it.
    let die = sim::chip::with_code_dictionary(sim::chip::compact_die(die, &layout), &code);
    let vision = layout.vision_info(&program, &tower.config).expect("el chip lleva el codificador");
    Chip { dir, program, die, layout, code, vision, tower, image }
}

fn engine(chip: &Chip) -> VmEngine {
    VmEngine::new(chip.program.clone(), Weights::open_with_vision(&chip.dir).unwrap(), chip.die, CHIP_TOKENS).unwrap()
}

fn logits(engine: &mut VmEngine, chip: &Chip) -> Vec<i64> {
    let base = chip.layout.logits_register(&chip.program) << chip.die.index_bits;
    (0..chip.program.vocab as u32).map(|i| engine.read(base + i)).collect()
}

fn greedy(engine: &mut VmEngine, chip: &Chip) -> u32 {
    let logits = logits(engine, chip);
    let mut sampler = SamplerModel { config: sampler_config(), rng: 0, table: exponential_table() };
    sampler.sample(&logits, 0xFFFF, 1, true).token
}

fn sampler_config() -> SamplerConfig {
    SamplerConfig { logit_bits: 16, index_bits: 12, entries: 4, lanes: 1 }
}

/// The host does what the firmware should: writes each patch and orders the encoder's
/// routines, and writes each image token's rotation, computed from its three positions, and
/// the text's after it. Gives the tokens the model then writes, picked greedily, and the
/// logits of the last.
fn host_ordered(chip: &Chip) -> (Vec<u32>, Vec<i64>) {
    let mut engine = engine(chip);
    let (v, bits) = (chip.vision, chip.die.cell.activation_bits);
    let count = (chip.image.rows * chip.image.cols) as u32;
    let write = |engine: &mut VmEngine, address: u32, values: &[i64]| values.iter().enumerate().for_each(|(i, value)| engine.write(address + i as u32, *value));
    let positions = chip.tower.positions(chip.image.rows, chip.image.cols).unwrap();
    let angles = sim::vision::patch_angles(&chip.tower.config, chip.image.rows, chip.image.cols);
    for (patch, ((pixels, place), angles)) in chip.image.values.chunks(v.pixels.2 as usize).zip(&positions).zip(&angles).enumerate() {
        for (vector, values) in [(v.pixels, pixels), (v.position, place.as_slice())] {
            let (mantissas, exponent) = netlist::block::encode_vector(&values.iter().map(|x| *x as f64).collect::<Vec<_>>(), bits);
            write(&mut engine, vector.1, &[exponent]);
            write(&mut engine, vector.0, &mantissas);
        }
        let row = chip.layout.patch_rotation(&angles.iter().map(|a| *a as f64).collect::<Vec<_>>());
        write(&mut engine, v.patch_rope.0 + patch as u32 * v.patch_rope.1, &row);
        engine.run(ENTRY_VISION, patch as u32, patch as u32);
    }
    for block in 0..v.blocks as usize {
        (0..count).for_each(|p| engine.run(ENTRY_VISION + 1 + 2 * block, p, p));
        (0..count).for_each(|p| engine.run(ENTRY_VISION + 2 + 2 * block, p, count - 1));
    }
    let merge = ENTRY_VISION + 1 + 2 * v.blocks as usize;
    let section = chip.tower.config.mrope_section;
    let width = 3 * v.rope_half + 1;
    let rotate = |engine: &mut VmEngine, slot: u32, positions: [u64; 3]| {
        let row = chip.layout.text_rotation(&chip.program, positions, section);
        row.iter().enumerate().for_each(|(i, value)| engine.write(v.rope + slot * width + i as u32, *value));
    };
    let (mut position, mut behind, mut last) = (0u32, 0u32, 0);
    for token in PROMPT {
        if token == v.image_token {
            let (image_positions, next) = sim::vision::image_positions(position as u64, chip.image.rows, chip.image.cols, chip.tower.config.merge);
            for (index, three) in image_positions.iter().enumerate() {
                engine.run(merge, index as u32, index as u32);
                rotate(&mut engine, position, *three);
                engine.run(ENTRY_TOKEN_FROM_IMAGE, 0, position);
                position += 1;
            }
            behind = position - next as u32;
            continue;
        }
        if behind > 0 {
            rotate(&mut engine, position, [(position - behind) as u64; 3]);
        }
        engine.run(ENTRY_TOKEN, token, position);
        (last, position) = (token, position + 1);
    }
    assert!(behind > 0, "la imagen ocupa más posiciones de caché que de rotación");
    let mut written = Vec::new();
    for _ in 0..NEW_TOKENS {
        engine.run(ENTRY_LOGITS, last, position - 1);
        last = greedy(&mut engine, chip);
        written.push(last);
        rotate(&mut engine, position, [(position - behind) as u64; 3]);
        engine.run(ENTRY_TOKEN, last, position);
        position += 1;
    }
    (written, logits(&mut engine, chip))
}

/// The chip's image and what the host sends it: the picture, then the prompt as tokens.
fn image_and_input(chip: &Chip) -> (Vec<u32>, Vec<u8>) {
    let firmware = firmware().expect("firmware");
    let setup = GenerationSetup { max_new: NEW_TOKENS as u32, greedy: true, limit: 1, temperature: 256, seed: 1, raw_tokens: true, stop: Vec::new(), draft_depth: 0, draft_logits: 0, think: None, think_budget: u32::MAX };
    let mut info = chip_info(&chip.die, &chip.code, chip.layout.logits_register(&chip.program), chip.program.vocab as u32, &setup);
    add_vision(&mut info, &chip.vision);
    let mut segments = image_segments(&chip.die, &chip.code, &chip.layout);
    segments.extend(cell_segments(&chip.die, &chip.layout, &Weights::open_with_vision(&chip.dir).unwrap()).unwrap());
    let image = build_image(&firmware.program, &firmware.data, &info, &[], &segments).unwrap();
    let mut input = picture_bytes(&chip.tower, Some(&chip.image), chip.die.cell.activation_bits).unwrap();
    input.extend(PROMPT.iter().chain(&[u32::MAX]).flat_map(|t| t.to_le_bytes()));
    (image, input)
}

fn soc() -> SocConfig {
    SocConfig { program_bits: 13, data_bits: 14, fifo_bits: 6 }
}

/// The chip at instruction level: its output bytes and cycles, and the last logits.
fn instruction_level(chip: &Chip, image: &[u32], input: &[u8]) -> (Vec<u8>, u64, Vec<i64>) {
    let model = ChipModel::new(engine(chip), chip.code.clone(), SamplerModel { config: sampler_config(), rng: 0, table: exponential_table() });
    let mut bus = SocBus::booted_with(soc(), image, input, model).unwrap();
    let mut cpu = Rv32Iss::default();
    while !cpu.halted {
        cpu.step(&mut bus);
        assert!(cpu.cycle < 200_000_000, "el firmware no termina");
    }
    (bus.tx.clone(), cpu.cycle, logits(&mut bus.peripheral.engine, chip))
}

fn tokens_of(bytes: &[u8]) -> Vec<u32> {
    // Each segment loaded at boot reports a dot first.
    let start = bytes.iter().position(|b| *b != b'.').unwrap_or(bytes.len());
    bytes[start..].chunks(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

#[test]
fn the_firmware_reads_the_picture_and_the_model_answers_as_the_host_ordered_it() {
    if firmware().is_none() {
        return;
    }
    let chip = chip("iss");
    let (expected, last_logits) = host_ordered(&chip);
    let (image, input) = image_and_input(&chip);
    let (output, cycles, logits) = instruction_level(&chip, &image, &input);
    eprintln!("{cycles} ciclos, tokens {:?}", tokens_of(&output));
    assert_eq!(tokens_of(&output), expected);
    assert_eq!(logits, last_logits, "los logits del último token");
    std::fs::remove_dir_all(&chip.dir).ok();
}

/// Runs the whole chip in gates, the host sending a byte whenever the chip can take it.
fn gates(design: &netlist::Design, image: &[u32], input: &[u8], limit: u64) -> (Vec<u8>, u64) {
    let mut sim = Simulator::new(design).unwrap();
    sim.set("reset", 1).unwrap();
    sim.set("cpu_hold", 1).unwrap();
    sim.tick();
    sim.set("reset", 0).unwrap();
    for _ in 0..=netlist::chip::RESET_STRETCH {
        sim.tick();
    }
    let mut pending = None;
    let mut sent = 0;
    let mut step = |sim: &mut Simulator, pending: &mut Option<u32>| {
        sim.set("image_resp_valid", pending.is_some() as u64).unwrap();
        sim.set("image_resp_data", pending.unwrap_or(0) as u64).unwrap();
        let offer = sent < input.len();
        sim.set("host_rx_valid", offer as u64).unwrap();
        sim.set("host_rx_data", input.get(sent).copied().unwrap_or(0) as u64).unwrap();
        sim.settle();
        if offer && sim.get("host_rx_ready").unwrap() == 1 {
            sent += 1;
        }
        *pending = (sim.get("image_req_valid").unwrap() == 1).then(|| image[sim.get("image_req_addr").unwrap() as usize]);
    };
    let mut boot = 0;
    loop {
        step(&mut sim, &mut pending);
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
        step(&mut sim, &mut pending);
        if sim.get("host_tx_valid").unwrap() == 1 {
            output.push(sim.get("host_tx_data").unwrap() as u8);
        }
        if sim.get("halted").unwrap() == 1 {
            return (output, cycle + 1);
        }
        sim.tick();
    }
    sim.set("debug_select", netlist::chip::DEBUG_CPU_PC).unwrap();
    sim.settle();
    panic!("el firmware no terminó en {limit} ciclos: salida {output:?}, CPU en {:#x}, {sent} de {} bytes enviados", sim.get("debug_value").unwrap(), input.len());
}

#[test]
#[ignore = "lenta: el chip entero en puertas codifica la imagen y genera, millones de ciclos"]
fn the_whole_chip_reads_a_picture_in_gates_as_at_instruction_level() {
    if firmware().is_none() {
        return;
    }
    let chip = chip("gates");
    let (image, input) = image_and_input(&chip);
    let (expected, cycles, _) = instruction_level(&chip, &image, &input);
    let config = ChipConfig { soc: soc(), die: chip.die, sampler: sampler_config() };
    let started = std::time::Instant::now();
    let (output, gate_cycles) = gates(&chip_design(config), &image, &input, cycles + 4096);
    eprintln!("puertas: {gate_cycles} ciclos en {:.0} s", started.elapsed().as_secs_f64());
    assert_eq!(output, expected);
    assert_eq!(gate_cycles, cycles, "ciclos");
    std::fs::remove_dir_all(&chip.dir).ok();
}

/// The app's whole chip: the real firmware drives a die run in software, the exact engine
/// that carries the encoder, and the host sends the picture over the chip's port as before.
#[test]
fn the_firmware_drives_a_software_die_that_reads_the_picture() {
    use netlist::die::Parameter;
    use sim::chip::{ImageTokens, Placement, RunnerEngine, FLOW_INDEX_BITS, FLOW_REGISTER_BITS};
    use sim::Runner;
    let Some(firmware) = firmware() else { return };
    let chip = chip("software");
    let merge = chip.tower.config.merge;
    let open = || {
        let mut engine = sim::ExactEngine::new(chip.program.clone(), Weights::open_with_vision(&chip.dir).unwrap(), 16).unwrap();
        engine.set_mrope(chip.tower.config.mrope_section);
        let count = Runner::encode_image(&mut engine, &chip.tower, &chip.image).unwrap().expect("lleva el codificador");
        (engine, count)
    };

    // The host steps the image's tokens at their three positions and picks greedily.
    let (mut engine, count) = open();
    let (mut position, mut logits) = (0u64, Vec::new());
    for token in PROMPT {
        if token == chip.vision.image_token {
            let (positions, next) = sim::vision::image_positions(position, chip.image.rows, chip.image.cols, merge);
            for (index, three) in positions.into_iter().enumerate() {
                logits = engine.step_image(index as u32, three, true).unwrap().unwrap();
            }
            engine.continue_text_at(next);
            position += count as u64;
            continue;
        }
        logits = sim::ExactEngine::step(&mut engine, token, true).unwrap().unwrap();
        position += 1;
    }
    let argmax = |l: &[f32]| l.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0 as u32;
    let mut expected = Vec::new();
    let mut chosen_from = Vec::new();
    for _ in 0..NEW_TOKENS {
        let next = argmax(&logits);
        expected.push(next);
        chosen_from = logits.clone();
        logits = sim::ExactEngine::step(&mut engine, next, true).unwrap().unwrap();
    }

    // The firmware, with the chip description the app gives a software die.
    let (mut engine, count) = open();
    let die = DieConfig { reg_bits: FLOW_REGISTER_BITS, index_bits: FLOW_INDEX_BITS, ..chip.die };
    let parameter = |register, token, position| Parameter { register, token, position, constant: 0, window_length: 0, window_slot: 0, window: 0 };
    let code = Microprogram { words: Vec::new(), entries: (0..ENTRY_VISION as u32 + 2).collect(), parameters: vec![parameter(1, 1, 0), parameter(2, 0, 1)] };
    let setup = GenerationSetup { max_new: NEW_TOKENS as u32, greedy: true, limit: 1, temperature: 256, seed: 1, raw_tokens: true, stop: Vec::new(), draft_depth: 0, draft_logits: 0, think: None, think_budget: u32::MAX };
    let mut info = chip_info(&die, &code, 1, chip.program.vocab as u32, &setup);
    add_vision(&mut info, &sim::vision_chip::software_vision_info(&chip.tower.config, count * merge * merge, FLOW_INDEX_BITS, FLOW_REGISTER_BITS));
    let image = build_image(&firmware.program, &firmware.data, &info, &[], &[]).unwrap();
    let mut input = picture_bytes(&chip.tower, Some(&chip.image), 16).unwrap();
    input.extend(PROMPT.iter().chain(&[u32::MAX]).flat_map(|t| t.to_le_bytes()));
    let runner = RunnerEngine::new(&mut engine, 16, 0).with_image(ImageTokens::OnChip(count), Placement::Grid(chip.image.rows, chip.image.cols, merge), chip.vision.image_token);
    let model = ChipModel::new(runner, code, SamplerModel { config: sampler_config(), rng: 0, table: exponential_table() });
    let mut bus = SocBus::booted_with(SocConfig { program_bits: 14, data_bits: 14, fifo_bits: 6 }, &image, &input, model).unwrap();
    let mut cpu = Rv32Iss::default();
    while !cpu.halted {
        cpu.step(&mut bus);
        assert!(bus.peripheral.engine.error.is_none(), "{:?}", bus.peripheral.engine.error);
        assert!(cpu.cycle < 200_000_000, "el firmware no termina");
    }
    assert_eq!(tokens_of(&bus.tx), expected);
    assert_eq!(bus.peripheral.engine.tokens[..PROMPT.len()], PROMPT, "el die ve el texto con la imagen en su sitio");
    // The logits the last token was picked from, in the die's format.
    let (mantissas, _) = netlist::block::encode_vector(&chosen_from.iter().map(|v| *v as f64).collect::<Vec<_>>(), 16);
    let read: Vec<i64> = (0..chip.program.vocab as u32).map(|i| bus.peripheral.engine.read(1 << FLOW_INDEX_BITS | i)).collect();
    assert_eq!(read, mantissas, "los logits del último token");
    std::fs::remove_dir_all(&chip.dir).ok();
}
