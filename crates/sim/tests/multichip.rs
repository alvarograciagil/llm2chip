//! Several identical chips on a simulated board: one design, told apart by its pins.

use compiler::compile_qwen3_5;
use netlist::chip::{chip_design, chip_info, ChipConfig, ChipModel, GenerationSetup, DEBUG_LINK_CORRECTED, DEBUG_LINK_FAILED, RESET_STRETCH};
use netlist::image::build_image;
use netlist::rv32::Rv32Iss;
use netlist::sampler::{exponential_table, SamplerConfig, SamplerModel};
use netlist::soc::{FirmwareImage, SocBus, SocConfig};
use netlist::{DieConfig, Simulator};
use sim::chip::{image_segments, VmEngine};
use sim::lower::Layout;
use sim::tokenizer::TokenizerTables;
use sim::Weights;
use std::collections::VecDeque;
use std::path::Path;
use std::process::Command;

mod common;
use common::*;

const CHIP_TOKENS: usize = 8;
const CHIPS: usize = 2;
const WIRE_DELAY: usize = 2;

fn build_firmware() -> Option<FirmwareImage> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../firmware");
    let built = Command::new("cargo").current_dir(&dir).env_remove("RUSTFLAGS").env_remove("CARGO_ENCODED_RUSTFLAGS").args(["build", "--release"]).output().ok()?;
    built.status.success().then(|| FirmwareImage::from_elf(&std::fs::read(dir.join("target/riscv32im-unknown-none-elf/release/firmware")).unwrap()).unwrap())
}

struct Noise(u64);

impl Noise {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

/// One direction of a link on the board: wires with delay, gaps and the odd flipped bit.
struct Wire {
    from: usize,
    to: usize,
    prefix: (&'static str, &'static str),
    forward: VecDeque<(u64, u128)>,
    backward: VecDeque<u64>,
    width: u32,
}

impl Wire {
    fn new(from: usize, to: usize, prefix: (&'static str, &'static str), width: u32) -> Self {
        Self { from, to, prefix, forward: VecDeque::from(vec![(0, 0); WIRE_DELAY]), backward: VecDeque::from(vec![0; WIRE_DELAY]), width }
    }

    /// Drives the receiving pins with what left the other chip `WIRE_DELAY` cycles ago.
    fn drive(&mut self, chips: &mut [Simulator], noise: &mut Noise, flips: &mut u32) {
        let (mut valid, mut bits) = self.forward.pop_front().unwrap();
        if valid == 1 && noise.next().is_multiple_of(40) {
            bits ^= 1 << (noise.next() % self.width as u64);
            *flips += 1;
        }
        chips[self.to].set(&format!("{}_valid", self.prefix.1), valid).unwrap();
        chips[self.to].set_wide(&format!("{}_bits", self.prefix.1), bits).unwrap();
        // Now and then the board holds the sender back for a cycle.
        let ready = self.backward.pop_front().unwrap() & u64::from(!noise.next().is_multiple_of(5));
        chips[self.from].set(&format!("{}_ready", self.prefix.0), ready).unwrap();
        valid = 0;
        let _ = valid;
    }

    fn sample(&mut self, chips: &[Simulator]) {
        self.forward.push_back((chips[self.from].get(&format!("{}_valid", self.prefix.0)).unwrap(), chips[self.from].get_wide(&format!("{}_bits", self.prefix.0)).unwrap()));
        self.backward.push_back(chips[self.to].get(&format!("{}_ready", self.prefix.1)).unwrap());
    }
}

#[test]
#[cfg_attr(debug_assertions, ignore = "dos chips completos en puertas: solo con optimizaciones")]
fn two_identical_chips_generate_the_same_text_as_one() {
    let Some(firmware) = build_firmware() else { return };
    let dir = std::env::temp_dir().join(format!("llm2chip-board-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    make_checkpoint(&dir);
    let program = compile_qwen3_5(CONFIG).unwrap();
    let die = DieConfig { microcode_bits: 15, ..die_config() };
    let soc = SocConfig { program_bits: 12, data_bits: 14, fifo_bits: 6 };
    let config = ChipConfig { soc, die, sampler: SamplerConfig { logit_bits: 16, index_bits: 12, entries: 4, lanes: 1 } };
    let tables = TokenizerTables { byte_symbols: (0..256).map(|b| b % 8).collect(), letters: vec![(97, 122)], token_bytes: (0..8).map(|t| vec![b'a' + t]).collect(), text_capacity: 64, ..Default::default() };
    let input = b"hoy\0";
    let setup = |dies: u32| GenerationSetup { max_new: 1, greedy: false, limit: 4, temperature: 256, seed: 0x1234_5678, raw_tokens: false, stop: [u32::MAX, u32::MAX], dies };

    // One chip at instruction level is the reference.
    let single = Layout::new(&program, &Weights::open(&dir).unwrap(), die, Some(CHIP_TOKENS)).unwrap();
    let code = single.microprogram(&program).unwrap();
    let logits_register = single.logits_register(&program);
    let info = chip_info(&die, &code, logits_register, 8, &setup(1));
    let image = build_image(&firmware.program, &firmware.data, &info, &tables.to_words(), &image_segments(&die, &code, &single)).unwrap();
    let engine = VmEngine::new(compile_qwen3_5(CONFIG).unwrap(), Weights::open(&dir).unwrap(), die, CHIP_TOKENS).unwrap();
    let model = ChipModel::new(engine, code.clone(), SamplerModel { config: config.sampler, rng: 0, table: exponential_table() });
    let mut bus = SocBus::booted_with(soc, &image, input, model).unwrap();
    let mut cpu = Rv32Iss::default();
    while !cpu.halted {
        cpu.step(&mut bus);
    }
    let reference = String::from_utf8_lossy(&bus.tx).trim_start_matches('.').to_string();
    assert_eq!(reference.len(), 1);

    // Two chips built from one design, each with its share of the model.
    let plan = program.split(CHIPS).unwrap();
    let design = chip_design(config);
    let mut images = Vec::new();
    for chip in 0..CHIPS {
        let mut layout = Layout::new(&program, &Weights::open(&dir).unwrap(), die, Some(CHIP_TOKENS)).unwrap();
        layout.keep_weights_of(&program, &plan, chip);
        let code = layout.die_microprogram(&program, &plan, chip).unwrap();
        let info = chip_info(&die, &code, logits_register, 8, &setup(CHIPS as u32));
        images.push(build_image(&firmware.program, &firmware.data, &info, &tables.to_words(), &image_segments(&die, &code, &layout)).unwrap());
    }

    let mut chips: Vec<Simulator> = (0..CHIPS).map(|_| Simulator::new(&design).unwrap()).collect();
    for (id, chip) in chips.iter_mut().enumerate() {
        chip.set("die_id", id as u64).unwrap();
        chip.set("reset", 1).unwrap();
        chip.tick();
        chip.set("reset", 0).unwrap();
        for _ in 0..=RESET_STRETCH {
            chip.tick();
        }
        chip.set("host_tx_ready", 1).unwrap();
    }
    let mut wires = vec![
        Wire::new(0, 1, ("link_tx", "link_rx"), netlist::link::wire_bits(16)),
        Wire::new(1, 0, ("link_tx", "link_rx"), netlist::link::wire_bits(16)),
        Wire::new(0, 1, ("chain_tx", "chain_rx"), netlist::link::wire_bits(32)),
    ];
    let mut pending: Vec<Option<u32>> = vec![None; CHIPS];
    let mut noise = Noise(0x0BAD_C0DE_1234_5678);
    let mut flips = 0;
    let mut output = Vec::new();
    let mut fed = 0;
    let mut cycles = 0u64;
    loop {
        for wire in &mut wires {
            wire.drive(&mut chips, &mut noise, &mut flips);
        }
        for (id, chip) in chips.iter_mut().enumerate() {
            chip.set("image_resp_valid", pending[id].is_some() as u64).unwrap();
            chip.set("image_resp_data", pending[id].unwrap_or(0) as u64).unwrap();
        }
        let feeding = fed < input.len() && chips[0].get("host_rx_ready") == Some(1);
        chips[0].set("host_rx_valid", feeding as u64).unwrap();
        chips[0].set("host_rx_data", input.get(fed).copied().unwrap_or(0) as u64).unwrap();
        fed += feeding as usize;
        for (id, chip) in chips.iter_mut().enumerate() {
            chip.settle();
            pending[id] = (chip.get("image_req_valid").unwrap() == 1).then(|| images[id][chip.get("image_req_addr").unwrap() as usize]);
        }
        if chips[0].get("host_tx_valid").unwrap() == 1 {
            output.push(chips[0].get("host_tx_data").unwrap() as u8);
        }
        if chips[0].get("halted").unwrap() == 1 {
            break;
        }
        for wire in &mut wires {
            wire.sample(&chips);
        }
        for chip in &mut chips {
            chip.tick();
        }
        cycles += 1;
        assert!(cycles < 8_000_000, "la placa no termina; salida {:?}", String::from_utf8_lossy(&output));
    }
    let text = String::from_utf8_lossy(&output);
    eprintln!("{cycles} ciclos, {flips} bits invertidos en los enlaces, salida {text:?}");
    assert_eq!(text.trim_start_matches('.'), reference, "el texto debe ser el mismo que con un solo chip");
    assert!(flips > 0, "la prueba debe inyectar errores");

    let debug = |chip: &mut Simulator, what: u64| {
        chip.set("debug_select", what).unwrap();
        chip.settle();
        chip.get("debug_value").unwrap()
    };
    let corrected: u64 = chips.iter_mut().map(|chip| debug(chip, DEBUG_LINK_CORRECTED)).sum();
    let failed: u64 = chips.iter_mut().map(|chip| debug(chip, DEBUG_LINK_FAILED)).sum();
    assert_eq!(failed, 0);
    assert_eq!(corrected, flips as u64, "cada bit invertido debe constar como corregido");

    for index in 0..8 {
        let address = (logits_register as usize) << die.index_bits | index;
        let gates = chips[0].peek_memory("/die/vectors_a", address).unwrap() as u64 & 0xFFFF;
        let exact = netlist::chip::DieEngine::read(&mut bus.peripheral.engine, address as u32) as u64 & 0xFFFF;
        assert_eq!(gates, exact, "logit {index}");
    }
    std::fs::remove_dir_all(&dir).ok();
}
