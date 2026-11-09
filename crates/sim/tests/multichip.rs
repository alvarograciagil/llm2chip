//! Several identical chips on a simulated board: one design, told apart by its pins.

use compiler::compile_qwen3_5;
use netlist::chip::{chip_design, ChipConfig, ChipModel, GenerationSetup, DEBUG_LINK_CORRECTED, DEBUG_LINK_FAILED, RESET_STRETCH};
use netlist::image::build_plan_image;
use netlist::rv32::Rv32Iss;
use netlist::sampler::{exponential_table, SamplerConfig, SamplerModel};
use netlist::soc::{FirmwareImage, SocBus, SocConfig};
use netlist::{Design, DieConfig, Simulator};
use sim::chip::{plan_parts, VmEngine};
use sim::lower::Layout;
use sim::tokenizer::TokenizerTables;
use sim::Weights;
use std::collections::VecDeque;
use std::path::Path;
use std::process::Command;

mod common;
use common::*;

const CHIP_TOKENS: usize = 8;
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

/// Chips joined in a ring on a board, all reading the same image.
struct Board<'a> {
    chips: Vec<Simulator>,
    wires: Vec<Wire>,
    pending: Vec<Option<u32>>,
    image: &'a [u32],
    noise: Noise,
    flips: u32,
    input: &'a [u8],
    fed: usize,
    output: Vec<u8>,
    cycles: u64,
}

impl<'a> Board<'a> {
    fn new(design: &Design, count: usize, image: &'a [u32], input: &'a [u8]) -> Self {
        let mut chips: Vec<Simulator> = (0..count).map(|_| Simulator::new(design).unwrap()).collect();
        for (index, chip) in chips.iter_mut().enumerate() {
            chip.set("follower", (index != 0) as u64).unwrap();
            chip.set("reset", 1).unwrap();
            chip.tick();
            chip.set("reset", 0).unwrap();
            for _ in 0..=RESET_STRETCH {
                chip.tick();
            }
            chip.set("host_tx_ready", 1).unwrap();
        }
        let mut wires = Vec::new();
        if count > 1 {
            for from in 0..count {
                let to = (from + 1) % count;
                wires.push(Wire::new(from, to, ("link_tx", "link_rx"), netlist::link::wire_bits(16)));
                wires.push(Wire::new(from, to, ("chain_tx", "chain_rx"), netlist::link::wire_bits(32)));
            }
        }
        Self { chips, wires, pending: vec![None; count], image, noise: Noise(0x0BAD_C0DE_1234_5678), flips: 0, input, fed: 0, output: Vec::new(), cycles: 0 }
    }

    /// Runs until the leading chip stops.
    fn run(&mut self, limit: u64) {
        loop {
            for wire in &mut self.wires {
                wire.drive(&mut self.chips, &mut self.noise, &mut self.flips);
            }
            for (chip, pending) in self.chips.iter_mut().zip(&self.pending) {
                chip.set("image_resp_valid", pending.is_some() as u64).unwrap();
                chip.set("image_resp_data", pending.unwrap_or(0) as u64).unwrap();
            }
            let feeding = self.fed < self.input.len() && self.chips[0].get("host_rx_ready") == Some(1);
            self.chips[0].set("host_rx_valid", feeding as u64).unwrap();
            self.chips[0].set("host_rx_data", self.input.get(self.fed).copied().unwrap_or(0) as u64).unwrap();
            self.fed += feeding as usize;
            for (chip, pending) in self.chips.iter_mut().zip(&mut self.pending) {
                chip.settle();
                *pending = (chip.get("image_req_valid").unwrap() == 1).then(|| self.image[chip.get("image_req_addr").unwrap() as usize]);
            }
            if self.chips[0].get("host_tx_valid").unwrap() == 1 {
                self.output.push(self.chips[0].get("host_tx_data").unwrap() as u8);
            }
            if self.chips[0].get("halted").unwrap() == 1 {
                return;
            }
            for wire in &mut self.wires {
                wire.sample(&self.chips);
            }
            for chip in &mut self.chips {
                chip.tick();
            }
            self.cycles += 1;
            assert!(self.cycles < limit, "la placa no termina; salida {:?}", String::from_utf8_lossy(&self.output));
        }
    }
}

struct Setup {
    design: Design,
    image: Vec<u32>,
    dir: std::path::PathBuf,
    config: ChipConfig,
    die: DieConfig,
}

fn setup(name: &str, firmware: &FirmwareImage) -> Setup {
    let dir = std::env::temp_dir().join(format!("llm2chip-board-{name}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    make_checkpoint(&dir);
    let program = compile_qwen3_5(CONFIG).unwrap();
    let die = DieConfig { microcode_bits: 15, ..die_config() };
    let config = ChipConfig { soc: SocConfig { program_bits: 12, data_bits: 14, fifo_bits: 6 }, die, sampler: SamplerConfig { logit_bits: 16, index_bits: 12, entries: 4, lanes: 1 } };
    let tables = TokenizerTables { byte_symbols: (0..256).map(|b| b % 8).collect(), letters: vec![(97, 122)], token_bytes: (0..8).map(|t| vec![b'a' + t]).collect(), text_capacity: 64, ..Default::default() };
    let generation = GenerationSetup { max_new: 1, greedy: false, limit: 4, temperature: 256, seed: 0x1234_5678, raw_tokens: false, stop: [u32::MAX, u32::MAX] };
    let plans: Vec<_> = [1, 2].iter().map(|chips| plan_parts(&program, &dir, die, CHIP_TOKENS, &generation, *chips).unwrap()).collect();
    let image = build_plan_image(&firmware.program, &firmware.data, &tables.to_words(), &plans).unwrap();
    Setup { design: chip_design(config), image, dir, config, die }
}

#[test]
#[cfg_attr(debug_assertions, ignore = "dos chips completos en puertas: solo con optimizaciones")]
fn the_same_image_adapts_to_one_or_two_chips() {
    let Some(firmware) = build_firmware() else { return };
    let setup = setup("two", &firmware);
    let input = b"hoy\0";
    let program = compile_qwen3_5(CONFIG).unwrap();

    // One chip, at instruction level, with the very same image.
    let engine = VmEngine::new(compile_qwen3_5(CONFIG).unwrap(), Weights::open(&setup.dir).unwrap(), setup.die, CHIP_TOKENS).unwrap();
    let code = Layout::new(&program, &Weights::open(&setup.dir).unwrap(), setup.die, Some(CHIP_TOKENS)).unwrap().die_microprogram(&program, &program.split(1).unwrap(), 0).unwrap();
    let model = ChipModel::new(engine, code, SamplerModel { config: setup.config.sampler, rng: 0, table: exponential_table() });
    let mut bus = SocBus::booted_with(setup.config.soc, &setup.image, input, model).unwrap();
    let mut cpu = Rv32Iss::default();
    while !cpu.halted {
        cpu.step(&mut bus);
        assert!(cpu.cycle < 100_000_000, "el firmware no termina");
    }
    let reference = String::from_utf8_lossy(&bus.tx).trim_start_matches('.').to_string();
    assert_eq!(reference.len(), 1, "{:?}", String::from_utf8_lossy(&bus.tx));

    let mut board = Board::new(&setup.design, 2, &setup.image, input);
    board.run(8_000_000);
    let text = String::from_utf8_lossy(&board.output).into_owned();
    eprintln!("{} ciclos, {} bits invertidos en los enlaces, salida {text:?}", board.cycles, board.flips);
    assert_eq!(text.trim_start_matches('.'), reference, "el texto debe ser el mismo que con un solo chip");
    assert!(board.flips > 0, "la prueba debe inyectar errores");

    let debug = |chip: &mut Simulator, what: u64| {
        chip.set("debug_select", what).unwrap();
        chip.settle();
        chip.get("debug_value").unwrap()
    };
    let corrected: u64 = board.chips.iter_mut().map(|chip| debug(chip, DEBUG_LINK_CORRECTED)).sum();
    let failed: u64 = board.chips.iter_mut().map(|chip| debug(chip, DEBUG_LINK_FAILED)).sum();
    assert_eq!(failed, 0);
    assert_eq!(corrected, board.flips as u64, "cada bit invertido debe constar como corregido");

    let logits_register = Layout::new(&program, &Weights::open(&setup.dir).unwrap(), setup.die, Some(CHIP_TOKENS)).unwrap().logits_register(&program);
    for index in 0..8 {
        let address = (logits_register as usize) << setup.die.index_bits | index;
        let gates = board.chips[0].peek_memory("/die/vectors_a", address).unwrap() as u64 & 0xFFFF;
        let exact = netlist::chip::DieEngine::read(&mut bus.peripheral.engine, address as u32) as u64 & 0xFFFF;
        assert_eq!(gates, exact, "logit {index}");
    }
    std::fs::remove_dir_all(&setup.dir).ok();
}

#[test]
#[cfg_attr(debug_assertions, ignore = "tres chips completos en puertas: solo con optimizaciones")]
fn a_number_of_chips_the_image_does_not_cover_is_reported() {
    let Some(firmware) = build_firmware() else { return };
    let setup = setup("three", &firmware);
    let mut board = Board::new(&setup.design, 3, &setup.image, b"hoy\0");
    board.run(2_000_000);
    assert_eq!(String::from_utf8_lossy(&board.output), "N");
    std::fs::remove_dir_all(&setup.dir).ok();
}
