//! Several identical chips on a simulated board: one design, told apart by its pins.

use compiler::compile_qwen3_5;
use netlist::chip::{chip_design, ChipConfig, ChipModel, GenerationSetup, DEBUG_LINK_CORRECTED, DEBUG_LINK_FAILED, RESET_RELEASE};
use netlist::image::build_plan_image;
use netlist::rv32::Rv32Iss;
use netlist::sampler::{exponential_table, SamplerConfig, SamplerModel};
use netlist::soc::{FirmwareImage, SocBus, SocConfig};
use netlist::{Design, DieConfig, Simulator};
use sim::chip::{plan_parts, Sharing, VmEngine};
use sim::lower::Layout;
use sim::Weights;
use std::collections::VecDeque;

mod common;
use common::*;

const CHIP_TOKENS: usize = 8;
const WIRE_DELAY: usize = 2;

fn build_firmware() -> Option<FirmwareImage> {
    common::firmware()
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

/// Often enough to hit a run of a few words, rarely enough to stay correctable.
const FLIP_EVERY: u64 = 8;
const DOUBLE_EVERY: u64 = 3;

/// One direction of a link on the board: wires with delay, gaps and the odd flipped bit.
struct Wire {
    from: usize,
    to: usize,
    prefix: (String, String),
    forward: VecDeque<(u64, u128)>,
    backward: VecDeque<(u64, u64)>,
    width: u32,
}

impl Wire {
    fn new(from: usize, to: usize, prefix: (String, String), width: u32) -> Self {
        Self { from, to, prefix, forward: VecDeque::from(vec![(0, 0); WIRE_DELAY]), backward: VecDeque::from(vec![(0, 0); WIRE_DELAY]), width }
    }

    /// Drives the receiving pins with what left the other chip `WIRE_DELAY` cycles ago.
    fn drive(&mut self, chips: &mut [Simulator], noise: &mut Noise, flips: &mut u32, doubles: &mut u32) {
        let (mut valid, mut bits) = self.forward.pop_front().unwrap();
        if valid == 1 && noise.next().is_multiple_of(FLIP_EVERY) {
            let width = self.width as u64;
            let first = noise.next() % width;
            bits ^= 1 << first;
            *flips += 1;
            // Every so often a second bit goes too, which only a repeat can repair.
            if u64::from(*flips - 1).is_multiple_of(DOUBLE_EVERY) {
                bits ^= 1 << ((first + 1 + noise.next() % (width - 1)) % width);
                *doubles += 1;
            }
        }
        chips[self.to].set(&format!("{}_valid", self.prefix.1), valid).unwrap();
        chips[self.to].set_wide(&format!("{}_bits", self.prefix.1), bits).unwrap();
        // Now and then the board holds the sender back for a cycle.
        let (ready, ack) = self.backward.pop_front().unwrap();
        let ready = ready & u64::from(!noise.next().is_multiple_of(5));
        chips[self.from].set(&format!("{}_ready", self.prefix.0), ready).unwrap();
        chips[self.from].set(&format!("{}_ack", self.prefix.0), ack).unwrap();
        valid = 0;
        let _ = valid;
    }

    fn sample(&mut self, chips: &[Simulator]) {
        self.forward.push_back((chips[self.from].get(&format!("{}_valid", self.prefix.0)).unwrap(), chips[self.from].get_wide(&format!("{}_bits", self.prefix.0)).unwrap()));
        self.backward.push_back((chips[self.to].get(&format!("{}_ready", self.prefix.1)).unwrap(), chips[self.to].get(&format!("{}_ack", self.prefix.1)).unwrap()));
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
    doubles: u32,
    input: &'a [u8],
    fed: usize,
    output: Vec<u8>,
    cycles: u64,
}

impl<'a> Board<'a> {
    /// With `group`, the chips are in groups of that many and the last of each closes its
    /// group's ring with a loop back to the first.
    fn new(design: &Design, count: usize, image: &'a [u32], input: &'a [u8], link_lanes: usize, group: Option<usize>) -> Self {
        let mut chips: Vec<Simulator> = (0..count).map(|_| Simulator::with_names(design).unwrap()).collect();
        for (index, chip) in chips.iter_mut().enumerate() {
            chip.set("follower", (index != 0) as u64).unwrap();
            chip.set("reset", 1).unwrap();
            chip.tick();
            chip.set("reset", 0).unwrap();
            for _ in 0..=RESET_RELEASE {
                chip.tick();
            }
            chip.set("host_tx_ready", 1).unwrap();
        }
        let mut wires = Vec::new();
        if count > 1 {
            for from in 0..count {
                let to = (from + 1) % count;
                // Every lane of the link is a wire of its own on the board.
                for lane in 0..link_lanes {
                    let p = if lane == 0 { "link".to_string() } else { format!("link{lane}") };
                    wires.push(Wire::new(from, to, (format!("{p}_tx"), format!("{p}_rx")), netlist::link::wire_bits(16)));
                }
                wires.push(Wire::new(from, to, ("chain_tx".into(), "chain_rx".into()), netlist::link::wire_bits(32)));
            }
            for last in group.into_iter().flat_map(|g| (g - 1..count).step_by(g)) {
                let group = group.unwrap();
                // The loop back is as wide as the link.
                for lane in 0..link_lanes {
                    let p = if lane == 0 { "loop".to_string() } else { format!("loop{lane}") };
                    wires.push(Wire::new(last, last + 1 - group, (format!("{p}_tx"), format!("{p}_rx")), netlist::link::wire_bits(16)));
                }
            }
        }
        Self { chips, wires, pending: vec![None; count], image, noise: Noise(0x0BAD_C0DE_1234_5678), flips: 0, doubles: 0, input, fed: 0, output: Vec::new(), cycles: 0 }
    }

    /// Runs until the leading chip stops.
    fn run(&mut self, limit: u64) {
        loop {
            for wire in &mut self.wires {
                wire.drive(&mut self.chips, &mut self.noise, &mut self.flips, &mut self.doubles);
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
            if self.cycles.is_multiple_of(500_000) {
                eprintln!("{} cycles, output {:?}", self.cycles, String::from_utf8_lossy(&self.output));
            }
            if self.cycles >= limit {
                // Where each chip stands: its CPU's address and whether its die is busy.
                let states: Vec<String> = self.chips.iter_mut().map(|chip| {
                    let mut read = |what: u64| {
                        chip.set("debug_select", what).unwrap();
                        chip.settle();
                        chip.get("debug_value").unwrap()
                    };
                    format!("pc {:#x}, die {}", read(netlist::chip::DEBUG_CPU_PC), read(netlist::chip::DEBUG_DIE_BUSY))
                }).collect();
                panic!("the board does not finish; output {:?}; chips: {states:?}", String::from_utf8_lossy(&self.output));
            }
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

fn setup(name: &str, firmware: &FirmwareImage, die: DieConfig, sharing: Sharing) -> Setup {
    setup_for(name, firmware, die, sharing, 2)
}

/// The image covers one chip and `chips` of them.
fn setup_for(name: &str, firmware: &FirmwareImage, die: DieConfig, sharing: Sharing, chips: usize) -> Setup {
    let dir = std::env::temp_dir().join(format!("llm2chip-board-{name}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    make_checkpoint(&dir);
    let program = compile_qwen3_5(CONFIG).unwrap();
    let config = ChipConfig { soc: SocConfig { program_bits: 13, data_bits: 14, fifo_bits: 6 }, die, sampler: SamplerConfig { logit_bits: 16, index_bits: 12, entries: 4, lanes: 1 } };
    let tables = tiny_tokenizer();
    let generation = GenerationSetup { max_new: 1, greedy: false, limit: 4, temperature: 256, seed: 0x1234_5678, raw_tokens: false, stop: Vec::new(), draft_depth: 0, draft_logits: 0, think: None, think_budget: u32::MAX };
    let plans: Vec<_> = [1, chips].iter().map(|chips| plan_parts(&program, &dir, die, CHIP_TOKENS, &generation, *chips, sharing).unwrap()).collect();
    let image = build_plan_image(&firmware.program, &firmware.data, &tables.to_words(), &plans).unwrap();
    Setup { design: chip_design(config), image, dir, config, die }
}

#[test]
#[ignore = "slow: two whole chips in gates: only with optimizations"]
fn the_same_image_adapts_to_one_or_two_chips() {
    let Some(firmware) = build_firmware() else { return };
    two_chips_say_what_one_says(&firmware, DieConfig { microcode_bits: 15, ..cell_die_config() }, "two", Sharing::Layers);
}

#[test]
#[ignore = "slow: two whole chips in gates: only with optimizations"]
fn two_chips_that_share_every_product_by_rows_say_what_one_says() {
    let Some(firmware) = build_firmware() else { return };
    two_chips_say_what_one_says(&firmware, DieConfig { microcode_bits: 15, ..cell_die_config() }, "rows", Sharing::Rows);
}

#[test]
#[ignore = "slow: four whole chips in gates: only with optimizations"]
fn two_groups_of_two_chips_sharing_rows_in_a_chain_say_what_one_says() {
    let Some(firmware) = build_firmware() else { return };
    chips_say_what_one_says(&firmware, DieConfig { microcode_bits: 15, ..cell_die_config() }, "mixed", Sharing::Mixed(2), 4);
}

#[test]
#[ignore = "slow: two whole chips in gates: only with optimizations"]
fn chips_sharing_rows_over_a_wide_link_say_what_one_says() {
    let Some(firmware) = build_firmware() else { return };
    let base = cell_die_config();
    let die = DieConfig { cells: 4, microcode_bits: 15, features: base.features.with_lane_bits(2).with_wide_link(true), ..base };
    two_chips_say_what_one_says(&firmware, die, "rows-wide", Sharing::Rows);
}

#[test]
#[ignore = "slow: four whole chips in gates: only with optimizations"]
fn four_chips_in_a_ring_pass_blocks_on_and_say_what_one_says() {
    let Some(firmware) = build_firmware() else { return };
    let base = cell_die_config();
    let die = DieConfig { cells: 4, microcode_bits: 15, features: base.features.with_lane_bits(2).with_wide_link(true), ..base };
    chips_say_what_one_says(&firmware, die, "rows-ring", Sharing::Rows, 4);
}

#[test]
#[ignore = "slow: six whole chips in gates: only with optimizations"]
fn six_chips_in_a_ring_agree_on_a_pass_in_one_round_and_say_what_one_says() {
    // Lanes of two: the gate and up products share every pass by rows, and in a ring this
    // long their narrowings agree on both shifts in one round, a lane each.
    let Some(firmware) = build_firmware() else { return };
    let base = cell_die_config();
    let die = DieConfig { cells: 4, microcode_bits: 15, features: base.features.with_lane_bits(1).with_wide_link(true), ..base };
    chips_say_what_one_says(&firmware, die, "rows-six", Sharing::Rows, 6);
}

#[test]
#[ignore = "slow: two whole chips in gates: only with optimizations"]
fn a_wide_link_moves_blocks_between_two_chips_and_keeps_the_text() {
    let Some(firmware) = build_firmware() else { return };
    let base = cell_die_config();
    let die = DieConfig { cells: 4, microcode_bits: 15, features: base.features.with_lane_bits(2).with_wide_link(true), ..base };
    two_chips_say_what_one_says(&firmware, die, "wide", Sharing::Layers);
}

/// Two chips on a noisy board say the same as one at instruction level, with the very
/// same image, and their links count every error they correct or have repeated.
fn two_chips_say_what_one_says(firmware: &FirmwareImage, die: DieConfig, name: &str, sharing: Sharing) {
    chips_say_what_one_says(firmware, die, name, sharing, 2);
}

/// Several chips on a noisy board say the same as one at instruction level.
fn chips_say_what_one_says(firmware: &FirmwareImage, die: DieConfig, name: &str, sharing: Sharing, chips: usize) {
    let setup = setup_for(name, firmware, die, sharing, chips);
    let input = b"hoy\0";
    let program = compile_qwen3_5(CONFIG).unwrap();

    // One chip, at instruction level, with the very same image.
    let engine = VmEngine::new(compile_qwen3_5(CONFIG).unwrap(), Weights::open(&setup.dir).unwrap(), setup.die, CHIP_TOKENS).unwrap();
    let code = Layout::with_cells(&program, &Weights::open(&setup.dir).unwrap(), setup.die, CHIP_TOKENS).unwrap().die_microprogram(&program, &program.split(1).unwrap(), 0).unwrap();
    let model = ChipModel::new(engine, code, SamplerModel { config: setup.config.sampler, rng: 0, table: exponential_table() });
    let mut bus = SocBus::booted_with(setup.config.soc, &setup.image, input, model).unwrap();
    let mut cpu = Rv32Iss::default();
    while !cpu.halted {
        cpu.step(&mut bus);
        assert!(cpu.cycle < 100_000_000, "the firmware does not finish");
    }
    let reference = String::from_utf8_lossy(&bus.tx).trim_start_matches('.').to_string();
    assert_eq!(reference.len(), 1, "{:?}", String::from_utf8_lossy(&bus.tx));

    let group = match sharing {
        Sharing::Mixed(g) if (g as usize) < chips => Some(g as usize),
        _ => None,
    };
    let mut board = Board::new(&setup.design, chips, &setup.image, input, setup.die.features.link_lanes(), group);
    board.run(std::env::var("LIMIT").ok().and_then(|l| l.parse().ok()).unwrap_or(8_000_000));
    let text = String::from_utf8_lossy(&board.output).into_owned();
    eprintln!("{} cycles, {} flipped bits on the links, output {text:?}", board.cycles, board.flips);
    assert_eq!(text.trim_start_matches('.'), reference, "the text must be the same as with a single chip");
    assert!(board.flips > 0, "the test must inject errors");

    let debug = |chip: &mut Simulator, what: u64| {
        chip.set("debug_select", what).unwrap();
        chip.settle();
        chip.get("debug_value").unwrap()
    };
    let corrected: u64 = board.chips.iter_mut().map(|chip| debug(chip, DEBUG_LINK_CORRECTED)).sum();
    let failed: u64 = board.chips.iter_mut().map(|chip| debug(chip, DEBUG_LINK_FAILED)).sum();
    assert!(board.doubles > 0, "the test must force retries");
    assert_eq!(failed, board.doubles as u64, "each word with two bad bits is logged as failed");
    assert_eq!(corrected, (board.flips - board.doubles) as u64, "each single flipped bit is logged as corrected");

    let logits_register = Layout::with_cells(&program, &Weights::open(&setup.dir).unwrap(), setup.die, CHIP_TOKENS).unwrap().logits_register(&program);
    for index in 0..8 {
        let address = (logits_register as usize) << setup.die.index_bits | index;
        // With lanes, each element sits in the bank of its lane, at its block.
        let lane_bits = setup.die.features.lane_bits;
        let bank = if lane_bits == 0 { "/die/vectors_a".to_string() } else { format!("/die/vectors_a{}", address & ((1 << lane_bits) - 1)) };
        let gates = board.chips[0].peek_memory(&bank, address >> lane_bits).unwrap() as u64 & 0xFFFF;
        let exact = netlist::chip::DieEngine::read(&mut bus.peripheral.engine, address as u32) as u64 & 0xFFFF;
        assert_eq!(gates, exact, "logit {index}");
    }
    std::fs::remove_dir_all(&setup.dir).ok();
}

#[test]
fn a_number_of_chips_the_image_does_not_cover_is_reported() {
    let Some(firmware) = build_firmware() else { return };
    let setup = setup("three", &firmware, DieConfig { microcode_bits: 15, ..cell_die_config() }, Sharing::Layers);
    let mut board = Board::new(&setup.design, 3, &setup.image, b"hoy\0", 1, None);
    board.run(2_000_000);
    assert_eq!(String::from_utf8_lossy(&board.output), "N");
    std::fs::remove_dir_all(&setup.dir).ok();
}

