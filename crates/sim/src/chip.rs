//! The whole chip at instruction level: real firmware on the CPU model, driving the
//! exact machine through the same registers that the gates expose.

use crate::lower::Layout;
use crate::vm::BlockVm;
use crate::weights::Weights;
use compiler::Program;
use crate::Runner;
use netlist::block::{encode_vector, exponent_register, tables, BlockOps, Place};
use netlist::chip::{chip_info, DieEngine, GenerationSetup};
use netlist::die::word_bits;
use netlist::image::{ChipPart, Segment, Target};
use netlist::sampler::exponential_table;
use netlist::{DieConfig, Microprogram};
use std::collections::{BTreeMap, HashMap};
use std::path::Path;
use std::sync::Arc;

/// Fills a machine's registers from the values a layout starts with.
pub(crate) fn load_initial(vm: &mut BlockVm, initial: &HashMap<Place, i64>, exponents: u32) {
    let mut by_register: BTreeMap<u32, Vec<(u32, i64)>> = BTreeMap::new();
    let mut scales = Vec::new();
    for ((register, index), value) in initial {
        if *register == exponents {
            scales.push((*index, *value));
        } else {
            by_register.entry(*register).or_default().push((*index, *value));
        }
    }
    for (register, mut entries) in by_register {
        entries.sort();
        let mantissas: Vec<i64> = entries.iter().map(|(_, v)| *v).collect();
        vm.load_mantissas(register, &mantissas, 0);
    }
    for (register, exponent) in scales {
        vm.set_exponent(register, exponent);
    }
}

/// The die's routines on the exact machine, with registers numbered as in the gates.
pub struct VmEngine {
    config: DieConfig,
    code: Microprogram,
    program: Program,
    layout: Layout,
    vm: BlockVm,
    index_bits: u32,
    exponents: u32,
}

impl VmEngine {
    pub fn new(program: Program, weights: Weights, config: DieConfig, chip_tokens: usize) -> Result<Self, String> {
        let weights = Arc::new(weights);
        let layout = Layout::new(&program, &weights, config, Some(chip_tokens))?;
        let exponents = exponent_register(&config);
        let mut vm = BlockVm::new(&config, layout.registers_used() as usize + 8, Some(weights));
        load_initial(&mut vm, &layout.setup().initial, exponents);
        let code = layout.microprogram(&program)?;
        Ok(Self { config, code, program, layout, vm, index_bits: config.index_bits, exponents })
    }
}

impl DieEngine for VmEngine {
    fn run(&mut self, routine: usize, token: u32, position: u32) {
        self.layout.run_routine(&mut self.vm, &self.program, routine, token, position).expect("rutina del die");
    }

    fn cycles(&self, routine: usize, token: u32, position: u32) -> u64 {
        self.code.cycles(&self.config, routine, token, position)
    }

    fn read(&mut self, address: u32) -> i64 {
        let (register, index) = (address >> self.index_bits, address & ((1 << self.index_bits) - 1));
        if register == self.exponents {
            self.vm.register_exponent(index)
        } else {
            self.vm.mantissa(register, index)
        }
    }
}

/// Addresses of the logits and their exponent when the die runs as one software engine.
pub const FLOW_INDEX_BITS: u32 = 18;
pub const FLOW_REGISTER_BITS: u32 = 12;

/// Any engine behind the die's registers, so the real firmware can drive a real model.
pub struct RunnerEngine<'a> {
    engine: &'a mut dyn Runner,
    /// A token waits here until it is known whether its logits are wanted.
    pending: Option<u32>,
    mantissas: Vec<i64>,
    exponent: i64,
    bits: u32,
    cycles_per_token: u64,
    pub steps: usize,
    pub last_seconds: f64,
    pub saturated: u64,
    pub die_cycles: u64,
    pub error: Option<String>,
    /// Every token the firmware has sent to the die, in order.
    pub tokens: Vec<u32>,
    /// How many times the firmware has asked for logits.
    pub generated: usize,
}

impl<'a> RunnerEngine<'a> {
    pub fn new(engine: &'a mut dyn Runner, bits: u32, cycles_per_token: u64) -> Self {
        Self { engine, pending: None, mantissas: Vec::new(), exponent: 0, bits, cycles_per_token, steps: 0, last_seconds: 0.0, saturated: 0, die_cycles: 0, error: None, tokens: Vec::new(), generated: 0 }
    }

    fn step(&mut self, token: u32, want_logits: bool) {
        match self.engine.step(token, want_logits) {
            Ok((logits, stats)) => {
                self.steps += 1;
                self.last_seconds = stats.seconds;
                self.saturated += stats.saturated;
                if let Some(logits) = logits {
                    let values: Vec<f64> = logits.iter().map(|v| *v as f64).collect();
                    (self.mantissas, self.exponent) = encode_vector(&values, self.bits);
                }
            }
            Err(error) => self.error = Some(error),
        }
    }

}

impl DieEngine for RunnerEngine<'_> {
    fn run(&mut self, routine: usize, token: u32, _position: u32) {
        if let Some(previous) = self.pending.take() {
            self.step(previous, routine == crate::lower::ENTRY_LOGITS);
        }
        if routine == crate::lower::ENTRY_TOKEN {
            self.die_cycles += self.cycles_per_token;
            self.pending = Some(token);
            self.tokens.push(token);
        } else if routine == crate::lower::ENTRY_LOGITS {
            self.generated += 1;
        }
    }

    fn cycles(&self, routine: usize, _token: u32, _position: u32) -> u64 {
        if routine == crate::lower::ENTRY_TOKEN { self.cycles_per_token } else { 0 }
    }

    fn read(&mut self, address: u32) -> i64 {
        let register = address >> FLOW_INDEX_BITS;
        if register == (1 << FLOW_REGISTER_BITS) - 1 {
            return self.exponent;
        }
        self.mantissas.get((address & ((1 << FLOW_INDEX_BITS) - 1)) as usize).copied().unwrap_or(0)
    }
}

/// What each chip takes from the image when the model is shared among `chips` of them.
pub fn plan_parts(program: &Program, dir: &Path, config: DieConfig, chip_tokens: usize, setup: &GenerationSetup, chips: usize) -> Result<Vec<ChipPart>, String> {
    let plan = program.split(chips)?;
    (0..chips)
        .map(|chip| {
            let mut layout = Layout::new(program, &Weights::open(dir)?, config, Some(chip_tokens))?;
            layout.keep_weights_of(program, &plan, chip);
            let code = layout.die_microprogram(program, &plan, chip)?;
            Ok(ChipPart { info: chip_info(&config, &code, layout.logits_register(program), program.vocab as u32, setup), segments: image_segments(&config, &code, &layout) })
        })
        .collect()
}

/// Everything the firmware has to load onto the die from the image.
pub fn image_segments(config: &DieConfig, code: &Microprogram, layout: &Layout) -> Vec<Segment> {
    let mask = |value: i64| value as u32 & ((1u64 << config.cell.activation_bits) - 1) as u32;
    let beats = word_bits(config).div_ceil(32);
    let mut segments = vec![Segment { target: Target::Microcode, unit: 0, address: 0, beats, words: code.words.iter().flat_map(|w| w.encode_words(config)).collect() }];

    // Consecutive addresses travel as one transfer.
    let mut places: Vec<(u32, i64)> = layout.setup().initial.iter().map(|((register, index), value)| ((*register << config.index_bits) | *index, *value)).collect();
    places.sort();
    let mut run: Option<Segment> = None;
    for (address, value) in places {
        match &mut run {
            Some(segment) if segment.address + segment.words.len() as u32 == address => segment.words.push(mask(value)),
            _ => {
                segments.extend(run.take());
                run = Some(Segment { target: Target::Vector, unit: 0, address, beats: 1, words: vec![mask(value)] });
            }
        }
    }
    segments.extend(run);
    segments.push(Segment { target: Target::Function, unit: 0, address: 0, beats: 1, words: tables(config).iter().map(|v| mask(*v)).collect() });
    segments.push(Segment { target: Target::SamplerTable, unit: 0, address: 0, beats: 1, words: exponential_table() });
    segments
}
