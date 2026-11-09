//! The whole chip at instruction level: real firmware on the CPU model, driving the
//! exact machine through the same registers that the gates expose.

use crate::lower::Layout;
use crate::vm::BlockVm;
use crate::weights::Weights;
use compiler::Program;
use netlist::block::{exponent_register, tables, BlockOps, Place};
use netlist::chip::DieEngine;
use netlist::die::word_bits;
use netlist::image::{Segment, Target};
use netlist::sampler::exponential_table;
use netlist::{DieConfig, Microprogram};
use std::collections::{BTreeMap, HashMap};
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
        Ok(Self { program, layout, vm, index_bits: config.index_bits, exponents })
    }
}

impl DieEngine for VmEngine {
    fn run(&mut self, routine: usize, token: u32, position: u32) {
        self.layout.run_routine(&mut self.vm, &self.program, routine, token, position).expect("rutina del die");
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
