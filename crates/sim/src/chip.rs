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
        let layout = Layout::with_cells(&program, &weights, config, chip_tokens)?;
        let exponents = exponent_register(&config);
        let mut vm = BlockVm::new(&config, layout.registers_used() as usize + 8, Some(weights));
        vm.compute_in_cells();
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
pub const FLOW_DRAFT_REGISTER: u32 = 2;
pub const FLOW_INDEX_BITS: u32 = 18;
pub const FLOW_REGISTER_BITS: u32 = 12;

/// Any engine behind the die's registers, so the real firmware can drive a real model.
pub struct RunnerEngine<'a> {
    engine: &'a mut dyn Runner,
    /// A token waits here until it is known whether its logits are wanted.
    pending: Option<u32>,
    mantissas: Vec<i64>,
    exponent: i64,
    draft_mantissas: Vec<i64>,
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
        Self { engine, pending: None, mantissas: Vec::new(), exponent: 0, draft_mantissas: Vec::new(), bits, cycles_per_token, steps: 0, last_seconds: 0.0, saturated: 0, die_cycles: 0, error: None, tokens: Vec::new(), generated: 0 }
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
    fn run(&mut self, routine: usize, token: u32, position: u32) {
        use crate::lower::{ENTRY_DRAFT, ENTRY_DRAFT_LOGITS, ENTRY_FEED_FROM_DRAFT, ENTRY_FEED_FROM_MAIN, ENTRY_LOGITS};
        if let Some(previous) = self.pending.take() {
            self.step(previous, routine == ENTRY_LOGITS || routine == ENTRY_FEED_FROM_MAIN);
        }
        match routine {
            ENTRY_FEED_FROM_MAIN => return self.engine.feed_draft_from_main(),
            ENTRY_FEED_FROM_DRAFT => return self.engine.feed_draft_from_draft(),
            ENTRY_DRAFT | ENTRY_DRAFT_LOGITS => {
                if (position as usize) < self.engine.draft_tokens() {
                    self.engine.truncate_draft(position as usize);
                }
                match self.engine.draft(token, routine == ENTRY_DRAFT_LOGITS) {
                    Ok(Some(logits)) => {
                        let values: Vec<f64> = logits.iter().map(|v| *v as f64).collect();
                        self.draft_mantissas = encode_vector(&values, self.bits).0;
                    }
                    Ok(None) => {}
                    Err(error) => self.error = Some(error),
                }
                return;
            }
            _ => {}
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
        let source = if register == FLOW_DRAFT_REGISTER { &self.draft_mantissas } else { &self.mantissas };
        source.get((address & ((1 << FLOW_INDEX_BITS) - 1)) as usize).copied().unwrap_or(0)
    }
}

/// How several chips share the model.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Sharing {
    /// Each chip runs a run of whole layers and passes the hidden vector down the chain.
    #[default]
    Layers,
    /// Every chip holds a slice of the rows of each matrix and they all work on every
    /// product at once, passing their slices around the ring.
    Rows,
    /// Groups of this many chips share each product by rows, each group in a ring closed by
    /// a loop back, and the groups take runs of layers in a chain.
    Mixed(u32),
}

/// How `chips` of them share the model: the groups' size when they are in several groups,
/// whether each product is shared by rows, and the runs of layers of the chain.
pub struct ShareOut {
    group: Option<usize>,
    rows: bool,
    plan: Vec<std::ops::Range<usize>>,
}

impl ShareOut {
    pub fn new(program: &Program, weights: &Weights, chips: usize, chip_tokens: usize, activation_bits: u32, sharing: Sharing) -> Result<Self, String> {
        // An image carries a plan for each number of chips: one that does not make whole
        // groups goes by layers, and one that fits in a single group shares by rows.
        let group = match sharing {
            Sharing::Mixed(group) if group > 1 && chips > group as usize && chips.is_multiple_of(group as usize) => Some(group as usize),
            _ => None,
        };
        let rows = group.is_none() && chips > 1 && match sharing {
            Sharing::Rows => true,
            Sharing::Mixed(group) => chips <= group as usize,
            Sharing::Layers => false,
        };
        let stages = group.map_or(chips, |g| chips / g);
        let plan = crate::lower::chip_plan(program, weights, stages, chip_tokens, activation_bits)?;
        Ok(Self { group, rows, plan })
    }

    /// Lays out chip `chip`'s part on `layout` and gives its microcode.
    pub fn microprogram(&self, layout: &mut Layout, program: &Program, chip: usize) -> Result<Microprogram, String> {
        if let Some(g) = self.group {
            let (stage, member) = (chip / g, chip % g);
            layout.keep_weights_of_part(program, &self.plan, stage, chip == 0);
            layout.share_rows(program, crate::lower::Shard { index: member as u32, chips: g as u32, grouped: true });
            layout.group_microprogram(program, &self.plan, stage)
        } else if self.rows {
            layout.share_rows(program, crate::lower::Shard { index: chip as u32, chips: self.plan.len() as u32, grouped: false });
            layout.microprogram(program)
        } else {
            layout.keep_weights_of(program, &self.plan, chip);
            layout.die_microprogram(program, &self.plan, chip)
        }
    }
}

/// What each chip takes from the image when the model is shared among `chips` of them.
pub fn plan_parts(program: &Program, dir: &Path, config: DieConfig, chip_tokens: usize, setup: &GenerationSetup, chips: usize, sharing: Sharing) -> Result<Vec<ChipPart>, String> {
    let share = ShareOut::new(program, &Weights::open(dir)?, chips, chip_tokens, config.cell.activation_bits, sharing)?;
    (0..chips)
        .map(|chip| {
            let weights = Weights::open(dir)?;
            let mut layout = Layout::with_cells(program, &weights, config, chip_tokens)?;
            let code = share.microprogram(&mut layout, program, chip)?;
            let mut segments = image_segments(&config, &code, &layout);
            segments.extend(cell_segments(&config, &layout, &weights)?);
            Ok(ChipPart { info: chip_info(&config, &code, layout.logits_register(program), program.vocab as u32, setup), segments })
        })
        .collect()
}

/// What the firmware writes into each cell: its weights, then its scales and biases.
pub fn cell_segments(config: &DieConfig, layout: &Layout, weights: &Weights) -> Result<Vec<Segment>, String> {
    let width = config.scale_bits.max(config.cell.weight_bits).max(config.cell.activation_bits);
    let mask = |value: i64| value as u32 & ((1u64 << width) - 1) as u32;
    let mut segments = Vec::with_capacity(2 * config.cells as usize);
    for cell in 0..config.cells {
        let (codes, factors) = layout.cell_contents(weights, cell)?;
        segments.push(Segment { target: Target::CellWeight, unit: cell, address: 0, beats: 1, words: codes.iter().map(|c| mask(*c)).collect() });
        segments.push(Segment { target: Target::CellScale, unit: cell, address: 0, beats: 1, words: factors.iter().flat_map(|(scale, bias)| [mask(*scale), mask(*bias)]).collect() });
    }
    Ok(segments)
}

/// Everything the firmware has to load onto the die from the image.
/// The entries of a compact vector memory's register table: each register's base, with
/// the step of the loops that visit it by number above it.
pub fn register_table(config: &DieConfig, layout: &Layout) -> Vec<u32> {
    let bits = config.features.vector_bits;
    assert!(bits + netlist::die::BASE_SHIFT_BITS <= 32, "la tabla de registros no cabe en palabras de 32 bits");
    let (bases, needed) = layout.register_bases();
    assert!(needed <= bits, "la memoria vectorial del die tiene 2^{bits} palabras y hacen falta 2^{needed}");
    bases.iter().map(|(base, shift)| base | shift << bits).collect()
}

/// The same die with its microcode kept as a dictionary of distinct words when that takes
/// less memory than the words one after another.
pub fn with_code_dictionary(config: DieConfig, code: &Microprogram) -> DieConfig {
    let (words, _) = netlist::die::code_dictionary(code, &config);
    let bits = u32::BITS - (words.len().max(2) as u32 - 1).leading_zeros();
    let width = word_bits(&config) as u64;
    let plain = (1u64 << config.microcode_bits) * width;
    let kept = (1u64 << bits) * width + (1u64 << config.microcode_bits) * 2 * bits as u64;
    if config.microcode_bits > netlist::die::ENTRY_INDEX_SHIFT || kept >= plain {
        return config;
    }
    DieConfig { features: config.features.with_dictionary_bits(bits), ..config }
}

/// The same die with memories just as large as the layout needs: the vector memory its
/// routines reach, as many scales and biases in each cell as it has groups, and the wide
/// memory as many rows as its sums reach.
/// How a chip for an FPGA board is put together.
#[derive(Clone, Copy, Debug)]
pub struct FpgaOptions {
    /// Cycles each cell takes to apply a group's scale and bias, a few bits at a time.
    pub factor_steps: u32,
    /// The embedding table in the vector memory, instead of a row at a time from the image.
    pub table_inside: bool,
    /// The microcode kept once per distinct word, with a list of numbers into it.
    pub dictionary: bool,
    /// Words of tokenizer tables the CPU keeps, when it tokenizes on the chip.
    pub tokenizer_words: usize,
    /// The manufacturing tests and the chain to other chips, for boards joined together.
    pub chain: bool,
}

impl Default for FpgaOptions {
    fn default() -> Self {
        Self { factor_steps: 1, table_inside: false, dictionary: true, tokenizer_words: 0, chain: false }
    }
}

/// A chip for an FPGA board, with what it was built from.
pub struct FpgaChip {
    pub design: netlist::Design,
    pub config: netlist::chip::ChipConfig,
    pub layout: Layout,
    pub code: Microprogram,
}

/// The chip for `program` on an FPGA: `trial` sized for it, then only the registers, logic
/// and memory its microprogram uses.
pub fn fpga_chip(program: &Program, weights: &Weights, trial: DieConfig, context: usize, options: FpgaOptions) -> Result<FpgaChip, String> {
    use netlist::chip::{chip_design, chip_design_alone, ChipConfig};
    use netlist::sampler::SamplerConfig;
    use netlist::soc::{SocConfig, TOKENIZER_WORD};
    let bits_for = |count: u64| (u64::BITS - count.max(2).saturating_sub(1).leading_zeros()).max(1);
    // Scale and bias a few bits a cycle cut the cell's longest path; the microprogram is
    // laid out for it.
    let trial = DieConfig { features: trial.features.with_factor_steps(options.factor_steps), ..trial };
    let roomy = Layout::with_cells(program, weights, trial, context)?;
    roomy.microprogram(program)?;
    let trial = with_registers_needed(trial, &roomy);
    let mut layout = Layout::with_cells(program, weights, trial, context)?;
    if !options.table_inside {
        layout.table_outside(program);
    }
    let code = layout.microprogram(program)?;
    let features = netlist::DieFeatures::of(&code).with_lane_bits(trial.features.lane_bits).with_cluster_bits(trial.features.cluster_bits).with_key_shift(trial.features.key_shift).with_factor_steps(options.factor_steps);
    let die = compact_die(DieConfig { features, microcode_bits: bits_for(code.words.len() as u64), ..trial }, &layout);
    let die = if options.dictionary { with_code_dictionary(die, &code) } else { die };
    // The firmware's data, its stack and the chip's description take the first 32 KB, and
    // the tokenizer's tables, if any, go after them.
    let data_bits = (TOKENIZER_WORD + options.tokenizer_words).next_power_of_two().ilog2();
    let config = ChipConfig { soc: SocConfig { program_bits: 13, data_bits, fifo_bits: 6 }, die, sampler: SamplerConfig { logit_bits: 16, index_bits: bits_for(program.vocab as u64).max(4), entries: 4, lanes: 1 } };
    let design = if options.chain { chip_design(config) } else { chip_design_alone(config) };
    Ok(FpgaChip { design, config, layout, code })
}

/// As many registers as the routines assembled on `layout` reach into, and the two the
/// die keeps for itself: the working registers past them are left out.
pub fn with_registers_needed(config: DieConfig, layout: &Layout) -> DieConfig {
    let needed = layout.registers_needed() as u64 + 2;
    DieConfig { reg_bits: u64::BITS - (needed - 1).leading_zeros(), ..config }
}

pub fn compact_die(config: DieConfig, layout: &Layout) -> DieConfig {
    let groups = layout.cell_depths().1.max(2);
    let factor_bits = u64::BITS - (groups - 1).leading_zeros();
    // The cells drain a whole pass at a time, so the wide memory holds whole passes.
    let wide = layout.wide_reach().next_multiple_of(config.cells.max(1 << config.features.lane_bits)).max(2);
    let wide_bits = u32::BITS - (wide - 1).leading_zeros();
    DieConfig { features: config.features.with_vector_bits(layout.register_bases().1).with_factor_bits(factor_bits).with_wide_bits(wide_bits), ..config }
}

pub fn image_segments(config: &DieConfig, code: &Microprogram, layout: &Layout) -> Vec<Segment> {
    let mask = |value: i64| value as u32 & ((1u64 << config.cell.activation_bits) - 1) as u32;
    let beats = word_bits(config).div_ceil(32);
    let dictionary = config.features.dictionary_bits;
    let mut segments = if dictionary > 0 {
        // Each distinct word once, then for each PC where its successors are among them.
        let (words, index) = netlist::die::code_dictionary(code, config);
        let links = netlist::die::dictionary_links(code, &index, dictionary);
        vec![Segment { target: Target::Microcode, unit: 0, address: 0, beats, words: words.concat() }, Segment { target: Target::MicrocodeIndex, unit: 0, address: 0, beats: 1, words: links }]
    } else {
        vec![Segment { target: Target::Microcode, unit: 0, address: 0, beats, words: code.words.iter().flat_map(|w| w.encode_words(config)).collect() }]
    };
    // A compact vector memory needs every register's place before anything goes into it.
    if config.features.vector_bits > 0 {
        segments.push(Segment { target: Target::RegisterBase, unit: 0, address: 0, beats: 1, words: register_table(config, layout) });
    }

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
    if let Some(table) = layout.outside_table() {
        segments.push(Segment { target: Target::RuntimeTable, unit: table.cols, address: table.register << config.index_bits, beats: 1, words: table.mantissas.iter().map(|v| mask(*v)).collect() });
    }
    segments.push(Segment { target: Target::SamplerTable, unit: 0, address: 0, beats: 1, words: exponential_table() });
    segments
}
