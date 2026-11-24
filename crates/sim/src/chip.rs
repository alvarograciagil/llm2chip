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
        // The register just below the exponents is the die's scratch, whose constants only
        // its own steps read: the exact machine has no use for them.
        if *register + 1 == exponents {
            continue;
        }
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
        Self::build(program, weights, config, chip_tokens, false)
    }

    /// The die with the embedding table in the image from the start
    /// (`Layout::with_cells_table_outside`): each token's row is what the firmware brings.
    pub fn new_table_outside(program: Program, weights: Weights, config: DieConfig, chip_tokens: usize) -> Result<Self, String> {
        Self::build(program, weights, config, chip_tokens, true)
    }

    fn build(program: Program, weights: Weights, config: DieConfig, chip_tokens: usize, table_outside: bool) -> Result<Self, String> {
        // The die's addresses put the register above the element in 32 bits.
        if config.reg_bits + config.index_bits > 32 {
            return Err(format!("{} register bits and {} element bits do not fit in a 32-bit address", config.reg_bits, config.index_bits));
        }
        let weights = Arc::new(weights);
        let layout = if table_outside { Layout::with_cells_table_outside(&program, &weights, config, chip_tokens)? } else { Layout::with_cells(&program, &weights, config, chip_tokens)? };
        let exponents = exponent_register(&config);
        let mut vm = BlockVm::new(&config, layout.registers_used() as usize + 8, Some(weights));
        vm.compute_in_cells();
        load_initial(&mut vm, &layout.setup().initial, exponents);
        let code = layout.microprogram(&program)?;
        Ok(Self { config, code, program, layout, vm, index_bits: config.index_bits, exponents })
    }
}

impl VmEngine {
    /// Starts the die over, as at boot: what the image loads, nothing else.
    pub fn reset(&mut self) {
        self.vm.clear();
        load_initial(&mut self.vm, &self.layout.setup().initial, self.exponents);
    }

    /// The same die with the embedding table left in the image: each token's row is what the
    /// firmware brings into its register.
    pub fn with_table_outside(mut self) -> Result<Self, String> {
        if self.layout.table_outside(&self.program) {
            self.code = self.layout.microprogram(&self.program)?;
        }
        Ok(self)
    }
}

impl DieEngine for VmEngine {
    fn run(&mut self, routine: usize, token: u32, position: u32) {
        self.layout.run_routine(&mut self.vm, &self.program, routine, token, position).expect("die routine");
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

    fn write(&mut self, address: u32, value: i64) {
        let (register, index) = (address >> self.index_bits, address & ((1 << self.index_bits) - 1));
        if register == self.exponents {
            self.vm.set_register_exponent(index, value)
        } else if register + 1 != self.exponents {
            // The die's scratch, just below the exponents, is not modelled here.
            self.vm.write_mantissa(register, index, value)
        }
    }
}

/// The control CPU's cycles beyond the die's to follow a schedule, at three cycles an
/// instruction, measured at instruction level (the firmware tests of Moonshine and DAC
/// check them): to start a routine of a repeat made only of calls and notice its end, the
/// lean path (about 270)…
pub const FIRMWARE_CYCLES_PER_ROUTINE: u64 = 300;
/// …a routine started alone, reading its record and working out its parameters (about 930)…
pub const FIRMWARE_CYCLES_PER_ROUTINE_ALONE: u64 = 1_000;
/// …and a word received from the host or sent to it, a byte at a time.
pub const FIRMWARE_CYCLES_PER_WORD: u64 = 200;

/// The CPU's cycles for `lean` routines in repeats of calls, `alone` routines on their own
/// and `words` exchanged with the host.
pub fn firmware_cycles(lean: u64, alone: u64, words: u64) -> u64 {
    lean * FIRMWARE_CYCLES_PER_ROUTINE + alone * FIRMWARE_CYCLES_PER_ROUTINE_ALONE + words * FIRMWARE_CYCLES_PER_WORD
}

/// Follows `schedule` as the firmware does, on `die`, with the host's `input`: the tokens the
/// chip sends back, and the scores each was picked from, as values. `logits` is the scores'
/// first element, their exponent's address and how many there are, in vectors of `bits`
/// bits; a token of `stops` ends the text. The pick is greedy, the first of the highest.
pub fn follow_schedule(die: &mut impl DieEngine, schedule: &[netlist::schedule::Record], input: &[u32], logits: (u32, u32, u32), bits: u32, stops: &[u32]) -> Result<(Vec<u32>, Vec<Vec<f32>>), String> {
    struct Host<'a, E> {
        die: &'a mut E,
        input: std::collections::VecDeque<u32>,
        logits: (u32, u32, u32),
        bits: u32,
        stops: &'a [u32],
        sent: Vec<u32>,
        scores: Vec<Vec<f32>>,
    }
    impl<E: DieEngine> netlist::schedule::Driver for Host<'_, E> {
        fn receive(&mut self) -> u32 {
            self.input.pop_front().unwrap_or(0)
        }
        fn write(&mut self, address: u32, value: u32) {
            self.die.write(address, value as i32 as i64);
        }
        fn read(&mut self, address: u32) -> u32 {
            self.die.read(address) as u32
        }
        fn run(&mut self, routine: u32, token: u32, position: u32) {
            self.die.run(routine as usize, token, position);
        }
        fn sample(&mut self) -> u32 {
            let (base, exponent, count) = self.logits;
            let mantissas: Vec<i64> = (0..count).map(|i| self.die.read(base + i)).collect();
            let scale = 2f32.powi(self.die.read(exponent) as i32 - self.bits as i32 + 1);
            self.scores.push(mantissas.iter().map(|m| *m as f32 * scale).collect());
            mantissas.iter().enumerate().fold((0, i64::MIN), |best, (i, m)| if *m > best.1 { (i, *m) } else { best }).0 as u32
        }
        fn stops(&self, token: u32) -> bool {
            self.stops.contains(&token)
        }
        fn send(&mut self, token: u32) {
            self.sent.push(token);
        }
    }
    let mut host = Host { die, input: input.iter().copied().collect(), logits, bits, stops, sent: Vec::new(), scores: Vec::new() };
    let words = netlist::schedule::words(schedule);
    netlist::schedule::expand(&words, &mut host)?;
    Ok((host.sent, host.scores))
}

/// A die whose program is a set of routines, run by the exact machine, as a chip's firmware
/// drives it: each routine by its number in the microprogram `code`, with whether it rotates.
pub struct RoutinesDie<'a> {
    layout: &'a Layout,
    program: &'a Program,
    vm: &'a mut BlockVm,
    lists: Vec<(Vec<compiler::Inst>, bool)>,
    code: netlist::Microprogram,
    exponents: u32,
}

impl<'a> RoutinesDie<'a> {
    pub fn new(layout: &'a Layout, program: &'a Program, vm: &'a mut BlockVm, lists: Vec<(Vec<compiler::Inst>, bool)>, code: netlist::Microprogram) -> Self {
        let exponents = exponent_register(&layout.config());
        Self { layout, program, vm, lists, code, exponents }
    }

    fn split(&self, address: u32) -> (u32, u32) {
        let bits = self.layout.config().index_bits;
        (address >> bits, address & ((1 << bits) - 1))
    }
}

impl DieEngine for RoutinesDie<'_> {
    fn run(&mut self, routine: usize, token: u32, position: u32) {
        let (list, rope) = &self.lists[routine];
        self.layout.run_ops(self.vm, self.program, list, token, position, *rope).expect("die routine");
    }

    fn cycles(&self, routine: usize, token: u32, position: u32) -> u64 {
        self.code.cycles(&self.layout.config(), routine, token, position)
    }

    fn read(&mut self, address: u32) -> i64 {
        let (register, index) = self.split(address);
        if register == self.exponents { self.vm.register_exponent(index) } else { self.vm.mantissa(register, index) }
    }

    fn write(&mut self, address: u32, value: i64) {
        let (register, index) = self.split(address);
        if register == self.exponents {
            self.vm.set_register_exponent(index, value)
        } else {
            self.vm.write_mantissa(register, index, value)
        }
    }
}

/// Addresses of the logits and their exponent when the die runs as one software engine.
pub const FLOW_DRAFT_REGISTER: u32 = 2;
pub const FLOW_INDEX_BITS: u32 = 18;
pub const FLOW_REGISTER_BITS: u32 = 12;

/// The tokens of an image a software die reads, as vectors the host computed or kept in the
/// engine that encoded them.
pub enum ImageTokens {
    Vectors(Vec<Vec<f32>>),
    OnChip(usize),
}

impl ImageTokens {
    pub fn len(&self) -> usize {
        match self {
            ImageTokens::Vectors(vectors) => vectors.len(),
            ImageTokens::OnChip(count) => *count,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// How an image's tokens take their positions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Placement {
    /// Rows and columns of patches merged `merge` by `merge`, with M-RoPE.
    Grid(usize, usize, usize),
    /// One after another.
    Line,
}

impl Placement {
    /// The three positions of each of `tokens` tokens of an image that starts at `start`, and
    /// where the text after it goes on from.
    pub fn positions(&self, start: u64, tokens: usize) -> (Vec<[u64; 3]>, u64) {
        match *self {
            Placement::Grid(rows, cols, merge) => crate::vision::image_positions(start, rows, cols, merge),
            Placement::Line => ((0..tokens as u64).map(|k| [start + k; 3]).collect(), start + tokens as u64),
        }
    }
}

/// What waits for the next routine to know whether its logits are wanted.
enum Waiting {
    Token(u32),
    /// Token `index` of the image, at its three positions.
    Image(usize, [u64; 3]),
}

/// Any engine behind the die's registers, so the real firmware can drive a real model.
pub struct RunnerEngine<'a> {
    engine: &'a mut dyn Runner,
    /// A token waits here until it is known whether its logits are wanted.
    pending: Option<Waiting>,
    /// An image the host sent, its tokens and their places, where the first one went, and
    /// the placeholder they take the place of.
    image: Option<(ImageTokens, Placement, Option<u64>)>,
    placeholder: u32,
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
        Self { engine, pending: None, image: None, placeholder: 0, mantissas: Vec::new(), exponent: 0, draft_mantissas: Vec::new(), bits, cycles_per_token, steps: 0, last_seconds: 0.0, saturated: 0, die_cycles: 0, error: None, tokens: Vec::new(), generated: 0 }
    }

    /// The image the firmware will put in its placeholder's place, already encoded.
    pub fn with_image(mut self, tokens: ImageTokens, placement: Placement, placeholder: u32) -> Self {
        self.image = Some((tokens, placement, None));
        self.placeholder = placeholder;
        self
    }

    fn step(&mut self, waiting: Waiting, want_logits: bool) {
        let token = match waiting {
            Waiting::Token(token) => token,
            Waiting::Image(index, positions) => return self.step_image(index, positions, want_logits),
        };
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

    fn step_image(&mut self, index: usize, positions: [u64; 3], want_logits: bool) {
        let Some((tokens, placement, Some(start))) = &self.image else { return };
        let done = match tokens {
            ImageTokens::Vectors(vectors) => self.engine.step_embedding(&vectors[index], positions, want_logits),
            ImageTokens::OnChip(_) => self.engine.step_image(index as u32, positions, want_logits),
        };
        let last = index + 1 == tokens.len();
        let next = placement.positions(*start, tokens.len()).1;
        match done {
            Ok((logits, stats)) => {
                self.last_seconds = stats.seconds;
                if let Some(logits) = logits {
                    let values: Vec<f64> = logits.iter().map(|v| *v as f64).collect();
                    (self.mantissas, self.exponent) = encode_vector(&values, self.bits);
                }
            }
            Err(error) => self.error = Some(error),
        }
        // The image counts as one step of the prompt, as its placeholder does.
        if last {
            self.steps += 1;
            self.engine.continue_text_at(next);
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
            self.pending = Some(Waiting::Token(token));
            self.tokens.push(token);
        } else if routine == crate::lower::ENTRY_TOKEN_FROM_IMAGE {
            // The image's tokens take the cache's positions from the first one on.
            let Some((tokens, placement, start)) = &mut self.image else {
                self.error = Some("the firmware asks for an image token without an image".into());
                return;
            };
            if start.is_none() {
                // To the firmware's tokenizer, the image is its placeholder.
                self.tokens.push(self.placeholder);
            }
            let first = *start.get_or_insert(position as u64);
            let index = (position as u64 - first) as usize;
            if index >= tokens.len() {
                self.error = Some("the firmware asks for more image tokens than there are".into());
                return;
            }
            self.die_cycles += self.cycles_per_token;
            self.pending = Some(Waiting::Image(index, placement.positions(first, tokens.len()).0[index]));
        } else if routine == crate::lower::ENTRY_LOGITS {
            self.generated += 1;
        }
    }

    fn cycles(&self, routine: usize, _token: u32, _position: u32) -> u64 {
        if routine == crate::lower::ENTRY_TOKEN || routine == crate::lower::ENTRY_TOKEN_FROM_IMAGE { self.cycles_per_token } else { 0 }
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

impl Sharing {
    /// What this asks for comes to with `chips` of them: groups that do not divide the chips
    /// go by layers, and a single group shares every product by rows.
    pub fn for_chips(self, chips: usize) -> Sharing {
        match self {
            Sharing::Mixed(group) if group > 1 && chips > group as usize && chips.is_multiple_of(group as usize) => self,
            Sharing::Mixed(group) if chips > 1 && chips <= group as usize => Sharing::Rows,
            Sharing::Mixed(_) => Sharing::Layers,
            Sharing::Rows if chips < 2 => Sharing::Layers,
            other => other,
        }
    }
}

/// How `chips` of them share the model: the groups' size when they are in several groups,
/// whether each product is shared by rows, and the runs of layers of the chain.
pub struct ShareOut {
    group: Option<usize>,
    rows: bool,
    chips: usize,
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
        // Chips that share every product by rows make one stage of the chain between them.
        let stages = group.map_or(if rows { 1 } else { chips }, |g| chips / g);
        let plan = crate::lower::chip_plan(program, weights, stages, chip_tokens, activation_bits)?;
        Ok(Self { group, rows, chips, plan })
    }

    /// The runs of layers of the chain: one for each chip, or for each group.
    pub fn plan(&self) -> &[std::ops::Range<usize>] {
        &self.plan
    }

    /// The size of the groups that share each product by rows, when the chips are in several.
    pub fn group(&self) -> Option<usize> {
        self.group
    }

    /// The cycles each operation of chip `chip`'s part takes for a token at `position`, after
    /// `microprogram` has laid it out on `layout`, in program order.
    pub fn cycles_by_op(&self, layout: &Layout, program: &Program, chip: usize, position: u32) -> Result<Vec<(String, u64)>, String> {
        let ops = match self.group {
            Some(g) => self.plan[chip / g].clone(),
            None if self.rows => 0..program.ops.len(),
            None => self.plan[chip].clone(),
        };
        ops.map(|i| {
            let inst = &program.ops[i];
            let code = layout.op_microprogram(program, inst, i == 0)?;
            let name = format!("{:?}", inst.op);
            Ok((name.split([' ', '{', '(']).next().unwrap_or("").to_string(), code.cycles(&layout.config(), 0, 1, position)))
        })
        .collect()
    }

    /// Lays out chip `chip`'s part on `layout` and gives its microcode.
    pub fn microprogram(&self, layout: &mut Layout, program: &Program, chip: usize) -> Result<Microprogram, String> {
        if let Some(g) = self.group {
            let (stage, member) = (chip / g, chip % g);
            layout.keep_weights_of_part(program, &self.plan, stage, stage == 0);
            layout.share_rows(program, crate::lower::Shard { index: member as u32, chips: g as u32, grouped: true });
            layout.group_microprogram(program, &self.plan, stage)
        } else if self.rows {
            layout.share_rows(program, crate::lower::Shard { index: chip as u32, chips: self.chips as u32, grouped: false });
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
    assert!(bits + netlist::die::BASE_SHIFT_BITS <= 32, "the register table does not fit in 32-bit words");
    let (bases, needed) = layout.register_bases();
    assert!(needed <= bits, "the die's vector memory has 2^{bits} words and 2^{needed} are needed");
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

/// A chip for an FPGA board, with what it was built from: a layout and a microprogram for
/// each model it runs.
pub struct FpgaChip {
    pub design: netlist::Design,
    pub config: netlist::chip::ChipConfig,
    pub models: Vec<(Layout, Microprogram)>,
}

/// The chip for `program` on an FPGA: `trial` sized for it, then only the registers, logic
/// and memory its microprogram uses.
pub fn fpga_chip(program: &Program, weights: &Weights, trial: DieConfig, context: usize, options: FpgaOptions) -> Result<FpgaChip, String> {
    fpga_chip_for(&[(program, weights)], trial, context, options)
}

/// One chip for several models: the logic any of their microprograms uses and, for each
/// memory, the most any of them needs. Each model then boots from its own image on the
/// same bitstream, since microcode, tables and weights all come from the image.
pub fn fpga_chip_for(models: &[(&Program, &Weights)], trial: DieConfig, context: usize, options: FpgaOptions) -> Result<FpgaChip, String> {
    use netlist::chip::{chip_design, chip_design_alone, ChipConfig};
    use netlist::sampler::SamplerConfig;
    use netlist::soc::{SocConfig, TOKENIZER_WORD};
    let bits_for = |count: u64| (u64::BITS - count.max(2).saturating_sub(1).leading_zeros()).max(1);
    // Scale and bias a few bits a cycle cut the cell's longest path; the microprograms are
    // laid out for it.
    let trial = DieConfig { features: trial.features.with_factor_steps(options.factor_steps), ..trial };
    let mut reg_bits = 1;
    for (program, weights) in models {
        let roomy = Layout::with_cells(program, weights, trial, context)?;
        roomy.microprogram(program)?;
        reg_bits = reg_bits.max(with_registers_needed(trial, &roomy).reg_bits);
    }
    let trial = DieConfig { reg_bits, ..trial };
    let mut built = Vec::new();
    let mut needs: Vec<DieConfig> = Vec::new();
    for (program, weights) in models {
        let mut layout = Layout::with_cells(program, weights, trial, context)?;
        if !options.table_inside {
            layout.table_outside(program);
        }
        let code = layout.microprogram(program)?;
        let features = netlist::DieFeatures::of(&code).with_host_writes(program.rows().is_some()).with_lane_bits(trial.features.lane_bits).with_cluster_bits(trial.features.cluster_bits).with_key_shift(trial.features.key_shift).with_factor_steps(options.factor_steps);
        let die = compact_die(DieConfig { features, microcode_bits: bits_for(code.words.len() as u64), ..trial }, &layout);
        needs.push(if options.dictionary { with_code_dictionary(die, &code) } else { die });
        built.push((layout, code));
    }
    let features = needs.iter().map(|d| d.features).reduce(netlist::DieFeatures::union).ok_or("no model")?;
    let die = DieConfig { features, microcode_bits: needs.iter().map(|d| d.microcode_bits).max().unwrap_or(1), ..trial };
    // The firmware's data, its stack and the chip's description take the first 32 KB, and
    // the tokenizer's tables, if any, go after them.
    let data_bits = (TOKENIZER_WORD + options.tokenizer_words).next_power_of_two().ilog2();
    let vocabulary = models.iter().map(|(p, _)| p.vocab as u64).max().unwrap_or(2);
    let config = ChipConfig { soc: SocConfig { program_bits: 13, data_bits, fifo_bits: 6 }, die, sampler: SamplerConfig { logit_bits: 16, index_bits: bits_for(vocabulary).max(4), entries: 4, lanes: 1 } };
    let design = if options.chain { chip_design(config) } else { chip_design_alone(config) };
    Ok(FpgaChip { design, config, models: built })
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
    let features = config.features.with_vector_bits(layout.register_bases().1).with_factor_bits(factor_bits).with_wide_bits(wide_bits);
    // Units that read across their group need the field's wiring.
    let features = layout.group_field().map_or(features, |field| features.with_group(field));
    DieConfig { features, ..config }
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
    if let Some(table) = layout.rotation_table_outside() {
        segments.push(Segment { target: Target::RotationTable, unit: table.cols, address: table.register << config.index_bits, beats: 1, words: table.mantissas.iter().map(|v| mask(*v)).collect() });
    }
    segments.push(Segment { target: Target::SamplerTable, unit: 0, address: 0, beats: 1, words: exponential_table() });
    segments
}
