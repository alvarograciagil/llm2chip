//! Traces of dies at work for the 3D viewer, and the gate-level board they come from.
//!
//! Nothing in the compiler or the simulators depends on this crate: it only runs them. It
//! holds the small models the gate-level tests run (`tiny`), dies loaded and joined by their
//! links (`board`), and the recorder that turns a run into a `Trace` (`record`).

pub mod board;
pub mod check;
pub mod record;
pub mod tiny;

use board::{lanes, loaded_die, Wire};
use netlist::{DieConfig, Microprogram, Simulator};
use record::{Recorder, Subject, Trace};
use sim::chip::{compact_die, ShareOut, Sharing};
use sim::sizing::sharing_name;
use sim::lower::{Layout, ENTRY_LOGITS, ENTRY_TOKEN};
use sim::Weights;
use std::path::Path;

/// The longest context the dies of a tiny board are built for.
pub const TINY_CONTEXT: usize = 8;

/// Which small model a board runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TinyModel {
    /// Two query heads on one key head: the model most gate-level tests run.
    Plain,
    /// Six query heads on two key heads, heads of eight elements: it gives attention's
    /// head sharing something to do.
    Grouped,
}

impl TinyModel {
    pub fn config(self) -> String {
        match self {
            TinyModel::Plain => tiny::CONFIG.to_string(),
            TinyModel::Grouped => tiny::grouped_config(8),
        }
    }

    /// Writes the model's checkpoint into `dir`.
    pub fn write(self, dir: &Path) {
        match self {
            TinyModel::Plain => tiny::make_checkpoint(dir),
            TinyModel::Grouped => tiny::grouped_checkpoint(dir, 8),
        }
    }
}

/// The shape of the dies of a tiny board.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TinyDies {
    /// log2 of the units of each die.
    pub cluster_bits: u32,
    /// log2 of the lanes of each unit.
    pub lane_bits: u32,
    /// Links a block wide.
    pub wide: bool,
}

impl Default for TinyDies {
    fn default() -> Self {
        Self { cluster_bits: 1, lane_bits: 2, wide: false }
    }
}

/// Dies running a tiny model, loaded and wired, ready to run.
pub struct TinyBoard {
    pub program: compiler::Program,
    pub dies: Vec<(Simulator, Microprogram)>,
    pub layouts: Vec<Layout>,
    /// Each die as built.
    pub configs: Vec<DieConfig>,
    pub lanes: Vec<Wire>,
    pub sharing: Sharing,
}

/// The words of operation `op` in `code`, from its mark to the next one or to the end of
/// its routine.
pub fn op_words(code: &Microprogram, op: u32) -> (usize, usize) {
    let at = code.marks.iter().position(|(_, o)| *o == op).expect("the operation runs on this die");
    let from = code.marks[at].0 as usize;
    let next_mark = code.marks.get(at + 1).map(|m| m.0 as usize);
    let next_entry = code.entries.iter().map(|e| *e as usize).filter(|e| *e > from).min();
    let to = [next_mark, next_entry, Some(code.words.len())].into_iter().flatten().min().unwrap();
    (from, to)
}

/// The links of `chips` chips sharing a model as `sharing` says: a ring through all of
/// them, and for groups that share products, a loop back from each group's last to its
/// first.
pub fn wires(chips: usize, sharing: Sharing) -> Vec<(usize, usize, &'static str)> {
    if chips < 2 {
        return Vec::new();
    }
    let mut wires: Vec<(usize, usize, &str)> = (0..chips).map(|c| (c, (c + 1) % chips, "link")).collect();
    // `for_chips` keeps groups only where several of them divide the chips.
    if let Sharing::Mixed(g) = sharing.for_chips(chips) {
        let g = g as usize;
        wires.extend((0..chips / g).map(|k| (k * g + g - 1, k * g, "loop")));
    }
    wires
}

/// How chips share a model, for a title.
pub fn sharing_label(sharing: Sharing) -> String {
    match sharing {
        Sharing::Layers => "each with its own layers".into(),
        Sharing::Rows => "sharing every product".into(),
        Sharing::Mixed(g) => format!("in groups of {g}"),
    }
}

/// Builds `chips` dies of shape `shape` running `model`, whose checkpoint is in `dir`.
pub fn tiny_board(model: TinyModel, dir: &Path, chips: usize, sharing: Sharing, shape: TinyDies) -> Result<TinyBoard, String> {
    let program = compiler::compile_qwen3_5(&model.config())?;
    let weights = Weights::open(dir)?;
    let base = tiny::cell_die_config();
    let base = DieConfig { microcode_bits: 15, cells: 4 << shape.cluster_bits, features: base.features.with_lane_bits(shape.lane_bits).with_cluster_bits(shape.cluster_bits).with_wide_link(shape.wide), ..base };
    let sharing = sharing.for_chips(chips);
    let share = (chips > 1).then(|| ShareOut::new(&program, &weights, chips, TINY_CONTEXT, base.cell.activation_bits, sharing)).transpose()?;
    let mut dies = Vec::new();
    let mut layouts = Vec::new();
    let mut configs = Vec::new();
    for chip in 0..chips {
        let mut layout = Layout::with_cells(&program, &weights, base, TINY_CONTEXT)?;
        let code = match &share {
            Some(share) => share.microprogram(&mut layout, &program, chip)?,
            None => layout.microprogram(&program)?,
        };
        let die = loaded_die(&layout, &code, &weights, base);
        configs.push(compact_die(base, &layout));
        layouts.push(layout);
        dies.push((die, code));
    }
    let lanes = lanes(&dies, &wires(chips, sharing));
    Ok(TinyBoard { program, dies, layouts, configs, lanes, sharing })
}

impl TinyBoard {
    /// The group and member of each die when they share products in groups.
    pub fn groups(&self) -> Vec<Option<(u32, u32)>> {
        let chips = self.dies.len();
        (0..chips)
            .map(|chip| match self.sharing {
                Sharing::Mixed(g) if g > 1 && chips > 1 => Some(((chip / g as usize) as u32, (chip % g as usize) as u32)),
                Sharing::Rows if chips > 1 => Some((0, chip as u32)),
                _ => None,
            })
            .collect()
    }

    /// Runs `tokens` one after another, then the logits of the last, recording them all.
    pub fn record(&mut self, title: &str, tokens: &[u32]) -> Result<Trace, String> {
        let sharing = sharing_name(self.sharing);
        let subject = Subject { title, sharing: &sharing, program: &self.program, context: TINY_CONTEXT };
        let mut recorder = Recorder::new(subject, &self.dies, &self.configs, &self.groups(), &self.lanes)?;
        for (position, token) in tokens.iter().enumerate() {
            recorder.run(&mut self.dies, &self.lanes, ENTRY_TOKEN, *token, position as u32);
        }
        if let Some(last) = tokens.last() {
            recorder.run(&mut self.dies, &self.lanes, ENTRY_LOGITS, *last, tokens.len() as u32 - 1);
        }
        Ok(recorder.finish())
    }

    /// Like `record`, checking after every routine each die against the exact machine of the
    /// model whose checkpoint is in `dir` (see `check::compare`).
    pub fn record_checked(&mut self, title: &str, tokens: &[u32], dir: &Path) -> Result<Trace, String> {
        let sharing = sharing_name(self.sharing);
        let subject = Subject { title, sharing: &sharing, program: &self.program, context: TINY_CONTEXT };
        let mut recorder = Recorder::new(subject, &self.dies, &self.configs, &self.groups(), &self.lanes)?;
        let mut exact = sim::ExactEngine::new(self.program.clone(), Weights::open(dir)?, self.configs[0].cell.activation_bits)?;
        let logits_from = self.program.logits_from;
        for (position, token) in tokens.iter().enumerate() {
            let last = position + 1 == tokens.len();
            recorder.run(&mut self.dies, &self.lanes, ENTRY_TOKEN, *token, position as u32);
            exact.step(*token, last)?;
            recorder.diverged(check::compare(self, &exact, 0..logits_from));
            if last {
                recorder.run(&mut self.dies, &self.lanes, ENTRY_LOGITS, *token, position as u32);
                recorder.diverged(check::compare(self, &exact, logits_from..self.program.ops.len()));
            }
        }
        Ok(recorder.finish())
    }

    /// Makes die `die` forget to store what operation `op` computes: none of its words
    /// writes to the registers. A fault to show the check finding it.
    pub fn break_op(&mut self, die: usize, op: u32) {
        let (sim, code) = &mut self.dies[die];
        let (from, to) = op_words(code, op);
        for word in &mut code.words[from..to] {
            word.write_en = false;
        }
        board::load_microprogram(sim, &self.configs[die], code);
    }

    /// The logits the first chip holds, as mantissas.
    pub fn logits(&mut self) -> Vec<i64> {
        let register = self.layouts[0].logits_register(&self.program);
        let c = self.configs[0];
        (0..self.program.vocab as u32).map(|i| netlist::demo::read_block_value(&mut self.dies[0].0, c, register, i).0).collect()
    }
}
