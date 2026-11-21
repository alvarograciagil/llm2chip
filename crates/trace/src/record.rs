//! A trace of dies at work, cycle by cycle, for the 3D viewer to play back.
//!
//! It holds what the gates did, read from the dies' own nets while they run: where each
//! sequencer was, whether it was stalled, and what every lane of every link carried. What
//! each word means (its text, its kind, the operation of the model it belongs to) is worked
//! out once per address, from the microcode and its marks.

use crate::board::{run_observed, LinkSample, Wire};
use compiler::{Op, Program};
use netlist::die::{decode_word, describe_word, OP_LINK};
use netlist::{DieConfig, Microprogram, Probe, Simulator};
use serde::{Deserialize, Serialize};

pub const TRACE_VERSION: u32 = 1;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Trace {
    pub version: u32,
    pub title: String,
    /// "gates": every cycle simulated gate by gate; "timing": operation by operation, from
    /// each chip's microprogram (see `design`).
    pub source: String,
    /// How the chips share the model: "layers", "rows" or "mixed:N".
    pub sharing: String,
    /// Tokens of cache each die has room for.
    pub context: usize,
    pub ops: Vec<OpInfo>,
    pub layers: usize,
    pub dies: Vec<DieInfo>,
    pub links: Vec<LinkInfo>,
    pub steps: Vec<StepInfo>,
    /// Slots of the cache written, in the order they were.
    pub kv: Vec<KvWrite>,
    /// Where each die first parted from the exact machine, if it did and it was checked.
    #[serde(default)]
    pub divergences: Vec<Divergence>,
}

/// One operation of the model.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct OpInfo {
    /// The operation's kind, as the compiler names it: `Matvec`, `KvAppend`, `AttnSum`…
    pub kind: String,
    /// What it computes, as the compiler names it.
    pub name: String,
    pub layer: Option<u32>,
    /// The cache or state slot it reads or writes, for attention and the delta rule.
    pub slot: Option<usize>,
}

/// What a microcode word does, in broad strokes, for colouring it.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum WordClass {
    /// The cells multiply and accumulate, or their sums move out.
    Cells,
    /// Every lane of every unit computes.
    Vector,
    /// One element at a time.
    Scalar,
    /// A word goes out to the next chip.
    Send,
    /// A word comes in from the previous chip.
    Take,
    Halt,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct WordInfo {
    pub class: WordClass,
    /// The operation of the model the word belongs to, an index into `Trace::ops`; none
    /// for the words a routine starts with before its first operation (transfers, tables).
    pub op: Option<u32>,
    /// What the word makes the die do, in a sentence.
    pub text: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct DieInfo {
    pub units: u32,
    pub lanes: u32,
    pub cells: u32,
    /// The group of chips sharing each product it belongs to, and its place in it.
    pub group: Option<(u32, u32)>,
    pub words: Vec<WordInfo>,
    /// The first word of each routine.
    pub entries: Vec<u32>,
    /// The word running in every cycle of the trace, or -1 once the die has halted.
    pub pcs: Vec<i32>,
    /// Cycles the sequencer waited on a link, as `(start, length)`.
    pub stalls: Vec<(u64, u64)>,
    /// In a trace by operations (no `pcs`): the word, here one per operation, the die runs
    /// from each `(start, length, word)`; a die is idle between them.
    #[serde(default)]
    pub spans: Vec<(u64, u64, u32)>,
}

/// One lane of a link between two dies.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct LinkInfo {
    pub from: usize,
    pub to: usize,
    pub port: String,
    /// What the lane did, as `(start, length, state)`: state 1 is a word crossing, 2 a word
    /// waiting because the receiver cannot take it. Cycles left out were idle.
    pub spans: Vec<(u64, u64, u8)>,
}

/// One run of a routine on every die at once.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct StepInfo {
    /// "token" or "logits".
    pub routine: String,
    pub token: u32,
    pub position: u32,
    /// Its first cycle in the trace.
    pub start: u64,
    pub cycles: u64,
}

/// A die starting the operation that writes a token into a layer's cache.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct KvWrite {
    pub cycle: u64,
    pub die: usize,
    pub op: u32,
    pub slot: usize,
    pub position: u32,
}

/// A value a die computed that the exact machine computes otherwise: the first, in the
/// program's order, of a routine.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Divergence {
    pub die: usize,
    /// The operation that wrote it.
    pub op: u32,
    pub buffer: usize,
    /// Its first element that differs.
    pub index: u32,
    /// The mantissas the die and the exact machine hold.
    pub got: i64,
    pub want: i64,
    /// The routine it was found after, an index into `Trace::steps`, and the cycle the die
    /// started that operation in it.
    pub step: usize,
    pub cycle: u64,
}

/// The operations of `program`, as a trace lists them.
pub fn op_infos(program: &Program) -> Vec<OpInfo> {
    program
        .ops
        .iter()
        .map(|inst| {
            let kind = format!("{:?}", inst.op).split([' ', '{', '(']).next().unwrap_or("").to_string();
            let slot = match &inst.op {
                Op::KvAppend { slot, .. } | Op::AttnSum { slot, .. } | Op::DeltaStep { slot, .. } | Op::Dconv { slot, .. } | Op::Mla { slot, .. } => Some(*slot),
                _ => None,
            };
            OpInfo { kind, name: inst.name.to_string(), layer: inst.layer, slot }
        })
        .collect()
}

/// What every word of `code` does, and which operation it belongs to.
pub fn word_infos(code: &Microprogram, c: &DieConfig) -> Vec<WordInfo> {
    let mut starts: Vec<u32> = code.entries.clone();
    starts.sort_unstable();
    code.words
        .iter()
        .enumerate()
        .map(|(address, word)| {
            let address = address as u32;
            // The routine the word is in starts at the last entry at or before it; its
            // operation is the last mark between there and the word.
            let routine = starts.iter().rev().find(|s| **s <= address).copied().unwrap_or(0);
            let op = code.marks.iter().rev().find(|(at, _)| *at <= address && *at >= routine).map(|(_, op)| *op);
            let bits: String = word.encode_words(c).iter().rev().map(|w| format!("{w:032b}")).collect();
            let text = describe_word(&decode_word(&bits, c));
            let class = if word.halt {
                WordClass::Halt
            } else if word.op == OP_LINK {
                WordClass::Take
            } else if word.link_send {
                WordClass::Send
            } else if word.cell_run || word.cell_flush || word.wide_write || word.chain_load || word.chain_shift {
                WordClass::Cells
            } else if word.vector {
                WordClass::Vector
            } else {
                WordClass::Scalar
            };
            WordInfo { class, op, text }
        })
        .collect()
}

/// What a trace is of: its title, the model and how the chips share it.
pub struct Subject<'a> {
    pub title: &'a str,
    pub sharing: &'a str,
    pub program: &'a Program,
    /// Tokens of cache each die has room for.
    pub context: usize,
}

/// Records routines run on joined dies into a `Trace`.
pub struct Recorder {
    trace: Trace,
    probes: Vec<(Probe, Option<Probe>)>,
    now: u64,
}

impl Recorder {
    /// `configs` are the dies' own configurations, as built (`compact_die`); `groups` the
    /// group and member of each die when they share products, `lanes` every lane of the
    /// links.
    pub fn new(subject: Subject, dies: &[(Simulator, Microprogram)], configs: &[DieConfig], groups: &[Option<(u32, u32)>], lanes: &[Wire]) -> Result<Self, String> {
        let Subject { title, sharing, program, context } = subject;
        let mut probes = Vec::new();
        let mut infos = Vec::new();
        for (i, ((sim, code), c)) in dies.iter().zip(configs).enumerate() {
            let pc = sim.probe("", "pc").ok_or("the die has no net named pc")?;
            probes.push((pc, sim.probe("", "not_stalled")));
            infos.push(DieInfo {
                units: 1 << c.features.cluster_bits,
                lanes: 1 << c.features.lane_bits,
                cells: c.cells,
                group: groups.get(i).copied().flatten(),
                words: word_infos(code, c),
                entries: code.entries.clone(),
                pcs: Vec::new(),
                stalls: Vec::new(),
                spans: Vec::new(),
            });
        }
        let links = lanes.iter().map(|w| LinkInfo { from: w.from, to: w.to, port: w.port.clone(), spans: Vec::new() }).collect();
        let layers = program.ops.iter().filter_map(|i| i.layer).max().map_or(0, |l| l as usize + 1);
        let trace = Trace { version: TRACE_VERSION, title: title.to_string(), source: "gates".into(), sharing: sharing.to_string(), context, ops: op_infos(program), layers, dies: infos, links, steps: Vec::new(), kv: Vec::new(), divergences: Vec::new() };
        Ok(Self { trace, probes, now: 0 })
    }

    /// Runs `routine` (`sim::lower::ENTRY_TOKEN` or `ENTRY_LOGITS`) on every die as
    /// `board::run_joined` does, recording it. Gives the cycles it took.
    pub fn run(&mut self, dies: &mut [(Simulator, Microprogram)], lanes: &[Wire], routine: usize, token: u32, position: u32) -> u64 {
        let start = self.now;
        let Self { trace, probes, .. } = self;
        let mut last_op: Vec<Option<u32>> = vec![None; dies.len()];
        let cycles = run_observed(dies, lanes, routine, token, position, |cycle, dies, samples: &[LinkSample]| {
            let at = start + cycle;
            for (d, ((sim, _), (pc, not_stalled))) in dies.iter().zip(probes.iter()).enumerate() {
                let info = &mut trace.dies[d];
                let halted = sim.get("halted") == Some(1);
                let pc = if halted { -1 } else { sim.read(pc) as i32 };
                info.pcs.push(pc);
                let stalled = !halted && not_stalled.as_ref().is_some_and(|p| sim.read(p) == 0);
                if stalled {
                    match info.stalls.last_mut() {
                        Some((from, length)) if *from + *length == at => *length += 1,
                        _ => info.stalls.push((at, 1)),
                    }
                }
                let op = usize::try_from(pc).ok().and_then(|pc| info.words.get(pc)).and_then(|w| w.op);
                if op != last_op[d] {
                    if let Some(op) = op {
                        let o = &trace.ops[op as usize];
                        if o.kind == "KvAppend" {
                            trace.kv.push(KvWrite { cycle: at, die: d, op, slot: o.slot.unwrap_or(0), position });
                        }
                    }
                    last_op[d] = op;
                }
            }
            for (link, sample) in trace.links.iter_mut().zip(samples) {
                let state = match (sample.valid, sample.ready) {
                    (true, true) => 1,
                    (true, false) => 2,
                    _ => continue,
                };
                match link.spans.last_mut() {
                    Some((from, length, s)) if *s == state && *from + *length == at => *length += 1,
                    _ => link.spans.push((at, 1, state)),
                }
            }
        });
        let routine_name = if routine == sim::lower::ENTRY_LOGITS { "logits" } else { "token" };
        self.trace.steps.push(StepInfo { routine: routine_name.into(), token, position, start, cycles });
        self.now += cycles;
        cycles
    }

    /// Notes what a check after the last routine found: each divergence gets that routine
    /// and the cycle its die started the operation in it.
    pub fn diverged(&mut self, found: Vec<Divergence>) {
        let step = self.trace.steps.len().saturating_sub(1);
        let Some(info) = self.trace.steps.get(step) else { return };
        let (start, end) = (info.start, info.start + info.cycles);
        for mut d in found {
            let die = &self.trace.dies[d.die];
            let runs = |pc: &i32| usize::try_from(*pc).ok().and_then(|pc| die.words.get(pc)).is_some_and(|w| w.op == Some(d.op));
            d.cycle = (start..end).find(|c| die.pcs.get(*c as usize).is_some_and(runs)).unwrap_or(start);
            d.step = step;
            self.trace.divergences.push(d);
        }
    }

    pub fn trace(&self) -> &Trace {
        &self.trace
    }

    pub fn finish(self) -> Trace {
        self.trace
    }
}
