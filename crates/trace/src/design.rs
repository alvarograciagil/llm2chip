//! A trace of a whole design at the grain of operations, for models too large for the
//! gates: each chip's own microprogram says how many cycles each operation of its part
//! takes, and the chips follow one another as the sizing has them.
//!
//! The chips of a group work on each operation at once and wait for the slowest of them
//! before the next (they exchange their slices within it); groups, or chips by layers,
//! take turns down the chain, and the first group makes the logits. How long a word takes
//! to cross a link is left out, as in the sizing.

use crate::record::{DieInfo, KvWrite, StepInfo, Trace, WordClass, WordInfo, TRACE_VERSION};
use compiler::Program;
use sim::chip::{ShareOut, Sharing};
use sim::lower::Layout;
use sim::sizing::{die_of, reference_die, sharing_name};
use sim::Weights;
use std::ops::Range;

/// The design to trace: its chips, how they share the model and the shape of each die.
#[derive(Clone, Copy, Debug)]
pub struct Design {
    pub chips: u32,
    pub sharing: Sharing,
    /// log2 of the vector units, of the lanes of each and of the cells of each die.
    pub cluster_bits: u32,
    pub lane_bits: u32,
    pub cell_bits: u32,
    /// The longest context the dies are built for.
    pub context: u32,
    /// Cycles each cell takes to apply a group's scale and bias.
    pub factor_steps: u32,
}

/// How an operation of a given kind looks: the cells for products, the lanes for the rest.
fn class_of(kind: &str) -> WordClass {
    match kind {
        "Matvec" | "Moe" | "SigmoidMoe" | "Mla" => WordClass::Cells,
        "EmbRead" | "LoadRow" | "LoadRows" | "StoreRow" => WordClass::Scalar,
        _ => WordClass::Vector,
    }
}

/// The cycles of each operation of each chip, for a token at `position`.
struct Costs {
    /// Per chip: the operations it runs, in order, with their cycles.
    chips: Vec<Vec<(usize, u64)>>,
}

/// Builds every chip's part and measures its operations at each of `positions`.
fn measure(program: &Program, weights: &Weights, design: Design, positions: &[u32]) -> Result<(ShareOut, Vec<Costs>), String> {
    let die = die_of(reference_die(design.factor_steps), design.context, (design.cluster_bits, design.lane_bits, design.cell_bits));
    let sharing = design.sharing.for_chips(design.chips as usize);
    let share = ShareOut::new(program, weights, design.chips as usize, design.context as usize, die.cell.activation_bits, sharing)?;
    let mut by_position: Vec<Costs> = positions.iter().map(|_| Costs { chips: Vec::new() }).collect();
    for chip in 0..design.chips as usize {
        let mut layout = match Layout::for_image_table_outside(program, weights, die, design.context as usize) {
            Ok(layout) => layout,
            Err(_) => Layout::for_image(program, weights, die, design.context as usize)?,
        };
        share.microprogram(&mut layout, program, chip)?;
        // Its part of the layers, and the logits if it is among the chips that make them.
        let mut ops: Vec<usize> = ops_of(&share, program, sharing, chip).filter(|op| *op < program.logits_from).collect();
        if makes_logits(&share, sharing, chip) {
            ops.extend(program.logits_from..program.ops.len());
        }
        for (costs, position) in by_position.iter_mut().zip(positions) {
            // Each operation on its own, as `ShareOut::cycles_by_op` measures them.
            let cycles = ops
                .iter()
                .map(|&op| Ok((op, layout.op_microprogram(program, &program.ops[op], op == 0)?.cycles(&layout.config(), 0, 1, *position))))
                .collect::<Result<Vec<_>, String>>()?;
            costs.chips.push(cycles);
        }
    }
    Ok((share, by_position))
}

/// The operations chip `chip` runs, as `ShareOut::cycles_by_op` lists them.
fn ops_of(share: &ShareOut, program: &Program, sharing: Sharing, chip: usize) -> Range<usize> {
    match share.group() {
        Some(g) => share.plan()[chip / g].clone(),
        None if sharing == Sharing::Rows => 0..program.ops.len(),
        None => share.plan()[chip].clone(),
    }
}

/// Whether chip `chip` is among those that make the logits: the first group, the first
/// chip of a chain, or every chip when they all share each product.
fn makes_logits(share: &ShareOut, sharing: Sharing, chip: usize) -> bool {
    match share.group() {
        Some(g) => chip < g,
        None => sharing == Sharing::Rows || chip == 0,
    }
}

/// Lays out a routine on the chips: `turns` are the sets of chips that work together, one
/// after another, each on the operations of `ops` that its chips run. Gives the cycle it
/// ends.
fn lay_out(trace: &mut Trace, costs: &Costs, turns: &[Vec<usize>], ops: Range<usize>, mut at: u64, position: u32) -> u64 {
    for members in turns {
        let lists: Vec<Vec<(usize, u64)>> = members.iter().map(|m| costs.chips[*m].iter().copied().filter(|(op, _)| ops.contains(op)).collect()).collect();
        let Some(first) = lists.first() else { continue };
        for (k, (op, _)) in first.iter().enumerate() {
            let longest = lists.iter().map(|l| l.get(k).map_or(0, |x| x.1)).max().unwrap_or(0).max(1);
            for (m, list) in members.iter().zip(&lists) {
                let cycles = list.get(k).map_or(0, |x| x.1).max(1);
                let die = &mut trace.dies[*m];
                die.spans.push((at, cycles, *op as u32));
                if cycles < longest {
                    die.stalls.push((at + cycles, longest - cycles));
                }
                if trace.ops[*op].kind == "KvAppend" {
                    trace.kv.push(KvWrite { cycle: at, die: *m, op: *op as u32, slot: trace.ops[*op].slot.unwrap_or(0), position });
                }
            }
            at += longest;
        }
    }
    at
}

/// Traces a token at each of `positions` (and its logits) through the whole design.
pub fn design_trace(title: &str, program: &Program, weights: &Weights, design: Design, positions: &[u32]) -> Result<Trace, String> {
    let (share, costs) = measure(program, weights, design, positions)?;
    let sharing = design.sharing.for_chips(design.chips as usize);
    let chips = design.chips as usize;
    let ops = crate::record::op_infos(program);
    let layers = program.ops.iter().filter_map(|i| i.layer).max().map_or(0, |l| l as usize + 1);
    // The chips that work together, in the order of the chain.
    let groups: Vec<Vec<usize>> = match (share.group(), sharing) {
        (Some(g), _) => (0..chips).collect::<Vec<_>>().chunks(g).map(<[usize]>::to_vec).collect(),
        (None, Sharing::Rows) => vec![(0..chips).collect()],
        _ => (0..chips).map(|c| vec![c]).collect(),
    };
    let words: Vec<WordInfo> = ops.iter().enumerate().map(|(i, o)| WordInfo { class: class_of(&o.kind), op: Some(i as u32), text: format!("{} ({})", o.name, o.kind) }).collect();
    let dies = (0..chips)
        .map(|chip| DieInfo {
            units: 1 << design.cluster_bits,
            lanes: 1 << design.lane_bits,
            cells: 1 << design.cell_bits,
            group: match share.group() {
                Some(g) => Some(((chip / g) as u32, (chip % g) as u32)),
                None if sharing == Sharing::Rows => Some((0, chip as u32)),
                None => None,
            },
            words: words.clone(),
            entries: Vec::new(),
            pcs: Vec::new(),
            stalls: Vec::new(),
            spans: Vec::new(),
        })
        .collect();
    let mut trace = Trace { version: TRACE_VERSION, title: title.to_string(), source: "timing".into(), sharing: sharing_name(sharing), context: design.context as usize, ops, layers, dies, links: Vec::new(), steps: Vec::new(), kv: Vec::new(), divergences: Vec::new() };
    let mut at = 0;
    for (costs, position) in costs.iter().zip(positions) {
        let start = at;
        at = lay_out(&mut trace, costs, &groups, 0..program.logits_from, at, *position);
        trace.steps.push(StepInfo { routine: "token".into(), token: 0, position: *position, start, cycles: at - start });
        let start = at;
        at = lay_out(&mut trace, costs, &groups[..1], program.logits_from..program.ops.len(), at, *position);
        trace.steps.push(StepInfo { routine: "logits".into(), token: 0, position: *position, start, cycles: at - start });
    }
    Ok(trace)
}
