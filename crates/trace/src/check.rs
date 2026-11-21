//! Where the dies part from the exact machine: after every routine, each buffer a die
//! keeps whole is read and set against the exact machine's, and the first operation, in
//! the program's order, whose result differs is the place to look.

use crate::record::Divergence;
use crate::TinyBoard;
use compiler::Program;
use netlist::demo::read_block_value;
use sim::ExactEngine;
use std::collections::BTreeMap;

/// The operation that writes each buffer, the last one to if several do.
fn writers(program: &Program) -> BTreeMap<usize, usize> {
    let mut writer = BTreeMap::new();
    for (index, inst) in program.ops.iter().enumerate() {
        for buffer in inst.op.buffers().1 {
            writer.insert(buffer, index);
        }
    }
    writer
}

/// The buffers die `die` runs the writer of, among `ops`, and keeps whole: what `compare`
/// reads, as `(op, buffer)` in program order.
pub fn checked(board: &TinyBoard, die: usize, ops: std::ops::Range<usize>) -> Vec<(usize, usize)> {
    let runs: std::collections::HashSet<u32> = board.dies[die].1.marks.iter().map(|(_, op)| *op).collect();
    let layout = &board.layouts[die];
    let mut checked: Vec<(usize, usize)> = writers(&board.program)
        .into_iter()
        .filter(|(buffer, op)| ops.contains(op) && runs.contains(&(*op as u32)) && !layout.is_spread(*buffer) && !layout.is_overwritten(*buffer))
        .map(|(buffer, op)| (op, buffer))
        .collect();
    checked.sort_unstable();
    checked
}

/// Compares, on every die, the buffers of `checked` (the slice of them the die makes)
/// against `exact`. Gives the first
/// difference of each die, by the order of the operations. Reading the dies' registers
/// takes clock cycles of their own, after they have halted, with their links idle.
pub fn compare(board: &mut TinyBoard, exact: &ExactEngine, ops: std::ops::Range<usize>) -> Vec<Divergence> {
    let mut found = Vec::new();
    for die in 0..board.dies.len() {
        let list = checked(board, die, ops.clone());
        let c = board.configs[die];
        for lane in board.lanes.iter().filter(|w| w.to == die) {
            board.dies[die].0.set(&format!("{}_rx_valid", lane.port), 0).unwrap();
        }
        'die: for (op, buffer) in list {
            let gates = board.layouts[die].register_of(buffer);
            let fast = exact.register_of(buffer);
            // Chips that share the heads each make their own slice of some buffers.
            let (first, count) = board.layouts[die].slice_of(&board.program, buffer);
            for index in first..first + count {
                let (got, _) = read_block_value(&mut board.dies[die].0, c, gates, index);
                let want = exact.mantissa(fast, index);
                if got != want {
                    found.push(Divergence { die, op: op as u32, buffer, index, got, want, cycle: 0, step: 0 });
                    break 'die;
                }
            }
        }
    }
    found
}
