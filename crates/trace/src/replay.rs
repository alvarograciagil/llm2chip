//! Any cycle of a recorded run again, net by net: the dies run the same routines as when
//! the trace was recorded, saving their state every so often, so a cycle is reached from
//! the last saved state before it rather than from the start.

use crate::board::{run_from, start};
use crate::{tiny_board, TinyBoard, TinyDies, TinyModel};
use netlist::{SimState, Simulator};
use serde::Serialize;
use sim::chip::Sharing;
use sim::lower::{ENTRY_LOGITS, ENTRY_TOKEN};
use std::collections::BTreeMap;
use std::path::Path;

/// Cycles between saved states: a cycle is never more than this far from one.
const EVERY: u64 = 256;

/// The state of every die at a cycle of a routine.
struct Checkpoint {
    /// The cycle in the whole trace, the step it is in and the cycle within it.
    cycle: u64,
    step: usize,
    local: u64,
    states: Vec<SimState>,
}

/// One routine of the run: where it begins in the trace and how long it takes.
struct Step {
    start: u64,
    cycles: u64,
}

/// What one module instance of one die did in one cycle.
#[derive(Clone, Debug, Serialize)]
pub struct Frame {
    pub cycle: u64,
    /// Every net of the instance, most significant bit first, as `Simulator::snapshot`.
    pub nets: BTreeMap<String, String>,
    /// For each instance inside it, the share of its bits that changed since the cycle
    /// before, everything inside it counted.
    pub activity: BTreeMap<String, f32>,
}

/// A run of a small model that can be visited at any cycle.
pub struct Replay {
    board: TinyBoard,
    steps: Vec<Step>,
    checkpoints: Vec<Checkpoint>,
}

impl Replay {
    /// Runs `tokens` on the board as `TinyBoard::record_checked` does, the same routines
    /// with the same checks between them, saving the dies' state along the way. `dir` is
    /// a folder of its own for the model's checkpoint.
    pub fn new(model: TinyModel, dir: &Path, chips: usize, sharing: Sharing, tokens: &[u32]) -> Result<Self, String> {
        model.write(dir);
        let mut board = tiny_board(model, dir, chips, sharing, TinyDies::default())?;
        let mut exact = sim::ExactEngine::new(board.program.clone(), sim::Weights::open(dir)?, board.configs[0].cell.activation_bits)?;
        let (logits_from, all) = (board.program.logits_from, board.program.ops.len());
        let (mut steps, mut checkpoints) = (Vec::new(), Vec::new());
        let mut at = 0u64;
        for (position, token) in tokens.iter().enumerate() {
            let last = position + 1 == tokens.len();
            let mut routines = vec![ENTRY_TOKEN];
            if last {
                routines.push(ENTRY_LOGITS);
            }
            for routine in routines {
                let step = steps.len();
                start(&mut board.dies, routine, *token, position as u32);
                let lanes = board.lanes.clone();
                let (cycles, _) = run_from(&mut board.dies, &lanes, 0, |local, dies, _| {
                    if local % EVERY == 0 {
                        checkpoints.push(Checkpoint { cycle: at + local, step, local, states: dies.iter().map(|(s, _)| s.save()).collect() });
                    }
                    true
                });
                steps.push(Step { start: at, cycles });
                at += cycles;
                // The same reads of the registers the recording made.
                if routine == ENTRY_TOKEN {
                    exact.step(*token, last)?;
                    crate::check::compare(&mut board, &exact, 0..logits_from);
                } else {
                    crate::check::compare(&mut board, &exact, logits_from..all);
                }
            }
        }
        Ok(Self { board, steps, checkpoints })
    }

    /// The netlist of die `die`, as built.
    pub fn design(&self, die: usize) -> netlist::Design {
        netlist::die_design(self.board.configs[die])
    }

    /// Up to `window` cycles of the instance at `path` of die `die` from `cycle` on, within
    /// the routine `cycle` is in.
    pub fn frames(&mut self, die: usize, path: &str, cycle: u64, window: u64) -> Result<Vec<Frame>, String> {
        if die >= self.board.dies.len() {
            return Err(format!("there is no die {die}"));
        }
        let step = self.steps.iter().rposition(|s| s.start <= cycle).ok_or("the cycle is before the run")?;
        let info = &self.steps[step];
        if cycle >= info.start + info.cycles {
            return Err("the cycle is past the run".into());
        }
        let saved = self.checkpoints.iter().rev().find(|c| c.step == step && c.cycle <= cycle).ok_or("no saved state before that cycle")?;
        for ((sim, _), state) in self.board.dies.iter_mut().zip(&saved.states) {
            sim.restore(state);
        }
        // The routine was started before the saved state: its entry and parameters are in it.
        let sim: &Simulator = &self.board.dies[die].0;
        if sim.snapshot(path).is_none() {
            return Err(format!("die {die} has no instance {path}"));
        }
        let children: Vec<(String, netlist::Probe)> = sim.children(path).into_iter().map(|child| (child.clone(), sim.probe_under(&format!("{path}/{child}")))).collect();
        let (first, last) = (cycle.saturating_sub(1).max(saved.cycle), (cycle + window.max(1)).min(info.start + info.cycles));
        let mut previous: Option<Vec<Vec<u8>>> = None;
        let mut frames = Vec::new();
        let lanes = self.board.lanes.clone();
        let start_local = saved.local;
        let base = info.start;
        run_from(&mut self.board.dies, &lanes, start_local, |local, dies, _| {
            let at = base + local;
            // The cycle before the first frame only gives what changed in it.
            if at + 1 < first {
                return true;
            }
            let sim = &dies[die].0;
            let now: Vec<Vec<u8>> = children.iter().map(|(_, p)| sim.read_bits(p)).collect();
            if at >= cycle {
                let activity = children
                    .iter()
                    .zip(&now)
                    .enumerate()
                    .map(|(k, ((name, _), bits))| {
                        let changed = previous.as_ref().map_or(0, |p| p[k].iter().zip(bits).filter(|(a, b)| a != b).count());
                        (name.clone(), changed as f32 / bits.len().max(1) as f32)
                    })
                    .collect();
                frames.push(Frame { cycle: at, nets: sim.snapshot(path).unwrap_or_default(), activity });
            }
            previous = Some(now);
            at + 1 < last
        });
        Ok(frames)
    }
}
