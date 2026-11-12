//! What the chip's CPU spends its cycles on, from where its program is and what it last
//! touched on the bus.
//!
//! The firmware does most of its work inline in `main`, so the bus says more than the
//! program counter: the instructions leading to an access to the host, the load controller
//! or the die's registers belong to what that access does, so preparing a token's
//! parameters is giving the die its orders. Only the tokenizer's own functions are told
//! apart by their symbols. Reads of the die's or the sampler's status that find it busy,
//! and the branches back to read it again, are waiting, kept apart from the CPU's work.

use crate::chip::{REG_DRAFT_RESULT, REG_SAMPLE_BASE, REG_SAMPLE_DONE, REG_SAMPLE_TOKEN, REG_STATUS, REG_VECTOR_DATA};
use crate::soc::PERIPHERAL_BITS;
use serde::Serialize;

/// What an instruction of the firmware was for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CpuTask {
    /// From reset to the first access to the outside.
    Start,
    /// Reading the prompt from the host.
    Receive,
    /// Turning text into tokens.
    Tokenize,
    /// Bringing rows of the image with the load controller.
    Load,
    /// Giving the die its parameters and orders, and reading its results.
    Order,
    /// Polling the die's status while it computes.
    WaitDie,
    /// Giving the sampling unit its logits and settings, and reading its token.
    Sample,
    /// Polling the sampling unit while it picks.
    WaitSampler,
    /// Turning a token back into text.
    Detokenize,
    /// Writing to the host.
    Send,
    /// The firmware's own rules between all of that: stop tokens, loops, reasoning.
    Control,
}

pub const CPU_TASKS: [CpuTask; 11] = [CpuTask::Start, CpuTask::Receive, CpuTask::Tokenize, CpuTask::Load, CpuTask::Order, CpuTask::WaitDie, CpuTask::Sample, CpuTask::WaitSampler, CpuTask::Detokenize, CpuTask::Send, CpuTask::Control];

/// One access of the CPU outside its own memories.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BusAccess {
    pub address: u32,
    pub write: bool,
    /// What a read returned, or what a write stored.
    pub value: u32,
}

/// Cycles and instructions for each task, in the order of `CPU_TASKS`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CpuCounts {
    pub cycles: [u64; CPU_TASKS.len()],
    pub instructions: [u64; CPU_TASKS.len()],
}

impl CpuCounts {
    pub fn total_cycles(&self) -> u64 {
        self.cycles.iter().sum()
    }

    /// What was counted since `before`.
    pub fn since(&self, before: &CpuCounts) -> CpuCounts {
        let mut out = *self;
        for i in 0..CPU_TASKS.len() {
            out.cycles[i] -= before.cycles[i];
            out.instructions[i] -= before.instructions[i];
        }
        out
    }

    pub fn cycles_of(&self, task: CpuTask) -> u64 {
        self.cycles[index(task)]
    }

    /// The tasks with cycles, as `(task, cycles, instructions)`.
    pub fn entries(&self) -> Vec<(CpuTask, u64, u64)> {
        CPU_TASKS.iter().enumerate().filter(|(i, _)| self.cycles[*i] > 0).map(|(i, task)| (*task, self.cycles[i], self.instructions[i])).collect()
    }
}

fn index(task: CpuTask) -> usize {
    CPU_TASKS.iter().position(|t| *t == task).unwrap()
}

/// Follows the firmware instruction by instruction.
#[derive(Clone, Debug)]
pub struct CpuProfile {
    /// Ranges of the program that belong to the tokenizer, `[start, end)` in bytes.
    functions: Vec<(u32, u32, CpuTask)>,
    /// Whether the CPU has touched the outside yet: before that it is starting.
    started: bool,
    /// Cycles and instructions not yet given to a task, waiting for the next access.
    pending: (u64, u64),
    last_cycle: u64,
    pub counts: CpuCounts,
}

impl CpuProfile {
    /// The firmware's tokenizer functions, from the symbols of its ELF file. Without symbols
    /// every instruction goes by the bus alone.
    pub fn from_elf(elf: &[u8]) -> Self {
        let functions = elf_functions(elf)
            .into_iter()
            .filter_map(|(start, size, name)| {
                if !name.contains("tokenizer_rt") {
                    return None;
                }
                // What writes a token's bytes turns tokens into text; the rest makes tokens.
                let writes = ["Writer", "4blob", "10blob_bytes", "10vocabulary"].iter().any(|part| name.contains(part));
                Some((start, start + size, if writes { CpuTask::Detokenize } else { CpuTask::Tokenize }))
            })
            .collect();
        Self { functions, started: false, pending: (0, 0), last_cycle: 0, counts: CpuCounts::default() }
    }

    /// One retired instruction at `pc`, the access it made outside the memories if any,
    /// and the CPU's cycle count after it.
    pub fn record(&mut self, pc: u32, access: Option<BusAccess>, cycle: u64) {
        let spent = cycle - self.last_cycle;
        self.last_cycle = cycle;
        if let Some((_, _, task)) = self.functions.iter().find(|(start, end, _)| (*start..*end).contains(&pc)) {
            self.add(*task, spent, 1);
            return;
        }
        self.pending = (self.pending.0 + spent, self.pending.1 + 1);
        if let Some(access) = access {
            let task = if self.started { task_of(access) } else { CpuTask::Start };
            self.started = true;
            let (cycles, instructions) = std::mem::take(&mut self.pending);
            self.add(task, cycles, instructions);
        }
    }

    /// Gives what is left since the last access to the firmware's own control: at the end
    /// of a run, or before reading the counts in the middle of one.
    pub fn settle(&mut self) {
        let (cycles, instructions) = std::mem::take(&mut self.pending);
        self.add(CpuTask::Control, cycles, instructions);
    }

    fn add(&mut self, task: CpuTask, cycles: u64, instructions: u64) {
        let at = index(task);
        self.counts.cycles[at] += cycles;
        self.counts.instructions[at] += instructions;
    }
}

/// The task an access to the outside starts.
fn task_of(access: BusAccess) -> CpuTask {
    match access.address >> 28 {
        2 if access.address & 0xC == 0 || access.address & 0xC == 4 => CpuTask::Receive,
        2 => CpuTask::Send,
        3 => CpuTask::Load,
        4 => {
            let register = (access.address >> 2) & ((1 << PERIPHERAL_BITS) - 1);
            let idle = !access.write && access.value == 0;
            match register {
                REG_STATUS if idle => CpuTask::WaitDie,
                REG_SAMPLE_DONE if idle => CpuTask::WaitSampler,
                0..=REG_VECTOR_DATA => CpuTask::Order,
                REG_SAMPLE_BASE..=REG_SAMPLE_TOKEN => CpuTask::Sample,
                REG_DRAFT_RESULT => CpuTask::Control,
                // The links to other chips are orders too.
                _ => CpuTask::Order,
            }
        }
        _ => CpuTask::Control,
    }
}

/// The functions of a 32-bit little-endian ELF file: start, size and name.
pub fn elf_functions(elf: &[u8]) -> Vec<(u32, u32, String)> {
    let read = |at: usize, n: usize| -> Option<usize> { elf.get(at..at + n).map(|b| b.iter().rev().fold(0usize, |v, x| (v << 8) | *x as usize)) };
    let mut out = Vec::new();
    let (Some(sections), Some(entry_size), Some(count)) = (read(32, 4), read(46, 2), read(48, 2)) else {
        return out;
    };
    let section = |i: usize, field: usize| read(sections + i * entry_size + field, 4);
    for i in 0..count {
        // A symbol table, and the string table it links to.
        if section(i, 4) != Some(2) {
            continue;
        }
        let (Some(offset), Some(size), Some(link)) = (section(i, 16), section(i, 20), section(i, 24)) else { continue };
        let Some(strings) = section(link, 16) else { continue };
        for symbol in (offset..offset + size).step_by(16) {
            let (Some(name), Some(value), Some(length), Some(info)) = (read(symbol, 4), read(symbol + 4, 4), read(symbol + 8, 4), read(symbol + 12, 1)) else { continue };
            if info & 0xF != 2 {
                continue;
            }
            let start = strings + name;
            let end = elf[start..].iter().position(|b| *b == 0).map_or(elf.len(), |n| start + n);
            out.push((value as u32, length as u32, String::from_utf8_lossy(&elf[start..end]).into_owned()));
        }
    }
    out.sort();
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chip::REG_ENTRY;

    const PERIPHERAL: u32 = 0x4000_0000;

    fn at(register: u32, write: bool, value: u32) -> Option<BusAccess> {
        Some(BusAccess { address: PERIPHERAL + 4 * register, write, value })
    }

    #[test]
    fn a_busy_status_is_waiting_and_the_rest_is_work() {
        let mut p = CpuProfile { functions: vec![(0x100, 0x200, CpuTask::Tokenize)], started: false, pending: (0, 0), last_cycle: 0, counts: CpuCounts::default() };
        p.record(0, Some(BusAccess { address: 0x2000_0004, write: false, value: 1 }), 3);
        p.record(4, None, 4);
        p.record(4, at(REG_ENTRY, true, 0), 5);
        p.record(8, at(REG_STATUS, false, 0), 7);
        p.record(12, None, 10);
        p.record(8, at(REG_STATUS, false, 1), 12);
        p.record(0x104, None, 20);
        p.record(16, None, 21);
        p.record(20, Some(BusAccess { address: 0x2000_0008, write: true, value: 65 }), 22);
        p.record(24, None, 25);
        p.settle();
        let c = p.counts;
        assert_eq!((c.cycles_of(CpuTask::Start), c.cycles_of(CpuTask::Order), c.cycles_of(CpuTask::WaitDie), c.cycles_of(CpuTask::Tokenize), c.cycles_of(CpuTask::Send), c.cycles_of(CpuTask::Control)), (3, 7, 2, 8, 2, 3));
        assert_eq!(c.total_cycles(), 25);
    }
}
