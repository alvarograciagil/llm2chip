//! Test patterns for the stuck-at faults of a design with a scan chain (`scan::insert_scan`).
//!
//! With the chain, the test of a sequential circuit becomes that of its logic between
//! flip-flops:
//! - **what the tester sets**: the input pins and the state it shifts in;
//! - **what it reads**: the output pins and what the flip-flops capture on the one clock
//!   with `scan_enable` low.
//!
//! The faults are the classic stuck-at ones:
//! - each signal stuck at 0 or at 1, at its source;
//! - each gate input stuck, where its signal goes to more than one place.
//!
//! The patterns are made in two passes:
//! 1. **Random patterns**, 64 at a time, keeping those that find a fault no earlier one
//!    found.
//! 2. **PODEM** on the faults left: it searches the inputs for a pattern that sets the
//!    fault's site to the opposite value and carries the difference to something the tester
//!    reads. If the whole search fails, the fault is untestable: no pattern can find it,
//!    usually because the logic around it is redundant.
//!
//! Values have three states: 0, 1, and unknown for what a memory outputs, which the test
//! does not control. A fault counts as detected only where the good and the faulty circuit
//! both give a known value, and the two differ.

use crate::scan::{SCAN_ENABLE, SCAN_IN, SCAN_OUT, TEST_MODE};
use crate::sim::{flatten, Flat, Gate, Op};
use crate::{Design, SimError};
use std::cmp::Reverse;
use std::collections::BinaryHeap;

const ZERO: usize = 0;
const ONE: usize = 1;

/// Where a fault sits: at a signal's source, or at one gate input.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Site {
    Stem(usize),
    /// Gate index and input: 0 is `a`, 1 is `b`, 2 a multiplexer's select.
    Branch(usize, usize),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Fault {
    pub site: Site,
    pub stuck: bool,
}

/// One test: what the tester sets and what it should read back.
#[derive(Clone, Debug)]
pub struct TestPattern {
    /// Each input pin but the four of the chain, bit by bit.
    pub inputs: Vec<(String, Vec<bool>)>,
    /// The state shifted in, the flip-flop next to `scan_in` first.
    pub load: Vec<bool>,
    /// Each output pin but `scan_out`, before the capture clock; `None` if unknown.
    pub outputs: Vec<(String, Vec<Option<bool>>)>,
    /// What the capture leaves in the chain, in the order of `load`.
    pub unload: Vec<Option<bool>>,
}

#[derive(Clone, Debug, Default)]
pub struct AtpgReport {
    pub faults: usize,
    /// Found by the capture patterns.
    pub detected: usize,
    /// On the shift path only: found by shifting a pattern through the chain (`chain_test`).
    pub chain: usize,
    /// No pattern can find them.
    pub untestable: usize,
    /// The search gave up on them.
    pub aborted: usize,
    pub random_patterns: usize,
    pub deterministic_patterns: usize,
    pub chain_length: usize,
    pub patterns: Vec<TestPattern>,
    /// What became of each fault the capture patterns aim at.
    pub outcomes: Vec<(Fault, Outcome)>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    Detected,
    Untestable,
    Aborted,
}

impl AtpgReport {
    /// Share of all faults found.
    pub fn fault_coverage(&self) -> f64 {
        (self.detected + self.chain) as f64 / self.faults.max(1) as f64
    }

    /// Share of the faults some pattern could find that the patterns do find.
    pub fn test_coverage(&self) -> f64 {
        (self.detected + self.chain) as f64 / (self.faults - self.untestable).max(1) as f64
    }
}

/// Values of 64 patterns at once: the patterns where a signal is 1, and where it is 0.
/// Neither means unknown.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Wide {
    one: u64,
    zero: u64,
}

const KNOWN_ONE: Wide = Wide { one: !0, zero: 0 };
const KNOWN_ZERO: Wide = Wide { one: 0, zero: !0 };
const UNKNOWN: Wide = Wide { one: 0, zero: 0 };

fn evaluate(op: Op, a: Wide, b: Wide, c: Wide) -> Wide {
    match op {
        Op::And => Wide { one: a.one & b.one, zero: a.zero | b.zero },
        Op::Or => Wide { one: a.one | b.one, zero: a.zero & b.zero },
        Op::Xor => Wide { one: (a.one & b.zero) | (a.zero & b.one), zero: (a.one & b.one) | (a.zero & b.zero) },
        Op::Not => Wide { one: a.zero, zero: a.one },
        // `y = c ? b : a`, known where both agree too.
        Op::Mux => Wide { one: (c.zero & a.one) | (c.one & b.one) | (a.one & b.one), zero: (c.zero & a.zero) | (c.one & b.zero) | (a.zero & b.zero) },
    }
}

/// Patterns where two values are both known and differ.
fn differ(good: Wide, bad: Wide) -> u64 {
    (good.one & bad.zero) | (good.zero & bad.one)
}

fn constant(value: bool) -> Wide {
    if value { KNOWN_ONE } else { KNOWN_ZERO }
}

/// The design as the test sees it: what it controls and observes, around one level of gates.
struct View {
    flat: Flat,
    driver: Vec<u32>,
    fanout_start: Vec<u32>,
    fanout: Vec<u32>,
    observed: Vec<bool>,
    /// Signals the tester sets: input pins first, then the flip-flops' outputs.
    controlled: Vec<usize>,
    /// Signals fixed during the capture: `test_mode` high and `scan_enable` low.
    fixed: Vec<(usize, bool)>,
    /// What the tester reads: output pins, then what each flip-flop captures.
    reads: Vec<usize>,
    /// The flip-flops along the chain from `scan_in`.
    chain: Vec<usize>,
    pins_in: Vec<(String, usize)>,
    pins_out: Vec<(String, usize)>,
    /// Each signal's place in `controlled`, or `NO_DRIVER`.
    position: Vec<u32>,
    /// The buffers that carry the chain from module to module.
    shift_path: Vec<bool>,
    /// How hard each signal is to set to 0 and to 1 (SCOAP): roughly, how many inputs and
    /// gates that takes. `HARD` if the tester cannot.
    effort: Vec<[u32; 2]>,
}

const HARD: u32 = u32::MAX / 4;

const NO_DRIVER: u32 = u32::MAX;

impl View {
    fn new(design: &Design) -> Result<Self, SimError> {
        Self::build(design, false)
    }

    fn build(design: &Design, named: bool) -> Result<Self, SimError> {
        let flat = flatten(design, named)?;
        let mut driver = vec![NO_DRIVER; flat.signals];
        for (i, g) in flat.gates.iter().enumerate() {
            driver[g.y] = i as u32;
        }
        let mut count = vec![0u32; flat.signals + 1];
        for g in &flat.gates {
            for pin in g.inputs() {
                count[g.input(*pin) + 1] += 1;
            }
        }
        for s in 0..flat.signals {
            count[s + 1] += count[s];
        }
        let mut next = count.clone();
        let mut fanout = vec![0u32; count[flat.signals] as usize];
        for (i, g) in flat.gates.iter().enumerate() {
            for pin in g.inputs() {
                let s = g.input(*pin);
                fanout[next[s] as usize] = i as u32;
                next[s] += 1;
            }
        }
        let pin = |name: &str| flat.inputs.iter().find(|(n, _)| n == name).map(|(_, bits)| bits[0]).ok_or_else(|| SimError(format!("el diseño no tiene el pin {name}: falta la cadena de scan")));
        let (enable, test_mode, scan_in) = (pin(SCAN_ENABLE)?, pin(TEST_MODE)?, pin(SCAN_IN)?);
        let scan_out = flat.outputs.iter().find(|(n, _)| n == SCAN_OUT).map(|(_, bits)| bits[0]).ok_or_else(|| SimError("el diseño no tiene scan_out".into()))?;
        let mut pins_in = Vec::new();
        let mut controlled = Vec::new();
        for (name, bits) in flat.inputs.iter().filter(|(n, _)| ![SCAN_ENABLE, TEST_MODE, SCAN_IN].contains(&n.as_str())) {
            pins_in.push((name.clone(), bits.len()));
            controlled.extend(bits);
        }
        let mut pins_out = Vec::new();
        let mut reads = Vec::new();
        for (name, bits) in flat.outputs.iter().filter(|(n, _)| n != SCAN_OUT) {
            pins_out.push((name.clone(), bits.len()));
            reads.extend(bits);
        }

        // The chain: from `scan_in`, the flip-flop whose shift input is the signal so far, then
        // its output, through the buffers that join modules.
        let through_buffers = |mut s: usize| {
            while driver[s] != NO_DRIVER {
                let g = flat.gates[driver[s] as usize];
                if g.op == Op::Or && g.b == ZERO {
                    s = g.a;
                } else {
                    break;
                }
            }
            s
        };
        let mut after = vec![usize::MAX; flat.signals];
        for (k, (d, _)) in flat.flops.iter().enumerate() {
            if driver[*d] != NO_DRIVER {
                let g = flat.gates[driver[*d] as usize];
                if g.op == Op::Mux && g.c == enable {
                    after[through_buffers(g.b)] = k;
                }
            }
        }
        let mut chain = Vec::with_capacity(flat.flops.len());
        let mut at = scan_in;
        while after[at] != usize::MAX && chain.len() < flat.flops.len() {
            let k = after[at];
            chain.push(k);
            at = flat.flops[k].1;
        }
        if chain.len() != flat.flops.len() || through_buffers(scan_out) != at {
            return Err(SimError(format!("la cadena de scan recorre {} de {} biestables", chain.len(), flat.flops.len())));
        }
        // The buffers between one link of the chain and the next, and up to `scan_out`.
        let mut shift_path = vec![false; flat.signals];
        let ends = flat.flops.iter().filter(|(d, _)| driver[*d] != NO_DRIVER).map(|(d, _)| flat.gates[driver[*d] as usize]).filter(|g| g.op == Op::Mux && g.c == enable).map(|g| g.b);
        for mut s in ends.chain([scan_out]) {
            while driver[s] != NO_DRIVER {
                let g = flat.gates[driver[s] as usize];
                if g.op != Op::Or || g.b != ZERO {
                    break;
                }
                shift_path[s] = true;
                s = g.a;
            }
        }
        controlled.extend(chain.iter().map(|k| flat.flops[*k].1));
        reads.extend(chain.iter().map(|k| flat.flops[*k].0));
        let mut observed = vec![false; flat.signals];
        for s in &reads {
            observed[*s] = true;
        }
        let mut position = vec![NO_DRIVER; flat.signals];
        for (k, s) in controlled.iter().enumerate() {
            position[*s] = k as u32;
        }
        let fixed = vec![(test_mode, true), (enable, false), (scan_in, false)];
        let mut effort = vec![[HARD, HARD]; flat.signals];
        effort[ZERO] = [0, HARD];
        effort[ONE] = [HARD, 0];
        for s in &controlled {
            effort[*s] = [1, 1];
        }
        for (s, value) in &fixed {
            effort[*s] = if *value { [HARD, 0] } else { [0, HARD] };
        }
        let add = |a: u32, b: u32| (a + b).min(HARD);
        for g in &flat.gates {
            let (a, b, c) = (effort[g.a], effort[g.b], effort[g.c]);
            effort[g.y] = match g.op {
                Op::And => [add(a[0].min(b[0]), 1), add(add(a[1], b[1]), 1)],
                Op::Or => [add(add(a[0], b[0]), 1), add(a[1].min(b[1]), 1)],
                Op::Xor => [add(add(a[0], b[0]).min(add(a[1], b[1])), 1), add(add(a[0], b[1]).min(add(a[1], b[0])), 1)],
                Op::Not => [add(a[1], 1), add(a[0], 1)],
                Op::Mux => [0, 1].map(|v| add(add(c[0], a[v]).min(add(c[1], b[v])), 1)),
            };
        }
        Ok(Self { shift_path, effort, position, driver, fanout_start: count, fanout, observed, controlled, fixed, reads, chain, pins_in, pins_out, flat })
    }

    fn fanout_of(&self, s: usize) -> &[u32] {
        &self.fanout[self.fanout_start[s] as usize..self.fanout_start[s + 1] as usize]
    }

    /// The good circuit for 64 patterns, given the controlled signals' values.
    fn simulate(&self, controlled: &[Wide]) -> Vec<Wide> {
        let mut v = vec![UNKNOWN; self.flat.signals];
        v[ZERO] = KNOWN_ZERO;
        v[ONE] = KNOWN_ONE;
        for (s, value) in self.controlled.iter().zip(controlled) {
            v[*s] = *value;
        }
        for (s, value) in &self.fixed {
            v[*s] = constant(*value);
        }
        for g in &self.flat.gates {
            v[g.y] = evaluate(g.op, v[g.a], v[g.b], v[g.c]);
        }
        v
    }

    /// The stuck-at faults, the shift path's apart: those the capture cannot reach.
    /// The faults the capture patterns aim at, and those on the shift path, which only the
    /// chain test can find.
    fn faults(&self) -> (Vec<Fault>, Vec<Fault>) {
        let enable = self.fixed[1].0;
        let scan_in = self.fixed[2].0;
        let test_mode = self.fixed[0].0;
        let shift_input = |g: &Gate, pin: usize| g.op == Op::Mux && g.c == enable && pin == 1;
        // Signals that only carry the chain: the buffers that take it from one module to the
        // next, up to a flip-flop's shift input or `scan_out`.
        let shift_only = |s: usize| {
            self.shift_path[s]
                && !self.observed[s]
                && self.fanout_of(s).iter().all(|g| {
                    let g = &self.flat.gates[*g as usize];
                    (shift_input(g, 1) && g.b == s && g.a != s && g.c != s) || (g.op == Op::Or && g.b == ZERO && g.a == s && self.shift_path[g.y])
                })
        };
        let mut set = vec![false; self.flat.signals];
        let mut memory = vec![false; self.flat.signals];
        for s in &self.controlled {
            set[*s] = true;
        }
        for s in &self.flat.memory_outputs {
            memory[*s] = true;
        }
        let mut faults = Vec::new();
        let mut chain = Vec::new();
        for s in 2..self.flat.signals {
            // The memories' own faults are the march test's.
            if s == test_mode || s == scan_in || memory[s] {
                continue;
            }
            let fan = self.fanout_of(s).len();
            if fan == 0 && !self.observed[s] && self.driver[s] == NO_DRIVER && !set[s] {
                continue;
            }
            if shift_only(s) {
                chain.extend([false, true].map(|stuck| Fault { site: Site::Stem(s), stuck }));
                continue;
            }
            for stuck in [false, true] {
                // `scan_enable` stuck low stops the shifting, which the chain test shows.
                if s == enable && !stuck {
                    chain.push(Fault { site: Site::Stem(s), stuck });
                    continue;
                }
                faults.push(Fault { site: Site::Stem(s), stuck });
            }
            if fan > 1 || (fan == 1 && self.observed[s]) {
                for g in self.fanout_of(s) {
                    let gate = &self.flat.gates[*g as usize];
                    for pin in gate.inputs() {
                        if gate.input(*pin) != s {
                            continue;
                        }
                        if shift_input(gate, *pin) {
                            chain.extend([false, true].map(|stuck| Fault { site: Site::Branch(*g as usize, *pin), stuck }));
                            continue;
                        }
                        for stuck in [false, true] {
                            // A flip-flop whose select is stuck low does not shift.
                            if gate.op == Op::Mux && gate.c == enable && *pin == 2 && !stuck {
                                chain.push(Fault { site: Site::Branch(*g as usize, *pin), stuck });
                                continue;
                            }
                            faults.push(Fault { site: Site::Branch(*g as usize, *pin), stuck });
                        }
                    }
                }
            }
        }
        (faults, chain)
    }

    /// Patterns, among the 64 of `good`, that find `fault`, or 0; event-driven from its site.
    fn detects(&self, good: &[Wide], fault: Fault, scratch: &mut Scratch) -> u64 {
        scratch.epoch += 1;
        let epoch = scratch.epoch;
        let forced = constant(fault.stuck);
        let mut found = 0u64;
        let set = |scratch: &mut Scratch, s: usize, value: Wide| -> u64 {
            scratch.bad[s] = value;
            scratch.stamp[s] = epoch;
            for g in self.fanout_of(s) {
                if scratch.queued[*g as usize] != epoch {
                    scratch.queued[*g as usize] = epoch;
                    scratch.heap.push(Reverse(*g));
                }
            }
            if self.observed[s] { differ(good[s], value) } else { 0 }
        };
        let value_of = |scratch: &Scratch, s: usize| if scratch.stamp[s] == epoch { scratch.bad[s] } else { good[s] };
        match fault.site {
            Site::Stem(s) => {
                if good[s] == forced {
                    return 0;
                }
                found |= set(scratch, s, forced);
            }
            Site::Branch(g, pin) => {
                let gate = self.flat.gates[g];
                let mut inputs = [good[gate.a], good[gate.b], good[gate.c]];
                inputs[pin] = forced;
                let y = evaluate(gate.op, inputs[0], inputs[1], inputs[2]);
                if y != good[gate.y] {
                    found |= set(scratch, gate.y, y);
                }
            }
        }
        while let Some(Reverse(g)) = scratch.heap.pop() {
            if found != 0 {
                break;
            }
            let gate = self.flat.gates[g as usize];
            let y = evaluate(gate.op, value_of(scratch, gate.a), value_of(scratch, gate.b), value_of(scratch, gate.c));
            if y != value_of(scratch, gate.y) {
                found |= set(scratch, gate.y, y);
            }
        }
        scratch.heap.clear();
        found
    }
}

struct Scratch {
    bad: Vec<Wide>,
    stamp: Vec<u32>,
    queued: Vec<u32>,
    heap: BinaryHeap<Reverse<u32>>,
    epoch: u32,
}

/// A small, fixed random sequence: the same design always gets the same patterns.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

/// Three states for one pattern: 0, 1, and `X` unknown.
const X: u8 = 2;

fn evaluate_one(op: Op, a: u8, b: u8, c: u8) -> u8 {
    match op {
        Op::And => match (a, b) {
            (0, _) | (_, 0) => 0,
            (1, 1) => 1,
            _ => X,
        },
        Op::Or => match (a, b) {
            (1, _) | (_, 1) => 1,
            (0, 0) => 0,
            _ => X,
        },
        Op::Xor => if a == X || b == X { X } else { a ^ b },
        Op::Not => if a == X { X } else { 1 ^ a },
        Op::Mux => match c {
            0 => a,
            1 => b,
            _ => if a == b { a } else { X },
        },
    }
}

enum Search {
    Found(Vec<u8>),
    Untestable,
    Aborted,
}

impl View {
    /// Good and faulty values of one pattern, inputs as `assigned`.
    fn simulate_one(&self, assigned: &[u8], fault: Fault, good: &mut [u8], bad: &mut [u8]) {
        good.fill(X);
        good[ZERO] = 0;
        good[ONE] = 1;
        for (s, value) in self.controlled.iter().zip(assigned) {
            good[*s] = *value;
        }
        for (s, value) in &self.fixed {
            good[*s] = *value as u8;
        }
        bad.copy_from_slice(good);
        let stuck = fault.stuck as u8;
        if let Site::Stem(s) = fault.site {
            bad[s] = stuck;
        }
        for (i, g) in self.flat.gates.iter().enumerate() {
            good[g.y] = evaluate_one(g.op, good[g.a], good[g.b], good[g.c]);
            let mut inputs = [bad[g.a], bad[g.b], bad[g.c]];
            if let Site::Branch(at, pin) = fault.site {
                if at == i {
                    inputs[pin] = stuck;
                }
            }
            bad[g.y] = evaluate_one(g.op, inputs[0], inputs[1], inputs[2]);
            if fault.site == Site::Stem(g.y) {
                bad[g.y] = stuck;
            }
        }
    }

    /// From a wish for signal `s`, the controlled input to set and its value; `None` if the
    /// way leads to something the tester cannot set.
    fn backtrace(&self, mut s: usize, mut value: u8, good: &[u8]) -> Option<(usize, u8)> {
        loop {
            if self.position[s] != NO_DRIVER {
                return (good[s] == X).then_some((self.position[s] as usize, value));
            }
            if self.driver[s] == NO_DRIVER {
                return None;
            }
            let g = self.flat.gates[self.driver[s] as usize];
            let unknown = |t: usize| good[t] == X;
            let cost = |t: usize, v: u8| self.effort[t][v as usize];
            // Of the unknown inputs, the easiest to set to `v`, or with `hardest` the hardest:
            // when every input must take it, the hardest first shows soonest if it cannot.
            let pick = |inputs: [usize; 2], v: u8, hardest: bool| {
                let mut open = inputs.into_iter().filter(|t| unknown(*t));
                let first = open.next().unwrap_or(inputs[1]);
                open.fold(first, |best, t| if (cost(t, v) > cost(best, v)) == hardest && cost(t, v) != cost(best, v) { t } else { best })
            };
            match g.op {
                Op::Not => {
                    s = g.a;
                    value ^= 1;
                }
                // All inputs at the non-controlling value, or any one at the controlling one.
                Op::And => s = pick([g.a, g.b], value, value == 1),
                Op::Or => s = pick([g.a, g.b], value, value == 0),
                Op::Xor => {
                    let to = if unknown(g.a) && unknown(g.b) {
                        if cost(g.a, 0).min(cost(g.a, 1)) <= cost(g.b, 0).min(cost(g.b, 1)) { g.a } else { g.b }
                    } else if unknown(g.a) {
                        g.a
                    } else {
                        g.b
                    };
                    let other = if to == g.a { g.b } else { g.a };
                    if good[other] != X {
                        value ^= good[other];
                    } else if cost(to, value) > cost(to, 1 - value) {
                        // The other input still free: ask this one for its easier value.
                        value ^= 1;
                    }
                    s = to;
                }
                Op::Mux => match good[g.c] {
                    0 => s = g.a,
                    1 => s = g.b,
                    _ => {
                        let side = |data: usize, select: u8| match good[data] {
                            X => (cost(g.c, select) + cost(data, value)).min(HARD),
                            known if known == value => cost(g.c, select),
                            _ => HARD,
                        };
                        s = g.c;
                        value = (side(g.b, 1) < side(g.a, 0)) as u8;
                    }
                },
            }
        }
    }

    /// PODEM: a pattern for `fault`, or why there is none.
    fn podem(&self, fault: Fault, limit: usize, good: &mut [u8], bad: &mut [u8]) -> Search {
        let mut assigned = vec![X; self.controlled.len()];
        let mut decisions: Vec<(usize, bool)> = Vec::new();
        let mut backtracks = 0;
        // A way blocked by something the tester cannot set makes a failed search inconclusive.
        let mut blocked = false;
        let (site, stuck) = match fault.site {
            Site::Stem(s) => (s, fault.stuck as u8),
            Site::Branch(g, pin) => (self.flat.gates[g].input(pin), fault.stuck as u8),
        };
        loop {
            self.simulate_one(&assigned, fault, good, bad);
            let detected = self.reads.iter().any(|s| good[*s] != X && bad[*s] != X && good[*s] != bad[*s]);
            if detected {
                return Search::Found(assigned);
            }
            let reaches = self.open_paths(good, bad);
            // Before setting the site, there must still be a way out from it.
            let way_out = match fault.site {
                Site::Stem(s) => self.observed[s] || self.fanout_of(s).iter().any(|f| reaches[self.flat.gates[*f as usize].y]),
                Site::Branch(g, _) => reaches[self.flat.gates[g].y],
            };
            let objective = if good[site] == X {
                way_out.then_some((site, 1 - stuck))
            } else if good[site] == stuck {
                None
            } else {
                self.propagation_objective(fault, good, bad, &reaches)
            };
            let step = objective.and_then(|(s, v)| {
                let traced = self.backtrace(s, v, good);
                blocked |= traced.is_none();
                traced
            });
            match step {
                Some((k, value)) => {
                    assigned[k] = value;
                    decisions.push((k, false));
                }
                None => loop {
                    let Some((k, flipped)) = decisions.pop() else {
                        return if blocked { Search::Aborted } else { Search::Untestable };
                    };
                    if !flipped {
                        backtracks += 1;
                        if backtracks > limit {
                            return Search::Aborted;
                        }
                        assigned[k] ^= 1;
                        decisions.push((k, true));
                        break;
                    }
                    assigned[k] = X;
                },
            }
        }
    }

    /// With the fault active, a value that carries its difference one gate further.
    /// Whether each signal still unsettled reaches something the tester reads through
    /// unsettled signals: a difference can only travel along such a path.
    fn open_paths(&self, good: &[u8], bad: &[u8]) -> Vec<bool> {
        let open = |s: usize| good[s] == X || bad[s] == X;
        let mut reaches = vec![false; self.flat.signals];
        for gate in self.flat.gates.iter().rev() {
            let y = gate.y;
            reaches[y] = open(y) && (self.observed[y] || self.fanout_of(y).iter().any(|f| reaches[self.flat.gates[*f as usize].y]));
        }
        reaches
    }

    fn propagation_objective(&self, fault: Fault, good: &[u8], bad: &[u8], reaches: &[bool]) -> Option<(usize, u8)> {
        let differs = |g: usize, pin: usize, s: usize| {
            let faulty = match fault.site {
                Site::Branch(at, p) if at == g && p == pin => fault.stuck as u8,
                _ => bad[s],
            };
            good[s] != X && faulty != X && good[s] != faulty
        };
        // The gates nearest the outputs first.
        for (i, gate) in self.flat.gates.iter().enumerate().rev() {
            if !reaches[gate.y] {
                continue;
            }
            let carrying: Vec<usize> = gate.inputs().iter().copied().filter(|pin| differs(i, *pin, gate.input(*pin))).collect();
            if carrying.is_empty() {
                continue;
            }
            let unknown = |s: usize| good[s] == X;
            let wish = match gate.op {
                Op::And => [gate.a, gate.b].into_iter().find(|s| unknown(*s)).map(|s| (s, 1)),
                Op::Or => [gate.a, gate.b].into_iter().find(|s| unknown(*s)).map(|s| (s, 0)),
                Op::Xor => [gate.a, gate.b].into_iter().find(|s| unknown(*s)).map(|s| (s, 0)),
                Op::Not => None,
                Op::Mux => {
                    if carrying.contains(&0) && unknown(gate.c) {
                        Some((gate.c, 0))
                    } else if carrying.contains(&1) && unknown(gate.c) {
                        Some((gate.c, 1))
                    } else if carrying.contains(&2) {
                        // The select differs: the two data inputs must too.
                        match (good[gate.a], good[gate.b]) {
                            (X, X) => Some((gate.a, 0)),
                            (X, b) => Some((gate.a, 1 - b)),
                            (a, X) => Some((gate.b, 1 - a)),
                            _ => None,
                        }
                    } else {
                        None
                    }
                }
            };
            if wish.is_some() {
                return wish;
            }
        }
        None
    }
}

/// Patterns for every stuck-at fault of `design`, which must have a scan chain. PODEM gives
/// up on a fault after `limit` backtracks.
pub fn generate(design: &Design, limit: usize) -> Result<AtpgReport, SimError> {
    let view = View::new(design)?;
    let (faults, chain) = view.faults();
    let mut remaining: Vec<Fault> = faults.clone();
    let signals = view.flat.signals;
    let mut scratch = Scratch { bad: vec![UNKNOWN; signals], stamp: vec![0; signals], queued: vec![0; view.flat.gates.len()], heap: BinaryHeap::new(), epoch: 0 };
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut kept: Vec<Vec<bool>> = Vec::new();
    let mut report = AtpgReport { faults: faults.len() + chain.len(), chain: chain.len(), chain_length: view.chain.len(), ..Default::default() };

    // Drops the faults a batch of 64 finds, and keeps the patterns that found one first.
    let apply = |controlled: Vec<Wide>, remaining: &mut Vec<Fault>, scratch: &mut Scratch, kept: &mut Vec<Vec<bool>>| -> usize {
        let good = view.simulate(&controlled);
        let mut useful = 0u64;
        let before = remaining.len();
        remaining.retain(|fault| {
            let found = view.detects(&good, *fault, scratch);
            useful |= found.isolate_lowest_one();
            found == 0
        });
        for bit in 0..64 {
            if useful >> bit & 1 == 1 {
                kept.push(controlled.iter().map(|w| w.one >> bit & 1 == 1).collect());
            }
        }
        before - remaining.len()
    };

    // Random patterns while they keep finding faults.
    let mut idle = 0;
    while idle < 4 && !remaining.is_empty() {
        let controlled: Vec<Wide> = view.controlled.iter().map(|_| {
            let bits = rng.next();
            Wide { one: bits, zero: !bits }
        }).collect();
        let found = apply(controlled, &mut remaining, &mut scratch, &mut kept);
        idle = if found * 1000 < faults.len() { idle + 1 } else { 0 };
    }
    report.random_patterns = kept.len();

    // PODEM on what is left, its patterns in batches of 64 that drop other faults too.
    let (mut good, mut bad) = (vec![X; signals], vec![X; signals]);
    let mut batch: Vec<Vec<u8>> = Vec::new();
    let flush = |batch: &mut Vec<Vec<u8>>, remaining: &mut Vec<Fault>, scratch: &mut Scratch, kept: &mut Vec<Vec<bool>>, rng: &mut Rng| {
        if batch.is_empty() {
            return;
        }
        let controlled: Vec<Wide> = (0..view.controlled.len())
            .map(|k| {
                let fill = rng.next();
                let mut one = 0u64;
                for bit in 0..64 {
                    let value = batch.get(bit).map_or(fill >> bit & 1, |p| if p[k] == X { fill >> bit & 1 } else { p[k] as u64 });
                    one |= value << bit;
                }
                Wide { one, zero: !one }
            })
            .collect();
        apply(controlled, remaining, scratch, kept);
        batch.clear();
    };
    let mut lost: std::collections::HashMap<Fault, Outcome> = std::collections::HashMap::new();
    let mut index = 0;
    while index < remaining.len() {
        let fault = remaining[index];
        match view.podem(fault, limit, &mut good, &mut bad) {
            Search::Found(pattern) => {
                batch.push(pattern);
                if batch.len() == 64 {
                    flush(&mut batch, &mut remaining, &mut scratch, &mut kept, &mut rng);
                    index = 0;
                    continue;
                }
                index += 1;
            }
            Search::Untestable => {
                report.untestable += 1;
                lost.insert(remaining.remove(index), Outcome::Untestable);
            }
            Search::Aborted => {
                report.aborted += 1;
                lost.insert(remaining.remove(index), Outcome::Aborted);
            }
        }
    }
    let pending = batch.len();
    flush(&mut batch, &mut remaining, &mut scratch, &mut kept, &mut rng);
    // Each PODEM pattern finds its own fault: none should be left.
    if !remaining.is_empty() {
        return Err(SimError(format!("{} fallos con patrón de PODEM no se detectan al simularlos (de {pending} pendientes)", remaining.len())));
    }
    report.deterministic_patterns = kept.len() - report.random_patterns;
    report.detected = faults.len() - report.untestable - report.aborted;
    report.patterns = kept.iter().map(|controlled| view.pattern(controlled)).collect();
    report.outcomes = faults.iter().map(|f| (*f, lost.get(f).copied().unwrap_or(Outcome::Detected))).collect();
    Ok(report)
}

impl View {
    /// The pattern with these controlled values, and what the good circuit answers.
    fn pattern(&self, controlled: &[bool]) -> TestPattern {
        let wide: Vec<Wide> = controlled.iter().map(|b| constant(*b)).collect();
        let good = self.simulate(&wide);
        let read = |s: usize| match (good[s].one & 1, good[s].zero & 1) {
            (1, _) => Some(true),
            (_, 1) => Some(false),
            _ => None,
        };
        let mut at = 0;
        let inputs = self.pins_in.iter().map(|(name, width)| {
            at += width;
            (name.clone(), controlled[at - width..at].to_vec())
        }).collect();
        let load = controlled[at..].to_vec();
        let mut at = 0;
        let outputs = self.pins_out.iter().map(|(name, width)| {
            at += width;
            (name.clone(), self.reads[at - width..at].iter().map(|s| read(*s)).collect())
        }).collect();
        let unload = self.reads[at..].iter().map(|s| read(*s)).collect();
        TestPattern { inputs, load, outputs, unload }
    }
}

/// Where each fault sits, by the name of its signal and the gate it enters.
pub fn describe(design: &Design, faults: &[Fault]) -> Result<Vec<String>, SimError> {
    let view = View::build(design, true)?;
    let name = |s: usize| view.flat.names.get(s).cloned().filter(|n| !n.is_empty()).unwrap_or_else(|| format!("#{s}"));
    Ok(faults
        .iter()
        .map(|f| match f.site {
            Site::Stem(s) => format!("{} a {}", name(s), f.stuck as u8),
            Site::Branch(g, pin) => {
                let gate = view.flat.gates[g];
                format!("{} a {} en la entrada {pin} de {:?} → {}", name(gate.input(pin)), f.stuck as u8, gate.op, name(gate.y))
            }
        })
        .collect())
}

/// For tests: each fault of `design` and whether one pattern (controlled values in the order
/// of `TestPattern`'s `inputs` then `load`) finds it, by simulating the whole faulty circuit.
#[doc(hidden)]
pub fn faults_found_by_full_simulation(design: &Design, controlled: &[bool]) -> Result<Vec<(Fault, bool)>, SimError> {
    let view = View::new(design)?;
    let (faults, _) = view.faults();
    let assigned: Vec<u8> = controlled.iter().map(|b| *b as u8).collect();
    let (mut good, mut bad) = (vec![X; view.flat.signals], vec![X; view.flat.signals]);
    Ok(faults
        .into_iter()
        .map(|fault| {
            view.simulate_one(&assigned, fault, &mut good, &mut bad);
            (fault, view.reads.iter().any(|s| good[*s] != X && bad[*s] != X && good[*s] != bad[*s]))
        })
        .collect())
}

/// The faults on the shift path, which only shifting a pattern through the chain finds.
pub fn chain_faults(design: &Design) -> Result<Vec<Fault>, SimError> {
    Ok(View::new(design)?.faults().1)
}

/// For tests: how many of `faults` some of `batches` × 64 random patterns finds.
#[doc(hidden)]
pub fn found_at_random(design: &Design, faults: &[Fault], batches: usize) -> Result<usize, SimError> {
    let view = View::new(design)?;
    let signals = view.flat.signals;
    let mut scratch = Scratch { bad: vec![UNKNOWN; signals], stamp: vec![0; signals], queued: vec![0; view.flat.gates.len()], heap: BinaryHeap::new(), epoch: 0 };
    let mut rng = Rng(0xD1B5_4A32_D192_ED03);
    let mut left = faults.to_vec();
    for _ in 0..batches {
        let controlled: Vec<Wide> = view.controlled.iter().map(|_| {
            let bits = rng.next();
            Wide { one: bits, zero: !bits }
        }).collect();
        let good = view.simulate(&controlled);
        left.retain(|f| view.detects(&good, *f, &mut scratch) == 0);
    }
    Ok(faults.len() - left.len())
}

/// For tests: the same, by the event-driven fault simulation `generate` uses.
#[doc(hidden)]
pub fn faults_found_by_fault_simulation(design: &Design, controlled: &[bool]) -> Result<Vec<(Fault, bool)>, SimError> {
    let view = View::new(design)?;
    let (faults, _) = view.faults();
    let wide: Vec<Wide> = controlled.iter().map(|b| constant(*b)).collect();
    let good = view.simulate(&wide);
    let signals = view.flat.signals;
    let mut scratch = Scratch { bad: vec![UNKNOWN; signals], stamp: vec![0; signals], queued: vec![0; view.flat.gates.len()], heap: BinaryHeap::new(), epoch: 0 };
    Ok(faults.into_iter().map(|fault| (fault, view.detects(&good, fault, &mut scratch) != 0)).collect())
}
