//! Two-state, cycle-based simulation of a design, one gate at a time. Only the gates whose
//! inputs changed are evaluated again, in their topological order: in a cycle most of a
//! chip holds still.

use crate::{Bit, Design, Dir, InstKind, Prim};
use std::collections::{BTreeMap, HashMap};

/// Flat signal 0 is always low and flat signal 1 is always high.
const ZERO: usize = 0;
const ONE: usize = 1;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SimError(pub String);

impl std::fmt::Display for SimError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for SimError {}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Op {
    And,
    Or,
    Xor,
    Not,
    Mux,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Gate {
    pub(crate) op: Op,
    pub(crate) a: usize,
    pub(crate) b: usize,
    /// A multiplexer's select.
    pub(crate) c: usize,
    pub(crate) y: usize,
}

impl Gate {
    /// The signals it reads: `[a]`, `[a, b]` or `[a, b, c]`.
    pub(crate) fn inputs(&self) -> &'static [usize] {
        match self.op {
            Op::Not => &[0],
            Op::Mux => &[0, 1, 2],
            _ => &[0, 1],
        }
    }

    pub(crate) fn input(&self, pin: usize) -> usize {
        [self.a, self.b, self.c][pin]
    }
}

/// A design as one level of gates in their order, its flip-flops `(d, q)`, the signals its
/// memories drive, and the signals of its top ports. Signal 0 is low and 1 is high.
pub(crate) struct Flat {
    pub(crate) signals: usize,
    pub(crate) gates: Vec<Gate>,
    pub(crate) flops: Vec<(usize, usize)>,
    pub(crate) memory_outputs: Vec<usize>,
    pub(crate) inputs: Vec<(String, Vec<usize>)>,
    pub(crate) outputs: Vec<(String, Vec<usize>)>,
    /// Each signal's name, when asked for.
    pub(crate) names: Vec<String>,
}

pub(crate) fn flatten(design: &Design, named: bool) -> Result<Flat, SimError> {
    let top = &design.modules[&design.top];
    let mut e = Elaborator { design, named, names: Vec::new(), next: 2, gates: Vec::new(), flops: Vec::new(), memories: Vec::new(), paths: HashMap::new() };
    let mut ports = HashMap::new();
    let (mut inputs, mut outputs) = (Vec::new(), Vec::new());
    for port in &top.ports {
        let bits: Vec<usize> = (0..top.nets[port.net].width).map(|_| e.fresh()).collect();
        ports.insert(port.name.clone(), bits.clone());
        match port.dir {
            Dir::Input => inputs.push((port.name.clone(), bits)),
            Dir::Output => outputs.push((port.name.clone(), bits)),
        }
    }
    e.elaborate(&design.top, "", &ports);
    let signals = e.next;
    let gates = levelize(e.gates, signals, &e.names)?;
    let memory_outputs = e.memories.iter().flat_map(|m| m.rdata.iter().copied()).collect();
    Ok(Flat { signals, gates, flops: e.flops, memory_outputs, inputs, outputs, names: e.names })
}

struct Memory {
    path: String,
    /// Differs from `addr` only for memories with a separate write port.
    write_addr: Vec<usize>,
    addr: Vec<usize>,
    we: usize,
    wdata: Vec<usize>,
    rdata: Vec<usize>,
    cells: Vec<u128>,
    /// `(address, bit, value)` of every defective bit, which always read as `value`.
    stuck: Vec<(usize, u32, bool)>,
}

/// A net's bits, found once by name: see `Simulator::probe`.
#[derive(Clone, Debug)]
pub struct Probe(Vec<usize>);

pub struct Simulator {
    values: Vec<u8>,
    gates: Vec<Gate>,
    flops: Vec<(usize, usize)>,
    memories: Vec<Memory>,
    ports: HashMap<String, Vec<usize>>,
    paths: HashMap<String, BTreeMap<String, Vec<usize>>>,
    cycle: u64,
    /// Every signal's name, for `critical_trace`, when built with `with_names`.
    names: Vec<String>,
    /// The gates each signal drives, by index into `gates`: those of signal `s` are
    /// `fanout[fanout_start[s]..fanout_start[s + 1]]`.
    fanout_start: Vec<u32>,
    fanout: Vec<u32>,
    /// Gates to evaluate again, one bit each, by index.
    pending: Vec<u64>,
    /// Signals held at 0 or 1 whatever drives them, a stuck-at fault each; 2 for the rest.
    /// Empty while there are none, which is almost always.
    stuck: Vec<u8>,
}

struct Elaborator<'a> {
    design: &'a Design,
    named: bool,
    names: Vec<String>,
    next: usize,
    gates: Vec<Gate>,
    flops: Vec<(usize, usize)>,
    memories: Vec<Memory>,
    paths: HashMap<String, BTreeMap<String, Vec<usize>>>,
}

impl Elaborator<'_> {
    fn fresh(&mut self) -> usize {
        self.next += 1;
        self.next - 1
    }

    fn elaborate(&mut self, module: &str, path: &str, bindings: &HashMap<String, Vec<usize>>) {
        let design = self.design;
        let m = &design.modules[module];
        let mut nets: Vec<Vec<usize>> = Vec::with_capacity(m.nets.len());
        for net in &m.nets {
            let bound = m.ports.iter().find(|p| m.nets[p.net].name == net.name).and_then(|p| bindings.get(&p.name));
            nets.push(match bound {
                Some(bits) => bits.clone(),
                None => (0..net.width).map(|_| self.fresh()).collect(),
            });
        }
        if self.named {
            for (id, (net, bits)) in m.nets.iter().zip(&nets).enumerate() {
                for (bit, index) in bits.iter().enumerate() {
                    if self.names.len() <= *index {
                        self.names.resize(*index + 1, String::new());
                    }
                    if self.names[*index].is_empty() {
                        self.names[*index] = format!("{path}/{}#{id}[{bit}]", net.name);
                    }
                }
            }
        }
        self.paths.insert(
            path.to_string(),
            m.nets.iter().zip(&nets).map(|(n, bits)| (n.name.clone(), bits.clone())).collect(),
        );

        let resolve = |bits: &[Bit]| -> Vec<usize> {
            bits.iter()
                .map(|b| match b {
                    Bit::Zero => ZERO,
                    Bit::One => ONE,
                    Bit::Net(id, bit) => nets[*id][*bit as usize],
                })
                .collect()
        };

        for inst in &m.instances {
            let conns: HashMap<String, Vec<usize>> = inst.conns.iter().map(|(k, v)| (k.clone(), resolve(v))).collect();
            let one = |port: &str| conns[port][0];
            match &inst.kind {
                InstKind::Module(child) => {
                    let child_path = format!("{path}/{}", inst.name);
                    self.elaborate(child, &child_path, &conns);
                }
                InstKind::Prim(Prim::And2) => self.gates.push(Gate { op: Op::And, a: one("a"), b: one("b"), c: 0, y: one("y") }),
                InstKind::Prim(Prim::Or2) => self.gates.push(Gate { op: Op::Or, a: one("a"), b: one("b"), c: 0, y: one("y") }),
                InstKind::Prim(Prim::Xor2) => self.gates.push(Gate { op: Op::Xor, a: one("a"), b: one("b"), c: 0, y: one("y") }),
                InstKind::Prim(Prim::Not) => self.gates.push(Gate { op: Op::Not, a: one("a"), b: 0, c: 0, y: one("y") }),
                InstKind::Prim(Prim::Mux2) => self.gates.push(Gate { op: Op::Mux, a: one("a"), b: one("b"), c: one("s"), y: one("y") }),
                InstKind::Prim(Prim::Dff) => self.flops.push((one("d"), one("q"))),
                InstKind::Prim(Prim::Ram2p { addr_bits, .. }) => self.memories.push(Memory {
                    path: format!("{path}/{}", inst.name),
                    write_addr: conns["waddr"].clone(),
                    addr: conns["raddr"].clone(),
                    we: one("we"),
                    wdata: conns["wdata"].clone(),
                    rdata: conns["rdata"].clone(),
                    cells: vec![0; 1usize << addr_bits],
                    stuck: Vec::new(),
                }),
                InstKind::Prim(Prim::Rom { addr_bits, .. }) => self.memories.push(Memory {
                    path: format!("{path}/{}", inst.name),
                    write_addr: conns["addr"].clone(),
                    addr: conns["addr"].clone(),
                    we: ZERO,
                    wdata: Vec::new(),
                    rdata: conns["rdata"].clone(),
                    cells: vec![0; 1usize << addr_bits],
                    stuck: Vec::new(),
                }),
                InstKind::Prim(Prim::Sram { addr_bits, .. }) => self.memories.push(Memory {
                    path: format!("{path}/{}", inst.name),
                    write_addr: conns["addr"].clone(),
                    addr: conns["addr"].clone(),
                    we: one("we"),
                    wdata: conns["wdata"].clone(),
                    rdata: conns["rdata"].clone(),
                    cells: vec![0; 1usize << addr_bits],
                    stuck: Vec::new(),
                }),
            }
        }
    }
}

/// The gates each signal drives, as offsets into one list.
fn fanout_of(gates: &[Gate], signals: usize) -> (Vec<u32>, Vec<u32>) {
    let inputs = |g: &Gate| -> [usize; 3] {
        match g.op {
            Op::Not => [g.a, ZERO, ZERO],
            Op::Mux => [g.a, g.b, g.c],
            _ => [g.a, g.b, ZERO],
        }
    };
    let mut count = vec![0u32; signals + 1];
    for g in gates {
        for s in inputs(g) {
            if s > ONE {
                count[s + 1] += 1;
            }
        }
    }
    for s in 0..signals {
        count[s + 1] += count[s];
    }
    let mut next = count.clone();
    let mut fanout = vec![0u32; count[signals] as usize];
    for (i, g) in gates.iter().enumerate() {
        for s in inputs(g) {
            if s > ONE {
                fanout[next[s] as usize] = i as u32;
                next[s] += 1;
            }
        }
    }
    (count, fanout)
}

/// Orders gates so each one runs after everything that drives it.
fn levelize(gates: Vec<Gate>, signals: usize, names: &[String]) -> Result<Vec<Gate>, SimError> {
    let mut driver = vec![usize::MAX; signals];
    for (i, g) in gates.iter().enumerate() {
        if driver[g.y] != usize::MAX {
            return Err(SimError(format!("signal {} has two drivers", g.y)));
        }
        driver[g.y] = i;
    }
    let inputs = |g: &Gate| -> [usize; 3] {
        match g.op {
            Op::Not => [g.a, ZERO, ZERO],
            Op::Mux => [g.a, g.b, g.c],
            _ => [g.a, g.b, ZERO],
        }
    };
    let mut pending = vec![0u8; gates.len()];
    let mut fanout: Vec<Vec<usize>> = vec![Vec::new(); gates.len()];
    for (i, g) in gates.iter().enumerate() {
        for s in inputs(g) {
            if s > ONE && driver[s] != usize::MAX {
                pending[i] += 1;
                fanout[driver[s]].push(i);
            }
        }
    }
    let mut ready: Vec<usize> = (0..gates.len()).filter(|i| pending[*i] == 0).collect();
    let mut order = Vec::with_capacity(gates.len());
    while let Some(i) = ready.pop() {
        order.push(gates[i]);
        for &next in &fanout[i] {
            pending[next] -= 1;
            if pending[next] == 0 {
                ready.push(next);
            }
        }
    }
    if order.len() != gates.len() {
        // Walk back from a gate left over through inputs also left over until one repeats:
        // that is the loop. With names, it says which signals it goes through.
        let mut walk = vec![(0..gates.len()).find(|i| pending[*i] > 0).unwrap_or(0)];
        let cycle = loop {
            let g = &gates[*walk.last().unwrap()];
            let Some(next) = inputs(g).into_iter().filter(|s| *s > ONE && driver[*s] != usize::MAX).map(|s| driver[s]).find(|d| pending[*d] > 0) else { break Vec::new() };
            if let Some(at) = walk.iter().position(|w| *w == next) {
                break walk[at..].to_vec();
            }
            walk.push(next);
        };
        let through: Vec<String> = cycle.iter().take(12).map(|i| names.get(gates[*i].y).cloned().unwrap_or_else(|| format!("#{}", gates[*i].y))).collect();
        return Err(SimError(format!("the circuit has a combinational loop: {}", through.join(" ← "))));
    }
    Ok(order)
}

fn read(values: &[u8], bits: &[usize]) -> u128 {
    bits.iter().enumerate().fold(0, |acc, (i, s)| acc | ((values[*s] as u128) << i))
}

impl Simulator {
    pub fn new(design: &Design) -> Result<Self, SimError> {
        Self::build(design, false)
    }

    /// Like `new`, keeping every signal's name for `critical_trace`.
    pub fn with_names(design: &Design) -> Result<Self, SimError> {
        Self::build(design, true)
    }

    fn build(design: &Design, named: bool) -> Result<Self, SimError> {
        let top = &design.modules[&design.top];
        let mut e = Elaborator {
            design,
            named,
            names: Vec::new(),
            next: 2,
            gates: Vec::new(),
            flops: Vec::new(),
            memories: Vec::new(),
            paths: HashMap::new(),
        };
        let mut ports = HashMap::new();
        for port in &top.ports {
            let bits: Vec<usize> = (0..top.nets[port.net].width).map(|_| e.fresh()).collect();
            ports.insert(port.name.clone(), bits);
        }
        e.elaborate(&design.top, "", &ports);

        let signals = e.next;
        let gates = levelize(e.gates, signals, &e.names)?;
        let mut values = vec![0u8; signals];
        values[ONE] = 1;
        let (fanout_start, fanout) = fanout_of(&gates, signals);
        // Everything is evaluated the first time.
        let mut pending = vec![u64::MAX; gates.len().div_ceil(64)];
        if !gates.len().is_multiple_of(64) {
            *pending.last_mut().unwrap() = (1u64 << (gates.len() % 64)) - 1;
        }
        let mut sim = Self {
            values,
            gates,
            flops: e.flops,
            memories: e.memories,
            ports,
            paths: e.paths,
            cycle: 0,
            names: e.names,
            fanout_start,
            fanout,
            pending,
            stuck: Vec::new(),
        };
        for port in top.ports.iter().filter(|p| p.dir == Dir::Input) {
            sim.set(&port.name, 0)?;
        }
        sim.settle();
        Ok(sim)
    }

    /// The number of bit `bit` of net `net` in the module instance at `path` (`""` for the
    /// top), as `stick_signal` takes it.
    pub fn signal(&self, path: &str, net: &str, bit: usize) -> Option<usize> {
        self.paths.get(path)?.get(net)?.get(bit).copied()
    }

    /// Holds signal `s`, numbered as `atpg` numbers them, at `value` from now on: a stuck-at
    /// fault on the wire, to check that the test patterns find it.
    #[doc(hidden)]
    pub fn stick_signal(&mut self, s: usize, value: bool) {
        if self.stuck.is_empty() {
            self.stuck = vec![2; self.values.len()];
        }
        self.stuck[s] = value as u8;
        self.values[s] = value as u8;
        self.mark_fanout(s);
        self.settle();
    }

    /// Writes directly into a memory, as a testbench preloads one.
    pub fn poke_memory(&mut self, path: &str, address: usize, value: u128) -> Result<(), SimError> {
        let memory = self.memories.iter_mut().find(|m| m.path == path).ok_or_else(|| SimError(format!("unknown memory: {path}")))?;
        let slot = memory.cells.get_mut(address).ok_or_else(|| SimError(format!("address outside {path}: {address}")))?;
        *slot = value;
        Ok(())
    }

    /// Inverts one stored bit, as a particle strike would.
    pub fn flip_memory_bit(&mut self, path: &str, address: usize, bit: u32) -> Result<(), SimError> {
        let value = self.peek_memory(path, address).ok_or_else(|| SimError(format!("address outside {path}: {address}")))?;
        self.poke_memory(path, address, value ^ (1 << bit))
    }

    /// Makes one bit defective: from now on it reads as `value`, whatever is written.
    pub fn stick_memory_bit(&mut self, path: &str, address: usize, bit: u32, value: bool) -> Result<(), SimError> {
        let memory = self.memories.iter_mut().find(|m| m.path == path).ok_or_else(|| SimError(format!("unknown memory: {path}")))?;
        memory.stuck.push((address, bit, value));
        Ok(())
    }

    /// The paths of every memory, for `peek_memory`.
    pub fn memory_paths(&self) -> Vec<String> {
        self.memories.iter().map(|m| m.path.clone()).collect()
    }

    pub fn peek_memory(&self, path: &str, address: usize) -> Option<u128> {
        self.memories.iter().find(|m| m.path == path).and_then(|m| m.cells.get(address).copied())
    }

    pub fn set(&mut self, port: &str, value: u64) -> Result<(), SimError> {
        self.set_wide(port, value as u128)
    }

    pub fn set_wide(&mut self, port: &str, value: u128) -> Result<(), SimError> {
        let bits = self.ports.get(port).ok_or_else(|| SimError(format!("puerto desconocido: {port}")))?.clone();
        for (i, s) in bits.iter().enumerate() {
            self.drive(*s, (value.checked_shr(i as u32).unwrap_or(0) & 1) as u8);
        }
        Ok(())
    }

    /// Gives signal `s` a value, and if it changes, marks the gates it drives.
    fn drive(&mut self, s: usize, value: u8) {
        let value = match self.stuck.get(s) {
            Some(held) if *held < 2 => *held,
            _ => value,
        };
        if self.values[s] != value {
            self.values[s] = value;
            self.mark_fanout(s);
        }
    }

    fn mark_fanout(&mut self, s: usize) {
        for &g in &self.fanout[self.fanout_start[s] as usize..self.fanout_start[s + 1] as usize] {
            self.pending[g as usize / 64] |= 1 << (g % 64);
        }
    }

    /// Sets a port of any width from 32-bit words, least significant first.
    pub fn set_words(&mut self, port: &str, words: &[u32]) -> Result<(), SimError> {
        let bits = self.ports.get(port).ok_or_else(|| SimError(format!("puerto desconocido: {port}")))?.clone();
        for (i, s) in bits.iter().enumerate() {
            let word = words.get(i / 32).copied().unwrap_or(0);
            self.drive(*s, (word >> (i % 32) & 1) as u8);
        }
        Ok(())
    }

    /// Propagates the current inputs through the combinational logic: the gates marked,
    /// in their order, and those their changes reach.
    pub fn settle(&mut self) {
        for word in 0..self.pending.len() {
            loop {
                let bits = self.pending[word];
                if bits == 0 {
                    break;
                }
                let bit = bits.trailing_zeros() as usize;
                self.pending[word] &= !(1 << bit);
                let g = self.gates[word * 64 + bit];
                let v = &self.values;
                let value = match g.op {
                    Op::And => v[g.a] & v[g.b],
                    Op::Or => v[g.a] | v[g.b],
                    Op::Xor => v[g.a] ^ v[g.b],
                    Op::Not => 1 ^ v[g.a],
                    Op::Mux => {
                        if v[g.c] == 1 {
                            v[g.b]
                        } else {
                            v[g.a]
                        }
                    }
                };
                // What it drives comes later in the order, so this one pass reaches it.
                self.drive(g.y, value);
            }
        }
    }

    /// One rising clock edge, followed by settling.
    pub fn tick(&mut self) {
        self.settle();
        let captured: Vec<u8> = self.flops.iter().map(|(d, _)| self.values[*d]).collect();
        let mut reads: Vec<u128> = Vec::with_capacity(self.memories.len());
        for m in &mut self.memories {
            let addr = read(&self.values, &m.addr) as usize & (m.cells.len() - 1);
            let mut word = m.cells[addr];
            for (at, bit, value) in &m.stuck {
                if *at == addr {
                    word = if *value { word | 1 << bit } else { word & !(1 << bit) };
                }
            }
            reads.push(word);
            if self.values[m.we] == 1 {
                let target = read(&self.values, &m.write_addr) as usize & (m.cells.len() - 1);
                m.cells[target] = read(&self.values, &m.wdata);
            }
        }
        for (k, value) in captured.into_iter().enumerate() {
            let q = self.flops[k].1;
            self.drive(q, value);
        }
        for (k, value) in reads.into_iter().enumerate() {
            for i in 0..self.memories[k].rdata.len() {
                let s = self.memories[k].rdata[i];
                self.drive(s, ((value >> i) & 1) as u8);
            }
        }
        self.cycle += 1;
        self.settle();
    }

    pub fn get(&self, port: &str) -> Option<u64> {
        self.ports.get(port).map(|bits| read(&self.values, bits) as u64)
    }

    /// Runs `cycles` with a stimulus and counts how often each gate kind's output
    /// changes value: AND, OR, XOR, NOT, MUX, then flip-flop.
    pub fn switching_activity(&mut self, cycles: u32, mut stimulus: impl FnMut(&mut Simulator, u32)) -> [u64; 6] {
        let mut counts = [0u64; 6];
        let mut last: Vec<u8> = self.values.clone();
        for cycle in 0..cycles {
            stimulus(self, cycle);
            self.tick();
            for g in &self.gates {
                if self.values[g.y] != last[g.y] {
                    counts[match g.op {
                        Op::And => 0,
                        Op::Or => 1,
                        Op::Xor => 2,
                        Op::Not => 3,
                        Op::Mux => 4,
                    }] += 1;
                }
            }
            counts[5] += self.flops.iter().filter(|(_, q)| self.values[*q] != last[*q]).count() as u64;
            last.clone_from(&self.values);
        }
        counts
    }

    /// The slowest combinational path, given the delay of an AND, OR, XOR, NOT and MUX.
    pub fn critical_path(&self, delays: [f64; 5]) -> f64 {
        let mut arrival = vec![0f64; self.values.len()];
        let mut worst = 0f64;
        for g in &self.gates {
            let (delay, input) = match g.op {
                Op::And => (delays[0], arrival[g.a].max(arrival[g.b])),
                Op::Or => (delays[1], arrival[g.a].max(arrival[g.b])),
                Op::Xor => (delays[2], arrival[g.a].max(arrival[g.b])),
                Op::Not => (delays[3], arrival[g.a]),
                Op::Mux => (delays[4], arrival[g.a].max(arrival[g.b]).max(arrival[g.c])),
            };
            arrival[g.y] = input + delay;
            worst = worst.max(arrival[g.y]);
        }
        worst
    }

    /// The nets along the slowest combinational path, from where it starts to where it
    /// ends, each with its arrival time: to see which structure sets the clock.
    pub fn critical_trace(&self, delays: [f64; 5]) -> Vec<(String, f64)> {
        self.trace_to(delays, |_| true)
    }

    /// The slowest path into a net whose name `ends` accepts, as `critical_trace` gives it.
    pub fn trace_to(&self, delays: [f64; 5], ends: impl Fn(&str) -> bool) -> Vec<(String, f64)> {
        let mut arrival = vec![0f64; self.values.len()];
        let mut from = vec![usize::MAX; self.values.len()];
        let (mut worst, mut end) = (0f64, 0);
        for g in &self.gates {
            let inputs: &[usize] = match g.op {
                Op::Not => &[g.a],
                Op::Mux => &[g.a, g.b, g.c],
                _ => &[g.a, g.b],
            };
            let delay = match g.op { Op::And => delays[0], Op::Or => delays[1], Op::Xor => delays[2], Op::Not => delays[3], Op::Mux => delays[4] };
            let latest = *inputs.iter().max_by(|x, y| arrival[**x].total_cmp(&arrival[**y])).expect("gate without inputs");
            arrival[g.y] = arrival[latest] + delay;
            from[g.y] = latest;
            if arrival[g.y] > worst && ends(self.names.get(g.y).map_or("", String::as_str)) {
                (worst, end) = (arrival[g.y], g.y);
            }
        }
        let mut trace = Vec::new();
        let mut at = end;
        while at != usize::MAX {
            let name = self.names.get(at).filter(|n| !n.is_empty()).cloned().unwrap_or_else(|| format!("#{at}"));
            trace.push((name, arrival[at]));
            at = from[at];
        }
        trace.reverse();
        trace
    }

    /// A port of any width, in 32-bit words starting from the least significant.
    pub fn get_words(&self, port: &str) -> Option<Vec<u32>> {
        self.ports.get(port).map(|bits| bits.chunks(32).map(|chunk| read(&self.values, chunk) as u32).collect())
    }

    pub fn get_wide(&self, port: &str) -> Option<u128> {
        self.ports.get(port).map(|bits| read(&self.values, bits))
    }

    pub fn cycle(&self) -> u64 {
        self.cycle
    }

    /// The bits of net `name` of the module instance at `path` ("" for the top), to read
    /// it every cycle with `read` without the cost of a `snapshot`.
    pub fn probe(&self, path: &str, name: &str) -> Option<Probe> {
        self.paths.get(path).and_then(|nets| nets.get(name)).map(|bits| Probe(bits.clone()))
    }

    /// The value of a probed net, least significant bit first, up to 128 bits.
    pub fn read(&self, probe: &Probe) -> u128 {
        read(&self.values, &probe.0)
    }

    /// Every net of the module instance at `path`, as bit strings with the MSB first.
    pub fn snapshot(&self, path: &str) -> Option<BTreeMap<String, String>> {
        self.paths.get(path).map(|nets| {
            nets.iter()
                .map(|(name, bits)| {
                    (name.clone(), bits.iter().rev().map(|s| if self.values[*s] == 1 { '1' } else { '0' }).collect())
                })
                .collect()
        })
    }
}
