//! Two-state, cycle-based simulation of a design, one gate at a time.

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

#[derive(Clone, Copy)]
enum Op {
    And,
    Or,
    Xor,
    Not,
    Mux,
}

#[derive(Clone, Copy)]
struct Gate {
    op: Op,
    a: usize,
    b: usize,
    c: usize,
    y: usize,
}

struct Memory {
    /// Differs from `addr` only for memories with a separate write port.
    write_addr: Vec<usize>,
    addr: Vec<usize>,
    we: usize,
    wdata: Vec<usize>,
    rdata: Vec<usize>,
    cells: Vec<u128>,
}

pub struct Simulator {
    values: Vec<u8>,
    gates: Vec<Gate>,
    flops: Vec<(usize, usize)>,
    memories: Vec<Memory>,
    ports: HashMap<String, Vec<usize>>,
    paths: HashMap<String, BTreeMap<String, Vec<usize>>>,
    cycle: u64,
}

struct Elaborator<'a> {
    design: &'a Design,
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
                    write_addr: conns["waddr"].clone(),
                    addr: conns["raddr"].clone(),
                    we: one("we"),
                    wdata: conns["wdata"].clone(),
                    rdata: conns["rdata"].clone(),
                    cells: vec![0; 1usize << addr_bits],
                }),
                InstKind::Prim(Prim::Sram { addr_bits, .. }) => self.memories.push(Memory {
                    write_addr: conns["addr"].clone(),
                    addr: conns["addr"].clone(),
                    we: one("we"),
                    wdata: conns["wdata"].clone(),
                    rdata: conns["rdata"].clone(),
                    cells: vec![0; 1usize << addr_bits],
                }),
            }
        }
    }
}

/// Orders gates so each one runs after everything that drives it.
fn levelize(gates: Vec<Gate>, signals: usize) -> Result<Vec<Gate>, SimError> {
    let mut driver = vec![usize::MAX; signals];
    for (i, g) in gates.iter().enumerate() {
        if driver[g.y] != usize::MAX {
            return Err(SimError(format!("la señal {} tiene dos fuentes", g.y)));
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
        return Err(SimError("el circuito tiene un bucle combinacional".into()));
    }
    Ok(order)
}

fn read(values: &[u8], bits: &[usize]) -> u128 {
    bits.iter().enumerate().fold(0, |acc, (i, s)| acc | ((values[*s] as u128) << i))
}

impl Simulator {
    pub fn new(design: &Design) -> Result<Self, SimError> {
        let top = &design.modules[&design.top];
        let mut e = Elaborator {
            design,
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
        let gates = levelize(e.gates, signals)?;
        let mut values = vec![0u8; signals];
        values[ONE] = 1;
        let mut sim = Self {
            values,
            gates,
            flops: e.flops,
            memories: e.memories,
            ports,
            paths: e.paths,
            cycle: 0,
        };
        for port in top.ports.iter().filter(|p| p.dir == Dir::Input) {
            sim.set(&port.name, 0)?;
        }
        sim.settle();
        Ok(sim)
    }

    pub fn set(&mut self, port: &str, value: u64) -> Result<(), SimError> {
        self.set_wide(port, value as u128)
    }

    pub fn set_wide(&mut self, port: &str, value: u128) -> Result<(), SimError> {
        let bits = self.ports.get(port).ok_or_else(|| SimError(format!("puerto desconocido: {port}")))?;
        for (i, s) in bits.iter().enumerate() {
            self.values[*s] = ((value >> i) & 1) as u8;
        }
        Ok(())
    }

    /// Propagates the current inputs through the combinational logic.
    pub fn settle(&mut self) {
        let v = &mut self.values;
        for g in &self.gates {
            v[g.y] = match g.op {
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
        }
    }

    /// One rising clock edge, followed by settling.
    pub fn tick(&mut self) {
        self.settle();
        let captured: Vec<u8> = self.flops.iter().map(|(d, _)| self.values[*d]).collect();
        let mut reads: Vec<u128> = Vec::with_capacity(self.memories.len());
        for m in &mut self.memories {
            let addr = read(&self.values, &m.addr) as usize & (m.cells.len() - 1);
            reads.push(m.cells[addr]);
            if self.values[m.we] == 1 {
                let target = read(&self.values, &m.write_addr) as usize & (m.cells.len() - 1);
                m.cells[target] = read(&self.values, &m.wdata);
            }
        }
        for ((_, q), value) in self.flops.iter().zip(captured) {
            self.values[*q] = value;
        }
        for (m, value) in self.memories.iter().zip(reads) {
            for (i, s) in m.rdata.iter().enumerate() {
                self.values[*s] = ((value >> i) & 1) as u8;
            }
        }
        self.cycle += 1;
        self.settle();
    }

    pub fn get(&self, port: &str) -> Option<u64> {
        self.ports.get(port).map(|bits| read(&self.values, bits) as u64)
    }

    pub fn cycle(&self) -> u64 {
        self.cycle
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
