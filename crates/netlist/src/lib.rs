//! Hierarchical structural netlists built from logic gates, flip-flops and memories.
//! The schematic, the simulator and the Verilog export all read this one structure.

pub mod block;
pub mod chip;
pub mod format;
pub mod image;
pub mod jtag;
pub mod link;
mod self_test;
mod optimize;
pub mod reliability;
pub mod multiplier;
mod cell;
pub mod demo;
pub mod export;
pub mod die;
pub mod rv32;
pub mod sampler;
mod sim;
pub mod soc;
mod verilog;
mod vhdl;

pub use cell::{cell_design, shared_unit_design, CellConfig, MAX_MULTIPLES_BITS};
pub use die::{die_design, package_design, word_bits, ControlWord, DieConfig, DieFeatures, DieProgram, Microprogram, Parameter};
pub use sim::{SimError, Simulator};
pub use verilog::{to_verilog, verilog_testbench, verilog_testbench_with_rom, RomWord, TestStep};
pub use vhdl::{to_vhdl, vhdl_testbench};

use serde::Serialize;
use std::collections::BTreeMap;

pub type NetId = usize;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Bit {
    Net(NetId, u32),
    Zero,
    One,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Dir {
    Input,
    Output,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Prim {
    And2,
    Or2,
    Xor2,
    Not,
    /// `y = s ? b : a`
    Mux2,
    Dff,
    /// One read/write port, read result available the next cycle.
    Sram { addr_bits: u32, width: u32 },
    /// Separate read and write ports; the read result arrives the next cycle.
    Ram2p { addr_bits: u32, width: u32 },
    /// Contents fixed at manufacture; the read result arrives the next cycle.
    Rom { addr_bits: u32, width: u32 },
}

impl Prim {
    pub fn outputs(self) -> &'static [&'static str] {
        match self {
            Prim::Dff => &["q"],
            Prim::Sram { .. } | Prim::Ram2p { .. } | Prim::Rom { .. } => &["rdata"],
            _ => &["y"],
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", content = "of", rename_all = "snake_case")]
pub enum InstKind {
    Prim(Prim),
    Module(String),
}

#[derive(Clone, Debug, Serialize)]
pub struct Instance {
    pub name: String,
    pub kind: InstKind,
    /// The functional area this instance is drawn in.
    pub group: String,
    pub conns: BTreeMap<String, Vec<Bit>>,
}

#[derive(Clone, Debug, Serialize)]
pub struct Net {
    pub name: String,
    pub width: u32,
}

#[derive(Clone, Debug, Serialize)]
pub struct Port {
    pub name: String,
    pub dir: Dir,
    pub net: NetId,
}

#[derive(Clone, Debug, Serialize)]
pub struct Module {
    pub name: String,
    pub title: String,
    pub description: String,
    pub ports: Vec<Port>,
    pub nets: Vec<Net>,
    pub instances: Vec<Instance>,
}

#[derive(Clone, Debug, Serialize)]
pub struct Design {
    pub top: String,
    pub modules: BTreeMap<String, Module>,
}

impl Design {
    /// Number of each primitive once every instance is expanded.
    pub fn flat_counts(&self) -> BTreeMap<String, u64> {
        fn walk(design: &Design, module: &str, times: u64, out: &mut BTreeMap<String, u64>) {
            for inst in &design.modules[module].instances {
                match &inst.kind {
                    InstKind::Prim(p) => {
                        let name = match p {
                            Prim::And2 => "and",
                            Prim::Or2 => "or",
                            Prim::Xor2 => "xor",
                            Prim::Not => "not",
                            Prim::Mux2 => "mux",
                            Prim::Dff => "dff",
                            Prim::Sram { .. } | Prim::Ram2p { .. } => "sram",
                            Prim::Rom { .. } => "rom",
                        };
                        *out.entry(name.into()).or_default() += times;
                    }
                    InstKind::Module(child) => walk(design, child, times, out),
                }
            }
        }
        let mut out = BTreeMap::new();
        walk(self, &self.top, 1, &mut out);
        out
    }

    /// Logic in NAND2 equivalents, memories left out: the usual weights of each gate.
    pub fn gate_equivalents(&self) -> f64 {
        let weight = |name: &str| match name {
            "and" | "or" => 1.33,
            "not" => 0.67,
            "xor" => 2.0,
            "mux" => 2.33,
            "dff" => 4.5,
            _ => 0.0,
        };
        self.flat_counts().iter().map(|(name, count)| weight(name) * *count as f64).sum()
    }
}

/// Builds one module. Generators pass bit vectors around, least significant first.
pub(crate) struct Builder {
    module: Module,
    counter: usize,
    group: String,
    prefix: String,
    /// Names given so far: a name used again gets a number, so every net has its own.
    names: std::collections::HashSet<String>,
}

impl Builder {
    pub fn new(name: &str, title: &str, description: &str) -> Self {
        Self {
            module: Module {
                name: name.into(),
                title: title.into(),
                description: description.into(),
                ports: Vec::new(),
                nets: Vec::new(),
                instances: Vec::new(),
            },
            counter: 0,
            group: String::new(),
            prefix: String::new(),
            names: std::collections::HashSet::new(),
        }
    }

    /// Everything named from now on starts with this, for circuits built several times.
    pub fn set_prefix(&mut self, prefix: &str) {
        self.prefix = prefix.into();
    }

    pub fn group(&mut self, name: &str) {
        self.group = name.into();
    }

    pub fn wire(&mut self, name: &str, width: u32) -> Vec<Bit> {
        let id = self.module.nets.len();
        let mut full = format!("{}{name}", self.prefix);
        if !self.names.insert(full.clone()) {
            full = format!("{full}_{id}");
            self.names.insert(full.clone());
        }
        self.module.nets.push(Net { name: full, width });
        (0..width).map(|bit| Bit::Net(id, bit)).collect()
    }

    fn port(&mut self, name: &str, width: u32, dir: Dir) -> Vec<Bit> {
        let bits = self.wire(name, width);
        self.module.ports.push(Port { name: name.into(), dir, net: self.module.nets.len() - 1 });
        bits
    }

    pub fn input(&mut self, name: &str, width: u32) -> Vec<Bit> {
        self.port(name, width, Dir::Input)
    }

    pub fn output(&mut self, name: &str, width: u32) -> Vec<Bit> {
        self.port(name, width, Dir::Output)
    }

    fn fresh(&mut self, prefix: &str) -> String {
        self.counter += 1;
        format!("{prefix}{}", self.counter)
    }

    fn add(&mut self, name: String, kind: InstKind, conns: &[(&str, Vec<Bit>)]) {
        self.module.instances.push(Instance {
            name: format!("{}{name}", self.prefix),
            kind,
            group: self.group.clone(),
            conns: conns.iter().map(|(p, b)| (p.to_string(), b.clone())).collect(),
        });
    }

    fn gate(&mut self, prim: Prim, prefix: &str, inputs: &[(&str, Bit)], y: Bit) {
        let name = self.fresh(prefix);
        let mut conns: Vec<(&str, Vec<Bit>)> = inputs.iter().map(|(p, b)| (*p, vec![*b])).collect();
        conns.push((prim.outputs()[0], vec![y]));
        self.add(name, InstKind::Prim(prim), &conns);
    }

    pub fn and_to(&mut self, a: Bit, b: Bit, y: Bit) {
        self.gate(Prim::And2, "and", &[("a", a), ("b", b)], y);
    }

    pub fn or_to(&mut self, a: Bit, b: Bit, y: Bit) {
        self.gate(Prim::Or2, "or", &[("a", a), ("b", b)], y);
    }

    pub fn xor_to(&mut self, a: Bit, b: Bit, y: Bit) {
        self.gate(Prim::Xor2, "xor", &[("a", a), ("b", b)], y);
    }

    pub fn not_to(&mut self, a: Bit, y: Bit) {
        self.gate(Prim::Not, "not", &[("a", a)], y);
    }

    pub fn mux_to(&mut self, s: Bit, a: Bit, b: Bit, y: Bit) {
        self.gate(Prim::Mux2, "mux", &[("s", s), ("a", a), ("b", b)], y);
    }

    pub fn dff_to(&mut self, d: Bit, q: Bit) {
        self.gate(Prim::Dff, "dff", &[("d", d)], q);
    }

    pub fn ram2p(&mut self, name: &str, addr_bits: u32, width: u32, conns: &[(&str, Vec<Bit>)]) {
        self.add(name.into(), InstKind::Prim(Prim::Ram2p { addr_bits, width }), conns);
    }

    pub fn sram(&mut self, name: &str, addr_bits: u32, width: u32, conns: &[(&str, Vec<Bit>)]) {
        self.add(name.into(), InstKind::Prim(Prim::Sram { addr_bits, width }), conns);
    }

    pub fn rom(&mut self, name: &str, addr_bits: u32, width: u32, conns: &[(&str, Vec<Bit>)]) {
        self.add(name.into(), InstKind::Prim(Prim::Rom { addr_bits, width }), conns);
    }

    pub fn inst(&mut self, name: &str, module: &str, conns: &[(&str, Vec<Bit>)]) {
        self.add(name.into(), InstKind::Module(module.into()), conns);
    }

    pub fn finish(self) -> Module {
        self.module
    }
}
