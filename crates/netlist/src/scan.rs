//! Scan chains for the factory test: every flip-flop of a design, and a shadow flip-flop per
//! bit read from each memory, joined into one shift register that goes from `scan_in` to
//! `scan_out`.
//!
//! - With `scan_enable` high, each clock shifts the chain one place, so a tester can load
//!   any state and read back whatever the chip holds.
//! - With it low, the flip-flops take their normal inputs: one clock captures what the logic
//!   computes from the loaded state.
//! - `test_mode` stays high for the whole test. It puts each memory behind its shadow: the
//!   logic reads the shadow flip-flops instead of the memory, and those capture a mix of the
//!   address, data and write signals going into it. The memory contents themselves are
//!   tested by the march test (`self_test`).
//!
//! With both low the chip works as before: the only difference is a multiplexer in front of
//! each flip-flop and after each memory output.

use crate::cell::Gen;
use crate::{Bit, Builder, Design, InstKind, Prim};
use std::collections::{HashMap, HashSet};

pub const SCAN_ENABLE: &str = "scan_enable";
pub const TEST_MODE: &str = "test_mode";
pub const SCAN_IN: &str = "scan_in";
pub const SCAN_OUT: &str = "scan_out";

/// A design with its scan chain, and how many flip-flops the chain goes through.
pub struct ScanDesign {
    pub design: Design,
    pub length: usize,
}

/// The input pins of a memory that its shadow flip-flops capture.
fn memory_inputs(prim: Prim) -> &'static [&'static str] {
    match prim {
        Prim::Sram { .. } => &["addr", "we", "wdata"],
        Prim::Ram2p { .. } => &["waddr", "raddr", "we", "wdata"],
        Prim::Rom { .. } => &["addr"],
        _ => &[],
    }
}

/// Threads the chain through module `name` and the modules under it; gives its length.
/// The instances in `skip` stay out of the chain: they get chains of their own.
fn visit(gen: &mut Gen, name: &str, lengths: &mut HashMap<String, usize>, skip: &HashSet<usize>) -> usize {
    if let Some(known) = lengths.get(name) {
        return *known;
    }
    let module = gen.modules[name].clone();
    let children: Vec<(usize, usize)> = module
        .instances
        .iter()
        .enumerate()
        .filter(|(i, _)| !skip.contains(i))
        .filter_map(|(i, inst)| match &inst.kind {
            InstKind::Module(child) => Some((i, visit(gen, child, lengths, &HashSet::new()))),
            _ => None,
        })
        .filter(|(_, length)| *length > 0)
        .collect();
    let flops: Vec<usize> = module.instances.iter().enumerate().filter(|(_, i)| matches!(i.kind, InstKind::Prim(Prim::Dff))).map(|(i, _)| i).collect();
    let memories: Vec<(usize, Prim, u32)> = module
        .instances
        .iter()
        .enumerate()
        .filter_map(|(i, inst)| match inst.kind {
            InstKind::Prim(prim @ (Prim::Sram { width, .. } | Prim::Ram2p { width, .. } | Prim::Rom { width, .. })) => Some((i, prim, width)),
            _ => None,
        })
        .collect();
    let length = flops.len() + memories.iter().map(|(_, _, width)| *width as usize).sum::<usize>() + children.iter().map(|(_, length)| length).sum::<usize>();
    lengths.insert(name.to_string(), length);
    if length == 0 {
        return 0;
    }

    let names = module.nets.iter().map(|net| net.name.clone()).collect();
    let mut b = Builder { prefix: String::new(), module, counter: 2_000_000, group: "Scan chain".into(), names };
    let enable = b.input(SCAN_ENABLE, 1)[0];
    // A module made testable already reads `test_mode`.
    let test_mode = match b.module.ports.iter().find(|p| p.name == TEST_MODE) {
        Some(port) => Bit::Net(port.net, 0),
        None => b.input(TEST_MODE, 1)[0],
    };
    let mut previous = b.input(SCAN_IN, 1)[0];
    let scan_out = b.output(SCAN_OUT, 1)[0];

    // Each flip-flop takes the one before it while shifting.
    for index in flops {
        let d = b.module.instances[index].conns["d"][0];
        let q = b.module.instances[index].conns["q"][0];
        let chosen = b.wire("scan_d", 1)[0];
        b.mux_to(enable, d, previous, chosen);
        b.module.instances[index].conns.insert("d".into(), vec![chosen]);
        previous = q;
    }

    // Each bit read from a memory gets a shadow flip-flop: in test mode the logic reads the
    // shadow, which captures the memory's inputs folded onto its width.
    for (index, prim, width) in memories {
        let inputs: Vec<Bit> = memory_inputs(prim).iter().flat_map(|pin| b.module.instances[index].conns[*pin].clone()).collect();
        let rdata = b.module.instances[index].conns["rdata"].clone();
        let raw = b.wire("scan_memory_raw", width);
        b.module.instances[index].conns.insert("rdata".into(), raw.clone());
        for bit in 0..width as usize {
            let mut folded = Bit::Zero;
            for input in inputs.iter().skip(bit).step_by(width as usize) {
                folded = if folded == Bit::Zero {
                    *input
                } else {
                    let y = b.wire("scan_fold", 1)[0];
                    b.xor_to(folded, *input, y);
                    y
                };
            }
            let (d, shadow) = (b.wire("scan_shadow_d", 1)[0], b.wire("scan_shadow", 1)[0]);
            b.mux_to(enable, folded, previous, d);
            b.dff_to(d, shadow);
            b.mux_to(test_mode, raw[bit], shadow, rdata[bit]);
            previous = shadow;
        }
    }

    for (index, _) in children {
        let out = b.wire("scan_child_out", 1)[0];
        let conns = &mut b.module.instances[index].conns;
        conns.insert(SCAN_ENABLE.into(), vec![enable]);
        conns.insert(TEST_MODE.into(), vec![test_mode]);
        conns.insert(SCAN_IN.into(), vec![previous]);
        conns.insert(SCAN_OUT.into(), vec![out]);
        previous = out;
    }
    b.or_to(previous, Bit::Zero, scan_out);
    gen.modules.insert(name.to_string(), b.module);
    length
}

/// Adds the scan chain, with its four pins on the top module.
pub fn insert_scan(design: Design) -> ScanDesign {
    let Design { top, modules } = design;
    let mut gen = Gen { modules };
    let length = visit(&mut gen, &top, &mut HashMap::new(), &HashSet::new());
    ScanDesign { design: Design { top, modules: gen.modules }, length }
}

pub const SCAN_CELLS_IN: &str = "scan_cells_in";
pub const SCAN_CELLS_OUT: &str = "scan_cells_out";
pub const SCAN_CELLS_DIFFER: &str = "scan_cells_differ";
/// The inputs a testable module reads in test mode instead of its own.
pub const TEST_IN: &str = "test_in";

/// A design whose identical modules, the compute cells, are tested side by side.
pub struct ParallelScan {
    pub design: Design,
    /// Flip-flops of the chain through the rest of the design.
    pub length: usize,
    /// Flip-flops of each cell's chain, its input wrapper included.
    pub cell_length: usize,
    /// Flip-flops of each group's input wrapper: as many as the cell has input bits.
    pub wrapper_width: usize,
    /// One cell with its input wrapper, alone, chained as each cell is in the design: the
    /// test patterns made for it go to every cell at once.
    pub slice: Design,
}

/// A copy of module `name` whose inputs, in test mode, come from `test_in` instead: the
/// logic then sees what the test shifted in, whatever drives the module. Gives the copy's
/// name and the width of `test_in`.
fn testable_module(gen: &mut Gen, name: &str) -> (String, u32) {
    let testable = format!("{name}_comprobable");
    let mut module = gen.modules[name].clone();
    let inputs: Vec<(usize, u32)> = module.ports.iter().filter(|p| p.dir == crate::Dir::Input).map(|p| (p.net, module.nets[p.net].width)).collect();
    let width: u32 = inputs.iter().map(|(_, w)| w).sum();
    if gen.modules.contains_key(&testable) {
        return (testable, width);
    }
    module.name = testable.clone();
    module.title = format!("{} (comprobable)", module.title);
    let names = module.nets.iter().map(|net| net.name.clone()).collect();
    let mut b = Builder { prefix: String::new(), module, counter: 3_000_000, group: "Test wrapper".into(), names };
    // What each input becomes inside: the pin, or in test mode its part of `test_in`.
    let inner: HashMap<usize, Vec<Bit>> = inputs.iter().map(|(net, w)| (*net, b.wire(&format!("{}_in_test", b.module.nets[*net].name), *w))).collect();
    for inst in &mut b.module.instances {
        for bits in inst.conns.values_mut() {
            for bit in bits.iter_mut() {
                if let Bit::Net(net, index) = *bit {
                    if let Some(inside) = inner.get(&net) {
                        *bit = inside[index as usize];
                    }
                }
            }
        }
    }
    let test_mode = b.input(TEST_MODE, 1)[0];
    let test_in = b.input(TEST_IN, width);
    let mut at = 0;
    for (net, w) in &inputs {
        for index in 0..*w {
            b.mux_to(test_mode, Bit::Net(*net, index), test_in[at], inner[net][index as usize]);
            at += 1;
        }
    }
    gen.modules.insert(testable.clone(), b.module);
    (testable, width)
}

/// The wrapper in front of a group of cells: `width` flip-flops that hold their value
/// except while shifting, chained from `scan_in`. Gives their outputs.
fn wrapper(b: &mut Builder, enable: Bit, scan_in: Bit, width: u32) -> Vec<Bit> {
    let mut previous = scan_in;
    (0..width)
        .map(|_| {
            let (d, q) = (b.wire("wrapper_d", 1)[0], b.wire("wrapper", 1)[0]);
            b.mux_to(enable, q, previous, d);
            b.dff_to(d, q);
            previous = q;
            q
        })
        .collect()
}

/// Adds a scan chain to the rest of the design and tests the cells side by side: the cells
/// named in each group, instances of one module in the top, read in test mode the inputs a
/// wrapper of that group holds, and their chains all start from `scan_cells_in`, after the
/// wrapper. So the same shifting loads every cell alike; `scan_cells_out` gives the first
/// cell's chain and `scan_cells_differ` rises when any cell's differs from it. Test time
/// is then that of one cell, however many there are.
pub fn insert_scan_parallel(design: Design, groups: &[Vec<String>]) -> Result<ParallelScan, String> {
    let Design { top, modules } = design;
    let mut gen = Gen { modules };
    let mut lengths = HashMap::new();
    let (length, cell_length, wrapper_width, slice) = parallel_chains(&mut gen, &top, groups, &mut lengths)?;
    Ok(ParallelScan { design: Design { top, modules: gen.modules }, length, cell_length, wrapper_width, slice })
}

/// The same for a design whose top holds, as instance `inner`, the module with the cells:
/// that module's chains as `insert_scan_parallel` makes them, with its main chain part of
/// the top's, which goes through everything else. The cells' three pins reach the top.
/// So a whole chip is tested with its die's cells side by side.
pub fn insert_scan_around(design: Design, inner: &str, groups: &[Vec<String>]) -> Result<ParallelScan, String> {
    let Design { top, modules } = design;
    let mut gen = Gen { modules };
    let index = gen.modules[&top].instances.iter().position(|i| i.name == inner).ok_or_else(|| format!("there is no “{inner}” instance in {top}"))?;
    let InstKind::Module(module) = gen.modules[&top].instances[index].kind.clone() else { return Err(format!("{inner} is not a module")) };
    // One record of the modules done, so a module the inner one shares with the rest gets
    // its chain once.
    let mut lengths = HashMap::new();
    let (_, cell_length, wrapper_width, slice) = parallel_chains(&mut gen, &module, groups, &mut lengths)?;
    let length = visit(&mut gen, &top, &mut lengths, &HashSet::new());
    let mut module_b = {
        let module = gen.modules[&top].clone();
        let names = module.nets.iter().map(|net| net.name.clone()).collect();
        Builder { prefix: String::new(), module, counter: 5_000_000, group: "Cells tested in parallel".into(), names }
    };
    let scan_in = module_b.input(SCAN_CELLS_IN, 1);
    let (out, differ) = (module_b.output(SCAN_CELLS_OUT, 1), module_b.output(SCAN_CELLS_DIFFER, 1));
    let (from_out, from_differ) = (module_b.wire("cells_out", 1), module_b.wire("cells_differ", 1));
    let conns = &mut module_b.module.instances[index].conns;
    conns.insert(SCAN_CELLS_IN.into(), scan_in);
    conns.insert(SCAN_CELLS_OUT.into(), from_out.clone());
    conns.insert(SCAN_CELLS_DIFFER.into(), from_differ.clone());
    module_b.or_to(from_out[0], Bit::Zero, out[0]);
    module_b.or_to(from_differ[0], Bit::Zero, differ[0]);
    gen.modules.insert(top.clone(), module_b.module);
    Ok(ParallelScan { design: Design { top, modules: gen.modules }, length, cell_length, wrapper_width, slice })
}

/// `insert_scan_parallel` on module `top` of `gen`: gives its main chain's length, each
/// cell's, the wrapper's width and the slice the patterns are made on.
/// The chains `parallel_chains` makes: the main one's length, each cell's, the wrapper's
/// width and the slice the patterns are made on.
type Chains = (usize, usize, usize, Design);

fn parallel_chains(gen: &mut Gen, top: &str, groups: &[Vec<String>], lengths: &mut HashMap<String, usize>) -> Result<Chains, String> {
    let top = top.to_string();
    let find = |gen: &Gen, name: &str| gen.modules[&top].instances.iter().position(|i| i.name == name).ok_or_else(|| format!("there is no “{name}” instance in {top}"));
    let mut cells = Vec::new();
    for group in groups {
        cells.push(group.iter().map(|name| find(gen, name)).collect::<Result<Vec<_>, _>>()?);
    }
    let first = *cells.iter().flatten().next().ok_or("there are no cells")?;
    let InstKind::Module(cell) = gen.modules[&top].instances[first].kind.clone() else { return Err("the cells must be modules".into()) };
    if cells.iter().flatten().any(|i| !matches!(&gen.modules[&top].instances[*i].kind, InstKind::Module(m) if *m == cell)) {
        return Err("all the cells must be of the same module".into());
    }
    let (testable, width) = testable_module(gen, &cell);
    for i in cells.iter().flatten() {
        gen.modules.get_mut(&top).unwrap().instances[*i].kind = InstKind::Module(testable.clone());
    }
    // The plain cell goes if nothing uses it any more, so the design has one version.
    if !gen.modules.values().any(|m| m.instances.iter().any(|i| matches!(&i.kind, InstKind::Module(n) if *n == cell))) {
        gen.modules.remove(&cell);
    }
    // One record of the modules done for both: a module shared by the cell and the rest
    // gets its chain once.
    let cell_chain = visit(gen, &testable, lengths, &HashSet::new());
    let skip: HashSet<usize> = cells.iter().flatten().copied().collect();
    let length = visit(gen, &top, lengths, &skip);

    // The cells' chains, after the main one has given the top its pins.
    let module = gen.modules[&top].clone();
    let pin = |name: &str| module.ports.iter().find(|p| p.name == name).map(|p| Bit::Net(p.net, 0)).ok_or_else(|| format!("the {name} pin is missing: the rest of the design has no flip-flops"));
    let (enable, test_mode) = (pin(SCAN_ENABLE)?, pin(TEST_MODE)?);
    let names = module.nets.iter().map(|net| net.name.clone()).collect();
    let mut b = Builder { prefix: String::new(), module, counter: 4_000_000, group: "Cells tested in parallel".into(), names };
    let scan_in = b.input(SCAN_CELLS_IN, 1)[0];
    let out = b.output(SCAN_CELLS_OUT, 1)[0];
    let differ = b.output(SCAN_CELLS_DIFFER, 1)[0];
    let (mut reference, mut any) = (None, Bit::Zero);
    for group in &cells {
        let held = wrapper(&mut b, enable, scan_in, width);
        let after = held.last().copied().unwrap_or(scan_in);
        for index in group {
            let chain_out = b.wire("cell_chain_out", 1)[0];
            let conns = &mut b.module.instances[*index].conns;
            conns.insert(TEST_MODE.into(), vec![test_mode]);
            conns.insert(TEST_IN.into(), held.clone());
            conns.insert(SCAN_ENABLE.into(), vec![enable]);
            conns.insert(SCAN_IN.into(), vec![after]);
            conns.insert(SCAN_OUT.into(), vec![chain_out]);
            match reference {
                None => reference = Some(chain_out),
                Some(first) => {
                    let (x, y) = (b.wire("cell_differs", 1)[0], b.wire("any_differs", 1)[0]);
                    b.xor_to(first, chain_out, x);
                    b.or_to(any, x, y);
                    any = y;
                }
            }
        }
    }
    b.or_to(reference.unwrap_or(Bit::Zero), Bit::Zero, out);
    b.or_to(any, Bit::Zero, differ);
    gen.modules.insert(top.clone(), b.module);

    // One cell alone, with its wrapper, for the patterns: its other inputs are pins, and
    // its outputs too, though in the design only the chain reads it.
    let cell_module = gen.modules[&testable].clone();
    let mut s = Builder::new("test_slice", "A cell for generating patterns", "The testable cell with its wrapper, chained as in the design.");
    let enable = s.input(SCAN_ENABLE, 1)[0];
    let test_mode = s.input(TEST_MODE, 1)[0];
    let scan_in = s.input(SCAN_IN, 1)[0];
    let scan_out = s.output(SCAN_OUT, 1)[0];
    let held = wrapper(&mut s, enable, scan_in, width);
    let after = held.last().copied().unwrap_or(scan_in);
    let mut conns: Vec<(String, Vec<Bit>)> = Vec::new();
    for port in &cell_module.ports {
        let w = cell_module.nets[port.net].width;
        let bits = match port.name.as_str() {
            TEST_MODE => vec![test_mode],
            TEST_IN => held.clone(),
            SCAN_ENABLE => vec![enable],
            SCAN_IN => vec![after],
            SCAN_OUT => vec![scan_out],
            name if port.dir == crate::Dir::Input => s.input(name, w),
            name => s.output(name, w),
        };
        conns.push((port.name.clone(), bits));
    }
    let refs: Vec<(&str, Vec<Bit>)> = conns.iter().map(|(n, b)| (n.as_str(), b.clone())).collect();
    s.inst("cell", &testable, &refs);
    let mut modules = gen.modules.clone();
    modules.insert("test_slice".into(), s.finish());
    let slice = Design { top: "test_slice".into(), modules };
    Ok((length, cell_chain + width as usize, width as usize, slice))
}
