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
use std::collections::HashMap;

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
fn visit(gen: &mut Gen, name: &str, lengths: &mut HashMap<String, usize>) -> usize {
    if let Some(known) = lengths.get(name) {
        return *known;
    }
    let module = gen.modules[name].clone();
    let children: Vec<(usize, usize)> = module
        .instances
        .iter()
        .enumerate()
        .filter_map(|(i, inst)| match &inst.kind {
            InstKind::Module(child) => Some((i, visit(gen, child, lengths))),
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
    let mut b = Builder { prefix: String::new(), module, counter: 2_000_000, group: "Cadena de scan".into(), names };
    let enable = b.input(SCAN_ENABLE, 1)[0];
    let test_mode = b.input(TEST_MODE, 1)[0];
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
    let length = visit(&mut gen, &top, &mut HashMap::new());
    ScanDesign { design: Design { top, modules: gen.modules }, length }
}
