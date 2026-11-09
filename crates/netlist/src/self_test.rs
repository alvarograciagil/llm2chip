//! Gives every read-write memory of a design a march test, reached through one signal.
//! A chip runs it before anything is loaded, since the test overwrites the memories.

use crate::cell::Gen;
use crate::{Bit, Builder, Design, InstKind, Module, Prim};
use std::collections::HashMap;

const ENABLE: &str = "bist_enable";
const DONE: &str = "bist_done";
const FAIL: &str = "bist_fail";

struct Hooks {
    enable: Bit,
    done: Bit,
    fail: Bit,
}

fn net_named(module: &Module, name: &str) -> Option<Bit> {
    module.nets.iter().position(|n| n.name == name).map(|id| Bit::Net(id, 0))
}

fn visit(gen: &mut Gen, name: &str, finished: &mut HashMap<String, bool>, top: bool) -> bool {
    if let Some(known) = finished.get(name) {
        return *known;
    }
    let module = gen.modules[name].clone();
    let children: Vec<(usize, bool)> = module
        .instances
        .iter()
        .enumerate()
        .filter_map(|(i, inst)| match &inst.kind {
            InstKind::Module(child) => Some((i, visit(gen, child, finished, false))),
            _ => None,
        })
        .collect();
    let memories: Vec<usize> = module.instances.iter().enumerate().filter(|(_, i)| matches!(i.kind, InstKind::Prim(Prim::Sram { .. } | Prim::Ram2p { .. }))).map(|(i, _)| i).collect();
    let has = !memories.is_empty() || children.iter().any(|(_, tested)| *tested);
    finished.insert(name.to_string(), has);
    if !has {
        return false;
    }

    let mut b = Builder { module, counter: 1_000_000, group: "Autotest de memorias".into() };
    let hooks = if top {
        let found = |b: &Builder, net: &str| net_named(&b.module, net).expect("el chip debe declarar las señales de autotest");
        Hooks { enable: found(&b, ENABLE), done: found(&b, DONE), fail: found(&b, FAIL) }
    } else {
        Hooks { enable: b.input(ENABLE, 1)[0], done: b.output(DONE, 1)[0], fail: b.output(FAIL, 1)[0] }
    };
    let enable = hooks.enable;
    let previous = b.wire("bist_enable_q", 1)[0];
    b.dff_to(enable, previous);
    let not_previous = b.wire("bist_first_n", 1)[0];
    b.not_to(previous, not_previous);
    let start = b.wire("bist_start", 1)[0];
    b.and_to(enable, not_previous, start);

    let (mut all_done, mut any_fail) = (Bit::One, Bit::Zero);
    let mut combine = |b: &mut Builder, done: Bit, fail: Bit| {
        let (d, f) = (b.wire("bist_done_so_far", 1)[0], b.wire("bist_fail_so_far", 1)[0]);
        b.and_to(all_done, done, d);
        b.or_to(any_fail, fail, f);
        (all_done, any_fail) = (d, f);
    };
    for (index, tested) in children {
        if !tested {
            continue;
        }
        let (done, fail) = (b.wire("bist_child_done", 1), b.wire("bist_child_fail", 1));
        let conns = &mut b.module.instances[index].conns;
        conns.insert(ENABLE.into(), vec![enable]);
        conns.insert(DONE.into(), done.clone());
        conns.insert(FAIL.into(), fail.clone());
        combine(&mut b, done[0], fail[0]);
    }
    for index in memories {
        let (addr_bits, width, two_ports) = match b.module.instances[index].kind {
            InstKind::Prim(Prim::Sram { addr_bits, width }) => (addr_bits, width, false),
            InstKind::Prim(Prim::Ram2p { addr_bits, width }) => (addr_bits, width, true),
            _ => unreachable!("solo se recogieron memorias"),
        };
        let engine = gen.march_test(addr_bits, width);
        let (address, we, data) = (b.wire("bist_address", addr_bits), b.wire("bist_we", 1), b.wire("bist_data", width));
        let (done, fail, where_failed) = (b.wire("bist_memory_done", 1), b.wire("bist_memory_fail", 1), b.wire("bist_where", addr_bits));
        let rdata = b.module.instances[index].conns["rdata"].clone();
        let name = format!("march_{}", b.module.instances[index].name);
        b.inst(&name, &engine, &[("start", vec![start]), ("mem_rdata", rdata), ("mem_addr", address.clone()), ("mem_we", we.clone()), ("mem_wdata", data.clone()), ("done", done.clone()), ("fail", fail.clone()), ("fail_addr", where_failed)]);
        let take = |b: &mut Builder, pin: &str, test: &[Bit]| {
            let original = b.module.instances[index].conns[pin].clone();
            let chosen: Vec<Bit> = original
                .iter()
                .zip(test)
                .map(|(functional, tested)| {
                    let y = b.wire("bist_mux", 1)[0];
                    b.mux_to(enable, *functional, *tested, y);
                    y
                })
                .collect();
            b.module.instances[index].conns.insert(pin.into(), chosen);
        };
        if two_ports {
            take(&mut b, "raddr", &address);
            take(&mut b, "waddr", &address);
        } else {
            take(&mut b, "addr", &address);
        }
        take(&mut b, "we", &we);
        take(&mut b, "wdata", &data);
        combine(&mut b, done[0], fail[0]);
    }
    b.or_to(all_done, Bit::Zero, hooks.done);
    b.or_to(any_fail, Bit::Zero, hooks.fail);
    gen.modules.insert(name.to_string(), b.module);
    true
}

/// Adds the march tests. The top module must have nets named for the three signals.
pub(crate) fn insert(design: Design) -> Design {
    let Design { top, modules } = design;
    let mut gen = Gen { modules };
    let mut finished: HashMap<String, bool> = HashMap::new();
    visit(&mut gen, &top, &mut finished, true);
    Design { top, modules: gen.modules }
}
