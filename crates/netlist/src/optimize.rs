//! Removes logic that can never matter once some signals are known to be constant.

use crate::{Bit, Design, Dir, InstKind, Module, Prim};
use std::collections::{BTreeSet, HashMap, HashSet};

fn resolve(aliases: &HashMap<Bit, Bit>, mut bit: Bit) -> Bit {
    while let Some(next) = aliases.get(&bit) {
        bit = *next;
    }
    bit
}

/// What a gate's output is known to equal, from what is known of its inputs.
fn known_output(prim: Prim, input: impl Fn(&str) -> Bit, q: Bit) -> Option<Bit> {
    match prim {
        Prim::And2 => match (input("a"), input("b")) {
            (Bit::Zero, _) | (_, Bit::Zero) => Some(Bit::Zero),
            (Bit::One, x) | (x, Bit::One) => Some(x),
            (a, b) if a == b => Some(a),
            _ => None,
        },
        Prim::Or2 => match (input("a"), input("b")) {
            (Bit::One, _) | (_, Bit::One) => Some(Bit::One),
            (Bit::Zero, x) | (x, Bit::Zero) => Some(x),
            (a, b) if a == b => Some(a),
            _ => None,
        },
        Prim::Xor2 => match (input("a"), input("b")) {
            (Bit::Zero, x) | (x, Bit::Zero) => Some(x),
            (a, b) if a == b => Some(Bit::Zero),
            _ => None,
        },
        Prim::Not => match input("a") {
            Bit::Zero => Some(Bit::One),
            Bit::One => Some(Bit::Zero),
            _ => None,
        },
        Prim::Mux2 => match (input("s"), input("a"), input("b")) {
            (Bit::Zero, a, _) => Some(a),
            (Bit::One, _, b) => Some(b),
            (_, a, b) if a == b => Some(a),
            _ => None,
        },
        // Flip-flops start at zero, so one that only ever loads zero stays there.
        Prim::Dff => (input("d") == Bit::Zero || input("d") == q).then_some(Bit::Zero),
        _ => None,
    }
}

fn simplify(module: &mut Module) {
    let port_bits: HashSet<Bit> = module.ports.iter().filter(|p| p.dir == Dir::Output).flat_map(|p| (0..module.nets[p.net].width).map(|b| Bit::Net(p.net, b))).collect();
    let mut aliases: HashMap<Bit, Bit> = HashMap::new();
    loop {
        let before = aliases.len();
        for instance in &module.instances {
            let InstKind::Prim(prim) = instance.kind else { continue };
            let Some(output) = prim.outputs().iter().find_map(|o| instance.conns.get(*o)) else { continue };
            let [y] = output.as_slice() else { continue };
            if aliases.contains_key(y) || port_bits.contains(y) {
                continue;
            }
            let input = |pin: &str| resolve(&aliases, instance.conns[pin][0]);
            if let Some(known) = known_output(prim, input, *y) {
                aliases.insert(*y, known);
            }
        }
        if aliases.len() == before {
            break;
        }
    }
    for instance in &mut module.instances {
        let outputs: &[&str] = match instance.kind {
            InstKind::Prim(prim) => prim.outputs(),
            InstKind::Module(_) => &[],
        };
        for (pin, bits) in instance.conns.iter_mut() {
            if !outputs.contains(&pin.as_str()) {
                bits.iter_mut().for_each(|b| *b = resolve(&aliases, *b));
            }
        }
    }
}

/// The names of each module's output pins, to tell inputs from outputs on an instance.
fn output_pins(design: &Design, instance: &crate::Instance) -> BTreeSet<String> {
    match &instance.kind {
        InstKind::Prim(prim) => prim.outputs().iter().map(|o| o.to_string()).collect(),
        InstKind::Module(name) => design.modules[name].ports.iter().filter(|p| p.dir == Dir::Output).map(|p| p.name.clone()).collect(),
    }
}

fn remove_dead(design: &Design, module: &mut Module) {
    let outputs: Vec<BTreeSet<String>> = module.instances.iter().map(|i| output_pins(design, i)).collect();
    let mut drivers: HashMap<Bit, usize> = HashMap::new();
    for (index, instance) in module.instances.iter().enumerate() {
        for pin in &outputs[index] {
            for bit in instance.conns.get(pin).into_iter().flatten() {
                drivers.insert(*bit, index);
            }
        }
    }
    let mut pending: Vec<Bit> = module.ports.iter().filter(|p| p.dir == Dir::Output).flat_map(|p| (0..module.nets[p.net].width).map(move |b| Bit::Net(p.net, b))).collect();
    let mut live = vec![false; module.instances.len()];
    while let Some(bit) = pending.pop() {
        let Some(index) = drivers.get(&bit).copied() else { continue };
        if std::mem::replace(&mut live[index], true) {
            continue;
        }
        for (pin, bits) in &module.instances[index].conns {
            if !outputs[index].contains(pin) {
                pending.extend(bits.iter().copied());
            }
        }
    }
    let mut keep = live.into_iter();
    module.instances.retain(|_| keep.next().unwrap_or(false));
}

/// Folds constants through the top module and drops what no output depends on.
pub fn trim_top(design: &mut Design) {
    let top = design.top.clone();
    let mut module = design.modules.remove(&top).expect("top module");
    simplify(&mut module);
    remove_dead(design, &mut module);
    design.modules.insert(top, module);
    // Modules no instance refers to any longer leave the design with them.
    let mut used = BTreeSet::from([design.top.clone()]);
    let mut pending = vec![design.top.clone()];
    while let Some(name) = pending.pop() {
        for instance in &design.modules[&name].instances {
            if let InstKind::Module(child) = &instance.kind {
                if used.insert(child.clone()) {
                    pending.push(child.clone());
                }
            }
        }
    }
    design.modules.retain(|name, _| used.contains(name));
}
