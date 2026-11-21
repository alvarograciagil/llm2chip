//! Dies loaded as the firmware leaves them and joined by their links, run gate by gate.

use netlist::block::tables;
use netlist::{die_design, DieConfig, Microprogram, Simulator};
use sim::lower::Layout;
use sim::Weights;

/// The mask of a value of the die's activation width.
pub fn mask(c: &DieConfig) -> u64 {
    (1u64 << c.cell.activation_bits) - 1
}

pub fn load_assembly(sim: &mut Simulator, c: &DieConfig, initial: &std::collections::HashMap<(u32, u32), i64>) {
    sim.set("load_vec", 1).unwrap();
    for ((reg, index), value) in initial {
        sim.set("vec_addr", ((*reg as u64) << c.index_bits) | *index as u64).unwrap();
        sim.set("vec_data", *value as u64 & mask(c)).unwrap();
        sim.tick();
    }
    sim.set("load_vec", 0).unwrap();
}

/// Where each register starts in a compact vector memory, before anything goes into it.
pub fn load_register_table(sim: &mut Simulator, c: &DieConfig, layout: &Layout) {
    sim.set("load_base", 1).unwrap();
    for (register, entry) in sim::chip::register_table(c, layout).iter().enumerate() {
        sim.set("base_addr", register as u64).unwrap();
        sim.set("base_data", *entry as u64).unwrap();
        sim.tick();
    }
    sim.set("load_base", 0).unwrap();
}

/// Writes the microprogram, once, as the load controller does at start-up.
pub fn load_microprogram(sim: &mut Simulator, c: &DieConfig, code: &Microprogram) {
    assert!(code.words.len() < 1 << c.microcode_bits, "{} words", code.words.len());
    sim.set("start", 1).unwrap();
    sim.tick();
    sim.set("load_code", 1).unwrap();
    for (i, word) in code.words.iter().enumerate() {
        sim.set("code_addr", i as u64).unwrap();
        sim.set_words("code_data", &word.encode_words(c)).unwrap();
        sim.tick();
    }
    sim.set("load_code", 0).unwrap();
}

/// A die of `c` with `layout`'s weights and `code` loaded, as the firmware would leave it.
pub fn loaded_die(layout: &Layout, code: &Microprogram, weights: &Weights, c: DieConfig) -> Simulator {
    let c = sim::chip::compact_die(c, layout);
    let mut sim = Simulator::with_names(&die_design(c)).unwrap();
    load_microprogram(&mut sim, &c, code);
    load_register_table(&mut sim, &c, layout);
    load_assembly(&mut sim, &c, &layout.setup().initial);
    let data_mask = (1u64 << c.scale_bits.max(c.cell.weight_bits).max(c.cell.activation_bits)) - 1;
    sim.set("load_fn", 1).unwrap();
    for (address, value) in tables(&c).iter().enumerate() {
        sim.set("fn_addr", address as u64).unwrap();
        sim.set("cell_data", *value as u64 & mask(&c)).unwrap();
        sim.tick();
    }
    sim.set("load_fn", 0).unwrap();
    let put = |sim: &mut Simulator, address: u64, value: i64| {
        sim.set("cell_addr", address).unwrap();
        sim.set("cell_data", value as u64 & data_mask).unwrap();
        sim.tick();
    };
    for cell in 0..c.cells {
        let (codes, factors) = layout.cell_contents(weights, cell).unwrap();
        sim.set("load_cell", 1 << cell).unwrap();
        sim.set("load_scale", 0).unwrap();
        for (address, code) in codes.iter().enumerate() {
            put(&mut sim, address as u64, *code);
        }
        sim.set("load_scale", 1).unwrap();
        for (group, (scale, bias)) in factors.iter().enumerate() {
            put(&mut sim, 2 * group as u64, *scale);
            put(&mut sim, 2 * group as u64 + 1, *bias);
        }
    }
    sim.set("load_cell", 0).unwrap();
    sim.set("load_scale", 0).unwrap();
    sim
}

/// One lane of a link between two dies: `port` is `link`, `loop`, or one of their lanes
/// after the first (`link1`, `link2`…).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Wire {
    pub from: usize,
    pub to: usize,
    pub port: String,
}

/// What a lane of a link did in a cycle, as the board saw its wires.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LinkSample {
    /// The sender has a word on the wire.
    pub valid: bool,
    /// The receiver can take it.
    pub ready: bool,
}

/// Every lane of `wires` `(from, to, port)`: a link a block wide has a wire of its own for
/// every lane after the first.
pub fn lanes(dies: &[(Simulator, Microprogram)], wires: &[(usize, usize, &str)]) -> Vec<Wire> {
    let mut lanes = Vec::new();
    for (from, to, p) in wires {
        lanes.push(Wire { from: *from, to: *to, port: p.to_string() });
        let mut k = 1;
        while dies[*from].0.get(&format!("{p}{k}_tx_valid")).is_some() {
            lanes.push(Wire { from: *from, to: *to, port: format!("{p}{k}") });
            k += 1;
        }
    }
    lanes
}

/// Sets the parameters of `routine` for `token` at `position` on every die and starts it,
/// as the firmware does.
pub fn start(dies: &mut [(Simulator, Microprogram)], routine: usize, token: u32, position: u32) {
    for (sim, code) in dies.iter_mut() {
        sim.set("start", 0).unwrap();
        sim.set("param_write", 1).unwrap();
        for parameter in &code.parameters {
            sim.set("param_index", parameter.register as u64).unwrap();
            sim.set("param_data", parameter.value(token, position) as u64).unwrap();
            sim.tick();
        }
        sim.set("param_write", 0).unwrap();
        sim.set("entry", code.entries[routine] as u64).unwrap();
        sim.set("start", 1).unwrap();
        sim.tick();
        sim.set("start", 0).unwrap();
    }
}

/// Runs `routine` with `token` at `position` on every die at once, joined by `wires`
/// `(from, to, port)`, until all have halted. Gives the cycles it took.
pub fn run_joined(dies: &mut [(Simulator, Microprogram)], wires: &[(usize, usize, &str)], routine: usize, token: u32, position: u32) -> u64 {
    let lanes = lanes(dies, wires);
    run_observed(dies, &lanes, routine, token, position, |_, _, _| {})
}

/// `run_joined` over lanes already expanded, calling `observe` before every clock edge with
/// the cycle (from 0), the dies and what each lane carries.
pub fn run_observed(dies: &mut [(Simulator, Microprogram)], lanes: &[Wire], routine: usize, token: u32, position: u32, mut observe: impl FnMut(u64, &[(Simulator, Microprogram)], &[LinkSample])) -> u64 {
    start(dies, routine, token, position);
    run_from(dies, lanes, 0, |cycle, dies, samples| {
        observe(cycle, dies, samples);
        true
    })
    .0
}

/// Runs the dies, already started, until all have halted or `observe` says to stop,
/// counting cycles from `first` (a routine resumed from a saved state goes on from its
/// cycle). Gives the cycle it stopped at and whether every die had halted.
pub fn run_from(dies: &mut [(Simulator, Microprogram)], lanes: &[Wire], first: u64, mut observe: impl FnMut(u64, &[(Simulator, Microprogram)], &[LinkSample]) -> bool) -> (u64, bool) {
    let names: Vec<[String; 8]> = lanes
        .iter()
        .map(|w| {
            let p = &w.port;
            [format!("{p}_tx_valid"), format!("{p}_tx_bits"), format!("{p}_rx_ready"), format!("{p}_rx_ack"), format!("{p}_rx_valid"), format!("{p}_rx_bits"), format!("{p}_tx_ready"), format!("{p}_tx_ack")]
        })
        .collect();
    let mut samples = vec![LinkSample::default(); lanes.len()];
    let mut cycles = first;
    while dies.iter().any(|(sim, _)| sim.get("halted").unwrap() == 0) {
        let outputs: Vec<_> = lanes
            .iter()
            .zip(&names)
            .map(|(w, n)| {
                let (a, b) = (&dies[w.from].0, &dies[w.to].0);
                (a.get(&n[0]).unwrap(), a.get_wide(&n[1]).unwrap(), b.get(&n[2]).unwrap(), b.get(&n[3]).unwrap())
            })
            .collect();
        for ((w, n), (valid, bits, ready, ack)) in lanes.iter().zip(&names).zip(&outputs) {
            dies[w.to].0.set(&n[4], *valid).unwrap();
            dies[w.to].0.set_wide(&n[5], *bits).unwrap();
            dies[w.from].0.set(&n[6], *ready).unwrap();
            dies[w.from].0.set(&n[7], *ack).unwrap();
        }
        for (sample, (valid, _, ready, _)) in samples.iter_mut().zip(&outputs) {
            *sample = LinkSample { valid: *valid == 1, ready: *ready == 1 };
        }
        for (sim, _) in dies.iter_mut() {
            sim.settle();
        }
        if !observe(cycles, dies, &samples) {
            return (cycles, false);
        }
        for (sim, _) in dies.iter_mut() {
            sim.tick();
        }
        cycles += 1;
        assert!(cycles < 2_000_000, "the dies do not finish");
    }
    (cycles, true)
}
