//! The sampling unit in gates must make the same choice as its model, in the same cycles.

use netlist::sampler::{exponential_table, sampler_design, SamplerConfig, SamplerModel};
use netlist::Simulator;

const CONFIG: SamplerConfig = SamplerConfig { logit_bits: 16, index_bits: 12, entries: 8 };

struct Noise(u64);

impl Noise {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
}

fn run_gates(sim: &mut Simulator, logits: &[i64], inverse: u32, limit: usize, greedy: bool) -> (u32, u64) {
    sim.set("start", 1).unwrap();
    sim.set("inverse", inverse as u64).unwrap();
    sim.set("limit", limit as u64).unwrap();
    sim.set("greedy", greedy as u64).unwrap();
    sim.tick();
    sim.set("start", 0).unwrap();
    for (i, value) in logits.iter().enumerate() {
        sim.set("logit_valid", 1).unwrap();
        sim.set("logit_data", *value as u64 & 0xFFFF).unwrap();
        sim.set("logit_last", (i + 1 == logits.len()) as u64).unwrap();
        sim.tick();
    }
    sim.set("logit_valid", 0).unwrap();
    sim.set("logit_last", 0).unwrap();
    let mut cycles = 1;
    sim.settle();
    while sim.get("done").unwrap() == 0 {
        sim.tick();
        cycles += 1;
        assert!(cycles < 10_000, "la unidad no termina");
    }
    (sim.get("token").unwrap() as u32, cycles)
}

#[test]
fn gates_match_the_model() {
    let mut sim = Simulator::new(&sampler_design(CONFIG)).unwrap();
    let table = exponential_table();
    for (address, value) in table.iter().enumerate() {
        sim.set("table_write", 1).unwrap();
        sim.set("table_address", address as u64).unwrap();
        sim.set("table_data", *value as u64).unwrap();
        sim.tick();
    }
    sim.set("table_write", 0).unwrap();
    let seed = 0x1234_5678;
    sim.set("seed_write", 1).unwrap();
    sim.set("seed", seed).unwrap();
    sim.tick();
    sim.set("seed_write", 0).unwrap();

    let mut model = SamplerModel { config: CONFIG, rng: seed as u32, table };
    let mut noise = Noise(5);
    for round in 0..200 {
        let length = 1 + noise.next() as usize % 300;
        let spread = 1 << (4 + noise.next() % 11);
        let logits: Vec<i64> = (0..length).map(|_| (noise.next() % spread) as i64 - spread as i64 / 2).collect();
        let inverse = noise.next() % 4096;
        let limit = 1 + noise.next() as usize % 8;
        let greedy = noise.next().is_multiple_of(8);
        let got = run_gates(&mut sim, &logits, inverse, limit, greedy);
        let expected = model.sample(&logits, inverse, limit, greedy);
        assert_eq!(got, expected, "ronda {round}, {length} logits, k={limit}, inversa {inverse}, voraz {greedy}");
    }
}

#[test]
fn a_cold_distribution_follows_the_logits() {
    let table = exponential_table();
    let mut model = SamplerModel { config: CONFIG, rng: 77, table };
    let logits = [3000i64, 4000, 2500, 3500];
    let mut hits = [0; 4];
    for _ in 0..4000 {
        // One logit step of 256 units per nat, at temperature 1, with the table's 1/16 steps.
        hits[model.sample(&logits, 16, 8, false).0 as usize] += 1;
    }
    assert!(hits[1] > hits[3] && hits[3] > hits[0] && hits[0] > hits[2], "{hits:?}");
}
