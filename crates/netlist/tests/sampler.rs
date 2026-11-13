//! The sampling unit in gates must make the same choice as its model, in the same cycles.

use netlist::sampler::{exponential_table, sampler_design, Sampled, SamplerConfig, SamplerModel};
use netlist::Simulator;

const CONFIG: SamplerConfig = SamplerConfig { logit_bits: 16, index_bits: 12, entries: 8, lanes: 1 };

struct Noise(u64);

impl Noise {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
}

/// Streams the logits in groups, holding each one until the unit is ready for the next.
fn run_gates(sim: &mut Simulator, config: SamplerConfig, logits: &[i64], inverse: u32, limit: usize, greedy: bool) -> Sampled {
    sim.set("start", 1).unwrap();
    sim.set("inverse", inverse as u64).unwrap();
    sim.set("limit", limit as u64).unwrap();
    sim.set("greedy", greedy as u64).unwrap();
    sim.tick();
    sim.set("start", 0).unwrap();
    let lanes = config.lanes as usize;
    let groups: Vec<&[i64]> = logits.chunks(lanes).collect();
    let mut stream_cycles = 0;
    for (g, group) in groups.iter().enumerate() {
        let data = group.iter().enumerate().fold(0u128, |word, (lane, value)| word | ((*value as u16 as u128) << (16 * lane)));
        sim.set("logit_valid", (1u64 << group.len()) - 1).unwrap();
        sim.set_wide("logit_data", data).unwrap();
        sim.set("logit_last", (g + 1 == groups.len()) as u64).unwrap();
        loop {
            sim.settle();
            let ready = sim.get("ready").unwrap() == 1;
            sim.tick();
            stream_cycles += 1;
            if ready {
                break;
            }
            assert!(stream_cycles < 100_000, "the filter does not finish");
        }
    }
    sim.set("logit_valid", 0).unwrap();
    sim.set("logit_last", 0).unwrap();
    let mut finish_cycles = 1;
    sim.settle();
    while sim.get("done").unwrap() == 0 {
        sim.tick();
        finish_cycles += 1;
        assert!(finish_cycles < 10_000, "the unit does not finish");
    }
    Sampled { token: sim.get("token").unwrap() as u32, stream_cycles, finish_cycles }
}

fn prepared(config: SamplerConfig, seed: u64) -> Simulator {
    let mut sim = Simulator::new(&sampler_design(config)).unwrap();
    for (address, value) in exponential_table().iter().enumerate() {
        sim.set("table_write", 1).unwrap();
        sim.set("table_address", address as u64).unwrap();
        sim.set("table_data", *value as u64).unwrap();
        sim.tick();
    }
    sim.set("table_write", 0).unwrap();
    sim.set("seed_write", 1).unwrap();
    sim.set("seed", seed).unwrap();
    sim.tick();
    sim.set("seed_write", 0).unwrap();
    sim
}

#[test]
fn gates_match_the_model() {
    for lanes in [1, 2, 4, 8] {
        let config = SamplerConfig { lanes, ..CONFIG };
        let seed = 0x1234_5678;
        let mut sim = prepared(config, seed);
        let mut model = SamplerModel { config, rng: seed as u32, table: exponential_table() };
        let mut noise = Noise(5 + lanes as u64);
        for round in 0..200 {
            let length = 1 + noise.next() as usize % 300;
            let spread = 1 << (4 + noise.next() % 11);
            let logits: Vec<i64> = (0..length).map(|_| (noise.next() % spread) as i64 - spread as i64 / 2).collect();
            let inverse = noise.next() % 65536;
            let limit = 1 + noise.next() as usize % 8;
            let greedy = noise.next().is_multiple_of(8);
            let got = run_gates(&mut sim, config, &logits, inverse, limit, greedy);
            let expected = model.sample(&logits, inverse, limit, greedy);
            assert_eq!(got, expected, "{lanes} carriles, ronda {round}, {length} logits, k={limit}, inversa {inverse}, voraz {greedy}");
        }
    }
}

#[test]
fn more_lanes_choose_the_same_token_faster() {
    let mut noise = Noise(9);
    let logits: Vec<i64> = (0..4096).map(|_| (noise.next() % 30_000) as i64 - 15_000).collect();
    let run = |lanes| SamplerModel { config: SamplerConfig { lanes, ..CONFIG }, rng: 77, table: exponential_table() }.sample(&logits, 256, 8, false);
    let (serial, parallel) = (run(1), run(64));
    assert_eq!(serial.token, parallel.token);
    assert_eq!(serial.stream_cycles, 4096);
    assert!(parallel.stream_cycles < 4096 / 32, "{} ciclos", parallel.stream_cycles);
}

#[test]
fn a_cold_distribution_follows_the_logits() {
    let table = exponential_table();
    let mut model = SamplerModel { config: CONFIG, rng: 77, table };
    let logits = [3000i64, 4000, 2500, 3500];
    let mut hits = [0; 4];
    for _ in 0..4000 {
        // One logit step of 256 units per nat, at temperature 1, with the table's 1/16 steps.
        hits[model.sample(&logits, 256, 8, false).token as usize] += 1;
    }
    assert!(hits[1] > hits[3] && hits[3] > hits[0] && hits[0] > hits[2], "{hits:?}");
}
