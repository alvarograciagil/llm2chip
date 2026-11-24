//! Moonshine, speech to text, taken to the chip on a toy checkpoint: the die's exact
//! arithmetic against the floating-point reference of `sim::moonshine`, and the die's gates
//! against the exact machine.

mod common;
use common::*;
#[allow(unused_imports)]
use common::{cell_die_config, enter, load_assembly, load_microprogram, mask};
use sim::moonshine::{Moonshine, MoonshineConfig};
use sim::moonshine_chip::MoonshineChip;
use std::path::Path;

/// Width 16 in 2 heads of 8 (6 values rotated), 2 encoder and 2 decoder layers, 24 tokens.
const HIDDEN: usize = 16;
const INTERMEDIATE: usize = 32;
const VOCAB: usize = 24;
/// 31 outputs of the first convolution, 9 of the second, 4 frames.
const SAMPLES: usize = 2100;

fn make_moonshine_checkpoint(dir: &Path) {
    let config = serde_json::json!({
        "model_type": "moonshine", "hidden_size": HIDDEN, "intermediate_size": INTERMEDIATE,
        "encoder_num_hidden_layers": 2, "decoder_num_hidden_layers": 2,
        "encoder_num_attention_heads": 2, "decoder_num_attention_heads": 2,
        "encoder_num_key_value_heads": 2, "decoder_num_key_value_heads": 2,
        "partial_rotary_factor": 0.75, "rope_theta": 100.0, "attention_bias": false,
        "vocab_size": VOCAB, "max_position_embeddings": 32,
        "decoder_start_token_id": 1, "bos_token_id": 1, "eos_token_id": 2
    });
    std::fs::write(dir.join("config.json"), serde_json::to_string_pretty(&config).unwrap()).unwrap();
    let mut rng = Rng(0x5EED_0F50_DA7A);
    let mut w = Writer::default();
    let mut put = |w: &mut Writer, name: &str, shape: &[usize], spread: f32, offset: f32| {
        let values: Vec<f32> = (0..shape.iter().product()).map(|_| offset + spread * rng.unit()).collect();
        w.tensor(name, "F32", shape, values.iter().flat_map(|v| v.to_le_bytes()).collect());
    };
    let h = HIDDEN;
    put(&mut w, "model.encoder.conv1.weight", &[h, 1, 127], 0.15, 0.0);
    put(&mut w, "model.encoder.groupnorm.weight", &[h], 0.2, 1.0);
    put(&mut w, "model.encoder.groupnorm.bias", &[h], 0.1, 0.0);
    put(&mut w, "model.encoder.conv2.weight", &[2 * h, h, 7], 0.1, 0.0);
    put(&mut w, "model.encoder.conv2.bias", &[2 * h], 0.1, 0.0);
    put(&mut w, "model.encoder.conv3.weight", &[h, 2 * h, 3], 0.1, 0.0);
    put(&mut w, "model.encoder.conv3.bias", &[h], 0.1, 0.0);
    for layer in 0..2 {
        let l = |n: &str| format!("model.encoder.layers.{layer}.{n}");
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"] {
            put(&mut w, &l(&format!("self_attn.{proj}.weight")), &[h, h], 0.3, 0.0);
        }
        put(&mut w, &l("input_layernorm.weight"), &[h], 0.2, 1.0);
        put(&mut w, &l("post_attention_layernorm.weight"), &[h], 0.2, 1.0);
        put(&mut w, &l("mlp.fc1.weight"), &[INTERMEDIATE, h], 0.3, 0.0);
        put(&mut w, &l("mlp.fc1.bias"), &[INTERMEDIATE], 0.1, 0.0);
        put(&mut w, &l("mlp.fc2.weight"), &[h, INTERMEDIATE], 0.2, 0.0);
        put(&mut w, &l("mlp.fc2.bias"), &[h], 0.1, 0.0);
    }
    put(&mut w, "model.encoder.layer_norm.weight", &[h], 0.2, 1.0);
    put(&mut w, "model.decoder.embed_tokens.weight", &[VOCAB, h], 0.5, 0.0);
    for layer in 0..2 {
        let l = |n: &str| format!("model.decoder.layers.{layer}.{n}");
        for attention in ["self_attn", "encoder_attn"] {
            for proj in ["q_proj", "k_proj", "v_proj", "o_proj"] {
                put(&mut w, &l(&format!("{attention}.{proj}.weight")), &[h, h], 0.3, 0.0);
            }
        }
        for norm in ["input_layernorm", "post_attention_layernorm", "final_layernorm"] {
            put(&mut w, &l(&format!("{norm}.weight")), &[h], 0.2, 1.0);
        }
        put(&mut w, &l("mlp.fc1.weight"), &[2 * INTERMEDIATE, h], 0.3, 0.0);
        put(&mut w, &l("mlp.fc1.bias"), &[2 * INTERMEDIATE], 0.1, 0.0);
        put(&mut w, &l("mlp.fc2.weight"), &[h, INTERMEDIATE], 0.2, 0.0);
        put(&mut w, &l("mlp.fc2.bias"), &[h], 0.1, 0.0);
    }
    put(&mut w, "model.decoder.norm.weight", &[h], 0.2, 1.0);
    w.write(&dir.join("model.safetensors"));
}

/// A clip with a few tones and some noise, the same every time.
fn test_clip(samples: usize) -> Vec<f32> {
    let mut rng = Rng(0xA0D10);
    (0..samples)
        .map(|i| {
            let t = i as f32 / 16_000.0;
            0.4 * (t * 440.0 * std::f32::consts::TAU).sin() + 0.2 * (t * 1250.0 * std::f32::consts::TAU).sin() + 0.05 * rng.unit()
        })
        .collect()
}

fn toy(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("llm2chip-moonshine-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    make_moonshine_checkpoint(&dir);
    dir
}

#[test]
fn the_toy_recogniser_makes_a_frame_every_384_samples_and_scores_every_token() {
    let dir = toy("reference");
    let model = Moonshine::open(&dir).unwrap();
    assert_eq!((model.config.head_dim, model.config.rotary), (8, 6));
    assert_eq!(MoonshineConfig::frames(SAMPLES), 4);
    let frames = model.encode(&test_clip(SAMPLES)).unwrap();
    assert_eq!(frames.len(), 4);
    let mut decoder = model.decoder(&frames).unwrap();
    let logits = decoder.step(model.config.start).unwrap();
    assert_eq!(logits.len(), VOCAB);
    assert!(logits.iter().all(|v| v.is_finite()));
    std::fs::remove_dir_all(&dir).ok();
}

/// The largest difference between two lists of scores, over the largest score.
fn relative(a: &[f32], b: &[f32]) -> f32 {
    let largest = b.iter().fold(0f32, |m, v| m.max(v.abs()));
    a.iter().zip(b).fold(0f32, |m, (x, y)| m.max((x - y).abs())) / largest
}

#[test]
fn moonshine_in_the_die_s_arithmetic_scores_as_floating_point_does() {
    let dir = toy("exact");
    let clip = test_clip(SAMPLES);
    let model = Moonshine::open(&dir).unwrap();
    let frames = model.encode(&clip).unwrap();
    let mut reference = model.decoder(&frames).unwrap();
    let mut chip = MoonshineChip::exact(&dir, SAMPLES, 16).unwrap();
    chip.encode(&clip).unwrap();
    // The same tokens in both, so that each step compares the same thing.
    let tokens = [1u32, 7, 12, 3, 19];
    for (position, token) in tokens.iter().enumerate() {
        let expected = reference.step(*token).unwrap();
        let (got, _) = chip.step(*token, position).unwrap();
        let error = relative(&got, &expected);
        eprintln!("token {position}: {:.3} % off", error * 100.0);
        // The products are exact to a few millionths; what differs is the die's tables of
        // functions (tanh, GELU, SiLU, exp), each a fraction of a percent, through four
        // layers of a toy with random weights. It does not shrink with wider vectors.
        assert!(error < 0.03, "step {position}: {:.2} % off", error * 100.0);
    }
    std::fs::remove_dir_all(&dir).ok();
}

/// The toy on the die's gates, for clips of up to `SAMPLES` samples, given a clip of
/// `length`: the scores of three tokens, which must be the exact machine's.
fn on_gates(length: usize) {
    let dir = toy(&format!("gates-{length}"));
    let cycles = gates_score_as_the_exact_machine(&dir, SAMPLES, &test_clip(length), VOCAB, &[1, 7, 12]);
    eprintln!("toy Moonshine in gates, a clip of {length} samples: {cycles} cycles");
    std::fs::remove_dir_all(&dir).ok();
}

/// The checkpoint in `dir` on a die's gates, for clips of up to `longest` samples, given
/// `clip`: the scores of `tokens` (of a vocabulary of `vocab`) must be the exact machine's,
/// and the microprogram's count of cycles the gates'. Returns the cycles.
fn gates_score_as_the_exact_machine(dir: &Path, longest: usize, clip: &[f32], vocab: usize, tokens: &[u32]) -> u64 {
    use netlist::demo::read_block_value;
    use netlist::{die_design, DieConfig, Simulator};
    let samples = longest;
    let length = clip.len();
    let bits_for = |count: u64| (u64::BITS - count.max(2).saturating_sub(1).leading_zeros()).max(1);
    // Cells as deep as the weights need, each a mantissa of the vectors' width, as many
    // registers as the routines reach, and key columns for every row of the clip.
    let base = cell_die_config();
    let roomy = DieConfig { microcode_bits: 15, reg_bits: 12, index_bits: 24, cell: netlist::CellConfig { addr_bits: 24, weight_bits: base.cell.activation_bits, ..base.cell }, features: base.features.with_key_shift(6), ..base };
    let places = 32;
    let mut probe = MoonshineChip::new(dir, samples, roomy, Some(places)).unwrap();
    probe.clip(length).unwrap();
    let (program, layout, _) = probe.parts();
    let (depth, groups) = layout.cell_depths();
    let addr_bits = bits_for(depth.max(2 * groups));
    let index_bits = bits_for(program.buffers.iter().map(|b| *b as u64).max().unwrap_or(1)).max(addr_bits);
    let sized = DieConfig { index_bits, cell: netlist::CellConfig { addr_bits, ..roomy.cell }, ..roomy };
    // Every routine the host runs, in order, with what it writes before some: each row of
    // samples, and the clip's `1 / count` before the encoder.
    type Write = Option<(u32, sim::moonshine_chip::Vector)>;
    let schedule = |chip: &MoonshineChip| {
        let (register, rows) = chip.sample_rows(clip).unwrap();
        let mut calls: Vec<(Write, sim::moonshine_chip::Call)> = rows.into_iter().enumerate().map(|(r, row)| (Some((register, row)), chip.store_samples_call(r))).collect();
        let mut encoder = chip.encoder_calls().into_iter();
        calls.push((Some(chip.inverse_write()), encoder.next().unwrap()));
        calls.extend(encoder.map(|c| (None, c)));
        for (position, token) in tokens.iter().enumerate() {
            calls.extend(chip.token_calls(*token, position).into_iter().map(|c| (None, c)));
        }
        calls
    };
    let calls = schedule(&probe);
    let routines = probe.microprogram(&calls.iter().map(|(_, c)| c.clone()).collect::<Vec<_>>()).unwrap().0;
    let only: Vec<sim::moonshine_chip::Call> = calls.iter().map(|(_, c)| c.clone()).collect();
    let microprogram = |chip: &MoonshineChip| chip.microprogram(&only).unwrap().1;
    // The routines assembled once on a die with room, to know how many registers they reach.
    let wide = MoonshineChip::new(dir, samples, sized, Some(places)).unwrap();
    microprogram(&wide);
    let die = sim::chip::with_registers_needed(sized, wide.parts().1);
    let mut chip = MoonshineChip::new(dir, samples, die, Some(places)).unwrap();

    // The exact machine's scores after each token.
    chip.encode(clip).unwrap();
    let exact: Vec<Vec<i64>> = tokens.iter().enumerate().map(|(position, token)| chip.step(*token, position).unwrap().1).collect();

    let code = microprogram(&chip);
    let (_, layout, weights) = chip.parts();
    let mut sim = Simulator::new(&die_design(die)).unwrap();
    load_microprogram(&mut sim, &die, &code);
    load_assembly(&mut sim, &die, &layout.setup().initial);
    sim.set("load_fn", 1).unwrap();
    for (address, value) in netlist::block::tables(&die).iter().enumerate() {
        sim.set("fn_addr", address as u64).unwrap();
        sim.set("cell_data", *value as u64 & mask(&die)).unwrap();
        sim.tick();
    }
    sim.set("load_fn", 0).unwrap();
    let data_mask = (1u64 << die.scale_bits.max(die.cell.weight_bits).max(die.cell.activation_bits)) - 1;
    let put = |sim: &mut Simulator, address: u64, value: i64| {
        sim.set("cell_addr", address).unwrap();
        sim.set("cell_data", value as u64 & data_mask).unwrap();
        sim.tick();
    };
    for cell in 0..die.cells {
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
    let exponents = netlist::block::exponent_register(&die);
    let logits = chip.logits_register();
    let (mut cycles, mut gates) = (0, Vec::new());
    for (write, call) in &calls {
        if let Some((register, (mantissas, exponent))) = write {
            // The host writes while the die is idle, as the firmware does.
            let mut values: std::collections::HashMap<(u32, u32), i64> = mantissas.iter().enumerate().map(|(i, m)| ((*register, i as u32), *m)).collect();
            values.insert((exponents, *register), *exponent);
            load_assembly(&mut sim, &die, &values);
        }
        let index = routines.iter().position(|(r, _)| *r == call.routine).unwrap();
        cycles += enter(&mut sim, &code, index, call.token, call.position);
        if call.read {
            gates.push((0..vocab as u32).map(|i| read_block_value(&mut sim, die, logits, i).0).collect::<Vec<i64>>());
        }
    }
    eprintln!("a clip of {length} samples: {} routines, {cycles} cycles", calls.len());
    assert_eq!(gates, exact, "scores");
    // The microprogram's own count is the gates' cycle for cycle, which a system's sizing uses.
    assert_eq!(chip.cycles(&only).unwrap(), cycles);
    let stage = chip.stage(1, 1.0, 1e9).unwrap();
    assert!(stage.output_cycles > 0 && stage.first_cycles + 2 * stage.output_cycles >= cycles / 2 && stage.first_cycles < cycles);
    cycles
}

#[test]
#[ignore = "slow: the toy Moonshine on the die's gates"]
fn moonshine_on_the_die_s_gates_gives_the_exact_machine_s_scores() {
    on_gates(SAMPLES);
}

#[test]
#[ignore = "slow: the toy Moonshine on the die's gates, a shorter clip"]
fn a_chip_for_longer_clips_takes_a_shorter_one_on_its_gates() {
    on_gates(1800);
}

/// A chip built for clips of up to `SAMPLES` scores a shorter clip as one built for exactly
/// that clip does, bit for bit: the stores are longer, the rows run are the clip's.
#[test]
fn a_chip_for_longer_clips_scores_a_shorter_one_as_one_built_for_it() {
    let dir = toy("lengths");
    let clip = test_clip(1800);
    assert_eq!(MoonshineConfig::frames(1800), 3);
    let mut long = MoonshineChip::exact(&dir, SAMPLES, 16).unwrap();
    let mut exact = MoonshineChip::exact(&dir, 1800, 16).unwrap();
    long.encode(&clip).unwrap();
    exact.encode(&clip).unwrap();
    for (position, token) in [1u32, 7, 12].iter().enumerate() {
        assert_eq!(long.step(*token, position).unwrap().1, exact.step(*token, position).unwrap().1, "token {position}");
    }
    // And the same chip goes back to the longest clip.
    let longest = test_clip(SAMPLES);
    long.encode(&longest).unwrap();
    let mut again = MoonshineChip::exact(&dir, SAMPLES, 16).unwrap();
    again.encode(&longest).unwrap();
    assert_eq!(long.step(1, 0).unwrap().1, again.step(1, 0).unwrap().1);
    std::fs::remove_dir_all(&dir).ok();
}

/// Moonshine-tiny's own first encoder and decoder layers, its stem and the first `vocab`
/// rows of its vocabulary, written to `dst` as a checkpoint of one layer each.
fn first_layers_of(src: &Path, dst: &Path, vocab: usize) {
    let mut config: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(src.join("config.json")).unwrap()).unwrap();
    config["encoder_num_hidden_layers"] = 1.into();
    config["decoder_num_hidden_layers"] = 1.into();
    config["vocab_size"] = vocab.into();
    // Positions enough for the test's frames and tokens, so the key columns stay short.
    config["max_position_embeddings"] = 32.into();
    std::fs::write(dst.join("config.json"), config.to_string()).unwrap();
    let bytes = std::fs::read(src.join("model.safetensors")).unwrap();
    let length = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let header: serde_json::Map<String, serde_json::Value> = serde_json::from_slice(&bytes[8..8 + length]).unwrap();
    let data = &bytes[8 + length..];
    let mut w = Writer::default();
    for (name, entry) in &header {
        if name == "__metadata__" || (name.contains(".layers.") && !name.contains(".layers.0.")) {
            continue;
        }
        let mut shape: Vec<usize> = serde_json::from_value(entry["shape"].clone()).unwrap();
        let [start, end]: [usize; 2] = serde_json::from_value(entry["data_offsets"].clone()).unwrap();
        let mut values = data[start..end].to_vec();
        if name == "model.decoder.embed_tokens.weight" {
            values.truncate(vocab * shape[1] * 4);
            shape[0] = vocab;
        }
        w.tensor(name, entry["dtype"].as_str().unwrap(), &shape, values);
    }
    w.write(&dst.join("model.safetensors"));
}

/// One layer of the real Moonshine-tiny on the die's gates, a spot check of the circuit
/// with the model's own weights (288 wide, 8 heads, 1,152 inner): the stem, its first
/// encoder and decoder layers and 64 rows of its vocabulary, a clip of 1,279 samples (two
/// frames) and two tokens, with `LLM2CHIP_REAL_MOONSHINE=1`. The model is read from `MOONSHINE` or
/// `~/.llm2chip/models/UsefulSensors--moonshine-tiny`.
#[test]
#[ignore = "very slow, and needs Moonshine-tiny: one real layer on the die's gates; with LLM2CHIP_REAL_MOONSHINE=1"]
fn one_real_moonshine_layer_on_the_die_s_gates_gives_the_exact_machine_s_scores() {
    // An hour or more of simulation: only when asked for.
    if std::env::var_os("LLM2CHIP_REAL_MOONSHINE").is_none() {
        eprintln!("skipped: LLM2CHIP_REAL_MOONSHINE=1 to run a real layer of Moonshine-tiny in gates");
        return;
    }
    let src = std::env::var("MOONSHINE").map(std::path::PathBuf::from).unwrap_or_else(|_| Path::new(&std::env::var("HOME").unwrap()).join(".llm2chip/models/UsefulSensors--moonshine-tiny"));
    if !src.join("model.safetensors").exists() {
        eprintln!("no Moonshine-tiny at {}: skipped", src.display());
        return;
    }
    let dir = std::env::temp_dir().join(format!("llm2chip-moonshine-real-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    first_layers_of(&src, &dir, 64);
    let samples = 1279;
    assert_eq!(MoonshineConfig::frames(samples), 2);
    let cycles = gates_score_as_the_exact_machine(&dir, samples, &test_clip(samples), 64, &[1, 7]);
    eprintln!("one layer of Moonshine-tiny in gates: {cycles} cycles");
    std::fs::remove_dir_all(&dir).ok();
}

/// The schedule the firmware follows runs the routines the host would, in the same order,
/// for clips of any length the chip takes.
#[test]
fn the_firmware_s_schedule_runs_what_the_host_runs() {
    use netlist::schedule::{expand, words, Driver};
    struct Script {
        input: std::collections::VecDeque<u32>,
        runs: Vec<(u32, u32, u32)>,
        picks: Vec<u32>,
        end: u32,
        sent: Vec<u32>,
    }
    impl Driver for Script {
        fn receive(&mut self) -> u32 {
            self.input.pop_front().expect("the host has sent everything")
        }
        fn write(&mut self, _: u32, _: u32) {}
        fn read(&mut self, _: u32) -> u32 {
            0
        }
        fn run(&mut self, routine: u32, token: u32, position: u32) {
            self.runs.push((routine, token, position));
        }
        fn sample(&mut self) -> u32 {
            self.picks.remove(0)
        }
        fn stops(&self, token: u32) -> bool {
            token == self.end
        }
        fn send(&mut self, token: u32) {
            self.sent.push(token);
        }
    }
    let dir = toy("schedule");
    // Long enough for three tokens: six a second.
    let longest = 6000;
    let mut chip = MoonshineChip::exact(&dir, longest, 16).unwrap();
    let (routines, _, schedule) = chip.schedule().unwrap();
    for length in [1279, 1800, SAMPLES, longest] {
        let clip = test_clip(length);
        chip.clip(length).unwrap();
        let input = chip.schedule_input(&clip).unwrap();
        let picks = vec![5, 9, chip.config.end];
        let mut script = Script { input: input.into(), runs: Vec::new(), picks: picks.clone(), end: chip.config.end, sent: Vec::new() };
        expand(&words(&schedule), &mut script).unwrap();
        assert!(script.input.is_empty(), "every word the host sends is read");
        let rows = sim::moonshine_chip::Lengths::of(length).unwrap().sample_rows;
        let mut calls: Vec<sim::moonshine_chip::Call> = (0..rows).map(|r| chip.store_samples_call(r)).collect();
        calls.extend(chip.encoder_calls());
        // As many tokens as the clip's limit allows, up to the end token.
        let limit = chip.limit().min(3);
        for (position, token) in [chip.config.start, 5, 9].into_iter().enumerate().take(limit) {
            calls.extend(chip.token_calls(token, position));
        }
        let expected: Vec<(u32, u32, u32)> = calls.iter().map(|c| (routines.iter().position(|(r, _)| *r == c.routine).unwrap() as u32, c.token, c.position)).collect();
        assert_eq!(script.runs, expected, "a clip of {length} samples");
        assert_eq!(script.sent, picks[..limit.min(2)]);
    }
    std::fs::remove_dir_all(&dir).ok();
}

/// The chip's own firmware transcribes a clip by itself: the host sends the clip's length,
/// `1 / count` and the rows of samples, and the firmware, at instruction level, follows the
/// schedule, driving a die run by the exact machine, and sends back the same tokens as the
/// host ordering every routine.
#[test]
fn the_firmware_transcribes_a_clip_as_the_host_ordering_the_routines() {
    use netlist::chip::{add_schedule, chip_info, ChipModel, GenerationSetup};
    use netlist::image::build_image;
    use netlist::rv32::Rv32Iss;
    use netlist::sampler::{exponential_table, SamplerConfig, SamplerModel};
    use netlist::soc::{SocBus, SocConfig};
    let Some(firmware) = firmware() else { return };
    let dir = toy("firmware");
    // Long enough for a few tokens, six a second.
    let longest = 8000;
    let base = cell_die_config();
    let die = netlist::DieConfig { microcode_bits: 15, reg_bits: 12, index_bits: 20, cell: netlist::CellConfig { addr_bits: 20, weight_bits: base.cell.activation_bits, ..base.cell }, features: base.features.with_host_writes(true).with_key_shift(7), ..base };
    let clip = test_clip(longest);

    let mut host = MoonshineChip::new(&dir, longest, die, Some(32)).unwrap();
    let expected = host.transcribe(&clip).unwrap();

    let mut chip = MoonshineChip::new(&dir, longest, die, Some(32)).unwrap();
    let input: Vec<u8> = chip.schedule_input(&clip).unwrap().iter().flat_map(|w| w.to_le_bytes()).collect();
    let (routines, code, schedule) = chip.schedule().unwrap();
    let end = chip.config.end;
    let setup = GenerationSetup { max_new: 0, greedy: true, limit: 1, temperature: 256, seed: 1, raw_tokens: true, stop: vec![end], draft_depth: 0, draft_logits: 0, think: None, think_budget: u32::MAX };
    let mut info = chip_info(&die, &code, chip.logits_register(), chip.config.vocab as u32, &setup);
    add_schedule(&mut info, &schedule);
    eprintln!("the chip's description: {} words, {} of them the schedule's {} records", info.len(), schedule.len() * netlist::schedule::RECORD_WORDS, schedule.len());
    let image = build_image(&firmware.program, &firmware.data, &info, &[], &[]).unwrap();
    let sampler = SamplerModel { config: SamplerConfig { logit_bits: 16, index_bits: 12, entries: 4, lanes: 1 }, rng: 0, table: exponential_table() };
    let count = routines.len();
    let model = ChipModel::new(chip.die_engine(&routines, code.clone()), code, sampler);
    let mut bus = SocBus::booted_with(SocConfig { program_bits: 13, data_bits: 14, fifo_bits: 6 }, &image, &input, model).unwrap();
    let mut cpu = Rv32Iss::default();
    while !cpu.halted {
        cpu.step(&mut bus);
        assert!(cpu.cycle < 200_000_000, "the firmware does not finish");
    }
    let start = bus.tx.iter().position(|b| *b != b'.').unwrap_or(bus.tx.len());
    let tokens: Vec<u32> = bus.tx[start..].chunks(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    eprintln!("{} cycles of the CPU, {} of the die, tokens {tokens:?}", cpu.cycle, bus.peripheral.die_busy);
    assert_eq!(tokens, expected);
    assert!(bus.peripheral.routines[..count].iter().all(|(runs, _)| *runs > 0), "every routine ran: {:?}", bus.peripheral.routines);
    std::fs::remove_dir_all(&dir).ok();
}
