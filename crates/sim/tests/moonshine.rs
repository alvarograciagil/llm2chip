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

#[test]
#[ignore = "slow: the toy Moonshine on the die's gates"]
fn moonshine_on_the_die_s_gates_gives_the_exact_machine_s_scores() {
    use netlist::demo::read_block_value;
    use netlist::{die_design, DieConfig, Simulator};
    let dir = toy("gates");
    let clip = test_clip(SAMPLES);
    let tokens = [1u32, 7, 12];
    let bits_for = |count: u64| (u64::BITS - count.max(2).saturating_sub(1).leading_zeros()).max(1);
    // Cells as deep as the weights need, each a mantissa of the vectors' width, as many
    // registers as the routines reach, and key columns for every row of the clip.
    let base = cell_die_config();
    let roomy = DieConfig { microcode_bits: 15, reg_bits: 12, index_bits: 20, cell: netlist::CellConfig { addr_bits: 20, weight_bits: base.cell.activation_bits, ..base.cell }, features: base.features.with_key_shift(6), ..base };
    let places = 32;
    let probe = MoonshineChip::new(&dir, SAMPLES, roomy, Some(places)).unwrap();
    let (program, layout, _) = probe.parts();
    let (depth, groups) = layout.cell_depths();
    let addr_bits = bits_for(depth.max(2 * groups));
    let index_bits = bits_for(program.buffers.iter().map(|b| *b as u64).max().unwrap_or(1)).max(addr_bits);
    let sized = DieConfig { index_bits, cell: netlist::CellConfig { addr_bits, ..roomy.cell }, ..roomy };
    // Every routine the host runs, in order, with the samples it writes before some.
    let schedule = |chip: &MoonshineChip| {
        let (_, rows) = chip.sample_rows(&clip).unwrap();
        let mut calls: Vec<(Option<sim::moonshine_chip::Vector>, sim::moonshine_chip::Call)> = rows.into_iter().enumerate().map(|(r, row)| (Some(row), chip.store_samples_call(r))).collect();
        calls.extend(chip.encoder_calls().into_iter().map(|c| (None, c)));
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
    let wide = MoonshineChip::new(&dir, SAMPLES, sized, Some(places)).unwrap();
    microprogram(&wide);
    let die = sim::chip::with_registers_needed(sized, wide.parts().1);
    let mut chip = MoonshineChip::new(&dir, SAMPLES, die, Some(places)).unwrap();
    let (register, _) = chip.sample_rows(&clip).unwrap();

    // The exact machine's scores after each token.
    chip.encode(&clip).unwrap();
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
        if let Some((mantissas, exponent)) = write {
            let exponent = *exponent;
            // The host writes the samples while the die is idle, as the firmware does.
            let mut values: std::collections::HashMap<(u32, u32), i64> = mantissas.iter().enumerate().map(|(i, m)| ((register, i as u32), *m)).collect();
            values.insert((exponents, register), exponent);
            load_assembly(&mut sim, &die, &values);
        }
        let index = routines.iter().position(|(r, _)| *r == call.routine).unwrap();
        cycles += enter(&mut sim, &code, index, call.token, call.position);
        if call.read {
            gates.push((0..VOCAB as u32).map(|i| read_block_value(&mut sim, die, logits, i).0).collect::<Vec<i64>>());
        }
    }
    eprintln!("toy Moonshine in gates: {} routines, {cycles} cycles", calls.len());
    assert_eq!(gates, exact, "scores");
    // The microprogram's own count is the gates' cycle for cycle, which a system's sizing uses.
    assert_eq!(chip.cycles(&only).unwrap(), cycles);
    let stage = chip.stage(1, 1.0, 1e9).unwrap();
    assert!(stage.output_cycles > 0 && stage.first_cycles + 2 * stage.output_cycles >= cycles / 2 && stage.first_cycles < cycles);
    std::fs::remove_dir_all(&dir).ok();
}
