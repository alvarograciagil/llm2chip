//! Streaming Moonshine taken to the chip on a toy checkpoint: the exact machine following
//! the schedule against the floating-point reference, and the firmware following it at
//! instruction level against the host.
#![cfg(feature = "listen")]

mod common;
use common::*;
#[allow(unused_imports)]
use common::{cell_die_config, enter, load_assembly, load_microprogram, mask};
use sim::listen::moonshine::{window, Moonshine, MoonshineConfig};
use sim::listen::moonshine_stream_chip::StreamingChip;
use std::path::Path;

/// Width 16 in 2 heads of 8 (6 values rotated in the decoder), 3 encoder layers seeing
/// 1, 0 and 1 frames ahead within 4, 2 decoder layers, 24 tokens.
const HIDDEN: usize = 16;
const INTERMEDIATE: usize = 32;
const VOCAB: usize = 24;
const FRAMES: usize = 20;

fn make_checkpoint(dir: &Path) {
    make_checkpoint_with(dir, &[[4, 2], [4, 0], [4, 2]]);
}

fn make_checkpoint_with(dir: &Path, windows: &[[usize; 2]]) {
    let config = serde_json::json!({
        "model_type": "moonshine_streaming", "hidden_size": HIDDEN, "intermediate_size": INTERMEDIATE,
        "num_hidden_layers": 2, "num_attention_heads": 2, "num_key_value_heads": 2, "head_dim": 8,
        "encoder_hidden_size": HIDDEN, "attention_bias": false, "vocab_size": VOCAB, "max_position_embeddings": 64,
        "rope_parameters": {"partial_rotary_factor": 0.75, "rope_theta": 100.0},
        "decoder_start_token_id": 1, "bos_token_id": 1, "eos_token_id": 2,
        "encoder_config": {
            "hidden_size": HIDDEN, "intermediate_size": INTERMEDIATE, "num_hidden_layers": windows.len(), "num_attention_heads": 2,
            "num_key_value_heads": 2, "head_dim": 8, "frame_ms": 5.0, "sample_rate": 16000,
            "sliding_windows": windows
        }
    });
    std::fs::write(dir.join("config.json"), serde_json::to_string_pretty(&config).unwrap()).unwrap();
    let mut rng = Rng(0x5EED_57EA);
    let mut w = Writer::default();
    let mut put = |w: &mut Writer, name: &str, shape: &[usize], spread: f32, offset: f32| {
        let values: Vec<f32> = (0..shape.iter().product()).map(|_| offset + spread * rng.unit()).collect();
        w.tensor(name, "F32", shape, values.iter().flat_map(|v| v.to_le_bytes()).collect());
    };
    let h = HIDDEN;
    let e = |n: &str| format!("model.encoder.embedder.{n}");
    put(&mut w, &e("comp.log_k"), &[], 0.0, 0.75f32.ln());
    put(&mut w, &e("linear.weight"), &[h, 80], 0.2, 0.0);
    put(&mut w, &e("conv1.weight"), &[2 * h, h, 5], 0.15, 0.0);
    put(&mut w, &e("conv1.bias"), &[2 * h], 0.1, 0.0);
    put(&mut w, &e("conv2.weight"), &[h, 2 * h, 5], 0.1, 0.0);
    put(&mut w, &e("conv2.bias"), &[h], 0.1, 0.0);
    for layer in 0..windows.len() {
        let l = |n: &str| format!("model.encoder.layers.{layer}.{n}");
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"] {
            put(&mut w, &l(&format!("self_attn.{proj}.weight")), &[h, h], 0.3, 0.0);
        }
        put(&mut w, &l("input_layernorm.gamma"), &[h], 0.2, 0.0);
        put(&mut w, &l("post_attention_layernorm.gamma"), &[h], 0.2, 0.0);
        put(&mut w, &l("mlp.fc1.weight"), &[INTERMEDIATE, h], 0.3, 0.0);
        put(&mut w, &l("mlp.fc1.bias"), &[INTERMEDIATE], 0.1, 0.0);
        put(&mut w, &l("mlp.fc2.weight"), &[h, INTERMEDIATE], 0.2, 0.0);
        put(&mut w, &l("mlp.fc2.bias"), &[h], 0.1, 0.0);
    }
    put(&mut w, "model.encoder.final_norm.gamma", &[h], 0.2, 0.0);
    put(&mut w, "model.decoder.embed_tokens.weight", &[VOCAB, h], 0.5, 0.0);
    put(&mut w, "model.decoder.pos_emb.weight", &[64, h], 0.3, 0.0);
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
    put(&mut w, "proj_out.weight", &[VOCAB, h], 0.5, 0.0);
    w.write(&dir.join("model.safetensors"));
}

/// A clip with a few tones and some noise, the same every time.
fn test_clip(samples: usize) -> Vec<f32> {
    let mut rng = Rng(0xA0D11);
    (0..samples).map(|i| {
        let t = i as f32 / 16_000.0;
        0.4 * (t * 440.0 * std::f32::consts::TAU).sin() + 0.2 * (t * 1250.0 * std::f32::consts::TAU).sin() + 0.05 * rng.unit()
    }).collect()
}

fn toy(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("llm2chip-moonshine-stream-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    make_checkpoint(&dir);
    dir
}

#[test]
fn a_streaming_window_sees_its_frames_and_no_more() {
    assert_eq!(window(10, 30, (16, 4)), 0..14);
    assert_eq!(window(20, 30, (16, 4)), 5..24);
    assert_eq!(window(28, 30, (16, 4)), 13..30);
    assert_eq!(window(20, 30, (16, 0)), 5..21);
    let config = MoonshineConfig::read(&toy("config")).unwrap();
    assert_eq!(config.frames_of(6400 + 100), FRAMES);
    assert_eq!(config.streaming.unwrap().ahead(0), 1);
}

/// The exact machine, following the schedule as the firmware does, writes the text the
/// floating-point reference would from the same tokens, its scores close to the reference's.
#[test]
fn streaming_moonshine_in_the_die_s_arithmetic_scores_as_floating_point_does() {
    let dir = toy("exact");
    let clip = test_clip(FRAMES * 320);
    let reference = Moonshine::open(&dir).unwrap();
    let frames = reference.encode(&clip).unwrap();
    assert_eq!(frames.len(), FRAMES);
    let mut chip = StreamingChip::exact(&dir, FRAMES, 16).unwrap();
    let (tokens, scores) = chip.transcribe(&clip).unwrap();
    let mut decoder = reference.decoder(&frames).unwrap();
    let mut token = chip.config.start;
    for (step, got) in scores.iter().enumerate() {
        let expected = decoder.step(token).unwrap();
        let largest = expected.iter().fold(0f32, |m, v| m.max(v.abs()));
        let worst = got.iter().zip(&expected).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        eprintln!("step {step}: scores within {:.3} %", worst / largest * 100.0);
        assert!(worst / largest < 0.03, "step {step}: {worst} of {largest}");
        if let Some(next) = tokens.get(step) {
            token = *next;
        }
    }
    std::fs::remove_dir_all(&dir).ok();
}

/// The chip's own firmware, at instruction level, listens as the host sends each 20 ms and
/// writes the same tokens as the schedule followed on the host.
#[test]
fn the_firmware_listens_while_the_speech_comes() {
    use netlist::chip::{add_schedule, chip_info, ChipModel, GenerationSetup};
    use netlist::image::build_image;
    use netlist::rv32::Rv32Iss;
    use netlist::sampler::{exponential_table, SamplerConfig, SamplerModel};
    use netlist::soc::{SocBus, SocConfig};
    let Some(firmware) = firmware() else { return };
    let dir = toy("firmware");
    // Long enough for a few tokens, six a second.
    let frames = 60;
    let base = cell_die_config();
    let die = netlist::DieConfig { microcode_bits: 15, reg_bits: 12, index_bits: 20, cell: netlist::CellConfig { addr_bits: 20, weight_bits: base.cell.activation_bits, ..base.cell }, features: base.features.with_host_writes(true).with_key_shift(8), ..base };
    let clip = test_clip(frames * 320);
    let (expected, _) = StreamingChip::new(&dir, frames, die, Some(32)).unwrap().transcribe(&clip).unwrap();

    let mut chip = StreamingChip::new(&dir, frames, die, Some(32)).unwrap();
    let input: Vec<u8> = chip.schedule_input(&clip).unwrap().iter().flat_map(|w| w.to_le_bytes()).collect();
    let schedule = chip.schedule().unwrap();
    let (routines, code) = chip.routines().unwrap();
    let setup = GenerationSetup { max_new: 0, greedy: true, limit: 1, temperature: 256, seed: 1, raw_tokens: true, stop: vec![chip.config.end], draft_depth: 0, draft_logits: 0, think: None, think_budget: u32::MAX };
    let mut info = chip_info(&die, &code, chip.logits_register(), chip.config.vocab as u32, &setup);
    let words = add_schedule(&mut info, &schedule);
    eprintln!("the chip's description: {} words, and a schedule of {} records", info.len(), schedule.len());
    let image = build_image(&firmware.program, &firmware.data, &info, &words, &[]).unwrap();
    let sampler = SamplerModel { config: SamplerConfig { logit_bits: 16, index_bits: 12, entries: 4, lanes: 1 }, rng: 0, table: exponential_table() };
    let model = ChipModel::new(chip.die_engine(&routines, code.clone()), code, sampler);
    let mut bus = SocBus::booted_with(SocConfig { program_bits: 13, data_bits: 14, fifo_bits: 6 }, &image, &input, model).unwrap();
    let mut cpu = Rv32Iss::default();
    while !cpu.halted {
        cpu.step(&mut bus);
        assert!(cpu.cycle < 400_000_000, "the firmware does not finish");
    }
    let start = bus.tx.iter().position(|b| *b != b'.').unwrap_or(bus.tx.len());
    let tokens: Vec<u32> = bus.tx[start..].chunks(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    eprintln!("{} cycles of the CPU, {} of the die, tokens {tokens:?}", cpu.cycle, bus.peripheral.die_busy);
    assert_eq!(tokens, expected);
    std::fs::remove_dir_all(&dir).ok();
}

/// Every kind of window: none ahead, some ahead, one wider than the clip, and no layers.
#[test]
fn every_kind_of_window_scores_as_floating_point_does() {
    for windows in [vec![], vec![[4, 0]], vec![[4, 2]], vec![[30, 0]], vec![[3, 3], [5, 1]]] {
        let dir = std::env::temp_dir().join(format!("llm2chip-moonshine-stream-windows-{}-{}", windows.len(), std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        make_checkpoint_with(&dir, &windows);
        let clip = test_clip(FRAMES * 320);
        let reference = Moonshine::open(&dir).unwrap();
        let frames = reference.encode(&clip).unwrap();
        let (_, scores) = StreamingChip::exact(&dir, FRAMES, 16).unwrap().transcribe(&clip).unwrap();
        let expected = reference.decoder(&frames).unwrap().step(1).unwrap();
        let largest = expected.iter().fold(0f32, |m, v| m.max(v.abs()));
        let worst = scores[0].iter().zip(&expected).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        eprintln!("windows {windows:?}: first scores within {:.3} %", worst / largest * 100.0);
        assert!(worst / largest < 0.01, "windows {windows:?}");
        std::fs::remove_dir_all(&dir).ok();
    }
}

/// The toy on the die's gates, run in step with the exact machine as the schedule orders
/// them: the scores of every token the same, bit for bit, and the microprogram's cycles the
/// gates'.
#[test]
#[ignore = "slow: the toy streaming Moonshine on the die's gates"]
fn streaming_moonshine_on_the_die_s_gates_gives_the_exact_machine_s_scores() {
    use netlist::demo::read_block_value;
    use netlist::schedule::{expand, words, Driver};
    use netlist::{die_design, DieConfig, Simulator};
    let dir = toy("gates");
    let clip = test_clip(FRAMES * 320);
    let bits_for = |count: u64| (u64::BITS - count.max(2).saturating_sub(1).leading_zeros()).max(1);
    let base = cell_die_config();
    let roomy = DieConfig { microcode_bits: 15, reg_bits: 12, index_bits: 20, cell: netlist::CellConfig { addr_bits: 20, weight_bits: base.cell.activation_bits, ..base.cell }, features: base.features.with_key_shift(7), function_select_bits: 4, ..base };
    let probe = StreamingChip::new(&dir, FRAMES, roomy, Some(32)).unwrap();
    let (program, layout, _) = probe.parts();
    let (depth, groups) = layout.cell_depths();
    let addr_bits = bits_for(depth.max(2 * groups));
    let index_bits = bits_for(program.buffers.iter().map(|b| *b as u64).max().unwrap_or(1)).max(addr_bits);
    let sized = DieConfig { index_bits, cell: netlist::CellConfig { addr_bits, ..roomy.cell }, ..roomy };
    let wide = StreamingChip::new(&dir, FRAMES, sized, Some(32)).unwrap();
    wide.routines().unwrap();
    let die = sim::chip::with_registers_needed(sized, wide.parts().1);
    let mut chip = StreamingChip::new(&dir, FRAMES, die, Some(32)).unwrap();
    let schedule = chip.schedule().unwrap();
    let input = chip.schedule_input(&clip).unwrap();
    let (routines, code) = chip.routines().unwrap();
    let (vocab, end, logits) = (chip.config.vocab as u32, chip.config.end, chip.logits_register());

    let mut sim = Simulator::new(&die_design(die)).unwrap();
    {
        let (_, layout, weights) = chip.parts();
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
    }

    // The schedule followed by the exact machine and the gates at once.
    struct Both<'a, E: netlist::chip::DieEngine> {
        exact: E,
        sim: &'a mut Simulator,
        die: DieConfig,
        code: &'a netlist::Microprogram,
        input: std::collections::VecDeque<u32>,
        writes: std::collections::HashMap<(u32, u32), i64>,
        logits: u32,
        vocab: u32,
        end: u32,
        cycles: u64,
        tokens: usize,
    }
    impl<E: netlist::chip::DieEngine> Driver for Both<'_, E> {
        fn receive(&mut self) -> u32 {
            self.input.pop_front().unwrap_or(0)
        }
        fn write(&mut self, address: u32, value: u32) {
            self.exact.write(address, value as i32 as i64);
            self.writes.insert((address >> self.die.index_bits, address & ((1 << self.die.index_bits) - 1)), value as i32 as i64);
        }
        fn read(&mut self, address: u32) -> u32 {
            self.exact.read(address) as u32
        }
        fn run(&mut self, routine: u32, token: u32, position: u32) {
            // The host writes while the die is idle, as the firmware does.
            if !self.writes.is_empty() {
                load_assembly(self.sim, &self.die, &std::mem::take(&mut self.writes));
            }
            self.exact.run(routine as usize, token, position);
            self.cycles += enter(self.sim, self.code, routine as usize, token, position);
        }
        fn sample(&mut self) -> u32 {
            let exact: Vec<i64> = (0..self.vocab).map(|i| self.exact.read(self.logits << self.die.index_bits | i)).collect();
            let gates: Vec<i64> = (0..self.vocab).map(|i| read_block_value(self.sim, self.die, self.logits, i).0).collect();
            assert_eq!(gates, exact, "token {}: the scores", self.tokens);
            self.tokens += 1;
            exact.iter().enumerate().fold((0, i64::MIN), |best, (i, m)| if *m > best.1 { (i, *m) } else { best }).0 as u32
        }
        fn stops(&self, token: u32) -> bool {
            token == self.end
        }
        fn send(&mut self, _: u32) {}
    }
    let exact = chip.die_engine(&routines, code.clone());
    let mut both = Both { exact, sim: &mut sim, die, code: &code, input: input.into(), writes: Default::default(), logits, vocab, end, cycles: 0, tokens: 0 };
    expand(&words(&schedule), &mut both).unwrap();
    eprintln!("toy streaming Moonshine in gates: {} tokens scored, {} cycles", both.tokens, both.cycles);
    assert!(both.tokens > 0);
    std::fs::remove_dir_all(&dir).ok();
}
