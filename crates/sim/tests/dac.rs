//! DAC's decoder, codes to sound, taken to the chip on toy checkpoints: the die's exact
//! arithmetic against the floating-point reference of `sim::dac`, and the die's gates
//! against the exact machine.
#![cfg(feature = "speak")]

mod common;
#[allow(unused_imports)]
use common::{cell_die_config, enter, load_assembly, load_microprogram, mask, Rng, Writer};
use sim::speak::dac::Dac;
use sim::speak::dac_chip::DacChip;
use std::path::Path;

/// Writes a tensor of random values: `(writer, name, shape, spread, offset)`.
type Put<'a> = dyn FnMut(&mut Writer, &str, &[usize], f32, f32) + 'a;

const LATENT: usize = 8;
const BOOKS: usize = 2;
const BOOK_SIZE: usize = 16;
const BOOK_DIM: usize = 4;

/// A decoder of `width` channels halving at each of `rates`, with random weights large
/// enough for Snake's arguments to go round the sine's period several times.
fn make_toy(dir: &Path, width: usize, rates: &[usize]) {
    let config = serde_json::json!({ "model_type": "dac", "sampling_rate": 24000, "latent_dim": LATENT, "decoder_dim": width, "decoder_rates": rates, "n_codebooks": BOOKS, "codebook_size": BOOK_SIZE, "codebook_dim": BOOK_DIM });
    std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
    let mut rng = Rng(0xDAC0_0000_0001);
    let mut w = Writer::default();
    let mut put = |w: &mut Writer, name: &str, shape: &[usize], spread: f32, offset: f32| {
        let values: Vec<f32> = (0..shape.iter().product()).map(|_| offset + spread * rng.unit()).collect();
        w.tensor(name, "F32", shape, values.iter().flat_map(|v| v.to_le_bytes()).collect());
    };
    let put = &mut put;
    for book in 0..BOOKS {
        let p = |n: &str| format!("quantizer.quantizers.{book}.{n}");
        put(&mut w, &p("codebook.weight"), &[BOOK_SIZE, BOOK_DIM], 1.0, 0.0);
        put(&mut w, &p("out_proj.weight"), &[LATENT, BOOK_DIM, 1], 0.8, 0.0);
        put(&mut w, &p("out_proj.bias"), &[LATENT], 0.2, 0.0);
    }
    fn conv(put: &mut Put, w: &mut Writer, name: &str, out: usize, inp: usize, k: usize, gain: f32) {
        put(w, &format!("{name}.weight"), &[out, inp, k], gain / ((inp * k) as f32).sqrt(), 0.0);
        put(w, &format!("{name}.bias"), &[out], 0.1, 0.0);
    }
    conv(put, &mut w, "decoder.model.0", width, LATENT, 7, 4.0);
    let mut channels = width;
    for (block, s) in rates.iter().enumerate() {
        let p = |n: &str| format!("decoder.model.{}.block.{n}", block + 1);
        put(&mut w, &format!("{}.alpha", p("0")), &[1, channels, 1], 1.0, 1.5);
        put(&mut w, &format!("{}.weight", p("1")), &[channels, channels / 2, 2 * s], 2.0 / ((channels * s) as f32).sqrt(), 0.0);
        put(&mut w, &format!("{}.bias", p("1")), &[channels / 2], 0.1, 0.0);
        channels /= 2;
        for unit in 0..3 {
            let q = |n: &str| p(&format!("{}.block.{n}", unit + 2));
            put(&mut w, &format!("{}.alpha", q("0")), &[1, channels, 1], 1.0, 1.5);
            conv(put, &mut w, &q("1"), channels, channels, 7, 2.0);
            put(&mut w, &format!("{}.alpha", q("2")), &[1, channels, 1], 1.0, 1.5);
            conv(put, &mut w, &q("3"), channels, channels, 1, 1.0);
        }
    }
    let last = rates.len() + 1;
    put(&mut w, &format!("decoder.model.{last}.alpha"), &[1, channels, 1], 1.0, 1.5);
    conv(put, &mut w, &format!("decoder.model.{}", last + 1), 1, channels, 7, 1.0);
    w.write(&dir.join("model.safetensors"));
}

fn toy(tag: &str, width: usize, rates: &[usize]) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("llm2chip-dac-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    make_toy(&dir, width, rates);
    dir
}

fn codes(frames: usize) -> Vec<Vec<u32>> {
    (0..BOOKS).map(|b| (0..frames).map(|f| ((f * 7 + b * 5 + f * f) % BOOK_SIZE) as u32).collect()).collect()
}

/// How far below the signal the difference is, in decibels.
fn below(got: &[f32], expected: &[f32]) -> f32 {
    let noise: f32 = got.iter().zip(expected).map(|(a, b)| (a - b) * (a - b)).sum();
    let signal: f32 = expected.iter().map(|v| v * v).sum();
    10.0 * (signal / noise).log10()
}

#[test]
fn the_die_s_arithmetic_decodes_codes_as_floating_point_does() {
    for (tag, width, rates) in [("plain", 8, vec![]), ("one", 8, vec![2]), ("odd", 8, vec![3, 2])] {
        let dir = toy(tag, width, &rates);
        let dac = Dac::open(&dir).unwrap();
        let codes = codes(5);
        let expected = dac.decode(&codes).unwrap();
        assert_eq!(expected.len(), dac.config.samples(5));
        let reach = dac.snake_reach(&codes).unwrap();
        let mut chip = DacChip::exact(&dir, 5, 20).unwrap();
        let (got, _) = chip.decode(&codes).unwrap();
        let db = below(&got, &expected);
        eprintln!("{tag}: {} samples, |αx| up to {reach:.1}, {db:.1} dB below the signal", got.len());
        // Below about 20-bit vectors Snake's sine, whose argument reaches dozens of
        // radians, loses the fraction of its period that matters: see docs/voice.md.
        assert!(db > 30.0, "{tag}: {db:.1} dB");
        std::fs::remove_dir_all(&dir).ok();
    }
}

#[test]
#[ignore = "slow: the toy DAC decoder on the die's gates"]
fn dac_s_decoder_on_the_die_s_gates_gives_the_exact_machine_s_sound() {
    use netlist::demo::read_block_value;
    use netlist::{die_design, DieConfig, Simulator};
    let dir = toy("gates", 8, &[3, 2]);
    let codes = codes(3);
    let bits_for = |count: u64| (u64::BITS - count.max(2).saturating_sub(1).leading_zeros()).max(1);
    let base = cell_die_config();
    // No attention, but a program over rows reaches as many rows as its key columns hold.
    let roomy = DieConfig { microcode_bits: 15, reg_bits: 12, index_bits: 20, cell: netlist::CellConfig { addr_bits: 20, weight_bits: base.cell.activation_bits, ..base.cell }, features: base.features.with_key_shift(7), ..base };
    let probe = DacChip::new(&dir, 3, roomy, true).unwrap();
    let (program, layout, _) = probe.parts();
    let (depth, groups) = layout.cell_depths();
    let addr_bits = bits_for(depth.max(2 * groups));
    let index_bits = bits_for(program.buffers.iter().map(|b| *b as u64).max().unwrap_or(1)).max(addr_bits);
    let sized = DieConfig { index_bits, cell: netlist::CellConfig { addr_bits, ..roomy.cell }, ..roomy };
    // The routines assembled once on a die with room, to know how many registers they reach.
    let wide = DacChip::new(&dir, 3, sized, true).unwrap();
    wide.microprogram().unwrap();
    let die = sim::chip::with_registers_needed(sized, wide.parts().1);
    // A die built for this microprogram: with the wrapping shift a general die leaves out.
    let mut widths = die.features.widths;
    widths[netlist::die::WRAP_LEFT_FIELD] = 1;
    let die = DieConfig { features: netlist::DieFeatures { widths, ..die.features }, ..die };
    let mut chip = DacChip::new(&dir, 3, die, true).unwrap();
    let (_, exact) = chip.decode(&codes).unwrap();
    let calls = chip.calls(&codes).unwrap();
    let code = chip.microprogram().unwrap();
    let (_, layout, weights) = chip.parts();
    let die = layout.config();
    assert!(netlist::DieFeatures::of(&code).widths[netlist::die::WRAP_LEFT_FIELD] > 0, "the sine's alignment wraps");
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
    let output = chip.output_register();
    let (mut cycles, mut gates) = (0, Vec::new());
    for call in &calls {
        cycles += enter(&mut sim, &code, call.routine, call.token, 0);
        if call.read {
            gates.push(read_block_value(&mut sim, die, output, 0).0);
        }
    }
    eprintln!("toy DAC in gates: {} routines, {cycles} cycles, {} samples", calls.len(), gates.len());
    assert_eq!(gates, exact, "sound");
    assert_eq!(chip.cycles(&calls).unwrap(), cycles);
    std::fs::remove_dir_all(&dir).ok();
}

/// Decoding a window at a time, each window as long as the frames it keeps plus the reach
/// of a frame on both sides, gives the sound decoded whole: in floating point to the last
/// rounding, and in the die's arithmetic bit for bit, since each row keeps its own exponent.
#[test]
fn decoding_window_by_window_gives_the_whole_sound() {
    let dir = toy("windows", 8, &[3, 2]);
    let dac = Dac::open(&dir).unwrap();
    let reach = dac.config.reach_frames();
    let frames = 2 * reach + 9;
    let codes = codes(frames);
    let whole = dac.decode(&codes).unwrap();
    for chunk in [1, 4, 7] {
        let chunked = dac.decode_chunked(&codes, chunk).unwrap();
        assert_eq!(chunked.len(), whole.len());
        let worst = chunked.iter().zip(&whole).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(worst < 1e-5, "chunks of {chunk}: {worst}");
    }
    let (_, whole) = DacChip::exact(&dir, frames, 20).unwrap().decode(&codes).unwrap();
    let (_, chunked) = sim::speak::dac_chip::decode_chunked(&dir, &codes, 4, 20).unwrap();
    assert_eq!(chunked, whole, "the die's mantissas");
    std::fs::remove_dir_all(&dir).ok();
}

/// The chip's own firmware decodes a sound by itself, window by window: the host sends the
/// codes and which samples of each window to keep, and the firmware, at instruction level,
/// follows the schedule, driving a die run by the exact machine, and sends back the same
/// samples as decoding window by window on the host.
#[test]
fn the_firmware_decodes_window_by_window_as_the_host_does() {
    use netlist::chip::{add_schedule, chip_info, ChipModel, GenerationSetup};
    use netlist::image::build_image;
    use netlist::rv32::Rv32Iss;
    use netlist::sampler::{exponential_table, SamplerConfig, SamplerModel};
    use netlist::soc::{SocBus, SocConfig};
    let Some(firmware) = common::firmware() else { return };
    // Routines about as short as the CPU's work for each (8 channels), and longer (64), as
    // the real decoder's are.
    for channels in [8, 64] {
    let dir = toy(&format!("firmware-{channels}"), channels, &[3, 2]);
    let config = sim::speak::dac::DacConfig::read(&dir).unwrap();
    let (reach, chunk) = (config.reach_frames(), 4);
    let frames = 2 * reach + 9;
    let codes = codes(frames);
    let width = config.window_frames(frames, chunk);
    let (_, expected) = sim::speak::dac_chip::decode_chunked(&dir, &codes, chunk, 20).unwrap();

    // Without a command queue before the die, and with one of 16 orders.
    for queue in [0, 4] {
    let base = cell_die_config();
    let die = netlist::DieConfig { microcode_bits: 15, reg_bits: 12, index_bits: 20, cell: netlist::CellConfig { addr_bits: 20, activation_bits: 20, accumulator_bits: 56, weight_bits: 20, ..base.cell }, scale_bits: 24, features: base.features.with_host_writes(true).with_key_shift(7).with_queue_bits(queue), ..base };
    let mut chip = DacChip::new(&dir, width, die, false).unwrap();
    let chip_samples = chip.samples();
    let (code, schedule) = chip.schedule().unwrap();
    let per_window: Vec<u64> = (0..=frames * config.hop()).map(|kept| chip.firmware_cycles(kept, &code)).collect();
    let firmware_per_window = |kept: usize| per_window[kept];
    let setup = GenerationSetup { max_new: 0, greedy: true, limit: 1, temperature: 256, seed: 1, raw_tokens: true, stop: Vec::new(), draft_depth: 0, draft_logits: 0, think: None, think_budget: u32::MAX };
    let mut info = chip_info(&die, &code, chip.output_register(), 1, &setup);
    let schedule_words = add_schedule(&mut info, &schedule);
    eprintln!("the chip's description: {} words, and a schedule of {} records", info.len(), schedule.len());
    let image = build_image(&firmware.program, &firmware.data, &info, &schedule_words, &[]).unwrap();
    let input = netlist::port::stream(&sim::speak::dac_chip::schedule_input(&config, &codes, chunk));
    let sampler = SamplerModel { config: SamplerConfig { logit_bits: 16, index_bits: 12, entries: 4, lanes: 1 }, rng: 0, table: exponential_table() };
    let model = ChipModel::new(chip.die_engine(code.clone()), code, sampler).with_queue(queue);
    let mut bus = SocBus::booted_with(SocConfig { program_words: netlist::soc::PROGRAM_WORDS, data_bits: 14, fifo_bits: 6, links: Default::default() }, &image, &input, model).unwrap();
    let mut cpu = Rv32Iss::default();
    while !cpu.halted {
        cpu.step(&mut bus);
        assert!(cpu.cycle < 400_000_000, "the firmware does not finish");
    }
    let words: Vec<i64> = netlist::port::words(&bus.tx).iter().map(|w| *w as i32 as i64).collect();
    eprintln!("{channels} channels, queue of {}: {} cycles of the CPU, {} of the die, {} routines run, {} samples", 1 << queue, cpu.cycle, bus.peripheral.die_busy, bus.peripheral.routines.iter().map(|r| r.0).sum::<u64>(), words.len() / 2);
    // What the planner counts for the firmware: each window's, and the count of windows.
    let windows = config.windows(frames, chunk).len() as u64;
    let firmware = cpu.cycle - bus.peripheral.die_busy;
    let counted = windows * firmware_per_window(0) + firmware_per_window(expected.len()) - firmware_per_window(0) + sim::chip::FIRMWARE_CYCLES_PER_WORD;
    eprintln!("the CPU's own cycles: {firmware}, counted {counted}");
    assert!(firmware <= counted && 2 * firmware > counted, "the CPU's own cycles: {firmware}, counted {counted}");
    let mantissas: Vec<i64> = words.chunks(2).map(|p| p[0]).collect();
    let first = mantissas.iter().zip(&expected).position(|(a, b)| a != b);
    assert_eq!(mantissas.len(), expected.len(), "samples");
    assert!(first.is_none(), "the die's mantissas differ from sample {first:?} of {}, windows of {} samples", expected.len(), chip_samples);
    }
    std::fs::remove_dir_all(&dir).ok();
    }
}

/// The real decoder's chip image holds its description and its schedule, for windows of
/// 54 and 86 frames (what the planner weighs), with the prebuilt firmware. Skipped without
/// the converted DAC.speech in `~/.llm2chip/models`.
#[test]
fn the_real_decoder_s_schedule_fits_its_chip_image() {
    use netlist::chip::{add_schedule, chip_info, GenerationSetup};
    let dir = Path::new(&std::env::var("HOME").unwrap()).join(".llm2chip/models/dac-speech-24khz-1.5kbps");
    if !dir.join("config.json").exists() {
        eprintln!("no DAC.speech at {}: skipped", dir.display());
        return;
    }
    let firmware = netlist::soc::FirmwareImage::from_elf(include_bytes!("../../../firmware/prebuilt/firmware.elf")).unwrap();
    let base = cell_die_config();
    let die = netlist::DieConfig { microcode_bits: 18, reg_bits: 12, index_bits: 20, cell: netlist::CellConfig { activation_bits: 20, accumulator_bits: 56, ..base.cell }, scale_bits: 24, features: base.features.with_key_shift(8), ..base };
    for frames in [54, 86] {
        let chip = DacChip::new(&dir, frames, die, false).unwrap();
        let (code, schedule) = chip.schedule().unwrap();
        let setup = GenerationSetup { max_new: 0, greedy: true, limit: 1, temperature: 256, seed: 1, raw_tokens: true, stop: Vec::new(), draft_depth: 0, draft_logits: 0, think: None, think_budget: u32::MAX };
        let mut info = chip_info(&die, &code, chip.output_register(), 1, &setup);
        let words = add_schedule(&mut info, &schedule);
        eprintln!("{frames} frames: a description of {} words, a schedule of {} records ({} words)", info.len(), schedule.len(), words.len());
        netlist::image::build_image(&firmware.program, &firmware.data, &info, &words, &[]).unwrap();
        assert!(netlist::soc::TOKENIZER_WORD + words.len() <= 1 << 14, "the schedule fits a data memory of 16,384 words");
    }
}
