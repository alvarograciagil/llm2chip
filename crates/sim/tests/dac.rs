//! DAC's decoder, codes to sound, taken to the chip on toy checkpoints: the die's exact
//! arithmetic against the floating-point reference of `sim::dac`, and the die's gates
//! against the exact machine.

mod common;
#[allow(unused_imports)]
use common::{cell_die_config, enter, load_assembly, load_microprogram, mask, Rng, Writer};
use sim::dac::Dac;
use sim::dac_chip::DacChip;
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
