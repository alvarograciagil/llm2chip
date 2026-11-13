//! Show-o's transformer in the die's arithmetic against its floating-point reference.

mod common;

use common::{cell_die_config, die_config, enter, load_assembly, load_microprogram, mask, Rng, Writer};
use sim::showo::ShowO;
use sim::showo_chip::ShowOChip;
use std::path::Path;

const TEXT_VOCABULARY: usize = 40;
const CODEBOOK: usize = 64;
const IMAGE: usize = 16;
const HIDDEN: usize = 64;
const LAYERS: usize = 2;

/// A Show-o with Phi-1.5's layout and random weights, small: two layers, two heads of 32.
fn make_toy(dir: &Path) {
    let vocab = TEXT_VOCABULARY + 10 + CODEBOOK + 1;
    let config = serde_json::json!({ "_class_name": "Showo", "llm_vocab_size": TEXT_VOCABULARY, "codebook_size": CODEBOOK, "mask_token_id": vocab - 1, "num_vq_tokens": IMAGE, "max_text": 8, "end_of_text": TEXT_VOCABULARY - 1 });
    std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
    std::fs::create_dir_all(dir.join("phi")).unwrap();
    std::fs::write(dir.join("phi/config.json"), serde_json::json!({ "num_attention_heads": 2, "partial_rotary_factor": 0.5 }).to_string()).unwrap();
    let mut rng = Rng(0x5EED_0F5A_0000_0001);
    let mut w = Writer::default();
    let mut put = |w: &mut Writer, name: &str, shape: &[usize], spread: f32, offset: f32| {
        let values: Vec<f32> = (0..shape.iter().product()).map(|_| offset + spread * rng.unit()).collect();
        w.tensor(&format!("showo.{name}"), "F32", shape, values.iter().flat_map(|v| v.to_le_bytes()).collect());
    };
    let (h, up) = (HIDDEN, 2 * HIDDEN);
    put(&mut w, "model.embed_tokens.weight", &[vocab, h], 1.0, 0.0);
    for layer in 0..LAYERS {
        let p = |s: &str| format!("model.layers.{layer}.{s}");
        for norm in ["input_layernorm", "self_attn.q_layernorm", "self_attn.k_layernorm"] {
            let width = if norm == "input_layernorm" { h } else { h / 2 };
            put(&mut w, &p(&format!("{norm}.weight")), &[width], 0.2, 1.0);
            put(&mut w, &p(&format!("{norm}.bias")), &[width], 0.1, 0.0);
        }
        for (name, rows, cols) in [("self_attn.q_proj", h, h), ("self_attn.k_proj", h, h), ("self_attn.v_proj", h, h), ("self_attn.dense", h, h), ("mlp.fc1", up, h), ("mlp.fc2", h, up)] {
            put(&mut w, &p(&format!("{name}.weight")), &[rows, cols], 0.3, 0.0);
            put(&mut w, &p(&format!("{name}.bias")), &[rows], 0.1, 0.0);
        }
    }
    put(&mut w, "model.final_layernorm.weight", &[h], 0.2, 1.0);
    put(&mut w, "model.final_layernorm.bias", &[h], 0.1, 0.0);
    put(&mut w, "lm_head.weight", &[vocab, h], 0.3, 0.0);
    put(&mut w, "lm_head.bias", &[vocab], 0.1, 0.0);
    w.write(&dir.join("model.safetensors"));
}

/// A text, and an image with half its codes known and half masked.
fn inputs(model: &ShowO) -> (Vec<u32>, Vec<u32>) {
    let text = vec![5, 17, 23, 2, 31];
    let first = model.config.first_code();
    let image = (0..IMAGE as u32).map(|i| if i % 2 == 0 { model.config.mask_token } else { first + (i * 7) % CODEBOOK as u32 }).collect();
    (text, image)
}

/// Largest difference relative to the largest score, and the masked places whose best
/// code is the same.
fn compare(chip: &[f32], float: &[f32], image: &[u32], mask: u32, codebook: usize) -> (f32, usize, usize) {
    let largest = float.iter().fold(0f32, |m, v| m.max(v.abs()));
    let worst = chip.iter().zip(float).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
    let best = |row: &[f32]| row.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
    let masked: Vec<usize> = (0..image.len()).filter(|i| image[*i] == mask).collect();
    let agree = masked.iter().filter(|i| best(&chip[*i * codebook..][..codebook]) == best(&float[*i * codebook..][..codebook])).count();
    (worst / largest, agree, masked.len())
}

#[test]
fn show_o_s_scores_in_the_die_arithmetic_match_the_floating_point_ones() {
    let dir = std::env::temp_dir().join(format!("llm2chip-showo-chip-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    make_toy(&dir);
    let model = ShowO::open(&dir).unwrap().unwrap();
    let (text, image) = inputs(&model);
    let float = model.scores(&text, &image);
    let (ids, positions) = model.text_prefix(&text);
    // Vectors of 16 bits, and cells as wide, so the weights go in whole.
    let base = die_config();
    let die = netlist::DieConfig { reg_bits: 12, index_bits: 16, cell: netlist::CellConfig { weight_bits: 16, accumulator_bits: 40, ..base.cell }, ..base };
    let mut chip = ShowOChip::open(&dir, sim::Weights::open(&dir).unwrap(), die).unwrap();
    let scores = chip.scores(&ids, &positions, &image).unwrap();
    assert_eq!(scores.len(), float.len());
    let (error, agree, masked) = compare(&scores, &float, &image, model.config.mask_token, CODEBOOK);
    eprintln!("diferencia mayor {:.3} % de la mayor puntuación; mejor código igual en {agree} de {masked}", error * 100.0);
    assert!(error < 0.02, "{error}");
    assert!(agree * 10 >= masked * 9, "{agree} de {masked}");
    // The text and the image's other rows matter: with another text the scores move far
    // more than the arithmetic's error.
    let other = model.scores(&[12, 7], &image);
    let (moved, _, _) = compare(&other, &float, &image, model.config.mask_token, CODEBOOK);
    assert!(moved > 10.0 * error, "otro texto mueve las puntuaciones {moved}");
    std::fs::remove_dir_all(&dir).ok();
}

/// The real Show-o, in `LLM2CHIP_SHOWO` or `~/.llm2chip/models/Show-o`: one pass over a
/// picture with its middle masked, on the chip's arithmetic and in floating point.
#[test]
#[ignore = "lenta: Show-o real, 1 300 M de parámetros, una pasada en la aritmética del die"]
fn the_real_show_o_scores_a_masked_picture_as_in_floating_point() {
    let dir = std::env::var_os("LLM2CHIP_SHOWO").map(std::path::PathBuf::from).unwrap_or_else(|| Path::new(&std::env::var("HOME").unwrap()).join(".llm2chip/models/Show-o"));
    if !dir.join("config.json").exists() {
        eprintln!("no está Show-o en {}: se omite", dir.display());
        return;
    }
    let model = ShowO::open(&dir).unwrap().unwrap();
    let tokenizer = tokenizers::Tokenizer::from_file(dir.join("phi/tokenizer.json")).unwrap();
    let text = tokenizer.encode("a green apple on a wooden table", false).unwrap().get_ids().to_vec();
    // A red disc on white, its codes, and the middle quarter of the grid masked.
    let side = model.config.side();
    let planes: Vec<f32> = (0..3)
        .flat_map(|c| (0..side * side).map(move |i| (c, i)))
        .map(|(c, i)| {
            let (x, y) = ((i % side) as f32 - side as f32 / 2.0, (i / side) as f32 - side as f32 / 2.0);
            let inside = x * x + y * y < (side as f32 * 0.3).powi(2);
            let value = if inside { [220.0, 30.0, 30.0][c] } else { 255.0 };
            value / 127.5 - 1.0
        })
        .collect();
    let codes = model.tokenizer().unwrap().encode(&planes).unwrap();
    let grid = model.config.grid();
    let image: Vec<u32> = codes
        .iter()
        .enumerate()
        .map(|(i, code)| if (grid / 4..3 * grid / 4).contains(&(i % grid)) && (grid / 4..3 * grid / 4).contains(&(i / grid)) { model.config.mask_token } else { model.config.first_code() + code })
        .collect();
    let started = std::time::Instant::now();
    let float = model.scores(&text, &image);
    eprintln!("coma flotante: {:.0} s", started.elapsed().as_secs_f64());
    let (ids, positions) = model.text_prefix(&text);
    let base = die_config();
    let die = netlist::DieConfig { reg_bits: 16, index_bits: 20, cell: netlist::CellConfig { weight_bits: 16, accumulator_bits: 40, ..base.cell }, ..base };
    let started = std::time::Instant::now();
    let mut chip = ShowOChip::open(&dir, sim::Weights::open(&dir).unwrap(), die).unwrap();
    eprintln!("chip preparado en {:.0} s", started.elapsed().as_secs_f64());
    let started = std::time::Instant::now();
    let scores = chip.scores(&ids, &positions, &image).unwrap();
    eprintln!("aritmética del die: {:.0} s", started.elapsed().as_secs_f64());
    let (error, agree, masked) = compare(&scores, &float, &image, model.config.mask_token, model.config.codebook as usize);
    eprintln!("diferencia mayor {:.2} % de la mayor puntuación; mejor código igual en {agree} de {masked}", error * 100.0);
    assert!(agree * 10 >= masked * 9, "{agree} de {masked}");
}

/// Show-o's routines, numbered as the chip's microprogram numbers them, on any die: the
/// text a token at a time, then each layer's two passes over the image's rows, then the
/// codes' scores of each row. Gives the scores' mantissas, a row after another, as `read`
/// reads them.
fn run_show_o(run: &mut dyn FnMut(usize, u32, u32), read: &mut dyn FnMut(u32) -> Vec<i64>, ids: &[u32], image: &[u32], layers: usize) -> Vec<i64> {
    use sim::lower::{ENTRY_TOKEN, ENTRY_VISION};
    for (slot, id) in ids.iter().enumerate() {
        run(ENTRY_TOKEN, *id, slot as u32);
    }
    let (text, rows) = (ids.len() as u32, image.len() as u32);
    for layer in 0..layers {
        for row in 0..rows {
            if layer == 0 {
                run(ENTRY_VISION, image[row as usize], 0);
            }
            run(ENTRY_VISION + 1 + 2 * layer, row, text + row);
        }
        for row in 0..rows {
            run(ENTRY_VISION + 2 + 2 * layer, row, text + rows - 1);
        }
    }
    (1..rows - 1)
        .flat_map(|row| {
            run(ENTRY_VISION + 1 + 2 * layers, row, 0);
            read(row)
        })
        .collect()
}

/// A die with cells as deep as Show-o's weights need, each a mantissa of the vectors'
/// width, with as many registers as its routines reach, for `context` places in the cache.
fn die_for(program: &compiler::Program, weights: &sim::Weights, context: usize, format: Option<netlist::format::WeightFormat>) -> netlist::DieConfig {
    use sim::lower::Layout;
    let bits_for = |count: u64| (u64::BITS - count.max(2).saturating_sub(1).leading_zeros()).max(1);
    let base = cell_die_config();
    // Cells as wide as the format's codes, or a whole mantissa each.
    let (weight_bits, coding) = format.map_or((base.cell.activation_bits, base.cell.coding), |f| (f.bits, f.coding));
    let roomy = netlist::DieConfig { microcode_bits: 15, reg_bits: 12, index_bits: 20, cell: netlist::CellConfig { addr_bits: 20, weight_bits, coding, ..base.cell }, features: base.features.with_key_shift(bits_for(context as u64)), ..base };
    let (depth, groups) = Layout::with_cells(program, weights, roomy, context).unwrap().cell_depths();
    let addr_bits = bits_for(depth.max(2 * groups));
    let index_bits = bits_for(program.buffers.iter().map(|b| *b as u64).max().unwrap_or(1).max(context as u64 * 512)).max(addr_bits);
    let roomy = netlist::DieConfig { index_bits, cell: netlist::CellConfig { addr_bits, ..roomy.cell }, ..roomy };
    let wide = Layout::with_cells(program, weights, roomy, context).unwrap();
    wide.microprogram(program).unwrap();
    sim::chip::with_registers_needed(roomy, &wide)
}

#[test]
#[ignore = "lenta: Show-o de juguete en las puertas del die"]
fn show_o_s_routines_on_the_die_s_gates_give_the_exact_machine_s_scores() {
    gates_match_the_exact_machine(None, "gates");
}

/// The same with weights that come in floating point, rewritten in an 8-bit affine format
/// with groups of 32: the cells keep each weight's 8-bit code and each group's scale and
/// bias, as they keep the weights of a checkpoint that comes quantized.
#[test]
#[ignore = "lenta: Show-o de juguete en las puertas del die, pesos de 8 bits"]
fn floating_point_weights_in_an_8_bit_format_go_into_8_bit_cells() {
    gates_match_the_exact_machine(Some(EIGHT_BITS), "8bits");
}

const EIGHT_BITS: netlist::format::WeightFormat = netlist::format::WeightFormat { coding: netlist::format::WeightCoding::Affine, bits: 8, exponent_bits: 0, bias: true, group_size: 32, fine_scale: false, keep_source: false };

/// Show-o's routines on the die's gates against its exact machine, with the weights in
/// `format`, if any, and cells as wide as its codes.
fn gates_match_the_exact_machine(format: Option<netlist::format::WeightFormat>, tag: &str) {
    use netlist::chip::DieEngine;
    use netlist::demo::read_block_value;
    use netlist::{die_design, Simulator};
    use sim::lower::Layout;
    let dir = std::env::temp_dir().join(format!("llm2chip-showo-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    make_toy(&dir);
    let open = || sim::showo_chip::weights(&dir, sim::Weights::open(&dir).unwrap().with_format(format)).unwrap();
    let (shape, config, weights) = open();
    let program = sim::showo_chip::program(&shape);
    // Six words fill the text: then each token's place in the cache is its position.
    let (ids, positions) = config.text_prefix(&[5, 17, 23, 2, 31, 11]);
    assert_eq!(positions, (0..ids.len() as u64).collect::<Vec<_>>());
    let first = config.first_code();
    let codes: Vec<u32> = (0..IMAGE as u32).map(|i| if i % 3 == 0 { config.mask_token } else { first + (i * 5) % CODEBOOK as u32 }).collect();
    let image: Vec<u32> = std::iter::once(config.start_of_image()).chain(codes).chain([config.end_of_image()]).collect();
    let context = ids.len() + image.len();

    let die = die_for(&program, &weights, context, format);
    let layout = Layout::with_cells(&program, &weights, die, context).unwrap();
    let code = layout.microprogram(&program).unwrap();
    let output = layout.register_of(program.vision.as_ref().unwrap().output);

    // The exact machine of the same die, routine by routine.
    let mut vm = sim::chip::VmEngine::new(program.clone(), open().2, die, context).unwrap();
    let exact = {
        let vm = std::cell::RefCell::new(&mut vm);
        run_show_o(
            &mut |r, t, p| vm.borrow_mut().run(r, t, p),
            &mut |_| (0..CODEBOOK as u32).map(|i| vm.borrow_mut().read(output << die.index_bits | i)).collect(),
            &ids,
            &image,
            shape.layers,
        )
    };

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
        let (codes, factors) = layout.cell_contents(&weights, cell).unwrap();
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
    if format.is_some() {
        let (codes, factors) = layout.cell_contents(&weights, 0).unwrap();
        assert!(codes.iter().all(|c| (-128..128).contains(c)), "códigos de 8 bits");
        assert!(factors.len() > 1, "una escala y un sesgo por grupo");
    }
    let mut cycles = 0;
    let gates = {
        let sim = std::cell::RefCell::new(sim);
        run_show_o(
            &mut |r, t, p| cycles += enter(&mut sim.borrow_mut(), &code, r, t, p),
            &mut |_| (0..CODEBOOK as u32).map(|i| read_block_value(&mut sim.borrow_mut(), die, output, i).0).collect(),
            &ids,
            &image,
            shape.layers,
        )
    };
    eprintln!("Show-o de juguete en puertas: {cycles} ciclos");
    assert_eq!(gates, exact, "puntuaciones de los códigos");
    std::fs::remove_dir_all(&dir).ok();
}

/// The chip's firmware runs a pass of Show-o on the instruction-level CPU, driving the
/// software die: the host sends a text shorter than the longest, whose tokens the firmware
/// rotates as their real positions, and the image's tokens; the scores it sends back are
/// those of the host ordering the same routines on the same machine.
#[test]
fn the_firmware_runs_show_o_s_pass_as_the_host_ordered_it() {
    use netlist::chip::{add_showo, chip_info, ChipModel, DieEngine, GenerationSetup};
    use netlist::image::build_image;
    use netlist::rv32::Rv32Iss;
    use netlist::sampler::{exponential_table, SamplerConfig, SamplerModel};
    use netlist::soc::{SocBus, SocConfig};
    use sim::lower::{Layout, ENTRY_TOKEN, ENTRY_VISION};
    let Some(firmware) = common::firmware() else { return };
    let dir = std::env::temp_dir().join(format!("llm2chip-showo-firmware-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    make_toy(&dir);
    let open = || sim::showo_chip::weights(&dir, sim::Weights::open(&dir).unwrap()).unwrap();
    let (shape, config, weights) = open();
    let program = sim::showo_chip::program(&shape);
    let (ids, positions) = config.text_prefix(&[5, 17, 23]);
    let first = positions[0] as u32;
    assert!(first > 0, "un texto corto empieza más adelante");
    let mask_token = config.mask_token;
    let codes: Vec<u32> = (0..IMAGE as u32).map(|i| if i % 3 == 0 { mask_token } else { config.first_code() + (i * 5) % CODEBOOK as u32 }).collect();
    let image: Vec<u32> = std::iter::once(config.start_of_image()).chain(codes).chain([config.end_of_image()]).collect();
    let context = (first as usize + ids.len() + image.len()).next_power_of_two();
    let die = die_for(&program, &weights, context, None);
    let layout = Layout::with_cells(&program, &weights, die, context).unwrap();
    let code = layout.microprogram(&program).unwrap();
    let showo = sim::showo_chip::firmware_info(&program, &layout, &die).unwrap();
    let mask_bits = |v: i64| v as u32 & ((1u32 << die.cell.activation_bits) - 1);

    // The host's order: each token rotated as its position, then the routines.
    let mut vm = sim::chip::VmEngine::new(program.clone(), open().2, die, context).unwrap();
    let width = 3 * showo.rope_half + 1;
    let (text, rows) = (ids.len() as u32, image.len() as u32);
    for slot in 0..text + rows {
        let row = layout.text_rotation(&program, [(first + slot) as u64; 3], [showo.rope_half as usize, 0, 0]);
        row.iter().enumerate().for_each(|(i, value)| vm.write(showo.rope + slot * width + i as u32, *value));
    }
    for (slot, id) in ids.iter().enumerate() {
        vm.run(ENTRY_TOKEN, *id, slot as u32);
    }
    for layer in 0..shape.layers {
        for row in 0..rows {
            if layer == 0 {
                vm.run(ENTRY_VISION, image[row as usize], 0);
            }
            vm.run(ENTRY_VISION + 1 + 2 * layer, row, text + row);
        }
        for row in 0..rows {
            vm.run(ENTRY_VISION + 2 + 2 * layer, row, text + rows - 1);
        }
    }
    let mut expected = Vec::new();
    for row in 1..rows - 1 {
        vm.run(ENTRY_VISION + 1 + 2 * shape.layers, row, 0);
        expected.push(mask_bits(vm.read(showo.scores.1)));
        expected.extend((0..showo.codebook).map(|i| mask_bits(vm.read(showo.scores.0 + i))));
    }

    // The firmware's order, on the instruction-level chip.
    let setup = GenerationSetup { max_new: 0, greedy: true, limit: 1, temperature: 256, seed: 1, raw_tokens: true, stop: Vec::new(), draft_depth: 0, draft_logits: 0, think: None, think_budget: u32::MAX };
    let mut info = chip_info(&die, &code, layout.logits_register(&program), program.vocab as u32, &setup);
    add_showo(&mut info, &showo);
    let mut segments = sim::chip::image_segments(&die, &code, &layout);
    segments.extend(sim::chip::cell_segments(&die, &layout, &weights).unwrap());
    let chip_image = build_image(&firmware.program, &firmware.data, &info, &[], &segments).unwrap();
    let input: Vec<u8> = [0, text, first].into_iter().chain(ids.iter().copied()).chain(image.iter().copied()).flat_map(|w| w.to_le_bytes()).collect();
    let engine = sim::chip::VmEngine::new(program.clone(), open().2, die, context).unwrap();
    let model = ChipModel::new(engine, code.clone(), SamplerModel { config: SamplerConfig { logit_bits: 16, index_bits: 12, entries: 4, lanes: 1 }, rng: 0, table: exponential_table() });
    let mut bus = SocBus::booted_with(SocConfig { program_bits: 13, data_bits: 14, fifo_bits: 6 }, &chip_image, &input, model).unwrap();
    let mut cpu = Rv32Iss::default();
    while !cpu.halted {
        cpu.step(&mut bus);
        assert!(cpu.cycle < 500_000_000, "el firmware no termina");
    }
    // Each segment loaded at boot reports a dot first.
    let start = bus.tx.iter().position(|b| *b != b'.').unwrap_or(bus.tx.len());
    let sent: Vec<u32> = bus.tx[start..].chunks(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).map(|w| w & ((1u32 << die.cell.activation_bits) - 1)).collect();
    eprintln!("firmware: {} ciclos de CPU, {} palabras", cpu.cycle, sent.len());
    assert_eq!(sent.len(), expected.len());
    assert_eq!(sent, expected, "puntuaciones de los códigos");
    std::fs::remove_dir_all(&dir).ok();
}

/// A toy Show-o chip set up to fill an image: its image, what the host sends it, and the
/// codes the host gets ordering the same routines on the same machine.
struct FillChip {
    dir: std::path::PathBuf,
    program: compiler::Program,
    die: netlist::DieConfig,
    context: usize,
    code: netlist::Microprogram,
    image: Vec<u32>,
    input: Vec<u8>,
    expected: Vec<u32>,
    /// The embedding table stays in the image and the loader brings each token's row.
    outside: bool,
}

/// The die's exact machine for a fill chip, with its table where the chip keeps it.
fn fill_engine(dir: &Path, program: &compiler::Program, die: netlist::DieConfig, context: usize, outside: bool) -> sim::chip::VmEngine {
    let weights = sim::showo_chip::weights(dir, sim::Weights::open(dir).unwrap()).unwrap().2;
    if outside {
        sim::chip::VmEngine::new_table_outside(program.clone(), weights, die, context).unwrap()
    } else {
        sim::chip::VmEngine::new(program.clone(), weights, die, context).unwrap()
    }
}

fn fill_soc() -> netlist::soc::SocConfig {
    netlist::soc::SocConfig { program_bits: 13, data_bits: 14, fifo_bits: 6 }
}

fn fill_sampler() -> netlist::sampler::SamplerConfig {
    netlist::sampler::SamplerConfig { logit_bits: 16, index_bits: 12, entries: 4, lanes: 1 }
}

fn fill_chip(tag: &str, steps: usize, outside: bool) -> Option<FillChip> {
    use netlist::chip::{add_showo, chip_info, GenerationSetup};
    use netlist::image::build_image;
    use sim::lower::Layout;
    use sim::showo_chip::{fill_on_die, schedule, Fill};
    let firmware = common::firmware()?;
    let dir = std::env::temp_dir().join(format!("llm2chip-showo-fill-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    make_toy(&dir);
    let open = || sim::showo_chip::weights(&dir, sim::Weights::open(&dir).unwrap()).unwrap();
    let (shape, config, weights) = open();
    let program = sim::showo_chip::program_guided(&shape, Some(1.75));
    let (text, first) = config.text_prefix(&[5, 17, 23]);
    let (plain, plain_first) = config.text_prefix(&[]);
    // Half the image known, half to fill.
    let codes: Vec<Option<u32>> = (0..IMAGE as u32).map(|i| (i % 2 == 1).then_some((i * 7) % CODEBOOK as u32)).collect();
    let steps = schedule(IMAGE, steps);
    let fill = Fill { text: (&text, first[0] as u32), plain: Some((&plain, plain_first[0] as u32)), schedule: &steps, seed: 7 };
    let context = (first[0] as usize + text.len() + IMAGE + 2).next_power_of_two();
    let die = if outside { sim::showo_chip::addressed_die(&program, &weights, die_for(&program, &weights, context, None), context).unwrap() } else { die_for(&program, &weights, context, None) };
    let layout = if outside { Layout::with_cells_table_outside(&program, &weights, die, context).unwrap() } else { Layout::with_cells(&program, &weights, die, context).unwrap() };
    assert_eq!(layout.outside_table().is_some(), outside);
    let code = layout.microprogram(&program).unwrap();
    let showo = sim::showo_chip::firmware_info(&program, &layout, &die).unwrap();
    assert_eq!(showo.guided, 1);

    let mut vm = fill_engine(&dir, &program, die, context, outside);
    let expected = fill_on_die(&mut vm, &layout, &program, &showo, &config, die.cell.activation_bits, &codes, &fill);
    for (place, known) in codes.iter().enumerate() {
        if let Some(code) = known {
            assert_eq!(expected[place], *code, "un código conocido se queda");
        }
    }

    let setup = GenerationSetup { max_new: 0, greedy: true, limit: 1, temperature: 256, seed: 1, raw_tokens: true, stop: Vec::new(), draft_depth: 0, draft_logits: 0, think: None, think_budget: u32::MAX };
    let mut info = chip_info(&die, &code, layout.logits_register(&program), program.vocab as u32, &setup);
    add_showo(&mut info, &showo);
    let mut segments = sim::chip::image_segments(&die, &code, &layout);
    segments.extend(sim::chip::cell_segments(&die, &layout, &weights).unwrap());
    let image = build_image(&firmware.program, &firmware.data, &info, &[], &segments).unwrap();
    let mut words = vec![steps.len() as u32, config.start_of_image(), config.end_of_image(), config.mask_token, config.first_code()];
    for (ids, first) in [(&text, first[0] as u32), (&plain, plain_first[0] as u32)] {
        words.extend([ids.len() as u32, first]);
        words.extend(ids.iter());
    }
    words.push(fill.seed);
    words.extend(steps.iter().flat_map(|(remaining, temperature)| [*remaining as u32, *temperature as u32]));
    words.extend(codes.iter().map(|c| c.unwrap_or(u32::MAX)));
    let input = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    Some(FillChip { dir, program, die, context, code, image, input, expected, outside })
}

/// The chip at instruction level, its die in software: what it sends and its cycles.
fn fill_on_iss(chip: &FillChip) -> (Vec<u8>, u64) {
    use netlist::chip::ChipModel;
    use netlist::rv32::Rv32Iss;
    use netlist::sampler::{exponential_table, SamplerModel};
    use netlist::soc::SocBus;
    let engine = fill_engine(&chip.dir, &chip.program, chip.die, chip.context, chip.outside);
    let model = ChipModel::new(engine, chip.code.clone(), SamplerModel { config: fill_sampler(), rng: 0, table: exponential_table() });
    let mut bus = SocBus::booted_with(fill_soc(), &chip.image, &chip.input, model).unwrap();
    let mut cpu = Rv32Iss::default();
    while !cpu.halted {
        cpu.step(&mut bus);
        assert!(cpu.cycle < 2_000_000_000, "el firmware no termina");
    }
    (bus.tx.clone(), cpu.cycle)
}

/// The codes in what the chip sent, after the dots of the segments loaded at boot.
fn sent_codes(bytes: &[u8]) -> Vec<u32> {
    let start = bytes.iter().position(|b| *b != b'.').unwrap_or(bytes.len());
    bytes[start..].chunks(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

/// The chip fills an image itself: its firmware runs MaskGIT's steps on the CPU, with
/// guidance in the die, and gives the codes the host gets ordering the same routines and
/// the same draws on the same machine.
#[test]
fn the_firmware_fills_an_image_as_the_host_ordered_it() {
    let Some(chip) = fill_chip("iss", 4, false) else { return };
    let (sent, cycles) = fill_on_iss(&chip);
    eprintln!("relleno en el chip: {cycles} ciclos de CPU, códigos {:?}", sent_codes(&sent));
    assert_eq!(sent_codes(&sent), chip.expected, "los códigos del relleno");
    std::fs::remove_dir_all(&chip.dir).ok();
}

/// The same with the embedding table left in the image, as the real Show-o needs it: the
/// firmware's loader brings each row, the text's and the image's, before the routine that
/// reads it, as the host does.
#[test]
fn the_firmware_fills_an_image_with_the_table_in_the_image() {
    let Some(chip) = fill_chip("fuera", 4, true) else { return };
    let (sent, _) = fill_on_iss(&chip);
    assert_eq!(sent_codes(&sent), chip.expected, "los códigos del relleno");
    std::fs::remove_dir_all(&chip.dir).ok();
}

/// The same chip in gates, CPU and all: the same bytes in the same cycles.
#[test]
#[ignore = "lenta: el chip entero en puertas rellena una imagen de juguete, millones de ciclos"]
fn the_whole_chip_fills_an_image_in_gates_as_at_instruction_level() {
    let Some(chip) = fill_chip("gates", 2, false) else { return };
    let (expected, cycles) = fill_on_iss(&chip);
    assert_eq!(sent_codes(&expected), chip.expected);
    let config = netlist::chip::ChipConfig { soc: fill_soc(), die: chip.die, sampler: fill_sampler() };
    let started = std::time::Instant::now();
    let (output, gate_cycles) = common::gates(&netlist::chip::chip_design(config), &chip.image, &chip.input, cycles + 4096);
    eprintln!("puertas: {gate_cycles} ciclos en {:.0} s", started.elapsed().as_secs_f64());
    assert_eq!(output, expected);
    assert_eq!(gate_cycles, cycles, "ciclos");
    std::fs::remove_dir_all(&chip.dir).ok();
}

/// The real Show-o fits the firmware's 32-bit addresses with its embedding table left in
/// the image and the guidance kept as the final state: 4 096 cells of 16-bit weights.
#[test]
#[ignore = "lenta: lee los 1 300 M de pesos del Show-o real para repartirlos en las celdas"]
fn the_real_show_o_fits_the_addressed_die_with_its_table_in_the_image() {
    let dir = std::env::var_os("LLM2CHIP_SHOWO").map(std::path::PathBuf::from).unwrap_or_else(|| Path::new(&std::env::var("HOME").unwrap()).join(".llm2chip/models/Show-o"));
    if !dir.join("config.json").exists() {
        eprintln!("no está Show-o en {}: se omite", dir.display());
        return;
    }
    let (shape, config, weights) = sim::showo_chip::weights(&dir, sim::Weights::open(&dir).unwrap()).unwrap();
    let program = sim::showo_chip::program_guided(&shape, Some(1.75));
    let context = (config.max_text + 1 + shape.image_rows).next_power_of_two();
    let base = die_config();
    let base = netlist::DieConfig { cells: 4096, cell: netlist::CellConfig { weight_bits: 16, accumulator_bits: 48, ..base.cell }, scale_bits: 21, ..base };
    let die = sim::showo_chip::addressed_die(&program, &weights, base, context).unwrap();
    eprintln!("{} bits de registro y {} de elemento", die.reg_bits, die.index_bits);
    assert!(die.reg_bits + die.index_bits <= 32);
    let layout = sim::lower::Layout::with_cells_table_outside(&program, &weights, die, context).unwrap();
    let table = layout.outside_table().expect("la tabla en la imagen");
    assert_eq!((table.cols as usize, table.mantissas.len()), (shape.hidden, shape.vocab * shape.hidden));
}
