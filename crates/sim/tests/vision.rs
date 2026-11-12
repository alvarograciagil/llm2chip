//! The image encoder taken to the chip, stage by stage, on a toy encoder: each stage is
//! checked against the floating-point reference of `sim::vision`.

mod common;
use common::*;
use sim::vision::{patches, VisionTower};

#[test]
fn the_toy_encoder_turns_a_picture_into_tokens_of_the_model_width() {
    let dir = std::env::temp_dir().join(format!("llm2chip-vision-toy-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    make_vision_checkpoint(&dir);
    let tower = VisionTower::open(&dir).unwrap().expect("codificador");
    // 16 × 8 pixels: 8 × 4 patches, 4 × 2 tokens.
    let image = patches(&tower.config, &test_picture(16, 8), 16, 8).unwrap();
    assert_eq!((image.rows, image.cols), (4, 8));
    let tokens = tower.encode(&image).unwrap();
    assert_eq!(tokens.len(), 8);
    assert!(tokens.iter().all(|t| t.len() == 8 && t.iter().all(|v| v.is_finite())));
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn a_picture_larger_than_the_chip_takes_is_resized_down_to_fit() {
    let dir = std::env::temp_dir().join(format!("llm2chip-vision-within-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    make_vision_checkpoint(&dir);
    let tower = VisionTower::open(&dir).unwrap().expect("codificador");
    // 64 × 32 pixels are 32 × 16 patches; the chip takes 32 at most.
    let image = patches(&tower.config.within(32), &test_picture(64, 32), 64, 32).unwrap();
    let count = image.rows * image.cols;
    assert!(count <= 32 && count > 8, "{} × {} parches", image.rows, image.cols);
    assert!(image.rows.is_multiple_of(tower.config.merge) && image.cols.is_multiple_of(tower.config.merge));
    assert!(image.cols > image.rows, "conserva la forma apaisada");
    // A small one is enlarged to the least the encoder takes, which here is also the most.
    let image = patches(&tower.config.within(32), &test_picture(12, 4), 12, 4).unwrap();
    assert!(image.rows * image.cols <= 32, "{} × {} parches", image.rows, image.cols);
    std::fs::remove_dir_all(&dir).ok();
}

/// A die with room for everything, its vectors of `bits` bits.
fn roomy_die(bits: u32) -> netlist::DieConfig {
    let base = die_config();
    netlist::DieConfig { reg_bits: 12, index_bits: 16, microcode_bits: 18, cell: netlist::CellConfig { activation_bits: bits, accumulator_bits: 2 * bits + 8, ..base.cell }, ..base }
}

#[test]
fn the_encoder_in_the_die_arithmetic_gives_the_same_tokens_within_rounding() {
    let dir = std::env::temp_dir().join(format!("llm2chip-vision-exact-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    make_vision_checkpoint(&dir);
    let tower = VisionTower::open(&dir).unwrap().unwrap();
    let image = patches(&tower.config, &test_picture(16, 8), 16, 8).unwrap();
    let reference = tower.encode(&image).unwrap();
    let config = std::fs::read_to_string(dir.join("config.json")).unwrap();
    let mut exact = sim::vision_chip::ExactVision::new(&config, sim::Weights::open_vision(&dir).unwrap(), 32, roomy_die(16)).unwrap();
    let tokens = exact.encode(&tower, &image).unwrap();
    assert_eq!(tokens.len(), reference.len());
    let largest = reference.iter().flatten().fold(0f32, |m, v| m.max(v.abs()));
    let worst = tokens.iter().flatten().zip(reference.iter().flatten()).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
    eprintln!("mayor valor {largest:.3}, mayor diferencia {worst:.4}");
    assert!(worst < 0.02 * largest, "{tokens:?}\nfrente a\n{reference:?}");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn the_language_model_reads_the_chip_encoder_s_tokens_as_the_reference_ones() {
    let dir = std::env::temp_dir().join(format!("llm2chip-vision-lm-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    make_checkpoint(&dir);
    make_vision_checkpoint(&dir);
    let tower = VisionTower::open(&dir).unwrap().unwrap();
    let image = patches(&tower.config, &test_picture(16, 8), 16, 8).unwrap();
    let reference = tower.encode(&image).unwrap();
    let config = std::fs::read_to_string(dir.join("config.json")).unwrap();
    let chip = sim::vision_chip::ExactVision::new(&config, sim::Weights::open_vision(&dir).unwrap(), 32, roomy_die(16)).unwrap().encode(&tower, &image).unwrap();
    // The same language model, in the die's arithmetic, reads either set of tokens.
    let last = |tokens: &[Vec<f32>]| {
        let mut lm = sim::ExactEngine::new(compiler::compile_qwen3_5(CONFIG).unwrap(), sim::Weights::open(&dir).unwrap(), 16).unwrap();
        let mut logits = Vec::new();
        for (position, token) in tokens.iter().enumerate() {
            logits = lm.step_embedding(token, [position as u64; 3], position + 1 == tokens.len()).unwrap().unwrap_or_default();
        }
        logits
    };
    let (from_chip, from_reference) = (last(&chip), last(&reference));
    let best = |v: &[f32]| v.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
    assert_eq!(best(&from_chip), best(&from_reference), "{from_chip:?} frente a {from_reference:?}");
    std::fs::remove_dir_all(&dir).ok();
}

mod gates {
    use super::*;
    use netlist::block::{encode_vector, exponent_register, tables};
    use netlist::demo::read_block_value;
    use netlist::{die_design, DieConfig, Microprogram, Simulator};
    use std::collections::HashMap;

    fn write(sim: &mut Simulator, c: &DieConfig, values: impl IntoIterator<Item = ((u32, u32), i64)>) {
        let mask = (1u64 << c.cell.activation_bits) - 1;
        sim.set("load_vec", 1).unwrap();
        for ((reg, index), value) in values {
            sim.set("vec_addr", ((reg as u64) << c.index_bits) | index as u64).unwrap();
            sim.set("vec_data", value as u64 & mask).unwrap();
            sim.tick();
        }
        sim.set("load_vec", 0).unwrap();
    }

    /// A vector into a register, as the host writes it: mantissas and the exponent.
    fn host_vector(c: &DieConfig, reg: u32, values: &[f32]) -> Vec<((u32, u32), i64)> {
        let (mantissas, exponent) = encode_vector(&values.iter().map(|v| *v as f64).collect::<Vec<_>>(), c.cell.activation_bits);
        mantissas.iter().enumerate().map(|(i, m)| ((reg, i as u32), *m)).chain([((exponent_register(c), reg), exponent)]).collect()
    }

    fn load_microprogram(sim: &mut Simulator, c: &DieConfig, code: &Microprogram) {
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

    fn enter(sim: &mut Simulator, code: &Microprogram, routine: usize, token: u32, position: u32) {
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
        let mut cycles = 0u64;
        while sim.get("halted").unwrap() == 0 {
            sim.tick();
            cycles += 1;
            assert!(cycles < 5_000_000, "el secuenciador no se detiene");
        }
    }

    #[test]
    fn each_encoder_routine_in_gates_gives_the_exact_machine_s_bits() {
        let dir = std::env::temp_dir().join(format!("llm2chip-vision-gates-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        make_vision_checkpoint(&dir);
        let tower = VisionTower::open(&dir).unwrap().unwrap();
        let image = patches(&tower.config, &test_picture(16, 8), 16, 8).unwrap();
        let count = image.rows * image.cols;
        let config = std::fs::read_to_string(dir.join("config.json")).unwrap();
        let c = DieConfig { index_bits: 10, microcode_bits: 14, ..die_config() };

        // The exact machine, with the products added up weight by weight as these gates do.
        let mut exact = sim::vision_chip::ExactVision::new(&config, sim::Weights::open_vision(&dir).unwrap(), count, c).unwrap().with_weights_in_registers();
        let expected = exact.encode(&tower, &image).unwrap();

        let compiler::VisionProgram { program, vision } = compiler::compile_vision(&config, count).unwrap();
        let program = &program;
        let layout = sim::lower::Layout::new(program, &sim::Weights::open_vision(&dir).unwrap(), c, Some(count)).unwrap();
        for (patch, angles) in sim::vision::patch_angles(&tower.config, image.rows, image.cols).iter().enumerate() {
            layout.set_rotation(patch as u32, &angles.iter().map(|a| *a as f64).collect::<Vec<_>>());
        }
        // One entry for each routine: embedding, each block's two and the fusion.
        let routines: Vec<(&[compiler::Inst], bool)> = vision.routines().into_iter().enumerate().map(|(i, range)| (&vision.ops[range], i % 2 == 1 && i < 2 * vision.blocks.len())).collect();
        let code = layout.routines_microprogram(program, &routines).unwrap();
        eprintln!("{} rutinas, {} palabras de microcódigo", routines.len(), code.words.len());

        let mut sim = Simulator::new(&die_design(c)).unwrap();
        load_microprogram(&mut sim, &c, &code);
        let mut initial: HashMap<(u32, u32), i64> = layout.setup().initial.clone();
        initial.extend(layout.rotation_writes());
        write(&mut sim, &c, initial);
        sim.set("load_fn", 1).unwrap();
        for (address, value) in tables(&c).iter().enumerate() {
            sim.set("fn_addr", address as u64).unwrap();
            sim.set("cell_data", *value as u64 & ((1 << c.cell.activation_bits) - 1)).unwrap();
            sim.tick();
        }
        sim.set("load_fn", 0).unwrap();

        let size = program.buffers[vision.pixels];
        let positions = tower.positions(image.rows, image.cols).unwrap();
        for (patch, (values, position)) in image.values.chunks(size).zip(&positions).enumerate() {
            let mut host = host_vector(&c, layout.register_of(vision.pixels), values);
            host.extend(host_vector(&c, layout.register_of(vision.position), position));
            write(&mut sim, &c, host);
            enter(&mut sim, &code, 0, patch as u32, patch as u32);
        }
        for block in 0..vision.blocks.len() {
            for patch in 0..count as u32 {
                enter(&mut sim, &code, 1 + 2 * block, patch, patch);
            }
            for patch in 0..count as u32 {
                enter(&mut sim, &code, 2 + 2 * block, patch, count as u32 - 1);
            }
        }
        let output = layout.register_of(vision.output);
        let width = program.buffers[vision.output] as u32;
        for (token, want) in expected.iter().enumerate() {
            enter(&mut sim, &code, routines.len() - 1, token as u32, token as u32);
            let got: Vec<f32> = (0..width).map(|i| read_block_value(&mut sim, c, output, i).1 as f32).collect();
            assert_eq!(&got, want, "token {token}");
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn one_die_encodes_the_picture_and_the_model_reads_it_in_gates() {
        use sim::lower::{ENTRY_LOGITS, ENTRY_TOKEN_FROM_IMAGE, ENTRY_VISION};
        let dir = std::env::temp_dir().join(format!("llm2chip-vision-joint-gates-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        make_checkpoint(&dir);
        make_vision_checkpoint(&dir);
        let tower = VisionTower::open(&dir).unwrap().unwrap();
        let image = patches(&tower.config, &test_picture(16, 8), 16, 8).unwrap();
        let count = image.rows * image.cols;
        let config = std::fs::read_to_string(dir.join("config.json")).unwrap();
        let program = compiler::attach_vision(compiler::compile_qwen3_5(CONFIG).unwrap(), &config, count).unwrap();
        let part = program.vision.clone().unwrap();
        let c = DieConfig { index_bits: 10, microcode_bits: 15, ..die_config() };

        // The exact machine, products added up weight by weight as these gates do.
        let mut exact = sim::ExactEngine::new(program.clone(), sim::Weights::open_with_vision(&dir).unwrap(), 16).unwrap().with_weights_in_registers();
        let tokens = exact.encode_image(&tower, &image).unwrap();
        let mut expected = Vec::new();
        for index in 0..tokens {
            expected = exact.step_image(index as u32, [index as u64; 3], index + 1 == tokens).unwrap().unwrap_or_default();
        }

        let layout = sim::lower::Layout::new(&program, &sim::Weights::open_with_vision(&dir).unwrap(), c, Some(8)).unwrap();
        for (patch, angles) in sim::vision::patch_angles(&tower.config, image.rows, image.cols).iter().enumerate() {
            layout.set_rotation(patch as u32, &angles.iter().map(|a| *a as f64).collect::<Vec<_>>());
        }
        let code = layout.microprogram(&program).unwrap();
        eprintln!("{} entradas, {} palabras de microcódigo", code.entries.len(), code.words.len());
        let mut sim = Simulator::new(&die_design(c)).unwrap();
        load_microprogram(&mut sim, &c, &code);
        let mut initial: HashMap<(u32, u32), i64> = layout.setup().initial.clone();
        initial.extend(layout.rotation_writes());
        write(&mut sim, &c, initial);
        sim.set("load_fn", 1).unwrap();
        for (address, value) in tables(&c).iter().enumerate() {
            sim.set("fn_addr", address as u64).unwrap();
            sim.set("cell_data", *value as u64 & ((1 << c.cell.activation_bits) - 1)).unwrap();
            sim.tick();
        }
        sim.set("load_fn", 0).unwrap();

        // The encoder, routine by routine in the firmware's order, the host writing each
        // patch's pixels and position before its embedding.
        let size = program.buffers[part.pixels];
        let positions = tower.positions(image.rows, image.cols).unwrap();
        let mut inputs = image.values.chunks(size).zip(&positions);
        let routines = part.routines();
        for (index, (list, token, position, _)) in sim::vision_chip::routine_order(&part, count).iter().enumerate() {
            if index < count {
                let (values, place) = inputs.next().unwrap();
                let mut host = host_vector(&c, layout.register_of(part.pixels), values);
                host.extend(host_vector(&c, layout.register_of(part.position), place));
                write(&mut sim, &c, host);
            }
            let entry = routines.iter().position(|r| std::ptr::eq(&part.ops[r.clone()][0], &list[0])).unwrap();
            enter(&mut sim, &code, ENTRY_VISION + entry, *token, *position);
        }
        // Each token: the fusion makes it and the model reads it in place of an embedding.
        for index in 0..tokens as u32 {
            enter(&mut sim, &code, ENTRY_VISION + routines.len() - 1, index, index);
            enter(&mut sim, &code, ENTRY_TOKEN_FROM_IMAGE, 0, index);
        }
        enter(&mut sim, &code, ENTRY_LOGITS, 0, tokens as u32 - 1);
        let logits = layout.logits_register(&program);
        let got: Vec<f32> = (0..program.vocab as u32).map(|i| read_block_value(&mut sim, c, logits, i).1 as f32).collect();
        assert_eq!(got, expected);
        std::fs::remove_dir_all(&dir).ok();
    }
}

#[test]
fn a_model_that_carries_its_encoder_reads_the_picture_as_the_separate_encoder_feeds_it() {
    let dir = std::env::temp_dir().join(format!("llm2chip-vision-joint-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    make_checkpoint(&dir);
    make_vision_checkpoint(&dir);
    let tower = VisionTower::open(&dir).unwrap().unwrap();
    let image = patches(&tower.config, &test_picture(16, 8), 16, 8).unwrap();
    let config = std::fs::read_to_string(dir.join("config.json")).unwrap();
    // The encoder on its own, its tokens fed to the model as vectors.
    let tokens = sim::vision_chip::ExactVision::new(&config, sim::Weights::open_vision(&dir).unwrap(), 32, roomy_die(16)).unwrap().encode(&tower, &image).unwrap();
    let mut apart = sim::ExactEngine::new(compiler::compile_qwen3_5(CONFIG).unwrap(), sim::Weights::open(&dir).unwrap(), 16).unwrap();
    // One model with the encoder inside, as one chip carries both.
    let joint = compiler::attach_vision(compiler::compile_qwen3_5(CONFIG).unwrap(), &config, 32).unwrap();
    let weights = sim::Weights::open_with_vision(&dir).unwrap();
    let mut together = sim::ExactEngine::new(joint, weights, 16).unwrap();
    assert_eq!(together.encode_image(&tower, &image).unwrap(), tokens.len());
    for (index, token) in tokens.iter().enumerate() {
        let last = index + 1 == tokens.len();
        let a = apart.step_embedding(token, [index as u64; 3], last).unwrap();
        let b = together.step_image(index as u32, [index as u64; 3], last).unwrap();
        assert_eq!(a, b, "token {index}");
    }
    std::fs::remove_dir_all(&dir).ok();
}
