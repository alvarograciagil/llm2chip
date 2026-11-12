//! A complete Qwen3.5-shaped model, weights and all, executed on the gates.

use compiler::compile_qwen3_5;
use netlist::block::tables;
use netlist::demo::read_block_value;
use netlist::{die_design, DieConfig, Microprogram, Simulator};
use sim::lower::{Layout, ENTRY_DRAFT, ENTRY_DRAFT_LOGITS, ENTRY_FEED_FROM_MAIN, ENTRY_LOGITS, ENTRY_TOKEN};
use sim::{ExactEngine, Engine, Numeric, Weights};
use std::path::Path;

mod common;
use common::*;

fn mask(c: &DieConfig) -> u64 {
    (1u64 << c.cell.activation_bits) - 1
}

fn load_assembly(sim: &mut Simulator, c: &DieConfig, initial: &std::collections::HashMap<(u32, u32), i64>) {
    sim.set("load_vec", 1).unwrap();
    for ((reg, index), value) in initial {
        sim.set("vec_addr", ((*reg as u64) << c.index_bits) | *index as u64).unwrap();
        sim.set("vec_data", *value as u64 & mask(c)).unwrap();
        sim.tick();
    }
    sim.set("load_vec", 0).unwrap();
}

/// Where each register starts in a compact vector memory, before anything goes into it.
fn load_register_table(sim: &mut Simulator, c: &DieConfig, layout: &Layout) {
    sim.set("load_base", 1).unwrap();
    for (register, entry) in sim::chip::register_table(c, layout).iter().enumerate() {
        sim.set("base_addr", register as u64).unwrap();
        sim.set("base_data", *entry as u64).unwrap();
        sim.tick();
    }
    sim.set("load_base", 0).unwrap();
}

fn make_moe_checkpoint(dir: &Path) {
    let mut rng = Rng(0x0FED_CBA9_8765_4321);
    let mut w = Writer::default();
    w.matrix("model.embed_tokens", 8, 8, &mut rng);
    w.matrix("lm_head", 8, 8, &mut rng);
    w.floats("model.norm.weight", &[1.0; 8]);
    let p = |s: &str| format!("model.layers.0.{s}");
    w.floats(&p("input_layernorm.weight"), &[1.0; 8]);
    w.floats(&p("post_attention_layernorm.weight"), &[1.0; 8]);
    w.matrix(&p("self_attn.q_proj"), 32, 8, &mut rng);
    w.matrix(&p("self_attn.k_proj"), 8, 8, &mut rng);
    w.matrix(&p("self_attn.v_proj"), 8, 8, &mut rng);
    w.matrix(&p("self_attn.o_proj"), 8, 16, &mut rng);
    w.floats(&p("self_attn.q_norm.weight"), &[1.0; 8]);
    w.floats(&p("self_attn.k_norm.weight"), &[1.0; 8]);
    w.matrix(&p("mlp.gate"), 4, 8, &mut rng);
    w.matrix(&p("mlp.switch_mlp.gate_proj"), 32, 8, &mut rng);
    w.matrix(&p("mlp.switch_mlp.up_proj"), 32, 8, &mut rng);
    w.matrix(&p("mlp.switch_mlp.down_proj"), 32, 8, &mut rng);
    w.matrix(&p("mlp.shared_expert.gate_proj"), 8, 8, &mut rng);
    w.matrix(&p("mlp.shared_expert.up_proj"), 8, 8, &mut rng);
    w.matrix(&p("mlp.shared_expert.down_proj"), 8, 8, &mut rng);
    w.matrix(&p("mlp.shared_expert_gate"), 1, 8, &mut rng);
    w.write(&dir.join("model.safetensors"));
}

/// The longest context the die of these tests is built for.
const CHIP_TOKENS: usize = 8;

/// Writes the microprogram, once, as the load controller does at start-up.
fn load_microprogram(sim: &mut Simulator, c: &DieConfig, code: &Microprogram) {
    assert!(code.words.len() < 1 << c.microcode_bits, "{} palabras", code.words.len());
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

/// What the firmware does for each routine: set the parameters and start the die.
fn enter(sim: &mut Simulator, code: &Microprogram, routine: usize, token: u32, position: u32) -> u64 {
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
        assert!(cycles < 20_000_000, "el secuenciador no se detiene");
    }
    cycles
}

struct GateRun {
    sim: Simulator,
    layout: Layout,
    code: Microprogram,
    cycles: u64,
}

fn gates_after(program: &compiler::Program, dir: &Path, c: DieConfig, tokens: &[u32]) -> GateRun {
    let layout = Layout::new(program, &Weights::open(dir).unwrap(), c, Some(CHIP_TOKENS)).unwrap();
    let code = layout.microprogram(program).unwrap();
    let mut sim = Simulator::new(&die_design(c)).unwrap();
    load_microprogram(&mut sim, &c, &code);
    load_assembly(&mut sim, &c, &layout.setup().initial);
    sim.set("load_fn", 1).unwrap();
    for (address, value) in tables(&c).iter().enumerate() {
        sim.set("fn_addr", address as u64).unwrap();
        sim.set("cell_data", *value as u64 & mask(&c)).unwrap();
        sim.tick();
    }
    sim.set("load_fn", 0).unwrap();
    let mut cycles = 0;
    for (position, token) in tokens.iter().enumerate() {
        cycles += enter(&mut sim, &code, ENTRY_TOKEN, *token, position as u32);
        if position + 1 == tokens.len() {
            cycles += enter(&mut sim, &code, ENTRY_LOGITS, *token, position as u32);
        }
    }
    GateRun { sim, layout, code, cycles }
}

fn assert_vm_matches_gates(program: &compiler::Program, config: &str, dir: &Path, c: DieConfig, tokens: &[u32], sim: &mut Simulator, layout: &Layout) {
    let mut vm = ExactEngine::new(compile_qwen3_5(config).unwrap(), Weights::open(dir).unwrap(), c.cell.activation_bits).unwrap().with_weights_in_registers();
    for (position, token) in tokens.iter().enumerate() {
        vm.step(*token, position + 1 == tokens.len()).unwrap();
    }
    for (buffer, length) in program.buffers.iter().enumerate().filter(|(buffer, _)| !layout.is_spread(*buffer) && !layout.is_overwritten(*buffer)) {
        let gates = layout.register_of(buffer);
        let fast = vm.register_of(buffer);
        for index in 0..*length as u32 {
            let (mantissa, _) = read_block_value(sim, c, gates, index);
            assert_eq!(mantissa, vm.mantissa(fast, index), "búfer {buffer}, elemento {index}");
        }
    }
}

fn draft_config() -> String {
    CONFIG.replacen(r#""model_type": "qwen3_5_text","#, r#""model_type": "qwen3_5_text", "mtp_num_hidden_layers": 1,"#, 1)
}

fn draft_checkpoint(dir: &Path) {
    std::fs::create_dir_all(dir).unwrap();
    make_checkpoint(dir);
    let mut rng = Rng(0xABCD_0123_4567_89EF);
    let mut w = Writer::default();
    for name in ["mtp.pre_fc_norm_embedding", "mtp.pre_fc_norm_hidden", "mtp.norm", "mtp.layers.0.input_layernorm", "mtp.layers.0.post_attention_layernorm", "mtp.layers.0.self_attn.q_norm", "mtp.layers.0.self_attn.k_norm"] {
        w.floats(&format!("{name}.weight"), &[1.0; 8]);
    }
    w.matrix("mtp.fc", 8, 16, &mut rng);
    let p = |s: &str| format!("mtp.layers.0.{s}");
    w.matrix(&p("self_attn.q_proj"), 32, 8, &mut rng);
    w.matrix(&p("self_attn.k_proj"), 8, 8, &mut rng);
    w.matrix(&p("self_attn.v_proj"), 8, 8, &mut rng);
    w.matrix(&p("self_attn.o_proj"), 8, 16, &mut rng);
    w.matrix(&p("mlp.gate_proj"), 16, 8, &mut rng);
    w.matrix(&p("mlp.up_proj"), 16, 8, &mut rng);
    w.matrix(&p("mlp.down_proj"), 8, 16, &mut rng);
    w.write(&dir.join("mtp.safetensors"));
}

#[test]
fn draft_head_runs_on_the_gates() {
    let config = draft_config();
    let dir = std::env::temp_dir().join(format!("llm2chip-mtp-{}", std::process::id()));
    draft_checkpoint(&dir);

    let program = compile_qwen3_5(&config).unwrap();
    let tokens = [3u32, 7];
    let next = 5u32;
    let mut reference = Engine::new(compile_qwen3_5(&config).unwrap(), Weights::open(&dir).unwrap(), Numeric::Float).unwrap();
    for (i, t) in tokens.iter().enumerate() {
        reference.step(*t, i + 1 == tokens.len()).unwrap();
    }
    reference.feed_draft_from_main();
    let expected = reference.draft(next, true).unwrap().unwrap().to_vec();

    let c = DieConfig { reg_bits: 10, index_bits: 9, ..die_config() };
    let GateRun { mut sim, layout, code, .. } = gates_after(&program, &dir, c, &tokens);
    enter(&mut sim, &code, ENTRY_FEED_FROM_MAIN, next, 0);
    enter(&mut sim, &code, ENTRY_DRAFT, next, 0);
    enter(&mut sim, &code, ENTRY_DRAFT_LOGITS, next, 0);

    let draft = program.draft.as_ref().unwrap();
    let got: Vec<f32> = (0..8).map(|i| read_block_value(&mut sim, c, layout.register_of(draft.logits), i).1 as f32).collect();
    let best = |v: &[f32]| v.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
    assert_eq!(best(&got), best(&expected), "{got:?} {expected:?}");
    let spread: f32 = expected.iter().map(|v| v.abs()).sum::<f32>() / 8.0;
    let error: f32 = got.iter().zip(&expected).map(|(a, b)| (a - b).abs()).sum::<f32>() / 8.0;
    assert!(error < 0.1 * spread.max(0.2), "error medio {error}, magnitud {spread}");

    let mut vm = ExactEngine::new(compile_qwen3_5(&config).unwrap(), Weights::open(&dir).unwrap(), 16).unwrap().with_weights_in_registers();
    for (position, token) in tokens.iter().enumerate() {
        vm.step(*token, position + 1 == tokens.len()).unwrap();
    }
    vm.feed_draft_from_main();
    vm.draft(next, true).unwrap();
    for index in 0..8 {
        let (mantissa, _) = read_block_value(&mut sim, c, layout.register_of(draft.logits), index);
        assert_eq!(mantissa, vm.mantissa(vm.register_of(draft.logits), index), "elemento {index}");
    }
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn other_mantissa_widths_match_the_gates_bit_for_bit() {
    let dir = std::env::temp_dir().join(format!("llm2chip-widths-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    make_checkpoint(&dir);
    let program = compile_qwen3_5(CONFIG).unwrap();
    let tokens = [3u32, 7, 1];
    for bits in [12u32, 18, 20] {
        let mut base = die_config();
        base.cell.activation_bits = bits;
        base.cell.accumulator_bits = 2 * bits;
        base.function_bits = 12.min(bits);
        let GateRun { mut sim, layout, .. } = gates_after(&program, &dir, base, &tokens);
        assert_vm_matches_gates(&program, CONFIG, &dir, base, &tokens, &mut sim, &layout);
    }
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn mixture_of_experts_runs_on_the_gates() {
    const MOE: &str = r#"{
      "model_type": "qwen3_5_moe_text",
      "hidden_size": 8, "vocab_size": 8,
      "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 8,
      "linear_num_key_heads": 1, "linear_key_head_dim": 4,
      "linear_num_value_heads": 2, "linear_value_head_dim": 4,
      "linear_conv_kernel_dim": 4, "partial_rotary_factor": 0.25,
      "layer_types": ["full_attention"],
      "num_experts": 4, "num_experts_per_tok": 2,
      "moe_intermediate_size": 8, "shared_expert_intermediate_size": 8,
      "rope_parameters": { "rope_theta": 10000.0 }
    }"#;
    let dir = std::env::temp_dir().join(format!("llm2chip-moe-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    make_moe_checkpoint(&dir);
    let program = compile_qwen3_5(MOE).unwrap();
    let tokens = [2u32, 5];

    let mut reference = Engine::new(compile_qwen3_5(MOE).unwrap(), Weights::open(&dir).unwrap(), Numeric::Float).unwrap();
    let mut expected = Vec::new();
    for (i, t) in tokens.iter().enumerate() {
        if let (Some(l), _) = reference.step(*t, i + 1 == tokens.len()).unwrap() {
            expected = l.to_vec();
        }
    }

    let c = DieConfig { reg_bits: 10, index_bits: 9, ..die_config() };
    let GateRun { mut sim, layout, .. } = gates_after(&program, &dir, c, &tokens);
    let got: Vec<f32> = (0..8).map(|i| read_block_value(&mut sim, c, layout.logits_register(&program), i).1 as f32).collect();
    let best = |v: &[f32]| v.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
    assert_eq!(best(&got), best(&expected), "{got:?} {expected:?}");
    let spread: f32 = expected.iter().map(|v| v.abs()).sum::<f32>() / 8.0;
    let error: f32 = got.iter().zip(&expected).map(|(a, b)| (a - b).abs()).sum::<f32>() / 8.0;
    assert!(error < 0.08 * spread.max(0.2), "error medio {error}, magnitud {spread}");
    assert_vm_matches_gates(&program, MOE, &dir, c, &tokens, &mut sim, &layout);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn tiny_model_runs_whole_on_the_gates() {
    let dir = std::env::temp_dir().join(format!("llm2chip-tiny-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    make_checkpoint(&dir);
    let program = compile_qwen3_5(CONFIG).unwrap();

    let tokens = [3u32, 7, 1];
    let mut reference = Engine::new(compile_qwen3_5(CONFIG).unwrap(), Weights::open(&dir).unwrap(), Numeric::Float).unwrap();
    let mut expected = Vec::new();
    for (i, t) in tokens.iter().enumerate() {
        let (logits, _) = reference.step(*t, i + 1 == tokens.len()).unwrap();
        if let Some(l) = logits {
            expected = l.to_vec();
        }
    }

    let c = die_config();
    let GateRun { mut sim, layout, code, cycles: total_cycles } = gates_after(&program, &dir, c, &tokens);
    eprintln!("microprograma: {} palabras, {} parámetros", code.words.len(), code.parameters.len());

    let logits_reg = layout.logits_register(&program);
    let got: Vec<f32> = (0..8).map(|i| read_block_value(&mut sim, c, logits_reg, i).1 as f32).collect();
    let best = |v: &[f32]| v.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
    eprintln!("ciclos: {total_cycles}\nreferencia: {expected:?}\npuertas:    {got:?}");
    assert_eq!(best(&got), best(&expected), "el token más probable debe coincidir");
    let spread: f32 = expected.iter().map(|v| v.abs()).sum::<f32>() / 8.0;
    let error: f32 = got.iter().zip(&expected).map(|(a, b)| (a - b).abs()).sum::<f32>() / 8.0;
    assert!(error < 0.08 * spread.max(0.2), "error medio {error}, magnitud {spread}");

    // The fast machine must reach the same bits as the gates, in every buffer.
    let mut vm = ExactEngine::new(compile_qwen3_5(CONFIG).unwrap(), Weights::open(&dir).unwrap(), 16).unwrap().with_weights_in_registers();
    for (position, token) in tokens.iter().enumerate() {
        vm.step(*token, position + 1 == tokens.len()).unwrap();
    }
    for (buffer, length) in program.buffers.iter().enumerate().filter(|(buffer, _)| !layout.is_spread(*buffer) && !layout.is_overwritten(*buffer)) {
        let gates = layout.register_of(buffer);
        let fast = vm.register_of(buffer);
        for index in 0..*length as u32 {
            let (mantissa, _) = read_block_value(&mut sim, c, gates, index);
            assert_eq!(mantissa, vm.mantissa(fast, index), "búfer {buffer}, elemento {index}");
        }
    }
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
#[ignore = "lenta: cientos de miles de ciclos: solo con optimizaciones"]
fn a_die_built_for_its_program_runs_it_identically_with_less_logic() {
    let dir = std::env::temp_dir().join(format!("llm2chip-trim-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    make_checkpoint(&dir);
    let program = compile_qwen3_5(CONFIG).unwrap();
    let c = die_config();
    let mut general = gates_after(&program, &dir, c, &[3, 5]);
    let trimmed = DieConfig { features: netlist::DieFeatures::of(&general.code), ..c };
    let mut specific = gates_after(&program, &dir, trimmed, &[3, 5]);
    assert_eq!(specific.code.words.len(), general.code.words.len(), "el programa no depende del recorte");
    assert_eq!(specific.cycles, general.cycles, "ciclos");
    let register = general.layout.logits_register(&program);
    let read = |run: &mut GateRun, config: &DieConfig| -> Vec<u64> {
        (0..8u32).map(|i| {
            run.sim.set("vec_addr", (register as u64) << config.index_bits | i as u64).unwrap();
            run.sim.tick();
            run.sim.settle();
            run.sim.get("vec_out").unwrap()
        }).collect()
    };
    assert_eq!(read(&mut specific, &trimmed), read(&mut general, &c), "logits");
    let count = |config: DieConfig| netlist::die_design(config).flat_counts().iter().filter(|(k, _)| *k != "sram").map(|(_, v)| *v).sum::<u64>();
    let (before, after) = (count(c), count(trimmed));
    eprintln!("puertas y biestables: {before} generales, {after} para este programa");
    assert!(after < before);
    assert!(netlist::word_bits(&trimmed) < netlist::word_bits(&c));
    std::fs::remove_dir_all(dir).unwrap();
}

/// Runs the tokens with every matrix in the cells.
fn gates_in_cells(program: &compiler::Program, dir: &Path, c: DieConfig, tokens: &[u32]) -> GateRun {
    let weights = Weights::open(dir).unwrap();
    let layout = Layout::with_cells(program, &weights, c, CHIP_TOKENS).unwrap();
    let code = layout.microprogram(program).unwrap();
    // Each register takes only the vector memory its routines reach.
    let c = sim::chip::compact_die(c, &layout);
    let mut sim = Simulator::new(&die_design(c)).unwrap();
    load_microprogram(&mut sim, &c, &code);
    load_register_table(&mut sim, &c, &layout);
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
        let (codes, factors) = layout.cell_contents(&weights, cell).unwrap();
        assert!(codes.len() <= 1 << c.cell.addr_bits && 2 * factors.len() <= 1 << c.cell.addr_bits, "las celdas no alcanzan");
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
    let mut cycles = 0;
    for (position, token) in tokens.iter().enumerate() {
        cycles += enter(&mut sim, &code, ENTRY_TOKEN, *token, position as u32);
        if position + 1 == tokens.len() {
            cycles += enter(&mut sim, &code, ENTRY_LOGITS, *token, position as u32);
        }
    }
    GateRun { sim, layout, code, cycles }
}

#[test]
#[ignore = "lenta: cientos de miles de ciclos: solo con optimizaciones"]
fn the_cells_reach_the_same_bits_as_the_exact_machine() {
    let dir = std::env::temp_dir().join(format!("llm2chip-cells-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    make_checkpoint(&dir);
    let program = compile_qwen3_5(CONFIG).unwrap();
    let tokens = [3u32, 5, 1];
    let mut vm = ExactEngine::new(compile_qwen3_5(CONFIG).unwrap(), Weights::open(&dir).unwrap(), 16).unwrap();
    for (position, token) in tokens.iter().enumerate() {
        vm.step(*token, position + 1 == tokens.len()).unwrap();
    }
    let base = die_config();
    let serial = gates_after(&program, &dir, base, &tokens).cycles;
    let mut cycles = Vec::new();
    for (cells, lane_bits, cluster_bits) in [(1u32, 0u32, 0u32), (2, 0, 0), (8, 0, 0), (8, 2, 0), (4, 0, 1), (8, 1, 2), (16, 2, 2), (16, 3, 1)] {
        let c = DieConfig { cells, features: cell_die_config().features.with_lane_bits(lane_bits).with_cluster_bits(cluster_bits), ..cell_die_config() };
        let mut run = gates_in_cells(&program, &dir, c, &tokens);
        // Some buffers stay a slice per unit: what reads them takes the same slices.
        for (buffer, length) in program.buffers.iter().enumerate().filter(|(buffer, _)| !run.layout.is_spread(*buffer) && !run.layout.is_overwritten(*buffer)) {
            let gates = run.layout.register_of(buffer);
            let fast = vm.register_of(buffer);
            for index in 0..*length as u32 {
                let (mantissa, _) = read_block_value(&mut run.sim, c, gates, index);
                assert_eq!(mantissa, vm.mantissa(fast, index), "{cells} celdas, búfer {buffer}, elemento {index}");
            }
        }
        cycles.push(run.cycles);
    }
    eprintln!("ciclos: {serial} en serie; con 1, 2 y 8 celdas, y con 8 y 4 carriles, y con carriles y unidades {cycles:?}");
    assert!(cycles[2] < cycles[1] && cycles[1] < cycles[0], "{cycles:?}");
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
#[ignore = "lenta: cientos de miles de ciclos: solo con optimizaciones"]
fn quantisation_groups_cut_among_the_units_keep_the_bits() {
    // Groups of four columns over four units: each unit takes two columns of a group, and
    // its cells move on to the next scale and bias only where the whole group ends.
    GROUP.with(|g| g.set(4));
    let dir = std::env::temp_dir().join(format!("llm2chip-cut-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    make_checkpoint(&dir);
    let program = compile_qwen3_5(CONFIG).unwrap();
    let tokens = [3u32, 5, 1];
    let mut vm = ExactEngine::new(compile_qwen3_5(CONFIG).unwrap(), Weights::open(&dir).unwrap(), 16).unwrap();
    for (position, token) in tokens.iter().enumerate() {
        vm.step(*token, position + 1 == tokens.len()).unwrap();
    }
    // The factors applied at once, over two cycles in parts of groups, and over four.
    for (lane_bits, cluster_bits, steps) in [(2, 2, 1), (2, 2, 2), (1, 0, 4)] {
        let c = DieConfig { cells: 16, features: cell_die_config().features.with_lane_bits(lane_bits).with_cluster_bits(cluster_bits).with_factor_steps(steps), ..cell_die_config() };
        let mut run = gates_in_cells(&program, &dir, c, &tokens);
        for (buffer, length) in program.buffers.iter().enumerate().filter(|(buffer, _)| !run.layout.is_spread(*buffer) && !run.layout.is_overwritten(*buffer)) {
            let (gates, fast) = (run.layout.register_of(buffer), vm.register_of(buffer));
            for index in 0..*length as u32 {
                let (mantissa, _) = read_block_value(&mut run.sim, c, gates, index);
                assert_eq!(mantissa, vm.mantissa(fast, index), "{steps} ciclos por factor, búfer {buffer}, elemento {index}");
            }
        }
    }
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn two_bit_codes_share_each_word_four_at_a_time() {
    // Words of eight bits: each keeps four codes, and the cells pick them in turn.
    GROUP.with(|g| g.set(4));
    BITS.with(|b| b.set(2));
    let dir = std::env::temp_dir().join(format!("llm2chip-two-bits-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    make_checkpoint(&dir);
    let program = compile_qwen3_5(CONFIG).unwrap();
    let tokens = [3u32, 5, 1];
    let mut vm = ExactEngine::new(compile_qwen3_5(CONFIG).unwrap(), Weights::open(&dir).unwrap(), 16).unwrap();
    for (position, token) in tokens.iter().enumerate() {
        vm.step(*token, position + 1 == tokens.len()).unwrap();
    }
    for (lane_bits, cluster_bits) in [(2, 0), (1, 1)] {
        let c = DieConfig { cells: 16, features: cell_die_config().features.with_lane_bits(lane_bits).with_cluster_bits(cluster_bits), ..cell_die_config() };
        let mut run = gates_in_cells(&program, &dir, c, &tokens);
        // On a real chip, buffers that are never alive at once share a register.
        assert!((0..program.buffers.len()).any(|b| run.layout.is_overwritten(b)), "ningún registro compartido");
        for (buffer, length) in program.buffers.iter().enumerate().filter(|(buffer, _)| !run.layout.is_spread(*buffer) && !run.layout.is_overwritten(*buffer)) {
            let (gates, fast) = (run.layout.register_of(buffer), vm.register_of(buffer));
            for index in 0..*length as u32 {
                let (mantissa, _) = read_block_value(&mut run.sim, c, gates, index);
                assert_eq!(mantissa, vm.mantissa(fast, index), "{lane_bits} y {cluster_bits}, búfer {buffer}, elemento {index}");
            }
        }
    }
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn parallel_lanes_compute_the_same_bits_in_fewer_cycles() {
    let dir = std::env::temp_dir().join(format!("llm2chip-lanes-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    make_checkpoint(&dir);
    let program = compile_qwen3_5(CONFIG).unwrap();
    let c = die_config();
    let mut one = gates_after(&program, &dir, c, &[3, 5]);
    let register = one.layout.logits_register(&program);
    let read = |run: &mut GateRun, config: &DieConfig| -> Vec<u64> {
        (0..8u32).map(|i| {
            run.sim.set("vec_addr", (register as u64) << config.index_bits | i as u64).unwrap();
            run.sim.tick();
            run.sim.settle();
            run.sim.get("vec_out").unwrap()
        }).collect()
    };
    let expected = read(&mut one, &c);
    let mut cycles = vec![one.cycles];
    for lane_bits in [1, 2] {
        let wider = DieConfig { features: c.features.with_lane_bits(lane_bits), ..c };
        let mut run = gates_after(&program, &dir, wider, &[3, 5]);
        assert_eq!(read(&mut run, &wider), expected, "logits con {} carriles", 1 << lane_bits);
        cycles.push(run.cycles);
    }
    eprintln!("ciclos con 1, 2 y 4 carriles: {cycles:?}");
    // Vectors of 8 and 16 elements are too short for four lanes to beat two.
    assert!(cycles[1] < cycles[0] && cycles[2] < cycles[0], "{cycles:?}");
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn sliding_windows_and_biases_run_on_the_gates() {
    for family in ["gemma3_text", "qwen2", "deepseek_v3"] {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden/plain").join(family);
        let config = std::fs::read_to_string(dir.join("config.json")).unwrap();
        let program = compiler::compile_model(&config).unwrap();
        // More tokens than the window holds, so the ring wraps around.
        let tokens = [3u32, 1, 4, 1, 5];
        let c = DieConfig { index_bits: 10, microcode_bits: 18, ..die_config() };
        let mut run = gates_after(&program, &dir, c, &tokens);
        let mut vm = ExactEngine::new(compiler::compile_model(&config).unwrap(), Weights::open(&dir).unwrap(), 16).unwrap().with_weights_in_registers();
        for (position, token) in tokens.iter().enumerate() {
            vm.step(*token, position + 1 == tokens.len()).unwrap();
        }
        // Some buffers stay a slice per unit: what reads them takes the same slices.
        for (buffer, length) in program.buffers.iter().enumerate().filter(|(buffer, _)| !run.layout.is_spread(*buffer) && !run.layout.is_overwritten(*buffer)) {
            let gates = run.layout.register_of(buffer);
            let fast = vm.register_of(buffer);
            for index in 0..*length as u32 {
                let (mantissa, _) = read_block_value(&mut run.sim, c, gates, index);
                assert_eq!(mantissa, vm.mantissa(fast, index), "{family}, búfer {buffer}, elemento {index}");
            }
        }
    }
}

#[test]
fn a_die_with_only_the_registers_its_routines_reach_computes_the_same() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden/plain/llama");
    let config = std::fs::read_to_string(dir.join("config.json")).unwrap();
    let program = compiler::compile_model(&config).unwrap();
    let roomy = DieConfig { index_bits: 10, microcode_bits: 18, ..die_config() };
    let layout = Layout::new(&program, &Weights::open(&dir).unwrap(), roomy, Some(CHIP_TOKENS)).unwrap();
    layout.microprogram(&program).unwrap();
    let trimmed = sim::chip::with_registers_needed(roomy, &layout);
    // The working registers past what the routines use are left out, and the die is smaller.
    assert!(trimmed.reg_bits < roomy.reg_bits, "{} registros de {} usados", layout.registers_needed(), layout.registers_used());
    // Only the logic and the function tables its microprogram uses: a Llama needs four of eight.
    let code = Layout::new(&program, &Weights::open(&dir).unwrap(), trimmed, Some(CHIP_TOKENS)).unwrap().microprogram(&program).unwrap();
    let c = DieConfig { features: netlist::DieFeatures::of(&code), ..trimmed };
    assert!(netlist::die::function_address_bits(&c) < c.function_bits + c.function_select_bits, "tablas de función");
    let tokens = [3u32, 1, 4];
    let mut run = gates_after(&program, &dir, c, &tokens);
    let mut vm = ExactEngine::new(compiler::compile_model(&config).unwrap(), Weights::open(&dir).unwrap(), 16).unwrap().with_weights_in_registers();
    for (position, token) in tokens.iter().enumerate() {
        vm.step(*token, position + 1 == tokens.len()).unwrap();
    }
    for (buffer, length) in program.buffers.iter().enumerate().filter(|(buffer, _)| !run.layout.is_spread(*buffer) && !run.layout.is_overwritten(*buffer)) {
        let (gates, fast) = (run.layout.register_of(buffer), vm.register_of(buffer));
        for index in 0..*length as u32 {
            assert_eq!(read_block_value(&mut run.sim, c, gates, index).0, vm.mantissa(fast, index), "búfer {buffer}, elemento {index}");
        }
    }
    // A die with fewer registers than the routines reach is refused, not run wrong.
    let short = DieConfig { reg_bits: trimmed.reg_bits - 1, ..trimmed };
    let refused = Layout::new(&program, &Weights::open(&dir).unwrap(), short, Some(CHIP_TOKENS)).and_then(|l| l.microprogram(&program).map(|_| ()));
    assert!(refused.is_err());
}

#[test]
fn the_firmware_scores_the_draft_head_like_the_software() {
    use netlist::chip::{chip_info, ChipModel, GenerationSetup};
    use netlist::image::build_image;
    use netlist::rv32::Rv32Iss;
    use netlist::sampler::{exponential_table, SamplerConfig, SamplerModel};
    use netlist::soc::{FirmwareImage, SocBus, SocConfig};
    use netlist::Parameter;
    use sim::chip::{RunnerEngine, FLOW_DRAFT_REGISTER, FLOW_INDEX_BITS, FLOW_REGISTER_BITS};

    let dir = std::env::temp_dir().join(format!("llm2chip-mtp-firmware-{}", std::process::id()));
    draft_checkpoint(&dir);
    let config = draft_config();
    let engine = || Engine::new(compile_qwen3_5(&config).unwrap(), Weights::open(&dir).unwrap(), Numeric::Float).unwrap();
    let best = |v: &[f32]| v.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0 as u32;
    let (prompt, depth, count) = ([3u32, 7, 1], 2usize, 6usize);

    let (mut hits, mut tries) = ([0u32; 4], [0u32; 4]);
    let mut software = engine();
    let mut next = 0;
    for (i, id) in prompt.iter().enumerate() {
        let (logits, _) = software.step(*id, true).unwrap();
        if i + 1 == prompt.len() {
            next = best(logits.unwrap());
        }
        software.feed_draft_from_main();
        let following = if i + 1 == prompt.len() { next } else { prompt[i + 1] };
        software.draft(following, false).unwrap();
    }
    let mut output: Vec<u32> = Vec::new();
    let mut pending: Vec<(usize, Vec<u32>)> = Vec::new();
    for _ in 0..count {
        output.push(next);
        let (logits, _) = software.step(next, true).unwrap();
        next = best(logits.unwrap());
        let position = output.len();
        pending.retain(|(from, guesses)| {
            let ahead = position - from;
            if ahead >= 1 && ahead <= guesses.len() {
                tries[ahead - 1] += 1;
                hits[ahead - 1] += (guesses[ahead - 1] == output[position - 1]) as u32;
            }
            ahead < guesses.len()
        });
        let mark = software.draft_tokens();
        let (mut guess, mut guesses) = (next, Vec::new());
        software.feed_draft_from_main();
        for _ in 0..depth {
            guess = best(software.draft(guess, true).unwrap().unwrap());
            guesses.push(guess);
            software.feed_draft_from_draft();
        }
        software.truncate_draft(mark);
        pending.push((position + 1, guesses));
        software.feed_draft_from_main();
        software.draft(next, false).unwrap();
    }

    let firmware = FirmwareImage::from_elf(include_bytes!("../../../firmware/prebuilt/firmware.elf")).unwrap();
    let cell = die_config().cell;
    let die = DieConfig { reg_bits: FLOW_REGISTER_BITS, index_bits: FLOW_INDEX_BITS, microcode_bits: 2, ..die_config() };
    let _ = cell;
    let code = Microprogram { words: Vec::new(), entries: (0..6).collect(), parameters: vec![Parameter { register: 1, token: 1, position: 0, constant: 0, window_length: 0, window_slot: 0, window: 0 }, Parameter { register: 2, token: 0, position: 1, constant: 0, window_length: 0, window_slot: 0, window: 0 }] };
    let setup = GenerationSetup { max_new: count as u32 + 1, greedy: true, limit: 1, temperature: 256, seed: 1, raw_tokens: true, stop: Vec::new(), draft_depth: depth as u32, draft_logits: FLOW_DRAFT_REGISTER << FLOW_INDEX_BITS, think: None, think_budget: u32::MAX };
    let info = chip_info(&die, &code, 1, 8, &setup);
    let image = build_image(&firmware.program, &firmware.data, &info, &[], &[]).unwrap();
    let input: Vec<u8> = prompt.iter().flat_map(|t| t.to_le_bytes()).chain(u32::MAX.to_le_bytes()).collect();
    let mut chip_engine = engine();
    let runner = RunnerEngine::new(&mut chip_engine, 16, 0);
    let sampler = SamplerModel { config: SamplerConfig { logit_bits: 16, index_bits: 4, entries: 4, lanes: 1 }, rng: 1, table: exponential_table() };
    let mut bus = SocBus::booted_with(SocConfig { program_bits: 14, data_bits: 13, fifo_bits: 6 }, &image, &input, ChipModel::new(runner, code, sampler)).unwrap();
    let mut cpu = Rv32Iss::default();
    while !cpu.halted {
        cpu.step(&mut bus);
        assert!(cpu.cycle < 200_000_000, "el firmware no termina");
    }
    assert!(bus.peripheral.engine.error.is_none(), "{:?}", bus.peripheral.engine.error);
    let tokens: Vec<u32> = bus.tx.as_chunks::<4>().0.iter().map(|t| u32::from_le_bytes(*t)).collect();
    assert_eq!(tokens[..count], output[..count], "los tokens generados");
    assert_eq!(bus.peripheral.draft_tries[..depth], tries[..depth], "intentos");
    assert_eq!(bus.peripheral.draft_hits[..depth], hits[..depth], "aciertos");
    assert!(tries[0] > 0);
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn the_firmware_closes_the_reasoning_when_its_budget_runs_out() {
    use netlist::chip::{chip_info, ChipModel, GenerationSetup};
    use netlist::image::build_image;
    use netlist::rv32::Rv32Iss;
    use netlist::sampler::{exponential_table, SamplerConfig, SamplerModel};
    use netlist::soc::{FirmwareImage, SocBus, SocConfig};
    use netlist::Parameter;
    use sim::chip::{RunnerEngine, FLOW_DRAFT_REGISTER, FLOW_INDEX_BITS, FLOW_REGISTER_BITS};

    let dir = std::env::temp_dir().join(format!("llm2chip-think-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    make_checkpoint(&dir);
    let engine = || Engine::new(compile_qwen3_5(CONFIG).unwrap(), Weights::open(&dir).unwrap(), Numeric::Float).unwrap();
    let best = |v: &[f32]| v.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0 as u32;
    // The prompt ends opening the reasoning, as a chat template does.
    let (prompt, count, budget) = ([3u32, 5, 6], 6usize, 2u32);
    let generate = |close: u32, budget: u32| -> Vec<u32> {
        let mut software = engine();
        let mut next = 0;
        for (i, id) in prompt.iter().enumerate() {
            let (logits, _) = software.step(*id, i + 1 == prompt.len()).unwrap();
            next = logits.map_or(next, best);
        }
        let (mut output, mut spent, mut inside) = (Vec::new(), 0, true);
        for _ in 0..count {
            if inside && next != close && spent >= budget {
                next = close;
            }
            if next == close {
                inside = false;
            } else if inside {
                spent += 1;
            }
            output.push(next);
            next = best(software.step(next, true).unwrap().0.unwrap());
        }
        output
    };
    // A closing token the model would not choose by itself in the first steps.
    let free = generate(u32::MAX, u32::MAX);
    let close = (0..8).find(|t| !free[..=budget as usize].contains(t) && *t != 6).unwrap();
    let expected = generate(close, budget);
    assert_eq!(expected[budget as usize], close, "{expected:?}");

    let firmware = FirmwareImage::from_elf(include_bytes!("../../../firmware/prebuilt/firmware.elf")).unwrap();
    let die = DieConfig { reg_bits: FLOW_REGISTER_BITS, index_bits: FLOW_INDEX_BITS, microcode_bits: 2, ..die_config() };
    let code = Microprogram { words: Vec::new(), entries: (0..6).collect(), parameters: vec![Parameter { register: 1, token: 1, position: 0, constant: 0, window_length: 0, window_slot: 0, window: 0 }, Parameter { register: 2, token: 0, position: 1, constant: 0, window_length: 0, window_slot: 0, window: 0 }] };
    let setup = GenerationSetup { max_new: count as u32, greedy: true, limit: 1, temperature: 256, seed: 1, raw_tokens: true, stop: Vec::new(), draft_depth: 0, draft_logits: FLOW_DRAFT_REGISTER << FLOW_INDEX_BITS, think: Some((6, close)), think_budget: budget };
    let info = chip_info(&die, &code, 1, 8, &setup);
    let image = build_image(&firmware.program, &firmware.data, &info, &[], &[]).unwrap();
    let input: Vec<u8> = prompt.iter().flat_map(|t| t.to_le_bytes()).chain(u32::MAX.to_le_bytes()).collect();
    let mut chip_engine = engine();
    let runner = RunnerEngine::new(&mut chip_engine, 16, 0);
    let sampler = SamplerModel { config: SamplerConfig { logit_bits: 16, index_bits: 4, entries: 4, lanes: 1 }, rng: 1, table: exponential_table() };
    let mut bus = SocBus::booted_with(SocConfig { program_bits: 14, data_bits: 13, fifo_bits: 6 }, &image, &input, ChipModel::new(runner, code, sampler)).unwrap();
    let mut cpu = Rv32Iss::default();
    while !cpu.halted {
        cpu.step(&mut bus);
        assert!(cpu.cycle < 200_000_000, "el firmware no termina");
    }
    assert!(bus.peripheral.engine.error.is_none(), "{:?}", bus.peripheral.engine.error);
    let tokens: Vec<u32> = bus.tx.as_chunks::<4>().0.iter().map(|t| u32::from_le_bytes(*t)).collect();
    assert_eq!(tokens, expected, "el chip debe cerrar el razonamiento tras {budget} tokens");
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn the_cpu_profile_adds_up_and_tells_waiting_from_work() {
    use netlist::activity::{ChipActivity, CpuProfile, CpuTask};
    use netlist::chip::{chip_info, ChipModel, GenerationSetup, REG_ENTRY};
    use netlist::image::build_image;
    use netlist::rv32::Rv32Iss;
    use netlist::sampler::{exponential_table, SamplerConfig, SamplerModel};
    use netlist::soc::{FirmwareImage, SocBus, SocConfig};
    use netlist::Parameter;
    use sim::chip::{RunnerEngine, FLOW_DRAFT_REGISTER, FLOW_INDEX_BITS, FLOW_REGISTER_BITS};

    let dir = std::env::temp_dir().join(format!("llm2chip-profile-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    make_checkpoint(&dir);
    let elf = include_bytes!("../../../firmware/prebuilt/firmware.elf");
    let firmware = FirmwareImage::from_elf(elf).unwrap();
    let die = DieConfig { reg_bits: FLOW_REGISTER_BITS, index_bits: FLOW_INDEX_BITS, microcode_bits: 2, ..die_config() };
    let code = Microprogram { words: Vec::new(), entries: (0..6).collect(), parameters: vec![Parameter { register: 1, token: 1, position: 0, constant: 0, window_length: 0, window_slot: 0, window: 0 }, Parameter { register: 2, token: 0, position: 1, constant: 0, window_length: 0, window_slot: 0, window: 0 }] };
    let setup = GenerationSetup { max_new: 4, greedy: true, limit: 1, temperature: 256, seed: 1, raw_tokens: true, stop: Vec::new(), draft_depth: 0, draft_logits: FLOW_DRAFT_REGISTER << FLOW_INDEX_BITS, think: None, think_budget: u32::MAX };
    let info = chip_info(&die, &code, 1, 8, &setup);
    let image = build_image(&firmware.program, &firmware.data, &info, &[], &[]).unwrap();
    let input: Vec<u8> = [3u32, 5, 6].iter().flat_map(|t| t.to_le_bytes()).chain(u32::MAX.to_le_bytes()).collect();
    let mut engine = Engine::new(compile_qwen3_5(CONFIG).unwrap(), Weights::open(&dir).unwrap(), Numeric::Float).unwrap();
    let runner = RunnerEngine::new(&mut engine, 16, 1000);
    let sampler = SamplerModel { config: SamplerConfig { logit_bits: 16, index_bits: 4, entries: 4, lanes: 1 }, rng: 1, table: exponential_table() };
    let mut bus = SocBus::booted_with(SocConfig { program_bits: 14, data_bits: 13, fifo_bits: 6 }, &image, &input, ChipModel::new(runner, code, sampler)).unwrap();
    let mut cpu = Rv32Iss::default();
    let mut profile = CpuProfile::from_elf(elf);
    let mut orders = 0;
    // The steps cut as the simulator's server cuts them: a prompt token's step ends just
    // before the order of the next routine, an answer's when its token reaches the host.
    let (mut last, mut prompt_steps, mut answer_steps) = (ChipActivity::default(), Vec::new(), Vec::new());
    while !cpu.halted {
        let before = ChipActivity::of(&profile, &bus.peripheral);
        let pc = cpu.pc;
        cpu.step(&mut bus);
        let access = bus.last_access.take();
        orders += access.is_some_and(|a| a.write && a.address == 0x4000_0000 + 4 * REG_ENTRY) as u64;
        profile.record(pc, access, cpu.cycle);
        if prompt_steps.len() < bus.peripheral.engine.steps.min(3) {
            prompt_steps.push(before.since(&last));
            last = before;
        }
        if bus.tx.len() / 4 > answer_steps.len() {
            let now = ChipActivity::of(&profile, &bus.peripheral);
            answer_steps.push(now.since(&last));
            last = now;
        }
        assert!(cpu.cycle < 50_000_000, "el firmware no termina");
    }
    // Each prompt step runs its own token through the layers once, and no more.
    for (i, step) in prompt_steps.iter().enumerate() {
        assert_eq!(step.routines[ENTRY_TOKEN], (1, 1000), "paso {i} del prompt: {:?}", step.routines);
    }
    // An answer's step picks it: the logits, the sampler and the host; from the second on it
    // also runs the token before it.
    for (i, step) in answer_steps.iter().enumerate() {
        assert_eq!((step.routines[ENTRY_LOGITS].0, step.samples.0, step.routines[ENTRY_TOKEN].0), (1, 1, (i > 0) as u64), "respuesta {i}: {step:?}");
    }
    profile.settle();
    let (counts, chip) = (profile.counts, &bus.peripheral);
    // Every cycle is somewhere, and every order the CPU gave is a routine the die ran.
    assert_eq!(counts.total_cycles(), cpu.cycle);
    assert_eq!(chip.routines.iter().map(|r| r.0).sum::<u64>(), orders);
    assert_eq!(chip.routines[ENTRY_TOKEN].0, 3 + 4, "un paso por token del prompt y por cada generado");
    // The CPU waits while the die works: at most the few cycles of one poll apart per order.
    let waited = counts.cycles_of(CpuTask::WaitDie);
    assert!(waited.abs_diff(chip.die_busy) <= 16 * orders, "espera {waited}, die {}", chip.die_busy);
    for task in [CpuTask::Receive, CpuTask::Order, CpuTask::Sample, CpuTask::Send] {
        assert!(counts.cycles_of(task) > 0, "{task:?}: {:?}", counts.entries());
    }
    // The tokens arrive ready, so nothing is tokenized.
    assert_eq!(counts.cycles_of(CpuTask::Tokenize), 0);
    eprintln!("{:?}", counts.entries());
    std::fs::remove_dir_all(dir).unwrap();
}

/// Three heads to each key head, in attention and in the delta rule: their units are laid
/// out a power of two apart, with the units in between sitting out.
const GROUPED: &str = r#"{
  "model_type": "qwen3_5_text",
  "hidden_size": 8, "intermediate_size": 16, "vocab_size": 8,
  "num_attention_heads": 6, "num_key_value_heads": 2, "head_dim": 4,
  "linear_num_key_heads": 2, "linear_key_head_dim": 4,
  "linear_num_value_heads": 6, "linear_value_head_dim": 4,
  "linear_conv_kernel_dim": 4, "partial_rotary_factor": 0.5,
  "layer_types": ["linear_attention", "full_attention"],
  "rope_parameters": { "rope_theta": 10000.0 }
}"#;

fn grouped_checkpoint(dir: &Path) {
    let mut rng = Rng(0x0F1E_2D3C_4B5A_6978);
    let mut w = Writer::default();
    w.matrix("model.embed_tokens", 8, 8, &mut rng);
    w.matrix("lm_head", 8, 8, &mut rng);
    w.floats("model.norm.weight", &[1.0; 8]);
    for (layer, full) in [(0, false), (1, true)] {
        let p = |s: &str| format!("model.layers.{layer}.{s}");
        w.floats(&p("input_layernorm.weight"), &[1.0; 8]);
        w.floats(&p("post_attention_layernorm.weight"), &[1.0; 8]);
        w.matrix(&p("mlp.gate_proj"), 16, 8, &mut rng);
        w.matrix(&p("mlp.up_proj"), 16, 8, &mut rng);
        w.matrix(&p("mlp.down_proj"), 8, 16, &mut rng);
        if full {
            w.matrix(&p("self_attn.q_proj"), 48, 8, &mut rng);
            w.matrix(&p("self_attn.k_proj"), 8, 8, &mut rng);
            w.matrix(&p("self_attn.v_proj"), 8, 8, &mut rng);
            w.matrix(&p("self_attn.o_proj"), 8, 24, &mut rng);
            w.floats(&p("self_attn.q_norm.weight"), &[1.0; 4]);
            w.floats(&p("self_attn.k_norm.weight"), &[1.0; 4]);
        } else {
            w.matrix(&p("linear_attn.in_proj_qkv"), 40, 8, &mut rng);
            w.matrix(&p("linear_attn.in_proj_z"), 24, 8, &mut rng);
            w.matrix(&p("linear_attn.in_proj_a"), 6, 8, &mut rng);
            w.matrix(&p("linear_attn.in_proj_b"), 6, 8, &mut rng);
            w.matrix(&p("linear_attn.out_proj"), 8, 24, &mut rng);
            let taps: Vec<f32> = (0..160).map(|i| if i % 4 == 3 { 0.9 } else { 0.05 }).collect();
            w.floats(&p("linear_attn.conv1d.weight"), &taps);
            w.floats(&p("linear_attn.A_log"), &[-1.0, -0.5, -0.7, -0.3, -0.9, -0.6]);
            w.floats(&p("linear_attn.dt_bias"), &[1.0; 6]);
            w.floats(&p("linear_attn.norm.weight"), &[1.0; 4]);
        }
    }
    w.write(&dir.join("model.safetensors"));
}

#[test]
#[ignore = "lenta: cientos de miles de ciclos: solo con optimizaciones"]
fn heads_in_groups_of_three_run_on_units_a_power_of_two_apart() {
    let dir = std::env::temp_dir().join(format!("llm2chip-grouped-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    grouped_checkpoint(&dir);
    let program = compile_qwen3_5(GROUPED).unwrap();
    let tokens = [3u32, 5, 1];
    let mut vm = ExactEngine::new(compile_qwen3_5(GROUPED).unwrap(), Weights::open(&dir).unwrap(), 16).unwrap();
    for (position, token) in tokens.iter().enumerate() {
        vm.step(*token, position + 1 == tokens.len()).unwrap();
    }
    // Eight units take one part of each head; sixteen take two.
    for cluster_bits in [3, 4] {
        let c = DieConfig { cells: 4 << cluster_bits, features: cell_die_config().features.with_lane_bits(2).with_cluster_bits(cluster_bits), ..cell_die_config() };
        let mut run = gates_in_cells(&program, &dir, c, &tokens);
        // Some buffers stay a slice per unit: what reads them takes the same slices.
        for (buffer, length) in program.buffers.iter().enumerate().filter(|(buffer, _)| !run.layout.is_spread(*buffer) && !run.layout.is_overwritten(*buffer)) {
            let (gates, fast) = (run.layout.register_of(buffer), vm.register_of(buffer));
            for index in 0..*length as u32 {
                let (mantissa, _) = read_block_value(&mut run.sim, c, gates, index);
                assert_eq!(mantissa, vm.mantissa(fast, index), "{} unidades, búfer {buffer}, elemento {index}", 1 << cluster_bits);
            }
        }
    }
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn an_image_rotates_by_three_positions_on_the_chip_as_in_float() {
    // Every dimension rotates, so the row's and column's frequencies show.
    let config = CONFIG.replace("\"partial_rotary_factor\": 0.25", "\"partial_rotary_factor\": 1.0");
    let dir = std::env::temp_dir().join(format!("llm2chip-mrope-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    make_checkpoint(&dir);
    let section = [2, 1, 1];
    let picture: Vec<Vec<f32>> = (0..4).map(|k| (0..8).map(|i| ((i * 7 + k * 3) % 11) as f32 / 11.0 - 0.45).collect()).collect();
    // Text, a 2×2 picture at time 1 with its rows and columns, then text from position 3.
    let grid = [[1u64, 1, 1], [1, 1, 2], [1, 2, 1], [1, 2, 2]];
    let run = |engine: &mut dyn sim::Runner| -> Vec<f32> {
        engine.set_mrope(section);
        engine.step(3, false).unwrap();
        for (vector, positions) in picture.iter().zip(grid) {
            engine.step_embedding(vector, positions, false).unwrap();
        }
        engine.continue_text_at(3);
        engine.step(5, true).unwrap().0.unwrap().to_vec()
    };
    let mut float = Engine::new(compile_qwen3_5(&config).unwrap(), Weights::open(&dir).unwrap(), Numeric::Float).unwrap();
    let expected = run(&mut float);
    let mut exact = ExactEngine::new(compile_qwen3_5(&config).unwrap(), Weights::open(&dir).unwrap(), 16).unwrap();
    let got = run(&mut exact);
    let scale = expected.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let error = got.iter().zip(&expected).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    assert!(error < 0.02 * scale, "{error} sobre {scale}: {got:?} {expected:?}");

    // The same picture as if all its tokens sat at one place gives other logits: the rows
    // and columns really move them.
    let mut flat = ExactEngine::new(compile_qwen3_5(&config).unwrap(), Weights::open(&dir).unwrap(), 16).unwrap();
    flat.set_mrope(section);
    flat.step(3, false).unwrap();
    for (k, vector) in picture.iter().enumerate() {
        flat.step_embedding(vector, [k as u64 + 1; 3], false).unwrap();
    }
    flat.continue_text_at(5);
    let other = flat.step(5, true).unwrap().unwrap();
    let moved = other.iter().zip(&got).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    assert!(moved > 4.0 * error, "{moved} frente a {error}");
    std::fs::remove_dir_all(dir).ok();
}

#[test]
fn the_hidden_state_an_image_head_reads_comes_from_the_chip_as_in_float() {
    let dir = std::env::temp_dir().join(format!("llm2chip-hidden-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    make_checkpoint(&dir);
    let input: Vec<f32> = (0..8).map(|i| (i as f32 - 3.5) / 6.0).collect();
    let mut float = Engine::new(compile_qwen3_5(CONFIG).unwrap(), Weights::open(&dir).unwrap(), Numeric::Float).unwrap();
    let mut exact = ExactEngine::new(compile_qwen3_5(CONFIG).unwrap(), Weights::open(&dir).unwrap(), 16).unwrap();
    for (token, embedding) in [(3u32, None), (5, None), (0, Some(input.as_slice()))] {
        let expected = float.step_hidden(token, embedding).unwrap();
        let got = exact.step_hidden(token, embedding).unwrap();
        let scale = expected.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let error = got.iter().zip(&expected).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(error < 0.02 * scale, "{error} sobre {scale}: {got:?} {expected:?}");
    }
    std::fs::remove_dir_all(dir).ok();
}

/// A die of `c` with `layout`'s weights and `code` loaded, as the firmware would leave it.
fn loaded_die(layout: &Layout, code: &Microprogram, weights: &Weights, c: DieConfig) -> Simulator {
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

#[test]
fn two_groups_of_two_dies_run_a_token_over_their_links() {
    let dir = std::env::temp_dir().join(format!("llm2chip-groups-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    make_checkpoint(&dir);
    let program = compile_qwen3_5(CONFIG).unwrap();
    let weights = Weights::open(&dir).unwrap();
    let base = DieConfig { microcode_bits: 15, ..cell_die_config() };
    let share = sim::chip::ShareOut::new(&program, &weights, 4, CHIP_TOKENS, base.cell.activation_bits, sim::chip::Sharing::Mixed(2)).unwrap();
    let mut first = None;
    let mut dies: Vec<(Simulator, Microprogram)> = (0..4)
        .map(|chip| {
            let mut layout = Layout::with_cells(&program, &weights, base, CHIP_TOKENS).unwrap();
            let code = share.microprogram(&mut layout, &program, chip).unwrap();
            let die = loaded_die(&layout, &code, &weights, base);
            if chip == 0 {
                first = Some((layout.logits_register(&program), sim::chip::compact_die(base, &layout)));
            }
            (die, code)
        })
        .collect();
    // The link goes round all four; each group's last closes its ring to its first.
    let wires = [(0, 1, "link"), (1, 2, "link"), (2, 3, "link"), (3, 0, "link"), (1, 0, "loop"), (3, 2, "loop")];
    for routine in [ENTRY_TOKEN, ENTRY_LOGITS] {
        for (sim, code) in &mut dies {
            sim.set("start", 0).unwrap();
            sim.set("param_write", 1).unwrap();
            for parameter in &code.parameters {
                sim.set("param_index", parameter.register as u64).unwrap();
                sim.set("param_data", parameter.value(3, 0) as u64).unwrap();
                sim.tick();
            }
            sim.set("param_write", 0).unwrap();
            sim.set("entry", code.entries[routine] as u64).unwrap();
            sim.set("start", 1).unwrap();
            sim.tick();
            sim.set("start", 0).unwrap();
        }
        let mut cycles = 0u64;
        let mut traffic = vec![0u64; wires.len()];
        while dies.iter().any(|(sim, _)| sim.get("halted").unwrap() == 0) {
            let outputs: Vec<_> = wires.iter().map(|(from, to, p)| {
                let (a, b) = (&dies[*from].0, &dies[*to].0);
                (a.get(&format!("{p}_tx_valid")).unwrap(), a.get_wide(&format!("{p}_tx_bits")).unwrap(), b.get(&format!("{p}_rx_ready")).unwrap(), b.get(&format!("{p}_rx_ack")).unwrap())
            }).collect();
            for (count, (valid, _, ready, _)) in traffic.iter_mut().zip(&outputs) {
                *count += valid & ready;
            }
            for ((from, to, p), (valid, bits, ready, ack)) in wires.iter().zip(outputs) {
                dies[*to].0.set(&format!("{p}_rx_valid"), valid).unwrap();
                dies[*to].0.set_wide(&format!("{p}_rx_bits"), bits).unwrap();
                dies[*from].0.set(&format!("{p}_tx_ready"), ready).unwrap();
                dies[*from].0.set(&format!("{p}_tx_ack"), ack).unwrap();
            }
            for (sim, _) in &mut dies {
                sim.tick();
            }
            cycles += 1;
            if cycles >= 300_000 {
                let pcs: Vec<String> = dies.iter().map(|(sim, code)| {
                    let pc = sim.snapshot("").and_then(|nets| nets.get("pc").map(|bits| usize::from_str_radix(bits, 2).unwrap())).unwrap_or(0);
                    let word = code.words.get(pc).map(|w| format!("op {} send {} loop {}", w.op, w.link_send, w.link_loop)).unwrap_or_default();
                    format!("pc {pc} ({word}), detenido {}", sim.get("halted").unwrap())
                }).collect();
                panic!("rutina {routine}: los dies no terminan: {pcs:?}; palabras por cable {:?}", wires.iter().zip(&traffic).collect::<Vec<_>>());
            }
        }
        eprintln!("rutina {routine}: {cycles} ciclos");
    }
    // The logits the first chip made are, bit for bit, those of one die on its own.
    let mut alone = gates_in_cells(&program, &dir, base, &[3]);
    let (logits, c) = first.unwrap();
    let single = alone.layout.logits_register(&program);
    let alone_c = sim::chip::compact_die(base, &alone.layout);
    for index in 0..8 {
        let (got, _) = read_block_value(&mut dies[0].0, c, logits, index);
        let (want, _) = read_block_value(&mut alone.sim, alone_c, single, index);
        assert_eq!(got, want, "logit {index}");
    }
    std::fs::remove_dir_all(dir).ok();
}
