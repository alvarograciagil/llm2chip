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
    let mut vm = ExactEngine::new(compile_qwen3_5(config).unwrap(), Weights::open(dir).unwrap(), c.cell.activation_bits).unwrap();
    for (position, token) in tokens.iter().enumerate() {
        vm.step(*token, position + 1 == tokens.len()).unwrap();
    }
    for (buffer, length) in program.buffers.iter().enumerate() {
        let gates = layout.register_of(buffer);
        let fast = vm.register_of(buffer);
        for index in 0..*length as u32 {
            let (mantissa, _) = read_block_value(sim, c, gates, index);
            assert_eq!(mantissa, vm.mantissa(fast, index), "búfer {buffer}, elemento {index}");
        }
    }
}

#[test]
fn draft_head_runs_on_the_gates() {
    let config = CONFIG.replacen(r#""model_type": "qwen3_5_text","#, r#""model_type": "qwen3_5_text", "mtp_num_hidden_layers": 1,"#, 1);
    let dir = std::env::temp_dir().join(format!("llm2chip-mtp-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    make_checkpoint(&dir);
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

    let mut vm = ExactEngine::new(compile_qwen3_5(&config).unwrap(), Weights::open(&dir).unwrap(), 16).unwrap();
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
    let mut vm = ExactEngine::new(compile_qwen3_5(CONFIG).unwrap(), Weights::open(&dir).unwrap(), 16).unwrap();
    for (position, token) in tokens.iter().enumerate() {
        vm.step(*token, position + 1 == tokens.len()).unwrap();
    }
    for (buffer, length) in program.buffers.iter().enumerate() {
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
#[cfg_attr(debug_assertions, ignore = "cientos de miles de ciclos: solo con optimizaciones")]
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

#[test]
#[cfg_attr(debug_assertions, ignore = "cientos de miles de ciclos: solo con optimizaciones")]
fn sliding_windows_and_biases_run_on_the_gates() {
    for family in ["gemma3_text", "qwen2", "deepseek_v3"] {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden/plain").join(family);
        let config = std::fs::read_to_string(dir.join("config.json")).unwrap();
        let program = compiler::compile_model(&config).unwrap();
        // More tokens than the window holds, so the ring wraps around.
        let tokens = [3u32, 1, 4, 1, 5];
        let c = DieConfig { index_bits: 10, microcode_bits: 18, ..die_config() };
        let mut run = gates_after(&program, &dir, c, &tokens);
        let mut vm = ExactEngine::new(compiler::compile_model(&config).unwrap(), Weights::open(&dir).unwrap(), 16).unwrap();
        for (position, token) in tokens.iter().enumerate() {
            vm.step(*token, position + 1 == tokens.len()).unwrap();
        }
        for (buffer, length) in program.buffers.iter().enumerate() {
            let gates = run.layout.register_of(buffer);
            let fast = vm.register_of(buffer);
            for index in 0..*length as u32 {
                let (mantissa, _) = read_block_value(&mut run.sim, c, gates, index);
                assert_eq!(mantissa, vm.mantissa(fast, index), "{family}, búfer {buffer}, elemento {index}");
            }
        }
    }
}
