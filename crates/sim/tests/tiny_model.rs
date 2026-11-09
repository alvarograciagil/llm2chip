//! A complete Qwen3.5-shaped model, weights and all, executed on the gates.

use compiler::compile_qwen3_5;
use netlist::block::tables;
use netlist::demo::read_block_value;
use netlist::{die_design, CellConfig, DieConfig, Microprogram, Simulator};
use sim::lower::{Layout, ENTRY_DRAFT, ENTRY_DRAFT_LOGITS, ENTRY_FEED_FROM_MAIN, ENTRY_LOGITS, ENTRY_TOKEN};
use sim::{ExactEngine, Engine, Numeric, Weights};
use std::collections::BTreeMap;
use std::path::Path;

const CONFIG: &str = r#"{
  "model_type": "qwen3_5_text",
  "hidden_size": 8, "intermediate_size": 16, "vocab_size": 8,
  "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 8,
  "linear_num_key_heads": 1, "linear_key_head_dim": 4,
  "linear_num_value_heads": 2, "linear_value_head_dim": 4,
  "linear_conv_kernel_dim": 4, "partial_rotary_factor": 0.25,
  "layer_types": ["linear_attention", "full_attention"],
  "rope_parameters": { "rope_theta": 10000.0 }
}"#;

fn die_config() -> DieConfig {
    DieConfig {
        cells: 1,
        cell: CellConfig { weight_bits: 8, activation_bits: 16, accumulator_bits: 32, mac_stages: 1, adder_stages: 1, addr_bits: 3, coding: Default::default(), exponent_bits: 0, multiplier: netlist::multiplier::MultiplierKind::Array },
        scale_bits: 16,
        reg_bits: 9,
        index_bits: 9,
        microcode_bits: 18,
        broadcast_stages: 1,
        function_bits: 12,
        function_select_bits: 3,
    }
}

struct Rng(u64);

impl Rng {
    fn unit(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    }
}

#[derive(Default)]
struct Writer {
    header: BTreeMap<String, serde_json::Value>,
    data: Vec<u8>,
}

impl Writer {
    fn tensor(&mut self, name: &str, dtype: &str, shape: &[usize], bytes: Vec<u8>) {
        let start = self.data.len();
        self.data.extend(bytes);
        self.header.insert(name.to_string(), serde_json::json!({ "dtype": dtype, "shape": shape, "data_offsets": [start, self.data.len()] }));
    }

    fn floats(&mut self, name: &str, values: &[f32]) {
        let bytes = values.iter().flat_map(|v| half::f16::from_f32(*v).to_le_bytes()).collect();
        self.tensor(name, "F16", &[values.len()], bytes);
    }

    /// One matrix in MLX's affine format: 4-bit codes in groups of two columns.
    fn matrix(&mut self, name: &str, rows: usize, cols: usize, rng: &mut Rng) {
        const GROUP: usize = 2;
        let groups = cols / GROUP;
        let mut codes = Vec::with_capacity(rows * cols);
        let (mut scales, mut biases) = (Vec::new(), Vec::new());
        for _ in 0..rows {
            for _ in 0..groups {
                scales.extend(half::f16::from_f32(0.02 + 0.03 * rng.unit().abs()).to_le_bytes());
                biases.extend(half::f16::from_f32(-0.2 * rng.unit().abs() - 0.05).to_le_bytes());
                for _ in 0..GROUP {
                    codes.push(((rng.unit() + 1.0) * 7.5) as u8 & 0xF);
                }
            }
        }
        let mut packed = vec![0u8; rows * cols.div_ceil(8) * 4];
        for (i, code) in codes.iter().enumerate() {
            packed[i / 2] |= code << (4 * (i & 1));
        }
        self.tensor(&format!("{name}.weight"), "U32", &[rows, cols.div_ceil(8)], packed);
        self.tensor(&format!("{name}.scales"), "F16", &[rows, groups], scales);
        self.tensor(&format!("{name}.biases"), "F16", &[rows, groups], biases);
    }

    fn write(self, path: &Path) {
        let header = serde_json::to_vec(&self.header).unwrap();
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend(header);
        file.extend(self.data);
        std::fs::write(path, file).unwrap();
    }
}

fn make_checkpoint(dir: &Path) {
    let mut rng = Rng(0x1234_5678_9ABC_DEF1);
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
            w.matrix(&p("self_attn.q_proj"), 32, 8, &mut rng);
            w.matrix(&p("self_attn.k_proj"), 8, 8, &mut rng);
            w.matrix(&p("self_attn.v_proj"), 8, 8, &mut rng);
            w.matrix(&p("self_attn.o_proj"), 8, 16, &mut rng);
            w.floats(&p("self_attn.q_norm.weight"), &[1.0; 8]);
            w.floats(&p("self_attn.k_norm.weight"), &[1.0; 8]);
        } else {
            w.matrix(&p("linear_attn.in_proj_qkv"), 16, 8, &mut rng);
            w.matrix(&p("linear_attn.in_proj_z"), 8, 8, &mut rng);
            w.matrix(&p("linear_attn.in_proj_a"), 2, 8, &mut rng);
            w.matrix(&p("linear_attn.in_proj_b"), 2, 8, &mut rng);
            w.matrix(&p("linear_attn.out_proj"), 8, 8, &mut rng);
            let taps: Vec<f32> = (0..64).map(|i| if i % 4 == 3 { 0.9 } else { 0.05 }).collect();
            w.floats(&p("linear_attn.conv1d.weight"), &taps);
            w.floats(&p("linear_attn.A_log"), &[-1.0, -0.5]);
            w.floats(&p("linear_attn.dt_bias"), &[1.0, 1.0]);
            w.floats(&p("linear_attn.norm.weight"), &[1.0; 4]);
        }
    }
    w.write(&dir.join("model.safetensors"));
}

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
