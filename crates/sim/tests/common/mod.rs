//! The small Qwen-shaped model that the gate-level tests run.
#![allow(dead_code)]

use netlist::{CellConfig, DieConfig};
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::OnceLock;

pub const CONFIG: &str = r#"{
  "model_type": "qwen3_5_text",
  "hidden_size": 8, "intermediate_size": 16, "vocab_size": 8,
  "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 8,
  "linear_num_key_heads": 1, "linear_key_head_dim": 4,
  "linear_num_value_heads": 2, "linear_value_head_dim": 4,
  "linear_conv_kernel_dim": 4, "partial_rotary_factor": 0.25,
  "layer_types": ["linear_attention", "full_attention"],
  "rope_parameters": { "rope_theta": 10000.0 }
}"#;

/// Eight letters, one token each: enough for the chip tests' text in very little memory.
pub fn tiny_tokenizer() -> sim::tokenizer::TokenizerTables {
    let json = r#"{"added_tokens":[],"normalizer":null,"pre_tokenizer":null,"post_processor":null,"decoder":{"type":"Fuse"},
      "model":{"type":"BPE","vocab":{"h":0,"y":1,"c":2,"d":3,"e":4,"f":5,"g":6,"o":7},"merges":[]}}"#;
    sim::tokenizer::TokenizerTables::from_json(json, 64, 256).unwrap()
}

pub fn die_config() -> DieConfig {
    DieConfig {
        cells: 1,
        cell: CellConfig { weight_bits: 8, activation_bits: 16, accumulator_bits: 32, mac_stages: 1, adder_stages: 1, addr_bits: 3, coding: Default::default(), exponent_bits: 0, multiplier: netlist::multiplier::MultiplierKind::Array, fixed_weights: false, ecc: false, skip_zeros: false },
        scale_bits: 16,
        reg_bits: 9,
        index_bits: 9,
        microcode_bits: 18,
        broadcast_stages: 1,
        function_bits: 12,
        function_select_bits: 3, features: Default::default(),
    }
}

/// A die with room for a model's matrices in its cells.
#[allow(dead_code)]
pub fn cell_die_config() -> DieConfig {
    let base = die_config();
    DieConfig { cell: CellConfig { accumulator_bits: 40, addr_bits: 12, ..base.cell }, scale_bits: 21, index_bits: 12, features: base.features.with_key_shift(4), ..base }
}

/// The smallest die that runs the test model with eight cells, whose memories fit in
/// an FPGA's block RAM: 2^9 registers of 2^7 words and 2^13 words of microcode.
#[allow(dead_code)]
pub fn fpga_die_config() -> DieConfig {
    let base = cell_die_config();
    DieConfig { cells: 8, cell: CellConfig { addr_bits: 8, ..base.cell }, index_bits: 7, microcode_bits: 13, ..base }
}

pub struct Rng(pub u64);

impl Rng {
    pub fn unit(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    }
}

thread_local! {
    /// Columns that share a scale and a bias in the matrices this thread writes.
    pub static GROUP: std::cell::Cell<usize> = const { std::cell::Cell::new(2) };
    /// Bits of each code, packed as MLX packs them.
    pub static BITS: std::cell::Cell<usize> = const { std::cell::Cell::new(4) };
}

#[derive(Default)]
pub struct Writer {
    header: BTreeMap<String, serde_json::Value>,
    data: Vec<u8>,
}

impl Writer {
    pub fn tensor(&mut self, name: &str, dtype: &str, shape: &[usize], bytes: Vec<u8>) {
        let start = self.data.len();
        self.data.extend(bytes);
        self.header.insert(name.to_string(), serde_json::json!({ "dtype": dtype, "shape": shape, "data_offsets": [start, self.data.len()] }));
    }

    pub fn floats(&mut self, name: &str, values: &[f32]) {
        let bytes = values.iter().flat_map(|v| half::f16::from_f32(*v).to_le_bytes()).collect();
        self.tensor(name, "F16", &[values.len()], bytes);
    }

    /// One matrix in MLX's affine format: 4-bit codes in groups of two columns.
    pub fn matrix(&mut self, name: &str, rows: usize, cols: usize, rng: &mut Rng) {
        let (group, bits) = (GROUP.with(|g| g.get()), BITS.with(|b| b.get()));
        let groups = cols / group;
        let mut codes = Vec::with_capacity(rows * cols);
        let (mut scales, mut biases) = (Vec::new(), Vec::new());
        for _ in 0..rows {
            for _ in 0..groups {
                scales.extend(half::f16::from_f32(0.02 + 0.03 * rng.unit().abs()).to_le_bytes());
                biases.extend(half::f16::from_f32(-0.2 * rng.unit().abs() - 0.05).to_le_bytes());
                for _ in 0..group {
                    codes.push(((rng.unit() + 1.0) * ((1 << bits) as f32 / 2.0 - 0.5)) as u8 & ((1 << bits) - 1) as u8);
                }
            }
        }
        let words = (cols * bits).div_ceil(32);
        let mut packed = vec![0u8; rows * words * 4];
        for (row, codes) in codes.chunks(cols).enumerate() {
            for (i, code) in codes.iter().enumerate() {
                packed[row * words * 4 + i * bits / 8] |= code << (i * bits % 8);
            }
        }
        self.tensor(&format!("{name}.weight"), "U32", &[rows, words], packed);
        self.tensor(&format!("{name}.scales"), "F16", &[rows, groups], scales);
        self.tensor(&format!("{name}.biases"), "F16", &[rows, groups], biases);
    }

    pub fn write(self, path: &Path) {
        let header = serde_json::to_vec(&self.header).unwrap();
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend(header);
        file.extend(self.data);
        std::fs::write(path, file).unwrap();
    }
}

pub fn make_checkpoint(dir: &Path) {
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


/// The firmware, built once per test run: tests running at the same time used to start
/// their own builds, and one could find the binary half-written by another.
pub fn firmware() -> Option<netlist::soc::FirmwareImage> {
    static FIRMWARE: OnceLock<Option<netlist::soc::FirmwareImage>> = OnceLock::new();
    FIRMWARE
        .get_or_init(|| {
            let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../firmware");
            let built = std::process::Command::new("cargo").current_dir(&dir).env_remove("RUSTFLAGS").env_remove("CARGO_ENCODED_RUSTFLAGS").args(["build", "--release"]).output().ok()?;
            if !built.status.success() {
                eprintln!("firmware omitido: {}", String::from_utf8_lossy(&built.stderr));
                return None;
            }
            let elf = std::fs::read(dir.join("target/riscv32im-unknown-none-elf/release/firmware")).ok()?;
            Some(netlist::soc::FirmwareImage::from_elf(&elf).expect("the firmware's ELF"))
        })
        .clone()
}

/// The toy image encoder: hidden 8 in 2 heads of 4, 2 blocks, patches of 2×2 pixels in 2
/// frames, 2×2 patches to a token of 8 values, the toy language model's width.
#[allow(dead_code)]
pub const VISION: (usize, usize, usize, usize, usize, usize, usize) = (8, 2, 2, 16, 2, 2, 2);

/// Writes the toy encoder next to a checkpoint: its config, the preprocessor's size limits
/// (images of 4 to 64 tokens) and its weights as plain floats.
#[allow(dead_code)]
pub fn make_vision_checkpoint(dir: &Path) {
    let (hidden, heads, depth, intermediate, patch, temporal, merge) = VISION;
    let side = 3;
    let config = serde_json::json!({
        "model_type": "qwen3_5",
        "image_token_id": 7, "vision_start_token_id": 5, "vision_end_token_id": 6,
        "text_config": { "rope_parameters": { "mrope_section": [1, 1, 0] } },
        "vision_config": { "depth": depth, "hidden_size": hidden, "num_heads": heads, "intermediate_size": intermediate, "patch_size": patch, "temporal_patch_size": temporal, "spatial_merge_size": merge, "out_hidden_size": 8, "num_position_embeddings": side * side, "rope_parameters": { "rope_theta": 100.0 } }
    });
    std::fs::write(dir.join("config.json"), serde_json::to_string_pretty(&config).unwrap()).unwrap();
    let pixels = (patch * merge) * (patch * merge);
    std::fs::write(dir.join("preprocessor_config.json"), serde_json::json!({ "size": { "shortest_edge": 4 * pixels, "longest_edge": 64 * pixels } }).to_string()).unwrap();

    let mut rng = Rng(0x0BAD_5EED_1234_5678);
    let mut w = Writer::default();
    let mut put = |w: &mut Writer, name: &str, shape: &[usize], spread: f32, offset: f32| {
        let values: Vec<f32> = (0..shape.iter().product()).map(|_| offset + spread * rng.unit()).collect();
        let bytes = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        w.tensor(&format!("vision_tower.{name}"), "F32", shape, bytes);
    };
    let input = 3 * temporal * patch * patch;
    put(&mut w, "patch_embed.proj.weight", &[hidden, input], 0.1, 0.0);
    put(&mut w, "patch_embed.proj.bias", &[hidden], 0.1, 0.0);
    put(&mut w, "pos_embed.weight", &[side * side, hidden], 0.2, 0.0);
    for block in 0..depth {
        let b = |s: &str| format!("blocks.{block}.{s}");
        put(&mut w, &b("norm1.weight"), &[hidden], 0.2, 1.0);
        put(&mut w, &b("norm1.bias"), &[hidden], 0.1, 0.0);
        put(&mut w, &b("attn.qkv.weight"), &[3 * hidden, hidden], 0.3, 0.0);
        put(&mut w, &b("attn.qkv.bias"), &[3 * hidden], 0.1, 0.0);
        put(&mut w, &b("attn.proj.weight"), &[hidden, hidden], 0.3, 0.0);
        put(&mut w, &b("attn.proj.bias"), &[hidden], 0.1, 0.0);
        put(&mut w, &b("norm2.weight"), &[hidden], 0.2, 1.0);
        put(&mut w, &b("norm2.bias"), &[hidden], 0.1, 0.0);
        put(&mut w, &b("mlp.linear_fc1.weight"), &[intermediate, hidden], 0.3, 0.0);
        put(&mut w, &b("mlp.linear_fc1.bias"), &[intermediate], 0.1, 0.0);
        put(&mut w, &b("mlp.linear_fc2.weight"), &[hidden, intermediate], 0.2, 0.0);
        put(&mut w, &b("mlp.linear_fc2.bias"), &[hidden], 0.1, 0.0);
    }
    let merged = hidden * merge * merge;
    put(&mut w, "merger.norm.weight", &[hidden], 0.2, 1.0);
    put(&mut w, "merger.norm.bias", &[hidden], 0.1, 0.0);
    put(&mut w, "merger.linear_fc1.weight", &[merged, merged], 0.2, 0.0);
    put(&mut w, "merger.linear_fc1.bias", &[merged], 0.1, 0.0);
    put(&mut w, "merger.linear_fc2.weight", &[8, merged], 0.2, 0.0);
    put(&mut w, "merger.linear_fc2.bias", &[8], 0.1, 0.0);
    w.write(&dir.join("model-vision.safetensors"));
}

/// A small picture, `width` × `height` RGB, the same every time.
#[allow(dead_code)]
pub fn test_picture(width: usize, height: usize) -> Vec<u8> {
    (0..width * height * 3).map(|i| ((i * 37 + i / 3 * 11) % 251) as u8).collect()
}

/// Runs a whole chip in Icarus Verilog, from its exported netlist, as the gate tests run it
/// in this crate's simulator: its output bytes and cycles. `None` without Icarus Verilog.
#[allow(dead_code)]
pub fn chip_in_iverilog(design: &netlist::Design, image: &[u32], input: &[u8], limit: u64, tag: &str) -> Option<(Vec<u8>, u64)> {
    std::process::Command::new("iverilog").arg("-V").output().ok()?;
    let dir = std::env::temp_dir().join(format!("llm2chip-iverilog-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let hex = |values: Vec<String>| values.join("\n") + "\n";
    std::fs::write(dir.join("image.hex"), hex(image.iter().map(|w| format!("{w:08x}")).collect())).unwrap();
    std::fs::write(dir.join("input.hex"), hex(input.iter().map(|b| format!("{b:02x}")).collect())).unwrap();
    let source = dir.join("chip.v");
    std::fs::write(&source, format!("{}\n{}", netlist::to_verilog(design), netlist::chip_testbench(design, image.len(), input.len(), limit))).unwrap();
    let binary = dir.join("chip.vvp");
    let compile = std::process::Command::new("iverilog").args(["-g2005", "-s", "testbench", "-o"]).arg(&binary).arg(&source).output().unwrap();
    assert!(compile.status.success(), "{}", String::from_utf8_lossy(&compile.stderr));
    let run = std::process::Command::new("vvp").current_dir(&dir).arg(&binary).output().unwrap();
    let text = String::from_utf8_lossy(&run.stdout);
    let output = text.lines().filter_map(|l| l.strip_prefix("B ")).map(|b| b.trim().parse().unwrap()).collect();
    let cycles = text.lines().find_map(|l| l.strip_prefix("CICLOS ")).map(|c| c.trim().parse().unwrap());
    std::fs::remove_dir_all(&dir).ok();
    Some((output, cycles.unwrap_or_else(|| panic!("Icarus Verilog did not finish: {}", text.lines().last().unwrap_or_default()))))
}

// The die driven on its gates as its firmware would, for the tests that need it.

#[allow(dead_code)]
pub fn mask(c: &DieConfig) -> u64 {
    (1u64 << c.cell.activation_bits) - 1
}
#[allow(dead_code)]
pub fn load_assembly(sim: &mut netlist::Simulator, c: &DieConfig, initial: &std::collections::HashMap<(u32, u32), i64>) {
    sim.set("load_vec", 1).unwrap();
    for ((reg, index), value) in initial {
        sim.set("vec_addr", ((*reg as u64) << c.index_bits) | *index as u64).unwrap();
        sim.set("vec_data", *value as u64 & mask(c)).unwrap();
        sim.tick();
    }
    sim.set("load_vec", 0).unwrap();
}
/// Where each register starts in a compact vector memory, before anything goes into it.
#[allow(dead_code)]
pub fn load_register_table(sim: &mut netlist::Simulator, c: &DieConfig, layout: &sim::lower::Layout) {
    sim.set("load_base", 1).unwrap();
    for (register, entry) in sim::chip::register_table(c, layout).iter().enumerate() {
        sim.set("base_addr", register as u64).unwrap();
        sim.set("base_data", *entry as u64).unwrap();
        sim.tick();
    }
    sim.set("load_base", 0).unwrap();
}
/// Writes the microprogram, once, as the load controller does at start-up.
#[allow(dead_code)]
pub fn load_microprogram(sim: &mut netlist::Simulator, c: &DieConfig, code: &netlist::Microprogram) {
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
#[allow(dead_code)]
pub fn enter(sim: &mut netlist::Simulator, code: &netlist::Microprogram, routine: usize, token: u32, position: u32) -> u64 {
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
        assert!(cycles < 20_000_000, "the sequencer does not stop");
    }
    cycles
}

/// Runs the whole chip in gates, the host sending a byte whenever the chip can take it.
#[allow(dead_code)]
pub fn gates(design: &netlist::Design, image: &[u32], input: &[u8], limit: u64) -> (Vec<u8>, u64) {
    let mut sim = netlist::Simulator::new(design).unwrap();
    sim.set("reset", 1).unwrap();
    sim.set("cpu_hold", 1).unwrap();
    sim.tick();
    sim.set("reset", 0).unwrap();
    for _ in 0..=netlist::chip::RESET_RELEASE {
        sim.tick();
    }
    let mut pending = None;
    let mut sent = 0;
    let mut step = |sim: &mut netlist::Simulator, pending: &mut Option<u32>| {
        sim.set("image_resp_valid", pending.is_some() as u64).unwrap();
        sim.set("image_resp_data", pending.unwrap_or(0) as u64).unwrap();
        let offer = sent < input.len();
        sim.set("host_rx_valid", offer as u64).unwrap();
        sim.set("host_rx_data", input.get(sent).copied().unwrap_or(0) as u64).unwrap();
        sim.settle();
        if offer && sim.get("host_rx_ready").unwrap() == 1 {
            sent += 1;
        }
        *pending = (sim.get("image_req_valid").unwrap() == 1).then(|| image[sim.get("image_req_addr").unwrap() as usize]);
    };
    let mut boot = 0;
    loop {
        step(&mut sim, &mut pending);
        if sim.get("booting").unwrap() == 0 {
            break;
        }
        sim.tick();
        boot += 1;
        assert!(boot < 1_000_000, "boot does not finish");
    }
    sim.set("cpu_hold", 0).unwrap();
    sim.set("host_tx_ready", 1).unwrap();
    let mut output = Vec::new();
    for cycle in 0..limit {
        step(&mut sim, &mut pending);
        if sim.get("host_tx_valid").unwrap() == 1 {
            output.push(sim.get("host_tx_data").unwrap() as u8);
        }
        if sim.get("halted").unwrap() == 1 {
            return (output, cycle + 1);
        }
        sim.tick();
    }
    sim.set("debug_select", netlist::chip::DEBUG_CPU_PC).unwrap();
    sim.settle();
    panic!("the firmware did not finish within {limit} cycles: output {output:?}, CPU at {:#x}, {sent} of {} bytes sent", sim.get("debug_value").unwrap(), input.len());
}
