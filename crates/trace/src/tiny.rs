//! The small Qwen3.5-shaped models the gate-level tests and the viewer run: their
//! configurations, dies and checkpoints, written with seeded random weights.

use netlist::{CellConfig, DieConfig};
use std::collections::BTreeMap;
use std::path::Path;

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
pub fn cell_die_config() -> DieConfig {
    let base = die_config();
    DieConfig { cell: CellConfig { accumulator_bits: 40, addr_bits: 12, ..base.cell }, scale_bits: 21, index_bits: 12, features: base.features.with_key_shift(4), ..base }
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

/// Three heads to each key head, in attention and in the delta rule: their units are laid
/// out a power of two apart, with the units in between sitting out.
pub const GROUPED: &str = r#"{
  "model_type": "qwen3_5_text",
  "hidden_size": 8, "intermediate_size": 16, "vocab_size": 8,
  "num_attention_heads": 6, "num_key_value_heads": 2, "head_dim": 4,
  "linear_num_key_heads": 2, "linear_key_head_dim": 4,
  "linear_num_value_heads": 6, "linear_value_head_dim": 4,
  "linear_conv_kernel_dim": 4, "partial_rotary_factor": 0.5,
  "layer_types": ["linear_attention", "full_attention"],
  "rope_parameters": { "rope_theta": 10000.0 }
}"#;

/// The grouped model with attention heads of `head_dim` elements.
pub fn grouped_config(head_dim: usize) -> String {
    GROUPED.replace(r#""num_key_value_heads": 2, "head_dim": 4"#, &format!(r#""num_key_value_heads": 2, "head_dim": {head_dim}"#))
}

pub fn grouped_checkpoint(dir: &Path, head_dim: usize) {
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
            w.matrix(&p("self_attn.q_proj"), 12 * head_dim, 8, &mut rng);
            w.matrix(&p("self_attn.k_proj"), 2 * head_dim, 8, &mut rng);
            w.matrix(&p("self_attn.v_proj"), 2 * head_dim, 8, &mut rng);
            w.matrix(&p("self_attn.o_proj"), 8, 6 * head_dim, &mut rng);
            w.floats(&p("self_attn.q_norm.weight"), &vec![1.0; head_dim]);
            w.floats(&p("self_attn.k_norm.weight"), &vec![1.0; head_dim]);
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
