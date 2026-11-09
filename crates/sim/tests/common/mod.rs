//! The small Qwen-shaped model that the gate-level tests run.
#![allow(dead_code)]

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

pub struct Rng(pub u64);

impl Rng {
    pub fn unit(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    }
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

