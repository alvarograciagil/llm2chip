use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WeightBackend {
    Sram,
    Rom,
    StackedDram,
    Hbm4,
    ExternalDdr,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KvBackend {
    Sram,
    Hbm4,
    /// Cache and recurrent state both live in the external memory.
    ExternalDdr,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LinkKind {
    Ucie,
    Serdes,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LoadInterface {
    Uart,
    Nvme,
    Pcie5,
}

impl LoadInterface {
    pub fn bytes_per_second(self) -> f64 {
        match self {
            LoadInterface::Uart => 11_520.0,
            LoadInterface::Nvme => 14e9,
            LoadInterface::Pcie5 => 50e9,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct ChipParams {
    pub node: String,

    pub max_params_g: f64,
    pub context_tokens: u64,
    pub mtp_enabled: bool,
    pub mtp_depth: u32,
    pub include_vision: bool,

    /// 0 lets the estimator pick the smallest number that fits.
    pub dies: u32,
    pub die_max_mm2: f64,
    pub macs_per_die: u64,
    pub freq_ghz: f64,

    pub mac_stages: u32,
    pub adder_stages: u32,
    pub noc_stages: u32,

    pub weight_bits: u32,
    /// Keeps the model's own 8-bit tensors, as real mixed checkpoints do.
    pub use_model_mix: bool,
    /// Weights sharing one scale and bias. 0 means no per-group affine terms.
    pub group_size: u32,
    pub scale_bits: u32,
    pub activation_bits: u32,
    pub accumulator_bits: u32,
    pub delta_state_bits: u32,
    pub kv_bits: u32,

    pub vector_lanes_per_die: u64,
    pub vector_bits: u32,
    pub function_units_per_die: u32,

    pub weight_backend: WeightBackend,
    pub kv_backend: KvBackend,
    pub hbm_stacks: u32,
    pub stacked_dram_tbps: f64,
    pub ddr_gbps: f64,
    pub ddr_capacity_gb: f64,

    pub link: LinkKind,
    pub load_interface: LoadInterface,

    pub optimizations: BTreeSet<String>,
}

impl Default for ChipParams {
    fn default() -> Self {
        Self {
            node: "n6".into(),
            max_params_g: 35.0,
            context_tokens: 8192,
            mtp_enabled: false,
            mtp_depth: 3,
            include_vision: false,
            dies: 0,
            die_max_mm2: 815.0,
            macs_per_die: 136_000,
            freq_ghz: 1.0,
            mac_stages: 2,
            adder_stages: 2,
            noc_stages: 1,
            weight_bits: 4,
            use_model_mix: true,
            group_size: 32,
            scale_bits: 16,
            activation_bits: 16,
            accumulator_bits: 32,
            delta_state_bits: 32,
            kv_bits: 16,
            vector_lanes_per_die: 32_768,
            vector_bits: 16,
            function_units_per_die: 16,
            weight_backend: WeightBackend::Sram,
            kv_backend: KvBackend::Sram,
            hbm_stacks: 8,
            stacked_dram_tbps: 100.0,
            ddr_gbps: 77.0,
            ddr_capacity_gb: 64.0,
            link: LinkKind::Ucie,
            load_interface: LoadInterface::Nvme,
            optimizations: BTreeSet::new(),
        }
    }
}

impl ChipParams {
    pub fn has(&self, optimization: &str) -> bool {
        self.optimizations.contains(optimization)
    }
}
