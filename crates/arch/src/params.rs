use netlist::format::{WeightCoding, WeightFormat, TABLE_BITS};
use netlist::CellConfig;
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

    pub weight_coding: WeightCoding,
    pub weight_bits: u32,
    /// Float: exponent bits. Posit: `es`. Logarithmic: fraction bits.
    pub weight_exponent_bits: u32,
    /// Whether each group also stores a bias, as MLX does; GGUF's Q4_0 does not.
    pub weight_bias: bool,
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
            weight_coding: WeightCoding::Affine,
            weight_bits: 4,
            weight_exponent_bits: 5,
            weight_bias: true,
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

    /// True when per-group factors exist in this format.
    pub fn uses_groups(&self) -> bool {
        self.group_size > 0 && matches!(self.weight_coding, WeightCoding::Affine | WeightCoding::Float)
    }

    /// Stored bits per weight that the group's factors add.
    pub fn group_overhead_bits(&self, group: u32) -> f64 {
        if !self.uses_groups() || group == 0 {
            return 0.0;
        }
        let factors = if self.weight_coding == WeightCoding::Affine && self.weight_bias { 2.0 } else { 1.0 };
        factors * self.scale_bits as f64 / group as f64
    }

    pub fn mixes_widths(&self) -> bool {
        self.use_model_mix && self.weight_coding == WeightCoding::Affine
    }

    /// Width of the cell's stored code.
    pub fn cell_weight_bits(&self) -> u32 {
        if self.mixes_widths() {
            self.weight_bits.max(8)
        } else {
            self.weight_bits
        }
    }

    /// The accumulator the format needs, never narrower than the one requested.
    pub fn effective_accumulator(&self) -> u32 {
        self.cell_config(1).accumulator_bits
    }

    pub fn cell_config(&self, addr_bits: u32) -> CellConfig {
        let weight_bits = self.cell_weight_bits().clamp(2, 32);
        let mut config = CellConfig {
            weight_bits,
            activation_bits: self.activation_bits,
            accumulator_bits: self.accumulator_bits,
            mac_stages: self.mac_stages,
            adder_stages: self.adder_stages,
            addr_bits,
            coding: self.weight_coding,
            exponent_bits: self.weight_coding.legal_exponent_bits(weight_bits, self.weight_exponent_bits),
        };
        config.accumulator_bits = config.accumulator_bits.max(config.required_accumulator());
        config
    }

    /// Bits of the lookup table each cell needs, for formats that have one.
    pub fn cell_table_bits(&self) -> f64 {
        if self.weight_coding == WeightCoding::Codebook {
            (1u64 << self.cell_weight_bits().min(16)) as f64 * TABLE_BITS as f64
        } else {
            0.0
        }
    }

    pub fn weight_format(&self) -> WeightFormat {
        WeightFormat {
            coding: self.weight_coding,
            bits: self.weight_bits.clamp(1, 32),
            exponent_bits: self.weight_coding.legal_exponent_bits(self.weight_bits.max(2), self.weight_exponent_bits),
            bias: self.weight_bias,
            group_size: if self.uses_groups() { self.group_size } else { 0 },
            keep_source: self.mixes_widths(),
        }
    }
}
