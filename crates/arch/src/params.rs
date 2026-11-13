use netlist::format::{WeightCoding, WeightFormat, TABLE_BITS};
use netlist::multiplier::MultiplierKind;
use netlist::CellConfig;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DieMode {
    /// Each die is built for exactly what it holds of this model.
    #[default]
    Custom,
    /// One die design, repeated: more dies fit larger models, up to the limits below.
    Identical,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CircuitGoal {
    /// The product of area and energy.
    #[default]
    Balanced,
    Area,
    Energy,
}

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

/// How the chips share the model.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChipSplit {
    /// Each chip runs a run of whole layers and passes the hidden vector on: one chip
    /// works at a time on a single user's token.
    Layers,
    /// Every chip holds a share of every layer and they all work on each layer at once,
    /// adding their partial results over the links twice a layer.
    Tensor,
    /// Groups of `tensor_group` chips share each layer, and the groups run in a chain.
    Mixed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LinkKind {
    /// UCIe inside each package and SerDes between packages, as the area requires; on an
    /// open shuttle, the pins.
    Auto,
    Ucie,
    Serdes,
    /// Plain I/O pins on a board: what chips from an open shuttle, each in its own package,
    /// have to talk to each other.
    Pins,
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
    /// With an FPGA node, the board the design is built for: see `fpga::boards`.
    pub fpga_board: String,
    /// A compute cell laid out with OpenROAD: which one (`cell_key`), and its measured area
    /// and clock, used instead of the estimates for that same cell.
    pub measured_cell: String,
    pub measured_cell_um2: f64,
    pub measured_cell_fmax_ghz: f64,

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

    /// The generator picks multiplier, stages and zero skipping for these parameters.
    pub auto_circuit: bool,
    pub circuit_goal: CircuitGoal,
    /// When above zero, the multipliers per die follow from this speed and the model.
    pub target_tok_s: f64,
    pub die_mode: DieMode,
    /// With identical dies: the deepest, widest model the design must accept.
    pub max_layers: u32,
    pub max_hidden_size: u32,
    pub max_vocabulary: u32,
    pub multiplier_kind: MultiplierKind,
    pub mac_stages: u32,
    pub adder_stages: u32,
    pub noc_stages: u32,

    pub weight_coding: WeightCoding,
    pub weight_bits: u32,
    /// Float: exponent bits. Posit: `es`. Logarithmic: fraction bits.
    pub weight_exponent_bits: u32,
    /// Whether each group also stores a bias, as MLX does; GGUF's Q4_0 does not.
    pub weight_bias: bool,
    /// Float groups: a full float scale (NVFP4) instead of a power of two (MXFP4).
    pub weight_fine_scale: bool,
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
    /// Copies of the vector unit, which compute several heads at once.
    #[serde(default = "default_vector_units")]
    pub vector_units_per_die: u64,
    pub vector_bits: u32,
    pub function_units_per_die: u32,
    /// Logits the sampling unit takes in per cycle.
    pub sampler_lanes: u32,
    /// Candidates the sampling unit keeps: the largest top-k it can draw from.
    pub sampler_entries: u32,
    /// log2 of the entries in each function table: its resolution.
    pub function_table_bits: u32,
    /// Cycles a cell takes to apply a group's scale and bias, a part of each a cycle,
    /// while it adds up the next group. One is a whole multiplier.
    pub factor_steps: u32,
    /// Activations each link between dies carries per cycle.
    pub link_words: u32,

    pub weight_backend: WeightBackend,
    pub kv_backend: KvBackend,
    pub hbm_stacks: u32,
    pub stacked_dram_tbps: f64,
    pub ddr_gbps: f64,
    pub ddr_capacity_gb: f64,

    pub link: LinkKind,
    pub chip_split: ChipSplit,
    /// Chips that share each layer when the split is mixed.
    pub tensor_group: u32,
    pub load_interface: LoadInterface,

    #[serde(deserialize_with = "optimization_ids")]
    pub optimizations: BTreeSet<String>,
}

/// The optimizations of a design, with the ids designs saved before they were in English
/// read as their English ones.
fn optimization_ids<'de, D: serde::Deserializer<'de>>(deserializer: D) -> Result<BTreeSet<String>, D::Error> {
    const RENAMED: [(&str, &str); 19] = [("embedding_externo", "external_embedding"), ("salto_ceros", "zero_skipping"), ("reparacion_por_compilador", "compiler_repair"), ("apagado_dies_inactivos", "idle_die_power_off"), ("checkpoint_estado", "state_checkpoints"), ("sin_ecc_con_refresco", "refresh_without_ecc"), ("sin_multiplicadores", "multiplier_free"), ("aritmetica_distribuida", "distributed_arithmetic"), ("compresion_pesos", "weight_compression"), ("sram_baja_tension", "low_voltage_sram"), ("sram_apilada_3d", "stacked_sram_3d"), ("poda_pesos", "weight_pruning"), ("compresion_kv", "kv_compression"), ("edram_ganancia", "gain_cell_edram"), ("cim_analogico", "analog_cim"), ("pesos_csd", "csd_weights"), ("baja_rango", "low_rank"), ("cabeza_recortada", "trimmed_head"), ("atencion_dispersa", "sparse_attention")];
    let ids = BTreeSet::<String>::deserialize(deserializer)?;
    Ok(ids.into_iter().map(|id| RENAMED.iter().find(|(old, _)| *old == id).map_or(id, |(_, new)| new.to_string())).collect())
}

fn default_vector_units() -> u64 {
    128
}

impl Default for ChipParams {
    fn default() -> Self {
        Self {
            node: "n6".into(),
            fpga_board: String::new(),
            measured_cell: String::new(),
            measured_cell_um2: 0.0,
            measured_cell_fmax_ghz: 0.0,
            max_params_g: 35.0,
            context_tokens: 8192,
            mtp_enabled: false,
            mtp_depth: 3,
            include_vision: false,
            dies: 0,
            die_max_mm2: 815.0,
            macs_per_die: 136_000,
            freq_ghz: 1.0,
            auto_circuit: true,
            circuit_goal: CircuitGoal::Balanced,
            target_tok_s: 0.0,
            die_mode: DieMode::Custom,
            max_layers: 128,
            max_hidden_size: 8192,
            max_vocabulary: 1 << 20,
            multiplier_kind: MultiplierKind::Wallace,
            mac_stages: 2,
            adder_stages: 2,
            noc_stages: 1,
            weight_coding: WeightCoding::Affine,
            weight_bits: 4,
            weight_exponent_bits: 5,
            weight_bias: true,
            weight_fine_scale: false,
            use_model_mix: true,
            group_size: 32,
            scale_bits: 16,
            activation_bits: 16,
            accumulator_bits: 32,
            delta_state_bits: 32,
            kv_bits: 16,
            vector_lanes_per_die: 32_768,
            vector_units_per_die: default_vector_units(),
            vector_bits: 16,
            function_units_per_die: 16,
            sampler_lanes: 64,
            sampler_entries: 64,
            function_table_bits: 12,
            factor_steps: 1,
            link_words: 16,
            weight_backend: WeightBackend::Sram,
            kv_backend: KvBackend::Sram,
            hbm_stacks: 8,
            stacked_dram_tbps: 100.0,
            ddr_gbps: 77.0,
            ddr_capacity_gb: 64.0,
            link: LinkKind::Auto,
            chip_split: ChipSplit::Layers,
            tensor_group: 2,
            load_interface: LoadInterface::Nvme,
            optimizations: BTreeSet::new(),
        }
    }
}

impl ChipParams {
    /// Chips that share each layer, out of `dies`.
    pub fn tensor_chips(&self, dies: u32) -> u32 {
        match self.chip_split {
            ChipSplit::Layers => 1,
            ChipSplit::Tensor => dies.max(1),
            ChipSplit::Mixed => self.tensor_group.clamp(1, dies.max(1)),
        }
    }

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

    /// Whether each group's scale needs a multiplier, not just a shifter.
    pub fn scale_is_multiplied(&self) -> bool {
        self.weight_coding != WeightCoding::Float || self.weight_fine_scale
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
            multiplier: if self.has("multiplier_free") { MultiplierKind::Multiples } else { self.multiplier_kind },
            skip_zeros: self.has("zero_skipping"),
            fixed_weights: self.weight_backend == WeightBackend::Rom,
            ecc: self.weight_backend == WeightBackend::Sram && !self.has("refresh_without_ecc"),
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
            fine_scale: self.weight_fine_scale,
            keep_source: self.mixes_widths(),
        }
    }
}
