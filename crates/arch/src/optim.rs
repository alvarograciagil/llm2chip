use serde::Serialize;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Maturity {
    Proven,
    Plausible,
    Experimental,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum QualityRisk {
    Lossless,
    Measure,
    High,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct OptInfo {
    pub id: &'static str,
    pub name: &'static str,
    pub maturity: Maturity,
    pub quality_risk: QualityRisk,
    pub what: &'static str,
    pub effect: &'static str,
}

const fn opt(
    id: &'static str,
    name: &'static str,
    maturity: Maturity,
    quality_risk: QualityRisk,
    what: &'static str,
    effect: &'static str,
) -> OptInfo {
    OptInfo { id, name, maturity, quality_risk, what, effect }
}

use Maturity::*;
use QualityRisk::*;

/// Effects are assumptions to be replaced by measurements as the project matures.
const CATALOG: &[OptInfo] = &[
    opt("external_embedding", "Embedding in external DRAM", Proven, Lossless,
        "The embedding table is read only one row per token, so it does not need to sit next to the compute.",
        "Takes the embedding out of the SRAM and adds a DRAM controller."),
    opt("zero_skipping", "Skip zero weights", Proven, Lossless,
        "The multiplier is not switched on when the weight is zero.",
        "Forces zero gating in the cell. The custom circuit only picks it if it saves something with the format's real zeros."),
    opt("compiler_repair", "Repair by the compiler", Proven, Lossless,
        "The compiler places the weights avoiding each chip's defective cells.",
        "Redundancy from 3 % to 1 %."),
    opt("idle_die_power_off", "Power off idle dies", Proven, Lossless,
        "With a single user only one die works at a time; the rest cut the power to their logic.",
        "−90 % logic leakage on idle dies."),
    opt("state_checkpoints", "Conversation checkpoints", Proven, Lossless,
        "Keeps the state in cheap DRAM so the common prefix of each request is not read again.",
        "Adds a DRAM controller; removes re-reading the prompt."),
    opt("refresh_without_ecc", "No ECC, periodic rewrite", Plausible, Measure,
        "A flipped bit in a weight is barely noticed. The image is rewritten from time to time.",
        "−11 % SRAM area."),
    opt("multiplier_free", "No multiplier per weight", Plausible, Lossless,
        "The multiples of each activation are precomputed and each weight only chooses which to add.",
        "Forces the cell without a multiplier. The custom circuit only picks it if it pays for the wires of the multiples."),
    opt("distributed_arithmetic", "Distributed arithmetic", Plausible, Lossless,
        "The activation is processed bit by bit with precomputed sums, as in classic FPGA filters.",
        "Multiplier at 15 % of the area, ×4 weight cycles, 50 % of the energy."),
    opt("weight_compression", "Weight compression", Plausible, Lossless,
        "Weights are read in order, so a streaming decompressor per cell is feasible.",
        "Weights at 75 % (to be measured); +8 µm² per cell."),
    opt("low_voltage_sram", "Low-voltage weight SRAM", Plausible, Measure,
        "Each weight is read once per token: a lower voltage and slow blocks can be used.",
        "−30 % read energy and −50 % leakage."),
    opt("stacked_sram_3d", "SRAM stacked in 3D", Plausible, Lossless,
        "An SRAM die bonded on top of the logic by hybrid bonding, as in stacked caches.",
        "Weight SRAM footprint at 50 %. Higher cost and heat."),
    opt("weight_pruning", "Pruning small weights", Plausible, Measure,
        "The smallest weights are removed and only what remains is stored.",
        "Weights at 85 %. Quality must be validated again."),
    opt("kv_compression", "KV cache compression", Plausible, Measure,
        "Each token keeps keys and values with a truncated mantissa and a compressed exponent, as in Cassandra.",
        "KV cache at 50 % (to be measured). Helps mostly with long contexts."),
    opt("gain_cell_edram", "Gain-cell eDRAM", Experimental, Measure,
        "A cell of few transistors whose constant reading could act as refresh.",
        "×2 density for the weights. Limited availability in advanced nodes."),
    opt("analog_cim", "Analog compute in SRAM", Experimental, High,
        "The weighted sum is done in the SRAM array itself, with currents.",
        "Multiplier at 30 % of the area and 20 % of the energy; adds noise to the results."),
    opt("csd_weights", "Weights in signed digits", Experimental, High,
        "With weights fixed in ROM, each weight is rewritten with few non-zero digits and the multiplier becomes a few additions.",
        "Multiplication logic at 23 % (CAxCNN, measured on CNNs). Only with ROM."),
    opt("low_rank", "Low-rank factorization", Experimental, High,
        "Large matrices are approximated as the product of two smaller ones.",
        "Weights at 60 %. It changes the model and quality must be measured."),
    opt("trimmed_head", "Trimmed vocabulary output", Experimental, Measure,
        "Only the logits of the most frequent tokens are computed and the rest are resolved apart.",
        "lm_head at 30 %. Rare tokens lose precision."),
    opt("sparse_attention", "Sparse attention", Experimental, Measure,
        "The query is compared only with a fraction of the KV cache chosen by an index.",
        "Attention compute and KV reads at 25 %."),
];

pub fn catalog() -> &'static [OptInfo] {
    CATALOG
}
