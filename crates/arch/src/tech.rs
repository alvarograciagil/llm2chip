use serde::{Deserialize, Serialize};
use std::sync::OnceLock;

/// Process parameters. Until calibration every value is an assumption, and the
/// report labels the numbers derived from them accordingly.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TechNode {
    pub id: String,
    pub label: String,
    pub sram_cell_um2: f64,
    /// Bigger means small SRAM macros lose more area to their periphery.
    pub sram_periphery_k: f64,
    pub rom_mbit_mm2: f64,
    pub fa_um2: f64,
    pub ff_um2: f64,
    pub and_um2: f64,
    pub or_um2: f64,
    pub xor_um2: f64,
    pub not_um2: f64,
    pub mux_um2: f64,
    pub fa_delay_ns: f64,
    pub ff_overhead_ns: f64,
    pub fmax_ghz: f64,
    pub e_fa_pj: f64,
    pub e_sram_fj_bit: f64,
    pub e_wire_pj_bit_mm: f64,
    pub sram_leak_pw_bit: f64,
    pub logic_leak_w_mm2: f64,
    pub die_max_mm2: f64,
    pub fixed_die_mm2: f64,
    pub cpu_mm2: f64,
    pub io_mm2: f64,
}

impl TechNode {
    /// Effective density in Mbit/mm² for an SRAM macro of `kbit` kilobits.
    pub fn sram_density(&self, kbit: f64) -> f64 {
        let efficiency = 1.0 / (1.0 + self.sram_periphery_k / kbit.max(0.25).sqrt());
        efficiency / self.sram_cell_um2
    }
}

const SOURCES: [&str; 7] = [
    include_str!("../../../tech/n6.json"),
    include_str!("../../../tech/n5.json"),
    include_str!("../../../tech/n3.json"),
    include_str!("../../../tech/asap7.json"),
    include_str!("../../../tech/sky130.json"),
    include_str!("../../../tech/gf180.json"),
    include_str!("../../../tech/fpga_u250.json"),
];

pub fn nodes() -> &'static [TechNode] {
    static NODES: OnceLock<Vec<TechNode>> = OnceLock::new();
    NODES.get_or_init(|| {
        SOURCES
            .iter()
            .map(|s| serde_json::from_str(s).expect("archivo de tecnología inválido"))
            .collect()
    })
}

pub fn node(id: &str) -> Option<&'static TechNode> {
    nodes().iter().find(|n| n.id == id)
}
