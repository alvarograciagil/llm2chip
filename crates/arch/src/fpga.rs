//! FPGA boards a design can target: what each offers, and how much of it the generated
//! chip takes, from synthesis of that chip rather than from a scale of an ASIC process.

use crate::estimate::{Confidence, Metric, Report, Severity, Violation};
use crate::params::{ChipParams, KvBackend, WeightBackend};
use serde::Serialize;
use std::sync::OnceLock;

/// One board: its FPGA, what it holds, and where to read about it. Figures from the
/// makers; prices are approximate and change with the shop and the region.
#[derive(Clone, Debug, Serialize)]
pub struct FpgaBoard {
    pub id: &'static str,
    pub maker: &'static str,
    pub name: &'static str,
    pub fpga: &'static str,
    /// The maker's own page for the board.
    pub url: &'static str,
    /// A photo on the maker's site, where one could be checked.
    pub image: Option<&'static str>,
    /// Look-up tables, in thousands, of the kind the family has: Intel's logic elements
    /// count like four-input tables.
    pub luts_k: f64,
    pub lut: &'static str,
    /// Block memory on the FPGA, UltraRAM included.
    pub bram_mbit: f64,
    pub dsp: u64,
    /// Memory on the board, and roughly what it delivers per second.
    pub ddr_gb: f64,
    pub ddr_gbps: f64,
    /// A clock the generated logic reaches without hand placement.
    pub clock_mhz: f64,
    /// Whether that clock comes from placing and routing the generated chip on this family.
    pub clock_measured: bool,
    pub price_usd: f64,
    pub price: &'static str,
    /// Free tools that can build for it.
    pub tools: &'static str,
}

/// The smallest chip the generator makes for one board on its own, synthesised with Yosys
/// for each family (`tools/synth/families.sh`): CPU, die with eight cells and one lane, and
/// sampler, with scale and bias in two cycles. `the_catalog_knows_how_much_logic_the_smallest_chip_takes`
/// checks these against a fresh synthesis.
pub fn base_logic_k(board: &FpgaBoard) -> f64 {
    match board.lut {
        // A Cyclone IV has no distributed memory, only blocks that read a cycle late, so
        // its register table stays in flip-flops (synth_intel, memories aside).
        "LE" if board.fpga.contains("Cyclone IV") => 53.6,
        // With the register table in distributed memory, counted in the look-up tables
        // it takes.
        "LUT6" => 12.8,
        // Look-up tables and carry cells, as Gowin's count them.
        "LE" => 20.2,
        // Lattice ECP5 with the products in DSP blocks and the sums on carry chains.
        _ => 17.9,
    }
}

/// The memories of that chip besides weights and caches: microcode as a dictionary,
/// CPU memory and function tables, in megabits.
pub const BASE_MEMORY_MBIT: f64 = 1.68;

macro_rules! board {
    ($id:literal, $maker:literal, $name:literal, $fpga:literal, $url:literal, $image:expr, $luts:literal, $lut:literal, $bram:literal, $dsp:literal, $ddr:literal, $ddr_gbps:literal, $clock:literal, $measured:literal, $usd:literal, $price:literal, $tools:literal) => {
        FpgaBoard { id: $id, maker: $maker, name: $name, fpga: $fpga, url: $url, image: $image, luts_k: $luts, lut: $lut, bram_mbit: $bram, dsp: $dsp, ddr_gb: $ddr, ddr_gbps: $ddr_gbps, clock_mhz: $clock, clock_measured: $measured, price_usd: $usd, price: $price, tools: $tools }
    };
}

pub fn boards() -> &'static [FpgaBoard] {
    static BOARDS: OnceLock<Vec<FpgaBoard>> = OnceLock::new();
    BOARDS.get_or_init(|| {
        vec![
            board!("tang-primer-25k", "Sipeed", "Sipeed Tang Primer 25K", "Gowin GW5A-25", "https://wiki.sipeed.com/hardware/en/tang/tang-primer-25k/primer-25k.html", Some("https://wiki.sipeed.com/hardware/zh/tang/tang-primer-25k/assets/25k_dock_45.jpg"), 23.0, "LUT4", 1.0, 28, 0.0, 0.0, 40.0, false, 35.0, "~35 $", "Gowin EDA"),
            board!("tang-mega-60k", "Sipeed", "Sipeed Tang Mega 60K Dock", "Gowin GW5AT-60", "https://wiki.sipeed.com/hardware/en/tang/tang-mega-60k/mega-60k.html", Some("https://wiki.sipeed.com/hardware/zh/tang/tang-mega-60k/assets/mega_60k_top.png"), 60.0, "LUT4", 2.1, 118, 0.25, 1.6, 40.0, false, 90.0, "~90 $", "Gowin EDA"),
            board!("tang-mega-138k-dock", "Sipeed", "Sipeed Tang Mega 138K Dock", "Gowin GW5AST-138", "https://wiki.sipeed.com/hardware/en/tang/tang-mega-138k/mega-138k.html", Some("https://wiki.sipeed.com/hardware/zh/tang/tang-mega-138k/assets/mega_138k_top.jpg"), 138.0, "LUT4", 6.1, 298, 0.5, 1.6, 40.0, false, 130.0, "~130 $", "Gowin EDA"),
            board!("tang-mega-138k", "Sipeed", "Sipeed Tang Mega 138K Pro", "Gowin GW5AST-138", "https://wiki.sipeed.com/hardware/en/tang/tang-mega-138k/mega-138k-pro.html", Some("https://wiki.sipeed.com/hardware/en/tang/tang-mega-138k/assets/mega_138k_pro_top.png"), 138.0, "LUT4", 6.1, 298, 1.0, 3.2, 40.0, false, 195.0, "~195 $", "Gowin EDA"),
            board!("ulx3s-85f", "Radiona", "Radiona ULX3S 85F", "Lattice ECP5 LFE5U-85F", "https://radiona.org/ulx3s/", None, 84.0, "LUT4", 3.7, 156, 0.032, 0.2, 34.5, true, 150.0, "~130-150 $", "Yosys + nextpnr, libres"),
            // The usual small boards with this chip carry 32 MB of SDRAM on 16 bits.
            board!("cyclone-iv-ep4ce10", "Altera", "Placa con Cyclone IV EP4CE10", "Altera Cyclone IV E EP4CE10", "https://www.intel.com/content/www/us/en/products/details/fpga/cyclone/iv/e.html", None, 10.3, "LE", 0.41, 23, 0.008, 0.2, 40.0, false, 30.0, "~25-40 $", "Quartus Prime Lite (gratuito)"),
            board!("atum-a3-nano", "Terasic", "Terasic Atum A3 Nano", "Altera Agilex 3 A3CZ135", "https://www.terasic.com.tw/cgi-bin/page/archive.pl?Language=English&CategoryNo=44&No=1373", None, 135.0, "LE", 8.3, 368, 0.064, 0.2, 80.0, false, 180.0, "180-245 $", "Quartus Prime Pro (licencia gratuita para Agilex 3)"),
            board!("pynq-z2", "TUL", "TUL PYNQ-Z2", "Xilinx Zynq XC7Z020", "https://www.tulembedded.com/FPGA/ProductsPYNQ-Z2.html", None, 53.0, "LUT6", 4.9, 220, 0.5, 2.1, 50.0, false, 200.0, "200-260 $", "Vivado (edición gratuita)"),
            board!("de10-nano", "Terasic", "Terasic DE10-Nano", "Intel Cyclone V SE 5CSEBA6", "https://www.terasic.com.tw/cgi-bin/page/archive.pl?Language=English&No=1046", None, 110.0, "LE", 5.6, 112, 1.0, 3.2, 40.0, false, 225.0, "~225 $", "Quartus Prime Lite (gratuito)"),
            board!("kv260", "AMD", "AMD Kria KV260", "Zynq UltraScale+ K26 (clase ZU5EV)", "https://www.amd.com/en/products/system-on-modules/kria/k26/kv260-vision-starter-kit.html", None, 117.0, "LUT6", 23.1, 1248, 4.0, 19.0, 100.0, false, 249.0, "~249 $", "Vivado"),
            board!("arty-a7-100t", "Digilent", "Digilent Arty A7-100T", "Xilinx XC7A100T", "https://digilent.com/shop/arty-a7-100t-artix-7-fpga-development-board/", None, 63.0, "LUT6", 4.9, 240, 0.25, 2.6, 50.0, false, 250.0, "~250 $", "Vivado (edición gratuita)"),
            board!("zybo-z7-20", "Digilent", "Digilent Zybo Z7-20", "Xilinx Zynq XC7Z020", "https://digilent.com/shop/zybo-z7-zynq-7000-arm-fpga-soc-development-board/", None, 53.0, "LUT6", 4.9, 220, 1.0, 4.3, 50.0, false, 300.0, "~300 $", "Vivado (edición gratuita)"),
            board!("kr260", "AMD", "AMD Kria KR260", "Zynq UltraScale+ K26 (clase ZU5EV)", "https://www.amd.com/en/products/system-on-modules/kria/k26/kr260-robotics-starter-kit.html", None, 117.0, "LUT6", 23.1, 1248, 4.0, 19.0, 100.0, false, 349.0, "~349 $", "Vivado"),
            board!("alinx-ax7202", "ALINX", "ALINX AX7202", "Xilinx XC7A200T", "https://www.en.alinx.com/Product/FPGA-Development-Boards/Artix-7/AX7202.html", None, 134.0, "LUT6", 13.1, 740, 1.0, 6.4, 50.0, false, 450.0, "400-500 $", "Vivado (edición gratuita)"),
            board!("nexys-video", "Digilent", "Digilent Nexys Video", "Xilinx XC7A200T", "https://digilent.com/shop/nexys-video-artix-7-fpga-trainer-board-for-multimedia-applications/", None, 134.0, "LUT6", 13.1, 740, 0.5, 3.2, 50.0, false, 550.0, "~500 €", "Vivado (edición gratuita)"),
            board!("genesys-2", "Digilent", "Digilent Genesys 2", "Xilinx Kintex-7 XC7K325T", "https://digilent.com/shop/genesys-2-kintex-7-fpga-development-board/", None, 204.0, "LUT6", 16.0, 840, 1.0, 7.2, 80.0, false, 1000.0, "~1 000 $", "Vivado (licencia del dispositivo incluida)"),
            board!("alveo-u250", "AMD", "AMD Alveo U250", "Xilinx UltraScale+ XCU250", "https://www.amd.com/en/products/accelerators/alveo/u250/a-u250-a64g-pq-g.html", None, 1728.0, "LUT6", 432.0, 12288, 64.0, 77.0, 200.0, false, 9000.0, "~9 000 $", "Vivado"),
        ]
    })
}

pub fn board(id: &str) -> Option<&'static FpgaBoard> {
    boards().iter().find(|b| b.id == id)
}

/// Whether the design targets an FPGA board, and which.
pub fn target(p: &ChipParams) -> Option<&'static FpgaBoard> {
    if p.node.starts_with("fpga") {
        board(&p.fpga_board)
    } else {
        None
    }
}

/// The design on a board: its clock, and the board's memory as the external memory.
pub fn on_board(p: &ChipParams, board: &FpgaBoard) -> ChipParams {
    // Square millimetres mean nothing on a board: what limits it is its memory and logic.
    ChipParams { freq_ghz: p.freq_ghz.min(board.clock_mhz / 1000.0), ddr_gbps: board.ddr_gbps.max(0.01), ddr_capacity_gb: board.ddr_gb.max(0.004), dies: 1, die_max_mm2: f64::MAX, ..p.clone() }
}

/// How much of the board the design takes, as metrics, and what does not fit, with the
/// change that would make it fit.
pub fn check(report: &mut Report, p: &ChipParams, board: &FpgaBoard) {
    let gb = |id: &str| report.metric(id).unwrap_or(0.0);
    let weights = if p.weight_backend == WeightBackend::Sram { gb("model_weights_gb") } else { 0.0 };
    let caches = if p.kv_backend == KvBackend::Sram { gb("kv_gb") + gb("delta_state_mb") / 1000.0 } else { 0.0 };
    let memory = BASE_MEMORY_MBIT + (weights + caches) * 8000.0;
    let logic = base_logic_k(board);
    // A cell's product takes one multiplier block, as does each lane's and the sampler's.
    let multipliers = report.macs_per_die + p.vector_lanes_per_die.saturating_mul(p.vector_units_per_die.max(1)) + 2;
    let share = |used: f64, total: f64| if total > 0.0 { 100.0 * used / total } else { f64::INFINITY };
    let metric = |id, label, value, unit, confidence, why: String| Metric { id, section: "FPGA", label, value, unit, confidence, why };
    report.metrics.push(metric("fpga_memory_pct", "Memoria interna usada", share(memory, board.bram_mbit), "%", Confidence::Calibrada, format!("{memory:.1} de {} Mbit: {BASE_MEMORY_MBIT} Mbit del chip básico medidos en síntesis, más los pesos y cachés que viven dentro.", board.bram_mbit)));
    report.metrics.push(metric("fpga_logic_pct", "Lógica usada", share(logic, board.luts_k), "%", Confidence::Calibrada, format!("{logic:.1} K {} del chip básico sintetizado con Yosys, de {} K. Con más carriles y unidades crece; mídelo con la síntesis real.", board.lut, board.luts_k)));
    report.metrics.push(metric("fpga_dsp_pct", "Multiplicadores usados", share(multipliers as f64, board.dsp as f64), "%", Confidence::Derivada, format!("{multipliers} productos por ciclo, cada uno en un bloque DSP, de {}.", board.dsp)));
    report.metrics.push(metric("fpga_clock_mhz", "Reloj en esta placa", board.clock_mhz.min(p.freq_ghz * 1000.0), "MHz", if board.clock_measured { Confidence::Calibrada } else { Confidence::Supuesta }, if board.clock_measured { "Medido colocando y rutando el chip con nextpnr en esta familia.".into() } else { "Sin colocar y rutar aún en esta familia: una cifra prudente para lógica de FPGA sin optimizar a mano.".into() }));

    let error = |message: String, fix: &str, field: &'static str, value: serde_json::Value| Violation { severity: Severity::Error, message, fix: fix.into(), field: Some(field), value: Some(value) };
    if logic > board.luts_k {
        report.violations.push(error(format!("El chip básico ocupa {logic:.0} K {} y esta FPGA tiene {} K.", board.lut, board.luts_k), "Elige una FPGA mayor.", "fpga_board", serde_json::json!("kv260")));
    }
    if memory > board.bram_mbit {
        // What moves out must leave the rest inside: the weights, the caches, both, or else
        // the board is too small and a larger one is the fix.
        let fits_outside = |gb: f64| board.ddr_gb > 0.0 && gb <= board.ddr_gb * 0.9;
        let inside = |gb: f64| BASE_MEMORY_MBIT + gb * 8000.0 <= board.bram_mbit;
        let (fix, field, value) = if weights > 0.0 && inside(caches) && fits_outside(weights) {
            ("Lleva los pesos a la memoria de la placa: más lento, pero cabe.", "weight_backend", serde_json::json!("external_ddr"))
        } else if caches > 0.0 && inside(weights) && fits_outside(caches) {
            ("Lleva la caché a la memoria de la placa, o reduce el contexto.", "kv_backend", serde_json::json!("external_ddr"))
        } else {
            let larger = boards().iter().filter(|b| b.bram_mbit >= memory && base_logic_k(b) <= b.luts_k).min_by(|a, b| a.price_usd.total_cmp(&b.price_usd));
            match larger {
                Some(b) => ("Elige una FPGA con más memoria interna, o reduce bits de peso o contexto.", "fpga_board", serde_json::json!(b.id)),
                None => ("Ninguna placa del catálogo guarda tanto dentro: reduce bits de peso o contexto.", "context_tokens", serde_json::json!((p.context_tokens / 2).max(64))),
            }
        };
        report.violations.push(error(format!("Hacen falta {memory:.1} Mbit de memoria interna y esta FPGA tiene {}.", board.bram_mbit), fix, field, value));
    }
    if multipliers > board.dsp {
        report.violations.push(Violation {
            severity: Severity::Aviso,
            message: format!("Hay {multipliers} productos por ciclo y {} bloques DSP: el resto se hace con LUT, mucho más grandes.", board.dsp),
            fix: "Baja los multiplicadores por die.".into(),
            field: Some("macs_per_die"),
            value: Some(serde_json::json!(board.dsp.saturating_sub(p.vector_lanes_per_die * p.vector_units_per_die.max(1) + 2).max(1))),
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use model_ir::ModelGraph;

    fn on(board: &str, config: &str, p: ChipParams) -> Report {
        let model = ModelGraph::from_config_json("modelo", config).unwrap();
        crate::estimate(&model, &ChipParams { node: "fpga_u250".into(), fpga_board: board.into(), max_params_g: 30.0, ..p }).unwrap()
    }

    const STORIES: &str = include_str!("../../../models/stories260k.json");
    const STORIES_110M: &str = include_str!("../../../models/stories110m.json");

    #[test]
    fn a_tiny_model_fits_a_cheap_board_and_the_board_sets_the_clock() {
        let p = ChipParams { weight_bits: 8, context_tokens: 256, target_tok_s: 0.0, macs_per_die: 8, vector_lanes_per_die: 1, vector_units_per_die: 1, freq_ghz: 1.0, ..Default::default() };
        let report = on("tang-mega-138k", STORIES, p);
        assert!(report.metric("fpga_memory_pct").unwrap() < 100.0, "{:?}", report.metric("fpga_memory_pct"));
        assert_eq!(report.metric("fpga_clock_mhz"), Some(40.0));
        assert!(!report.violations.iter().any(|v| v.severity == Severity::Error && v.message.contains("memoria interna")));
    }

    #[test]
    fn what_does_not_fit_is_said_with_the_change_that_fixes_it() {
        let p = ChipParams { weight_bits: 8, context_tokens: 2048, target_tok_s: 0.0, macs_per_die: 1024, ..Default::default() };
        // 110 M weights do not fit inside the KV260, but do in the memory on its board; with
        // a short context the cache stays inside.
        let report = on("kv260", STORIES_110M, ChipParams { context_tokens: 64, ..p.clone() });
        let memory = report.violations.iter().find(|v| v.message.contains("memoria interna")).expect("no cabe dentro");
        assert_eq!((memory.field, memory.value.clone()), (Some("weight_backend"), Some(serde_json::json!("external_ddr"))));
        // The smallest board has no room even for the basic chip.
        let tiny = on("tang-primer-25k", STORIES, p);
        assert!(tiny.violations.iter().any(|v| v.field == Some("fpga_board")));
    }
}


#[cfg(test)]
mod corrections {
    use crate::estimate::Severity;
    use crate::params::ChipParams;
    use model_ir::ModelGraph;

    /// The design with one parameter set to `value`, as the interface sets it.
    fn with(p: &ChipParams, field: &str, value: &serde_json::Value) -> ChipParams {
        let mut json = serde_json::to_value(p).unwrap();
        json[field] = value.clone();
        serde_json::from_value(json).unwrap()
    }

    #[test]
    fn every_offered_correction_clears_its_warning() {
        let model = |config: &str| ModelGraph::from_config_json("modelo", config).unwrap();
        let stories = model(include_str!("../../../models/stories260k.json"));
        let stories_110m = model(include_str!("../../../models/stories110m.json"));
        let qwen = model(include_str!("../../../models/qwen3.8-27b.json"));
        let fpga = |board: &str, context: u64| ChipParams { node: "fpga_u250".into(), fpga_board: board.into(), weight_bits: 8, context_tokens: context, target_tok_s: 0.0, macs_per_die: 64, max_params_g: 1.0, ..Default::default() };
        let cases = [
            (&stories, fpga("tang-mega-138k", 4096)),
            (&stories_110m, fpga("kv260", 2048)),
            (&stories, fpga("tang-primer-25k", 256)),
            (&stories, ChipParams { mtp_enabled: true, context_tokens: 1 << 20, ..Default::default() }),
            (&qwen, ChipParams { max_params_g: 1.0, ..Default::default() }),
            (&qwen, ChipParams { dies: 2, target_tok_s: 0.0, ..Default::default() }),
            (&qwen, ChipParams { freq_ghz: 9.0, target_tok_s: 0.0, ..Default::default() }),
        ];
        let mut checked = 0;
        for (model, p) in cases {
            let report = crate::estimate(model, &p).unwrap();
            for v in report.violations.iter().filter(|v| v.value.is_some()) {
                let (field, value) = (v.field.unwrap(), v.value.as_ref().unwrap());
                let fixed = crate::estimate(model, &with(&p, field, value)).unwrap();
                let kind = |m: &str| m.chars().take(24).collect::<String>();
                let still = fixed.violations.iter().find(|w| w.field == v.field && w.severity == v.severity && kind(&w.message) == kind(&v.message));
                assert!(still.is_none() || v.severity == Severity::Aviso && field == "target_tok_s", "«{}» sigue tras poner {field} = {value}: {:?}", v.message, still.map(|w| &w.message));
                checked += 1;
            }
        }
        assert!(checked >= 6, "solo se comprobaron {checked} correcciones");
    }
}
