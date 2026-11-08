use crate::floorplan::{build_floorplan, Floorplan};
use crate::optim::catalog;
use crate::params::{ChipParams, KvBackend, LinkKind, WeightBackend};
use crate::tech::{node, TechNode};
use netlist::format::WeightCoding;
use netlist::{cell_design, CellConfig};
use model_ir::{Block, ModelGraph, StageKind, WeightTensor};
use serde::Serialize;

/// Tokens accepted at each draft depth, measured by MTPLX on Qwen3.8-27B.
const MTP_ACCEPTANCE: [f64; 3] = [0.938, 0.855, 0.743];
const HBM4_STACK_BYTES_PER_S: f64 = 2e12;
const HBM4_PJ_PER_BIT: f64 = 3.5;
const STACKED_DRAM_PJ_PER_BIT: f64 = 0.37;
const DDR_PJ_PER_BIT: f64 = 15.0;
const MAX_AUTO_DIES: u32 = 512;
const ROUTING_OVERHEAD: f64 = 1.2;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Confidence {
    Derivada,
    Calibrada,
    Supuesta,
}

#[derive(Clone, Debug, Serialize)]
pub struct Metric {
    pub id: &'static str,
    pub section: &'static str,
    pub label: &'static str,
    pub value: f64,
    pub unit: &'static str,
    pub confidence: Confidence,
    pub why: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Category {
    Memoria,
    Calculo,
    Interconexion,
    Vectorial,
    Estado,
    Sistema,
}

#[derive(Clone, Debug, Serialize)]
pub struct BlockArea {
    pub id: &'static str,
    pub label: &'static str,
    pub per_die_mm2: f64,
    pub what: &'static str,
    pub category: Category,
    /// Parameters that change this block's size, as `ChipParams` field names.
    pub driven_by: &'static [&'static str],
    pub why: String,
    /// Fraction of a token's cycles during which the block is busy.
    pub activity: f64,
    pub energy_mj: f64,
    pub bytes_token: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Severity {
    Error,
    Aviso,
}

#[derive(Clone, Debug, Serialize)]
pub struct Violation {
    pub severity: Severity,
    pub message: String,
    pub fix: String,
    /// The parameter to reveal when the user asks how to fix this.
    pub field: Option<&'static str>,
}

#[derive(Clone, Debug, Serialize)]
pub struct Report {
    pub model: String,
    pub dies: u32,
    pub die_area_mm2: f64,
    pub total_area_mm2: f64,
    pub blocks: Vec<BlockArea>,
    pub metrics: Vec<Metric>,
    pub violations: Vec<Violation>,
    pub floorplan: Floorplan,
}

impl Report {
    pub fn metric(&self, id: &str) -> Option<f64> {
        self.metrics.iter().find(|m| m.id == id).map(|m| m.value)
    }
}

fn coding_name(coding: WeightCoding) -> &'static str {
    match coding {
        WeightCoding::Fixed => "punto fijo",
        WeightCoding::Affine => "formato afín",
        WeightCoding::Float => "coma flotante",
        WeightCoding::Posit => "posit",
        WeightCoding::Log => "formato logarítmico",
        WeightCoding::Codebook => "libro de códigos",
    }
}

struct Storage {
    /// Weight bits held by the chip for the chosen budget.
    chip_bits: f64,
    /// Bits for the loaded model only, used for the capacity check and load time.
    model_bits: f64,
    external_bits: f64,
    /// Average stored bits per matrix weight actually read.
    bits_per_read_weight: f64,
    widest_weight_bits: u32,
}

fn tensor_included(t: &WeightTensor, p: &ChipParams) -> bool {
    match t.block {
        Block::Vision => p.include_vision,
        Block::Mtp => p.mtp_enabled,
        _ => true,
    }
}

fn is_small_float(t: &WeightTensor) -> bool {
    t.block == Block::Norm
        || t.block == Block::Vision
        || t.rows == 1
        || t.name.ends_with("conv1d")
}

fn storage(model: &ModelGraph, p: &ChipParams, budget_params: f64) -> Storage {
    let mut model_bits = 0.0;
    let mut external_bits = 0.0;
    let mut read_bits = 0.0;
    let mut read_weights = 0.0;
    let mut has_wide = false;
    // Small float tensors never lose precision below what the weights themselves keep.
    let small_bits = 16.0f64.max(if p.weight_coding.shifts() || p.weight_coding == WeightCoding::Fixed { p.weight_bits as f64 } else { 0.0 });

    for t in model.tensors.iter().filter(|t| tensor_included(t, p)) {
        let params = t.params() as f64;
        if is_small_float(t) {
            model_bits += params * small_bits;
            continue;
        }
        let per_weight = if p.mixes_widths() && t.wide {
            has_wide = true;
            8.0 + p.group_overhead_bits(p.group_size * 2)
        } else {
            p.weight_bits as f64 + p.group_overhead_bits(p.group_size)
        };
        let mut factor = 1.0;
        if p.has("compresion_pesos") {
            factor *= 0.75;
        }
        if p.has("poda_pesos") {
            factor *= 0.85;
        }
        if p.has("baja_rango") {
            factor *= 0.6;
        }
        if t.block == Block::Head && p.has("cabeza_recortada") {
            factor *= 0.3;
        }
        let bits = params * per_weight * factor;
        if t.block == Block::Embedding && p.has("embedding_externo") {
            external_bits += bits;
            continue;
        }
        model_bits += bits;
        if t.block != Block::Embedding {
            read_bits += bits;
            read_weights += params;
        }
    }

    let included = model.params_where(|t| tensor_included(t, p)) as f64;
    let scale = budget_params / included.max(1.0);
    Storage {
        chip_bits: model_bits * scale,
        model_bits,
        external_bits,
        bits_per_read_weight: read_bits / read_weights.max(1.0),
        widest_weight_bits: if has_wide { p.weight_bits.max(8) } else { p.weight_bits },
    }
}

#[derive(Default)]
struct StageTotals {
    weights: f64,
    matvec_stages: f64,
    vector_ops: f64,
    vector_stages: f64,
    mtp_weights: f64,
    mtp_matvec_stages: f64,
    mtp_vector_ops: f64,
    mtp_vector_stages: f64,
}

fn stage_totals(model: &ModelGraph, p: &ChipParams) -> StageTotals {
    let attention_factor = if p.has("atencion_dispersa") { 0.25 } else { 1.0 };
    let context = p.context_tokens as f64;
    let mut t = StageTotals::default();
    for stage in &model.stages {
        let mtp = stage.block == Block::Mtp;
        match stage.kind {
            StageKind::Matvec { weights } => {
                if mtp {
                    t.mtp_weights += weights as f64;
                    t.mtp_matvec_stages += 1.0;
                } else {
                    t.weights += weights as f64;
                    t.matvec_stages += 1.0;
                }
            }
            StageKind::Vector { ops, ops_per_ctx } => {
                let total = ops as f64 + ops_per_ctx as f64 * context * attention_factor;
                if mtp {
                    t.mtp_vector_ops += total;
                    t.mtp_vector_stages += 1.0;
                } else {
                    t.vector_ops += total;
                    t.vector_stages += 1.0;
                }
            }
            StageKind::EmbeddingRead { .. } => {}
        }
    }
    t
}

/// Logic area of one cell, from the gates its generated netlist actually contains.
fn netlist_cell_um2(tech: &TechNode, p: &ChipParams, widest: u32, addr_bits: u32) -> f64 {
    let config = CellConfig { weight_bits: widest, ..p.cell_config(addr_bits) };
    cell_design(config)
        .flat_counts()
        .iter()
        .map(|(gate, count)| {
            *count as f64
                * match gate.as_str() {
                    "and" => tech.and_um2,
                    "or" => tech.or_um2,
                    "xor" => tech.xor_um2,
                    "not" => tech.not_um2,
                    "mux" => tech.mux_um2,
                    "dff" => tech.ff_um2,
                    _ => 0.0,
                }
        })
        .sum()
}

struct Multiplier {
    area_um2: f64,
    energy_pj: f64,
    delay_factor: f64,
    cycle_factor: f64,
}

fn weight_multiplier(tech: &TechNode, p: &ChipParams, widest: u32, cell_bits: f64) -> Multiplier {
    let significand = CellConfig { weight_bits: widest, ..p.cell_config(1) }.significand_bits();
    let products = significand as f64 * p.activation_bits as f64;
    let addr_bits = (cell_bits / widest as f64).max(2.0).log2().ceil() as u32;
    let cell_logic = netlist_cell_um2(tech, p, widest, addr_bits);
    let accumulator_bits = p.effective_accumulator() as f64;
    let accumulator = accumulator_bits * (tech.fa_um2 + tech.ff_um2);
    let mut area = (cell_logic - accumulator).max(0.0);
    let mut energy = (products + accumulator_bits) * tech.e_fa_pj;
    let mut delay_factor = 1.0;
    let mut cycle_factor = 1.0;
    if p.has("sin_multiplicadores") {
        area *= 0.4;
        energy *= 0.6;
        delay_factor = 0.5;
    }
    if p.has("aritmetica_distribuida") {
        area *= 0.15;
        energy *= 0.5;
        delay_factor = 0.3;
        cycle_factor = 4.0;
    }
    if p.has("cim_analogico") {
        area *= 0.3;
        energy *= 0.2;
    }
    if p.has("salto_ceros") {
        energy *= 0.9;
    }
    let decoder = if p.has("compresion_pesos") { 8.0 } else { 0.0 };
    Multiplier {
        area_um2: area + accumulator + decoder,
        energy_pj: energy,
        delay_factor,
        cycle_factor,
    }
}

struct AreaInputs<'a> {
    tech: &'a TechNode,
    p: &'a ChipParams,
    model: &'a ModelGraph,
    store: &'a Storage,
    mult: &'a Multiplier,
    kv_bits: f64,
    state_bits: f64,
    rollback_bits: f64,
}

fn sram_margin(p: &ChipParams) -> f64 {
    let ecc = if p.has("sin_ecc_con_refresco") { 1.0 } else { 1.125 };
    let spare = if p.has("reparacion_por_compilador") { 1.01 } else { 1.03 };
    ecc * spare
}

fn blocks_for(i: &AreaInputs<'_>, dies: u32) -> Vec<BlockArea> {
    let AreaInputs { tech, p, model, store, mult, .. } = *i;
    let d = dies as f64;
    let n = p.macs_per_die as f64;
    let um2 = 1e-6;
    let mut b = Vec::new();
    let mut push = |id, label, per_die_mm2: f64, what| {
        if per_die_mm2 > 0.0 {
            b.push(BlockArea {
                id,
                label,
                per_die_mm2,
                what,
                category: category_of(id),
                driven_by: driven_by(id),
                why: String::new(),
                activity: 0.0,
                energy_mj: 0.0,
                bytes_token: 0.0,
            });
        }
    };

    let cell_kbit = store.chip_bits / (d * n) / 1e3;
    let weights_mm2 = match p.weight_backend {
        WeightBackend::Sram => {
            let mut density = tech.sram_density(cell_kbit);
            if p.has("edram_ganancia") {
                density *= 2.0;
            }
            let mut area = store.chip_bits * sram_margin(p) / 1e6 / density / d;
            if p.has("sram_apilada_3d") {
                area *= 0.5;
            }
            area
        }
        WeightBackend::Rom => store.chip_bits / 1e6 / tech.rom_mbit_mm2 / d,
        WeightBackend::StackedDram | WeightBackend::Hbm4 | WeightBackend::ExternalDdr => {
            n * 4.0 * 8.0 / 1e6 / tech.sram_density(32.0)
        }
    };
    push("cell_weights", "Celdas: memoria de pesos", weights_mm2,
        "Cada celda guarda un trozo de cada matriz junto a su multiplicador, para que los pesos nunca viajen.");
    push("cell_macs", "Celdas: multiplicadores", n * mult.area_um2 * um2 * ROUTING_OVERHEAD,
        "Multiplican peso por activación y acumulan la suma de cada fila.");

    if p.cell_table_bits() > 0.0 {
        let density = tech.sram_density((p.cell_table_bits() / 1e3).max(1.0));
        push("weight_tables", "Tablas de pesos", n * p.cell_table_bits() * sram_margin(p) / 1e6 / density,
            "Cada celda guarda los valores reales a los que apuntan sus códigos.");
    }

    if p.uses_groups() {
        let unit = (p.scale_bits * p.effective_accumulator()) as f64 * tech.fa_um2;
        push("scale_units", "Escala y sesgo por grupo", n * unit * um2 * ROUTING_OVERHEAD,
            "Aplican la escala y el sesgo que comparten los pesos de cada grupo.");
    }

    push("activation_noc", "Red de activaciones",
        n * p.activation_bits as f64 * tech.ff_um2 * p.noc_stages.max(1) as f64 * 4.0 * um2,
        "Reparte cada vector de entrada a todas las celdas y recoge las sumas parciales.");

    let lane = (p.vector_bits * p.vector_bits) as f64 * tech.fa_um2
        + 2.0 * p.vector_bits as f64 * (tech.fa_um2 + tech.ff_um2);
    push("vector_units", "Unidades vectoriales",
        p.vector_lanes_per_die as f64 * lane * um2 * ROUTING_OVERHEAD,
        "Multiplican activación por activación o estado: atención Q·K, regla delta y puertas.");

    let table_bits = 65_536.0;
    push("function_units", "Unidades de función",
        p.function_units_per_die as f64 * (table_bits / 1e6 / tech.sram_density(64.0) + 5000.0 * um2),
        "Tablas para SiLU, sigmoide, softplus, exp, raíz inversa y softmax.");

    let widest_vector = model
        .tensors
        .iter()
        .filter(|t| t.block != Block::Vision)
        .map(|t| t.rows.max(t.cols))
        .filter(|&len| len < model.vocab_size)
        .max()
        .unwrap_or(0);
    let buffer_bits = (model.hidden_size * 8 + widest_vector * 2) as f64 * p.activation_bits as f64;
    push("activation_buffers", "Búferes de activación",
        buffer_bits / 1e6 / tech.sram_density(256.0),
        "Guardan los vectores intermedios entre etapas.");

    if p.kv_backend == KvBackend::Sram {
        push("kv_sram", "Caché KV de atención",
            i.kv_bits * sram_margin(p) / 1e6 / tech.sram_density(1024.0) / d,
            "Claves y valores de cada token del contexto. Crece con el contexto.");
    }
    if p.kv_backend != KvBackend::ExternalDdr {
        push("delta_state", "Estado de DeltaNet",
            i.state_bits * sram_margin(p) / 1e6 / tech.sram_density(1024.0) / d,
            "Matrices recurrentes de las capas lineales. No crece con el contexto.");
    }
    if p.mtp_enabled {
        push("mtp_rollback", "MTP: copia de estado",
            i.rollback_bits * sram_margin(p) / 1e6 / tech.sram_density(1024.0) / d,
            "Permite deshacer el estado cuando se rechaza un token especulado.");
    }

    push("control", "Secuenciador y microprograma", 0.8,
        "Ejecuta el microprograma que dirige cada etapa.");
    push("system", "CPU de control y carga", (tech.cpu_mm2 + tech.io_mm2) / d,
        "Tokenizador, muestreo y volcado de pesos desde la imagen.");
    if dies > 1 {
        let endpoint = match p.link {
            LinkKind::Ucie => 2.0,
            LinkKind::Serdes => 6.0,
        };
        push("links", "Enlaces entre dies", endpoint, "Pasan un vector de activación entre capas consecutivas.");
    }
    if p.has("checkpoint_estado") || p.has("embedding_externo") {
        push("dram_ctrl", "Controlador de DRAM", 4.0 / d, "Acceso a la DRAM externa.");
    }
    let hbm = p.weight_backend == WeightBackend::Hbm4 || p.kv_backend == KvBackend::Hbm4;
    if hbm {
        push("hbm_phy", "PHY de HBM4", 12.0 * p.hbm_stacks as f64 / d, "Interfaz con las pilas de memoria HBM4.");
    }
    push("fixed", "Reloj, alimentación y test", tech.fixed_die_mm2, "Infraestructura necesaria en cada die.");
    b
}

fn category_of(id: &str) -> Category {
    match id {
        "cell_weights" | "weight_tables" => Category::Memoria,
        "cell_macs" | "scale_units" => Category::Calculo,
        "activation_noc" | "links" => Category::Interconexion,
        "vector_units" | "function_units" => Category::Vectorial,
        "activation_buffers" | "kv_sram" | "delta_state" | "mtp_rollback" => Category::Estado,
        _ => Category::Sistema,
    }
}

fn driven_by(id: &str) -> &'static [&'static str] {
    match id {
        "cell_weights" => &["max_params_g", "weight_coding", "weight_bits", "group_size", "weight_backend"],
        "weight_tables" => &["weight_coding", "weight_bits"],
        "cell_macs" => &["macs_per_die", "weight_coding", "weight_bits", "weight_exponent_bits", "activation_bits", "mac_stages"],
        "scale_units" => &["group_size", "scale_bits"],
        "activation_noc" => &["macs_per_die", "activation_bits", "noc_stages"],
        "vector_units" => &["vector_lanes_per_die", "vector_bits"],
        "function_units" => &["function_units_per_die"],
        "activation_buffers" => &["activation_bits"],
        "kv_sram" => &["context_tokens", "kv_bits", "kv_backend"],
        "delta_state" => &["delta_state_bits"],
        "mtp_rollback" => &["mtp_enabled", "mtp_depth", "delta_state_bits"],
        "links" => &["link", "dies"],
        "hbm_phy" => &["hbm_stacks"],
        _ => &[],
    }
}

fn total(blocks: &[BlockArea]) -> f64 {
    blocks.iter().map(|b| b.per_die_mm2).sum()
}

fn check_optimizations(p: &ChipParams, out: &mut Vec<Violation>) {
    let aviso = |message: &str, fix: &str| Violation {
        severity: Severity::Aviso,
        message: message.into(),
        fix: fix.into(),
        field: None,
    };
    let error = |message: &str, fix: &str| Violation {
        severity: Severity::Error,
        message: message.into(),
        fix: fix.into(),
        field: None,
    };
    for id in &p.optimizations {
        if !catalog().iter().any(|o| o.id == id) {
            out.push(aviso(&format!("Optimización desconocida: {id}"), "Quítala del diseño."));
        }
    }
    let sram_only = ["edram_ganancia", "sram_apilada_3d", "sram_baja_tension", "cim_analogico", "sin_ecc_con_refresco"];
    if p.weight_backend != WeightBackend::Sram {
        for id in sram_only.iter().filter(|id| p.has(id)) {
            out.push(error(
                &format!("«{id}» solo tiene sentido con pesos en SRAM."),
                "Cambia la memoria de pesos a SRAM o desactiva la optimización.",
            ));
        }
    }
    for (a, b) in [
        ("sin_multiplicadores", "aritmetica_distribuida"),
        ("sin_multiplicadores", "cim_analogico"),
        ("aritmetica_distribuida", "cim_analogico"),
        ("edram_ganancia", "cim_analogico"),
    ] {
        if p.has(a) && p.has(b) {
            out.push(error(&format!("«{a}» y «{b}» son incompatibles."), "Elige una de las dos."));
        }
    }
    if p.has("sin_multiplicadores") && (p.weight_bits > 4 || p.mixes_widths() || p.weight_coding.shifts()) {
        out.push(aviso(
            "«sin_multiplicadores» solo rinde lo anunciado con pesos de 4 bits o menos.",
            "Baja los bits de peso y desactiva la mezcla de 8 bits.",
        ));
    }
    if p.effective_accumulator() > p.accumulator_bits {
        out.push(Violation {
            severity: Severity::Aviso,
            message: format!("Este formato de pesos necesita un acumulador de {} bits; se usa ese ancho en lugar de {}.", p.effective_accumulator(), p.accumulator_bits),
            fix: "Sube los bits de acumulador o usa un formato más estrecho.".into(),
            field: Some("accumulator_bits"),
        });
    }
    if p.weight_backend == WeightBackend::Rom {
        out.push(aviso(
            "Con ROM el chip solo sirve para este modelo.",
            "Usa SRAM para poder cargar otros pesos.",
        ));
    }
}

/// Estimates area, speed and energy for one design running one user at a time.
pub fn estimate(model: &ModelGraph, p: &ChipParams) -> Result<Report, String> {
    let tech = node(&p.node).ok_or_else(|| format!("nodo tecnológico desconocido: {}", p.node))?;
    let mut violations = Vec::new();
    check_optimizations(p, &mut violations);

    let included = model.params_where(|t| tensor_included(t, p)) as f64;
    let budget = (p.max_params_g * 1e9).max(1.0);
    let store = storage(model, p, budget);
    let totals = stage_totals(model, p);
    let cell_bits_estimate = store.chip_bits / p.macs_per_die.max(1) as f64 / (p.dies.max(10) as f64);
    let mult = weight_multiplier(tech, p, store.widest_weight_bits, cell_bits_estimate);
    let context = p.context_tokens as f64;

    if included > budget {
        violations.push(Violation {
            severity: Severity::Error,
            message: format!(
                "El modelo no cabe: tiene {:.2} G de parámetros y el chip admite {:.2} G.",
                included / 1e9,
                budget / 1e9
            ),
            fix: "Sube los parámetros máximos.".into(),
            field: Some("max_params_g"),
        });
    }
    if model.max_context > 0 && p.context_tokens > model.max_context {
        violations.push(Violation {
            severity: Severity::Aviso,
            message: format!("El modelo declara {} tokens de contexto nativos; por encima puede perder calidad.", model.max_context),
            fix: "Reduce el contexto si la calidad cae.".into(),
            field: Some("context_tokens"),
        });
    }
    if p.mtp_enabled && model.params_in(Block::Mtp) == 0 {
        violations.push(Violation {
            severity: Severity::Aviso,
            message: "Este modelo no tiene cabeza MTP.".into(),
            fix: "Desactiva MTP.".into(),
            field: Some("mtp_enabled"),
        });
    }

    let kv_bits = model.state.kv_values_per_token as f64 * context * p.kv_bits as f64;
    let state_bits = model.state.delta_values as f64 * p.delta_state_bits as f64
        + model.state.conv_values as f64 * p.activation_bits as f64;
    let depth = if p.mtp_enabled { p.mtp_depth.min(3) } else { 0 };
    let rollback_bits = state_bits * depth as f64;
    let inputs = AreaInputs {
        tech,
        p,
        model,
        store: &store,
        mult: &mult,
        kv_bits,
        state_bits,
        rollback_bits,
    };

    if p.weight_backend == WeightBackend::ExternalDdr && store.chip_bits / 8e9 > p.ddr_capacity_gb {
        violations.push(Violation {
            severity: Severity::Error,
            message: format!(
                "Los pesos ocupan {:.1} GB y la memoria externa tiene {:.0} GB.",
                store.chip_bits / 8e9,
                p.ddr_capacity_gb
            ),
            fix: "Amplía la memoria externa o reduce los bits de peso.".into(),
            field: Some("ddr_capacity_gb"),
        });
    }

    let die_limit = p.die_max_mm2.min(tech.die_max_mm2);
    let mut dies = if p.dies > 0 { p.dies } else { 1 };
    let mut blocks = blocks_for(&inputs, dies);
    if p.dies == 0 {
        while total(&blocks) > die_limit && dies < MAX_AUTO_DIES {
            dies += 1;
            blocks = blocks_for(&inputs, dies);
        }
    }
    let die_area = total(&blocks);
    if die_area > die_limit {
        violations.push(Violation {
            severity: Severity::Error,
            message: format!("Cada die ocupa {die_area:.0} mm² y el máximo es {die_limit:.0} mm²."),
            fix: "Añade dies, reduce multiplicadores o contexto, o activa optimizaciones de memoria.".into(),
            field: Some("dies"),
        });
    }
    let d = dies as f64;
    let n = p.macs_per_die as f64;
    let total_area = die_area * d;

    // Timing: each token visits every stage in order, so stage times add up.
    let cell = CellConfig { weight_bits: store.widest_weight_bits, ..p.cell_config(1) };
    // Decoding and placing a floating product take a subtraction and a shifter's levels.
    let decode_levels = if cell.shifts() { cell.exponent_width() + 8 } else { (cell.coding == WeightCoding::Codebook) as u32 };
    let logic_ns = (cell.significand_bits() + p.activation_bits + decode_levels) as f64 * tech.fa_delay_ns * mult.delay_factor;
    let fmax = tech.fmax_ghz.min(1.0 / (logic_ns / p.mac_stages.max(1) as f64 + tech.ff_overhead_ns));
    let freq = p.freq_ghz.min(fmax);
    if p.freq_ghz > fmax {
        violations.push(Violation {
            severity: Severity::Aviso,
            message: format!("Con {} etapas de multiplicador el máximo son {fmax:.2} GHz; se simula a esa frecuencia.", p.mac_stages),
            fix: "Añade etapas o baja la frecuencia.".into(),
            field: Some("freq_ghz"),
        });
    }
    let tree_levels = n.log2().ceil().max(1.0);
    let matvec_latency = (p.mac_stages + p.adder_stages) as f64 + 2.0 * tree_levels * p.noc_stages.max(1) as f64;
    let group_overhead = if p.uses_groups() { 1.0 + 1.0 / p.group_size as f64 } else { 1.0 };
    let weight_cycles = totals.weights / n * group_overhead * mult.cycle_factor;
    let lanes = p.vector_lanes_per_die.max(1) as f64;
    let vector_latency = 6.0;
    let vector_cycles = totals.vector_ops / lanes;
    let latency_cycles = totals.matvec_stages * matvec_latency + totals.vector_stages * vector_latency;
    let hop_ns = match p.link {
        LinkKind::Ucie => 10.0,
        LinkKind::Serdes => 150.0,
    };
    let link_cycles = if dies > 1 { d * hop_ns * freq } else { 0.0 };
    let total_cycles = weight_cycles + vector_cycles + latency_cycles + link_cycles;
    let compute_time = total_cycles / (freq * 1e9);

    let read_bytes = totals.weights * store.bits_per_read_weight / 8.0;
    let kv_factor = if p.has("atencion_dispersa") { 0.25 } else { 1.0 };
    let kv_read_bytes = kv_bits / 8.0 * kv_factor;
    let weight_bandwidth = match p.weight_backend {
        WeightBackend::Hbm4 => Some(p.hbm_stacks as f64 * HBM4_STACK_BYTES_PER_S),
        WeightBackend::StackedDram => Some(p.stacked_dram_tbps * 1e12),
        WeightBackend::ExternalDdr => Some(p.ddr_gbps * 1e9),
        _ => None,
    };
    let weight_time = weight_bandwidth.map_or(0.0, |b| read_bytes / b);
    let kv_time = match p.kv_backend {
        KvBackend::Hbm4 => kv_read_bytes / (p.hbm_stacks as f64 * HBM4_STACK_BYTES_PER_S),
        KvBackend::ExternalDdr => (kv_read_bytes + 2.0 * state_bits / 8.0) / (p.ddr_gbps * 1e9),
        KvBackend::Sram => 0.0,
    };
    // Weights and state share the same external channel when both live there.
    let shared_memory = p.weight_backend == WeightBackend::ExternalDdr && p.kv_backend == KvBackend::ExternalDdr;
    let memory_time = if shared_memory { weight_time + kv_time } else { weight_time.max(kv_time) };
    let token_time = compute_time.max(memory_time);
    let stream_rate = 1.0 / token_time;
    let prefill_rate = (d / compute_time).min(1.0 / memory_time.max(1e-12));

    // Speculative rounds fill the idle dies: drafts are checked in one pipelined pass.
    let (effective_rate, tokens_per_round) = if depth > 0 {
        let head_cycles = totals.mtp_weights / n * group_overhead * mult.cycle_factor
            + totals.mtp_vector_ops / lanes
            + totals.mtp_matvec_stages * matvec_latency
            + totals.mtp_vector_stages * vector_latency;
        let head_time = head_cycles / (freq * 1e9);
        let accepted: f64 = MTP_ACCEPTANCE.iter().take(depth as usize).sum();
        let tokens = 1.0 + accepted;
        let round = token_time + depth as f64 * (head_time + token_time / d);
        (tokens / round, tokens)
    } else {
        (stream_rate, 1.0)
    };

    // Energy per token.
    let e_mac = totals.weights * mult.energy_pj;
    let read_bits = read_bytes * 8.0;
    let e_read = read_bits
        * match p.weight_backend {
            WeightBackend::Sram => {
                let low_voltage = if p.has("sram_baja_tension") { 0.7 } else { 1.0 };
                tech.e_sram_fj_bit * low_voltage / 1e3
            }
            WeightBackend::Rom => tech.e_sram_fj_bit * 0.3 / 1e3,
            WeightBackend::Hbm4 => HBM4_PJ_PER_BIT,
            WeightBackend::StackedDram => STACKED_DRAM_PJ_PER_BIT,
            WeightBackend::ExternalDdr => DDR_PJ_PER_BIT,
        };
    let e_vector = totals.vector_ops * ((p.vector_bits * p.vector_bits + 2 * p.vector_bits) as f64) * tech.e_fa_pj;
    let kv_bit_energy = match p.kv_backend {
        KvBackend::Hbm4 => HBM4_PJ_PER_BIT,
        KvBackend::ExternalDdr => DDR_PJ_PER_BIT,
        KvBackend::Sram => tech.e_sram_fj_bit / 1e3,
    };
    let e_kv = kv_read_bytes * 8.0 * kv_bit_energy;
    let e_delta = 2.0 * state_bits * tech.e_sram_fj_bit / 1e3;
    let e_state = e_kv + e_delta;
    let hop_mm = die_area.sqrt() * 0.02;
    let e_noc = totals.weights * p.activation_bits as f64 * tech.e_wire_pj_bit_mm * hop_mm;
    let e_scale = if p.uses_groups() {
        totals.weights / p.group_size as f64 * (p.scale_bits * p.effective_accumulator()) as f64 * tech.e_fa_pj
    } else {
        0.0
    };
    let energy_token_mj = (e_mac + e_read + e_vector + e_state + e_noc + e_scale) * 1e-9;

    let sram_on_chip_bits = match p.weight_backend {
        WeightBackend::Sram => store.chip_bits * sram_margin(p),
        _ => 0.0,
    } + if p.kv_backend == KvBackend::Sram { kv_bits } else { 0.0 }
        + if p.kv_backend == KvBackend::ExternalDdr { 0.0 } else { state_bits + rollback_bits };
    let sram_leak_scale = if p.has("sram_baja_tension") { 0.5 } else { 1.0 };
    let leak_sram = sram_on_chip_bits * tech.sram_leak_pw_bit * 1e-12 * sram_leak_scale;
    let logic_area = (die_area - blocks.iter().filter(|b| b.id == "cell_weights").map(|b| b.per_die_mm2).sum::<f64>()) * d;
    let gated = if p.has("apagado_dies_inactivos") { (1.0 + 0.1 * (d - 1.0)) / d } else { 1.0 };
    let leak_logic = logic_area * tech.logic_leak_w_mm2 * gated;
    let leakage_w = leak_sram + leak_logic;
    let dynamic_w = energy_token_mj * 1e-3 * stream_rate;
    let power_w = dynamic_w + leakage_w;

    let image_bytes = (store.model_bits + store.external_bits) / 8.0;
    let load_seconds = image_bytes / p.load_interface.bytes_per_second();

    let cell_bytes = store.chip_bits / (d * n) / 8.0;
    let weight_share = weight_cycles / total_cycles;
    let vector_share = vector_cycles / total_cycles;
    let noc_bytes = totals.matvec_stages * n * (totals.weights / totals.matvec_stages.max(1.0) / n).sqrt().max(1.0)
        * p.activation_bits as f64 / 8.0;
    let link_bytes = model.hidden_size as f64 * p.activation_bits as f64 / 8.0 * d;
    let density = tech.sram_density(store.chip_bits / (d * n) / 1e3);
    for b in &mut blocks {
        let (activity, energy_pj, bytes, why) = match b.id {
            "cell_weights" => (weight_share, e_read, read_bytes,
                format!("{:.1} KB por celda con {} celdas por die, a unos {density:.0} Mbit/mm².", cell_bytes / 1e3, p.macs_per_die)),
            "cell_macs" => (weight_share, e_mac, 0.0,
                format!("{} multiplicadores de {} × {} bits para pesos de {} bits en {}, de {:.0} µm² cada uno.", p.macs_per_die, cell.significand_bits(), p.activation_bits, store.widest_weight_bits, coding_name(p.weight_coding), mult.area_um2)),
            "weight_tables" => (weight_share, 0.0, 0.0,
                format!("{:.1} Kbit de tabla por celda.", p.cell_table_bits() / 1e3)),
            "scale_units" => (weight_share, e_scale, 0.0,
                format!("Una unidad por celda, usada una vez cada {} pesos.", p.group_size)),
            "activation_noc" => (weight_share, e_noc, noc_bytes,
                format!("Árbol de {tree_levels:.0} niveles sobre {} celdas.", p.macs_per_die)),
            "vector_units" => (vector_share, e_vector, 0.0,
                format!("{} carriles de {} bits.", p.vector_lanes_per_die, p.vector_bits)),
            "function_units" => (vector_share, 0.0, 0.0,
                format!("{} unidades con tablas de 64 Kbit.", p.function_units_per_die)),
            "activation_buffers" => (1.0, 0.0, 0.0, "Ocho vectores del tamaño oculto y dos del ancho intermedio.".into()),
            "kv_sram" => (vector_share, e_kv, kv_read_bytes,
                format!("{:.2} GB para {} tokens, repartidos entre {dies} dies.", kv_bits / 8e9, p.context_tokens)),
            "delta_state" => (vector_share, e_delta, 2.0 * state_bits / 8.0,
                format!("{:.0} MB a {} bits, repartidos entre {dies} dies.", state_bits / 8e6, p.delta_state_bits)),
            "mtp_rollback" => (0.0, 0.0, 0.0, format!("{depth} copias del estado recurrente.")),
            "control" => (1.0, 0.0, 0.0, "Área fija estimada por die.".into()),
            "links" => (link_cycles / total_cycles, 0.0, link_bytes, "Un punto de enlace por die.".into()),
            _ => (0.0, 0.0, 0.0, "Área fija estimada.".into()),
        };
        b.activity = activity.clamp(0.0, 1.0);
        b.energy_mj = energy_pj * 1e-9;
        b.bytes_token = bytes;
        b.why = why;
    }
    let floorplan = build_floorplan(model, p, &blocks, dies, die_area);

    let mut metrics = Vec::new();
    let mut m = |id, section, label, value, unit, confidence, why: String| {
        metrics.push(Metric { id, section, label, value, unit, confidence, why });
    };
    use Confidence::*;

    m("model_params_g", "Modelo", "Parámetros del modelo cargado", included / 1e9, "G", Derivada,
        "Suma de los tensores del config.json con las opciones activas.".into());
    m("model_weights_gb", "Modelo", "Pesos del modelo en memoria", store.model_bits / 8e9, "GB", Derivada,
        "Bits por peso, escalas y sesgos por grupo y optimizaciones de almacenamiento.".into());
    m("chip_capacity_gb", "Modelo", "Capacidad de pesos del chip", store.chip_bits / 8e9, "GB", Derivada,
        "El chip se dimensiona para el máximo de parámetros, no solo para este modelo.".into());
    m("load_seconds", "Modelo", "Tiempo de carga de la imagen", load_seconds, "s", Derivada,
        "Tamaño de la imagen dividido por la interfaz de carga.".into());

    m("dies", "Silicio", "Dies", d, "", Supuesta, "El mínimo que cumple el área máxima por die.".into());
    m("die_area_mm2", "Silicio", "Área por die", die_area, "mm²", Supuesta, "Suma de los bloques del die.".into());
    m("total_area_mm2", "Silicio", "Área total", total_area, "mm²", Supuesta, "Área por die por número de dies.".into());
    m("macs_total", "Silicio", "Multiplicadores de pesos en total", n * d, "", Derivada,
        "Con un solo usuario solo trabajan los de un die cada vez.".into());
    m("cell_sram_kb", "Silicio", "Memoria de pesos por celda", cell_bytes / 1e3, "KB", Derivada,
        "Capacidad total repartida entre todas las celdas.".into());

    m("tok_s", "Velocidad", "Generación con un usuario", effective_rate, "tok/s", Supuesta,
        if depth > 0 { format!("Incluye MTP: {tokens_per_round:.2} tokens por ronda con las tasas de aceptación medidas por MTPLX.") }
        else { "Un token recorre todas las etapas antes de empezar el siguiente.".into() });
    m("tok_s_base", "Velocidad", "Sin especulación", stream_rate, "tok/s", Supuesta,
        "Inverso del tiempo de un token.".into());
    m("prefill_tok_s", "Velocidad", "Lectura de prompt", prefill_rate, "tok/s", Supuesta,
        "Los tokens ya se conocen, así que todos los dies trabajan a la vez.".into());
    m("freq_ghz", "Velocidad", "Frecuencia efectiva", freq, "GHz", Supuesta,
        format!("Limitada a {fmax:.2} GHz por las etapas del multiplicador."));
    m("cycles_token", "Velocidad", "Ciclos por token", total_cycles, "", Supuesta,
        format!("Pesos {weight_cycles:.0}, vectores {vector_cycles:.0}, latencia {latency_cycles:.0}, enlaces {link_cycles:.0}."));

    m("weight_bytes_token", "Ancho de banda", "Pesos leídos por token", read_bytes / 1e9, "GB", Derivada,
        "Cada peso de las matrices se usa una vez por token.".into());
    m("weight_bandwidth", "Ancho de banda", "Ancho de banda de pesos necesario", read_bytes * stream_rate / 1e12, "TB/s", Derivada,
        "Pesos por token multiplicados por la velocidad.".into());
    m("link_bandwidth", "Ancho de banda", "Tráfico entre dies", model.hidden_size as f64 * p.activation_bits as f64 / 8.0 * stream_rate / 1e6, "MB/s", Derivada,
        "Solo viaja el vector de activación entre capas.".into());

    m("kv_gb", "Estado", "Caché KV", kv_bits / 8e9, "GB", Derivada,
        "Valores por token de contexto por bits y por tokens de contexto.".into());
    m("delta_state_mb", "Estado", "Estado de DeltaNet y convolución", state_bits / 8e6, "MB", Derivada,
        "Matrices recurrentes de las capas lineales e historia de las convoluciones.".into());
    m("kv_kib_token", "Estado", "KV por token de contexto", model.state.kv_values_per_token as f64 * p.kv_bits as f64 / 8.0 / 1024.0, "KiB", Derivada,
        "Crece linealmente con el contexto.".into());

    m("energy_token_mj", "Energía", "Energía por token", energy_token_mj, "mJ", Supuesta,
        "Multiplicar, leer pesos, unidades vectoriales, estado y red.".into());
    m("power_w", "Energía", "Potencia generando", power_w, "W", Supuesta,
        format!("Dinámica {dynamic_w:.0} W y fugas {leakage_w:.0} W."));

    Ok(Report { model: model.name.clone(), dies, die_area_mm2: die_area, total_area_mm2: total_area, blocks, metrics, violations, floorplan })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    const QWEN: &str = include_str!("../../../models/qwen3.8-27b.json");

    fn qwen() -> ModelGraph {
        ModelGraph::from_config_json("qwen", QWEN).unwrap()
    }

    #[test]
    fn real_mixed_checkpoint_size_is_reproduced() {
        // Sum of the text tensors in the local MTPLX Speed checkpoint: 19.523 GB.
        let report = estimate(&qwen(), &ChipParams::default()).unwrap();
        let gb = report.metric("model_weights_gb").unwrap();
        assert!((gb / 19.523 - 1.0).abs() < 0.015, "{gb}");
    }

    #[test]
    fn state_sizes_match_the_earlier_analysis() {
        let report = estimate(&qwen(), &ChipParams::default()).unwrap();
        assert!((report.metric("kv_kib_token").unwrap() - 64.0).abs() < 1e-9);
        let state = report.metric("delta_state_mb").unwrap();
        assert!(state > 151.0 && state < 156.0, "{state}");
    }

    #[test]
    fn default_design_needs_several_dies_and_is_fast() {
        let report = estimate(&qwen(), &ChipParams::default()).unwrap();
        assert!((4..=14).contains(&report.dies), "{} dies", report.dies);
        assert!(report.violations.iter().all(|v| v.severity != Severity::Error));
        let rate = report.metric("tok_s").unwrap();
        assert!(rate > 1_000.0 && rate < 8_000.0, "{rate}");
    }

    #[test]
    fn estimator_tracks_the_published_hc1_within_twenty_percent() {
        // Taalas HC1: Llama 3.1 8B in ROM, 815 mm², 14.4-17 k tok/s measured, 200-250 W.
        let llama = ModelGraph::from_config_json("llama", include_str!("../../../models/llama-3.1-8b.json")).unwrap();
        let p = ChipParams {
            max_params_g: 8.1,
            weight_backend: WeightBackend::Rom,
            use_model_mix: false,
            group_size: 0,
            weight_bits: 4,
            context_tokens: 1024,
            vector_lanes_per_die: 65_536,
            ..Default::default()
        };
        let r = estimate(&llama, &p).unwrap();
        let rate = r.metric("tok_s").unwrap();
        let power = r.metric("power_w").unwrap();
        assert_eq!(r.dies, 1);
        assert!((r.die_area_mm2 / 815.0 - 1.0).abs() < 0.2, "área {}", r.die_area_mm2);
        assert!((14_400.0 * 0.8..17_000.0 * 1.2).contains(&rate), "velocidad {rate}");
        assert!((200.0 * 0.8..250.0 * 1.2).contains(&power), "potencia {power}");
    }

    #[test]
    fn fixed_rom_needs_less_silicon_than_sram() {
        let sram = estimate(&qwen(), &ChipParams::default()).unwrap();
        let rom = estimate(&qwen(), &ChipParams { weight_backend: WeightBackend::Rom, ..Default::default() }).unwrap();
        assert!(rom.total_area_mm2 < sram.total_area_mm2);
    }

    #[test]
    fn longer_context_grows_kv_and_slows_generation() {
        let short = estimate(&qwen(), &ChipParams::default()).unwrap();
        let long = estimate(&qwen(), &ChipParams { context_tokens: 262_144, ..Default::default() }).unwrap();
        assert!(long.metric("kv_gb").unwrap() > 30.0 * short.metric("kv_gb").unwrap());
        assert!(long.metric("tok_s").unwrap() < short.metric("tok_s").unwrap());
    }

    #[test]
    fn a_million_token_context_is_sized_and_flagged() {
        let sram = estimate(&qwen(), &ChipParams { context_tokens: 1_048_576, ..Default::default() }).unwrap();
        assert!((sram.metric("kv_gb").unwrap() - 68.7).abs() < 0.2);
        assert!(sram.violations.iter().any(|v| v.severity == Severity::Aviso && v.message.contains("262144")));
        let base = estimate(&qwen(), &ChipParams::default()).unwrap();
        assert!(sram.dies > base.dies, "la caché en SRAM debe costar más silicio");

        let hbm = estimate(&qwen(), &ChipParams { context_tokens: 1_048_576, kv_backend: KvBackend::Hbm4, ..Default::default() }).unwrap();
        assert!(hbm.total_area_mm2 < sram.total_area_mm2);

        let narrow = estimate(&qwen(), &ChipParams { context_tokens: 1_048_576, kv_backend: KvBackend::Hbm4, hbm_stacks: 2, ..Default::default() }).unwrap();
        assert!(narrow.metric("tok_s").unwrap() < hbm.metric("tok_s").unwrap(), "con pocas pilas debe limitar el ancho de banda");
    }

    #[test]
    fn hbm_is_limited_by_memory_bandwidth() {
        let p = ChipParams { weight_backend: WeightBackend::Hbm4, ..Default::default() };
        let report = estimate(&qwen(), &p).unwrap();
        let rate = report.metric("tok_s").unwrap();
        assert!(rate > 500.0 && rate < 1_500.0, "{rate}");
    }

    #[test]
    fn mtp_speeds_up_generation_and_costs_area() {
        let base = estimate(&qwen(), &ChipParams::default()).unwrap();
        let mtp = estimate(&qwen(), &ChipParams { mtp_enabled: true, ..Default::default() }).unwrap();
        assert!(mtp.metric("tok_s").unwrap() > base.metric("tok_s").unwrap());
        assert!(mtp.blocks.iter().any(|b| b.id == "mtp_rollback"));
    }

    #[test]
    fn every_optimization_changes_the_design() {
        let base = serde_json::to_string(&estimate(&qwen(), &ChipParams::default()).unwrap().metrics).unwrap();
        for option in catalog() {
            let mut p = ChipParams::default();
            p.optimizations.insert(option.id.to_string());
            let report = estimate(&qwen(), &p).unwrap();
            let blocks = report.blocks.iter().map(|b| b.per_die_mm2).sum::<f64>();
            let default_blocks = estimate(&qwen(), &ChipParams::default()).unwrap().die_area_mm2;
            let metrics = serde_json::to_string(&report.metrics).unwrap();
            assert!(metrics != base || (blocks - default_blocks).abs() > 1e-9, "{} no cambia nada", option.id);
        }
    }

    #[test]
    fn incompatible_optimizations_are_reported() {
        let mut p = ChipParams::default();
        p.optimizations.insert("sin_multiplicadores".into());
        p.optimizations.insert("cim_analogico".into());
        let report = estimate(&qwen(), &p).unwrap();
        assert!(report.violations.iter().any(|v| v.severity == Severity::Error));
    }

    #[test]
    fn oversized_model_is_rejected_with_a_fix() {
        let report = estimate(&qwen(), &ChipParams { max_params_g: 10.0, ..Default::default() }).unwrap();
        let v = report.violations.iter().find(|v| v.severity == Severity::Error).unwrap();
        assert!(v.message.contains("no cabe") && !v.fix.is_empty());
    }

    #[test]
    fn fpga_preset_is_slow_but_runs_the_full_model() {
        let p = ChipParams {
            node: "fpga_u250".into(),
            max_params_g: 28.0,
            context_tokens: 2048,
            macs_per_die: 1024,
            vector_lanes_per_die: 256,
            freq_ghz: 0.25,
            weight_backend: WeightBackend::ExternalDdr,
            kv_backend: KvBackend::ExternalDdr,
            ..Default::default()
        };
        let report = estimate(&qwen(), &p).unwrap();
        let rate = report.metric("tok_s").unwrap();
        assert_eq!(report.dies, 1);
        assert!(report.violations.iter().all(|v| v.severity != Severity::Error));
        assert!(rate > 2.0 && rate < 7.0, "{rate}");
    }

    #[test]
    fn small_models_fit_on_one_small_die() {
        let stories = ModelGraph::from_config_json("s", include_str!("../../../models/stories260k.json")).unwrap();
        let p = ChipParams {
            max_params_g: 0.00026,
            macs_per_die: 16,
            vector_lanes_per_die: 4,
            use_model_mix: false,
            group_size: 0,
            weight_bits: 8,
            context_tokens: 256,
            ..Default::default()
        };
        let report = estimate(&stories, &p).unwrap();
        assert_eq!(report.dies, 1);
        assert!(report.die_area_mm2 < 20.0, "{}", report.die_area_mm2);
    }

    #[test]
    fn estimate_is_fast_enough_for_live_sliders() {
        let model = qwen();
        let start = Instant::now();
        for _ in 0..100 {
            estimate(&model, &ChipParams::default()).unwrap();
        }
        // Debug build, so this bound is loose for the 16 ms target in release.
        assert!(start.elapsed().as_millis() < 1600, "{:?}", start.elapsed());
    }
}
