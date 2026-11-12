use crate::floorplan::{build_floorplan, Floorplan};
use crate::optim::catalog;
use crate::params::{ChipParams, CircuitGoal, DieMode, KvBackend, LinkKind, WeightBackend};
use crate::tech::{node, TechNode};
use netlist::format::{switching_per_mac, WeightCoding};
use netlist::multiplier::MultiplierKind;
use netlist::{cell_design, shared_unit_design, CellConfig, Simulator, MAX_MULTIPLES_BITS};
use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap};
use std::rc::Rc;
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
    /// A value of that parameter that clears this, when there is one to offer.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub value: Option<serde_json::Value>,
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
    /// What the generator chose for these parameters, per weight width.
    pub decisions: Vec<CircuitDecision>,
    pub macs_per_die: u64,
    #[serde(skip)]
    pub resolved: ChipParams,
}

impl Report {
    pub fn metric(&self, id: &str) -> Option<f64> {
        self.metrics.iter().find(|m| m.id == id).map(|m| m.value)
    }
}

/// AND, OR, XOR, NOT and MUX delays as a fraction of one full adder's carry path.
/// One weighted gate toggle measured in the cell, in full-adder energy units, calibrated
/// so that an 8 × 16 multiplier matches the published energy of a full-adder array.
const SWITCHING_TO_FULL_ADDERS: f64 = 0.226;

const GATE_DELAYS: [f64; 5] = [0.25, 0.25, 0.5, 0.12, 0.4];

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

/// The cell whose gates set the clock: the real one, with its memory as shallow as can be
/// while it keeps the same error correction on its read path. The path does not grow with
/// the depth past that, so cells of any size share one timing.
pub fn timed_cell(p: &ChipParams, weight_bits: u32, cell_bits: f64) -> CellConfig {
    let addr_bits = (cell_bits / weight_bits.max(1) as f64).max(2.0).log2().ceil() as u32;
    let real = CellConfig { weight_bits, ..p.cell_config(addr_bits) };
    let shallow = if real.has_ecc() { real.ecc_lanes().ilog2() + 1 } else { 1 };
    CellConfig { addr_bits: shallow, ..real }
}

/// Logic area of one cell, from the gates its generated netlist actually contains.
fn netlist_cell_um2(tech: &TechNode, p: &ChipParams, widest: u32, addr_bits: u32) -> f64 {
    cell_logic_um2(tech, CellConfig { weight_bits: widest, ..p.cell_config(addr_bits) })
}

/// Gate counts of a cell, and of the unit it shares with the die.
type CellGates = Rc<(BTreeMap<String, u64>, BTreeMap<String, u64>)>;

thread_local! {
    /// Gate counts and critical paths of generated cells: building their netlists is the
    /// slow part of an estimate, and the same cells come back on every change of a slider.
    static CELL_GATES: RefCell<HashMap<String, CellGates>> = RefCell::new(HashMap::new());
    static CELL_PATHS: RefCell<HashMap<String, Option<f64>>> = RefCell::new(HashMap::new());
    static CELL_SWITCHING: RefCell<HashMap<String, f64>> = RefCell::new(HashMap::new());
}

/// Gates that toggle per multiplication, simulated on the cell with random data.
fn cell_switching(config: CellConfig, zeros: f64) -> f64 {
    let key = format!("{config:?}{zeros}");
    if let Some(found) = CELL_SWITCHING.with(|c| c.borrow().get(&key).copied()) {
        return found;
    }
    let switching = switching_per_mac(config, zeros);
    CELL_SWITCHING.with(|c| c.borrow_mut().insert(key, switching));
    switching
}

/// The gates of a cell and of the unit it shares with the die, if any.
fn cell_gates(config: CellConfig) -> CellGates {
    let key = format!("{config:?}");
    if let Some(found) = CELL_GATES.with(|c| c.borrow().get(&key).cloned()) {
        return found;
    }
    let gates = Rc::new((cell_design(config).flat_counts(), shared_unit_design(config).map(|unit| unit.flat_counts()).unwrap_or_default()));
    CELL_GATES.with(|c| c.borrow_mut().insert(key, gates.clone()));
    gates
}

/// The critical path of a cell in full-adder delays, or None if its netlist does not simulate.
fn cell_path(config: CellConfig) -> Option<f64> {
    let key = format!("{config:?}");
    if let Some(found) = CELL_PATHS.with(|c| c.borrow().get(&key).copied()) {
        return found;
    }
    let path = Simulator::new(&cell_design(config)).ok().map(|sim| sim.critical_path(GATE_DELAYS));
    CELL_PATHS.with(|c| c.borrow_mut().insert(key, path));
    path
}

/// The logic of a cell, without its weight memory, after synthesis.
fn cell_logic_um2(tech: &TechNode, config: CellConfig) -> f64 {
    // Yosys and ABC shrink each generated multiplier by a measured factor (tests/golden/synthesis.md).
    let optimized = match config.multiplier {
        MultiplierKind::Array => 0.65,
        MultiplierKind::Wallace => 0.78,
        MultiplierKind::Booth => 0.75,
        MultiplierKind::Multiples => MULTIPLES_SYNTHESIS,
    };
    let area_of = |counts: &BTreeMap<String, u64>| -> f64 {
        counts
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
    };
    // Floats keep more after synthesis than the integer multipliers that factor came from:
    // their 8-bit cell measured 0.79 of the others' ratio on all three open processes
    // (tests/golden/asic.json).
    let format = if config.coding == WeightCoding::Float { FLOAT_SYNTHESIS } else { 1.0 };
    // The unit of shared multiples is paid once per die, not once per cell.
    let gates = cell_gates(config);
    optimized * format * (area_of(&gates.0) - area_of(&gates.1))
}

/// How much more of a float cell survives synthesis than of an integer one.
const FLOAT_SYNTHESIS: f64 = 1.26;

/// How much of a cell's synthesis shrinks when it uses precomputed multiples.
const MULTIPLES_SYNTHESIS: f64 = 0.75;

/// Share of stored codes that are zero, measured on real 4 and 8-bit checkpoints.
fn zero_fraction(weight_bits: u32) -> f64 {
    if weight_bits <= 4 { 0.106 } else { 0.016 }
}

struct Multiplier {
    area_um2: f64,
    energy_pj: f64,
    delay_factor: f64,
    cycle_factor: f64,
}

/// What the estimator expects OpenROAD to measure on a cell laid out on its own, as the
/// layout flow builds it: its logic, its few words of weights in flip-flops (each bit a
/// flip-flop with a hold multiplexer, read through a tree of multiplexers), and the clock its
/// critical path allows, without the process cap.
pub fn laid_out_cell(tech: &TechNode, config: CellConfig) -> (f64, f64) {
    let design = cell_design(config);
    let bits = design.memory_bits() as f64;
    let words = (1u64 << config.addr_bits) as f64;
    let memory = bits * (tech.ff_um2 + tech.mux_um2) + bits / words * (words - 1.0) * tech.mux_um2;
    let path = Simulator::new(&design).map_or(0.0, |sim| sim.critical_path(GATE_DELAYS));
    (cell_logic_um2(tech, config) + memory, 1.0 / (path * tech.fa_delay_ns + tech.ff_overhead_ns))
}

/// The fastest clock the weights' SRAM macros allow, when the process's are known to be
/// slower than its logic.
fn sram_clock_ghz(tech: &TechNode, p: &ChipParams) -> f64 {
    match tech.sram_min_period_ns {
        Some(period) if p.weight_backend == WeightBackend::Sram => 1.0 / period,
        _ => f64::INFINITY,
    }
}

/// The name of a compute cell, as a measured layout refers to it: everything that shapes its
/// gates except the depth of its weight memory, which the layout replaces by a few words.
pub fn cell_key(config: &CellConfig) -> String {
    let name = |v: serde_json::Value| v.as_str().unwrap_or("").to_string();
    let c = config;
    let memory = if c.fixed_weights { "-rom" } else if c.has_ecc() { "-ecc" } else { "" };
    format!(
        "{}{}-{}x{}-acc{}-{}-{}+{}{memory}{}",
        name(serde_json::json!(c.coding)),
        c.exponent_bits,
        c.weight_bits,
        c.activation_bits,
        c.accumulator_bits.max(c.required_accumulator()),
        name(serde_json::json!(c.multiplier)),
        c.mac_stages,
        c.adder_stages,
        if c.skip_zeros { "-ceros" } else { "" },
    )
}

/// The measured layout of this very cell on this process, if the design carries one: the
/// key is `node:cell_key`, so a change of circuit or process drops back to the estimate.
fn measured(p: &ChipParams, config: &CellConfig) -> Option<(f64, f64)> {
    (p.measured_cell_um2 > 0.0 && p.measured_cell == format!("{}:{}", p.node, cell_key(config))).then_some((p.measured_cell_um2, p.measured_cell_fmax_ghz))
}

fn weight_multiplier(tech: &TechNode, p: &ChipParams, widest: u32, cell_bits: f64) -> Multiplier {
    let switching = cell_switching(CellConfig { weight_bits: widest, ..p.cell_config(5) }, zero_fraction(widest));
    let addr_bits = (cell_bits / widest as f64).max(2.0).log2().ceil() as u32;
    // A cell laid out with OpenROAD counts what it measured instead.
    let cell_logic = measured(p, &CellConfig { weight_bits: widest, ..p.cell_config(addr_bits) }).map_or_else(|| netlist_cell_um2(tech, p, widest, addr_bits), |(area, _)| area);
    let accumulator_bits = p.effective_accumulator() as f64;
    let accumulator = accumulator_bits * (tech.fa_um2 + tech.ff_um2);
    let mut area = (cell_logic - accumulator).max(0.0);
    let mut energy = switching * SWITCHING_TO_FULL_ADDERS * tech.e_fa_pj;
    let mut delay_factor = 1.0;
    let mut cycle_factor = 1.0;
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
    if p.has("pesos_csd") && p.weight_backend == WeightBackend::Rom {
        area *= 0.23;
        energy *= 0.5;
        delay_factor *= 0.7;
    }
    let decoder = if p.has("compresion_pesos") { 8.0 } else { 0.0 };
    Multiplier {
        area_um2: area + accumulator + decoder,
        energy_pj: energy,
        delay_factor,
        cycle_factor,
    }
}

/// How much more state a die reserves than this model needs: for depth and for width.
fn envelope_scale(model: &ModelGraph, p: &ChipParams) -> (f64, f64) {
    if p.die_mode == DieMode::Custom {
        return (1.0, 1.0);
    }
    ((p.max_layers as f64 / model.num_layers.max(1) as f64).max(1.0), (p.max_hidden_size as f64 / model.hidden_size.max(1) as f64).max(1.0))
}

fn vocabulary_room(model: &ModelGraph, p: &ChipParams) -> f64 {
    if p.die_mode == DieMode::Custom { model.vocab_size as f64 } else { (p.max_vocabulary as f64).max(model.vocab_size as f64) }
}

fn check_envelope(model: &ModelGraph, p: &ChipParams, out: &mut Vec<Violation>) {
    if p.weight_backend == WeightBackend::Rom {
        out.push(Violation {
            severity: Severity::Error,
            message: "Con ROM cada die lleva grabados pesos distintos, así que no pueden ser todos iguales.".into(),
            fix: "Usa SRAM o genera cada die a medida.".into(),
            field: Some("die_mode"), value: Some(serde_json::json!("custom")),
        });
    }
    for (too_big, what, field, model_value) in [
        (model.num_layers > p.max_layers, "capas", "max_layers", model.num_layers as u64),
        (model.hidden_size > p.max_hidden_size as u64, "ancho interno", "max_hidden_size", model.hidden_size),
        (model.vocab_size > p.max_vocabulary as u64, "vocabulario", "max_vocabulary", model.vocab_size),
    ] {
        if too_big {
            out.push(Violation {
                severity: Severity::Error,
                message: format!("El modelo supera el límite de {what} de este diseño."),
                fix: "Sube ese límite.".into(),
                field: Some(field), value: Some(serde_json::json!(model_value)),
            });
        }
    }
}

/// The largest measured, Llama 3.2's; a test keeps real tables under it. Qwen uses half.
/// The largest multi-die package in volume, about five and a half reticles.
const PACKAGE_MAX_MM2: f64 = 4700.0;
const UCIE_ENDPOINT_MM2: f64 = 2.0;
const SERDES_ENDPOINT_MM2: f64 = 6.0;
const UCIE_HOP_NS: f64 = 10.0;
/// What one endpoint of each kind carries, in bits per nanosecond: a UCIe module of 64
/// lanes at 32 GT/s, and eight SerDes lanes of 112 Gb/s.
const UCIE_ENDPOINT_GBPS: f64 = 2048.0;
const SERDES_ENDPOINT_GBPS: f64 = 896.0;
const SERDES_HOP_NS: f64 = 150.0;
/// A link over I/O pins: 16 data pins each way at 50 MHz, about what the general-purpose
/// pads of an open shuttle's harness give, and a hop through two pads, the board and a
/// synchroniser.
const PIN_LINK_BITS_PER_NS: f64 = 16.0 * 0.05;
const PIN_HOP_NS: f64 = 60.0;

/// How a ring of dies is cut into packages, and what each hop travels over.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LinkPlan {
    pub packages: u32,
    pub ucie_hops: u32,
    pub serdes_hops: u32,
    pub pin_hops: u32,
}

impl LinkPlan {
    pub fn delay_ns(&self) -> f64 {
        self.ucie_hops as f64 * UCIE_HOP_NS + self.serdes_hops as f64 * SERDES_HOP_NS + self.pin_hops as f64 * PIN_HOP_NS
    }
}

/// The link the dies really use: an open shuttle hands each die back in its own package,
/// with only its I/O pins to reach the others.
pub fn effective_link(p: &ChipParams, tech: &TechNode) -> LinkKind {
    if p.link == LinkKind::Auto && tech.open_pdk && tech.fabricable { LinkKind::Pins } else { p.link }
}

/// Fills each package before opening the next; a ring needs one board hop per package.
pub fn link_plan(link: LinkKind, dies: u32, die_area_mm2: f64) -> LinkPlan {
    let per_package = ((PACKAGE_MAX_MM2 / die_area_mm2.max(1.0)).floor() as u32).max(1);
    let packages = dies.div_ceil(per_package).max(1);
    match link {
        LinkKind::Serdes => LinkPlan { packages: dies, ucie_hops: 0, serdes_hops: dies, pin_hops: 0 },
        LinkKind::Pins => LinkPlan { packages: dies, ucie_hops: 0, serdes_hops: 0, pin_hops: dies },
        LinkKind::Ucie => LinkPlan { packages: 1, ucie_hops: dies, serdes_hops: 0, pin_hops: 0 },
        LinkKind::Auto if packages == 1 => LinkPlan { packages, ucie_hops: dies, serdes_hops: 0, pin_hops: 0 },
        LinkKind::Auto => LinkPlan { packages, ucie_hops: dies - packages, serdes_hops: packages, pin_hops: 0 },
    }
}

/// Measured on the generated cell, for words of 64 weight bits.
const ECC_GATES_PER_CELL: f64 = 430.0;
pub const TOKENIZER_BITS_PER_TOKEN: f64 = 500.0;
/// The CPU's program memory, the firmware's size rounded up to a power of two.
pub const FIRMWARE_PROGRAM_BYTES: u32 = 1 << 15;

/// Cycles sampling adds to each generated token. Logits are filtered as `lm_head`
/// produces them, so only a unit too narrow to keep up delays the die.
fn sampling_cycles(p: &ChipParams, model: &ModelGraph) -> f64 {
    let vocabulary = model.vocab_size as f64;
    let head = vocabulary * model.hidden_size as f64 / p.macs_per_die as f64;
    let stream = vocabulary / p.sampler_lanes.max(1) as f64;
    let entries = p.sampler_entries.max(1) as f64;
    // A random stream improves on the k-th best about k·ln(n/k) times.
    let insertions = entries * (1.0 + (vocabulary / entries).max(1.0).ln());
    let selection = 2.0 + 3.0 * entries;
    (stream - head).max(0.0) + insertions + selection
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
    /// Bits sent to every cell for each activation.
    broadcast_bits: f64,
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
    if p.weight_backend == WeightBackend::Sram && !p.has("sin_ecc_con_refresco") {
        push("cell_ecc", "Celdas: corrección ECC", n * ECC_GATES_PER_CELL * 0.5 * tech.fa_um2 * um2 * ROUTING_OVERHEAD,
            "Comprueba cada palabra de pesos y corrige el peso que se está leyendo.");
    }
    push("cell_macs", "Celdas: multiplicadores", n * mult.area_um2 * um2 * ROUTING_OVERHEAD,
        "Multiplican peso por activación y acumulan la suma de cada fila.");

    if p.cell_table_bits() > 0.0 {
        let table_mm2 = if p.weight_backend == WeightBackend::Rom {
            n * p.cell_table_bits() / 1e6 / tech.rom_mbit_mm2
        } else {
            n * p.cell_table_bits() * sram_margin(p) / 1e6 / tech.sram_density((p.cell_table_bits() / 1e3).max(1.0))
        };
        push("weight_tables", "Tablas de pesos", table_mm2,
            "Cada celda guarda los valores reales a los que apuntan sus códigos.");
    }

    if p.uses_groups() {
        let width = p.effective_accumulator() as f64;
        // A power-of-two scale is a barrel shifter, not a multiplier.
        // Applied over several cycles, each cycle takes only a part of the scale's bits.
        let steps = p.factor_steps.clamp(1, p.scale_bits.max(1)) as f64;
        let unit = if p.scale_is_multiplied() { (p.scale_bits as f64 / steps).ceil() * width * tech.fa_um2 + width * tech.ff_um2 * (steps > 1.0) as u8 as f64 } else { width * width.log2().ceil() * tech.mux_um2 };
        push("scale_units", "Escala y sesgo por grupo", n * unit * um2 * ROUTING_OVERHEAD,
            "Aplican la escala y el sesgo que comparten los pesos de cada grupo.");
    }

    push("activation_noc", "Red de activaciones",
        n * i.broadcast_bits * tech.ff_um2 * p.noc_stages.max(1) as f64 * 4.0 * um2,
        "Reparte cada vector de entrada a todas las celdas y recoge las sumas parciales.");

    let lane = (p.vector_bits * p.vector_bits) as f64 * tech.fa_um2
        + 2.0 * p.vector_bits as f64 * (tech.fa_um2 + tech.ff_um2);
    push("vector_units", "Unidades vectoriales",
        p.vector_lanes_per_die as f64 * lane * um2 * ROUTING_OVERHEAD,
        "Multiplican activación por activación o estado: atención Q·K, regla delta y puertas.");

    let table_bits = (1u64 << p.function_table_bits.clamp(4, 20)) as f64 * p.activation_bits as f64;
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

    let logit_bits = p.activation_bits as f64;
    let index_bits = (model.vocab_size.max(2) as f64).log2().ceil();
    let sampler_um2 = p.sampler_entries.max(1) as f64 * ((logit_bits + index_bits) * (tech.ff_um2 + tech.mux_um2) + logit_bits * tech.fa_um2)
        + p.sampler_lanes as f64 * logit_bits * tech.fa_um2
        + 24.0 * 16.0 * tech.fa_um2;
    push("sampler", "Unidad de muestreo", sampler_um2 * um2 * ROUTING_OVERHEAD / d,
        "Conserva los mejores logits y elige el siguiente token, con top-k y temperatura.");
    // Merge table, token offsets and the text of each token, read only by the CPU.
    let tokenizer_bits = vocabulary_room(model, p) * TOKENIZER_BITS_PER_TOKEN;
    let tokenizer_density = if p.weight_backend == WeightBackend::Rom { tech.rom_mbit_mm2 } else { tech.sram_density(1024.0) };
    push("tokenizer_tables", "Tablas del tokenizador", tokenizer_bits / 1e6 / tokenizer_density / d,
        "Vocabulario y fusiones que el firmware usa para convertir texto en tokens y volver.");
    // Not in the tech file's CPU figure: this grows with the context the design allows.
    let (text_bytes, arena_words) = tokenizer_rt::memory_for_context(p.context_tokens);
    let cpu_memory_bits = 8.0 * (FIRMWARE_PROGRAM_BYTES as f64 + text_bytes as f64 + 4.0 * arena_words as f64);
    let cpu_memory_share = if p.die_mode == DieMode::Identical { 1.0 } else { 1.0 / d };
    push("cpu_memory", "Memoria de la CPU", cpu_memory_bits / 1e6 / tech.sram_density(1024.0) * cpu_memory_share,
        "Programa del firmware y memoria donde el tokenizador trabaja con el texto de entrada.");
    push("control", "Secuenciador y microprograma", 0.8,
        "Ejecuta el microprograma que dirige cada etapa.");
    // Interchangeable dies all carry the CPU, even though only the first one leads.
    let control_share = if p.die_mode == DieMode::Identical { 1.0 } else { 1.0 / d };
    push("system", "CPU de control y carga", (tech.cpu_mm2 + tech.io_mm2) * control_share,
        if p.weight_backend == WeightBackend::Rom {
            "Tokenizador, muestreo y carga del microprograma. Los pesos ya vienen grabados."
        } else {
            "Tokenizador, muestreo y volcado de pesos desde la imagen."
        });
    if dies > 1 {
        // Sized for the worst each die can be, since its own area depends on this.
        // The pads of pin links are already in the die's I/O ring.
        let plan = link_plan(effective_link(p, tech), dies, p.die_max_mm2.min(tech.die_max_mm2));
        // A wider link takes as many endpoints as its bits per nanosecond need.
        let rate = p.link_words.max(1) as f64 * p.activation_bits as f64 * p.freq_ghz;
        let endpoints = |capacity: f64| (rate / capacity).ceil().max(1.0);
        let endpoint = (plan.ucie_hops as f64 * UCIE_ENDPOINT_MM2 * endpoints(UCIE_ENDPOINT_GBPS) + plan.serdes_hops as f64 * SERDES_ENDPOINT_MM2 * endpoints(SERDES_ENDPOINT_GBPS)) / d;
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
        "cell_macs" | "cell_ecc" | "scale_units" => Category::Calculo,
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
        "cpu_memory" => &["context_tokens"],
        "cell_ecc" => &["macs_per_die"],
        "cell_macs" => &["macs_per_die", "multiplier_kind", "weight_coding", "weight_bits", "weight_exponent_bits", "activation_bits", "mac_stages"],
        "scale_units" => &["group_size", "scale_bits", "factor_steps"],
        "activation_noc" => &["macs_per_die", "activation_bits", "noc_stages"],
        "vector_units" => &["vector_lanes_per_die", "vector_bits"],
        "function_units" => &["function_units_per_die", "function_table_bits"],
        "sampler" => &["sampler_lanes", "sampler_entries"],
        "activation_buffers" => &["activation_bits"],
        "kv_sram" => &["context_tokens", "kv_bits", "kv_backend"],
        "delta_state" => &["delta_state_bits"],
        "mtp_rollback" => &["mtp_enabled", "mtp_depth", "delta_state_bits"],
        "links" => &["link", "link_words", "dies", "chip_split", "tensor_group"],
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
        value: None,
    };
    let error = |message: &str, fix: &str| Violation {
        severity: Severity::Error,
        message: message.into(),
        fix: fix.into(),
        field: None,
        value: None,
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
            field: Some("accumulator_bits"), value: Some(serde_json::json!(p.effective_accumulator())),
        });
    }
    if p.has("pesos_csd") && p.weight_backend != WeightBackend::Rom {
        out.push(aviso(
            "Los pesos CSD solo ahorran área si están grabados en ROM: con memoria cargable la celda necesita un multiplicador general.",
            "Elige ROM como memoria de pesos.",
        ));
    }
}

/// One internal decision the generator took for these parameters, and what it measured.
#[derive(Clone, Debug, Serialize)]
pub struct CircuitDecision {
    /// The cell itself, as the gate generator builds it, and its name for a measured layout.
    pub cell: CellConfig,
    pub cell_key: String,
    pub weight_bits: u32,
    /// Share of the multiplications done with weights of this width.
    pub share: f64,
    pub multiplier: MultiplierKind,
    pub mac_stages: u32,
    pub adder_stages: u32,
    pub skip_zeros: bool,
    pub area_um2: f64,
    pub energy_pj: f64,
    pub fmax_ghz: f64,
}

#[derive(Clone, Copy)]
struct Candidate {
    multiplier: MultiplierKind,
    mac_stages: u32,
    adder_stages: u32,
    skip_zeros: bool,
    area_um2: f64,
    energy_pj: f64,
    fmax_ghz: f64,
}

thread_local! {
    /// Measured candidates: generating them costs real netlists, so each set is kept.
    static CANDIDATES: RefCell<HashMap<String, Rc<Vec<Candidate>>>> = RefCell::new(HashMap::new());
}

fn with_choice(p: &ChipParams, multiplier: MultiplierKind, mac_stages: u32, adder_stages: u32, skip_zeros: bool) -> ChipParams {
    let mut q = p.clone();
    q.multiplier_kind = multiplier;
    q.mac_stages = mac_stages;
    q.adder_stages = adder_stages;
    if skip_zeros {
        q.optimizations.insert("salto_ceros".into());
    } else {
        q.optimizations.remove("salto_ceros");
    }
    q
}

/// Wire millimetres of broadcast per cell, in the die's activation tree.
const BROADCAST_MM: f64 = 0.5;

/// Every way of building this cell that the generator can offer, each one measured.
fn candidates_for(tech: &TechNode, p: &ChipParams, weight_bits: u32, cell_bits: f64) -> Rc<Vec<Candidate>> {
    let addr_bits = (cell_bits / weight_bits as f64).max(2.0).log2().ceil() as u32;
    let key = format!("{}|{weight_bits}|{}|{}|{:?}|{}|{}|{:?}|{:?}|{addr_bits}|{}|{}|{}|{}", p.node, p.activation_bits, p.accumulator_bits, p.weight_coding, p.weight_exponent_bits, p.noc_stages, p.weight_backend, p.optimizations, p.scale_bits, p.measured_cell, p.measured_cell_um2, p.measured_cell_fmax_ghz);
    if let Some(found) = CANDIDATES.with(|c| c.borrow().get(&key).cloned()) {
        return found;
    }
    let forced_multiples = p.has("sin_multiplicadores");
    let forced_skip = p.has("salto_ceros");
    let mut list = Vec::new();
    for multiplier in [MultiplierKind::Array, MultiplierKind::Wallace, MultiplierKind::Booth, MultiplierKind::Multiples] {
        let is_multiples = multiplier == MultiplierKind::Multiples;
        if forced_multiples != is_multiples && (forced_multiples || p.weight_coding == WeightCoding::Codebook) {
            continue;
        }
        for mac_stages in 1..=if is_multiples { 1 } else { 3 } {
            for adder_stages in 1..=2 {
                for skip_zeros in [false, true] {
                    if (skip_zeros && is_multiples) || (forced_skip && !skip_zeros) {
                        continue;
                    }
                    let q = with_choice(p, multiplier, mac_stages, adder_stages, skip_zeros);
                    let cell = timed_cell(&q, weight_bits, cell_bits);
                    if is_multiples && cell.significand_bits() > MAX_MULTIPLES_BITS {
                        continue;
                    }
                    let mult = weight_multiplier(tech, &q, weight_bits, cell_bits);
                    let Some(path) = cell_path(cell) else { continue };
                    let logic_ns = path * tech.fa_delay_ns * mult.delay_factor;
                    // A wider broadcast costs registers on the way to the cell and energy on its wires.
                    let bits = cell.broadcast_bits() as f64;
                    list.push(Candidate {
                        multiplier,
                        mac_stages,
                        adder_stages,
                        skip_zeros,
                        area_um2: mult.area_um2 + bits * tech.ff_um2 * p.noc_stages.max(1) as f64 * 4.0,
                        energy_pj: mult.energy_pj + bits * tech.e_wire_pj_bit_mm * BROADCAST_MM,
                        fmax_ghz: match measured(&q, &cell) {
                            Some((_, ghz)) if ghz > 0.0 => ghz,
                            _ => tech.fmax_ghz.min(1.0 / (logic_ns + tech.ff_overhead_ns)),
                        }
                        .min(sram_clock_ghz(tech, &q)),
                    });
                }
            }
        }
    }
    let list = Rc::new(list);
    CANDIDATES.with(|c| c.borrow_mut().insert(key, list.clone()));
    list
}

/// The best candidate that reaches the clock, or the fastest if none does.
fn pick(list: &[Candidate], frequency: f64, goal: CircuitGoal) -> Option<Candidate> {
    let score = |c: &Candidate| match goal {
        CircuitGoal::Balanced => c.area_um2 * c.energy_pj,
        CircuitGoal::Area => c.area_um2,
        CircuitGoal::Energy => c.energy_pj,
    };
    let latency = |c: &Candidate| c.mac_stages + c.adder_stages;
    list.iter()
        .filter(|c| c.fmax_ghz >= frequency)
        .min_by(|a, b| score(a).total_cmp(&score(b)).then(latency(a).cmp(&latency(b))))
        .or_else(|| list.iter().max_by(|a, b| a.fmax_ghz.total_cmp(&b.fmax_ghz)))
        .copied()
}

/// Share of the matrix weights that are stored at the wider width.
fn wide_share(model: &ModelGraph, p: &ChipParams) -> f64 {
    let matrices = |wide_only: bool| model.params_where(|t| tensor_included(t, p) && !is_small_float(t) && t.block != Block::Embedding && (!wide_only || t.wide)) as f64;
    matrices(true) / matrices(false).max(1.0)
}

struct Resolved {
    params: ChipParams,
    decisions: Vec<CircuitDecision>,
    /// The narrower cell's parameters, its width and the share of work it does.
    narrow: Option<(ChipParams, u32, f64)>,
}

/// Chooses the circuit for these parameters, separately for each weight width.
fn resolve_circuit(tech: &TechNode, model: &ModelGraph, p: &ChipParams, widest: u32, cell_bits: f64) -> Resolved {
    if !p.auto_circuit {
        return Resolved { params: p.clone(), decisions: Vec::new(), narrow: None };
    }
    let target = p.freq_ghz.min(tech.fmax_ghz);
    let mut decisions = Vec::new();
    let mut choose = |weight_bits: u32, share: f64| {
        let choice = pick(&candidates_for(tech, p, weight_bits, cell_bits), target, p.circuit_goal)?;
        let cell = timed_cell(&with_choice(p, choice.multiplier, choice.mac_stages, choice.adder_stages, choice.skip_zeros), weight_bits, cell_bits);
        decisions.push(CircuitDecision { cell, cell_key: cell_key(&cell), weight_bits, share, multiplier: choice.multiplier, mac_stages: choice.mac_stages, adder_stages: choice.adder_stages, skip_zeros: choice.skip_zeros, area_um2: choice.area_um2, energy_pj: choice.energy_pj, fmax_ghz: choice.fmax_ghz });
        Some(with_choice(p, choice.multiplier, choice.mac_stages, choice.adder_stages, choice.skip_zeros))
    };
    // Dies that must be interchangeable all need the widest cell.
    let mixed = p.mixes_widths() && widest > p.weight_bits && p.die_mode == DieMode::Custom;
    let wide = if mixed { wide_share(model, p) } else { 1.0 };
    let Some(params) = choose(widest, wide) else {
        return Resolved { params: p.clone(), decisions: Vec::new(), narrow: None };
    };
    let narrow = if mixed { choose(p.weight_bits, 1.0 - wide).map(|q| (q, p.weight_bits, 1.0 - wide)) } else { None };
    Resolved { params, decisions, narrow }
}

/// Estimates area, speed and energy. A speed target decides the multipliers per die. On
/// an FPGA board, the board sets the clock and the external memory, and the report says
/// how much of the board the design takes.
pub fn estimate(model: &ModelGraph, p: &ChipParams) -> Result<Report, String> {
    let Some(board) = crate::fpga::target(p) else { return estimate_on_silicon(model, p) };
    let on_board = crate::fpga::on_board(p, board);
    let mut report = estimate_on_silicon(model, &on_board)?;
    crate::fpga::check(&mut report, &on_board, board);
    Ok(report)
}

fn estimate_on_silicon(model: &ModelGraph, p: &ChipParams) -> Result<Report, String> {
    if p.target_tok_s <= 0.0 {
        return estimate_design(model, p);
    }
    let mut q = p.clone();
    let mut report = estimate_design(model, &q)?;
    for _ in 0..8 {
        let rate = report.metric("tok_s_base").unwrap_or(p.target_tok_s).max(1e-9);
        let ratio = p.target_tok_s / rate;
        if (ratio - 1.0).abs() < 0.01 {
            break;
        }
        let mut next = q.clone();
        next.macs_per_die = ((q.macs_per_die as f64 * ratio).ceil() as u64).clamp(1, MAX_MACS_PER_DIE);
        let tried = estimate_design(model, &next)?;
        // More multipliers that give little more speed (links, vectors or memory set it)
        // are mostly silicon: they must buy at least a third of what they cost.
        let growth = next.macs_per_die as f64 / q.macs_per_die.max(1) as f64;
        let gain = tried.metric("tok_s_base").unwrap_or(0.0) / rate;
        if growth > 1.0 && gain - 1.0 < (growth - 1.0) / 3.0 {
            break;
        }
        (q, report) = (next, tried);
    }
    if report.metric("tok_s_base").unwrap_or(0.0) < p.target_tok_s * 0.97 {
        report.violations.push(Violation {
            severity: Severity::Aviso,
            message: "No se alcanza la velocidad objetivo solo con más multiplicadores.".into(),
            fix: "Sube los carriles vectoriales, la frecuencia o el ancho de banda de memoria.".into(),
            field: Some("target_tok_s"), value: Some(serde_json::json!((report.metric("tok_s_base").unwrap_or(0.0) / 100.0).floor() * 100.0)),
        });
    }
    Ok(report)
}

/// The parameters the circuit is really generated from.
pub fn resolved_params(model: &ModelGraph, p: &ChipParams) -> Result<ChipParams, String> {
    estimate(model, p).map(|report| report.resolved)
}

/// The largest number of multipliers a die can be asked to hold.
const MAX_MACS_PER_DIE: u64 = 64_000_000;

fn estimate_design(model: &ModelGraph, p: &ChipParams) -> Result<Report, String> {
    let tech = node(&p.node).ok_or_else(|| format!("nodo tecnológico desconocido: {}", p.node))?;
    let mut violations = Vec::new();
    check_optimizations(p, &mut violations);

    let included = model.params_where(|t| tensor_included(t, p)) as f64;
    let budget = (p.max_params_g * 1e9).max(1.0);
    let store = storage(model, p, budget);
    let totals = stage_totals(model, p);
    let cell_bits_estimate = store.chip_bits / p.macs_per_die.max(1) as f64 / (p.dies.max(10) as f64);
    let resolved = resolve_circuit(tech, model, p, store.widest_weight_bits, cell_bits_estimate);
    let p = &resolved.params;
    let mut mult = weight_multiplier(tech, p, store.widest_weight_bits, cell_bits_estimate);
    let widest_cell = CellConfig { weight_bits: store.widest_weight_bits, ..p.cell_config(1) };
    let mut broadcast_bits = widest_cell.broadcast_bits() as f64;
    if let Some((narrow, bits, share)) = &resolved.narrow {
        // Dies and regions holding only narrow tensors get narrower cells.
        let small = weight_multiplier(tech, narrow, *bits, cell_bits_estimate);
        mult.area_um2 = mult.area_um2 * (1.0 - share) + small.area_um2 * share;
        mult.energy_pj = mult.energy_pj * (1.0 - share) + small.energy_pj * share;
        let narrow_cell = CellConfig { weight_bits: *bits, ..narrow.cell_config(1) };
        broadcast_bits = broadcast_bits * (1.0 - share) + narrow_cell.broadcast_bits() as f64 * share;
    }
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
            field: Some("max_params_g"), value: Some(serde_json::json!((included / 1e9 * 100.0).ceil() / 100.0)),
        });
    }
    if model.max_context > 0 && p.context_tokens > model.max_context {
        violations.push(Violation {
            severity: Severity::Aviso,
            message: format!("El modelo declara {} tokens de contexto nativos; por encima puede perder calidad.", model.max_context),
            fix: "Reduce el contexto si la calidad cae.".into(),
            field: Some("context_tokens"), value: Some(serde_json::json!(model.max_context)),
        });
    }
    if p.mtp_enabled && model.params_in(Block::Mtp) == 0 {
        violations.push(Violation {
            severity: Severity::Aviso,
            message: "Este modelo no tiene cabeza MTP.".into(),
            fix: "Desactiva MTP.".into(),
            field: Some("mtp_enabled"), value: Some(serde_json::json!(false)),
        });
    }

    let kv_compression = if p.has("compresion_kv") { 0.5 } else { 1.0 };
    // A die meant for any model reserves state for the largest one it accepts.
    let (layer_room, width_room) = envelope_scale(model, p);
    let kv_bits = model.state.kv_values_per_token as f64 * layer_room * width_room * context * p.kv_bits as f64 * kv_compression;
    let state_bits = model.state.delta_values as f64 * layer_room * width_room * width_room * p.delta_state_bits as f64
        + model.state.conv_values as f64 * layer_room * width_room * p.activation_bits as f64;
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
        broadcast_bits,
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
            field: Some("ddr_capacity_gb"), value: Some(serde_json::json!((store.chip_bits / 8e9).ceil())),
        });
    }

    // On an FPGA board the board itself is the limit, which `fpga::check` applies.
    let die_limit = if crate::fpga::target(p).is_some() { f64::MAX } else { p.die_max_mm2.min(tech.die_max_mm2) };
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
            field: Some("dies"), value: Some(serde_json::json!(0)),
        });
    }
    let d = dies as f64;
    let n = p.macs_per_die as f64;
    let total_area = die_area * d;

    // Timing: each token visits every stage in order, so stage times add up.
    let cell = timed_cell(p, store.widest_weight_bits, cell_bits_estimate);
    // Static timing on the generated gates, already cut by its pipeline stages.
    let path = cell_path(cell).ok_or("el circuito de la celda no se puede simular")?;
    let logic_ns = path * tech.fa_delay_ns * mult.delay_factor;
    let fmax = match measured(p, &cell) {
        Some((_, ghz)) if ghz > 0.0 => ghz,
        _ => tech.fmax_ghz.min(1.0 / (logic_ns + tech.ff_overhead_ns)),
    }
    .min(sram_clock_ghz(tech, p));
    let freq = p.freq_ghz.min(fmax);
    if p.freq_ghz > fmax {
        violations.push(Violation {
            severity: Severity::Aviso,
            message: format!("Con {} etapas de multiplicador el máximo son {fmax:.2} GHz; se simula a esa frecuencia.", p.mac_stages),
            fix: "Añade etapas o baja la frecuencia.".into(),
            field: Some("freq_ghz"), value: Some(serde_json::json!((fmax * 100.0).floor() / 100.0)),
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
    let link = effective_link(p, tech);
    let plan = link_plan(link, dies, die_area);
    if p.link == LinkKind::Ucie && dies > 1 && total_area > PACKAGE_MAX_MM2 {
        violations.push(Violation {
            severity: Severity::Aviso,
            message: format!("{total_area:.0} mm² de silicio no caben en un encapsulado de {PACKAGE_MAX_MM2:.0} mm²."),
            fix: "Elige el enlace automático, que reparte los dies en varios encapsulados unidos por SerDes.".into(),
            field: Some("link"), value: Some(serde_json::json!("auto")),
        });
    }
    // Chips that share each layer, and stages in a chain: by layers each chip is a stage
    // of its own, by tensor all of them are one group.
    let group = p.tensor_chips(dies);
    let stages = dies.div_ceil(group).max(1);
    let g = group as f64;
    // A group splits every matrix and every head among its chips: each does its share
    // of the products and of the attention and delta rule.
    let (weight_cycles, vector_cycles) = (weight_cycles / g, vector_cycles / g);
    // Its chips add their partial results twice a layer, round a ring: 2·(g − 1) steps,
    // each moving a g-th of the hidden vector and paying a hop.
    let pins = link == LinkKind::Pins;
    let hop_cycles = if pins { PIN_HOP_NS } else { UCIE_HOP_NS } * freq;
    // Cycles to move this many activations: `link_words` a cycle, or what the pins carry.
    let moving = |activations: f64| if pins { activations * p.activation_bits as f64 / PIN_LINK_BITS_PER_NS * freq } else { activations / p.link_words.max(1) as f64 };
    let share = moving(model.hidden_size as f64 / g);
    let tensor_cycles = if group > 1 { model.num_layers as f64 * 2.0 * 2.0 * (g - 1.0) * (hop_cycles + share) } else { 0.0 };
    // Each stage passes on the hidden vector, `link_words` activations a cycle, after the
    // hops' own delay.
    let transfer = moving(model.hidden_size as f64);
    let chain_share = if dies > 1 { (stages - 1) as f64 / (dies - 1) as f64 } else { 0.0 };
    let link_cycles = if stages > 1 { plan.delay_ns() * freq * chain_share + (stages - 1) as f64 * transfer } else { 0.0 } + tensor_cycles;
    let total_cycles = weight_cycles + vector_cycles + latency_cycles + link_cycles;
    if group > 1 && plan.packages > 1 && group > dies.div_ceil(plan.packages) {
        violations.push(Violation {
            severity: Severity::Aviso,
            message: format!("Los grupos de {group} chips que comparten cada capa no caben en un encapsulado: sus sumas cruzarían la placa, mucho más lenta."),
            fix: "Usa grupos más pequeños (reparto mixto) o dies menores para que cada grupo quepa en un encapsulado.".into(),
            field: Some("tensor_group"), value: Some(serde_json::json!(dies.div_ceil(plan.packages).max(1))),
        });
    }
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
    if p.die_mode == DieMode::Identical {
        check_envelope(model, p, &mut violations);
    }
    let sampling = sampling_cycles(p, model);
    let token_time = (compute_time + sampling / (freq * 1e9)).max(memory_time);
    let stream_rate = 1.0 / token_time;
    // Reading a prompt, every stage of the chain works on a token of its own at once.
    let prefill_rate = (stages as f64 / compute_time).min(1.0 / memory_time.max(1e-12));

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
    let e_noc = totals.weights * broadcast_bits * tech.e_wire_pj_bit_mm * hop_mm;
    let e_scale = if p.uses_groups() {
        let width = p.effective_accumulator() as f64;
        let bits = if p.scale_is_multiplied() { p.scale_bits as f64 * width } else { width * width.log2().ceil() * 0.5 };
        totals.weights / p.group_size as f64 * bits * tech.e_fa_pj
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

    // Weights in ROM are part of the silicon; only what lives elsewhere still has to arrive.
    let image_bytes = if p.weight_backend == WeightBackend::Rom { store.external_bits / 8.0 } else { (store.model_bits + store.external_bits) / 8.0 };
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
            "tokenizer_tables" => (0.0, 0.0, 0.0, format!("{:.0} Mbit para un vocabulario de {} tokens.", model.vocab_size as f64 * TOKENIZER_BITS_PER_TOKEN / 1e6, model.vocab_size)),
            "cpu_memory" => (0.0, 0.0, 0.0, format!("Para prompts de hasta {} tokens.", p.context_tokens)),
            "sampler" => (0.0, 0.0, 0.0, format!("{} carriles de filtro y {} candidatos.", p.sampler_lanes, p.sampler_entries)),
            "links" => (link_cycles / total_cycles, 0.0, link_bytes, if pins { format!("{} enlaces por pines de E/S, uno por encapsulado.", plan.pin_hops) } else { format!("{} enlaces UCIe y {} SerDes, en {} encapsulados.", plan.ucie_hops, plan.serdes_hops, plan.packages) }),
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
        if p.weight_backend == WeightBackend::Rom { "Los pesos vienen grabados en el chip y no se cargan.".into() } else { "Tamaño de la imagen dividido por la interfaz de carga.".into() });

    m("dies", "Silicio", "Dies", d, "", Supuesta, "El mínimo que cumple el área máxima por die.".into());
    m("packages", "Silicio", "Encapsulados", plan.packages as f64, "", Supuesta, format!("Hasta {PACKAGE_MAX_MM2:.0} mm² de silicio en cada uno."));
    m("die_area_mm2", "Silicio", "Área por die", die_area, "mm²", Supuesta, "Suma de los bloques del die.".into());
    m("total_area_mm2", "Silicio", "Área total", total_area, "mm²", Supuesta, "Área por die por número de dies.".into());
    m("macs_total", "Silicio", "Multiplicadores de pesos en total", n * d, "", Derivada,
        "Con un solo usuario solo trabajan los de un die cada vez.".into());
    m("cell_sram_kb", "Silicio", "Memoria de pesos por celda", cell_bytes / 1e3, "KB", Derivada,
        "Capacidad total repartida entre todas las celdas.".into());

    m("tok_s", "Velocidad", "Generación con un usuario", effective_rate, "tok/s", Supuesta,
        if depth > 0 { format!("Incluye MTP: {tokens_per_round:.2} tokens por ronda con las tasas de aceptación medidas por MTPLX.") }
        else { "Un token recorre todas las etapas antes de empezar el siguiente.".into() });
    m("sampling_cycles", "Velocidad", "Muestreo por token", sampling, "ciclos", Derivada,
        "Filtra los logits mientras salen de lm_head; solo cuenta lo que no se solapa con el cálculo.".into());
    m("tok_s_base", "Velocidad", "Sin especulación", stream_rate, "tok/s", Supuesta,
        "Inverso del tiempo de un token.".into());
    m("prefill_tok_s", "Velocidad", "Lectura de prompt", prefill_rate, "tok/s", Supuesta,
        "Los tokens ya se conocen, así que todos los dies trabajan a la vez.".into());
    if let (true, Some(vision)) = (p.include_vision, model.vision) {
        // The encoder runs on one die's cells, before the prompt's text.
        let macs = vision.macs_per_image(512, 512) as f64;
        m("image_ms", "Velocidad", "Imagen de 512×512", macs / (p.macs_per_die.max(1) as f64 * freq * 1e9) * 1e3, "ms", Supuesta,
            format!("{:.1} G multiplicaciones del codificador por imagen, en los multiplicadores de un die.", macs / 1e9));
    }
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

    Ok(Report { model: model.name.clone(), dies, die_area_mm2: die_area, total_area_mm2: total_area, blocks, metrics, violations, floorplan, decisions: resolved.decisions.clone(), macs_per_die: p.macs_per_die, resolved: p.clone() })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::ChipSplit;
    use std::time::Instant;

    const QWEN: &str = include_str!("../../../models/qwen3.8-27b.json");

    fn qwen() -> ModelGraph {
        ModelGraph::from_config_json("qwen", QWEN).unwrap()
    }

    #[test]
    fn sharing_each_layer_among_the_chips_makes_one_user_faster() {
        let model = qwen();
        let speed = |chip_split, tensor_group| {
            let p = ChipParams { dies: 8, link: LinkKind::Ucie, link_words: 128, chip_split, tensor_group, ..Default::default() };
            let report = estimate(&model, &p).unwrap();
            (report.metric("tok_s_base").unwrap(), report.metric("prefill_tok_s").unwrap())
        };
        let (layers, layers_prompt) = speed(ChipSplit::Layers, 1);
        let (mixed, _) = speed(ChipSplit::Mixed, 4);
        let (tensor, tensor_prompt) = speed(ChipSplit::Tensor, 1);
        assert!(layers < mixed && mixed < tensor, "{layers} {mixed} {tensor}");
        assert!(tensor > 3.0 * layers, "{layers} {tensor}");
        // A chain reads a prompt with every stage busy; a single group has one stage.
        assert!(layers_prompt / layers > tensor_prompt / tensor);
        assert_eq!(speed(ChipSplit::Mixed, 1).0, layers);
    }

    #[test]
    fn the_generator_adapts_the_circuit_to_each_set_of_parameters() {
        let model = qwen();
        let cell = |p: &ChipParams| estimate(&model, p).unwrap().decisions;
        let slow = cell(&ChipParams { freq_ghz: 0.5, use_model_mix: false, weight_bits: 4, ..Default::default() });
        let fast = cell(&ChipParams { freq_ghz: 2.5, use_model_mix: false, weight_bits: 16, ..Default::default() });
        let (a, b) = (slow[0].clone(), fast[0].clone());
        assert!(a.multiplier != b.multiplier || a.mac_stages != b.mac_stages || a.adder_stages != b.adder_stages, "{a:?} {b:?}");
        assert!(fast[0].fmax_ghz >= 2.0, "{:?}", fast[0]);

        // Each width of a mixed checkpoint gets its own cell, and never a worse one.
        let mixed = cell(&ChipParams::default());
        assert_eq!(mixed.len(), 2);
        assert!(mixed[0].weight_bits > mixed[1].weight_bits);
        assert!(mixed[1].area_um2 < mixed[0].area_um2);
        assert!((mixed[0].share + mixed[1].share - 1.0).abs() < 1e-9);
    }

    #[test]
    fn the_automatic_circuit_is_at_least_as_good_as_any_fixed_choice() {
        let model = qwen();
        let cost = |p: &ChipParams| {
            let report = estimate(&model, p).unwrap();
            let mac = report.blocks.iter().find(|b| b.id == "cell_macs").unwrap().per_die_mm2;
            (mac, report.metric("freq_ghz").unwrap())
        };
        let automatic = ChipParams { circuit_goal: CircuitGoal::Area, use_model_mix: false, weight_bits: 8, ..Default::default() };
        let (best, frequency) = cost(&automatic);
        for multiplier_kind in [MultiplierKind::Array, MultiplierKind::Wallace, MultiplierKind::Booth] {
            for mac_stages in 1..=3 {
                let (area, reached) = cost(&ChipParams { auto_circuit: false, multiplier_kind, mac_stages, adder_stages: 1, ..automatic.clone() });
                assert!(best <= area * 1.0001 || reached < frequency, "{multiplier_kind:?} × {mac_stages}: {area} frente a {best}");
            }
        }
    }

    #[test]
    fn a_speed_target_sizes_the_multipliers_for_the_model() {
        let model = qwen();
        let small = estimate(&model, &ChipParams { target_tok_s: 2000.0, ..Default::default() }).unwrap();
        let fast = estimate(&model, &ChipParams { target_tok_s: 4000.0, ..Default::default() }).unwrap();
        for (report, target) in [(&small, 2000.0), (&fast, 4000.0)] {
            assert!((report.metric("tok_s_base").unwrap() / target - 1.0).abs() < 0.02);
        }
        assert!(fast.macs_per_die > small.macs_per_die * 3 / 2);
    }

    #[test]
    fn identical_dies_cost_more_than_custom_ones_and_check_their_limits() {
        let model = qwen();
        let custom = estimate(&model, &ChipParams::default()).unwrap();
        let repeated = ChipParams { die_mode: DieMode::Identical, ..Default::default() };
        let identical = estimate(&model, &repeated).unwrap();
        assert_eq!(identical.decisions.len(), 1, "todas las celdas iguales");
        assert!(identical.total_area_mm2 > custom.total_area_mm2);
        let errors = |p: &ChipParams| estimate(&model, p).unwrap().violations.iter().filter(|v| v.severity == Severity::Error).count();
        assert_eq!(errors(&repeated), 0);
        assert_eq!(errors(&ChipParams { max_layers: 32, ..repeated.clone() }), 1);
        assert_eq!(errors(&ChipParams { weight_backend: WeightBackend::Rom, ..repeated }), 1);
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
        let metric = |report: &Report, id: &str| report.metrics.iter().find(|m| m.id == id).unwrap().value;
        assert!(metric(&rom, "load_seconds") < metric(&sram, "load_seconds") / 100.0, "los pesos grabados no se cargan");
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
        for option in catalog() {
            // Hardwired weights are the only place this optimization applies.
            let start = ChipParams { weight_backend: if option.id == "pesos_csd" { WeightBackend::Rom } else { WeightBackend::Sram }, ..Default::default() };
            let base_report = estimate(&qwen(), &start).unwrap();
            let base = serde_json::to_string(&base_report.metrics).unwrap();
            let mut p = start.clone();
            p.optimizations.insert(option.id.to_string());
            let report = estimate(&qwen(), &p).unwrap();
            let blocks = report.blocks.iter().map(|b| b.per_die_mm2).sum::<f64>();
            let default_blocks = base_report.die_area_mm2;
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
    fn a_measured_cell_replaces_the_estimate_only_for_that_cell_and_process() {
        let stories = ModelGraph::from_config_json("s", include_str!("../../../models/stories260k.json")).unwrap();
        let p = ChipParams { node: "sky130".into(), max_params_g: 0.00026, use_model_mix: false, group_size: 0, weight_bits: 8, context_tokens: 256, ..Default::default() };
        let estimated = estimate(&stories, &p).unwrap().decisions[0].clone();
        let key = format!("sky130:{}", estimated.cell_key);
        // Anything that changes the gates changes the name, error correction included; the
        // memory's depth does not, once it has the same correction.
        let deep = CellConfig { addr_bits: 12, ecc: true, ..estimated.cell };
        let other = |change: &dyn Fn(&mut CellConfig)| {
            let mut cell = deep;
            change(&mut cell);
            cell_key(&cell)
        };
        assert_eq!(other(&|c| c.addr_bits += 3), cell_key(&deep));
        for change in [&(|c: &mut CellConfig| c.skip_zeros = !c.skip_zeros) as &dyn Fn(&mut CellConfig), &|c| c.ecc = false, &|c| c.addr_bits = 1, &|c| c.mac_stages += 1, &|c| c.accumulator_bits += 40] {
            assert_ne!(other(change), cell_key(&deep));
        }
        let measured = ChipParams { measured_cell: key, measured_cell_um2: 100.0, measured_cell_fmax_ghz: 0.5, ..p.clone() };
        // The measured cell is smaller and faster than any estimate, so the generator keeps it.
        let laid_out = estimate(&stories, &measured).unwrap().decisions[0].clone();
        assert_eq!((laid_out.multiplier, laid_out.mac_stages, laid_out.adder_stages), (estimated.multiplier, estimated.mac_stages, estimated.adder_stages));
        assert!(laid_out.area_um2 < estimated.area_um2, "{laid_out:?} {estimated:?}");
        // The measured clock, or the SRAM macros' if they are slower.
        let expected = 0.5f64.min(1.0 / node("sky130").unwrap().sram_min_period_ns.unwrap());
        assert!((laid_out.fmax_ghz - expected).abs() < 1e-9, "{laid_out:?}");
        // On another process it is not what was measured.
        let elsewhere = estimate(&stories, &ChipParams { node: "gf180".into(), ..measured }).unwrap().decisions[0].clone();
        assert!((elsewhere.fmax_ghz - expected).abs() > 1e-9, "{elsewhere:?}");
    }

    #[test]
    fn chips_from_an_open_shuttle_talk_over_their_pins() {
        let stories = ModelGraph::from_config_json("s", include_str!("../../../models/stories260k.json")).unwrap();
        let p = ChipParams { node: "sky130".into(), max_params_g: 0.0003, macs_per_die: 16, vector_lanes_per_die: 4, use_model_mix: false, group_size: 0, weight_bits: 8, context_tokens: 256, dies: 8, ..Default::default() };
        let pins = estimate(&stories, &p).unwrap();
        let serdes = estimate(&stories, &ChipParams { link: LinkKind::Serdes, ..p.clone() }).unwrap();
        assert!(pins.metric("tok_s").unwrap() < serdes.metric("tok_s").unwrap(), "{} {}", pins.metric("tok_s").unwrap(), serdes.metric("tok_s").unwrap());
        // A speed the links cannot give does not buy multipliers that would only wait.
        let hopeless = estimate(&stories, &ChipParams { target_tok_s: 1e6, ..p.clone() }).unwrap();
        assert!(hopeless.resolved.macs_per_die < 100_000, "{}", hopeless.resolved.macs_per_die);
        // A process a foundry makes in advanced packages keeps its UCIe.
        assert_eq!(effective_link(&ChipParams { node: "n6".into(), ..p }, node("n6").unwrap()), LinkKind::Auto);
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
    fn the_firmware_fits_the_program_memory_the_estimate_counts() {
        let elf = include_bytes!("../../../firmware/prebuilt/firmware.elf");
        let word = |at: usize| u32::from_le_bytes(elf[at..at + 4].try_into().unwrap()) as usize;
        let entries = u16::from_le_bytes(elf[44..46].try_into().unwrap()) as usize;
        let at_address_zero = (0..entries).map(|i| word(28) + 32 * i).find(|at| word(*at) == 1 && word(*at + 8) == 0).unwrap();
        assert!(word(at_address_zero + 16) <= FIRMWARE_PROGRAM_BYTES as usize);
    }

    #[test]
    fn links_follow_the_packages_the_area_needs() {
        assert_eq!(link_plan(LinkKind::Auto, 4, 800.0), LinkPlan { packages: 1, ucie_hops: 4, serdes_hops: 0, pin_hops: 0 });
        assert_eq!(link_plan(LinkKind::Auto, 12, 800.0), LinkPlan { packages: 3, ucie_hops: 9, serdes_hops: 3, pin_hops: 0 });
        assert_eq!(link_plan(LinkKind::Ucie, 12, 800.0).serdes_hops, 0);
    }

    #[test]
    fn working_memory_follows_the_context() {
        let area = |context| {
            let report = estimate(&qwen(), &ChipParams { context_tokens: context, ..Default::default() }).unwrap();
            report.blocks.iter().find(|b| b.id == "cpu_memory").unwrap().per_die_mm2
        };
        assert!(area(32768) > 3.0 * area(8192));
    }

    #[test]
    fn estimate_is_fast_enough_for_live_sliders() {
        let model = qwen();
        let start = Instant::now();
        for _ in 0..100 {
            estimate(&model, &ChipParams::default()).unwrap();
        }
        // Debug build, about six times slower than the release build the app uses.
        assert!(start.elapsed().as_millis() < 6000, "{:?}", start.elapsed());
    }
}
