//! Whether the design can be built as it stands, before anyone lays it out: what stops it,
//! with the numbers that matter, and which changes of the parameters make it buildable.
//! Every change offered has been estimated again and passes the same check.

use crate::estimate::{estimate, link_plan, Report, Severity};
use crate::params::{ChipParams, DieMode, KvBackend, WeightBackend};
use crate::tech::{node, TechNode};
use model_ir::ModelGraph;
use serde::Serialize;

/// The verdict on a design, for a chip of its own.
#[derive(Clone, Debug, Serialize)]
pub struct Feasibility {
    /// The design fits its process with the chips it allows.
    pub possible: bool,
    /// It can also be laid out with OpenROAD: the process has an open PDK and every memory
    /// it uses exists there.
    pub layout: bool,
    /// What stops it, most important first.
    pub reasons: Vec<String>,
    /// What is worth knowing though it stops nothing.
    pub notes: Vec<String>,
    pub dies: u32,
    pub die_area_mm2: f64,
    pub die_limit_mm2: f64,
    /// The largest blocks of each die, to see where the area goes.
    pub largest: Vec<(String, f64)>,
    pub tok_s: f64,
}

/// One change of a parameter.
#[derive(Clone, Debug, Serialize)]
pub struct Change {
    pub field: &'static str,
    pub value: serde_json::Value,
    pub label: String,
    /// What it may cost besides area, when it is more than speed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub risk: Option<String>,
}

/// A set of changes that makes the design buildable, and what it gives then.
#[derive(Clone, Debug, Serialize)]
pub struct Suggestion {
    pub changes: Vec<Change>,
    pub dies: u32,
    pub die_area_mm2: f64,
    pub tok_s: f64,
    pub layout: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct Advice {
    pub verdict: Feasibility,
    pub suggestions: Vec<Suggestion>,
}

fn memory_name(backend: WeightBackend) -> Option<&'static str> {
    match backend {
        WeightBackend::Sram => None,
        WeightBackend::Rom => Some("ROM"),
        WeightBackend::StackedDram => Some("DRAM apilada"),
        WeightBackend::Hbm4 => Some("HBM4"),
        WeightBackend::ExternalDdr => Some("memoria externa DDR"),
    }
}

/// Packages a board takes, with their power and cooling: about what an eight- or
/// sixteen-accelerator server carries.
pub const BOARD_PACKAGES: u32 = 16;
/// Small chips of a few milliwatts in plastic packages: many more fit a board.
pub const BOARD_SMALL_PACKAGES: u32 = 64;

/// Area as a person reads it: whole square millimetres once there are many.
fn mm2(area: f64) -> String {
    if area >= 100.0 { format!("{area:.0}") } else { format!("{area:.1}") }
}

/// What the number of chips means physically: packages, boards, and the time each token
/// spends crossing between them.
fn chip_notes(p: &ChipParams, report: &Report, tech: Option<&TechNode>) -> Vec<String> {
    let mut notes = Vec::new();
    if report.dies <= 1 {
        return notes;
    }
    let plan = link_plan(p.link, report.dies, report.die_area_mm2);
    let crossing = plan.delay_ns() / 1000.0;
    let shuttle = tech.is_some_and(|t| t.open_pdk && t.fabricable);
    if shuttle {
        // An open shuttle hands each die back in its own package; they meet on a board.
        let per_board = if report.die_area_mm2 > 100.0 { BOARD_PACKAGES } else { BOARD_SMALL_PACKAGES };
        let boards = report.dies.div_ceil(per_board);
        let where_ = if boards > 1 { format!("{boards} placas unidas por cable") } else { "una placa".into() };
        notes.push(format!(
            "Una lanzadera abierta entrega cada die en su propio encapsulado: {} chips sobre {where_}, unidos por sus pines de E/S, más lentos que los enlaces que cuenta el estimador.",
            report.dies
        ));
    } else if plan.packages > BOARD_PACKAGES {
        let boards = plan.packages.div_ceil(BOARD_PACKAGES);
        notes.push(format!(
            "{} chips en {} encapsulados no caben en una placa (unos {BOARD_PACKAGES} encapsulados por placa): harían falta {boards} placas unidas por cable. Funciona, pero cada token cruza todos los enlaces: {crossing:.1} µs por token solo en ellos.",
            report.dies, plan.packages
        ));
    } else if plan.packages > 1 {
        notes.push(format!("{} chips en {} encapsulados sobre una placa, unidos por SerDes: {crossing:.1} µs por token en enlaces.", report.dies, plan.packages));
    } else {
        notes.push(format!("{} chips en un solo encapsulado, unidos por UCIe.", report.dies));
    }
    if p.die_mode == DieMode::Custom && tech.is_some_and(|t| t.open_pdk && t.fabricable) {
        notes.push(format!("Cada uno de los {} dies es un diseño distinto: en una lanzadera abierta ocupa una plaza propia. Con «Todos iguales» basta un diseño fabricado {} veces.", report.dies, report.dies));
    }
    notes
}

/// The verdict on these parameters, from their estimate.
pub fn judge(p: &ChipParams, report: &Report) -> Feasibility {
    let tech = node(&p.node);
    let limit = tech.map_or(p.die_max_mm2, |t| p.die_max_mm2.min(t.die_max_mm2));
    let mut reasons = Vec::new();
    let mut blocks: Vec<(String, f64)> = report.blocks.iter().filter(|b| b.per_die_mm2 > 0.0).map(|b| (b.label.to_string(), b.per_die_mm2)).collect();
    blocks.sort_by(|a, b| b.1.total_cmp(&a.1));
    blocks.truncate(4);
    if report.die_area_mm2 > limit {
        let list = blocks.iter().take(3).map(|(label, area)| format!("{label} {} mm²", mm2(*area))).collect::<Vec<_>>().join(", ");
        let chips = if report.dies > 1 { format!(" aun repartido en {} chips", report.dies) } else { String::new() };
        reasons.push(format!("Cada die necesita {} mm²{chips} y el proceso admite {limit:.0} mm²: {list}.", mm2(report.die_area_mm2)));
    }
    for v in report.violations.iter().filter(|v| v.severity == Severity::Error && !v.message.starts_with("Cada die ocupa")) {
        reasons.push(v.message.clone());
    }
    let possible = reasons.is_empty();
    let open = tech.is_some_and(|t| t.open_pdk);
    let mut layout_reasons = Vec::new();
    if !open {
        layout_reasons.push(format!("{} no tiene un PDK abierto: se estima, pero no se puede hacer su layout. Para eso, SkyWater 130, GF180 o ASAP7.", tech.map_or(p.node.as_str(), |t| t.label.as_str())));
    } else {
        if let Some(memory) = memory_name(p.weight_backend) {
            layout_reasons.push(format!("Los pesos van en {memory}, que el PDK abierto no ofrece: solo tiene macros de SRAM."));
        }
        if p.kv_backend != KvBackend::Sram {
            layout_reasons.push("La caché KV va fuera del chip, y el PDK abierto no tiene la interfaz para esa memoria.".into());
        }
    }
    let layout = possible && layout_reasons.is_empty();
    reasons.extend(layout_reasons);
    // How the chips would be put together matters once they can be made.
    let mut notes = if reasons.is_empty() { chip_notes(p, report, tech) } else { Vec::new() };
    if tech.is_some_and(|t| !t.fabricable) {
        notes.push(format!("{} es un proceso predictivo: su layout enseña cómo sería, pero ninguna fábrica lo produce.", tech.map_or("", |t| t.label.as_str())));
    }
    Feasibility {
        possible,
        layout,
        reasons,
        notes,
        dies: report.dies,
        die_area_mm2: report.die_area_mm2,
        die_limit_mm2: limit,
        largest: blocks,
        tok_s: report.metric("tok_s").unwrap_or(0.0),
    }
}

/// The changes worth trying from these parameters, each one lighter than what there is.
fn levers(model: &ModelGraph, p: &ChipParams, tech: &TechNode) -> Vec<Change> {
    let mut out = Vec::new();
    let mut add = |field: &'static str, value: serde_json::Value, label: String, risk: Option<&str>| {
        out.push(Change { field, value, label, risk: risk.map(str::to_string) });
    };
    if p.dies != 0 {
        add("dies", 0.into(), "Repartir el modelo en los chips que hagan falta".into(), None);
    }
    // A design sized for a bigger model carries room this one never uses.
    let params_g = model.total_params() as f64 / 1e9;
    if p.max_params_g > params_g * 1.1 {
        let room = (params_g * 1.05 * 1e4).ceil() / 1e4;
        add("max_params_g", room.into(), format!("Dimensionar para este modelo, {} G parámetros", room), None);
    }
    if model.max_context > 0 && p.context_tokens > model.max_context {
        add("context_tokens", model.max_context.into(), format!("El contexto nativo del modelo, {} tokens", model.max_context), None);
    }
    if p.die_max_mm2 < tech.die_max_mm2 {
        add("die_max_mm2", tech.die_max_mm2.into(), format!("Usar todo el die que admite el proceso, {:.0} mm²", tech.die_max_mm2), None);
    }
    // A mixed checkpoint keeps some matrices wider: all of them at one width is lighter.
    let mixed = p.use_model_mix && model.tensors.iter().any(|t| t.wide);
    for bits in [6, 4, 3] {
        if bits < p.weight_bits || (mixed && bits <= p.weight_bits) {
            let risk = if bits >= 6 { "calidad: pérdida pequeña" } else if bits == 4 { "calidad: pérdida apreciable sin cuantización cuidada" } else { "calidad: pérdida grande" };
            add("weight_bits", bits.into(), format!("Pesos de {bits} bits"), Some(risk));
        }
    }
    for divisor in [2, 4, 8] {
        let context = p.context_tokens / divisor;
        if context >= 128 {
            add("context_tokens", context.into(), format!("Contexto de {context} tokens"), Some("conversaciones más cortas"));
        }
    }
    if p.kv_bits > 8 {
        add("kv_bits", 8.into(), "Caché KV de 8 bits".into(), Some("calidad: pérdida pequeña en contextos largos"));
    }
    if p.function_units_per_die > 1 {
        add("function_units_per_die", 1.into(), "Una sola unidad de función por die".into(), None);
    }
    if p.function_table_bits > 8 {
        let bits = (p.function_table_bits - 2).max(8);
        add("function_table_bits", bits.into(), format!("Tablas de función de {bits} bits de entrada"), Some("precisión de SiLU, exp y raíz algo menor"));
    }
    // Fewer units per die, down to what a small model needs: each a quarter of the last.
    for divisor in [2, 8, 32, 128, 512, 2048] {
        let macs = p.macs_per_die / divisor;
        if macs >= 16 {
            add("macs_per_die", macs.into(), format!("{macs} multiplicadores por die"), Some("menos tok/s"));
        }
        let lanes = p.vector_lanes_per_die / divisor;
        if lanes >= 4 {
            add("vector_lanes_per_die", lanes.into(), format!("{lanes} carriles vectoriales por die"), Some("menos tok/s"));
        }
    }
    if tech.open_pdk {
        if p.weight_backend != WeightBackend::Sram {
            add("weight_backend", "sram".into(), "Pesos en SRAM, la memoria del PDK abierto".into(), None);
        }
        if p.kv_backend != KvBackend::Sram {
            add("kv_backend", "sram".into(), "Caché KV en SRAM dentro del chip".into(), None);
        }
    }
    for other in crate::tech::nodes().iter().filter(|n| n.open_pdk && n.fabricable && n.id != tech.id) {
        add("node", other.id.clone().into(), format!("Proceso {}", other.label), None);
    }
    out
}

fn apply(p: &ChipParams, changes: &[Change]) -> Option<ChipParams> {
    let mut value = serde_json::to_value(p).ok()?;
    for change in changes {
        value[change.field] = change.value.clone();
        if change.field == "weight_bits" {
            value["use_model_mix"] = false.into();
        }
    }
    serde_json::from_value(value).ok()
}

/// How far a design is from buildable: what still stops it, then the excess of area.
fn distance(f: &Feasibility) -> (usize, f64) {
    let oversized = f.die_area_mm2 > f.die_limit_mm2;
    (f.reasons.len() - usize::from(oversized), (f.die_area_mm2 / f.die_limit_mm2).max(1.0))
}

/// The verdict, and when the design cannot be built or laid out and `search` asks for it,
/// the changes that make it so.
pub fn advise(model: &ModelGraph, p: &ChipParams, search: bool) -> Result<Advice, String> {
    let report = estimate(model, p)?;
    let verdict = judge(p, &report);
    let Some(tech) = node(&p.node) else { return Ok(Advice { verdict, suggestions: Vec::new() }) };
    if !search || verdict.layout || (verdict.possible && !tech.open_pdk) {
        return Ok(Advice { verdict, suggestions: Vec::new() });
    }
    let goal = |f: &Feasibility| if tech.open_pdk { f.layout } else { f.possible };
    let try_changes = |changes: &[Change]| -> Option<(Feasibility, Suggestion)> {
        let q = apply(p, changes)?;
        let report = estimate(model, &q).ok()?;
        let f = judge(&q, &report);
        let s = Suggestion { changes: changes.to_vec(), dies: f.dies, die_area_mm2: f.die_area_mm2, tok_s: f.tok_s, layout: f.layout };
        Some((f, s))
    };
    let options = levers(model, p, tech);

    // One change alone, if any does it: the fastest first.
    let mut suggestions: Vec<Suggestion> = options.iter().filter_map(|c| try_changes(std::slice::from_ref(c))).filter(|(f, _)| goal(f)).map(|(_, s)| s).collect();
    suggestions.sort_by(|a, b| b.tok_s.total_cmp(&a.tok_s));
    suggestions.truncate(5);
    if !suggestions.is_empty() {
        return Ok(Advice { verdict, suggestions });
    }

    // Otherwise a few changes together: each step takes the one that brings it closest.
    let mut chosen: Vec<Change> = Vec::new();
    for _ in 0..8 {
        let best = options
            .iter()
            .filter(|c| chosen.iter().all(|d| d.field != c.field))
            .filter_map(|c| {
                let mut changes = chosen.clone();
                changes.push(c.clone());
                try_changes(&changes).map(|(f, s)| (distance(&f), goal(&f), s))
            })
            .min_by(|a, b| b.1.cmp(&a.1).then(a.0 .0.cmp(&b.0 .0)).then(a.0 .1.total_cmp(&b.0 .1)).then(b.2.tok_s.total_cmp(&a.2.tok_s)));
        let Some((_, reached, s)) = best else { break };
        chosen = s.changes.clone();
        if reached {
            suggestions.push(s);
            break;
        }
    }
    // Then, accepting more changes, the fewest chips it can take.
    if let Some(first) = suggestions.first().cloned() {
        let mut fewest = first.clone();
        for _ in 0..6 {
            let better = options
                .iter()
                .filter(|c| fewest.changes.iter().all(|d| d.field != c.field))
                // Not at any price: a large loss of quality is a choice for the person.
                .filter(|c| !c.risk.as_deref().is_some_and(|r| r.contains("grande")))
                .filter_map(|c| {
                    let mut changes = fewest.changes.clone();
                    changes.push(c.clone());
                    try_changes(&changes).filter(|(f, _)| goal(f)).map(|(_, s)| s)
                })
                .filter(|s| s.dies < fewest.dies)
                .min_by(|a, b| a.dies.cmp(&b.dies).then(b.tok_s.total_cmp(&a.tok_s)));
            let Some(s) = better else { break };
            fewest = s;
        }
        if fewest.dies < first.dies {
            suggestions.push(fewest);
        }
    }
    Ok(Advice { verdict, suggestions })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stories() -> ModelGraph {
        ModelGraph::from_config_json("s", include_str!("../../../models/stories260k.json")).unwrap()
    }

    fn small(node: &str) -> ChipParams {
        ChipParams { node: node.into(), max_params_g: 0.0003, macs_per_die: 16, vector_lanes_per_die: 4, use_model_mix: false, group_size: 0, weight_bits: 8, context_tokens: 256, dies: 1, die_max_mm2: 858.0, ..Default::default() }
    }

    #[test]
    fn a_small_model_fits_a_modern_process_but_not_an_open_shuttle() {
        let model = stories();
        let n6 = advise(&model, &small("n6"), true).unwrap();
        assert!(n6.verdict.possible && !n6.verdict.layout, "{:?}", n6.verdict);
        let sky = advise(&model, &small("sky130"), true).unwrap();
        assert!(!sky.verdict.possible, "{:?}", sky.verdict);
        assert!(sky.verdict.reasons[0].contains("10 mm²"), "{:?}", sky.verdict.reasons);
    }

    #[test]
    fn many_chips_say_how_many_packages_and_boards_they_take() {
        let model = ModelGraph::from_config_json("qwen", include_str!("../../../models/qwen3.8-27b.json")).unwrap();
        let p = ChipParams { dies: 0, ..Default::default() };
        let report = estimate(&model, &p).unwrap();
        let notes = chip_notes(&p, &report, node(&p.node));
        assert!(report.dies > 1 && notes[0].contains("encapsulado"), "{} {notes:?}", report.dies);
        // Each package of 32 small dies on its own: far more than a board takes.
        let crowded = Report { dies: 40 * BOARD_PACKAGES, die_area_mm2: 800.0, ..report };
        assert!(chip_notes(&ChipParams { link: crate::params::LinkKind::Auto, ..p }, &crowded, None)[0].contains("placas"));
    }

    #[test]
    fn every_suggestion_makes_the_design_buildable() {
        let model = stories();
        for p in [small("sky130"), small("gf180"), ChipParams { weight_backend: WeightBackend::Hbm4, ..small("asap7") }, ChipParams { die_max_mm2: 1.0, ..small("n6") }] {
            let advice = advise(&model, &p, true).unwrap();
            for s in &advice.suggestions {
                let q = apply(&p, &s.changes).unwrap();
                let f = judge(&q, &estimate(&model, &q).unwrap());
                let goal = if node(&p.node).unwrap().open_pdk { f.layout } else { f.possible };
                assert!(goal, "{}: {:?} → {:?}", p.node, s.changes, f.reasons);
            }
        }
    }

    #[test]
    fn the_whole_qwen_on_an_open_shuttle_is_impossible_and_said_so() {
        let model = ModelGraph::from_config_json("qwen", include_str!("../../../models/qwen3.8-27b.json")).unwrap();
        let verdict = advise(&model, &ChipParams { node: "sky130".into(), dies: 1, ..Default::default() }, false).unwrap().verdict;
        assert!(!verdict.possible && !verdict.layout);
        assert!(!verdict.reasons.is_empty() && !verdict.largest.is_empty());
    }
}
