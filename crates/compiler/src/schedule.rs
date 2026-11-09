//! Places the model's stages on dies and assigns each one its clock cycles.

use arch::{ChipParams, Floorplan};
use model_ir::{Block, ModelGraph, StageKind};
use serde::Serialize;

#[derive(Clone, Debug, Serialize)]
pub struct ScheduledStage {
    pub name: String,
    pub layer: Option<u32>,
    pub die: u32,
    pub kind: &'static str,
    pub start: f64,
    pub cycles: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct Schedule {
    pub stages: Vec<ScheduledStage>,
    pub total_cycles: f64,
    pub token_ns: f64,
}

fn die_of(plan: &Floorplan, layer: Option<u32>, block: Block) -> u32 {
    match layer {
        Some(l) => plan
            .dies
            .iter()
            .find(|d| d.first_layer.is_some_and(|f| f <= l) && d.last_layer.is_some_and(|e| l <= e))
            .map_or(0, |d| d.index),
        None if block == Block::Embedding => 0,
        None => plan.dies.len() as u32 - 1,
    }
}

/// The cycle-by-cycle plan for one token, using the same cost model as the estimator.
pub fn schedule(model: &ModelGraph, p: &ChipParams, plan: &Floorplan, effective_ghz: f64, cycle_factor: f64) -> Schedule {
    let n = p.macs_per_die.max(1) as f64;
    let lanes = p.vector_lanes_per_die.max(1) as f64;
    let attention = if p.has("atencion_dispersa") { 0.25 } else { 1.0 };
    let tree = n.log2().ceil().max(1.0);
    let matvec_latency = (p.mac_stages + p.adder_stages) as f64 + 2.0 * tree * p.noc_stages.max(1) as f64;
    let group = if p.group_size > 0 { 1.0 + 1.0 / p.group_size as f64 } else { 1.0 };
    // A hop's cost is the ring's average, since only some of them leave the package.
    let dies = plan.dies.len().max(1) as u32;
    let die_area = plan.dies.first().map_or(1.0, |d| d.rect.w * d.rect.h);
    let hop_ns = arch::link_plan(p.link, dies, die_area).delay_ns() / dies as f64;
    let hop_cycles = hop_ns * effective_ghz;

    let mut stages = Vec::new();
    let mut clock = 0.0;
    let mut previous = 0;
    for stage in model.stages.iter().filter(|s| s.block != Block::Mtp) {
        let die = die_of(plan, stage.layer, stage.block);
        if die != previous {
            stages.push(ScheduledStage { name: "Enlace entre dies".into(), layer: stage.layer, die, kind: "link", start: clock, cycles: hop_cycles });
            clock += hop_cycles;
            previous = die;
        }
        let (kind, cycles) = match stage.kind {
            StageKind::Matvec { weights } => ("matvec", weights as f64 / n * group * cycle_factor + matvec_latency),
            StageKind::Vector { ops, ops_per_ctx } => {
                ("vector", (ops as f64 + ops_per_ctx as f64 * p.context_tokens as f64 * attention) / lanes + 6.0)
            }
            StageKind::EmbeddingRead { .. } => ("embedding", 1.0),
        };
        stages.push(ScheduledStage { name: stage.name.clone(), layer: stage.layer, die, kind, start: clock, cycles });
        clock += cycles;
    }
    Schedule { stages, total_cycles: clock, token_ns: clock / effective_ghz }
}
