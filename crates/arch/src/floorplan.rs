//! Deterministic chip floorplan: every area comes from the estimator and every
//! coordinate is in millimetres. It explains the design; it is not place-and-route.

use crate::estimate::BlockArea;
use crate::params::{ChipParams, KvBackend, WeightBackend};
use model_ir::{Block, ModelGraph};
use serde::Serialize;

/// Blocks that are interleaved across the cell array rather than placed apart.
const ARRAY_BLOCKS: [&str; 4] = ["cell_weights", "cell_macs", "scale_units", "activation_noc"];
const HBM_STACK_MM: f64 = 11.0;

#[derive(Clone, Copy, Debug, Serialize)]
pub struct Rect {
    pub x: f64,
    pub y: f64,
    pub w: f64,
    pub h: f64,
}

impl Rect {
    fn area(&self) -> f64 {
        self.w * self.h
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct PlacedBlock {
    pub id: &'static str,
    pub rect: Rect,
}

#[derive(Clone, Debug, Serialize)]
pub struct CellSegment {
    pub label: String,
    pub layer: Option<u32>,
    pub fraction: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct CellPlan {
    pub cols: u64,
    pub rows: u64,
    pub count: u64,
    /// The array's rectangle, relative to its die.
    pub rect: Rect,
    pub pitch_mm: f64,
    /// Share of a cell taken by each part. They add up to one.
    pub sram_fraction: f64,
    pub mac_fraction: f64,
    pub scale_fraction: f64,
    pub noc_fraction: f64,
    /// How this cell's memory divides between the matrices of the die.
    pub segments: Vec<CellSegment>,
}

#[derive(Clone, Debug, Serialize)]
pub struct DiePlan {
    pub index: u32,
    /// Position and size inside the package.
    pub rect: Rect,
    pub first_layer: Option<u32>,
    pub last_layer: Option<u32>,
    pub has_embedding: bool,
    pub has_head: bool,
    pub weights: u64,
    /// Blocks relative to the die, excluding those interleaved in the array.
    pub blocks: Vec<PlacedBlock>,
    pub cells: CellPlan,
}

#[derive(Clone, Debug, Serialize)]
pub struct MemoryStack {
    pub label: &'static str,
    pub rect: Rect,
}

#[derive(Clone, Debug, Serialize)]
pub struct Floorplan {
    pub width: f64,
    pub height: f64,
    pub num_layers: u32,
    pub dies: Vec<DiePlan>,
    pub memory: Vec<MemoryStack>,
}

struct DieLayers {
    first: Option<u32>,
    last: Option<u32>,
    embedding: bool,
    head: bool,
    weights: u64,
    segments: Vec<(String, Option<u32>, u64)>,
}

fn layer_weights(model: &ModelGraph, layer: u32) -> u64 {
    model.params_where(|t| t.layer == Some(layer) && t.block != Block::Norm)
}

/// Assigns consecutive layers to dies so each holds a similar number of weights.
fn partition(model: &ModelGraph, dies: u32) -> Vec<DieLayers> {
    let embedding = model.params_in(Block::Embedding);
    let head = model.params_in(Block::Head);
    let layer_total: u64 = (0..model.num_layers).map(|l| layer_weights(model, l)).sum();
    let mut remaining = embedding + head + layer_total;
    let mut target = remaining as f64 / dies as f64;

    let mut out: Vec<DieLayers> = (0..dies)
        .map(|_| DieLayers { first: None, last: None, embedding: false, head: false, weights: 0, segments: Vec::new() })
        .collect();
    out[0].embedding = true;
    out[0].weights = embedding;
    out[0].segments.push(("Embedding".into(), None, embedding));

    let mut current = 0usize;
    for layer in 0..model.num_layers {
        let weights = layer_weights(model, layer);
        let layers_left = (model.num_layers - layer) as usize;
        let dies_left = dies as usize - current - 1;
        let full = out[current].weights as f64 + weights as f64 / 2.0 > target;
        if dies_left > 0 && (full && out[current].first.is_some() || dies_left >= layers_left) {
            // Re-aim at what is left so the last die does not end up nearly empty.
            remaining -= out[current].weights;
            current += 1;
            target = remaining as f64 / (dies as usize - current) as f64;
        }
        let die = &mut out[current];
        die.first.get_or_insert(layer);
        die.last = Some(layer);
        die.weights += weights;
        die.segments.push((format!("Capa {layer}"), Some(layer), weights));
    }

    let last = out.last_mut().expect("al menos un die");
    last.head = true;
    last.weights += head;
    last.segments.push(("Salida de vocabulario".into(), None, head));
    out
}

/// Lays `areas` out inside `bounds`, keeping rectangles close to square.
fn squarify(areas: &[(usize, f64)], bounds: Rect) -> Vec<(usize, Rect)> {
    let mut sorted: Vec<_> = areas.iter().copied().filter(|(_, a)| *a > 0.0).collect();
    sorted.sort_by(|a, b| b.1.total_cmp(&a.1));
    let total: f64 = sorted.iter().map(|(_, a)| a).sum();
    if total <= 0.0 {
        return Vec::new();
    }
    let scale = bounds.area() / total;
    let scaled: Vec<_> = sorted.iter().map(|(i, a)| (*i, a * scale)).collect();

    let mut out = Vec::new();
    let mut free = bounds;
    let mut index = 0;
    while index < scaled.len() {
        let short = free.w.min(free.h);
        let mut end = index + 1;
        let mut sum = scaled[index].1;
        let mut best = worst_ratio(&scaled[index..end], sum, short);
        while end < scaled.len() {
            let candidate = sum + scaled[end].1;
            let ratio = worst_ratio(&scaled[index..=end], candidate, short);
            if ratio > best {
                break;
            }
            best = ratio;
            sum = candidate;
            end += 1;
        }

        let row = &scaled[index..end];
        let horizontal = free.w >= free.h;
        let thickness = sum / short;
        let mut offset = 0.0;
        for (i, area) in row {
            let length = area / thickness;
            let rect = if horizontal {
                Rect { x: free.x, y: free.y + offset, w: thickness, h: length }
            } else {
                Rect { x: free.x + offset, y: free.y, w: length, h: thickness }
            };
            out.push((*i, rect));
            offset += length;
        }
        if horizontal {
            free.x += thickness;
            free.w -= thickness;
        } else {
            free.y += thickness;
            free.h -= thickness;
        }
        index = end;
    }
    out
}

fn worst_ratio(row: &[(usize, f64)], sum: f64, short: f64) -> f64 {
    let (min, max) = row
        .iter()
        .fold((f64::MAX, 0.0f64), |(lo, hi), (_, a)| (lo.min(*a), hi.max(*a)));
    let s2 = sum * sum;
    let w2 = short * short;
    (w2 * max / s2).max(s2 / (w2 * min))
}

pub(crate) fn build_floorplan(
    model: &ModelGraph,
    p: &ChipParams,
    blocks: &[BlockArea],
    dies: u32,
    die_area: f64,
) -> Floorplan {
    let side = die_area.sqrt();
    let is_array = |b: &BlockArea| ARRAY_BLOCKS.contains(&b.id);
    let array_area: f64 = blocks.iter().filter(|b| is_array(b)).map(|b| b.per_die_mm2).sum();
    let array_width = side * array_area / die_area;
    let array = Rect { x: 0.0, y: 0.0, w: array_width, h: side };

    let others: Vec<_> = blocks
        .iter()
        .enumerate()
        .filter(|(_, b)| !is_array(b))
        .map(|(i, b)| (i, b.per_die_mm2))
        .collect();
    let column = Rect { x: array_width, y: 0.0, w: side - array_width, h: side };
    let placed: Vec<PlacedBlock> = squarify(&others, column)
        .into_iter()
        .map(|(i, rect)| PlacedBlock { id: blocks[i].id, rect })
        .collect();

    let part = |id: &str| blocks.iter().find(|b| b.id == id).map_or(0.0, |b| b.per_die_mm2 / array_area.max(1e-12));
    let count = p.macs_per_die.max(1);
    let aspect = array.w / array.h.max(1e-12);
    let rows = ((count as f64 / aspect.max(1e-6)).sqrt().ceil() as u64).max(1);
    let cols = count.div_ceil(rows).max(1);

    let columns = (dies as f64).sqrt().ceil() as u32;
    let gap = (side * 0.14).max(0.4);
    let pitch = side + gap;
    let rows_of_dies = dies.div_ceil(columns);
    let package_margin = gap;
    let mut width = columns as f64 * pitch - gap + 2.0 * package_margin;
    let mut height = rows_of_dies as f64 * pitch - gap + 2.0 * package_margin;

    let dies_out: Vec<DiePlan> = partition(model, dies)
        .into_iter()
        .enumerate()
        .map(|(i, layers)| {
            // Snake order keeps consecutive dies next to each other.
            let row = i as u32 / columns;
            let col = if row.is_multiple_of(2) { i as u32 % columns } else { columns - 1 - i as u32 % columns };
            let total = layers.weights.max(1) as f64;
            DiePlan {
                index: i as u32,
                rect: Rect {
                    x: package_margin + col as f64 * pitch,
                    y: package_margin + row as f64 * pitch,
                    w: side,
                    h: side,
                },
                first_layer: layers.first,
                last_layer: layers.last,
                has_embedding: layers.embedding,
                has_head: layers.head,
                weights: layers.weights,
                blocks: placed.clone(),
                cells: CellPlan {
                    cols,
                    rows,
                    count,
                    rect: array,
                    pitch_mm: array.h / rows as f64,
                    sram_fraction: part("cell_weights"),
                    mac_fraction: part("cell_macs"),
                    scale_fraction: part("scale_units"),
                    noc_fraction: part("activation_noc"),
                    segments: layers
                        .segments
                        .into_iter()
                        .map(|(label, layer, weights)| CellSegment { label, layer, fraction: weights as f64 / total })
                        .collect(),
                },
            }
        })
        .collect();

    let mut memory = Vec::new();
    let hbm = p.weight_backend == WeightBackend::Hbm4 || p.kv_backend == KvBackend::Hbm4;
    let external: Option<(&'static str, u32, f64)> = if hbm {
        Some(("HBM4", p.hbm_stacks, HBM_STACK_MM))
    } else if p.weight_backend == WeightBackend::ExternalDdr {
        Some(("Memoria externa", 1, side * 0.5))
    } else if p.has("checkpoint_estado") || p.has("embedding_externo") {
        Some(("DRAM externa", 1, side * 0.35))
    } else {
        None
    };
    if let Some((label, count, size)) = external {
        let y = height;
        for i in 0..count {
            memory.push(MemoryStack {
                label,
                rect: Rect { x: package_margin + i as f64 * (size + gap), y, w: size, h: size },
            });
        }
        width = width.max(package_margin * 2.0 + count as f64 * (size + gap) - gap);
        height += size + package_margin;
    }

    Floorplan {
        width,
        height,
        num_layers: model.num_layers,
        dies: dies_out,
        memory,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::estimate::estimate;

    fn qwen() -> ModelGraph {
        ModelGraph::from_config_json("qwen", include_str!("../../../models/qwen3.8-27b.json")).unwrap()
    }

    #[test]
    fn squarify_preserves_area_and_stays_inside() {
        let bounds = Rect { x: 2.0, y: 3.0, w: 10.0, h: 40.0 };
        let input = [(0, 30.0), (1, 12.0), (2, 8.0), (3, 2.0), (4, 0.5)];
        let out = squarify(&input, bounds);
        assert_eq!(out.len(), input.len());
        let sum: f64 = out.iter().map(|(_, r)| r.area()).sum();
        assert!((sum - bounds.area()).abs() < 1e-6);
        for (_, r) in &out {
            assert!(r.x >= 2.0 - 1e-9 && r.y >= 3.0 - 1e-9);
            assert!(r.x + r.w <= 12.0 + 1e-9 && r.y + r.h <= 43.0 + 1e-9);
        }
    }

    #[test]
    fn placed_blocks_cover_the_die_without_overlap() {
        let report = estimate(&qwen(), &ChipParams::default()).unwrap();
        let die = &report.floorplan.dies[0];
        let covered: f64 = die.blocks.iter().map(|b| b.rect.area()).sum::<f64>() + die.cells.rect.area();
        assert!((covered / die.rect.area() - 1.0).abs() < 1e-6, "{covered}");
        for (i, a) in die.blocks.iter().enumerate() {
            for b in &die.blocks[i + 1..] {
                let overlap_w = (a.rect.x + a.rect.w).min(b.rect.x + b.rect.w) - a.rect.x.max(b.rect.x);
                let overlap_h = (a.rect.y + a.rect.h).min(b.rect.y + b.rect.h) - a.rect.y.max(b.rect.y);
                assert!(overlap_w < 1e-6 || overlap_h < 1e-6, "{} y {}", a.id, b.id);
            }
        }
    }

    #[test]
    fn every_layer_lands_on_exactly_one_die_in_order() {
        let report = estimate(&qwen(), &ChipParams::default()).unwrap();
        let mut next = 0;
        for die in &report.floorplan.dies {
            let (first, last) = (die.first_layer.unwrap(), die.last_layer.unwrap());
            assert_eq!(first, next);
            next = last + 1;
        }
        assert_eq!(next, 64);
        assert!(report.floorplan.dies[0].has_embedding);
        assert!(report.floorplan.dies.last().unwrap().has_head);
    }

    #[test]
    fn dies_hold_a_balanced_share_of_weights() {
        let report = estimate(&qwen(), &ChipParams::default()).unwrap();
        let weights: Vec<_> = report.floorplan.dies.iter().map(|d| d.weights as f64).collect();
        let mean = weights.iter().sum::<f64>() / weights.len() as f64;
        for w in weights {
            assert!((w / mean - 1.0).abs() < 0.35, "{w} frente a {mean}");
        }
    }

    #[test]
    fn cell_parts_and_segments_add_up() {
        let report = estimate(&qwen(), &ChipParams::default()).unwrap();
        let cells = &report.floorplan.dies[3].cells;
        let parts = cells.sram_fraction + cells.mac_fraction + cells.scale_fraction + cells.noc_fraction;
        assert!((parts - 1.0).abs() < 1e-9);
        let segments: f64 = cells.segments.iter().map(|s| s.fraction).sum();
        assert!((segments - 1.0).abs() < 1e-9);
        assert!(cells.cols * cells.rows >= cells.count);
    }

    #[test]
    fn single_die_holds_every_layer() {
        let p = ChipParams { dies: 1, die_max_mm2: 815.0, ..Default::default() };
        let report = estimate(&qwen(), &p).unwrap();
        let die = &report.floorplan.dies[0];
        assert_eq!((die.first_layer, die.last_layer), (Some(0), Some(63)));
        assert!(die.has_embedding && die.has_head);
    }

    #[test]
    fn external_memory_is_drawn_outside_the_dies() {
        let p = ChipParams { weight_backend: WeightBackend::Hbm4, hbm_stacks: 4, ..Default::default() };
        let report = estimate(&qwen(), &p).unwrap();
        assert_eq!(report.floorplan.memory.len(), 4);
        let bottom = report.floorplan.dies.iter().map(|d| d.rect.y + d.rect.h).fold(0.0, f64::max);
        assert!(report.floorplan.memory[0].rect.y >= bottom);
    }
}
