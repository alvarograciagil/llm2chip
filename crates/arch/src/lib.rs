//! Chip parameters, technology data and the architecture estimator.
//! Everything here is pure computation so it also runs in the browser through WASM.

mod estimate;
mod floorplan;
mod optim;
mod params;
mod tech;

pub use estimate::{link_plan, LinkPlan, FIRMWARE_PROGRAM_BYTES, TOKENIZER_BITS_PER_TOKEN, estimate, resolved_params, BlockArea, CircuitDecision, Category, Confidence, Metric, Report, Severity, Violation};
pub use floorplan::{CellPlan, CellSegment, DiePlan, Floorplan, MemoryStack, PlacedBlock, Rect};
pub use optim::{catalog, Maturity, OptInfo, QualityRisk};
pub use params::{ChipParams, ChipSplit, CircuitGoal, DieMode, KvBackend, LinkKind, LoadInterface, WeightBackend};
pub use tech::{node, nodes, TechNode};
