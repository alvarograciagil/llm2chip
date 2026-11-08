//! Chip parameters, technology data and the architecture estimator.
//! Everything here is pure computation so it also runs in the browser through WASM.

mod estimate;
mod floorplan;
mod optim;
mod params;
mod tech;

pub use estimate::{estimate, BlockArea, Category, Confidence, Metric, Report, Severity, Violation};
pub use floorplan::{CellPlan, CellSegment, DiePlan, Floorplan, MemoryStack, PlacedBlock, Rect};
pub use optim::{catalog, Maturity, OptInfo, QualityRisk};
pub use params::{ChipParams, KvBackend, LinkKind, LoadInterface, WeightBackend};
pub use tech::{node, nodes, TechNode};
