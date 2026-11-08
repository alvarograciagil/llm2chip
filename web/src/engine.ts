import init, {
  CellSimulation,
  DieSimulation,
  builtin_models,
  default_params,
  estimate_design,
  optimization_catalog,
  presets,
  schedule_design,
  sweep_design,
  tech_nodes,
} from "./wasm/llm2chip_wasm.js";

export type Confidence = "derivada" | "calibrada" | "supuesta";
export type Maturity = "probada" | "plausible" | "experimental";
export type QualityRisk = "ninguno" | "medir" | "alto";

export interface Metric {
  id: string;
  section: string;
  label: string;
  value: number;
  unit: string;
  confidence: Confidence;
  why: string;
}

export type Category = "memoria" | "calculo" | "interconexion" | "vectorial" | "estado" | "sistema";

export interface BlockArea {
  id: string;
  label: string;
  per_die_mm2: number;
  what: string;
  category: Category;
  driven_by: string[];
  why: string;
  activity: number;
  energy_mj: number;
  bytes_token: number;
}

export interface Rect {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface CellPlan {
  cols: number;
  rows: number;
  count: number;
  rect: Rect;
  pitch_mm: number;
  sram_fraction: number;
  mac_fraction: number;
  scale_fraction: number;
  noc_fraction: number;
  segments: { label: string; layer: number | null; fraction: number }[];
}

export interface DiePlan {
  index: number;
  rect: Rect;
  first_layer: number | null;
  last_layer: number | null;
  has_embedding: boolean;
  has_head: boolean;
  weights: number;
  blocks: { id: string; rect: Rect }[];
  cells: CellPlan;
}

export interface Floorplan {
  width: number;
  height: number;
  num_layers: number;
  dies: DiePlan[];
  memory: { label: string; rect: Rect }[];
}

export interface Violation {
  severity: "error" | "aviso";
  message: string;
  fix: string;
  field: string | null;
}

export interface Report {
  model: string;
  dies: number;
  die_area_mm2: number;
  total_area_mm2: number;
  blocks: BlockArea[];
  metrics: Metric[];
  violations: Violation[];
  floorplan: Floorplan;
}

export interface Optimization {
  id: string;
  name: string;
  maturity: Maturity;
  quality_risk: QualityRisk;
  what: string;
  effect: string;
}

export interface TechNode {
  id: string;
  label: string;
}

export interface ModelSource {
  id: string;
  name: string;
  config: string;
}

export interface Preset {
  id: string;
  name: string;
  model: string;
  description: string;
  params: Record<string, unknown>;
}

export type ChipParams = Record<string, unknown> & { optimizations: string[] };

export interface ScheduledStage {
  name: string;
  layer: number | null;
  die: number;
  kind: "matvec" | "vector" | "embedding" | "link";
  start: number;
  cycles: number;
}

export interface Schedule {
  stages: ScheduledStage[];
  total_cycles: number;
  token_ns: number;
}

export interface SweepResult {
  metrics: Record<string, number>;
  errors: number;
}

export type Bit = { net: [number, number] } | "zero" | "one";
export type PrimType = "and2" | "or2" | "xor2" | "not" | "mux2" | "dff" | "sram" | "ram2p";

export interface Instance {
  name: string;
  kind: { kind: "prim"; of: { type: PrimType; addr_bits?: number; width?: number } } | { kind: "module"; of: string };
  group: string;
  conns: Record<string, Bit[]>;
}

export interface NetlistModule {
  name: string;
  title: string;
  description: string;
  ports: { name: string; dir: "input" | "output"; net: number }[];
  nets: { name: string; width: number }[];
  instances: Instance[];
}

export interface Design {
  top: string;
  modules: Record<string, NetlistModule>;
}

export interface CellInfo {
  config: { weight_bits: number; activation_bits: number; accumulator_bits: number; mac_stages: number; adder_stages: number; addr_bits: number; coding: string; exponent_bits: number };
  latency: number;
  counts: Record<string, number>;
}

export { CellSimulation, DieSimulation };

export async function initEngine() {
  await init();
}

export const builtinModels = (): ModelSource[] => JSON.parse(builtin_models());
export const defaultParams = (): ChipParams => JSON.parse(default_params());
export const optimizationCatalog = (): Optimization[] => JSON.parse(optimization_catalog());
export const designPresets = (): Preset[] => JSON.parse(presets());
export const techNodes = (): TechNode[] => JSON.parse(tech_nodes());

export function scheduleDesign(model: ModelSource, params: ChipParams): Schedule {
  return JSON.parse(schedule_design(model.name, model.config, JSON.stringify(params)));
}

export function sweepDesign(model: ModelSource, params: ChipParams, variants: Record<string, unknown>[]): SweepResult[] {
  return JSON.parse(sweep_design(model.name, model.config, JSON.stringify(params), JSON.stringify(variants)));
}

export function estimateDesign(model: ModelSource, params: ChipParams): Report {
  return JSON.parse(estimate_design(model.name, model.config, JSON.stringify(params)));
}
