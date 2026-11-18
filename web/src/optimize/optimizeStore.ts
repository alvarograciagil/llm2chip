import { create } from "zustand";

export type Goal = "speed" | "area" | "efficiency" | "energy";
export type Effort = "quick" | "thorough";

/** One design the search measured on its own circuit, as the engine reports it. */
export interface Design {
  chips: number;
  cluster_bits: number;
  lane_bits: number;
  cell_bits: number;
  cycles: number;
  clock_hz: number;
  tokens_per_second: number;
  prompt_seconds: number;
  gates: number;
  area_mm2: number;
  die_mm2: number;
  /** Logic, cell memories, vector memories and microcode of each die, in mm². */
  parts_mm2: [number, number, number, number];
  sharing_name: string;
  /** Estimated by the architecture model at the measured speed. */
  energy_mj: number | null;
  power_w: number | null;
  area_per_tok_s: number;
  /** The designer's parameters for this design. */
  params: Record<string, unknown> | null;
}

export interface Report {
  goal: Goal;
  recommended: Design | null;
  alternatives: { label: "fastest" | "smallest" | "efficient" | "frugal"; design: Design }[];
  frontier: Design[];
  limiting: string[];
  too_few_chips: number[];
  measured: number;
  from_cache: number;
  failure: string | null;
}

export interface Progress {
  phase: "pruning" | "screening" | "climbing" | "done";
  done: number;
  planned: number;
  best: Design | null;
}

export type Status =
  | { state: "idle" }
  | { state: "running"; name: string; progress: Progress | null }
  | { state: "done"; name: string; report: Report }
  | { state: "error"; name: string; error: string };

/** What the person asks of the search, as the engine reads it. */
export interface Request {
  goal: Goal;
  limits: { min_tok_s: number; max_first_token_ms: number; max_chips: number; max_area_mm2: number; max_power_w: number };
  space: { chips: number[]; sharing: string; cluster_bits: [number, number]; lane_bits: [number, number]; cell_bits: [number, number] };
  effort: Effort;
  context: number;
  prompt: number;
  max_clock_ghz: number;
}

export const DEFAULT_REQUEST: Request = {
  goal: "speed",
  limits: { min_tok_s: 0, max_first_token_ms: 0, max_chips: 16, max_area_mm2: 0, max_power_w: 0 },
  space: { chips: [1, 2, 4, 6, 8, 12, 16, 24, 32], sharing: "layers,rows,mixed:auto", cluster_bits: [5, 11], lane_bits: [6, 7], cell_bits: [13, 24] },
  effort: "quick",
  context: 8192,
  prompt: 1024,
  max_clock_ghz: 5,
};

interface OptimizeState {
  status: Status;
  error: string | null;
  /** What the form asks, kept here so the project wizard can fill it in. */
  draft: Request;
  setDraft: (draft: Request) => void;
  start: (path: string, request: Request, factorSteps: number) => Promise<void>;
  poll: () => Promise<void>;
  cancel: () => Promise<void>;
}

const json = { "content-type": "application/json" };

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, init);
  if (!response.ok) throw new Error((await response.text()) || `Error ${response.status}`);
  return response.status === 202 || response.status === 204 ? (undefined as T) : response.json();
}

const text = (e: unknown) => (e instanceof Error ? e.message : String(e));

export const useOptimize = create<OptimizeState>((set, get) => ({
  status: { state: "idle" },
  error: null,
  draft: DEFAULT_REQUEST,
  setDraft: (draft) => set({ draft }),
  async start(path, request, factorSteps) {
    try {
      await call("/api/optimize", { method: "POST", headers: json, body: JSON.stringify({ path, request, factor_steps: factorSteps }) });
      set({ error: null });
      await get().poll();
    } catch (e) {
      set({ error: text(e) });
    }
  },
  async poll() {
    try {
      set({ status: await call<Status>("/api/optimize") });
    } catch (e) {
      set({ error: text(e) });
    }
  },
  async cancel() {
    try {
      await call("/api/optimize/cancel", { method: "POST" });
      await get().poll();
    } catch (e) {
      set({ error: text(e) });
    }
  },
}));

/** The parameters of the designer a measured design sets: its chips, how they share the
 * model, its units, lanes and cells, and its clock. The rest of the design stays. */
export function shapeOf(design: Design): Record<string, unknown> {
  const units = 2 ** design.cluster_bits;
  const [split, group] = design.sharing_name === "layers" ? ["layers", null] : design.sharing_name === "rows" ? ["tensor", design.chips] : ["mixed", Number(design.sharing_name.split(":")[1])];
  return {
    dies: design.chips,
    chip_split: split,
    ...(group ? { tensor_group: group } : {}),
    macs_per_die: 2 ** design.cell_bits,
    vector_units_per_die: units,
    vector_lanes_per_die: units * 2 ** design.lane_bits,
    freq_ghz: Math.round(design.clock_hz / 1e7) / 100,
    target_tok_s: 0,
  };
}
