import { create } from "zustand";

/** What OpenROAD measured on the laid-out cell. */
export interface LayoutResult {
  platform: string;
  cell_area_um2: number | null;
  die_area_um2: number | null;
  cells: number | null;
  power_w: number | null;
  fmax_mhz: number | null;
  drc_errors: number;
  period_ns: number;
  image: string | null;
  gds: string;
}

/** The cell to lay out, as the design's circuit chose it. */
export interface LayoutRequest {
  node: string;
  /** Exactly the cell the estimator chose, and its name. */
  cell: Record<string, unknown>;
  cell_key: string;
  period_ns: number;
}

interface LayoutState {
  state: "idle" | "running" | "done" | "error";
  phase: string;
  seconds: number;
  result: LayoutResult | null;
  /** What the result was made for, to tell when the design has moved on. */
  request: LayoutRequest | null;
  error: string | null;
  start: (request: LayoutRequest) => Promise<void>;
  poll: () => Promise<void>;
}

export const useLayout = create<LayoutState>((set, get) => ({
  state: "idle",
  phase: "",
  seconds: 0,
  result: null,
  request: null,
  error: null,

  async start(request) {
    set({ state: "running", phase: "Empezando", seconds: 0, error: null, request });
    try {
      const response = await fetch("/api/layout", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(request) });
      if (!response.ok) throw new Error((await response.text()) || `Error ${response.status}`);
      await get().poll();
    } catch (e) {
      set({ state: "error", error: e instanceof Error ? e.message : String(e) });
    }
  },

  async poll() {
    const started = performance.now();
    for (;;) {
      const response = await fetch("/api/layout");
      const status = await response.json();
      if (status.state === "running") {
        set({ phase: status.phase, seconds: Math.round((performance.now() - started) / 1000) });
        await new Promise((resolve) => setTimeout(resolve, 1500));
        continue;
      }
      if (status.state === "done") set({ state: "done", result: status.result });
      else if (status.state === "error") set({ state: "error", error: status.error });
      return;
    }
  },
}));
