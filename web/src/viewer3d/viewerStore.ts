import { create } from "zustand";
import { firstBreak, nextStop, NO_BREAKPOINTS, readTrace, totalCycles, type Breakpoints, type Stride, type Trace } from "./trace";

export type TinyModelName = "grouped" | "plain";

export interface RecordRequest {
  model: TinyModelName;
  chips: number;
  sharing: string;
  tokens: number[];
  /** A whole design of a real model, traced by operations instead of on the gates. */
  design?: DesignRequest;
}

export interface DesignRequest {
  path: string;
  cluster_bits: number;
  lane_bits: number;
  cell_bits: number;
  context: number;
}

/** The design on the sidebar as a request to trace it, or why it cannot be traced. */
export function designRequest(path: string | undefined, params: Record<string, unknown>): RecordRequest | string {
  if (!path) return "Trace a design of a model in the known folders: the built-in ones have no weights.";
  const chips = Number(params.dies) || 0;
  if (chips < 1) return "Choose how many dies the design has.";
  const split = String(params.chip_split ?? "layers");
  const sharing = chips === 1 || split === "layers" ? "layers" : split === "tensor" ? "rows" : `mixed:${Number(params.tensor_group) || 2}`;
  const log2 = (v: unknown) => Math.max(0, Math.round(Math.log2(Math.max(1, Number(v) || 1))));
  const units = Math.max(1, Number(params.vector_units_per_die) || 1);
  const lanes = Math.max(1, (Number(params.vector_lanes_per_die) || units) / units);
  const design = { path, cluster_bits: log2(units), lane_bits: log2(lanes), cell_bits: log2(params.macs_per_die), context: Math.max(16, Number(params.context_tokens) || 8192) };
  return { model: "plain", chips, sharing, tokens: [], design };
}

type RecordState = { state: "idle" } | { state: "running" } | { state: "error"; error: string };

interface ViewerState {
  trace: Trace | null;
  /** The cycle on show; fractional while playing, so slow speeds move smoothly. */
  cycle: number;
  playing: boolean;
  /** Cycles a second of playback. */
  speed: number;
  stride: Stride;
  selected: number | null;
  recording: RecordState;
  error: string | null;
  breakpoints: Breakpoints;
  /** Lanes lit by what each die runs now, or by how much it has worked so far. */
  colouring: "now" | "heat";
  /** A second trace to compare with. */
  other: Trace | null;
  /** The camera turns round the chips by itself and captions say what happens. */
  touring: boolean;
  /** Why playback last stopped by itself. */
  stopped: string | null;
  load: (trace: Trace) => void;
  open: (file: File) => Promise<void>;
  record: (request: RecordRequest) => Promise<void>;
  setCycle: (cycle: number) => void;
  advance: (seconds: number) => void;
  setPlaying: (playing: boolean) => void;
  setSpeed: (speed: number) => void;
  setStride: (stride: Stride) => void;
  step: (direction: 1 | -1) => void;
  select: (die: number | null) => void;
  setBreakpoints: (breakpoints: Breakpoints) => void;
  setColouring: (colouring: "now" | "heat") => void;
  openOther: (file: File | null) => Promise<void>;
  setTouring: (touring: boolean) => void;
  /** Moves on to the next breakpoint, or to the end. */
  runToBreak: () => void;
}

export const DEFAULT_RECORD: RecordRequest = { model: "grouped", chips: 4, sharing: "mixed:2", tokens: [3, 5, 1] };

async function poll(): Promise<Trace> {
  for (;;) {
    await new Promise((resolve) => setTimeout(resolve, 500));
    const response = await fetch("/api/trace");
    if (!response.ok) throw new Error(await response.text());
    const status = (await response.json()) as { state: string; trace?: unknown; error?: string };
    if (status.state === "done") return readTrace(status.trace);
    if (status.state === "error") throw new Error(status.error ?? "The recording failed.");
    if (status.state === "idle") throw new Error("The engine lost the recording.");
  }
}

export const useViewer = create<ViewerState>((set, get) => ({
  trace: null,
  cycle: 0,
  playing: false,
  speed: 200,
  stride: "op",
  selected: null,
  recording: { state: "idle" },
  error: null,
  breakpoints: NO_BREAKPOINTS,
  stopped: null,
  colouring: "now",
  other: null,
  touring: false,
  load: (trace) => set({ trace, cycle: 0, playing: false, selected: null, error: null, stopped: null }),
  open: async (file) => {
    try {
      get().load(readTrace(JSON.parse(await file.text())));
    } catch (e) {
      set({ error: e instanceof Error ? e.message : String(e) });
    }
  },
  record: async (request) => {
    set({ recording: { state: "running" }, error: null });
    try {
      const response = await fetch("/api/trace", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(request) });
      if (!response.ok) throw new Error(await response.text());
      const trace = await poll();
      get().load(trace);
      set({ recording: { state: "idle" } });
    } catch (e) {
      set({ recording: { state: "error", error: e instanceof Error ? e.message : String(e) } });
    }
  },
  setCycle: (cycle) => {
    const trace = get().trace;
    if (!trace) return;
    set({ cycle: Math.min(Math.max(0, cycle), totalCycles(trace) - 1) });
  },
  advance: (seconds) => {
    const { trace, cycle, speed, playing, breakpoints } = get();
    if (!trace || !playing) return;
    const end = totalCycles(trace) - 1;
    const next = Math.min(cycle + speed * seconds, end);
    const hit = firstBreak(trace, cycle, next, breakpoints);
    if (hit) set({ cycle: hit.cycle, playing: false, stopped: hit.reason });
    else set(next >= end ? { cycle: end, playing: false } : { cycle: next });
  },
  setPlaying: (playing) => {
    const { trace, cycle } = get();
    // Playing from the end starts again.
    if (playing && trace && cycle >= totalCycles(trace) - 1) set({ cycle: 0 });
    set({ playing, stopped: playing ? null : get().stopped });
  },
  setSpeed: (speed) => set({ speed }),
  setStride: (stride) => set({ stride }),
  step: (direction) => {
    const { trace, cycle, stride, selected } = get();
    if (!trace) return;
    set({ playing: false, cycle: nextStop(trace, cycle, stride, direction, selected ?? undefined) });
  },
  select: (die) => set({ selected: die }),
  setBreakpoints: (breakpoints) => set({ breakpoints }),
  setColouring: (colouring) => set({ colouring }),
  openOther: async (file) => {
    if (!file) return set({ other: null });
    try {
      set({ other: readTrace(JSON.parse(await file.text())), error: null });
    } catch (e) {
      set({ error: e instanceof Error ? e.message : String(e) });
    }
  },
  setTouring: (touring) => set({ touring }),
  runToBreak: () => {
    const { trace, cycle, breakpoints } = get();
    if (!trace) return;
    const end = totalCycles(trace) - 1;
    const hit = firstBreak(trace, cycle, end, breakpoints);
    set(hit ? { cycle: hit.cycle, playing: false, stopped: hit.reason } : { cycle: end, playing: false, stopped: null });
  },
}));
