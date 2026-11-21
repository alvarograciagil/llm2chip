import { create } from "zustand";
import { nextStop, readTrace, totalCycles, type Stride, type Trace } from "./trace";

export type TinyModelName = "grouped" | "plain";

export interface RecordRequest {
  model: TinyModelName;
  chips: number;
  sharing: string;
  tokens: number[];
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
  load: (trace) => set({ trace, cycle: 0, playing: false, selected: null, error: null }),
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
    const { trace, cycle, speed, playing } = get();
    if (!trace || !playing) return;
    const end = totalCycles(trace) - 1;
    const next = cycle + speed * seconds;
    set(next >= end ? { cycle: end, playing: false } : { cycle: next });
  },
  setPlaying: (playing) => {
    const { trace, cycle } = get();
    // Playing from the end starts again.
    if (playing && trace && cycle >= totalCycles(trace) - 1) set({ cycle: 0 });
    set({ playing });
  },
  setSpeed: (speed) => set({ speed }),
  setStride: (stride) => set({ stride }),
  step: (direction) => {
    const { trace, cycle, stride, selected } = get();
    if (!trace) return;
    set({ playing: false, cycle: nextStop(trace, cycle, stride, direction, selected ?? undefined) });
  },
  select: (die) => set({ selected: die }),
}));
