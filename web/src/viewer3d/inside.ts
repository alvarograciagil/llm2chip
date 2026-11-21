// Inside a die: its netlist from the engine, and the value of every net of the instance on
// show for a window of cycles around the one on screen, as the engine replays the run.

import { create } from "zustand";
import type { Design, NetlistModule } from "../engine";
import type { Trace } from "./trace";

/** What one instance did in one cycle, as the engine replays it. */
export interface Frame {
  cycle: number;
  nets: Record<string, string>;
  /** For each instance inside it, the share of its bits that changed since the cycle before. */
  activity: Record<string, number>;
}

/** Cycles fetched at a time around the one on screen. */
export const WINDOW = 256;

interface InsideState {
  /** The die looked into, the instances entered from its top, and the area on show. */
  die: number | null;
  path: string[];
  area: string | null;
  selected: string | null;
  designs: Record<number, Design>;
  frames: { die: number; path: string; list: Frame[] } | null;
  loading: boolean;
  error: string | null;
  enter: (die: number) => void;
  leave: () => void;
  open: (instance: string) => void;
  goTo: (depth: number) => void;
  setArea: (area: string | null) => void;
  select: (id: string | null) => void;
  /** Makes sure the design of the die and the frames around `cycle` are there. */
  fetch: (trace: Trace, cycle: number) => Promise<void>;
  reset: () => void;
}

/** The engine's path of an instance from the instances entered. */
export function enginePath(path: string[]): string {
  return path.map((p) => `/${p}`).join("");
}

/** The frame of `cycle`, if the frames hold it. */
export function frameAt(frames: Frame[] | undefined, cycle: number): Frame | null {
  return frames?.find((f) => f.cycle === Math.floor(cycle)) ?? null;
}

/** How busy each instance and each area of `module` was, from 0 to 1: the busiest of them
 * is 1, so that what little moves still shows. */
export function scaledActivity(module: NetlistModule | null, frame: Frame | null): Record<string, number> {
  if (!module || !frame) return {};
  const most = Math.max(0, ...Object.values(frame.activity));
  if (most <= 0) return {};
  const out: Record<string, number> = {};
  for (const inst of module.instances) {
    const a = (frame.activity[inst.name] ?? 0) / most;
    if (a <= 0) continue;
    out[inst.name] = a;
    out[inst.group] = Math.max(out[inst.group] ?? 0, a);
  }
  return out;
}

let pending: Promise<void> | null = null;

async function post<T>(url: string, body: unknown): Promise<T> {
  const response = await fetch(url, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
  if (!response.ok) throw new Error(await response.text());
  return (await response.json()) as T;
}

export const useInside = create<InsideState>((set, get) => ({
  die: null,
  path: [],
  area: null,
  selected: null,
  designs: {},
  frames: null,
  loading: false,
  error: null,
  enter: (die) => set({ die, path: [], area: null, selected: null, frames: null, error: null }),
  leave: () => set({ die: null, path: [], area: null, selected: null, frames: null }),
  open: (instance) => set({ path: [...get().path, instance], area: null, selected: null }),
  goTo: (depth) => set({ path: get().path.slice(0, depth), area: null, selected: null }),
  setArea: (area) => set({ area, selected: null }),
  select: (selected) => set({ selected }),
  reset: () => set({ die: null, path: [], area: null, selected: null, designs: {}, frames: null, error: null }),
  fetch: async (trace, cycle) => {
    const { die, path, designs, frames } = get();
    if (die === null || pending) return;
    if (!trace.request) {
      set({ error: "Only a run this engine simulated on the logic gates can be looked into." });
      return;
    }
    const at = Math.floor(cycle);
    const where = enginePath(path);
    const covered = frames && frames.die === die && frames.path === where && frameAt(frames.list, at);
    if (designs[die] && covered) return;
    set({ loading: true, error: null });
    pending = (async () => {
      try {
        if (!designs[die]) {
          const design = await post<Design>("/api/trace/design", { request: trace.request, die });
          set({ designs: { ...get().designs, [die]: design } });
        }
        if (!covered) {
          const { frames: list } = await post<{ frames: Frame[] }>("/api/trace/inspect", { request: trace.request, die, path: where, cycle: at, window: WINDOW });
          set({ frames: { die, path: where, list } });
        }
      } catch (e) {
        set({ error: e instanceof Error ? e.message : String(e) });
      } finally {
        set({ loading: false });
        pending = null;
      }
    })();
    await pending;
  },
}));
