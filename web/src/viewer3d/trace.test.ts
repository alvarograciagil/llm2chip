import { describe, expect, it } from "vitest";
import { cacheAt, cacheHolders, compareTraces, cyclesByKind, dieAt, useShare, divergedAt, firstBreak, linkWordsIn, NO_BREAKPOINTS, dieTotals, linkAt, nextStop, readTrace, stepAt, totalCycles, type Trace } from "./trace";
import { layoutScene, linkPoint } from "./layout3d";

/** Two dies over ten cycles: the first works through two operations of layer 0 and waits a
 * cycle on the link; the second takes what the first sends and halts early. */
function sample(): Trace {
  return {
    version: 1,
    title: "sample",
    source: "gates",
    sharing: "layers",
    context: 4,
    layers: 1,
    ops: [
      { kind: "Matvec", name: "q_proj", layer: 0, slot: null },
      { kind: "KvAppend", name: "kv", layer: 0, slot: 0 },
    ],
    dies: [
      {
        units: 2,
        lanes: 4,
        cells: 8,
        group: null,
        words: [
          { class: "cells", op: 0, text: "a" },
          { class: "cells", op: 0, text: "b" },
          { class: "vector", op: 1, text: "c" },
          { class: "send", op: 1, text: "d" },
          { class: "halt", op: 1, text: "e" },
        ],
        entries: [0],
        pcs: [0, 1, 1, 2, 2, 3, 3, 4, -1, -1],
        stalls: [[6, 1]],
        spans: [],
      },
      {
        units: 2,
        lanes: 4,
        cells: 8,
        group: null,
        words: [
          { class: "take", op: null, text: "f" },
          { class: "halt", op: null, text: "g" },
        ],
        entries: [0],
        pcs: [0, 0, 0, 0, 0, 0, 0, 1, -1, -1],
        stalls: [[0, 6]],
        spans: [],
      },
    ],
    links: [{ from: 0, to: 1, port: "link", spans: [[5, 1, 1], [6, 1, 2]] }],
    steps: [
      { routine: "token", token: 3, position: 0, start: 0, cycles: 8 },
      { routine: "logits", token: 3, position: 0, start: 8, cycles: 2 },
    ],
    kv: [{ cycle: 3, die: 0, op: 1, slot: 0, position: 0 }],
    divergences: [{ die: 0, op: 1, buffer: 2, index: 0, got: 1, want: 2, step: 0, cycle: 3 }],
  };
}

describe("a trace read at a cycle", () => {
  it("tells what each die runs, and whether it waits or has finished", () => {
    const trace = sample();
    expect(totalCycles(trace)).toBe(10);
    expect(dieAt(trace, 0, 3)).toMatchObject({ pc: 2, op: { name: "kv" }, stalled: false, halted: false });
    expect(dieAt(trace, 0, 6).stalled).toBe(true);
    expect(dieAt(trace, 0, 9).halted).toBe(true);
    expect(dieAt(trace, 1, 2)).toMatchObject({ op: null, stalled: true });
    expect(stepAt(trace, 8)).toBe(1);
  });

  it("tells a word crossing a link from one held back", () => {
    const link = sample().links[0];
    expect([4, 5, 6, 7].map((c) => linkAt(link, c))).toEqual([0, 1, 2, 0]);
  });

  it("counts the words a link carried and held back over a window", () => {
    const link = sample().links[0];
    expect(linkWordsIn(link, 0, 9)).toEqual({ crossed: 1, waited: 1 });
    expect(linkWordsIn(link, 6, 6)).toEqual({ crossed: 0, waited: 1 });
    expect(linkWordsIn(link, 0, 4)).toEqual({ crossed: 0, waited: 0 });
  });

  it("knows which dies keep each layer's cache", () => {
    expect([...(cacheHolders(sample()).get(1) ?? [])]).toEqual([0]);
  });

  it("fills the cache from the cycle the operation starts", () => {
    const trace = sample();
    expect(cacheAt(trace, 2).size).toBe(0);
    expect([...(cacheAt(trace, 3).get(0)?.get(1) ?? [])]).toEqual([0]);
  });

  it("steps by word, operation and token, forwards and back", () => {
    const trace = sample();
    expect(nextStop(trace, 1, "word", 1, 0)).toBe(3);
    expect(nextStop(trace, 0, "op", 1, 0)).toBe(3);
    expect(nextStop(trace, 4, "op", -1, 0)).toBe(0);
    expect(nextStop(trace, 2, "step", 1)).toBe(8);
    expect(nextStop(trace, 9, "step", -1)).toBe(8);
    expect(nextStop(trace, 9, "cycle", 1)).toBe(9);
  });

  it("counts the cycles each die worked, waited and sat finished", () => {
    expect(dieTotals(sample(), 1)).toEqual({ work: 2, stalled: 6, halted: 2 });
  });

  it("marks a die from the cycle it starts the operation that parts from the exact machine", () => {
    const trace = sample();
    expect(divergedAt(trace, 0, 2)).toBeNull();
    expect(divergedAt(trace, 0, 3)?.op).toBe(1);
    expect(divergedAt(trace, 1, 9)).toBeNull();
  });

  it("stops playback at the first breakpoint it passes", () => {
    const trace = sample();
    expect(firstBreak(trace, 0, 9, NO_BREAKPOINTS)).toBeNull();
    expect(firstBreak(trace, 0, 9, { ...NO_BREAKPOINTS, stall: true })).toEqual({ cycle: 6, reason: "Die 0 waits on a link" });
    expect(firstBreak(trace, 0, 9, { ...NO_BREAKPOINTS, divergence: true })?.cycle).toBe(3);
    expect(firstBreak(trace, 0, 9, { ...NO_BREAKPOINTS, kind: "KvAppend" })).toEqual({ cycle: 3, reason: "Die 0 starts kv" });
    expect(firstBreak(trace, 3, 9, { ...NO_BREAKPOINTS, kind: "KvAppend" })).toBeNull();
    expect(firstBreak(trace, -1, 9, { ...NO_BREAKPOINTS, layer: 0 })?.cycle).toBe(0);
  });

  it("refuses what is not a trace of this version", () => {
    expect(() => readTrace({ version: 99, dies: [], steps: [], ops: [] })).toThrow(/version 99/);
    expect(() => readTrace("nothing")).toThrow();
  });
});

/** The same two dies traced by operations: the first runs op 0 then op 1, the second
 * waits for its turn and then runs op 1, slower, so the first waits for it. */
function byOps(): Trace {
  const t = sample();
  t.source = "timing";
  t.dies[0] = { ...t.dies[0], pcs: [], words: [{ class: "cells", op: 0, text: "q" }, { class: "vector", op: 1, text: "kv" }], spans: [[0, 3, 0], [3, 2, 1]], stalls: [[5, 2]] };
  t.dies[1] = { ...t.dies[1], pcs: [], words: [{ class: "cells", op: 0, text: "q" }, { class: "vector", op: 1, text: "kv" }], spans: [[3, 4, 1]], stalls: [] };
  return t;
}

describe("a trace by operations", () => {
  it("tells what each die runs, waits for or sits idle in", () => {
    const t = byOps();
    expect(dieAt(t, 0, 1)).toMatchObject({ op: { name: "q_proj" }, halted: false });
    expect(dieAt(t, 0, 6)).toMatchObject({ stalled: true, halted: false });
    expect(dieAt(t, 1, 1)).toMatchObject({ halted: true });
    expect(dieAt(t, 0, 8).halted).toBe(true);
    expect(dieTotals(t, 0)).toEqual({ work: 5, stalled: 2, halted: 3 });
  });

  it("steps from one operation to the next and back", () => {
    const t = byOps();
    expect(nextStop(t, 0, "op", 1)).toBe(3);
    expect(nextStop(t, 3, "op", 1)).toBe(5);
    expect(nextStop(t, 5, "op", -1)).toBe(3);
  });

  it("stops where an operation of a kind starts", () => {
    expect(firstBreak(byOps(), 0, 9, { ...NO_BREAKPOINTS, kind: "KvAppend" })).toEqual({ cycle: 3, reason: "Die 0 starts kv" });
  });
});

describe("where the cycles go", () => {
  it("counts each die's cycles by kind of operation, waiting and idle", () => {
    const t = sample();
    expect(Object.fromEntries(cyclesByKind(t, 0))).toEqual({ Matvec: 3, KvAppend: 4, waiting: 1, idle: 2 });
    expect(Object.fromEntries(cyclesByKind(byOps(), 1))).toEqual({ KvAppend: 4, idle: 6 });
    expect(Object.fromEntries(cyclesByKind(t, 0, 3))).toEqual({ Matvec: 3 });
  });

  it("gives each die's share of work so far", () => {
    const [a, b] = useShare(sample(), 9);
    expect(a).toBeCloseTo(0.7);
    expect(b).toBeCloseTo(0.2);
  });

  it("sets two traces' routines and shares side by side", () => {
    const c = compareTraces(sample(), byOps());
    expect(c.steps.map((s) => [s.label, s.a, s.b])).toEqual([["token 0", 8, 8], ["logits 0", 2, 2]]);
    expect(c.share.a[0] + c.share.a[1] + c.share.a[2]).toBeCloseTo(1);
  });
});

describe("the scene's layout", () => {
  it("gives every lane, cell and cache slot a place, the dies apart round the ring", () => {
    const layout = layoutScene(sample());
    expect(layout.dies.map((d) => d.lanes.length)).toEqual([8, 8]);
    expect(layout.dies[0].cells).toHaveLength(8);
    expect(layout.dies[0].cache).toHaveLength(1);
    expect(layout.dies[0].cache[0]).toHaveLength(4);
    const [a, b] = layout.dies.map((d) => d.center);
    expect(Math.hypot(a.x - b.x, a.z - b.z)).toBeGreaterThan(layout.dies[0].width);
  });

  it("starts and ends a link at its dies", () => {
    const a = { x: 0, y: 0, z: 0 };
    const b = { x: 4, y: 0, z: 2 };
    expect(linkPoint(a, b, 0, 0, false)).toMatchObject({ x: 0, z: 0 });
    expect(linkPoint(a, b, 1, 0, false)).toMatchObject({ x: 4, z: 2 });
  });
});
