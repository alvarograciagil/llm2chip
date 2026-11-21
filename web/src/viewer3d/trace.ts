// A trace of dies at work, as the engine records it gate by gate (crates/trace), and how
// to read it at any cycle: which word each die runs, which operation of the model that is,
// what every link carries and how far the KV cache has filled.

export type WordClass = "cells" | "vector" | "scalar" | "send" | "take" | "halt";

export interface OpInfo {
  kind: string;
  name: string;
  layer: number | null;
  slot: number | null;
}

export interface WordInfo {
  class: WordClass;
  op: number | null;
  text: string;
}

export interface DieInfo {
  units: number;
  lanes: number;
  cells: number;
  group: [number, number] | null;
  words: WordInfo[];
  entries: number[];
  /** The word running in every cycle, or -1 once the die has halted. */
  pcs: number[];
  /** `[start, length]` of the cycles the sequencer waited on a link. */
  stalls: [number, number][];
}

/** 1: a word crosses; 2: a word waits because the receiver cannot take it. */
export type LinkState = 0 | 1 | 2;

export interface LinkInfo {
  from: number;
  to: number;
  port: string;
  /** `[start, length, state]`; cycles left out are idle. */
  spans: [number, number, LinkState][];
}

export interface StepInfo {
  routine: "token" | "logits";
  token: number;
  position: number;
  start: number;
  cycles: number;
}

export interface KvWrite {
  cycle: number;
  die: number;
  op: number;
  slot: number;
  position: number;
}

export interface Trace {
  version: number;
  title: string;
  source: string;
  sharing: string;
  context: number;
  ops: OpInfo[];
  layers: number;
  dies: DieInfo[];
  links: LinkInfo[];
  steps: StepInfo[];
  kv: KvWrite[];
}

export const TRACE_VERSION = 1;

/** Reads a trace as the engine or a file gives it, or says why it cannot. */
export function readTrace(raw: unknown): Trace {
  const t = raw as Partial<Trace> | null;
  if (!t || typeof t !== "object") throw new Error("This is not a trace.");
  if (t.version !== TRACE_VERSION) throw new Error(`This trace is version ${String(t.version)}; this viewer reads version ${TRACE_VERSION}.`);
  if (!Array.isArray(t.dies) || !Array.isArray(t.steps) || !Array.isArray(t.ops)) throw new Error("This is not a trace.");
  return { ...(t as Trace), links: t.links ?? [], kv: t.kv ?? [] };
}

export function totalCycles(trace: Trace): number {
  return trace.steps.reduce((sum, s) => sum + s.cycles, 0);
}

/** The last element whose start is at or before `cycle`, by bisection over `start`. */
function lastStartingBy<T>(items: T[], cycle: number, start: (item: T) => number): number {
  let lo = 0;
  let hi = items.length - 1;
  let found = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (start(items[mid]) <= cycle) {
      found = mid;
      lo = mid + 1;
    } else hi = mid - 1;
  }
  return found;
}

/** The step running at `cycle`. */
export function stepAt(trace: Trace, cycle: number): number {
  return Math.max(0, lastStartingBy(trace.steps, cycle, (s) => s.start));
}

export interface DieState {
  /** The word running, or -1 when the die has halted. */
  pc: number;
  word: WordInfo | null;
  op: OpInfo | null;
  stalled: boolean;
  halted: boolean;
}

export function dieAt(trace: Trace, die: number, cycle: number): DieState {
  const d = trace.dies[die];
  const pc = d.pcs[Math.min(Math.max(0, Math.floor(cycle)), d.pcs.length - 1)] ?? -1;
  const word = pc >= 0 ? (d.words[pc] ?? null) : null;
  const op = word && word.op !== null ? (trace.ops[word.op] ?? null) : null;
  const s = lastStartingBy(d.stalls, cycle, (x) => x[0]);
  const stalled = s >= 0 && cycle < d.stalls[s][0] + d.stalls[s][1];
  return { pc, word, op, stalled, halted: pc < 0 };
}

export function linkAt(link: LinkInfo, cycle: number): LinkState {
  const s = lastStartingBy(link.spans, cycle, (x) => x[0]);
  return s >= 0 && cycle < link.spans[s][0] + link.spans[s][1] ? link.spans[s][2] : 0;
}

/** Cycles in `[from, to]` a word crossed the link or waited at it, by state. */
export function linkWordsIn(link: LinkInfo, from: number, to: number): { crossed: number; waited: number } {
  let crossed = 0;
  let waited = 0;
  // Spans are in order and do not overlap: start from the last one starting before `from`.
  for (let i = Math.max(0, lastStartingBy(link.spans, from, (x) => x[0])); i < link.spans.length; i++) {
    const [start, length, state] = link.spans[i];
    if (start > to) break;
    const overlap = Math.min(to, start + length - 1) - Math.max(from, start) + 1;
    if (overlap <= 0) continue;
    if (state === 1) crossed += overlap;
    else waited += overlap;
  }
  return { crossed, waited };
}

/** The dies that keep each attention operation's cache: those that write it at some point
 * of the trace. */
export function cacheHolders(trace: Trace): Map<number, Set<number>> {
  const holders = new Map<number, Set<number>>();
  for (const k of trace.kv) {
    const dies = holders.get(k.op) ?? new Set<number>();
    dies.add(k.die);
    holders.set(k.op, dies);
  }
  return holders;
}

/** For each die, the cache positions each attention operation has written by `cycle`, keyed
 * by the operation. */
export function cacheAt(trace: Trace, cycle: number): Map<number, Map<number, Set<number>>> {
  const filled = new Map<number, Map<number, Set<number>>>();
  for (const k of trace.kv) {
    if (k.cycle > cycle) break;
    const byOp = filled.get(k.die) ?? new Map<number, Set<number>>();
    const positions = byOp.get(k.op) ?? new Set<number>();
    positions.add(k.position);
    byOp.set(k.op, positions);
    filled.set(k.die, byOp);
  }
  return filled;
}

/** The attention operations that write a cache, in program order. */
export function cacheOps(trace: Trace): number[] {
  return trace.ops.flatMap((o, i) => (o.kind === "KvAppend" ? [i] : []));
}

export type Stride = "cycle" | "word" | "op" | "layer" | "step";

/** What a die is doing at a cycle, for one stride: the cycles where it changes are where
 * stepping by that stride stops. */
function key(trace: Trace, die: number, cycle: number, stride: Stride): string {
  const s = dieAt(trace, die, cycle);
  switch (stride) {
    case "word":
      return `${s.pc}`;
    case "op":
      return `${s.word?.op ?? "-"}`;
    case "layer":
      return `${s.op?.layer ?? "-"}`;
    default:
      return "";
  }
}

/** The next cycle after `cycle` (or before it, going back) where any die changes what it
 * does at `stride`. `die` limits it to one die. */
export function nextStop(trace: Trace, cycle: number, stride: Stride, direction: 1 | -1, die?: number): number {
  const total = totalCycles(trace);
  const here = Math.floor(cycle);
  if (stride === "cycle") return clamp(here + direction, 0, total - 1);
  if (stride === "step") {
    const i = stepAt(trace, here);
    if (direction > 0) return i + 1 < trace.steps.length ? trace.steps[i + 1].start : total - 1;
    return here > trace.steps[i].start ? trace.steps[i].start : trace.steps[Math.max(0, i - 1)].start;
  }
  const dies = die === undefined ? trace.dies.map((_, i) => i) : [die];
  const start = dies.map((d) => key(trace, d, here, stride));
  for (let c = here + direction; c >= 0 && c < total; c += direction) {
    if (dies.some((d, i) => key(trace, d, c, stride) !== start[i])) {
      if (direction > 0) return c;
      // Going back stops at the start of the run before.
      const before = dies.map((d) => key(trace, d, c, stride));
      while (c - 1 >= 0 && dies.every((d, i) => key(trace, d, c - 1, stride) === before[i])) c--;
      return c;
    }
  }
  return direction > 0 ? total - 1 : 0;
}

function clamp(v: number, lo: number, hi: number): number {
  return Math.min(Math.max(v, lo), hi);
}

/** How many cycles each die spent working, waiting on a link and halted. */
export function dieTotals(trace: Trace, die: number): { work: number; stalled: number; halted: number } {
  const d = trace.dies[die];
  const halted = d.pcs.filter((pc) => pc < 0).length;
  const stalled = d.stalls.reduce((sum, s) => sum + s[1], 0);
  return { work: d.pcs.length - halted - stalled, stalled, halted };
}
