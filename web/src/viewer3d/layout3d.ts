// Where each piece of the chips goes in the 3D scene, worked out without Three.js so it
// can be tested: the dies around their ring, the units and cells on each die, and the
// columns of the KV cache beside it.

import type { Trace } from "./trace";
import { cacheOps } from "./trace";

export interface Vec3 {
  x: number;
  y: number;
  z: number;
}

export interface DieLayout {
  center: Vec3;
  /** Width and depth of the die's slab. */
  width: number;
  depth: number;
  /** The centre of every unit's lanes, unit by unit, lane by lane. */
  lanes: Vec3[];
  /** The centre of every cell. */
  cells: Vec3[];
  /** For each cache operation, the centre of each of its slots, from the first position. */
  cache: Vec3[][];
}

export interface SceneLayout {
  dies: DieLayout[];
  /** The ring's radius, to frame it. */
  radius: number;
  lane: number;
}

/** Size of one lane's cube and the gap around it. */
export const LANE = 0.32;
const GAP = 0.08;

/** Columns and rows of a grid of `n` items, wider than deep. */
function grid(n: number): [number, number] {
  const columns = Math.ceil(Math.sqrt(n));
  return [columns, Math.ceil(n / columns)];
}

export function layoutScene(trace: Trace): SceneLayout {
  const caches = cacheOps(trace);
  const sizes = trace.dies.map((d) => {
    const [unitColumns, unitRows] = grid(d.units);
    const [laneColumns, laneRows] = grid(d.lanes);
    const unitWidth = laneColumns * (LANE + GAP) + GAP * 2;
    const unitDepth = laneRows * (LANE + GAP) + GAP * 2;
    return { unitColumns, unitRows, laneColumns, laneRows, unitWidth, unitDepth, width: unitColumns * unitWidth + 1.2, depth: unitRows * unitDepth + 1.6 };
  });
  const widest = Math.max(...sizes.map((s) => Math.max(s.width, s.depth)), 1);
  const n = trace.dies.length;
  // Dies round a circle, in the order of the ring, far enough apart not to touch.
  const radius = n < 2 ? 0 : Math.max(widest * 1.25, (n * widest * 1.6) / (2 * Math.PI));
  const dies = trace.dies.map((d, i): DieLayout => {
    const angle = n < 2 ? 0 : (i / n) * Math.PI * 2 - Math.PI / 2;
    const center = { x: radius * Math.cos(angle), y: 0, z: radius * Math.sin(angle) };
    const s = sizes[i];
    const x0 = center.x - (s.unitColumns * s.unitWidth) / 2;
    const z0 = center.z - s.depth / 2 + 0.5;
    const lanes: Vec3[] = [];
    for (let u = 0; u < d.units; u++) {
      const ux = x0 + (u % s.unitColumns) * s.unitWidth + GAP * 2;
      const uz = z0 + Math.floor(u / s.unitColumns) * s.unitDepth + GAP * 2;
      for (let l = 0; l < d.lanes; l++) {
        lanes.push({ x: ux + (l % s.laneColumns) * (LANE + GAP) + LANE / 2, y: 0.25, z: uz + Math.floor(l / s.laneColumns) * (LANE + GAP) + LANE / 2 });
      }
    }
    // The cells in a row along the die's far edge.
    const cellStep = Math.min(0.3, (s.width - 0.6) / Math.max(d.cells, 1));
    const cells = Array.from({ length: d.cells }, (_, k) => ({ x: center.x - ((d.cells - 1) * cellStep) / 2 + k * cellStep, y: 0.18, z: center.z + s.depth / 2 - 0.35 }));
    // The cache: a column per attention layer standing at the die's left, a slot per token.
    const cache = caches.map((_, k) => Array.from({ length: trace.context }, (_, p) => ({ x: center.x - s.width / 2 - 0.35 - k * 0.4, y: 0.2 + p * 0.22, z: center.z })));
    return { center, width: s.width, depth: s.depth, lanes, cells, cache };
  });
  return { dies, radius, lane: LANE };
}

/** A point along the link from die `a` to die `b`: lanes of one link side by side, loops
 * back bowing towards the centre. `t` from 0 to 1. */
export function linkPoint(a: Vec3, b: Vec3, t: number, lane: number, inward: boolean): Vec3 {
  const mid = { x: (a.x + b.x) / 2, y: 0.6 + lane * 0.12, z: (a.z + b.z) / 2 };
  const bow = inward ? 0.45 : -0.15;
  const control = { x: mid.x * (1 - bow), y: mid.y + 0.8, z: mid.z * (1 - bow) };
  const u = 1 - t;
  return {
    x: u * u * a.x + 2 * u * t * control.x + t * t * b.x,
    y: u * u * 0.4 + 2 * u * t * control.y + t * t * 0.4,
    z: u * u * a.z + 2 * u * t * control.z + t * t * b.z,
  };
}
