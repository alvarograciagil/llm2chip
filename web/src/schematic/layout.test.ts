import { describe, expect, it } from "vitest";
import type { Bit, Design, Instance, NetlistModule } from "../engine";
import { fitCamera, layoutModule, valuePlace, type Layout, type Node, type Pin } from "./layout";

const bit = (net: number, index = 0): Bit => ({ net: [net, index] });

function gate(name: string, a: Bit, b: Bit, y: Bit): Instance {
  return { name, group: "", kind: { kind: "prim", of: { type: "and2" } }, conns: { a: [a], b: [b], y: [y] } };
}

function overlaps(points: [number, number][][]) {
  const segments = points.flatMap((line, owner) => line.slice(1).map((end, i) => ({ owner, a: line[i], b: end })));
  let total = 0;
  for (const s of segments) {
    for (const q of segments) {
      if (s.owner >= q.owner) continue;
      const horizontal = s.a[1] === s.b[1] && q.a[1] === q.b[1] && s.a[1] === q.a[1];
      const vertical = s.a[0] === s.b[0] && q.a[0] === q.b[0] && s.a[0] === q.a[0];
      const axis = horizontal ? 0 : 1;
      if (!horizontal && !vertical) continue;
      const shared = Math.min(Math.max(s.a[axis], s.b[axis]), Math.max(q.a[axis], q.b[axis])) - Math.max(Math.min(s.a[axis], s.b[axis]), Math.min(q.a[axis], q.b[axis]));
      if (shared > 1) total++;
    }
  }
  return total;
}

function design(module: NetlistModule): Design {
  return { top: module.name, modules: { [module.name]: module } };
}

describe("layoutModule", () => {
  it("ends the wires of a bus fed from several places at a join, so none overlap", async () => {
    const module: NetlistModule = {
      name: "m",
      title: "M",
      description: "",
      nets: [{ name: "p", width: 1 }, { name: "q", width: 1 }, { name: "r", width: 1 }, { name: "bus", width: 2 }, { name: "out", width: 2 }],
      ports: [{ name: "p", dir: "input", net: 0 }, { name: "q", dir: "input", net: 1 }, { name: "r", dir: "input", net: 2 }, { name: "out", dir: "output", net: 4 }],
      instances: [
        gate("g0", bit(0), bit(1), bit(3, 0)),
        gate("g1", bit(1), bit(2), bit(3, 1)),
        { name: "reg", group: "", kind: { kind: "prim", of: { type: "sram", addr_bits: 1, width: 2 } }, conns: { addr: [bit(0)], we: [bit(1)], wdata: [bit(3, 0), bit(3, 1)], rdata: [bit(4, 0), bit(4, 1)] } },
      ],
    };
    const layout = await layoutModule(module, design(module));
    const join = layout.nodes.find((n) => n.kind === "join");
    expect(join?.pins.filter((p) => p.side === "left")).toHaveLength(2);
    const separate = layout.edges.filter((e) => e.to.node === join?.id).map((e) => e.points);
    expect(overlaps(separate)).toBe(0);
  });

  it("gives every wire a drawable path", async () => {
    const module: NetlistModule = {
      name: "m",
      title: "M",
      description: "",
      nets: [{ name: "a", width: 1 }, { name: "b", width: 1 }, { name: "x", width: 1 }, { name: "y", width: 1 }],
      ports: [{ name: "a", dir: "input", net: 0 }, { name: "b", dir: "input", net: 1 }, { name: "x", dir: "output", net: 2 }, { name: "y", dir: "output", net: 3 }],
      instances: [gate("g0", bit(1), bit(1), bit(2)), gate("g1", bit(0), bit(0), bit(3))],
    };
    const layout = await layoutModule(module, design(module));
    for (const edge of layout.edges) expect(edge.d.startsWith("M")).toBe(true);
  });
});

describe("the start of the camera", () => {
  const block = (id: string, x: number, y: number, w = 160, h = 80): Node => ({ id, kind: "instance", title: id, group: "", x, y, w, h, pins: [] });
  const drawing = (nodes: Node[], width: number, height: number): Layout => ({ nodes, edges: [], groups: [], junctions: [], width, height });

  it("shows the whole drawing, centred, when it fits", () => {
    const camera = fitCamera(drawing([block("a", 0, 0)], 400, 200), 1000, 800, 0.7)!;
    expect(camera.k).toBe(1.4);
    expect(camera.x).toBeCloseTo((1000 - 400 * 1.4) / 2);
  });

  it("starts at the top left of a drawing too large to read whole", () => {
    const camera = fitCamera(drawing([block("a", 20, 20), block("b", 3000, 5000)], 3200, 5100), 1000, 800, 0.7)!;
    expect(camera).toEqual({ x: 24, y: 62, k: 0.7 });
  });

  it("starts at the middle when the top left of such a drawing is empty", () => {
    // A tall block of gates down the middle, as the sequencer lays out.
    const gates = Array.from({ length: 40 }, (_, i) => block(`g${i}`, 1500, 200 * i));
    const camera = fitCamera(drawing(gates, 2300, 8400), 1000, 900, 0.7)!;
    const middle = gates.reduce((a, b) => (Math.abs(b.y + 40 - 4200) < Math.abs(a.y + 40 - 4200) ? b : a));
    // The block nearest the middle of the drawing lands in the middle of the stage.
    expect((middle.x + 80) * camera.k + camera.x).toBeCloseTo(500);
    expect((middle.y + 40) * camera.k + camera.y).toBeCloseTo(450);
  });
});

describe("the values inside a box", () => {
  const pin = (name: string, side: "left" | "right", y: number, width = 16): Pin => ({ name, width, x: side === "left" ? 0 : 164, y, side });
  const box = (pins: Pin[]): Node => ({ id: "r", kind: "instance", title: "Register", group: "", x: 0, y: 0, w: 164, h: 80, pins });

  it("sit in the middle when a row has one", () => {
    const node = box([pin("d", "left", 50), pin("en", "left", 70, 1), pin("q", "right", 70)]);
    expect(valuePlace(node, node.pins[0])).toEqual({ x: 82, anchor: "middle" });
    expect(valuePlace(node, node.pins[2])).toEqual({ x: 82, anchor: "middle" });
  });

  it("take a half each when both sides of a row have one", () => {
    const node = box([pin("d", "left", 50), pin("q", "right", 50)]);
    expect(valuePlace(node, node.pins[0])).toEqual({ x: 78, anchor: "end" });
    expect(valuePlace(node, node.pins[1])).toEqual({ x: 86, anchor: "start" });
  });
});
