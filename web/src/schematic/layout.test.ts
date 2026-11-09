import { describe, expect, it } from "vitest";
import type { Bit, Design, Instance, NetlistModule } from "../engine";
import { layoutModule } from "./layout";

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
