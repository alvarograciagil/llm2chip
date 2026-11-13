import ELK, { type ElkExtendedEdge, type ElkNode } from "elkjs/lib/elk.bundled.js";
import type { Bit, Design, Instance, NetlistModule, PrimType } from "../engine";

export interface Pin {
  name: string;
  width: number;
  x: number;
  y: number;
  side: "left" | "right";
}

export interface Node {
  id: string;
  kind: "instance" | "input" | "output" | "area" | "join";
  instance?: Instance;
  prim?: PrimType;
  title: string;
  subtitle?: string;
  group: string;
  x: number;
  y: number;
  w: number;
  h: number;
  pins: Pin[];
  /** The layout may reorder this node's pins along each side to avoid crossings. */
  freePins?: boolean;
}

export interface Edge {
  id: string;
  from: { node: string; port: string };
  to: { node: string; port: string };
  bits: Bit[];
  net: string;
  points: [number, number][];
  /** SVG path of the wire, with a small jump where it crosses another net. */
  d: string;
  back: boolean;
  /** Edges of the same net share this key. */
  driver: string;
}

export interface GroupFrame {
  name: string;
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface Layout {
  nodes: Node[];
  edges: Edge[];
  groups: GroupFrame[];
  junctions: [number, number][];
  width: number;
  height: number;
}

const PRIM_INPUTS: Record<PrimType, string[]> = {
  and2: ["a", "b"],
  or2: ["a", "b"],
  xor2: ["a", "b"],
  not: ["a"],
  mux2: ["s", "a", "b"],
  dff: ["d"],
  sram: ["addr", "we", "wdata"],
  ram2p: ["raddr", "waddr", "we", "wdata"],
  rom: ["addr"],
};
const PRIM_OUTPUT: Record<PrimType, string> = { and2: "y", or2: "y", xor2: "y", not: "y", mux2: "y", dff: "q", sram: "rdata", ram2p: "rdata", rom: "rdata" };

const PIN_STEP = 18;
const BOX_W = 164;
const GATE = { w: 56, h: 42 };
const JOIN_W = 6;
const JOIN_STEP = 12;
const JUMP_RADIUS = 4;
/** Beyond this many crossings a view is too dense for jumps to help. */
const MAX_JUMPS = 4000;
const PORT_W = 136;
const PORT_H = 24;

function pinsOf(inst: Instance, design: Design): { inputs: [string, number][]; outputs: [string, number][] } {
  const width = (port: string) => inst.conns[port]?.length ?? 1;
  if (inst.kind.kind === "prim") {
    const type = inst.kind.of.type;
    return { inputs: PRIM_INPUTS[type].map((p) => [p, width(p)]), outputs: [[PRIM_OUTPUT[type], width(PRIM_OUTPUT[type])]] };
  }
  const child = design.modules[inst.kind.of];
  const list = (dir: "input" | "output"): [string, number][] => child.ports.filter((p) => p.dir === dir).map((p) => [p.name, child.nets[p.net].width]);
  return { inputs: list("input"), outputs: list("output") };
}

function makeNode(inst: Instance, design: Design): Node {
  const { inputs, outputs } = pinsOf(inst, design);
  if (inst.kind.kind === "prim" && inst.kind.of.type !== "sram" && inst.kind.of.type !== "ram2p" && inst.kind.of.type !== "rom") {
    const type = inst.kind.of.type;
    const h = type === "mux2" ? 60 : GATE.h;
    const step = h / (inputs.length + 1);
    return {
      id: inst.name,
      kind: "instance",
      instance: inst,
      prim: type,
      title: inst.name,
      group: inst.group,
      x: 0,
      y: 0,
      w: GATE.w,
      h,
      pins: [
        ...inputs.map(([name, width], i): Pin => ({ name, width, x: 0, y: step * (i + 1), side: "left" })),
        ...outputs.map(([name, width]): Pin => ({ name, width, x: GATE.w, y: h / 2, side: "right" })),
      ],
    };
  }
  const rows = Math.max(inputs.length, outputs.length, 1);
  const h = 40 + rows * PIN_STEP + 6;
  const isSram = inst.kind.kind === "prim";
  const title = isSram ? ((inst.kind.of as { type: PrimType }).type === "rom" ? "ROM" : "SRAM") : design.modules[(inst.kind as { of: string }).of].title;
  const prim = inst.kind.kind === "prim" ? inst.kind.of : null;
  return {
    id: inst.name,
    kind: "instance",
    instance: inst,
    prim: isSram ? (inst.kind.of as { type: PrimType }).type : undefined,
    title,
    subtitle: prim ? `${2 ** (prim.addr_bits ?? 0)} × ${prim.width} bits` : inst.name,
    group: inst.group,
    x: 0,
    y: 0,
    w: BOX_W,
    h,
    pins: [
      ...inputs.map(([name, width], i): Pin => ({ name, width, x: 0, y: 40 + PIN_STEP * (i + 0.5), side: "left" })),
      ...outputs.map(([name, width], i): Pin => ({ name, width, x: BOX_W, y: 40 + PIN_STEP * (i + 0.5), side: "right" })),
    ],
  };
}

const bitKey = (b: Bit) => (typeof b === "string" ? b : `${b.net[0]}:${b.net[1]}`);

interface Link {
  from: { node: string; port: string };
  to: { node: string; port: string };
  bits: Bit[];
  net: string;
  /** Everything the sink pin receives, in pin order, from every driver. */
  whole: Bit[];
}

function collectLinks(module: NetlistModule, nodes: Map<string, Node>): Link[] {
  const drivers = new Map<string, { node: string; port: string }>();
  for (const port of module.ports.filter((p) => p.dir === "input")) {
    const width = module.nets[port.net].width;
    for (let bit = 0; bit < width; bit++) drivers.set(`${port.net}:${bit}`, { node: `in:${port.name}`, port: port.name });
  }
  for (const inst of module.instances) {
    const node = nodes.get(inst.name)!;
    for (const pin of node.pins.filter((p) => p.side === "right")) {
      for (const bit of inst.conns[pin.name] ?? []) {
        if (typeof bit !== "string") drivers.set(bitKey(bit), { node: inst.name, port: pin.name });
      }
    }
  }

  const links = new Map<string, Link>();
  const addSink = (node: string, port: string, bits: Bit[]) => {
    const whole = bits.filter((b) => typeof b !== "string");
    for (const bit of bits) {
      if (typeof bit === "string") continue;
      const from = drivers.get(bitKey(bit));
      if (!from) continue;
      const key = `${from.node}.${from.port}>${node}.${port}`;
      const link = links.get(key) ?? { from, to: { node, port }, bits: [], net: module.nets[bit.net[0]].name, whole };
      link.bits.push(bit);
      links.set(key, link);
    }
  };
  for (const inst of module.instances) {
    const node = nodes.get(inst.name)!;
    for (const pin of node.pins.filter((p) => p.side === "left")) addSink(inst.name, pin.name, inst.conns[pin.name] ?? []);
  }
  for (const port of module.ports.filter((p) => p.dir === "output")) {
    const width = module.nets[port.net].width;
    const bits: Bit[] = Array.from({ length: width }, (_, bit) => ({ net: [port.net, bit] }));
    addSink(`out:${port.name}`, port.name, bits);
  }
  return [...links.values()];
}

/**
 * A pin fed from several places would make their wires run on top of each other, so
 * those wires end at a small bus join, whose single output goes to the pin.
 */
function addJoins(links: Link[], nodes: Map<string, Node>): Link[] {
  const bySink = new Map<string, Link[]>();
  for (const link of links) {
    const key = `${link.to.node}.${link.to.port}`;
    bySink.set(key, [...(bySink.get(key) ?? []), link]);
  }
  const out: Link[] = [];
  for (const [key, list] of bySink) {
    if (list.length < 2) {
      out.push(list[0]);
      continue;
    }
    const sink = list[0].to;
    const id = `join:${key}`;
    const h = Math.max(16, list.length * JOIN_STEP);
    nodes.set(id, {
      id,
      kind: "join",
      title: "",
      group: nodes.get(sink.node)!.group,
      x: 0,
      y: 0,
      w: JOIN_W,
      h,
      freePins: true,
      pins: [
        ...list.map((l, i): Pin => ({ name: `i${i}`, width: l.bits.length, x: 0, y: 0, side: "left" })),
        { name: "y", width: list[0].whole.length, x: JOIN_W, y: h / 2, side: "right" },
      ],
    });
    list.forEach((l, i) => out.push({ ...l, to: { node: id, port: `i${i}` } }));
    out.push({ from: { node: id, port: "y" }, to: sink, bits: list[0].whole, net: list[0].net, whole: list[0].whole });
  }
  return out;
}

const pinOf = (node: Node, port: string) => node.pins.find((p) => p.name === port)!;

/** Above this many parts, a module is first shown as its functional areas. */
export const AREA_VIEW_THRESHOLD = 30;

const elk = new ELK();

const ELK_OPTIONS = {
  "elk.algorithm": "layered",
  "elk.direction": "RIGHT",
  "elk.edgeRouting": "ORTHOGONAL",
  "elk.hierarchyHandling": "INCLUDE_CHILDREN",
  "elk.json.shapeCoords": "ROOT",
  "elk.json.edgeCoords": "ROOT",
  "elk.layered.crossingMinimization.thoroughness": "60",
  "elk.layered.mergeEdges": "true",
  "elk.layered.nodePlacement.strategy": "NETWORK_SIMPLEX",
  "elk.layered.spacing.nodeNodeBetweenLayers": "40",
  "elk.layered.spacing.edgeNodeBetweenLayers": "20",
  "elk.layered.spacing.edgeEdgeBetweenLayers": "13",
  "elk.spacing.nodeNode": "22",
  "elk.spacing.edgeNode": "16",
  "elk.spacing.edgeEdge": "13",
  "elk.padding": "[top=16,left=16,bottom=16,right=16]",
};

function portNode(name: string, width: number, kind: "input" | "output"): Node {
  return {
    id: `${kind === "input" ? "in" : "out"}:${name}`,
    kind,
    title: name,
    subtitle: width > 1 ? `${width} bits` : undefined,
    group: "",
    x: 0,
    y: 0,
    w: PORT_W,
    h: PORT_H,
    pins: [{ name, width, x: kind === "input" ? PORT_W : 0, y: PORT_H / 2, side: kind === "input" ? "right" : "left" }],
  };
}

const portId = (node: string, port: string) => `${node}\u0000${port}`;

function elkNode(node: Node): ElkNode {
  return {
    id: node.id,
    width: node.w,
    height: node.h,
    layoutOptions: { "elk.portConstraints": node.freePins ? "FIXED_SIDE" : "FIXED_POS" },
    ports: node.pins.map((pin) => ({ id: portId(node.id, pin.name), x: pin.x, y: pin.y, width: 0, height: 0, layoutOptions: { "elk.port.side": pin.side === "left" ? "WEST" : "EAST" } })),
  };
}

/** Fills in each wire's path, jumping over the wires of other nets that it crosses. */
function drawWires(edges: Edge[]) {
  const verticals: { x: number; y1: number; y2: number; driver: string }[] = [];
  for (const edge of edges) {
    edge.points.slice(1).forEach(([x, y], i) => {
      const [px, py] = edge.points[i];
      if (px === x && py !== y) verticals.push({ x, y1: Math.min(py, y), y2: Math.max(py, y), driver: edge.driver });
    });
  }
  verticals.sort((a, b) => a.x - b.x);
  const firstFrom = (x: number) => {
    let [low, high] = [0, verticals.length];
    while (low < high) {
      const middle = (low + high) >> 1;
      if (verticals[middle].x < x) low = middle + 1;
      else high = middle;
    }
    return low;
  };
  const crossings = (edge: Edge, x1: number, x2: number, y: number) => {
    const found: number[] = [];
    for (let v = firstFrom(x1 + JUMP_RADIUS); v < verticals.length && verticals[v].x <= x2 - JUMP_RADIUS; v++) {
      const { x, y1, y2, driver } = verticals[v];
      if (driver !== edge.driver && y1 < y && y < y2 && found[found.length - 1] !== x) found.push(x);
    }
    return found;
  };

  let total = 0;
  for (const edge of edges) {
    edge.points.slice(1).forEach(([x, y], i) => {
      const [px, py] = edge.points[i];
      if (py === y) total += crossings(edge, Math.min(px, x), Math.max(px, x), y).length;
    });
  }
  const jump = total <= MAX_JUMPS;
  for (const edge of edges) {
    let d = `M${edge.points[0][0]} ${edge.points[0][1]}`;
    edge.points.slice(1).forEach(([x, y], i) => {
      const [px, py] = edge.points[i];
      if (jump && py === y && px !== x) {
        const right = x > px;
        const hops = crossings(edge, Math.min(px, x), Math.max(px, x), y);
        let last = -Infinity;
        for (const hop of right ? hops : [...hops].reverse()) {
          if (Math.abs(hop - last) < 2 * JUMP_RADIUS + 1) continue;
          const [before, after] = right ? [hop - JUMP_RADIUS, hop + JUMP_RADIUS] : [hop + JUMP_RADIUS, hop - JUMP_RADIUS];
          d += ` L${before} ${y} A${JUMP_RADIUS} ${JUMP_RADIUS} 0 0 ${right ? 1 : 0} ${after} ${y}`;
          last = hop;
        }
      }
      d += ` L${x} ${y}`;
    });
    edge.d = d;
  }
}

/** Places the nodes and routes the links; groups become labelled frames. */
async function place(nodes: Map<string, Node>, links: Link[], groups: boolean): Promise<Layout> {
  const members = new Map<string, Node[]>();
  const loose: Node[] = [];
  for (const node of nodes.values()) {
    if (groups && node.group) members.set(node.group, [...(members.get(node.group) ?? []), node]);
    else loose.push(node);
  }
  const children: ElkNode[] = [
    ...loose.map(elkNode),
    ...[...members.entries()].map(([name, list]) => ({ id: `group:${name}`, layoutOptions: { "elk.padding": "[top=30,left=14,bottom=14,right=14]" }, children: list.map(elkNode) })),
  ];
  const edges = links.map((link, index) => ({ id: `${index}`, sources: [portId(link.from.node, link.from.port)], targets: [portId(link.to.node, link.to.port)] }));
  const root: ElkNode = { id: "root", layoutOptions: ELK_OPTIONS, children, edges };
  const result: ElkNode = await elk.layout(root);

  const frames: GroupFrame[] = [];
  const visit = (list: ElkNode[]) => {
    for (const child of list) {
      if (child.id.startsWith("group:")) {
        frames.push({ name: child.id.slice(6), x: child.x ?? 0, y: child.y ?? 0, w: child.width ?? 0, h: child.height ?? 0 });
        visit(child.children ?? []);
      } else {
        const node = nodes.get(child.id)!;
        node.x = child.x ?? 0;
        node.y = child.y ?? 0;
        if (node.freePins) {
          for (const port of child.ports ?? []) {
            const pin = node.pins.find((p) => portId(node.id, p.name) === port.id)!;
            pin.x = (port.x ?? 0) - node.x;
            pin.y = (port.y ?? 0) - node.y;
          }
        }
      }
    }
  };
  visit(result.children ?? []);

  const junctions = new Map<string, [number, number]>();
  const routed: Edge[] = (result.edges as ElkExtendedEdge[]).map((edge, index) => {
    const link = links[index];
    const section = edge.sections?.[0];
    const a = nodes.get(link.from.node)!;
    const b = nodes.get(link.to.node)!;
    const start = pinOf(a, link.from.port);
    const end = pinOf(b, link.to.port);
    const points: [number, number][] = section
      ? [section.startPoint, ...(section.bendPoints ?? []), section.endPoint].map((p) => [p.x, p.y])
      : [[a.x + start.x, a.y + start.y], [b.x + end.x, b.y + end.y]];
    for (const j of (edge as ElkExtendedEdge & { junctionPoints?: { x: number; y: number }[] }).junctionPoints ?? []) junctions.set(`${j.x},${j.y}`, [j.x, j.y]);
    return { id: `${index}`, ...link, points, d: "", back: points[points.length - 1][0] <= points[0][0], driver: `${link.from.node}.${link.from.port}` };
  });
  drawWires(routed);
  return { nodes: [...nodes.values()], edges: routed, groups: frames, junctions: [...junctions.values()], width: result.width ?? 0, height: result.height ?? 0 };
}

/** One box per functional area, joined by the buses that cross between them. */
export function layoutAreas(module: NetlistModule, design: Design): Promise<Layout> {
  const inner = new Map<string, Node>();
  for (const inst of module.instances) inner.set(inst.name, makeNode(inst, design));
  for (const p of module.ports) {
    const node = portNode(p.name, module.nets[p.net].width, p.dir);
    inner.set(node.id, node);
  }
  const groupOf = new Map(module.instances.map((i) => [i.name, i.group]));
  const areaId = (node: string) => (groupOf.has(node) ? `area:${groupOf.get(node)}` : node);

  const merged = new Map<string, Link>();
  for (const link of collectLinks(module, inner)) {
    const from = areaId(link.from.node);
    const to = areaId(link.to.node);
    if (from === to) continue;
    const key = `${from}>${to}`;
    const entry = merged.get(key) ?? { from: { node: from, port: "out" }, to: { node: to, port: "in" }, bits: [], net: link.net, whole: [] };
    entry.bits.push(...link.bits);
    entry.whole = entry.bits;
    merged.set(key, entry);
  }
  // Each pair of areas gets its own pins, so that no two connections share a wire.
  for (const link of merged.values()) {
    link.from.port = link.from.node.startsWith("area:") ? `to ${link.to.node}` : "out";
    link.to.port = link.to.node.startsWith("area:") ? `from ${link.from.node}` : "in";
  }

  const nodes = new Map<string, Node>();
  const AREA_W = 200;
  const AREA_H = 64;
  const AREA_PIN_STEP = 16;
  for (const name of new Set(module.instances.map((i) => i.group))) {
    const count = module.instances.filter((i) => i.group === name).length;
    const id = `area:${name}`;
    const links = [...merged.values()];
    const inputs = links.filter((l) => l.to.node === id);
    const outputs = links.filter((l) => l.from.node === id);
    const h = Math.max(AREA_H, Math.max(inputs.length, outputs.length) * AREA_PIN_STEP + 12);
    nodes.set(`area:${name}`, {
      id: `area:${name}`,
      kind: "area",
      title: name,
      subtitle: `${count} ${count === 1 ? "elemento" : "elementos"}`,
      group: "",
      x: 0,
      y: 0,
      w: AREA_W,
      h,
      freePins: true,
      pins: [
        ...inputs.map((l): Pin => ({ name: l.to.port, width: l.bits.length, x: 0, y: 0, side: "left" })),
        ...outputs.map((l): Pin => ({ name: l.from.port, width: l.bits.length, x: AREA_W, y: 0, side: "right" })),
      ],
    });
  }
  for (const [id, node] of inner) {
    if (id.startsWith("in:") || id.startsWith("out:")) nodes.set(id, { ...node, pins: node.pins.map((p) => ({ ...p, name: id.startsWith("in:") ? "out" : "in" })) });
  }
  return place(nodes, [...merged.values()], false);
}

export function layoutModule(fullModule: NetlistModule, design: Design, area: string | null = null): Promise<Layout> {
  const module = area ? { ...fullModule, instances: fullModule.instances.filter((i) => i.group === area), ports: [] } : fullModule;
  const nodes = new Map<string, Node>();
  for (const inst of module.instances) nodes.set(inst.name, makeNode(inst, design));
  for (const p of module.ports) {
    const node = portNode(p.name, module.nets[p.net].width, p.dir);
    nodes.set(node.id, node);
  }
  const showGroups = new Set(module.instances.map((i) => i.group)).size > 1;
  return place(nodes, addJoins(collectLinks(module, nodes), nodes), showGroups);
}
