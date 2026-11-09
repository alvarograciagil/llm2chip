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
  kind: "instance" | "input" | "output" | "area";
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
}

export interface Edge {
  id: string;
  from: { node: string; port: string };
  to: { node: string; port: string };
  bits: Bit[];
  net: string;
  points: [number, number][];
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
};
const PRIM_OUTPUT: Record<PrimType, string> = { and2: "y", or2: "y", xor2: "y", not: "y", mux2: "y", dff: "q", sram: "rdata", ram2p: "rdata" };

const PIN_STEP = 18;
const BOX_W = 164;
const GATE = { w: 56, h: 42 };
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
  if (inst.kind.kind === "prim" && inst.kind.of.type !== "sram" && inst.kind.of.type !== "ram2p") {
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
  const title = isSram ? "Memoria SRAM" : design.modules[(inst.kind as { of: string }).of].title;
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
    for (const bit of bits) {
      if (typeof bit === "string") continue;
      const from = drivers.get(bitKey(bit));
      if (!from) continue;
      const key = `${from.node}.${from.port}>${node}.${port}`;
      const link = links.get(key) ?? { from, to: { node, port }, bits: [], net: module.nets[bit.net[0]].name };
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
  "elk.layered.mergeEdges": "true",
  "elk.layered.nodePlacement.strategy": "NETWORK_SIMPLEX",
  "elk.layered.spacing.nodeNodeBetweenLayers": "40",
  "elk.layered.spacing.edgeNodeBetweenLayers": "20",
  "elk.layered.spacing.edgeEdgeBetweenLayers": "11",
  "elk.spacing.nodeNode": "22",
  "elk.spacing.edgeNode": "16",
  "elk.spacing.edgeEdge": "11",
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
    layoutOptions: { "elk.portConstraints": "FIXED_POS" },
    ports: node.pins.map((pin) => ({ id: portId(node.id, pin.name), x: pin.x, y: pin.y, width: 0, height: 0, layoutOptions: { "elk.port.side": pin.side === "left" ? "WEST" : "EAST" } })),
  };
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
    return { id: `${index}`, ...link, points, back: points[points.length - 1][0] <= points[0][0], driver: `${link.from.node}.${link.from.port}` };
  });
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
    const entry = merged.get(key) ?? { from: { node: from, port: "out" }, to: { node: to, port: "in" }, bits: [], net: link.net };
    entry.bits.push(...link.bits);
    merged.set(key, entry);
  }

  const nodes = new Map<string, Node>();
  const AREA_W = 200;
  const AREA_H = 64;
  for (const name of new Set(module.instances.map((i) => i.group))) {
    const count = module.instances.filter((i) => i.group === name).length;
    nodes.set(`area:${name}`, {
      id: `area:${name}`,
      kind: "area",
      title: name,
      subtitle: `${count} ${count === 1 ? "elemento" : "elementos"}`,
      group: "",
      x: 0,
      y: 0,
      w: AREA_W,
      h: AREA_H,
      pins: [
        { name: "in", width: 2, x: 0, y: AREA_H / 2, side: "left" },
        { name: "out", width: 2, x: AREA_W, y: AREA_H / 2, side: "right" },
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
  return place(nodes, collectLinks(module, nodes), showGroups);
}
