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
const BOX_W = 176;
const GATE = { w: 56, h: 42 };
const COL_GAP = 62;
const ROW_GAP = 16;
const GROUP_PAD = 16;
const GROUP_TITLE = 24;
const GROUP_GAP = 52;
const PORT_W = 110;
const PORT_H = 24;
const ROW_SPACE = 58;
const TARGET_ASPECT = 1.5;

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

/** Ranks nodes by longest path from the inputs, skipping edges that close feedback loops. */
function rank(ids: string[], links: Link[]): { ranks: Map<string, number>; back: Set<Link> } {
  const out = new Map<string, Link[]>();
  for (const link of links) out.set(link.from.node, [...(out.get(link.from.node) ?? []), link]);

  const state = new Map<string, 1 | 2>();
  const back = new Set<Link>();
  const order: string[] = [];
  const visit = (start: string) => {
    const stack: { id: string; i: number }[] = [{ id: start, i: 0 }];
    state.set(start, 1);
    while (stack.length) {
      const top = stack[stack.length - 1];
      const edges = out.get(top.id) ?? [];
      if (top.i < edges.length) {
        const link = edges[top.i++];
        const seen = state.get(link.to.node);
        if (seen === 1) back.add(link);
        else if (!seen) {
          state.set(link.to.node, 1);
          stack.push({ id: link.to.node, i: 0 });
        }
      } else {
        state.set(top.id, 2);
        order.push(top.id);
        stack.pop();
      }
    }
  };
  for (const id of ids) if (!state.has(id)) visit(id);

  const ranks = new Map(ids.map((id) => [id, 0]));
  for (const id of order.reverse()) {
    for (const link of out.get(id) ?? []) {
      if (!back.has(link)) ranks.set(link.to.node, Math.max(ranks.get(link.to.node) ?? 0, (ranks.get(id) ?? 0) + 1));
    }
  }
  return { ranks, back };
}

const pinOf = (node: Node, port: string) => node.pins.find((p) => p.name === port)!;

/** Above this many parts, a module is first shown as its functional areas. */
export const AREA_VIEW_THRESHOLD = 30;

/** One box per functional area, joined by the buses that cross between them. */
export function layoutAreas(module: NetlistModule, design: Design): Layout {
  const inner = new Map<string, Node>();
  for (const inst of module.instances) inner.set(inst.name, makeNode(inst, design));
  for (const p of module.ports) {
    const prefix = p.dir === "input" ? "in" : "out";
    inner.set(`${prefix}:${p.name}`, { id: `${prefix}:${p.name}`, kind: p.dir, title: p.name, group: "", x: 0, y: 0, w: PORT_W, h: PORT_H, pins: [{ name: p.name, width: module.nets[p.net].width, x: p.dir === "input" ? PORT_W : 0, y: PORT_H / 2, side: p.dir === "input" ? "right" : "left" }] });
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
  const AREA_W = 190;
  const AREA_H = 64;
  for (const name of new Set(module.instances.map((i) => i.group))) {
    const count = module.instances.filter((i) => i.group === name).length;
    nodes.set(`area:${name}`, {
      id: `area:${name}`,
      kind: "area",
      title: name,
      subtitle: `${count} ${count === 1 ? "elemento" : "elementos"}`,
      group: name,
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

  const links = [...merged.values()];
  const { ranks, back } = rank([...nodes.keys()], links);
  const areas = [...nodes.values()].filter((n) => n.kind === "area");
  const columns = new Map<number, Node[]>();
  for (const area of areas) {
    const r = ranks.get(area.id) ?? 0;
    columns.set(r, [...(columns.get(r) ?? []), area]);
  }
  const sorted = [...columns.entries()].sort((a, b) => a[0] - b[0]).map(([, list]) => list);
  const COLUMN_GAP = 110;
  const ROW = AREA_H + 44;
  let x = PORT_W + COLUMN_GAP;
  const tallestColumn = Math.max(1, ...sorted.map((c) => c.length));
  const totalHeight = tallestColumn * ROW;
  for (const column of sorted) {
    column.forEach((node, i) => {
      node.x = x;
      node.y = (totalHeight - column.length * ROW) / 2 + i * ROW + 12;
    });
    x += AREA_W + COLUMN_GAP;
  }

  const place = (list: Node[], px: number) => {
    const step = PORT_H + 8;
    let y = Math.max(0, (totalHeight - list.length * step) / 2);
    for (const node of list) {
      node.x = px;
      node.y = y;
      y += step;
    }
  };
  place([...nodes.values()].filter((n) => n.kind === "input"), 0);
  place([...nodes.values()].filter((n) => n.kind === "output"), x);
  const width = x + PORT_W;

  let backLane = 0;
  const edges: Edge[] = links.map((link, index) => {
    const a = nodes.get(link.from.node)!;
    const b = nodes.get(link.to.node)!;
    const pa = pinOf(a, "out");
    const pb = pinOf(b, "in");
    const [x1, y1, x2, y2] = [a.x + pa.x, a.y + pa.y, b.x + pb.x, b.y + pb.y];
    const fan = ((index * 7) % 13) * 4;
    const isBack = back.has(link) || x2 <= x1 + 8;
    let points: [number, number][];
    if (isBack) {
      backLane += 1;
      const lane = totalHeight + 24 + backLane * 10;
      points = [[x1, y1], [x1 + 18 + fan, y1], [x1 + 18 + fan, lane], [x2 - 18 - fan, lane], [x2 - 18 - fan, y2], [x2, y2]];
    } else {
      const mx = Math.max(x1 + 14, x2 - 30 - fan);
      points = [[x1, y1], [mx, y1], [mx, y2], [x2, y2]];
    }
    return { id: `${index}`, ...link, points, back: isBack };
  });

  return { nodes: [...nodes.values()], edges, groups: [], width, height: totalHeight + 40 + backLane * 10 };
}

export function layoutModule(fullModule: NetlistModule, design: Design, area: string | null = null): Layout {
  const module = area ? { ...fullModule, instances: fullModule.instances.filter((i) => i.group === area), ports: [] } : fullModule;
  const nodes = new Map<string, Node>();
  for (const inst of module.instances) nodes.set(inst.name, makeNode(inst, design));

  const portNode = (name: string, width: number, kind: "input" | "output"): Node => ({
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
  });
  for (const p of module.ports) {
    const node = portNode(p.name, module.nets[p.net].width, p.dir);
    nodes.set(node.id, node);
  }

  const links = collectLinks(module, nodes);
  const { ranks, back } = rank([...nodes.keys()], links);

  const groupNames = [...new Set(module.instances.map((i) => i.group))];

  // Lay each area out on its own first, at the origin.
  const areas = groupNames.map((name) => {
    const members = module.instances.filter((i) => i.group === name).map((i) => nodes.get(i.name)!);
    const layerOf = (n: Node) => ranks.get(n.id) ?? 0;
    const layers = [...new Set(members.map(layerOf))].sort((a, b) => a - b);
    const columns = layers.map((l) => members.filter((n) => layerOf(n) === l));
    // A long chain reads better as a vertical stack than as one very wide row.
    const stacked = columns.length > 4 && columns.every((c) => c.length === 1);
    const lanes = stacked ? [members] : columns;

    let cx = GROUP_PAD;
    let height = 0;
    for (const lane of lanes) {
      let y = GROUP_TITLE;
      const laneW = Math.max(...lane.map((n) => n.w));
      for (const node of lane) {
        node.x = cx + (laneW - node.w) / 2;
        node.y = y;
        y += node.h + ROW_GAP;
      }
      height = Math.max(height, y - ROW_GAP);
      cx += laneW + COL_GAP;
    }
    return { name, members, w: cx - COL_GAP + GROUP_PAD, h: height + GROUP_PAD };
  });

  // Wrap areas onto rows so the diagram stays close to the shape of a window.
  const totalWidth = areas.reduce((sum, a) => sum + a.w + GROUP_GAP, 0);
  const averageHeight = areas.reduce((sum, a) => sum + a.h, 0) / Math.max(areas.length, 1);
  const rowCount = Math.min(areas.length || 1, Math.max(1, Math.round(Math.sqrt(totalWidth / (averageHeight * TARGET_ASPECT)))));
  const rowLimit = totalWidth / rowCount;

  const rows: (typeof areas)[] = [[]];
  let used = 0;
  for (const area of areas) {
    if (used > 0 && used + area.w / 2 > rowLimit && rows.length < rowCount) {
      rows.push([]);
      used = 0;
    }
    rows[rows.length - 1].push(area);
    used += area.w + GROUP_GAP;
  }

  const groups: GroupFrame[] = [];
  const rowOf = new Map<string, number>();
  const rowBottom: number[] = [];
  let top = 0;
  let width = 0;
  rows.forEach((row, rowIndex) => {
    const rowHeight = Math.max(...row.map((a) => a.h), 0);
    let x = PORT_W + GROUP_GAP;
    for (const area of row) {
      const y = top + (rowHeight - area.h) / 2;
      groups.push({ name: area.name, x, y, w: area.w, h: area.h });
      for (const node of area.members) {
        node.x += x;
        node.y += y;
        rowOf.set(node.id, rowIndex);
      }
      x += area.w + GROUP_GAP;
    }
    width = Math.max(width, x);
    top += rowHeight;
    rowBottom.push(top);
    top += ROW_SPACE;
  });
  const tallest = Math.max(120, top - ROW_SPACE);

  const placePorts = (dir: "input" | "output", px: number, center: number) => {
    const list = module.ports.filter((p) => p.dir === dir);
    const total = list.length * (PORT_H + ROW_GAP) - ROW_GAP;
    let y = Math.max(0, center - total / 2);
    for (const p of list) {
      const node = nodes.get(`${dir === "input" ? "in" : "out"}:${p.name}`)!;
      node.x = px;
      node.y = y;
      rowOf.set(node.id, dir === "input" ? 0 : rows.length - 1);
      y += PORT_H + ROW_GAP;
    }
  };
  placePorts("input", 0, (rowBottom[0] ?? 120) / 2);
  placePorts("output", width, (rowBottom[rowBottom.length - 1] + (rowBottom[rowBottom.length - 2] ?? 0)) / 2);
  width += PORT_W;

  let backLane = 0;
  const edges: Edge[] = links.map((link, index) => {
    const a = nodes.get(link.from.node)!;
    const b = nodes.get(link.to.node)!;
    const pa = pinOf(a, link.from.port);
    const pb = pinOf(b, link.to.port);
    const [x1, y1, x2, y2] = [a.x + pa.x, a.y + pa.y, b.x + pb.x, b.y + pb.y];
    const fan = ((index * 7) % 13) * 2.2;
    const rowA = rowOf.get(a.id) ?? 0;
    const rowB = rowOf.get(b.id) ?? 0;
    let points: [number, number][];
    const isBack = back.has(link) || rowB < rowA || x2 <= x1 + 8;
    if (rowB > rowA && !back.has(link)) {
      // Forward across rows: use the free channel just below the source row.
      const lane = rowBottom[rowA] + 12 + ((index * 5) % 9) * 3.4;
      points = [[x1, y1], [x1 + 14 + fan, y1], [x1 + 14 + fan, lane], [x2 - 14 - fan, lane], [x2 - 14 - fan, y2], [x2, y2]];
    } else if (isBack) {
      backLane += 1;
      const lane = tallest + 22 + backLane * 7;
      points = [[x1, y1], [x1 + 14 + fan, y1], [x1 + 14 + fan, lane], [x2 - 14 - fan, lane], [x2 - 14 - fan, y2], [x2, y2]];
    } else {
      const mx = Math.max(x1 + 10, x2 - 16 - fan);
      points = [[x1, y1], [mx, y1], [mx, y2], [x2, y2]];
    }
    return { id: `${index}`, ...link, points, back: isBack };
  });

  return { nodes: [...nodes.values()], edges, groups, width, height: tallest + 34 + backLane * 7 };
}
