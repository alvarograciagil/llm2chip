import { useCallback, useEffect, useMemo, useRef, useState, type PointerEvent as ReactPointerEvent } from "react";
import type { Bit, NetlistModule } from "../engine";
import { IconButton } from "../components/controls";
import { AREA_VIEW_THRESHOLD, layoutAreas, layoutModule, type Edge, type Layout, type Node, type Pin } from "./layout";
import { currentModule, useCellSim } from "./simStore";

const MIN_FIT = 0.7;

interface Camera {
  x: number;
  y: number;
  k: number;
}

function netBit(module: NetlistModule, snapshot: Record<string, string>, bit: Bit): string {
  if (bit === "zero") return "0";
  if (bit === "one") return "1";
  const [id, index] = bit.net;
  const net = module.nets[id];
  const text = snapshot[net.name];
  return text ? text[text.length - 1 - index] ?? "0" : "0";
}

function busValue(module: NetlistModule, snapshot: Record<string, string>, bits: Bit[]): string {
  return [...bits].reverse().map((b) => netBit(module, snapshot, b)).join("");
}

function showValue(bits: string): string {
  if (bits.length <= 1) return bits;
  return `${BigInt.asIntN(bits.length, BigInt(`0b${bits}`))}`;
}

/** At most six characters, so a value never runs into the labels beside it. */
function compactValue(bits: string): string {
  const full = showValue(bits);
  if (full.length <= 6) return full;
  const value = Number(full);
  const [limit, suffix] = Math.abs(value) >= 1e12 ? [1e12, "T"] : Math.abs(value) >= 1e9 ? [1e9, "G"] : Math.abs(value) >= 1e6 ? [1e6, "M"] : [1e3, "k"];
  return `${(value / limit).toFixed(Math.abs(value / limit) >= 100 ? 0 : 1).replace(".", ",")}${suffix}`;
}

const PIN_HELP: Record<string, string> = {
  clear: "pone a cero el estado",
  clear_total: "borra la suma acumulada",
  clear_group: "borra la suma del grupo",
  load_en: "escribe un peso en memoria",
  load_addr: "dirección del peso que se escribe",
  load_data: "valor del peso que se escribe",
  load_scale: "escribe la escala o el sesgo del grupo",
  run: "avanza el cálculo un ciclo",
  x: "activación de entrada",
  total: "suma acumulada de la fila",
  weight: "peso leído de memoria",
  product: "peso por activación",
  valid: "el dato de este ciclo es válido",
  value: "valor que se suma",
  addr: "dirección de memoria",
  we: "activa la escritura",
  wdata: "dato que se escribe",
  rdata: "dato leído",
  raddr: "dirección de lectura",
  waddr: "dirección de escritura",
  d: "dato que se guarda",
  q: "dato guardado",
  advance: "avanza el contador",
  address: "dirección actual",
  s: "elige la entrada",
  a: "entrada a",
  b: "entrada b",
  y: "salida",
};

function GateSymbol({ node, selected }: { node: Node; selected: boolean }) {
  const { w, h } = node;
  const cls = selected ? "gate selected" : "gate";
  const body = {
    and2: `M8 2 H${w / 2 - 2} A${h / 2 - 2} ${h / 2 - 2} 0 0 1 ${w / 2 - 2} ${h - 2} H8 Z`,
    or2: `M8 2 C${w * 0.45} 2 ${w * 0.72} ${h * 0.2} ${w - 6} ${h / 2} C${w * 0.72} ${h * 0.8} ${w * 0.45} ${h - 2} 8 ${h - 2} C${w * 0.3} ${h * 0.7} ${w * 0.3} ${h * 0.3} 8 2 Z`,
    xor2: `M12 2 C${w * 0.5} 2 ${w * 0.74} ${h * 0.2} ${w - 6} ${h / 2} C${w * 0.74} ${h * 0.8} ${w * 0.5} ${h - 2} 12 ${h - 2} C${w * 0.34} ${h * 0.7} ${w * 0.34} ${h * 0.3} 12 2 Z`,
    not: `M10 3 L${w - 14} ${h / 2} L10 ${h - 3} Z`,
    mux2: `M8 2 L${w - 8} ${h * 0.22} V${h * 0.78} L8 ${h - 2} Z`,
    dff: `M8 2 H${w - 8} V${h - 2} H8 Z`,
  }[node.prim as "and2"];
  return (
    <g>
      <path className={cls} d={body} />
      {node.prim === "xor2" && <path className="gate-line" d={`M6 2 C${w * 0.26} ${h * 0.3} ${w * 0.26} ${h * 0.7} 6 ${h - 2}`} />}
      {node.prim === "not" && <circle className={cls} cx={w - 10} cy={h / 2} r={3.5} />}
      {node.prim === "dff" && <path className="gate-line" d={`M8 ${h - 15} l7 5 l-7 5`} />}
      {node.prim === "dff" && <text className="gate-text" x={w / 2} y={h / 2 + 4}>D</text>}
      {node.prim === "mux2" && <text className="gate-text" x={w / 2} y={h / 2 + 4}>MUX</text>}
      {node.pins.map((p) => (
        <line key={p.name} className="gate-line" x1={p.x} y1={p.y} x2={p.side === "left" ? 8 : w - 6} y2={p.y} />
      ))}
    </g>
  );
}

function BoxSymbol({ node, selected, value }: { node: Node; selected: boolean; value: (pin: Pin) => string }) {
  const clickable = node.instance?.kind.kind === "module";
  return (
    <g>
      <rect className={`${selected ? "box selected" : "box"} ${node.prim === "sram" || node.prim === "ram2p" || node.prim === "rom" ? "box-memory" : ""}`} width={node.w} height={node.h} rx={8} />
      <text className="box-title" x={10} y={17}>{node.title}</text>
      {node.subtitle && <text className="box-sub" x={10} y={31}>{node.subtitle}</text>}
      {clickable && (
        <path className="box-chevron" d={`M${node.w - 14} 12 l5 5 l-5 5`} />
      )}
      {node.pins.map((p) => (
        <g key={p.name}>
          <title>{`${p.name}${PIN_HELP[p.name] ? `: ${PIN_HELP[p.name]}` : ""} (${p.width} ${p.width === 1 ? "bit" : "bits"}) = ${showValue(value(p))}`}</title>
          <circle className="pin" cx={p.x} cy={p.y} r={2.5} />
          <text className="pin-name" x={p.side === "left" ? 8 : node.w - 8} y={p.y + 3.5} textAnchor={p.side === "left" ? "start" : "end"}>
            {p.name}
          </text>
          {p.width > 1 && (
            <text className="pin-value" x={node.w / 2} y={p.y + 3.5} textAnchor="middle">
              {compactValue(value(p))}
            </text>
          )}
        </g>
      ))}
    </g>
  );
}

function AreaSymbol({ node }: { node: Node }) {
  return (
    <g>
      <rect className="box area" width={node.w} height={node.h} rx={10} />
      <text className="box-title" x={14} y={27}>{node.title}</text>
      <text className="box-sub" x={14} y={44}>{node.subtitle}</text>
      <path className="box-chevron" d={`M${node.w - 18} ${node.h / 2 - 5} l5 5 l-5 5`} />
    </g>
  );
}

function PortSymbol({ node, value }: { node: Node; value: string }) {
  return (
    <g>
      <rect className="port" width={node.w} height={node.h} rx={node.h / 2} />
      <text className="port-name" x={node.kind === "input" ? 10 : node.w - 10} y={node.h / 2 + 4} textAnchor={node.kind === "input" ? "start" : "end"}>
        {node.title}
      </text>
      <text className="port-value" x={node.kind === "input" ? node.w - 10 : 10} y={node.h / 2 + 4} textAnchor={node.kind === "input" ? "end" : "start"}>
        {compactValue(value)}
      </text>
    </g>
  );
}

function EdgePath({ edge, value, state, onHover }: { edge: Edge; value: string; state: "" | "hot" | "dim"; onHover: (e: Edge | null) => void }) {
  const d = edge.points.map(([x, y], i) => `${i ? "L" : "M"}${x} ${y}`).join(" ");
  const active = value.includes("1");
  return (
    <g className={state} onPointerEnter={() => onHover(edge)} onPointerLeave={() => onHover(null)}>
      <path className="wire-hit" d={d} />
      <path className={`wire ${active ? "on" : ""} ${value.length > 1 ? "bus" : ""}`} d={d} />
    </g>
  );
}

function Diagram({ module, layout, snapshot }: { module: NetlistModule; layout: Layout; snapshot: Record<string, string> }) {
  const selected = useCellSim((s) => s.selected);
  const select = useCellSim((s) => s.select);
  const enter = useCellSim((s) => s.enter);
  const setArea = useCellSim((s) => s.setArea);
  const [hovered, setHovered] = useState<Edge | null>(null);

  // Hovering a wire or selecting a block lights its nets and fades the rest.
  const focus = useMemo(() => {
    if (hovered) return new Set([hovered.driver]);
    if (!selected) return null;
    return new Set(layout.edges.filter((e) => e.from.node === selected || e.to.node === selected).map((e) => e.driver));
  }, [hovered, selected, layout]);

  const pinValue = (node: Node, pin: Pin) => {
    if (node.kind !== "instance") {
      return snapshot[node.title] ?? "0";
    }
    return busValue(module, snapshot, node.instance!.conns[pin.name] ?? []);
  };

  return (
    <>
      {layout.groups.map((g) => (
        <g key={g.name}>
          <rect className="group-frame" x={g.x} y={g.y} width={g.w} height={g.h} rx={12} />
          <text className="group-title" x={g.x + 12} y={g.y + 16}>{g.name}</text>
        </g>
      ))}
      {layout.edges.map((edge) => (
        <EdgePath key={edge.id} edge={edge} value={busValue(module, snapshot, edge.bits)} state={!focus ? "" : focus.has(edge.driver) ? "hot" : "dim"} onHover={setHovered} />
      ))}
      {layout.junctions.map(([x, y]) => (
        <circle key={`${x},${y}`} className="junction" cx={x} cy={y} r={2.6} />
      ))}
      {layout.nodes.map((node) => (
        <g
          key={node.id}
          transform={`translate(${node.x} ${node.y})`}
          className={node.kind === "instance" || node.kind === "area" ? "node" : undefined}
          onClick={(e) => {
            e.stopPropagation();
            if (node.kind === "instance") select(node.id);
          }}
          onDoubleClick={(e) => {
            e.stopPropagation();
            if (node.kind === "area") setArea(node.title);
            else if (node.instance?.kind.kind === "module") enter(node.id);
          }}
        >
          {node.kind === "area" ? (
            <AreaSymbol node={node} />
          ) : node.kind !== "instance" ? (
            <PortSymbol node={node} value={pinValue(node, node.pins[0])} />
          ) : node.prim && node.prim !== "sram" && node.prim !== "ram2p" && node.prim !== "rom" ? (
            <GateSymbol node={node} selected={selected === node.id} />
          ) : (
            <BoxSymbol node={node} selected={selected === node.id} value={(pin) => pinValue(node, pin)} />
          )}
        </g>
      ))}
      {hovered && (
        <g className="wire-label" transform={`translate(${hovered.points[1][0] + 6} ${hovered.points[1][1] - 8})`}>
          <text>
            {hovered.net}{hovered.bits.length > 1 ? ` [${hovered.bits.length}]` : ""} = {showValue(busValue(module, snapshot, hovered.bits))}
          </text>
        </g>
      )}
    </>
  );
}

export function SchematicView() {
  const design = useCellSim((s) => s.design);
  const path = useCellSim((s) => s.path);
  const goTo = useCellSim((s) => s.goTo);
  const select = useCellSim((s) => s.select);
  const snapshot = useCellSim((s) => s.snapshot);
  const module = currentModule(design, path);
  const area = useCellSim((s) => s.area);
  const overview = !!module && !area && module.instances.length > AREA_VIEW_THRESHOLD;
  // The layout arrives later than the module it was computed for, so the two travel together.
  const [placed, setPlaced] = useState<{ module: NetlistModule; layout: Layout } | null>(null);
  const layout = placed && placed.module === module ? placed.layout : null;
  useEffect(() => {
    if (!module || !design) return;
    let current = true;
    void (overview ? layoutAreas(module, design) : layoutModule(module, design, area)).then((next) => {
      if (current) setPlaced({ module, layout: next });
    });
    return () => {
      current = false;
    };
  }, [module, design, area, overview]);

  const host = useRef<HTMLDivElement>(null);
  const world = useRef<SVGGElement>(null);
  const camera = useRef<Camera>({ x: 0, y: 0, k: 1 });
  const drag = useRef<{ x: number; y: number } | null>(null);

  const apply = useCallback(() => {
    const { x, y, k } = camera.current;
    world.current?.setAttribute("transform", `translate(${x} ${y}) scale(${k})`);
  }, []);

  const fit = useCallback(() => {
    const el = host.current;
    if (!el || !layout) return;
    const { width, height } = el.getBoundingClientRect();
    // Never shrink so far that the labels stop being readable: the rest is reached by panning.
    const k = Math.max(Math.min(MIN_FIT, 1.4), Math.min(1.4, (width - 80) / layout.width, (height - 130) / layout.height));
    if (!(k > 0)) return;
    const overflow = layout.width * k > width - 48;
    camera.current = { k, x: overflow ? 24 : (width - layout.width * k) / 2, y: Math.max(62, 62 + (height - 110 - layout.height * k) / 2) };
    apply();
  }, [layout, apply]);

  useEffect(() => {
    fit();
    const el = host.current;
    if (!el) return;
    const observer = new ResizeObserver(fit);
    observer.observe(el);
    return () => observer.disconnect();
  }, [fit]);

  const zoom = useCallback(
    (factor: number, px: number, py: number) => {
      const cam = camera.current;
      const k = Math.min(8, Math.max(0.05, cam.k * factor));
      camera.current = { k, x: px - ((px - cam.x) / cam.k) * k, y: py - ((py - cam.y) / cam.k) * k };
      apply();
    },
    [apply],
  );

  useEffect(() => {
    const el = host.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      if ((e.target as Element).closest(".hud")) return;
      e.preventDefault();
      const box = el.getBoundingClientRect();
      const isMouseWheel = e.deltaMode !== 0 || (e.deltaX === 0 && Number.isInteger(e.deltaY) && Math.abs(e.deltaY) >= 100);
      if (e.ctrlKey || isMouseWheel) {
        zoom(Math.exp(-e.deltaY * (e.ctrlKey ? 0.012 : 0.0025)), e.clientX - box.left, e.clientY - box.top);
      } else {
        const cam = camera.current;
        camera.current = { ...cam, x: cam.x - e.deltaX, y: cam.y - e.deltaY };
        apply();
      }
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, [zoom, apply]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const state = useCellSim.getState();
      if (e.key !== "Escape" || (e.target as HTMLElement).matches("input,textarea")) return;
      if (state.area) state.setArea(null);
      else if (state.path.length > 0) goTo(state.path.length - 1);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [goTo]);

  const onDown = (e: ReactPointerEvent<HTMLDivElement>) => {
    if ((e.target as Element).closest(".hud")) return;
    drag.current = { x: e.clientX, y: e.clientY };
    e.currentTarget.setPointerCapture(e.pointerId);
  };

  const onMove = (e: ReactPointerEvent<HTMLDivElement>) => {
    if (!drag.current) return;
    const cam = camera.current;
    camera.current = { ...cam, x: cam.x + e.clientX - drag.current.x, y: cam.y + e.clientY - drag.current.y };
    drag.current = { x: e.clientX, y: e.clientY };
    apply();
  };

  const crumbs = useMemo(() => {
    if (!design) return [];
    const names = [design.modules[design.top].title];
    let current = design.modules[design.top];
    for (const segment of path) {
      const inst = current.instances.find((i) => i.name === segment);
      if (!inst || inst.kind.kind !== "module") break;
      current = design.modules[inst.kind.of];
      names.push(current.title);
    }
    return names;
  }, [design, path]);

  const center = () => {
    const box = host.current!.getBoundingClientRect();
    return [box.width / 2, box.height / 2] as const;
  };

  return (
    <section
      className="stage schematic"
      ref={host}
      onPointerDown={onDown}
      onPointerMove={onMove}
      onPointerUp={() => (drag.current = null)}
      onClick={(e) => {
        if (!(e.target as Element).closest(".hud")) select(null);
      }}
    >
      <svg role="img" aria-label={module ? `Esquemático: ${module.title}. ${module.description}` : "Esquemático"}>
        <g ref={world}>{layout && module && <Diagram module={module} layout={layout} snapshot={snapshot} />}</g>
      </svg>
      <div className="hud top">
        <nav className="breadcrumb" aria-label="Posición en el circuito">
          {crumbs.map((name, depth) => (
            <span key={depth} className="crumb">
              {depth > 0 && <span aria-hidden="true">›</span>}
              {depth < crumbs.length - 1 || area ? <button onClick={() => goTo(depth)}>{name}</button> : <span>{name}</span>}
            </span>
          ))}
          {area && (
            <span className="crumb">
              <span aria-hidden="true">›</span>
              <span>{area}</span>
            </span>
          )}
        </nav>
      </div>
      <div className="hud bottom">
        <div className="legend">
          <span>
            <i className="wire-key on" aria-hidden="true" />
            Con valor
          </span>
          <span>
            <i className="wire-key" aria-hidden="true" />
            A cero
          </span>
          <span>
            <i className="wire-key thick" aria-hidden="true" />
            Bus
          </span>
          <span className="legend-hint">Pasa por un cable para seguirlo · Doble clic para entrar · Esc para subir</span>
        </div>
        <div className="zoom">
          <IconButton label="Alejar" onClick={() => zoom(0.7, ...center())}>
            <path d="M5 10h10" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
          </IconButton>
          <IconButton label="Encajar" onClick={fit}>
            <path d="M4 8V4h4M16 8V4h-4M4 12v4h4M16 12v4h-4" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
          </IconButton>
          <IconButton label="Acercar" onClick={() => zoom(1.4, ...center())}>
            <path d="M5 10h10M10 5v10" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
          </IconButton>
        </div>
      </div>
    </section>
  );
}
