import { useEffect, useMemo, useState } from "react";
import ELK, { type ElkExtendedEdge, type ElkNode } from "elkjs/lib/elk.bundled.js";
import { create } from "zustand";
import { programDesign, type Program, type ProgramInst } from "../engine";
import { formatNumber } from "../format";
import { useCellSim } from "../schematic/simStore";
import { useDesign } from "../store";
import { flow, layerKinds, multiplications, OP_INFO, reads, writes, type LayerKind } from "./logicData";

interface LogicState {
  kind: string | null;
  selected: ProgramInst | null;
  setKind: (kind: string | null) => void;
  select: (selected: ProgramInst | null) => void;
}

const useLogic = create<LogicState>((set) => ({
  kind: null,
  selected: null,
  setKind: (kind) => set({ kind, selected: null }),
  select: (selected) => set({ selected }),
}));

function useProgram(): Program | null {
  const model = useDesign((s) => s.model);
  return useMemo(() => (model ? programDesign(model) : null), [model]);
}

const unitClass = (op: string) => {
  if (op === "matvec" || op === "emb_read" || op === "moe") return "weights";
  if (op === "kv_append" || op === "attn_sum" || op === "delta_step" || op === "dconv") return "state";
  return "vector";
};

const compact = (n: number) => (n >= 1e9 ? `${formatNumber(n / 1e9)} G` : n >= 1e6 ? `${formatNumber(n / 1e6)} M` : n >= 1e3 ? `${formatNumber(n / 1e3)} k` : formatNumber(n));

function dims(inst: ProgramInst, program: Program) {
  const inputs = reads(inst.op).map((b) => program.buffers[b]);
  const outputs = writes(inst.op).map((b) => program.buffers[b]);
  return `${inputs.join(" + ") || "token"} → ${outputs.join(" + ") || "memoria"}`;
}

function Step({ title, text, tag, onClick, unit }: { title: string; text: string; tag?: string; onClick?: () => void; unit?: string }) {
  const body = (
    <>
      <strong>{title}</strong>
      <span>{text}</span>
      {tag && <small>{tag}</small>}
    </>
  );
  return onClick ? (
    <button className={`logic-step ${unit ?? ""}`} onClick={onClick}>{body}</button>
  ) : (
    <div className={`logic-step ${unit ?? "outside"}`}>{body}</div>
  );
}

function Arrow() {
  return <div className="logic-arrow" aria-hidden="true" />;
}

function Overview({ program }: { program: Program }) {
  const setKind = useLogic((s) => s.setKind);
  const select = useLogic((s) => s.select);
  const kinds = useMemo(() => layerKinds(program), [program]);
  const total = useMemo(() => program.ops.reduce((sum, i) => sum + multiplications(i.op, program), 0), [program]);
  const outer = program.ops.filter((i) => i.layer === null);
  const kindOf = useMemo(() => new Map(kinds.flatMap((k, index) => k.layers.map((l) => [l, index] as const))), [kinds]);
  const opStep = (inst: ProgramInst) => <Step key={inst.name} title={inst.name} text={OP_INFO[inst.op.op]?.formula ?? ""} tag={dims(inst, program)} unit={unitClass(inst.op.op)} onClick={() => select(inst)} />;
  return (
    <div className="logic-page">
      <h2>Recorrido de un token</h2>
      <p className="muted-text">
        {program.num_layers} capas y {compact(total)} multiplicaciones por token, sin contar la atención, que crece con el contexto, ni la regla delta. Pulsa cualquier paso para ver qué calcula y cómo lo hace el circuito.
      </p>
      <div className="logic-flow">
        <Step title="Texto de entrada" text="Bytes UTF-8 que llegan por el puerto de host" tag="CPU de control: cola de entrada" unit="system" />
        <Arrow />
        <Step title="Tokenizador" text="El firmware parte el texto en tokens con las tablas BPE del modelo, cargadas con la imagen" tag="CPU de control: firmware RISC-V" unit="system" />
        <Arrow />
        {outer.filter((i) => i.op.op === "emb_read").map(opStep)}
        <Arrow />
        <div className="logic-layers">
          <h3>{program.num_layers} capas en serie</h3>
          <div className="logic-strip" role="img" aria-label="Tipo de cada capa">
            {Array.from({ length: program.num_layers }, (_, l) => (
              <i key={l} className={`kind-${kindOf.get(l) ?? 0}`} title={`Capa ${l}: ${kinds[kindOf.get(l) ?? 0]?.label}`} />
            ))}
          </div>
          <div className="logic-kinds">
            {kinds.map((kind, index) => (
              <button key={kind.key} className="logic-step" onClick={() => setKind(kind.key)}>
                <strong><i className={`logic-dot kind-${index}`} aria-hidden="true" />{kind.label} × {kind.layers.length}</strong>
                <span>{kind.ops.length} operaciones, {compact(kind.ops.reduce((sum, i) => sum + multiplications(i.op, program), 0))} multiplicaciones cada una</span>
                <small>Ver dentro ›</small>
              </button>
            ))}
          </div>
        </div>
        <Arrow />
        {outer.filter((i) => i.op.op !== "emb_read").map((inst, i, list) => (
          <div key={inst.name} className="logic-flow">
            {opStep(inst)}
            {i < list.length - 1 && <Arrow />}
          </div>
        ))}
        <Arrow />
        <Step title="Muestreo" text="Filtra los logits a medida que salen, guarda los mejores candidatos y sortea uno con la temperatura; el firmware aplica la lista de parada y el presupuesto de razonamiento" tag="Unidad de muestreo y firmware" unit="system" />
        <Arrow />
        <Step title="Texto de salida" text="El firmware convierte el token en bytes UTF-8 y lo devuelve como entrada del siguiente paso" tag="CPU de control: cola de salida" unit="system" />
      </div>
    </div>
  );
}

const NODE_W = 280;
const NODE_H = 66;
const elk = new ELK();

interface Placed {
  nodes: { id: string; x: number; y: number; index: number }[];
  edges: { id: string; d: string }[];
  width: number;
  height: number;
}

const ENTRY = -1;
const EXIT = -2;

function LayerFlow({ kind, program }: { kind: LayerKind; program: Program }) {
  const selected = useLogic((s) => s.selected);
  const select = useLogic((s) => s.select);
  const [placed, setPlaced] = useState<Placed | null>(null);
  useEffect(() => {
    let current = true;
    const { links, outgoing } = flow(kind.ops, program);
    const entries = [...new Set(kind.ops.flatMap((inst, index) => (reads(inst.op).some((b) => !kind.ops.some((o) => writes(o.op).includes(b))) ? [index] : [])))];
    const pairs: [number, number][] = [...links.map((l): [number, number] => [l.from, l.to]), ...entries.map((i): [number, number] => [ENTRY, i]), ...outgoing.map((i): [number, number] => [i, EXIT])];
    const child = (index: number): ElkNode => ({ id: `${index}`, width: index < 0 ? 170 : NODE_W, height: index < 0 ? 32 : NODE_H });
    const graph: ElkNode = {
      id: "root",
      layoutOptions: { "elk.algorithm": "layered", "elk.direction": "DOWN", "elk.edgeRouting": "ORTHOGONAL", "elk.layered.spacing.nodeNodeBetweenLayers": "34", "elk.spacing.nodeNode": "26", "elk.spacing.edgeEdge": "10", "elk.spacing.edgeNode": "14", "elk.layered.mergeEdges": "true", "elk.layered.nodePlacement.strategy": "NETWORK_SIMPLEX", "elk.layered.considerModelOrder.strategy": "NODES_AND_EDGES" },
      children: [child(ENTRY), ...kind.ops.map((_, i) => child(i)), child(EXIT)],
      edges: [...new Set(pairs.map(([a, b]) => `${a}>${b}`))].map((key, index) => {
        const [a, b] = key.split(">");
        return { id: `${index}`, sources: [a], targets: [b] };
      }),
    };
    void elk.layout(graph).then((result: ElkNode) => {
      if (!current) return;
      setPlaced({
        nodes: (result.children ?? []).map((n) => ({ id: n.id, x: n.x ?? 0, y: n.y ?? 0, index: Number(n.id) })),
        edges: (result.edges as ElkExtendedEdge[]).map((edge) => {
          const section = edge.sections![0];
          const points = [section.startPoint, ...(section.bendPoints ?? []), section.endPoint];
          return { id: edge.id, d: points.map((p, i) => `${i ? "L" : "M"}${p.x} ${p.y}`).join(" ") };
        }),
        width: result.width ?? 0,
        height: result.height ?? 0,
      });
    });
    return () => {
      current = false;
    };
  }, [kind, program]);

  if (!placed) return null;
  return (
    <svg className="logic-graph" width={placed.width + 8} height={placed.height + 8} role="img" aria-label={`Operaciones de ${kind.label}`}>
      <defs>
        <marker id="logic-head" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M0 0 L8 4 L0 8 Z" className="logic-head" />
        </marker>
      </defs>
      <g transform="translate(4 4)">
        {placed.edges.map((e) => (
          <path key={e.id} className="logic-edge" d={e.d} markerEnd="url(#logic-head)" />
        ))}
        {placed.nodes.map((node) => {
          if (node.index < 0) {
            return (
              <g key={node.id} transform={`translate(${node.x} ${node.y})`}>
                <rect className="logic-port" width={170} height={32} rx={16} />
                <text className="logic-port-text" x={85} y={20} textAnchor="middle">{node.index === ENTRY ? "Entrada de la capa" : "Salida de la capa"}</text>
              </g>
            );
          }
          const inst = kind.ops[node.index];
          const info = OP_INFO[inst.op.op];
          const mults = multiplications(inst.op, program);
          return (
            <g key={node.id} transform={`translate(${node.x} ${node.y})`} className={`logic-node ${unitClass(inst.op.op)} ${selected === inst ? "selected" : ""}`} role="button" tabIndex={0} aria-label={inst.name} onClick={() => select(inst)} onKeyDown={(e) => e.key === "Enter" && select(inst)}>
              <rect width={NODE_W} height={NODE_H} rx={9} />
              <rect className="logic-stripe" width={5} height={NODE_H} rx={2.5} />
              <text className="logic-title" x={16} y={21}>{inst.name}</text>
              <text className="logic-formula" x={16} y={40}>{info?.formula.length > 40 ? `${info.formula.slice(0, 39)}…` : info?.formula}</text>
              <text className="logic-dims" x={16} y={57}>{dims(inst, program)}{mults > 0 ? ` · ${compact(mults)} ×` : ""}</text>
            </g>
          );
        })}
      </g>
    </svg>
  );
}

export function LogicView() {
  const program = useProgram();
  const kindKey = useLogic((s) => s.kind);
  const setKind = useLogic((s) => s.setKind);
  const kind = useMemo(() => (program && kindKey ? layerKinds(program).find((k) => k.key === kindKey) ?? null : null), [program, kindKey]);
  if (!program) {
    return (
      <section className="stage logic">
        <div className="logic-page"><h2>Funcionamiento</h2><p className="muted-text">El compilador aún no admite este modelo. Elige un Qwen3.5 para ver sus operaciones.</p></div>
      </section>
    );
  }
  return (
    <section className="stage logic">
      {kind ? (
        <div className="logic-page">
          <nav className="breadcrumb static" aria-label="Posición">
            <span className="crumb"><button onClick={() => setKind(null)}>Recorrido de un token</button></span>
            <span className="crumb"><span aria-hidden="true">›</span><span>{kind.label}</span></span>
          </nav>
          <h2>{kind.label}</h2>
          <p className="muted-text">Capas {kind.layers.length > 12 ? `${kind.layers.slice(0, 12).join(", ")}…` : kind.layers.join(", ")}. Cada caja es una operación real del programa compilado, y las flechas son los vectores que se pasan.</p>
          <div className="legend logic-legend">
            <span><i className="logic-dot weights" aria-hidden="true" />Celdas de pesos</span>
            <span><i className="logic-dot vector" aria-hidden="true" />Carril vectorial</span>
            <span><i className="logic-dot state" aria-hidden="true" />Memoria de estado</span>
          </div>
          <LayerFlow kind={kind} program={program} />
        </div>
      ) : (
        <Overview program={program} />
      )}
    </section>
  );
}

export function LogicPanel() {
  const program = useProgram();
  const selected = useLogic((s) => s.selected);
  const setView = useDesign((s) => s.setView);
  const setScope = useCellSim((s) => s.setScope);
  const setDemo = useCellSim((s) => s.setDemo);
  if (!program) return <aside className="inspector" />;
  if (!selected) {
    return (
      <aside className="inspector">
        <h2 className="detail-title">Funcionamiento</h2>
        <p className="detail-sub">Esta vista explica qué calcula el chip. Sale del programa compilado para el modelo que tengas elegido.</p>
        <p className="muted-text">Para ver las puertas y los cables que lo hacen, usa el esquemático. Cada operación tiene aquí un enlace al circuito que la ejecuta.</p>
      </aside>
    );
  }
  const info = OP_INFO[selected.op.op];
  const mults = multiplications(selected.op, program);
  const matrix = typeof selected.op.matrix === "string" ? selected.op.matrix : typeof selected.op.weight === "string" ? selected.op.weight : null;
  return (
    <aside className="inspector">
      <h2 className="detail-title">{selected.name}</h2>
      <p className="logic-big-formula">{info?.formula}</p>
      <p className="detail-sub">{info?.what}</p>
      <dl className="inset metrics">
        <div><dt>Quién lo ejecuta</dt><dd>{info?.unit}</dd></div>
        <div><dt>Entradas</dt><dd>{reads(selected.op).map((b) => formatNumber(program.buffers[b])).join(" + ") || "Token"}</dd></div>
        <div><dt>Salidas</dt><dd>{writes(selected.op).map((b) => formatNumber(program.buffers[b])).join(" + ") || "Memoria"}</dd></div>
        <div><dt>Multiplicaciones</dt><dd>{mults > 0 ? formatNumber(mults) : "—"}</dd></div>
        {selected.layer !== null && <div><dt>Primera capa</dt><dd>{selected.layer}</dd></div>}
      </dl>
      {matrix && <p className="muted-text logic-tensor">Pesos: <code>{matrix}</code></p>}
      {info?.circuit && (
        <button
          className="pill primary-pill"
          onClick={() => {
            const circuit = info.circuit!;
            setScope(circuit.scope);
            if (circuit.demo) setDemo(circuit.demo);
            setView("esquematico");
          }}
        >
          {info.circuit.label}
        </button>
      )}
      {!info?.circuit && <p className="muted-text">Esta memoria aún no tiene circuito propio en el esquemático.</p>}
    </aside>
  );
}
