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
  return `${inputs.join(" + ") || "token"} → ${outputs.join(" + ") || "memory"}`;
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
      <h2>A token's path</h2>
      <p className="muted-text">
        {program.num_layers} layers and {compact(total)} multiplications per token, not counting attention, which grows with the context, nor the delta rule. Click any step to see what it computes and how the circuit does it.
      </p>
      <div className="logic-flow">
        <Step title="Input text" text="UTF-8 bytes that arrive through the host port" tag="Control CPU: input queue" unit="system" />
        <Arrow />
        <Step title="Tokenizer" text="The firmware splits the text into tokens with the model's BPE tables, loaded with the image" tag="Control CPU: RISC-V firmware" unit="system" />
        <Arrow />
        {outer.filter((i) => i.op.op === "emb_read").map(opStep)}
        <Arrow />
        <div className="logic-layers">
          <h3>{program.num_layers} layers in series</h3>
          <div className="logic-strip" role="img" aria-label="Type of each layer">
            {Array.from({ length: program.num_layers }, (_, l) => (
              <i key={l} className={`kind-${kindOf.get(l) ?? 0}`} title={`Layer ${l}: ${kinds[kindOf.get(l) ?? 0]?.label}`} />
            ))}
          </div>
          <div className="logic-kinds">
            {kinds.map((kind, index) => (
              <button key={kind.key} className="logic-step" onClick={() => setKind(kind.key)}>
                <strong><i className={`logic-dot kind-${index}`} aria-hidden="true" />{kind.label} × {kind.layers.length}</strong>
                <span>{kind.ops.length} operations, {compact(kind.ops.reduce((sum, i) => sum + multiplications(i.op, program), 0))} multiplications each</span>
                <small>See inside ›</small>
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
        <Step title="Sampling" text="Filters the logits as they come out, keeps the best candidates and draws one with the temperature; the firmware applies the stop list and the reasoning budget" tag="Sampling unit and firmware" unit="system" />
        <Arrow />
        <Step title="Output text" text="The firmware turns the token into UTF-8 bytes and feeds it back as the next step's input" tag="Control CPU: output queue" unit="system" />
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
    <svg className="logic-graph" width={placed.width + 8} height={placed.height + 8} role="img" aria-label={`Operations of the ${kind.label}`}>
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
                <text className="logic-port-text" x={85} y={20} textAnchor="middle">{node.index === ENTRY ? "Layer input" : "Layer output"}</text>
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
        <div className="logic-page"><h2>How it works</h2><p className="muted-text">The compiler does not take this model's architecture yet, so there are no operations to show. It takes Qwen3.5, Llama (and llama2.c), Mistral, Qwen2 and 3, Gemma, DeepSeek V3 and Janus.</p></div>
      </section>
    );
  }
  return (
    <section className="stage logic">
      {kind ? (
        <div className="logic-page">
          <nav className="breadcrumb static" aria-label="Position">
            <span className="crumb"><button onClick={() => setKind(null)}>A token's path</button></span>
            <span className="crumb"><span aria-hidden="true">›</span><span>{kind.label}</span></span>
          </nav>
          <h2>{kind.label}</h2>
          <p className="muted-text">Layers {kind.layers.length > 12 ? `${kind.layers.slice(0, 12).join(", ")}…` : kind.layers.join(", ")}. Each box is a real operation of the compiled program, and the arrows are the vectors passed on.</p>
          <div className="legend logic-legend">
            <span><i className="logic-dot weights" aria-hidden="true" />Weight cells</span>
            <span><i className="logic-dot vector" aria-hidden="true" />Vector lane</span>
            <span><i className="logic-dot state" aria-hidden="true" />State memory</span>
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
        <h2 className="detail-title">How it works</h2>
        <p className="detail-sub">This view explains what the chip computes. It comes from the program compiled for the model you have chosen.</p>
        <p className="muted-text">To see the gates and wires that do it, use the schematic. Each operation here has a link to the circuit that runs it.</p>
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
        <div><dt>What runs it</dt><dd>{info?.unit}</dd></div>
        <div><dt>Inputs</dt><dd>{reads(selected.op).map((b) => formatNumber(program.buffers[b])).join(" + ") || "Token"}</dd></div>
        <div><dt>Outputs</dt><dd>{writes(selected.op).map((b) => formatNumber(program.buffers[b])).join(" + ") || "Memory"}</dd></div>
        <div><dt>Multiplications</dt><dd>{mults > 0 ? formatNumber(mults) : "—"}</dd></div>
        {selected.layer !== null && <div><dt>First layer</dt><dd>{selected.layer}</dd></div>}
      </dl>
      {matrix && <p className="muted-text logic-tensor">Weights: <code>{matrix}</code></p>}
      {info?.circuit && (
        <button
          className="pill primary-pill"
          onClick={() => {
            const circuit = info.circuit!;
            setScope(circuit.scope);
            if (circuit.demo) setDemo(circuit.demo);
            setView("schematic");
          }}
        >
          {info.circuit.label}
        </button>
      )}
      {!info?.circuit && <p className="muted-text">This memory does not have a circuit of its own in the schematic yet.</p>}
    </aside>
  );
}
