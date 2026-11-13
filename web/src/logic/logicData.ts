import type { Program, ProgramInst, ProgramOp } from "../engine";
import type { Scope, DieDemo } from "../schematic/simStore";

export interface Circuit {
  label: string;
  scope: Scope;
  demo?: DieDemo;
}

interface OpInfo {
  formula: string;
  what: string;
  unit: string;
  circuit?: Circuit;
}

const CELL: Circuit = { label: "See the compute cell", scope: "cell" };
const LANE: Circuit = { label: "See the vector lane on the die", scope: "die", demo: "mlp" };

export const OP_INFO: Record<string, OpInfo> = {
  emb_read: { formula: "x = E[token]", what: "Reads the row of the embedding table that belongs to the token. It multiplies nothing.", unit: "Weight cells", circuit: CELL },
  matvec: { formula: "y = W · x", what: "Each row of the matrix is stored in memory next to a multiplier. The activation network sends x to every cell, and each one accumulates its row: Σ weight × activation.", unit: "Weight cells and activation network", circuit: CELL },
  copy: { formula: "y = x", what: "Moves a vector from one register to another.", unit: "Vector bank", circuit: LANE },
  add: { formula: "y = a + b", what: "Adds element by element, first aligning the two vectors' exponents.", unit: "Vector lane", circuit: LANE },
  mul_elem: { formula: "y = a ⊙ b", what: "Element-wise product. It multiplies activation by activation, which is why it cannot be done in the weight cells.", unit: "Vector lane", circuit: LANE },
  silu: { formula: "y = x · σ(x)", what: "SiLU activation function. The value comes from a table and is multiplied by the input.", unit: "Function table and lane", circuit: LANE },
  sigmoid: { formula: "y = σ(x)", what: "Sigmoid from a table, which keeps each gate between 0 and 1.", unit: "Function table", circuit: LANE },
  rms_norm: { formula: "y = x · (1 + w) / √(mean(x²) + ε)", what: "Normalizes each block: adds up the squares, computes the inverse square root from a table and scales the vector by its learned weight.", unit: "Vector lane and table", circuit: LANE },
  rms_gated: { formula: "y = x · g(z) / √(mean(x²) + ε)", what: "A norm like the previous one, but the factor depends on another vector z acting as a gate.", unit: "Vector lane and table", circuit: LANE },
  split_query_gate: { formula: "q, g = split(x)", what: "The Q projection interleaves each head's query and gate. Here they are split.", unit: "Vector bank", circuit: LANE },
  rope: { formula: "(a, b) → (a cos θ − b sin θ, a sin θ + b cos θ)", what: "Encodes the token's position by rotating pairs of values by an angle that depends on that position.", unit: "Vector lane and table", circuit: LANE },
  kv_append: { formula: "K[t], V[t] = k, v", what: "Stores this token's key and value. The cache grows with each token of the context.", unit: "KV cache SRAM" },
  attn_sum: { formula: "y = Σ softmax(q · K) · V", what: "Compares the query with each earlier token, turns the similarities into probabilities and adds up the weighted values. Its cost grows with the context.", unit: "Vector lane, table and KV cache", circuit: { label: "See attention on the die", scope: "die", demo: "attention" } },
  dconv: { formula: "y[t] = Σ w · x[t − i]", what: "Causal convolution: mixes the vector with those of the earlier tokens.", unit: "Vector lane and convolution state", circuit: LANE },
  delta_step: { formula: "S ← αS;  v̂ = S·k;  S ← S + β(v − v̂)kᵀ;  y = S·q", what: "Gated DeltaNet's delta rule. Instead of keeping the whole context, it updates a state matrix per head, so the cost does not grow with the context.", unit: "Vector lane and state SRAM", circuit: { label: "See the delta rule on the die", scope: "die", demo: "delta" } },
  concat: { formula: "y = [a, b]", what: "Joins two vectors into one.", unit: "Vector bank", circuit: LANE },
  moe: { formula: "y = Σ pᵢ · MLPᵢ(x), for the best k", what: "A network decides which experts handle this token. Only those run, plus a shared one.", unit: "Cells, lane and top-k selector", circuit: LANE },
};

const fieldNames = (op: ProgramOp, names: string[]) => names.flatMap((n) => (typeof op[n] === "number" ? [op[n] as number] : []));

export function reads(op: ProgramOp): number[] {
  if (op.op === "rope") return [op.buf as number];
  return fieldNames(op, ["src", "a", "b", "mixed", "z", "query", "key", "value"]).filter((buffer, i, all) => all.indexOf(buffer) === i);
}

export function writes(op: ProgramOp): number[] {
  if (op.op === "rope") return [op.buf as number];
  if (op.op === "split_query_gate") return [op.query as number, op.gate as number];
  return typeof op.dst === "number" ? [op.dst] : [];
}

/** `split_query_gate` names its outputs with fields that are inputs of other operations. */
function inputs(op: ProgramOp): number[] {
  return op.op === "split_query_gate" ? [op.src as number] : reads(op);
}

/** Multiplications per token, leaving out those that depend on the context. */
export function multiplications(op: ProgramOp, program: Program): number {
  const len = (field: string) => program.buffers[op[field] as number] ?? 0;
  switch (op.op) {
    case "matvec":
      return len("src") * len("dst");
    case "mul_elem":
    case "silu":
    case "gelu":
    case "scale":
      return len("dst");
    case "rms_norm":
    case "rms_gated":
      return 2 * len("dst");
    case "rope":
      return 4 * (op.rotary as number) * (program.buffers[op.buf as number] / (op.block as number));
    case "moe":
      return 2 * len("src") * (op.top_k as number) * (op.intermediate as number);
    default:
      return 0;
  }
}

export interface LayerKind {
  key: string;
  label: string;
  layers: number[];
  ops: ProgramInst[];
}

export function layerKinds(program: Program): LayerKind[] {
  const byLayer = new Map<number, ProgramInst[]>();
  for (const inst of program.ops) {
    if (inst.layer !== null) byLayer.set(inst.layer, [...(byLayer.get(inst.layer) ?? []), inst]);
  }
  const kinds = new Map<string, LayerKind>();
  for (const [layer, ops] of byLayer) {
    const key = ops.map((i) => i.name).join("|");
    const label = ops.some((i) => i.op.op === "delta_step") ? "DeltaNet layer" : "Attention layer";
    const kind = kinds.get(key) ?? { key, label, layers: [], ops };
    kind.layers.push(layer);
    kinds.set(key, kind);
  }
  return [...kinds.values()];
}

export interface FlowLink {
  from: number;
  to: number;
  length: number;
}

/** Links each operation to the ones that use its results. */
export function flow(ops: ProgramInst[], program: Program): { links: FlowLink[]; incoming: number[]; outgoing: number[] } {
  const writer = new Map<number, number>();
  const links: FlowLink[] = [];
  const incoming = new Set<number>();
  const used = new Set<number>();
  ops.forEach((inst, index) => {
    for (const buffer of inputs(inst.op)) {
      const from = writer.get(buffer);
      if (from === undefined) incoming.add(buffer);
      else {
        links.push({ from, to: index, length: program.buffers[buffer] });
        used.add(buffer);
      }
    }
    for (const buffer of writes(inst.op)) writer.set(buffer, index);
  });
  const outgoing = [...writer.entries()].filter(([buffer]) => !used.has(buffer)).map(([, index]) => index);
  return { links, incoming: [...incoming], outgoing: [...new Set(outgoing)] };
}
