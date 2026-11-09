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

const CELL: Circuit = { label: "Ver la celda de cálculo", scope: "cell" };
const LANE: Circuit = { label: "Ver el carril vectorial en el die", scope: "die", demo: "mlp" };

export const OP_INFO: Record<string, OpInfo> = {
  emb_read: { formula: "x = E[token]", what: "Lee la fila de la tabla de embeddings que corresponde al token. No multiplica nada.", unit: "Celdas de pesos", circuit: CELL },
  matvec: { formula: "y = W · x", what: "Cada fila de la matriz está guardada en memoria junto a un multiplicador. La red de activaciones reparte x a todas las celdas, y cada una acumula su fila: Σ peso × activación.", unit: "Celdas de pesos y red de activaciones", circuit: CELL },
  copy: { formula: "y = x", what: "Mueve un vector de un registro a otro.", unit: "Banco de vectores", circuit: LANE },
  add: { formula: "y = a + b", what: "Suma elemento a elemento, alineando antes los exponentes de los dos vectores.", unit: "Carril vectorial", circuit: LANE },
  mul_elem: { formula: "y = a ⊙ b", what: "Producto elemento a elemento. Multiplica activación por activación, y por eso no puede hacerse en las celdas de pesos.", unit: "Carril vectorial", circuit: LANE },
  silu: { formula: "y = x · σ(x)", what: "Función de activación SiLU. El valor sale de una tabla y se multiplica por la entrada.", unit: "Tabla de funciones y carril", circuit: LANE },
  sigmoid: { formula: "y = σ(x)", what: "Sigmoide por tabla, que deja cada puerta entre 0 y 1.", unit: "Tabla de funciones", circuit: LANE },
  rms_norm: { formula: "y = x · (1 + w) / √(media(x²) + ε)", what: "Normaliza cada bloque: suma los cuadrados, calcula la raíz inversa por tabla y escala el vector con su peso aprendido.", unit: "Carril vectorial y tabla", circuit: LANE },
  rms_gated: { formula: "y = x · g(z) / √(media(x²) + ε)", what: "Norma como la anterior, pero el factor depende de otro vector z que hace de puerta.", unit: "Carril vectorial y tabla", circuit: LANE },
  split_query_gate: { formula: "q, g = separar(x)", what: "La proyección Q intercala la consulta y la puerta de cada cabeza. Aquí se separan.", unit: "Banco de vectores", circuit: LANE },
  rope: { formula: "(a, b) → (a cos θ − b sin θ, a sin θ + b cos θ)", what: "Codifica la posición del token girando parejas de valores un ángulo que depende de esa posición.", unit: "Carril vectorial y tabla", circuit: LANE },
  kv_append: { formula: "K[t], V[t] = k, v", what: "Guarda la clave y el valor de este token. La caché crece con cada token del contexto.", unit: "SRAM de caché KV" },
  attn_sum: { formula: "y = Σ softmax(q · K) · V", what: "Compara la consulta con cada token anterior, convierte las similitudes en probabilidades y suma los valores ponderados. Su coste crece con el contexto.", unit: "Carril vectorial, tabla y caché KV", circuit: { label: "Ver la atención en el die", scope: "die", demo: "attention" } },
  dconv: { formula: "y[t] = Σ w · x[t − i]", what: "Convolución causal: mezcla el vector con los de los tokens anteriores.", unit: "Carril vectorial y estado de convolución", circuit: LANE },
  delta_step: { formula: "S ← αS;  v̂ = S·k;  S ← S + β(v − v̂)kᵀ;  y = S·q", what: "Regla delta de Gated DeltaNet. En vez de guardar todo el contexto, actualiza una matriz de estado por cabeza, con lo que el coste no crece con el contexto.", unit: "Carril vectorial y SRAM de estado", circuit: { label: "Ver la regla delta en el die", scope: "die", demo: "delta" } },
  concat: { formula: "y = [a, b]", what: "Une dos vectores en uno.", unit: "Banco de vectores", circuit: LANE },
  moe: { formula: "y = Σ pᵢ · MLPᵢ(x), para los k mejores", what: "Una red decide qué expertos atienden este token. Solo se ejecutan esos, más uno compartido.", unit: "Celdas, carril y selector top-k", circuit: LANE },
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
    const label = ops.some((i) => i.op.op === "delta_step") ? "Capa DeltaNet" : "Capa de atención";
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
