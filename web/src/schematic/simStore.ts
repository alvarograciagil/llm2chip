import { create } from "zustand";
import { CellSimulation, DieSimulation, type CellInfo, type Design } from "../engine";

export interface ScriptStep {
  label: string;
  inputs: [string, string][];
}

export type Scope = "cell" | "die";

export type DieDemo = "mlp" | "attention" | "delta";

export interface DieResults {
  got: number[];
  expected: number[];
  exact: boolean;
  tolerance: number;
}

interface CellSimState {
  scope: Scope;
  die: DieSimulation | null;
  dieCounts: Record<string, number> | null;
  dieDone: boolean;
  dieResults: DieResults | null;
  setScope: (scope: Scope) => void;
  demo: DieDemo;
  setDemo: (demo: DieDemo) => void;
  handle: CellSimulation | null;
  design: Design | null;
  info: CellInfo | null;
  path: string[];
  area: string | null;
  setArea: (area: string | null) => void;
  snapshot: Record<string, string>;
  cycle: number;
  script: ScriptStep[];
  index: number;
  weightsText: string;
  activationsText: string;
  inputError: string | null;
  selected: string | null;
  source: { config: string; params: string } | null;

  rebuild: (config: string, params: string) => void;
  setWeights: (text: string) => void;
  setActivations: (text: string) => void;
  restart: () => void;
  step: () => void;
  runAll: () => void;
  enter: (instance: string) => void;
  goTo: (depth: number) => void;
  select: (instance: string | null) => void;
}

export function parseList(text: string): bigint[] | null {
  const parts = text.split(/[\s,;]+/).filter(Boolean);
  if (parts.length === 0 || !parts.every((p) => /^-?\d+$/.test(p))) return null;
  return parts.map((p) => BigInt(p));
}

const usesShifter = (info: CellInfo) => ["float", "posit", "log"].includes(info.config.coding);

const pathText = (path: string[]) => path.map((p) => `/${p}`).join("");
const toUnsigned = (value: bigint, bits: number) => BigInt.asUintN(bits, value).toString();

export function toSigned(bits: string): bigint {
  return BigInt.asIntN(bits.length, BigInt(`0b${bits || "0"}`));
}

/** The expected accumulator value, wrapping at the accumulator's width. */
export function expectedTotal(weights: bigint[], activations: bigint[], bits: number): bigint {
  const sum = weights.reduce((total, w, i) => total + w * activations[i], 0n);
  return BigInt.asIntN(bits, sum);
}

function buildScript(info: CellInfo, weights: bigint[], activations: bigint[]): ScriptStep[] {
  const { weight_bits, activation_bits, addr_bits } = info.config;
  const off: [string, string][] = [["load_en", "0"], ["run", "0"]];
  const steps: ScriptStep[] = [{ label: "Puesta a cero", inputs: [["clear", "1"], ...off, ...(usesShifter(info) ? ([["exp_base", "0"]] as [string, string][]) : [])] }];
  if (info.config.coding === "codebook") {
    for (let i = 0; i < 2 ** info.config.weight_bits; i++) {
      const signed = BigInt.asIntN(info.config.weight_bits, BigInt(i));
      steps.push({ label: `Tabla ${i + 1}`, inputs: [["clear", "0"], ["table_en", "1"], ["table_addr", String(i)], ["table_data", toUnsigned(signed, 16)]] });
    }
    steps.push({ label: "Fin de la tabla", inputs: [["table_en", "0"]] });
  }
  weights.forEach((w, i) =>
    steps.push({
      label: `Carga de pesos ${i + 1}/${weights.length}`,
      inputs: [["clear", "0"], ["load_en", "1"], ["load_addr", toUnsigned(BigInt(i), addr_bits)], ["load_data", toUnsigned(w, weight_bits)]],
    }),
  );
  steps.push({ label: "Fin de la carga", inputs: [["clear", "1"], ...off] });
  activations.forEach((x, i) =>
    steps.push({
      label: `Multiplicación ${i + 1}/${activations.length}`,
      inputs: [["clear", "0"], ["load_en", "0"], ["run", "1"], ["x", toUnsigned(x, activation_bits)]],
    }),
  );
  for (let i = 0; i < info.latency; i++) {
    steps.push({ label: `Vaciando etapas ${i + 1}/${info.latency}`, inputs: [["clear", "0"], ...off] });
  }
  return steps;
}

export function validateInputs(info: CellInfo, weightsText: string, activationsText: string): { error: string } | { weights: bigint[]; activations: bigint[] } {
  const weights = parseList(weightsText);
  const activations = parseList(activationsText);
  if (!weights) return { error: "Escribe los pesos como números enteros separados por comas." };
  if (!activations) return { error: "Escribe las activaciones como números enteros separados por comas." };
  if (weights.length !== activations.length) return { error: `Hay ${weights.length} pesos y ${activations.length} activaciones: deben ser los mismos.` };
  const { weight_bits, activation_bits, addr_bits } = info.config;
  if (weights.length > 2 ** addr_bits) return { error: `La memoria de la celda solo admite ${2 ** addr_bits} pesos.` };
  const fits = (v: bigint, bits: number) => BigInt.asIntN(bits, v) === v;
  if (!weights.every((w) => fits(w, weight_bits))) return { error: `Algún peso no cabe en ${weight_bits} bits con signo.` };
  if (!activations.every((x) => fits(x, activation_bits))) return { error: `Alguna activación no cabe en ${activation_bits} bits con signo.` };
  return { weights, activations };
}

export const useCellSim = create<CellSimState>((set, get) => {
  const refresh = () => {
    const { handle, die, scope, path } = get();
    const active = scope === "die" ? die : handle;
    if (!active) return;
    set({ snapshot: JSON.parse(active.snapshot(pathText(path))) ?? {}, cycle: active.cycle() });
  };

  const freshDie = (params: string) => {
    get().die?.free();
    const die = new DieSimulation(params, get().demo);
    set({ die, design: JSON.parse(die.design()), dieCounts: JSON.parse(die.counts()), dieDone: false, dieResults: null, selected: null, path: [], area: null });
    refresh();
  };

  const startScript = () => {
    const { info, weightsText, activationsText } = get();
    if (!info) return;
    const result = validateInputs(info, weightsText, activationsText);
    if ("error" in result) {
      set({ script: [], index: 0, inputError: result.error });
      return;
    }
    set({ script: buildScript(info, result.weights, result.activations), index: 0, inputError: null });
  };

  const fresh = (config: string, params: string, keepPath: boolean) => {
    if (get().scope === "die") {
      set({ source: { config, params } });
      freshDie(params);
      return;
    }
    get().handle?.free();
    const handle = new CellSimulation(config, params);
    const design: Design = JSON.parse(handle.design());
    const info: CellInfo = JSON.parse(handle.config());
    set({ handle, design, info, source: { config, params }, selected: null, path: keepPath ? get().path : [] });
    startScript();
    refresh();
  };

  return {
    scope: "cell",
    die: null,
    dieCounts: null,
    dieDone: false,
    dieResults: null,
    setScope(scope) {
      set({ scope, path: [], selected: null, area: null });
      const { source } = get();
      if (source) fresh(source.config, source.params, false);
    },
    demo: "mlp",
    setDemo(demo) {
      set({ demo });
      const { source } = get();
      if (source) freshDie(source.params);
    },
    handle: null,
    design: null,
    info: null,
    path: [],
    area: null,
    setArea: (area) => set({ area, selected: null }),
    snapshot: {},
    cycle: 0,
    script: [],
    index: 0,
    weightsText: "3, -2, 5, 1, -4, 7, 0, 2",
    activationsText: "100, -50, 25, 8, -12, 30, 99, -1",
    inputError: null,
    selected: null,
    source: null,

    rebuild: (config, params) => fresh(config, params, false),

    setWeights(weightsText) {
      set({ weightsText });
      const { source } = get();
      if (source) fresh(source.config, source.params, true);
    },

    setActivations(activationsText) {
      set({ activationsText });
      const { source } = get();
      if (source) fresh(source.config, source.params, true);
    },

    restart() {
      const { source } = get();
      if (source) fresh(source.config, source.params, true);
    },

    step() {
      const { die, scope, dieDone } = get();
      if (scope === "die") {
        if (!die || dieDone) return;
        if (die.tick()) set({ dieDone: true, dieResults: JSON.parse(die.results()) });
        refresh();
        return;
      }
      const { handle, script, index } = get();
      if (!handle || index >= script.length) return;
      for (const [port, value] of script[index].inputs) handle.set(port, value);
      handle.tick();
      set({ index: index + 1 });
      refresh();
    },

    runAll() {
      const { die, scope, dieDone } = get();
      if (scope === "die") {
        if (!die || dieDone) return;
        for (let i = 0; i < 100000; i++) {
          if (die.tick()) {
            set({ dieDone: true, dieResults: JSON.parse(die.results()) });
            break;
          }
        }
        refresh();
        return;
      }
      const { handle, script } = get();
      if (!handle) return;
      for (let i = get().index; i < script.length; i++) {
        for (const [port, value] of script[i].inputs) handle.set(port, value);
        handle.tick();
      }
      set({ index: script.length });
      refresh();
    },

    enter(instance) {
      set({ path: [...get().path, instance], selected: null, area: null });
      refresh();
    },

    goTo(depth) {
      set({ path: get().path.slice(0, depth), selected: null, area: null });
      refresh();
    },

    select: (selected) => set({ selected }),
  };
});

/** The module shown at the current path. */
export function currentModule(design: Design | null, path: string[]) {
  if (!design) return null;
  let module = design.modules[design.top];
  for (const segment of path) {
    const instance = module.instances.find((i) => i.name === segment);
    if (!instance || instance.kind.kind !== "module") return null;
    module = design.modules[instance.kind.of];
  }
  return module;
}
