import { create } from "zustand";
import { CellSimulation, DieSimulation, type CellInfo, type Design } from "../engine";

export interface ScriptStep {
  label: string;
  /** What this step does in the circuit, with its values, in plain words. */
  explain: string;
  inputs: [string, string][];
}

export type Scope = "cell" | "die";

export type DieDemo = "mlp" | "attention" | "delta" | "boot";

export interface DieResults {
  got: number[];
  expected: number[];
  exact: boolean;
  tolerance: number;
  text?: string;
}

interface CellSimState {
  scope: Scope;
  die: DieSimulation | null;
  dieCounts: Record<string, number> | null;
  dieDone: boolean;
  dieResults: DieResults | null;
  /** The microcode word the die runs now, and what it does. */
  dieActivity: { pc: number; text: string } | null;
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

const TICKS_PER_SLICE = 1500;
const MAX_TICKS = 400_000;
let pumping = false;

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
  const steps: ScriptStep[] = [{ label: "Reset", explain: "clear = 1 empties the accumulator and takes the address counter to 0. With load_en and run at 0, the memory and the multiplier do nothing.", inputs: [["clear", "1"], ...off, ...(usesShifter(info) ? ([["exp_base", "0"]] as [string, string][]) : [])] }];
  if (info.config.coding === "codebook" && !info.config.fixed_weights) {
    for (let i = 0; i < 2 ** info.config.weight_bits; i++) {
      const signed = BigInt.asIntN(info.config.weight_bits, BigInt(i));
      steps.push({ label: `Table ${i + 1}`, explain: `Writes into the cell's table the real value of code ${signed}: the weights keep indices into this table.`, inputs: [["clear", "0"], ["table_en", "1"], ["table_addr", String(i)], ["table_data", toUnsigned(signed, 16)]] });
    }
    steps.push({ label: "End of the table", explain: "table_en = 0: the table stays fixed and the cell only reads it.", inputs: [["table_en", "0"]] });
  }
  if (!info.config.fixed_weights) weights.forEach((w, i) =>
    steps.push({
      label: `Loading weights ${i + 1}/${weights.length}`,
      explain: `load_en = 1 writes code ${w} (${BigInt.asUintN(weight_bits, w).toString(2).padStart(weight_bits, "0")} in ${weight_bits} bits) at address ${i} of the weight memory. That is how each weight reaches its cell when the chip boots.`,
      inputs: [["clear", "0"], ["load_en", "1"], ["load_addr", toUnsigned(BigInt(i), addr_bits)], ["load_data", toUnsigned(w, weight_bits)]],
    }),
  );
  steps.push({ label: info.config.fixed_weights ? "Weights already burnt in" : "End of loading", explain: info.config.fixed_weights ? "The weights are burnt into the ROM at the factory: there is no loading. clear = 1 leaves the accumulator at zero." : "Loading ends and clear = 1 takes the counter back to address 0, so the cell reads the weights in the same order the activations arrive.", inputs: [["clear", "1"], ...off] });
  let partial = 0n;
  activations.forEach((x, i) => {
    partial += weights[i] * x;
    steps.push({
      label: `Multiplication ${i + 1}/${activations.length}`,
      explain: `run = 1: the activation network brings x = ${x}; the cell reads weight ${i} (${weights[i]}) from its memory, multiplies them (${weights[i]} × ${x} = ${weights[i] * x}) and the counter moves to address ${i + 1}. ${info.latency > 0 ? `The product goes through ${info.latency} register stages before being added` : "The product is added at once"}: the reference sum so far is ${partial}.`,
      inputs: [["clear", "0"], ["load_en", "0"], ["run", "1"], ["x", toUnsigned(x, activation_bits)]],
    });
  });
  for (let i = 0; i < info.latency; i++) {
    steps.push({ label: `Draining stages ${i + 1}/${info.latency}`, explain: `No more activations come in (run = 0), but there are still products in the multiplier's and adder's stages: in ${info.latency - i} more cycle${info.latency - i === 1 ? "" : "s"} the accumulator will have the whole sum, ${partial}.`, inputs: [["clear", "0"], ...off] });
  }
  // A cell with its weights in ROM has no loading ports to drive.
  return info.config.fixed_weights ? steps.map((s) => ({ ...s, inputs: s.inputs.filter(([port]) => !port.startsWith("load_")) })) : steps;
}

export function validateInputs(info: CellInfo, weightsText: string, activationsText: string): { error: string } | { weights: bigint[]; activations: bigint[] } {
  const weights = parseList(weightsText);
  const activations = parseList(activationsText);
  if (!weights) return { error: "Write the weights as integers separated by commas." };
  if (!activations) return { error: "Write the activations as integers separated by commas." };
  if (weights.length !== activations.length) return { error: `There are ${weights.length} weights and ${activations.length} activations: they must be the same number.` };
  const { weight_bits, activation_bits, addr_bits } = info.config;
  if (weights.length > 2 ** addr_bits) return { error: `The cell's memory only takes ${2 ** addr_bits} weights.` };
  const fits = (v: bigint, bits: number) => BigInt.asIntN(bits, v) === v;
  if (!weights.every((w) => fits(w, weight_bits))) return { error: `Some weight does not fit in ${weight_bits} signed bits.` };
  if (!activations.every((x) => fits(x, activation_bits))) return { error: `Some activation does not fit in ${activation_bits} signed bits.` };
  return { weights, activations };
}

export const useCellSim = create<CellSimState>((set, get) => {
  const refresh = () => {
    const { handle, die, scope, path } = get();
    const active = scope === "die" ? die : handle;
    if (!active) return;
    set({ snapshot: JSON.parse(active.snapshot(pathText(path))) ?? {}, cycle: active.cycle(), dieActivity: scope === "die" && die ? JSON.parse(die.activity()) : null });
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
    const handle = get().handle;
    if (info.config.fixed_weights && handle) {
      result.weights.forEach((w, i) => handle.burn("weight_rom", i, toUnsigned(w, info.config.weight_bits)));
      if (info.config.coding === "codebook") {
        for (let i = 0; i < 2 ** info.config.weight_bits; i++) handle.burn("codebook", i, toUnsigned(BigInt.asIntN(info.config.weight_bits, BigInt(i)), 16));
      }
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
    dieActivity: null,
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
        if (!die || dieDone || pumping) return;
        pumping = true;
        let ticks = 0;
        // Works in slices so that a long start-up leaves the interface responsive.
        const slice = () => {
          if (get().die !== die) {
            pumping = false;
            return;
          }
          const end = Math.min(ticks + TICKS_PER_SLICE, MAX_TICKS);
          for (; ticks < end; ticks++) {
            if (die.tick()) {
              pumping = false;
              set({ dieDone: true, dieResults: JSON.parse(die.results()) });
              refresh();
              return;
            }
          }
          refresh();
          if (ticks < MAX_TICKS) setTimeout(slice, 0);
          else pumping = false;
        };
        slice();
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
