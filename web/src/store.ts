import { create } from "zustand";
import {
  builtinModels,
  defaultParams,
  designPresets,
  estimateDesign,
  fitDesign,
  optimizationCatalog,
  techNodes,
  type ChipParams,
  type ModelSource,
  type Optimization,
  type Preset,
  type Report,
  type TechNode,
} from "./engine";

export type Selection =
  | { kind: "die"; die: number }
  | { kind: "array"; die: number }
  | { kind: "block"; die: number; id: string }
  | { kind: "cell"; die: number; row: number; col: number }
  | null;

export type Overlay = "function" | "part" | "layer" | "activity" | "energy" | "data" | "density";
export type SidebarTab = "parameters" | "optimizations";
export type View = "floorplan" | "logic" | "schematic" | "simulator" | "images" | "listen" | "speak" | "explorer" | "optimize" | "models" | "numeric" | "parameters" | "viewer";
export type Appearance = "auto" | "light" | "dark";
import { readLanguage, writeLanguage, type Language } from "./i18n";
import { LINK } from "./link";

/** Whether a view belongs to the design: Listen only when the chip listens, Speak only when it
 * speaks, Images only when it draws. A chip that does none shows nothing of them. */
export function viewShown(view: View, params: ChipParams): boolean {
  if (view === "listen") return params.speech_in === true;
  if (view === "speak") return params.speech_out === true;
  if (view === "images") return params.draw_images === true;
  return true;
}

export interface SavedDesign {
  name: string;
  model: string;
  params: ChipParams;
}

interface DesignState {
  models: ModelSource[];
  nodes: TechNode[];
  catalog: Optimization[];
  presets: Preset[];
  defaults: ChipParams;
  model: ModelSource | null;
  params: ChipParams;
  report: Report | null;
  error: string | null;
  estimateMs: number;
  past: ChipParams[];
  future: ChipParams[];
  lastEdit: { key: string; at: number };
  baseline: { name: string; report: Report } | null;
  saved: SavedDesign[];
  designName: string;
  selection: Selection;
  overlay: Overlay;
  view: View;
  appearance: Appearance;
  language: Language;
  sidebarTab: SidebarTab;
  help: string | null;
  /** The parameter the documentation shows first. */
  docsFocus: string | null;
  revealed: { key: string; at: number } | null;

  bootstrap: () => void;
  setParam: (key: string, value: unknown) => void;
  /** Several parameters at once, as one step to undo. */
  setParams: (patch: Record<string, unknown>) => void;
  resetParam: (key: string) => void;
  toggleOptimization: (id: string) => void;
  selectModel: (id: string) => void;
  addModel: (name: string, config: string, path?: string) => void;
  reset: () => void;
  undo: () => void;
  redo: () => void;
  applyPreset: (id: string) => void;
  saveDesign: (name: string) => void;
  /** Starts a design from a model and the parameters the wizard chose, and keeps it. */
  createProject: (name: string, model: string, params: Partial<ChipParams>) => void;
  loadDesign: (name: string) => void;
  deleteDesign: (name: string) => void;
  pinBaseline: () => void;
  clearBaseline: () => void;
  setSelection: (s: Selection) => void;
  setOverlay: (o: Overlay) => void;
  setView: (v: View) => void;
  setAppearance: (a: Appearance) => void;
  setLanguage: (l: Language) => void;
  setSidebarTab: (t: SidebarTab) => void;
  setHelp: (text: string | null) => void;
  openDocs: (key: string) => void;
  reveal: (key: string) => void;
}

const STORAGE_KEY = "llm2chip.designs";
const APPEARANCE_KEY = "llm2chip.appearance";
const COALESCE_MS = 500;

function readSaved(): SavedDesign[] {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY) ?? "[]");
  } catch {
    return [];
  }
}

function writeSaved(designs: SavedDesign[]) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(designs));
  } catch {
    // Saving is a convenience; the session keeps working without it.
  }
}

function readAppearance(): Appearance {
  try {
    const stored = localStorage.getItem(APPEARANCE_KEY);
    return stored === "light" || stored === "dark" ? stored : "auto";
  } catch {
    return "auto";
  }
}

function applyAppearance(appearance: Appearance) {
  if (appearance === "auto") delete document.documentElement.dataset.theme;
  else document.documentElement.dataset.theme = appearance;
}

const ADDED_MODELS_KEY = "llm2chip-added-models";
/** Models added from a file; `archivo:` is how earlier versions named them. */
const isAddedFile = (id: string) => id.startsWith("file:") || id.startsWith("archivo:");

/** Models chosen from a config or from Hugging Face, kept between visits. */
function readAddedModels(): ModelSource[] {
  try {
    return JSON.parse(localStorage.getItem(ADDED_MODELS_KEY) ?? "[]");
  } catch {
    return [];
  }
}

/** The models listed, with the added ones last: one added under the name of another, such as
 * the real stories260K the online demo loads beside the built-in one, takes its place. */
export function withoutReplaced(models: ModelSource[], added: ModelSource[]): ModelSource[] {
  const names = new Set(added.map((m) => m.name));
  return [...models.filter((m) => !names.has(m.name)), ...added];
}

function writeAddedModels(models: ModelSource[]) {
  try {
    localStorage.setItem(ADDED_MODELS_KEY, JSON.stringify(models.filter((m) => isAddedFile(m.id))));
  } catch {
    // Without storage they simply last until the page reloads.
  }
}

function readHash(): { model: string; params: ChipParams } | null {
  const match = location.hash.match(/^#d=(.+)$/);
  if (!match) return null;
  try {
    return JSON.parse(decodeURIComponent(atob(match[1])));
  } catch {
    return null;
  }
}

let hashTimer: ReturnType<typeof setTimeout> | undefined;
function writeHash(model: ModelSource | null, params: ChipParams) {
  clearTimeout(hashTimer);
  hashTimer = setTimeout(() => {
    if (!model || isAddedFile(model.id)) return;
    const payload = btoa(encodeURIComponent(JSON.stringify({ model: model.id, params })));
    history.replaceState(null, "", `#d=${payload}`);
  }, 300);
}

/** The parameters sized for a model, or none if the engine cannot read it. */
function fitted(model: ModelSource, params: ChipParams): Partial<ChipParams> {
  try {
    return fitDesign(model, params);
  } catch {
    return {};
  }
}

function run(model: ModelSource | null, params: ChipParams) {
  if (!model) return { report: null, error: null, estimateMs: 0 };
  const start = performance.now();
  writeHash(model, params);
  try {
    const report = estimateDesign(model, params);
    return { report, error: null, estimateMs: performance.now() - start };
  } catch (e) {
    return { error: String(e).replace(/^Error: /, ""), estimateMs: performance.now() - start };
  }
}

export const useDesign = create<DesignState>((set, get) => {
  /** Applies new parameters; rapid edits of one control collapse into one undo step. */
  const commit = (params: ChipParams, key: string, extra: Partial<DesignState> = {}) => {
    const { past, params: previous, lastEdit, model } = get();
    const now = performance.now();
    const coalesce = key === lastEdit.key && key !== "" && now - lastEdit.at < COALESCE_MS;
    set({
      params,
      past: coalesce ? past : [...past.slice(-99), previous],
      future: [],
      lastEdit: { key, at: now },
      ...run(extra.model === undefined ? model : extra.model, params),
      ...extra,
    });
  };

  return {
    models: [],
    nodes: [],
    catalog: [],
    presets: [],
    defaults: { optimizations: [] },
    model: null,
    params: { optimizations: [] },
    report: null,
    error: null,
    estimateMs: 0,
    past: [],
    future: [],
    lastEdit: { key: "", at: 0 },
    baseline: null,
    saved: [],
    designName: "Untitled",
    selection: null,
    overlay: "function",
    view: LINK?.view ?? "floorplan",
    appearance: "auto",
    language: readLanguage(),
    sidebarTab: "parameters",
    help: null,
    docsFocus: null,
    revealed: null,

    bootstrap() {
      const models = withoutReplaced(builtinModels(), readAddedModels());
      const defaults = defaultParams();
      const shared = readHash();
      const model = models.find((m) => m.id === shared?.model) ?? models[0];
      const params = shared ? { ...defaults, ...shared.params } : defaults;
      const appearance = readAppearance();
      applyAppearance(appearance);
      set({
        appearance,
        models,
        defaults,
        nodes: techNodes(),
        catalog: optimizationCatalog(),
        presets: designPresets(),
        saved: readSaved(),
        model,
        params,
        ...run(model, params),
      });
    },

    setParam: (key, value) => commit({ ...get().params, [key]: value }, key),
    setParams: (patch) => commit({ ...get().params, ...patch }, ""),

    resetParam(key) {
      commit({ ...get().params, [key]: get().defaults[key] }, "");
    },

    toggleOptimization(id) {
      const current = get().params.optimizations;
      const optimizations = current.includes(id) ? current.filter((o) => o !== id) : [...current, id];
      commit({ ...get().params, optimizations }, "");
    },

    selectModel(id) {
      const model = get().models.find((m) => m.id === id) ?? null;
      if (!model || model.id === get().model?.id) return;
      // A design moves to another model sized for it: its limits, context and units.
      commit({ ...get().params, ...fitted(model, get().params) } as ChipParams, "", { model, selection: null });
    },

    addModel(name, config, path) {
      const model = { id: `file:${name}`, name, config, ...(path ? { path } : {}) };
      const added = withoutReplaced(get().models.filter((m) => m.id !== model.id), [model]);
      writeAddedModels(added);
      set({ models: added });
      commit({ ...get().params, ...fitted(model, get().params) } as ChipParams, "", { model, selection: null });
    },

    reset: () => commit(get().defaults, "", { selection: null, designName: "Untitled" }),

    undo() {
      const { past, future, params, model } = get();
      const previous = past[past.length - 1];
      if (!previous) return;
      set({ params: previous, past: past.slice(0, -1), future: [params, ...future], lastEdit: { key: "", at: 0 }, ...run(model, previous) });
    },

    redo() {
      const { past, future, params, model } = get();
      const next = future[0];
      if (!next) return;
      set({ params: next, past: [...past, params], future: future.slice(1), lastEdit: { key: "", at: 0 }, ...run(model, next) });
    },

    applyPreset(id) {
      const preset = get().presets.find((p) => p.id === id);
      if (!preset) return;
      const model = get().models.find((m) => m.id === preset.model) ?? get().model;
      commit({ ...get().defaults, ...preset.params } as ChipParams, "", { model, selection: null, designName: preset.name });
    },

    createProject(name, modelId, params) {
      const model = get().models.find((m) => m.id === modelId) ?? get().model;
      commit({ ...get().defaults, ...params } as ChipParams, "", { model, selection: null, designName: name, view: "floorplan" });
      get().saveDesign(name);
    },

    saveDesign(name) {
      const { saved, model, params } = get();
      if (!model) return;
      const next = [...saved.filter((d) => d.name !== name), { name, model: model.id, params }];
      writeSaved(next);
      set({ saved: next, designName: name });
    },

    loadDesign(name) {
      const design = get().saved.find((d) => d.name === name);
      if (!design) return;
      const model = get().models.find((m) => m.id === design.model) ?? get().model;
      commit({ ...get().defaults, ...design.params }, "", { model, selection: null, designName: name });
    },

    deleteDesign(name) {
      const next = get().saved.filter((d) => d.name !== name);
      writeSaved(next);
      set({ saved: next });
    },

    pinBaseline() {
      const { report, designName } = get();
      if (report) set({ baseline: { name: designName, report } });
    },

    clearBaseline: () => set({ baseline: null }),
    setSelection: (selection) => set({ selection }),
    setOverlay: (overlay) => set({ overlay }),
    setView: (view) => set({ view }),
    setAppearance(appearance) {
      applyAppearance(appearance);
      try {
        localStorage.setItem(APPEARANCE_KEY, appearance);
      } catch {
        // The choice still applies to this session.
      }
      set({ appearance });
    },
    setLanguage(language) {
      writeLanguage(language);
      set({ language });
    },
    setSidebarTab: (sidebarTab) => set({ sidebarTab }),
    setHelp: (help) => set({ help }),
    openDocs: (key) => set({ view: "parameters", docsFocus: key }),
    reveal: (key) => set({ sidebarTab: "parameters", revealed: { key, at: performance.now() } }),
  };
});
