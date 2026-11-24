import { create } from "zustand";
import type { Adaptation } from "../models/modelsStore";
import { useDesign } from "../store";
import { addActivity, EMPTY_ACTIVITY, readActivity, readCpuInfo, type ChipStep, type CpuInfo, type StepActivity } from "./activity";
import { DEMO } from "../demo";
import { describeModel, forgetModel, keptModels, loadModel, runModel, stopModel } from "./browserChip";
import { LINK_LOAD_LIMIT, PROMPTS, SUGGESTED, type BrowserModel, type BrowserSource } from "./browserModels";
import { LINK } from "../link";

/** Steps kept for the list: the oldest go once there are more. */
const MAX_STEPS = 2000;

export interface ServerModel {
  name: string;
  path: string;
  architecture: string;
  supported: boolean;
  size_gb: number;
  config: string;
  adaptation: Adaptation;
  /** What the model does with images, and what the simulator cannot follow yet. */
  images: { reads: boolean; makes: boolean; edits: boolean; note: string | null };
  /** What it does in a spoken turn: listens (Moonshine), speaks (OuteTTS) or sounds (DAC). */
  voice?: "listens" | "speaks" | "sounds" | null;
}

export type Numeric = "float" | "fixed" | "exact";

/** The integer formats of the design, as the engine takes them. */
export function chipNumeric(p: Record<string, unknown>) {
  return { activation_bits: Number(p.activation_bits), state_bits: Number(p.delta_state_bits), table_bits: Number(p.function_table_bits) };
}

/** The weight format of the design, as the engine expects it. */
export function chipFormat(p: Record<string, unknown>): WeightFormat {
  const coding = String(p.weight_coding);
  const grouped = coding === "affine" || coding === "float";
  return {
    coding,
    bits: Number(p.weight_bits),
    exponent_bits: Number(p.weight_exponent_bits),
    bias: p.weight_bias === true,
    group_size: grouped ? Number(p.group_size) : 0,
    fine_scale: p.weight_fine_scale === true,
    keep_source: coding === "affine" && p.use_model_mix === true,
  };
}

/** What a load depends on: the weights and the design's formats. */
export function loadSignature(path: string, p: Record<string, unknown>): string {
  return JSON.stringify([path, chipNumeric(p), chipFormat(p)]);
}

/** How deep the design's draft head guesses, or 0 without one. */
export function chipDraftDepth(p: Record<string, unknown>): number {
  return p.mtp_enabled === true ? Math.min(3, Number(p.mtp_depth) || 0) : 0;
}

export interface WeightFormat {
  coding: string;
  bits: number;
  exponent_bits: number;
  bias: boolean;
  group_size: number;
  fine_scale: boolean;
  keep_source: boolean;
}

interface SimulatorState {
  online: boolean | null;
  models: ServerModel[];
  path: string;
  loaded: string | null;
  accelerator: string | null;
  useGpu: boolean;
  loading: boolean;
  numeric: Numeric;
  numericLoaded: Numeric | null;
  activationBits: number;
  stateBits: number;
  tableBits: number;
  useDesignFormat: boolean;
  designFormat: WeightFormat | null;
  mtpDepth: number;
  /** The model reasons before answering, if its template allows it. */
  thinking: boolean;
  /** Most reasoning tokens before the chip closes the reasoning itself; empty is unbounded. */
  thinkBudget: string;
  firmware: boolean;
  samplerUnit: boolean;
  firmwareTokenizer: boolean;
  transactionLoad: boolean;
  /** What the loaded engine was built for: a different one means the design changed since. */
  loadedSignature: string | null;
  loadReport: { segments: number; words: number; cycles: number; seconds: number; created: boolean; crc_errors: string[] } | null;
  chip: { cycles: number; dieCycles: number; tokenizerMatch: boolean | null } | null;
  acceptance: { hits: number[]; tries: number[] };
  prompt: string;
  /** A picture the question is about: its pixels as RGB, and a preview to show. */
  image: { width: number; height: number; rgb: number[]; preview: string } | null;
  /** What the encoder made of the picture in the last run. */
  imageReport: { patches: number; tokens: number; seconds: number; onChip: boolean } | null;
  /** Draw a picture of the prompt instead of answering it, with a model that draws. */
  draw: boolean;
  /** How far the drawing has gone: image tokens written, and whether it is decoding. */
  drawing: { done: number; total: number; seconds: number; decoding: boolean } | null;
  /** The last picture drawn, ready for an <img>. */
  picture: { data: string; side: number; seconds: number } | null;
  maxTokens: number;
  temperature: number;
  topK: number;
  output: string;
  running: boolean;
  phase: string;
  error: string | null;
  /** Why the last answer ended, when it was not the model's own end token. */
  ending: "limit" | "loop" | "context" | null;
  promptTokens: number;
  prefilled: number;
  tokens: number;
  seconds: number[];
  saturated: number;
  pulse: number;
  /** What the chip's CPU is, when the run goes through it. */
  cpuInfo: CpuInfo | null;
  /** What the chip did in each step, with the CPU and its firmware. */
  steps: ChipStep[];
  /** All of it added up since the run began. */
  totals: StepActivity | null;
  /** In the online demo: the models it can run in the browser, as described. */
  browserModels: BrowserModel[];
  /** In the online demo: how the download or the build of the chip goes. */
  download: { phase: string; done: number; total: number } | null;
  /** In the online demo: why a model asked for cannot run here. */
  browserRefusal: string | null;
  /** In the online demo: the bytes of each model the browser keeps, and of everything kept. */
  kept: Record<string, number>;
  keptTotal: number;
  /** In the online demo: the last load had no room to keep a copy. */
  noRoom: boolean;
  /** In the online demo: whether downloads stay in the browser; by default they do not. */
  keepModels: boolean;
  setKeepModels: (keep: boolean) => void;

  connect: () => Promise<void>;
  /** In the online demo: adds a model from Hugging Face if the browser can run it. */
  addBrowserModel: (source: BrowserSource) => Promise<BrowserModel | null>;
  /** In the online demo: makes a model the one simulated, and the design's model. */
  chooseBrowserModel: (model: BrowserModel) => void;
  /** In the online demo: follows a link to the simulator opened in the page already. */
  openLink: (model: string | null) => Promise<void>;
  /** In the online demo: deletes the browser's copy of a model. */
  forgetBrowserModel: (model: BrowserModel | null) => Promise<void>;
  set: (patch: Partial<SimulatorState>) => void;
  load: () => Promise<void>;
  generate: () => void;
  stop: () => void;
}

const KEEP_KEY = "llm2chip.keepModels";

function readKeep(): boolean {
  try {
    return localStorage.getItem(KEEP_KEY) === "1";
  } catch {
    return false;
  }
}

/** The question the simulator starts with. */
const QUESTION = "What is the capital of France? Answer in one sentence.";

let socket: WebSocket | null = null;
let described: Promise<void> | null = null;

/** A model the browser runs, as the list of the engine's models shows one. */
function asServerModel(model: BrowserModel): ServerModel {
  return {
    name: model.name,
    path: model.id,
    architecture: model.architecture,
    supported: true,
    size_gb: model.bytes / 1e9,
    config: model.config,
    adaptation: { design: true, simulate: true, approximate: false, reason: null, parameters: null, layers: null },
    images: { reads: false, makes: false, edits: false, note: null },
  };
}

const message = async (r: Response) => (await r.text()) || `Error ${r.status}`;

type Setter = (patch: Partial<SimulatorState> | ((s: SimulatorState) => Partial<SimulatorState>)) => void;
type Getter = () => SimulatorState;

/** What a new run starts from. */
const FRESH_RUN: Partial<SimulatorState> = { ending: null, drawing: null, chip: null, output: "", running: true, error: null, phase: "Connecting", tokens: 0, prefilled: 0, promptTokens: 0, seconds: [], saturated: 0, acceptance: { hits: [], tries: [] }, cpuInfo: null, steps: [], totals: null };

/** A step the chip reported, with what it did, at the end of the list. */
const step = (kind: ChipStep["kind"], index: number, text: string, data: Record<string, unknown>) => (s: SimulatorState) => {
  if (!data.activity) return {};
  const activity = readActivity(data.activity);
  return { steps: [...s.steps, { kind, index, text, cycles: Number(data.cycles) || 0, activity }].slice(-MAX_STEPS), totals: addActivity(s.totals ?? EMPTY_ACTIVITY, activity) };
};

/** Applies an event of a run, from the local engine or from the simulator in the browser;
 * `close` ends the run's connection. */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
function handleEvent(set: Setter, data: any, close: () => void) {
  if (data.type === "image") set({ imageReport: { patches: data.patches, tokens: data.tokens, seconds: data.seconds, onChip: Boolean(data.on_chip) } });
  if (data.type === "drawing") set((s) => ({ phase: "Drawing", drawing: { done: data.done, total: data.total, seconds: data.seconds, decoding: false }, pulse: s.pulse + 1 }));
  if (data.type === "decoding") set((s) => ({ phase: "Decoding the image", drawing: s.drawing && { ...s.drawing, decoding: true } }));
  if (data.type === "picture") set({ picture: { data: data.data, side: data.side, seconds: data.seconds } });
  if (data.type === "start") set({ promptTokens: data.prompt_tokens, phase: "Reading the prompt" });
  if (data.type === "cpu") set({ cpuInfo: readCpuInfo(data) });
  if (data.type === "prefill") set((s) => ({ prefilled: data.done, saturated: s.saturated, pulse: s.pulse + 1, ...step("prefill", data.done, "", data)(s) }));
  if (data.type === "token") {
    set((s) => ({
      output: s.output + data.text,
      tokens: data.index,
      seconds: [...s.seconds, data.seconds],
      saturated: s.saturated + data.saturated,
      phase: "Generating",
      pulse: s.pulse + 1,
      acceptance: data.tries ? { hits: data.hits, tries: data.tries } : s.acceptance,
      ...step("token", data.index, data.text, data)(s),
    }));
  }
  if (data.type === "done") {
    // What the firmware did after its last token: closing the answer.
    if (data.activity) set((s) => ({ totals: addActivity(s.totals ?? EMPTY_ACTIVITY, readActivity(data.activity)) }));
    set({ phase: "Finished", running: false, ending: ["limit", "loop", "context"].includes(data.reason) ? data.reason : null, chip: data.chip ? { cycles: data.chip.cycles, dieCycles: data.chip.die_cycles, tokenizerMatch: data.chip.tokenizer_match ?? null } : null });
    close();
  }
  if (data.type === "error") {
    set({ error: data.message, phase: "", running: false });
    close();
  }
}

/** The online demo: the chip runs in the browser on WebAssembly, with the design's arithmetic. */
async function loadInBrowser(get: Getter, set: Setter) {
  const { path, browserModels } = get();
  const model = browserModels.find((m) => m.id === path);
  if (!model) return;
  const params = useDesign.getState().params;
  const signature = loadSignature(path, params);
  set({ loading: true, error: null, download: null });
  try {
    const setup = { activation_bits: chipNumeric(params).activation_bits, format: chipFormat(params), context_tokens: Number(params.context_tokens) || 0 };
    const keep = get().keepModels;
    const { kept, threads } = await loadModel(model, setup, keep, (phase, done, total) => set({ download: { phase, done, total } }));
    await refreshKept(get, set);
    set({ noRoom: keep && !kept, loaded: model.name, accelerator: threads > 1 ? `WebAssembly, ${threads} threads` : "WebAssembly", loadReport: null, loading: false, download: null, numericLoaded: "exact", loadedSignature: signature });
  } catch (e) {
    set({ error: e instanceof Error ? e.message : String(e), loading: false, download: null });
  }
}

async function refreshKept(get: Getter, set: Setter, sweep = false) {
  try {
    const { models, total } = await keptModels(get().browserModels, sweep);
    set({ kept: models, keptTotal: total });
  } catch {
    set({ kept: {}, keptTotal: 0 });
  }
}

function generateInBrowser(get: Getter, set: Setter) {
  const { prompt, maxTokens, temperature, topK, thinking, thinkBudget } = get();
  const budget = thinkBudget.trim() === "" ? null : Math.max(0, Math.floor(Number(thinkBudget) || 0));
  set({ ...FRESH_RUN, image: null, draw: false });
  runModel({ prompt, max_tokens: maxTokens, temperature, top_k: topK, thinking, think_budget: budget, seed: Math.floor(Math.random() * 2 ** 32) }, (event) => handleEvent(set, event, () => undefined));
}

/** A link to the simulator: chooses its model, stories260K if it names none, and runs it if
 * it is small enough to download without asking. */
async function followLink(name: string | null, get: Getter) {
  const wanted = name ?? "stories260K";
  const { browserModels, addBrowserModel } = get();
  const model = browserModels.find((m) => m.name === wanted || m.id === `hf:${wanted}`) ?? (wanted.includes("/") ? await addBrowserModel({ repo: wanted }) : null);
  if (!model) return;
  // A link followed during a run ends it first; the engine is rebuilt only for another model.
  if (get().running) get().stop();
  get().chooseBrowserModel(model);
  if (model.bytes > LINK_LOAD_LIMIT) return;
  if (get().loaded !== model.name) await get().load();
  if (get().loaded === model.name && !get().error) get().generate();
}

export const useSimulator = create<SimulatorState>((set, get) => ({
  online: null,
  models: [],
  path: "",
  loaded: null,
  accelerator: null,
  useGpu: true,
  loading: false,
  numeric: "float",
  numericLoaded: null,
  activationBits: 16,
  stateBits: 16,
  tableBits: 12,
  useDesignFormat: false,
  designFormat: null,
  mtpDepth: 0,
  thinking: false,
  thinkBudget: "",
  firmware: false,
  samplerUnit: false,
  firmwareTokenizer: false,
  transactionLoad: false,
  loadedSignature: null,
  loadReport: null,
  chip: null,
  acceptance: { hits: [], tries: [] },
  prompt: QUESTION,
  image: null,
  imageReport: null,
  draw: false,
  drawing: null,
  picture: null,
  maxTokens: 64,
  temperature: 0,
  topK: 20,
  output: "",
  running: false,
  phase: "",
  error: null,
  ending: null,
  promptTokens: 0,
  prefilled: 0,
  tokens: 0,
  seconds: [],
  saturated: 0,
  pulse: 0,
  cpuInfo: null,
  steps: [],
  totals: null,
  browserModels: [],
  download: null,
  browserRefusal: null,
  kept: {},
  keptTotal: 0,
  noRoom: false,
  keepModels: readKeep(),
  setKeepModels(keep) {
    try {
      localStorage.setItem(KEEP_KEY, keep ? "1" : "0");
    } catch {
      // Remembered for this visit only.
    }
    set({ keepModels: keep });
  },

  async connect() {
    if (DEMO) {
      // The browser is the engine: it only has to find out what it can run, once.
      if (described) return described;
      described = (async () => {
        set({ online: true });
        await Promise.all(SUGGESTED.map((source) => get().addBrowserModel(source)));
        set({ browserRefusal: null });
        // Once per visit: what was left halfway, or not used for a month, goes.
        await refreshKept(get, set, true);
        if (LINK?.view === "simulator") await followLink(LINK.model, get);
      })();
      return described;
    }
    try {
      const [models, status] = await Promise.all([fetch("/api/models"), fetch("/api/status")]);
      if (!models.ok || !status.ok) throw new Error();
      const list: ServerModel[] = await models.json();
      const { loaded } = await status.json();
      const preferred = list.find((m) => m.supported && m.name.includes("Qwen3.8-27B") && m.name.includes("Speed")) ?? list.find((m) => m.supported);
      set({ online: true, models: list, loaded, path: get().path || preferred?.path || "" });
    } catch {
      set({ online: false });
    }
  },

  set: (patch) => set(patch),

  async addBrowserModel(source) {
    set({ browserRefusal: null });
    try {
      const model = await describeModel(source);
      const others = get().browserModels.filter((m) => m.id !== model.id);
      set({ browserModels: [...others, model], models: [...get().models.filter((m) => m.path !== model.id), asServerModel(model)] });
      if (described) void refreshKept(get, set);
      return model;
    } catch (e) {
      set({ browserRefusal: e instanceof Error ? e.message : String(e) });
      return null;
    }
  },

  chooseBrowserModel(model) {
    useDesign.getState().addModel(model.name, model.config, model.id);
    // A model that only continues text gets a beginning to continue, unless the prompt is one's own.
    const { prompt } = get();
    const ours = prompt === QUESTION || Object.values(PROMPTS).includes(prompt);
    set({ path: model.id, ...(ours ? { prompt: PROMPTS[model.id] ?? QUESTION } : {}) });
  },

  async openLink(model) {
    await get().connect();
    await followLink(model, get);
  },

  async forgetBrowserModel(model) {
    await forgetModel(model).catch(() => undefined);
    await refreshKept(get, set);
  },

  async load() {
    if (DEMO) return loadInBrowser(get, set);
    const { path, useGpu, transactionLoad } = get();
    set({ loading: true, error: null });
    // The simulator runs the chip being designed: its integer arithmetic and its weight format.
    const params = useDesign.getState().params;
    const body = {
      path,
      numeric: chipNumeric(params),
      exact: true,
      format: chipFormat(params),
      gpu: useGpu,
      transaction: transactionLoad,
    };
    const signature = loadSignature(path, params);
    try {
      const response = await fetch("/api/load", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
      if (!response.ok) throw new Error(await message(response));
      const { loaded, accelerator, load_report: loadReport } = await response.json();
      set({ loaded, accelerator, loadReport: loadReport ?? null, loading: false, numericLoaded: "exact", loadedSignature: signature });
    } catch (e) {
      set({ error: e instanceof Error ? e.message : String(e), loading: false });
    }
  },

  generate() {
    if (DEMO) return generateInBrowser(get, set);
    const { prompt, maxTokens, temperature, topK, thinking, thinkBudget, firmware, samplerUnit, firmwareTokenizer, image, draw } = get();
    // The draft head is part of the chip only when the design has one.
    const mtpDepth = chipDraftDepth(useDesign.getState().params);
    const budget = thinkBudget.trim() === "" ? null : Math.max(0, Math.floor(Number(thinkBudget) || 0));
    const metric = (id: string) => useDesign.getState().report?.metrics.find((m) => m.id === id)?.value ?? 0;
    const rate = metric("tok_s");
    const fidelity = { firmware, sampler_unit: samplerUnit || firmware, cycles_per_token: rate > 0 ? (metric("freq_ghz") * 1e9) / rate : 0, sampler_lanes: useDesign.getState().params.sampler_lanes, tokenizer: firmware && firmwareTokenizer, context_tokens: useDesign.getState().params.context_tokens };
    socket?.close();
    set({ ...FRESH_RUN, picture: draw ? null : get().picture });
    const protocol = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${protocol}://${location.host}/api/generate`);
    socket = ws;
    ws.onopen = () => ws.send(JSON.stringify({ prompt, max_tokens: maxTokens, temperature, top_k: topK, thinking, think_budget: budget, mtp_depth: mtpDepth, fidelity, image: image && { width: image.width, height: image.height, rgb: image.rgb }, draw }));
    ws.onerror = () => set({ error: "The connection to the engine was lost.", running: false });
    ws.onclose = () => set({ running: false });
    ws.onmessage = (event) => handleEvent(set, JSON.parse(event.data), () => ws.close());
  },

  stop() {
    if (DEMO) stopModel();
    socket?.close();
    set({ running: false, phase: "Stopped" });
  },
}));
