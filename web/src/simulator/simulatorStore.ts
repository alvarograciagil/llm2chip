import { create } from "zustand";
import type { Adaptation } from "../models/modelsStore";
import { useDesign } from "../store";

export interface ServerModel {
  name: string;
  path: string;
  architecture: string;
  supported: boolean;
  size_gb: number;
  config: string;
  adaptation: Adaptation;
  /** What the model does with images, and what the simulator cannot follow yet. */
  images: { reads: boolean; makes: boolean; note: string | null };
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
  imageReport: { patches: number; tokens: number; seconds: number } | null;
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

  connect: () => Promise<void>;
  set: (patch: Partial<SimulatorState>) => void;
  load: () => Promise<void>;
  generate: () => void;
  stop: () => void;
}

let socket: WebSocket | null = null;

const message = async (r: Response) => (await r.text()) || `Error ${r.status}`;

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
  prompt: "¿Cuál es la capital de Francia? Responde en una frase.",
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

  async connect() {
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

  async load() {
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
    const { prompt, maxTokens, temperature, topK, thinking, thinkBudget, firmware, samplerUnit, firmwareTokenizer, image, draw } = get();
    // The draft head is part of the chip only when the design has one.
    const mtpDepth = chipDraftDepth(useDesign.getState().params);
    const budget = thinkBudget.trim() === "" ? null : Math.max(0, Math.floor(Number(thinkBudget) || 0));
    const metric = (id: string) => useDesign.getState().report?.metrics.find((m) => m.id === id)?.value ?? 0;
    const rate = metric("tok_s");
    const fidelity = { firmware, sampler_unit: samplerUnit || firmware, cycles_per_token: rate > 0 ? (metric("freq_ghz") * 1e9) / rate : 0, sampler_lanes: useDesign.getState().params.sampler_lanes, tokenizer: firmware && firmwareTokenizer, context_tokens: useDesign.getState().params.context_tokens };
    socket?.close();
    set({ ending: null, drawing: null, picture: draw ? null : get().picture, chip: null, output: "", running: true, error: null, phase: "Conectando", tokens: 0, prefilled: 0, promptTokens: 0, seconds: [], saturated: 0, acceptance: { hits: [], tries: [] } });
    const protocol = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${protocol}://${location.host}/api/generate`);
    socket = ws;
    ws.onopen = () => ws.send(JSON.stringify({ prompt, max_tokens: maxTokens, temperature, top_k: topK, thinking, think_budget: budget, mtp_depth: mtpDepth, fidelity, image: image && { width: image.width, height: image.height, rgb: image.rgb }, draw }));
    ws.onerror = () => set({ error: "Se perdió la conexión con el motor.", running: false });
    ws.onclose = () => set({ running: false });
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === "image") set({ imageReport: { patches: data.patches, tokens: data.tokens, seconds: data.seconds } });
      if (data.type === "drawing") set((s) => ({ phase: "Dibujando", drawing: { done: data.done, total: data.total, seconds: data.seconds, decoding: false }, pulse: s.pulse + 1 }));
      if (data.type === "decoding") set((s) => ({ phase: "Decodificando la imagen", drawing: s.drawing && { ...s.drawing, decoding: true } }));
      if (data.type === "picture") set({ picture: { data: data.data, side: data.side, seconds: data.seconds } });
      if (data.type === "start") set({ promptTokens: data.prompt_tokens, phase: "Leyendo el prompt" });
      if (data.type === "prefill") set((s) => ({ prefilled: data.done, saturated: s.saturated, pulse: s.pulse + 1 }));
      if (data.type === "token") {
        set((s) => ({
          output: s.output + data.text,
          tokens: data.index,
          seconds: [...s.seconds, data.seconds],
          saturated: s.saturated + data.saturated,
          phase: "Generando",
          pulse: s.pulse + 1,
          acceptance: data.tries ? { hits: data.hits, tries: data.tries } : s.acceptance,
        }));
      }
      if (data.type === "done") {
        set({ phase: "Terminado", running: false, ending: ["limit", "loop", "context"].includes(data.reason) ? data.reason : null, chip: data.chip ? { cycles: data.chip.cycles, dieCycles: data.chip.die_cycles, tokenizerMatch: data.chip.tokenizer_match ?? null } : null });
        ws.close();
      }
      if (data.type === "error") {
        set({ error: data.message, phase: "", running: false });
        ws.close();
      }
    };
  },

  stop() {
    socket?.close();
    set({ running: false, phase: "Detenido" });
  },
}));
