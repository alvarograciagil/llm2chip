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
}

export type Numeric = "float" | "fixed" | "exact";

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
  loadReport: { segments: number; words: number; cycles: number; seconds: number; created: boolean; crc_errors: string[] } | null;
  chip: { cycles: number; dieCycles: number; tokenizerMatch: boolean | null } | null;
  acceptance: { hits: number[]; tries: number[] };
  prompt: string;
  maxTokens: number;
  temperature: number;
  topK: number;
  output: string;
  running: boolean;
  phase: string;
  error: string | null;
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
  loadReport: null,
  chip: null,
  acceptance: { hits: [], tries: [] },
  prompt: "¿Cuál es la capital de Francia? Responde en una frase.",
  maxTokens: 64,
  temperature: 0,
  topK: 20,
  output: "",
  running: false,
  phase: "",
  error: null,
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
    const { path, numeric, activationBits, stateBits, tableBits, useDesignFormat, designFormat, useGpu, transactionLoad } = get();
    set({ loading: true, error: null });
    const body = {
      path,
      numeric: numeric !== "float" ? { activation_bits: activationBits, state_bits: stateBits, table_bits: tableBits } : null,
      exact: numeric === "exact",
      format: useDesignFormat ? designFormat : null,
      gpu: useGpu,
      transaction: transactionLoad,
    };
    try {
      const response = await fetch("/api/load", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
      if (!response.ok) throw new Error(await message(response));
      const { loaded, accelerator, load_report: loadReport } = await response.json();
      set({ loaded, accelerator, loadReport: loadReport ?? null, loading: false, numericLoaded: numeric });
    } catch (e) {
      set({ error: e instanceof Error ? e.message : String(e), loading: false });
    }
  },

  generate() {
    const { prompt, maxTokens, temperature, topK, mtpDepth, thinking, thinkBudget, firmware, samplerUnit, firmwareTokenizer } = get();
    const budget = thinkBudget.trim() === "" ? null : Math.max(0, Math.floor(Number(thinkBudget) || 0));
    const metric = (id: string) => useDesign.getState().report?.metrics.find((m) => m.id === id)?.value ?? 0;
    const rate = metric("tok_s");
    const fidelity = { firmware, sampler_unit: samplerUnit || firmware, cycles_per_token: rate > 0 ? (metric("freq_ghz") * 1e9) / rate : 0, sampler_lanes: useDesign.getState().params.sampler_lanes, tokenizer: firmware && firmwareTokenizer, context_tokens: useDesign.getState().params.context_tokens };
    socket?.close();
    set({ chip: null, output: "", running: true, error: null, phase: "Conectando", tokens: 0, prefilled: 0, promptTokens: 0, seconds: [], saturated: 0, acceptance: { hits: [], tries: [] } });
    const protocol = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${protocol}://${location.host}/api/generate`);
    socket = ws;
    ws.onopen = () => ws.send(JSON.stringify({ prompt, max_tokens: maxTokens, temperature, top_k: topK, thinking, think_budget: budget, mtp_depth: mtpDepth, fidelity }));
    ws.onerror = () => set({ error: "Se perdió la conexión con el motor.", running: false });
    ws.onclose = () => set({ running: false });
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
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
        set({ phase: "Terminado", running: false, chip: data.chip ? { cycles: data.chip.cycles, dieCycles: data.chip.die_cycles, tokenizerMatch: data.chip.tokenizer_match ?? null } : null });
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
