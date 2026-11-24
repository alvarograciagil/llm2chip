import { create } from "zustand";
import type { ServerModel } from "../simulator/simulatorStore";
import { useDesign } from "../store";

/** The two views: listening (a spoken question, its text and the answer) and speaking (a
 * text, or the answer to it, said aloud). */
export type Mode = "listen" | "speak";

/** The parts the project's chip has. */
export function chipParts(): { listens: boolean; speaks: boolean } {
  const params = useDesign.getState().params;
  return { listens: params.speech_in === true, speaks: params.speech_out === true };
}

/** What each stage of a spoken turn gave, as the engine reports it. */
export interface Stage {
  text?: string;
  seconds: number;
}

export interface Spoken {
  tokens: number;
  seconds: number;
  duration: number;
  audio: string;
}

export interface VoiceResult {
  heard?: Stage;
  answer?: Stage;
  spoken?: Spoken;
  bits?: number | null;
}

/** A die of the planned system: the stages on it and its shape. */
export interface PlannedDie {
  stages: string[];
  count: number;
  units: number;
  lanes: number;
  cells: number;
  mm2: number;
  parts_mm2: [number, number, number, number];
  clock_mhz: number;
}

/** What the planner chose, and every grouping it weighed. */
export interface SystemPlan {
  plan: {
    area_mm2: number;
    /** The first sound, or the answer's first token when the chip only listens. */
    first_output_ms: number;
    /** A turn's energy: each stage's work and every die's leakage while it lasts. */
    energy: { total_j: number; leakage_j: number; seconds: number; stages_j: number[] };
    dies: PlannedDie[];
    stages: { name: string; first_ms: number; per_second: number; variant: string }[];
  } | null;
  weighed: { groups: string[][]; area_mm2: number | null; first_output_ms: number | null; reason: string | null }[];
}

/** Where the stages run: floating point, or the die's arithmetic with vectors of so many bits. */
export type Arithmetic = "float" | "16" | "20" | "24";

interface VoiceState {
  online: boolean | null;
  /** The parts the engine was built with, once known. */
  built: { listen: boolean; speak: boolean } | null;
  models: ServerModel[];
  listen: string;
  think: string;
  speak: string;
  sound: string;
  arithmetic: Arithmetic;
  /** The question, a 16 kHz WAV, and how long it is. */
  question: { wav: string; seconds: number } | null;
  /** The Speak view's text, and whether the language model answers it first. */
  text: string;
  answerFirst: boolean;
  /** The text the Speak view's last turn took. */
  asked: string | null;
  recording: boolean;
  running: boolean;
  phase: string;
  result: VoiceResult | null;
  error: string | null;
  /** The conversation's earlier turns, and whether the next question starts a new one. */
  history: { heard: string; answer: string }[];
  fresh: boolean;
  /** Speak in the voice of the profile kept with OuteTTS. */
  speaker: boolean;
  /** The system's planning: what it must meet, what it is doing and what it chose. */
  firstOutputMs: number;
  questionSeconds: number;
  areaMm2: string;
  planning: boolean;
  planDoing: string;
  plan: SystemPlan | null;
  planError: string | null;
  /** Plan the chip as soon as the models are known: the project wizard asks for it. */
  planOnOpen: boolean;
  set: (partial: Partial<VoiceState>) => void;
  startPlan: () => Promise<void>;
  connect: () => Promise<void>;
  record: () => Promise<void>;
  stop: () => void;
  open: (file: File) => Promise<void>;
  start: (mode: Mode) => Promise<void>;
}

const RATE = 16_000;

/** Samples in [−1, 1] as a 16-bit PCM WAV, mono. */
export function wav(samples: Float32Array, rate: number): Uint8Array {
  const bytes = new Uint8Array(44 + samples.length * 2);
  const view = new DataView(bytes.buffer);
  const text = (at: number, s: string) => [...s].forEach((c, i) => view.setUint8(at + i, c.charCodeAt(0)));
  text(0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  text(8, "WAVEfmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, rate, true);
  view.setUint32(28, rate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  text(36, "data");
  view.setUint32(40, samples.length * 2, true);
  samples.forEach((s, i) => view.setInt16(44 + i * 2, Math.round(Math.max(-1, Math.min(1, s)) * 32767), true));
  return bytes;
}

function base64(bytes: Uint8Array): string {
  let text = "";
  for (let i = 0; i < bytes.length; i += 0x8000) text += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return btoa(text);
}

/** Any sound the browser can decode, as a 16 kHz mono WAV in base64. */
export async function toQuestion(data: ArrayBuffer): Promise<{ wav: string; seconds: number }> {
  const decoded = await new AudioContext().decodeAudioData(data);
  const length = Math.max(1, Math.ceil(decoded.duration * RATE));
  const offline = new OfflineAudioContext(1, length, RATE);
  const source = offline.createBufferSource();
  source.buffer = decoded;
  source.connect(offline.destination);
  source.start();
  const mono = (await offline.startRendering()).getChannelData(0);
  return { wav: base64(wav(mono, RATE)), seconds: decoded.duration };
}

let recorder: MediaRecorder | null = null;
let timer: ReturnType<typeof setInterval> | null = null;
let planTimer: ReturnType<typeof setInterval> | null = null;

/** The first model of the role the stage has, keeping the one chosen if it still is there. */
function pick(models: ServerModel[], chosen: string, keep: (m: ServerModel) => boolean) {
  return models.some((m) => m.path === chosen && keep(m)) ? chosen : (models.find(keep)?.path ?? "");
}

export const useVoice = create<VoiceState>((set, get) => ({
  online: null,
  built: null,
  models: [],
  listen: "",
  think: "",
  speak: "",
  sound: "",
  arithmetic: "float",
  question: null,
  text: "",
  answerFirst: true,
  asked: null,
  recording: false,
  running: false,
  phase: "",
  result: null,
  error: null,
  history: [],
  fresh: true,
  speaker: true,
  firstOutputMs: 500,
  questionSeconds: 3,
  areaMm2: "",
  planning: false,
  planDoing: "",
  plan: null,
  planError: null,
  planOnOpen: false,
  set: (partial) => set(partial),

  startPlan: async () => {
    const { listen, think, speak, sound, firstOutputMs, questionSeconds, areaMm2, speaker } = get();
    const parts = chipParts();
    set({ planning: true, planError: null, plan: null, planDoing: "" });
    // Only the parts the project's chip has.
    const body = { listen: parts.listens ? listen : null, think, speak: parts.speaks ? speak : null, sound: parts.speaks ? sound : null, first_output_ms: firstOutputMs, seconds: questionSeconds, area_mm2: areaMm2.trim() ? Number(areaMm2) : null, speaker: parts.speaks && speaker };
    const response = await fetch("/api/voice/plan", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
    if (!response.ok) {
      set({ planning: false, planError: await response.text() });
      return;
    }
    if (planTimer) clearInterval(planTimer);
    planTimer = setInterval(async () => {
      try {
        const status = await (await fetch("/api/voice/plan")).json();
        if (status.state === "running") {
          set({ planDoing: status.doing });
          return;
        }
        if (planTimer) clearInterval(planTimer);
        planTimer = null;
        if (status.state === "done") {
          set({ planning: false, plan: status.result });
          keepPlan(status.result, parts);
        } else set({ planning: false, planError: status.error ?? null });
      } catch {
        set({ online: false });
      }
    }, 1000);
  },

  connect: async () => {
    try {
      const models = (await (await fetch("/api/models")).json()) as ServerModel[];
      const s = get();
      // The language model is the project's own, when its weights are on this computer.
      const own = useDesign.getState().model?.path;
      const answers = (m: ServerModel) => m.supported && !m.voice;
      const think = own && models.some((m) => m.path === own && answers(m)) ? own : pick(models, s.think, answers);
      set({
        online: true,
        models,
        listen: pick(models, s.listen, (m) => m.voice === "listens"),
        think,
        speak: pick(models, s.speak, (m) => m.voice === "speaks"),
        sound: pick(models, s.sound, (m) => m.voice === "sounds"),
      });
      const reply = await fetch("/api/voice");
      // An engine built with neither part has no voice at all.
      if (!reply.ok) {
        set({ built: { listen: false, speak: false } });
        return;
      }
      const status = await reply.json();
      if (status.state === "idle") set({ built: { listen: status.listen !== false, speak: status.speak !== false } });
      if (status.state === "running" && !get().running) {
        set({ running: true });
        poll(set);
      }
    } catch {
      set({ online: false });
    }
  },

  record: async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const chunks: Blob[] = [];
    recorder = new MediaRecorder(stream);
    recorder.ondataavailable = (e) => chunks.push(e.data);
    recorder.onstop = async () => {
      stream.getTracks().forEach((track) => track.stop());
      set({ recording: false, question: await toQuestion(await new Blob(chunks).arrayBuffer()) });
    };
    recorder.start();
    set({ recording: true, error: null });
  },

  stop: () => recorder?.stop(),

  open: async (file) => set({ question: await toQuestion(await file.arrayBuffer()), error: null }),

  start: async (mode) => {
    const { question, text, answerFirst, listen, think, speak, sound, arithmetic, fresh, result, speaker } = get();
    const parts = chipParts();
    if (mode === "listen" && !question) return;
    if (mode === "speak" && !text.trim()) return;
    // The last turn joins the conversation's history before the next is asked.
    const asked = result?.heard?.text ?? get().asked;
    const last = !fresh && asked && result?.answer?.text ? [{ heard: asked, answer: result.answer.text }] : [];
    set({ running: true, error: null, result: null, phase: mode === "listen" ? "listening" : answerFirst ? "thinking" : "speaking", history: fresh ? [] : [...get().history, ...last] });
    const bits = arithmetic === "float" ? null : Number(arithmetic);
    // Listening: the clip, heard and answered, and said too when the chip also speaks.
    // Speaking: the text, answered first if asked, and said.
    const says = mode === "speak" || parts.speaks;
    const body =
      mode === "listen"
        ? { wav: question!.wav, listen, think, speak: says ? speak : null, sound: says ? sound : null, bits, fresh, speaker: says && speaker }
        : { text, think: answerFirst ? think : null, speak, sound, bits, fresh, speaker };
    set({ asked: mode === "speak" ? text : null });
    const response = await fetch("/api/voice", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
    if (!response.ok) {
      set({ running: false, error: await response.text() });
      return;
    }
    poll(set);
  },
}));

/** The planned voice dies, kept with the project so its estimate and floorplan count them:
 * every die but the language model's, which the estimator lays out. */
export function voiceDies(plan: SystemPlan, parts: { listens: boolean; speaks: boolean }) {
  if (!plan.plan) return null;
  const dies = plan.plan.dies
    .filter((d) => !d.stages.includes("language model"))
    .map(({ stages, count, units, lanes, cells, parts_mm2, clock_mhz }) => ({ stages, count, units, lanes, cells, parts_mm2, clock_mhz }));
  return dies.length ? { ...parts, dies } : null;
}

function keepPlan(plan: SystemPlan, parts: { listens: boolean; speaks: boolean }) {
  const kept = voiceDies(plan, parts);
  if (kept) useDesign.getState().setParams({ voice_plan: kept });
}

function poll(set: (partial: Partial<VoiceState>) => void) {
  if (timer) clearInterval(timer);
  timer = setInterval(async () => {
    try {
      const status = await (await fetch("/api/voice")).json();
      if (status.state === "running") {
        set({ phase: status.phase, result: status.partial });
        return;
      }
      if (timer) clearInterval(timer);
      timer = null;
      if (status.state === "done") set({ running: false, result: status.result, fresh: false });
      else if (status.state === "error") set({ running: false, error: status.error, result: status.partial });
      else set({ running: false });
    } catch {
      set({ online: false });
    }
  }, 1000);
}
