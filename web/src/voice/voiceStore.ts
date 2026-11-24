import { create } from "zustand";
import type { ServerModel } from "../simulator/simulatorStore";

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
    first_sound_ms: number;
    dies: PlannedDie[];
    stages: { name: string; first_ms: number; per_second: number; variant: string }[];
  } | null;
  weighed: { groups: string[][]; area_mm2: number | null; first_sound_ms: number | null; reason: string | null }[];
}

/** Where the stages run: floating point, or the die's arithmetic with vectors of so many bits. */
export type Arithmetic = "float" | "16" | "20" | "24";

interface VoiceState {
  online: boolean | null;
  models: ServerModel[];
  listen: string;
  think: string;
  speak: string;
  sound: string;
  arithmetic: Arithmetic;
  /** The question, a 16 kHz WAV, and how long it is. */
  question: { wav: string; seconds: number } | null;
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
  firstSoundMs: number;
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
  start: () => Promise<void>;
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
  models: [],
  listen: "",
  think: "",
  speak: "",
  sound: "",
  arithmetic: "float",
  question: null,
  recording: false,
  running: false,
  phase: "",
  result: null,
  error: null,
  history: [],
  fresh: true,
  speaker: true,
  firstSoundMs: 500,
  questionSeconds: 3,
  areaMm2: "",
  planning: false,
  planDoing: "",
  plan: null,
  planError: null,
  planOnOpen: false,
  set: (partial) => set(partial),

  startPlan: async () => {
    const { listen, think, speak, sound, firstSoundMs, questionSeconds, areaMm2, speaker } = get();
    set({ planning: true, planError: null, plan: null, planDoing: "" });
    const body = { listen, think, speak, sound, first_sound_ms: firstSoundMs, seconds: questionSeconds, area_mm2: areaMm2.trim() ? Number(areaMm2) : null, speaker };
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
        if (status.state === "done") set({ planning: false, plan: status.result });
        else set({ planning: false, planError: status.error ?? null });
      } catch {
        set({ online: false });
      }
    }, 1000);
  },

  connect: async () => {
    try {
      const models = (await (await fetch("/api/models")).json()) as ServerModel[];
      const s = get();
      set({
        online: true,
        models,
        listen: pick(models, s.listen, (m) => m.voice === "listens"),
        think: pick(models, s.think, (m) => m.supported && !m.voice),
        speak: pick(models, s.speak, (m) => m.voice === "speaks"),
        sound: pick(models, s.sound, (m) => m.voice === "sounds"),
      });
      const status = await (await fetch("/api/voice")).json();
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

  start: async () => {
    const { question, listen, think, speak, sound, arithmetic, fresh, result, speaker } = get();
    if (!question) return;
    // The last turn joins the conversation's history before the next is asked.
    const last = !fresh && result?.heard?.text && result.answer?.text ? [{ heard: result.heard.text, answer: result.answer.text }] : [];
    set({ running: true, error: null, result: null, phase: "listening", history: fresh ? [] : [...get().history, ...last] });
    const body = { wav: question.wav, listen, think, speak, sound, bits: arithmetic === "float" ? null : Number(arithmetic), fresh, speaker };
    const response = await fetch("/api/voice", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
    if (!response.ok) {
      set({ running: false, error: await response.text() });
      return;
    }
    poll(set);
  },
}));

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
