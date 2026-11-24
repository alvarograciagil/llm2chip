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
  set: (partial: Partial<VoiceState>) => void;
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
  set: (partial) => set(partial),

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
    const { question, listen, think, speak, sound, arithmetic } = get();
    if (!question) return;
    set({ running: true, error: null, result: null, phase: "listening" });
    const body = { wav: question.wav, listen, think, speak, sound, bits: arithmetic === "float" ? null : Number(arithmetic) };
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
      if (status.state === "done") set({ running: false, result: status.result });
      else if (status.state === "error") set({ running: false, error: status.error, result: status.partial });
      else set({ running: false });
    } catch {
      set({ online: false });
    }
  }, 1000);
}
