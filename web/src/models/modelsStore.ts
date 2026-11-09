import { create } from "zustand";

export interface Adaptation {
  design: boolean;
  simulate: boolean;
  approximate: boolean;
  reason: string | null;
  parameters: number | null;
  layers: number | null;
}

export interface Settings {
  model_dirs: string[];
  download_dir: string;
  download_is_default: boolean;
  default_dirs: string[];
}

export interface SearchHit {
  id: string;
  downloads: number;
  likes: number;
  architecture: string | null;
  adaptation: Adaptation;
}

export interface Details {
  id: string;
  gated: boolean;
  files: { name: string; size: number }[];
  total_bytes: number;
  config: string | null;
  architecture: string | null;
  adaptation: Adaptation;
  target: string;
  present: boolean;
  resumable_bytes: number;
  free_bytes: number | null;
}

export interface Job {
  id: string;
  state: "queued" | "downloading" | "done" | "cancelled" | "error";
  total_bytes: number;
  done_bytes: number;
  file: string;
  error: string | null;
  target: string;
}

export type Source = "local" | "hub";

interface ModelsState {
  source: Source;
  settings: Settings | null;
  tokenSource: string | null;
  query: string;
  searched: string | null;
  hits: SearchHit[];
  searching: boolean;
  selected: string | null;
  details: Details | null;
  jobs: Job[];
  /** Bytes per second of each running download. */
  speeds: Record<string, number>;
  error: string | null;
  settingsError: string | null;

  set: (patch: Partial<ModelsState>) => void;
  loadSettings: () => Promise<void>;
  saveSettings: (dirs: string[], downloadDir: string | null) => Promise<boolean>;
  saveToken: (token: string) => Promise<boolean>;
  forgetToken: () => Promise<void>;
  search: () => Promise<void>;
  select: (id: string | null) => Promise<void>;
  download: (id: string) => Promise<void>;
  cancel: (id: string) => Promise<void>;
  discard: (id: string) => Promise<void>;
  pollJobs: () => Promise<Job[]>;
}

const json = { "content-type": "application/json" };

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, init);
  if (!response.ok) throw new Error((await response.text()) || `Error ${response.status}`);
  return response.status === 204 ? (undefined as T) : response.json();
}

const text = (e: unknown) => (e instanceof Error ? e.message : String(e));

let lastPoll = 0;

export const useModels = create<ModelsState>((set, get) => ({
  source: "local",
  settings: null,
  tokenSource: null,
  query: "",
  searched: null,
  hits: [],
  searching: false,
  selected: null,
  details: null,
  jobs: [],
  speeds: {},
  error: null,
  settingsError: null,

  set: (patch) => set(patch),

  async loadSettings() {
    try {
      const [settings, token] = await Promise.all([call<Settings>("/api/settings"), call<{ source: string | null }>("/api/hf/token")]);
      set({ settings, tokenSource: token.source });
    } catch {
      set({ settings: null });
    }
  },

  async saveSettings(dirs, downloadDir) {
    try {
      const settings = await call<Settings>("/api/settings", { method: "PUT", headers: json, body: JSON.stringify({ model_dirs: dirs, download_dir: downloadDir }) });
      set({ settings, settingsError: null });
      return true;
    } catch (e) {
      set({ settingsError: text(e) });
      return false;
    }
  },

  async saveToken(token) {
    try {
      const { source } = await call<{ source: string | null }>("/api/hf/token", { method: "PUT", headers: json, body: JSON.stringify({ token }) });
      set({ tokenSource: source, settingsError: null });
      return true;
    } catch (e) {
      set({ settingsError: text(e) });
      return false;
    }
  },

  async forgetToken() {
    const { source } = await call<{ source: string | null }>("/api/hf/token", { method: "DELETE" });
    set({ tokenSource: source });
  },

  async search() {
    const query = get().query.trim();
    if (query.includes("/") && !query.includes(" ")) {
      // A full repository name goes straight to its page, even if search would bury it.
      set({ hits: [], searched: query });
      await get().select(query);
      return;
    }
    set({ searching: true, error: null, selected: null, details: null });
    try {
      const hits = await call<SearchHit[]>(`/api/hf/search?q=${encodeURIComponent(query)}`);
      set({ hits, searched: query, searching: false });
    } catch (e) {
      set({ error: text(e), hits: [], searching: false });
    }
  },

  async select(id) {
    set({ selected: id, details: null, error: null });
    if (!id) return;
    try {
      const details = await call<Details>(`/api/hf/model?id=${encodeURIComponent(id)}`);
      if (get().selected === id) set({ details });
    } catch (e) {
      if (get().selected === id) set({ error: text(e) });
    }
  },

  async download(id) {
    try {
      await call("/api/hf/downloads", { method: "POST", headers: json, body: JSON.stringify({ id }) });
      set({ error: null });
      await get().pollJobs();
    } catch (e) {
      set({ error: text(e) });
    }
  },

  async cancel(id) {
    await call("/api/hf/cancel", { method: "POST", headers: json, body: JSON.stringify({ id }) });
    await get().pollJobs();
  },

  async discard(id) {
    try {
      await call("/api/hf/discard", { method: "POST", headers: json, body: JSON.stringify({ id }) });
      await get().pollJobs();
      if (get().selected === id) await get().select(id);
    } catch (e) {
      set({ error: text(e) });
    }
  },

  async pollJobs() {
    try {
      const jobs = await call<Job[]>("/api/hf/downloads");
      const now = performance.now();
      const elapsed = (now - lastPoll) / 1000;
      lastPoll = now;
      const before = get().jobs;
      const speeds: Record<string, number> = {};
      for (const job of jobs) {
        const earlier = before.find((b) => b.id === job.id);
        if (job.state === "downloading" && earlier && elapsed > 0) {
          const measured = Math.max(0, job.done_bytes - earlier.done_bytes) / elapsed;
          speeds[job.id] = 0.7 * (get().speeds[job.id] ?? measured) + 0.3 * measured;
        }
      }
      set({ jobs, speeds });
      return jobs;
    } catch {
      return get().jobs;
    }
  },
}));

export function formatBytes(bytes: number): string {
  const units: [number, string][] = [[1e12, "TB"], [1e9, "GB"], [1e6, "MB"], [1e3, "kB"]];
  for (const [size, unit] of units) {
    if (bytes >= size) return `${(bytes / size).toLocaleString("es-ES", { maximumFractionDigits: 1 })} ${unit}`;
  }
  return `${bytes} B`;
}

export function formatCount(value: number): string {
  if (value >= 1e9) return `${(value / 1e9).toLocaleString("es-ES", { maximumFractionDigits: 1 })} G`;
  if (value >= 1e6) return `${(value / 1e6).toLocaleString("es-ES", { maximumFractionDigits: 1 })} M`;
  if (value >= 1e3) return `${(value / 1e3).toLocaleString("es-ES", { maximumFractionDigits: 1 })} k`;
  return String(value);
}

/** Time left, once there is a speed to base it on. */
export function formatRemaining(job: Job, speed: number | undefined): string {
  if (!speed || speed < 1) return "";
  const seconds = (job.total_bytes - job.done_bytes) / speed;
  if (seconds < 60) return "menos de un minuto";
  if (seconds < 3600) return `${Math.ceil(seconds / 60)} min`;
  return `${(seconds / 3600).toLocaleString("es-ES", { maximumFractionDigits: 1 })} h`;
}
