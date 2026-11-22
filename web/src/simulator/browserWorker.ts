// The chip simulator of the online demo: it describes, downloads and runs a model in the
// browser, away from the interface, and speaks as the local engine does.
import init, { BrowserChip, Checkpoint, simulation_refusal } from "../wasm-sim/llm2chip_wasm_sim.js";
import { DOWNLOAD_LIMIT, TEXT_FILES, type BrowserModel, type BrowserSource } from "./browserModels";

export type WorkerRequest =
  | { type: "describe"; source: BrowserSource }
  | { type: "load"; model: BrowserModel; setup: { activation_bits: number; format: unknown; context_tokens: number } }
  | { type: "generate"; request: Record<string, unknown> }
  | { type: "stop" };

const ready = init();
const MB = 2 ** 20;
let chip: BrowserChip | null = null;
let running = 0;

const megabytes = (bytes: number) => `${Math.round(bytes / MB)} MB`;

async function describe(source: BrowserSource): Promise<BrowserModel> {
  if ("site" in source) {
    const base = new URL(`${import.meta.env.BASE_URL}models/${source.site}/`, location.origin).href;
    const size = async (name: string) => Number((await fetch(base + name, { method: "HEAD" })).headers.get("content-length")) || 0;
    const config = await (await fetch(base + "config.json")).text();
    const weights = [{ name: "model.safetensors", size: await size("model.safetensors") }];
    const bytes = weights[0].size + (await size("tokenizer.json")) + config.length;
    return { id: `site:${source.site}`, name: source.name, base, texts: ["config.json", "tokenizer.json"], weights, bytes, config, architecture: JSON.parse(config).model_type ?? "", credit: source.credit };
  }
  const repo = source.repo.trim().replace(/^https:\/\/huggingface\.co\//, "").replace(/\/+$/, "");
  if (!/^[\w.-]+\/[\w.-]+$/.test(repo)) throw new Error(`“${source.repo}” is not a Hugging Face model name, such as HuggingFaceTB/SmolLM2-135M-Instruct.`);
  const response = await fetch(`https://huggingface.co/api/models/${repo}?blobs=true`);
  if (!response.ok) throw new Error(`There is no public model ${repo} on Hugging Face.`);
  const info = await response.json();
  if (info.gated || info.private) throw new Error(`${repo} asks to accept its licence on Hugging Face, which the online demo cannot do: run it with the app.`);
  const files: { rfilename: string; size?: number }[] = (info.siblings ?? []).filter((f: { rfilename: string }) => !f.rfilename.includes("/"));
  const texts = TEXT_FILES.filter((name) => files.some((f) => f.rfilename === name));
  if (!texts.includes("config.json") || !texts.includes("tokenizer.json")) throw new Error(`${repo} has no config.json and tokenizer.json, which the simulator needs.`);
  const weights = files.filter((f) => f.rfilename.endsWith(".safetensors") && !f.rfilename.startsWith("model-vision")).map((f) => ({ name: f.rfilename, size: f.size ?? 0 }));
  if (weights.length === 0) throw new Error(`${repo} has no weights in safetensors.`);
  const bytes = [...weights, ...files.filter((f) => texts.includes(f.rfilename)).map((f) => ({ size: f.size ?? 0 }))].reduce((sum, f) => sum + f.size, 0);
  if (bytes > DOWNLOAD_LIMIT) throw new Error(`${repo} takes ${megabytes(bytes)}, and the online demo downloads up to ${megabytes(DOWNLOAD_LIMIT)}: run it with the app on your computer.`);
  const base = `https://huggingface.co/${repo}/resolve/${info.sha}/`;
  const config = await (await fetch(base + "config.json")).text();
  await ready;
  const refusal = simulation_refusal(config);
  if (refusal) throw new Error(`The simulator cannot run ${repo}: ${refusal}`);
  return { id: `hf:${repo}`, name: repo.split("/")[1], base, texts, weights, bytes, config, architecture: JSON.parse(config).model_type ?? "", credit: null };
}

const STORE = "llm2chip-models";

async function storeFolder(): Promise<FileSystemDirectoryHandle | null> {
  try {
    return await (await navigator.storage.getDirectory()).getDirectoryHandle(STORE, { create: true });
  } catch {
    return null;
  }
}

/** A file kept in the browser's private storage; a second file marks it complete. */
const storedName = (url: string) => url.replace(/^https?:\/\//, "").replace(/[^\w.-]+/g, "_");

interface SyncHandle {
  write(data: Uint8Array, options: { at: number }): number;
  truncate(size: number): void;
  flush(): void;
  close(): void;
}

async function readAll(stream: ReadableStream<Uint8Array>, take: (part: Uint8Array) => void): Promise<number> {
  const reader = stream.getReader();
  let total = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) return total;
    take(value);
    total += value.length;
  }
}

/** A file's bytes, given to `take` as they come: from the browser's private storage when a
 * complete copy is there, otherwise from the network, keeping a copy as it arrives. Models
 * come at a fixed commit, so a copy never goes stale. Returns how many bytes there were. */
async function download(url: string, take: (part: Uint8Array) => void): Promise<number> {
  const folder = await storeFolder();
  const name = storedName(url);
  if (folder) {
    try {
      await folder.getFileHandle(`${name}.done`);
      const file = await (await folder.getFileHandle(name)).getFile();
      return await readAll(file.stream(), take);
    } catch {
      // Not kept yet, or kept only in part.
    }
  }
  const response = await fetch(url);
  if (!response.ok || !response.body) throw new Error(`${url}: ${response.status}`);
  let copy: SyncHandle | null = null;
  try {
    const handle = (await folder?.getFileHandle(name, { create: true })) as (FileSystemFileHandle & { createSyncAccessHandle(): Promise<SyncHandle> }) | undefined;
    copy = (await handle?.createSyncAccessHandle()) ?? null;
    copy?.truncate(0);
  } catch {
    copy = null;
  }
  let at = 0;
  try {
    await readAll(response.body, (part) => {
      take(part);
      try {
        copy?.write(part, { at });
      } catch {
        // Without room to keep it (a full disk, a private window), it downloads again next time.
        copy?.close();
        copy = null;
        void folder?.removeEntry(name).catch(() => undefined);
      }
      at += part.length;
    });
    if (copy) {
      copy.flush();
      copy.close();
      copy = null;
      await folder?.getFileHandle(`${name}.done`, { create: true });
    }
  } finally {
    copy?.close();
  }
  return at;
}

async function downloadText(url: string): Promise<string> {
  const decoder = new TextDecoder();
  let text = "";
  await download(url, (part) => (text += decoder.decode(part, { stream: true })));
  return text + decoder.decode();
}

async function load(model: BrowserModel, setup: unknown) {
  await ready;
  chip?.free();
  chip = null;
  const checkpoint = new Checkpoint();
  try {
    for (const name of model.texts) checkpoint.add_text(name, await downloadText(model.base + name));
    const total = model.weights.reduce((sum, w) => sum + w.size, 0);
    let done = 0;
    let shown = 0;
    for (const weights of model.weights) {
      const file = checkpoint.reserve(weights.name, weights.size);
      let offset = 0;
      // The checkpoint refuses bytes beyond the size Hugging Face announced.
      const size = await download(model.base + weights.name, (part) => {
        checkpoint.write(file, offset, part);
        offset += part.length;
        done += part.length;
        if (done - shown > MB || done === total) {
          shown = done;
          postMessage({ type: "progress", phase: "Downloading", done, total });
        }
      });
      if (size !== weights.size) throw new Error(`${weights.name} arrived incomplete.`);
    }
    postMessage({ type: "progress", phase: "Building the chip", done: total, total });
    const started = performance.now();
    chip = new BrowserChip(checkpoint, JSON.stringify(setup));
    postMessage({ type: "loaded", seconds: (performance.now() - started) / 1000 });
  } catch (e) {
    checkpoint.free();
    throw e;
  }
}

async function generate(request: Record<string, unknown>) {
  if (!chip) throw new Error("Load the chip first.");
  const run = ++running;
  postMessage(JSON.parse(chip.start(JSON.stringify(request))));
  for (;;) {
    // Yielding lets a stop arrive between steps.
    await new Promise((resolve) => setTimeout(resolve));
    if (run !== running) return;
    const event = JSON.parse(chip.advance());
    postMessage(event);
    if (event.type === "done") return;
  }
}

onmessage = async (event: MessageEvent<WorkerRequest>) => {
  const message = event.data;
  try {
    if (message.type === "describe") postMessage({ type: "described", source: message.source, model: await describe(message.source) });
    if (message.type === "load") await load(message.model, message.setup);
    if (message.type === "generate") await generate(message.request);
    if (message.type === "stop") running++;
  } catch (e) {
    // A panic leaves the engine unusable: it must be loaded again.
    if (e instanceof WebAssembly.RuntimeError) chip = null;
    postMessage({ type: "error", during: message.type, source: message.type === "describe" ? message.source : undefined, message: e instanceof Error ? e.message : String(e) });
  }
};
