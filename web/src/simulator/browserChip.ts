// Talks to the simulator that runs in the browser, started the first time it is needed.
import { sourceId, type BrowserModel, type BrowserSource } from "./browserModels";
import type { WorkerRequest } from "./browserWorker";

type Event = Record<string, unknown> & { type: string };
type Pending<T> = { resolve: (value: T) => void; reject: (error: Error) => void };

let worker: Worker | null = null;
const describing = new Map<string, Pending<BrowserModel>>();
let loading: (Pending<number> & { progress: (phase: string, done: number, total: number) => void }) | null = null;
let onEvent: ((event: Event) => void) | null = null;

function send(message: WorkerRequest) {
  if (!worker) {
    worker = new Worker(new URL("./browserWorker.ts", import.meta.url), { type: "module" });
    worker.onmessage = (e: MessageEvent<Event>) => receive(e.data);
  }
  worker.postMessage(message);
}

function receive(event: Event) {
  if (event.type === "described") {
    const model = event.model as BrowserModel;
    const key = sourceId(event.source as BrowserSource);
    describing.get(key)?.resolve(model);
    describing.delete(key);
  } else if (event.type === "progress") {
    loading?.progress(String(event.phase), Number(event.done), Number(event.total));
  } else if (event.type === "loaded") {
    loading?.resolve(Number(event.seconds));
    loading = null;
  } else if (event.type === "error" && event.during === "describe") {
    const key = sourceId(event.source as BrowserSource);
    describing.get(key)?.reject(new Error(String(event.message)));
    describing.delete(key);
  } else if (event.type === "error" && event.during === "load") {
    loading?.reject(new Error(String(event.message)));
    loading = null;
  } else {
    onEvent?.(event);
  }
}

/** What a model would download and whether the simulator runs it, or why not. */
export function describeModel(source: BrowserSource): Promise<BrowserModel> {
  return new Promise((resolve, reject) => {
    describing.set(sourceId(source), { resolve, reject });
    send({ type: "describe", source });
  });
}

/** Downloads (or takes from the browser's cache) a model and builds the chip's engine for it. */
export function loadModel(model: BrowserModel, setup: { activation_bits: number; format: unknown; context_tokens: number }, progress: (phase: string, done: number, total: number) => void): Promise<number> {
  return new Promise((resolve, reject) => {
    loading?.reject(new Error("Replaced by another load."));
    loading = { resolve, reject, progress };
    send({ type: "load", model, setup });
  });
}

/** Runs a prompt; `listen` gets the same events the local engine sends. */
export function runModel(request: Record<string, unknown>, listen: (event: Event) => void) {
  onEvent = listen;
  send({ type: "generate", request });
}

export function stopModel() {
  // A step already on its way must not undo the stop.
  onEvent = null;
  send({ type: "stop" });
}
