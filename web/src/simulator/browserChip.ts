// Talks to the simulator that runs in the browser, started the first time it is needed.
import { sourceId, type BrowserModel, type BrowserSource } from "./browserModels";
import type { WorkerRequest } from "./browserWorker";

type Event = Record<string, unknown> & { type: string };
type Pending<T> = { resolve: (value: T) => void; reject: (error: Error) => void };

let worker: Worker | null = null;
const describing = new Map<string, Pending<BrowserModel>>();
let loading: (Pending<{ seconds: number; kept: boolean }> & { progress: (phase: string, done: number, total: number) => void }) | null = null;
let onEvent: ((event: Event) => void) | null = null;
const replies = new Map<number, Pending<unknown>>();
let asked = 0;

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
    loading?.resolve({ seconds: Number(event.seconds), kept: event.kept !== false });
    loading = null;
  } else if (event.type === "reply") {
    const pending = replies.get(Number(event.id));
    replies.delete(Number(event.id));
    if (event.error !== undefined) pending?.reject(new Error(String(event.error)));
    else pending?.resolve(event.value);
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

function ask<T>(message: (id: number) => WorkerRequest): Promise<T> {
  const id = ++asked;
  return new Promise((resolve, reject) => {
    replies.set(id, { resolve: resolve as (value: unknown) => void, reject });
    send(message(id));
  });
}

/** How many bytes of each model the browser keeps, so they need no download, and in all. */
export const keptModels = (models: BrowserModel[]) => ask<{ models: Record<string, number>; total: number }>((id) => ({ type: "kept", id, models }));

/** Deletes the browser's copy of a model, or with none, of every model. */
export const forgetModel = (model: BrowserModel | null) => ask<void>((id) => ({ type: "forget", id, model }));

/** Downloads (or takes from the browser's storage) a model and builds the chip's engine for
 * it; `kept` says whether there was room to keep a copy. */
export function loadModel(model: BrowserModel, setup: { activation_bits: number; format: unknown; context_tokens: number }, progress: (phase: string, done: number, total: number) => void): Promise<{ seconds: number; kept: boolean }> {
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
