// Sweeps build real circuits for every design, so they run away from the interface.
import init, { sweep_design } from "../wasm/llm2chip_wasm.js";

export interface SweepJob {
  channel: string;
  id: number;
  name: string;
  config: string;
  params: string;
  variants: Record<string, unknown>[];
}

const SLICE = 3;
const latest = new Map<string, number>();
const queue: SweepJob[] = [];
const ready = init();
let busy = false;

async function run(job: SweepJob) {
  const results: unknown[] = [];
  for (let at = 0; at < job.variants.length; at += SLICE) {
    // Yielding lets a newer request arrive, which makes this one pointless.
    await new Promise((resolve) => setTimeout(resolve));
    if (latest.get(job.channel) !== job.id) return;
    results.push(...JSON.parse(sweep_design(job.name, job.config, job.params, JSON.stringify(job.variants.slice(at, at + SLICE)))));
    postMessage({ channel: job.channel, id: job.id, done: results.length });
  }
  postMessage({ channel: job.channel, id: job.id, done: results.length, results });
}

async function pump() {
  if (busy) return;
  busy = true;
  await ready;
  // The shortest sweeps first, so that something appears as soon as possible.
  while (queue.length > 0) {
    queue.sort((a, b) => a.variants.length - b.variants.length);
    await run(queue.shift()!);
  }
  busy = false;
}

self.onmessage = (event: MessageEvent<SweepJob>) => {
  latest.set(event.data.channel, event.data.id);
  queue.push(event.data);
  void pump();
};
