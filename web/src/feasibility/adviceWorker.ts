// The search for a buildable design estimates it dozens of times, so it runs away from the interface.
import init, { advise_design } from "../wasm/llm2chip_wasm.js";

export interface AdviceJob {
  id: number;
  name: string;
  config: string;
  params: string;
}

const ready = init();
let latest = 0;

self.onmessage = async (event: MessageEvent<AdviceJob>) => {
  const job = event.data;
  latest = job.id;
  await ready;
  // A newer request makes this one pointless.
  await new Promise((resolve) => setTimeout(resolve, 150));
  if (latest !== job.id) return;
  try {
    // The verdict takes one estimate; the search for changes, dozens.
    const verdict = JSON.parse(advise_design(job.name, job.config, job.params, false));
    postMessage({ id: job.id, advice: verdict, searching: !verdict.verdict.layout });
    if (verdict.verdict.layout) return;
    await new Promise((resolve) => setTimeout(resolve));
    if (latest !== job.id) return;
    postMessage({ id: job.id, advice: JSON.parse(advise_design(job.name, job.config, job.params, true)), searching: false });
  } catch (e) {
    postMessage({ id: job.id, error: e instanceof Error ? e.message : String(e) });
  }
};
