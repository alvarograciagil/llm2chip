import { useEffect, useState } from "react";
import type { ChipParams, ModelSource, SweepResult } from "../engine";

type Variant = Record<string, unknown>;

export interface Sweep<V extends Variant> {
  variants: V[];
  results: SweepResult[] | null;
  done: number;
  total: number;
}

interface Reply {
  channel: string;
  id: number;
  done: number;
  results?: SweepResult[];
}

let worker: Worker | null = null;
let nextId = 0;
const listeners = new Map<string, (reply: Reply) => void>();

function theWorker() {
  worker ??= new Worker(new URL("./sweepWorker.ts", import.meta.url), { type: "module" });
  worker.onmessage = (event: MessageEvent<Reply>) => listeners.get(event.data.channel)?.(event.data);
  return worker;
}

/** Runs a sweep in the background; what was shown stays until the new results are ready. */
export function useSweep<V extends Variant>(channel: string, model: ModelSource | null, params: ChipParams, variants: V[]): Sweep<V> {
  const [shown, setShown] = useState<{ variants: V[]; results: SweepResult[] | null }>({ variants: [], results: null });
  const [progress, setProgress] = useState({ done: 0, total: 0 });

  useEffect(() => {
    if (!model || variants.length === 0) return;
    const id = ++nextId;
    setProgress({ done: 0, total: variants.length });
    listeners.set(channel, (reply) => {
      if (reply.id !== id) return;
      setProgress({ done: reply.done, total: variants.length });
      if (reply.results) setShown({ variants, results: reply.results });
    });
    theWorker().postMessage({ channel, id, name: model.name, config: model.config, params: JSON.stringify(params), variants });
    return () => {
      listeners.delete(channel);
    };
  }, [channel, model, params, variants]);

  return { ...shown, ...progress };
}
