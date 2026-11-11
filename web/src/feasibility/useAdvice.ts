import { useEffect, useState } from "react";
import type { ChipParams, ModelSource } from "../engine";

export interface Change {
  field: string;
  value: unknown;
  label: string;
  risk?: string;
}

export interface Suggestion {
  changes: Change[];
  dies: number;
  die_area_mm2: number;
  tok_s: number;
  layout: boolean;
}

export interface Feasibility {
  possible: boolean;
  layout: boolean;
  reasons: string[];
  notes: string[];
  dies: number;
  die_area_mm2: number;
  die_limit_mm2: number;
  largest: [string, number][];
  tok_s: number;
}

export interface Advice {
  verdict: Feasibility;
  suggestions: Suggestion[];
}

interface Reply {
  id: number;
  advice?: Advice;
  /** The verdict is in and the changes that fix it are still being looked for. */
  searching?: boolean;
  error?: string;
}

let worker: Worker | null = null;
let nextId = 0;
const listeners = new Set<(reply: Reply) => void>();

function theWorker() {
  if (!worker) {
    worker = new Worker(new URL("./adviceWorker.ts", import.meta.url), { type: "module" });
    worker.onmessage = (event: MessageEvent<Reply>) => listeners.forEach((listen) => listen(event.data));
  }
  return worker;
}

/** Whether the design can be built and how to make it so, worked out in the background.
 * What was shown stays until the new answer is ready; `checking` says one is on its way. */
interface AdviceState {
  advice: Advice | null;
  checking: boolean;
  searching: boolean;
  error: string | null;
}

export function useAdvice(model: ModelSource | null, params: ChipParams, enabled: boolean): AdviceState {
  const [state, setState] = useState<AdviceState>({ advice: null, checking: false, searching: false, error: null });

  useEffect(() => {
    if (!model || !enabled) return;
    const id = ++nextId;
    setState((s) => ({ ...s, checking: true }));
    const listen = (reply: Reply) => {
      if (reply.id !== id) return;
      setState({ advice: reply.advice ?? null, checking: false, searching: reply.searching ?? false, error: reply.error ?? null });
    };
    listeners.add(listen);
    theWorker().postMessage({ id, name: model.name, config: model.config, params: JSON.stringify(params) });
    return () => {
      listeners.delete(listen);
    };
  }, [model, params, enabled]);

  return state;
}
