// The models the online demo runs in the browser, and what it will download for them.

/** The online demo downloads at most this much for a model: a model takes about three
 * times its size in memory to run, and a larger one would run too slowly on one thread. */
export const DOWNLOAD_LIMIT = 300 * 2 ** 20;

/** The files of a checkpoint the simulator reads besides its weights. */
export const TEXT_FILES = ["config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json", "chat_template.jinja"];

export interface BrowserModel {
  /** `site:<folder>` for a model served with the demo, `hf:<repository>` for one on Hugging Face. */
  id: string;
  name: string;
  /** Where its files are, ending in “/”: the demo's own folder, or the repository at a fixed commit. */
  base: string;
  texts: string[];
  weights: { name: string; size: number }[];
  /** Everything it downloads. */
  bytes: number;
  config: string;
  architecture: string;
  /** Who made it and under what licence, when the demo serves it itself. */
  credit: string | null;
}

/** Where a model comes from, before it is described. */
export type BrowserSource = { site: string; name: string; credit: string } | { repo: string };

/** The models offered from the start: one that loads at once, and one that answers questions. */
export const SUGGESTED: BrowserSource[] = [
  { site: "stories260K", name: "stories260K", credit: "Andrej Karpathy, llama2.c (MIT)" },
  { repo: "HuggingFaceTB/SmolLM2-135M-Instruct" },
];

/** What to ask a model that only continues text, instead of the question the simulator starts with. */
export const PROMPTS: Record<string, string> = { "site:stories260K": "Once upon a time, there was a little robot" };

/** A link loads and runs a model on its own only up to this size; a larger one waits for a click. */
export const LINK_LOAD_LIMIT = 16 * 2 ** 20;

/** Why a model may not run on this device: it takes about three times its download in
 * memory. Browsers that say how much the device has are told apart; phones and tablets that
 * do not are warned above half a gigabyte. */
export function memoryWarning(bytes: number): string | null {
  const needed = Math.round((3 * bytes) / 2 ** 20);
  const device = (navigator as Navigator & { deviceMemory?: number }).deviceMemory;
  if (device) return 3 * bytes > (device * 2 ** 30) / 4 ? `It needs about ${needed} MB of memory, and this device has ${device} GB: the page may close.` : null;
  const touch = typeof matchMedia === "function" && matchMedia("(pointer: coarse)").matches;
  return touch && 3 * bytes > 2 ** 29 ? `It needs about ${needed} MB of memory: on a phone or tablet the page may close.` : null;
}

export const sourceId = (source: BrowserSource) => ("site" in source ? `site:${source.site}` : `hf:${source.repo}`);
