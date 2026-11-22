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

export const sourceId = (source: BrowserSource) => ("site" in source ? `site:${source.site}` : `hf:${source.repo}`);
