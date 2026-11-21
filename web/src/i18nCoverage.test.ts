import { appendFileSync, readdirSync, readFileSync, statSync } from "node:fs";
import { join, relative } from "node:path";
import { describe, expect, it } from "vitest";
import { translate } from "./i18n";

const ROOT = new URL(".", import.meta.url).pathname;

/** With I18N_REPORT set, the missing texts are also written there, one per line. */
const report = (lines: string[]) => {
  if (process.env.I18N_REPORT) appendFileSync(process.env.I18N_REPORT, lines.map((l) => l + "\n").join(""));
};

/** Texts that read the same in Spanish: names, units and symbols. */
const SAME = new Set([
  "DeltaNet", "RoPE", "SiLU", "GELU", "Show-o", "Hugging Face", "GitHub", "ECC", "SRAM", "ROM", "HBM4", "tok/s", "Top-k", "top-k", "MTP", "CPU", "GPU", "FPGA", "UCIe", "Verilog", "VHDL", "MLX", "GGUF",
  "Posit", "Error", "Total", "Booth", "Wallace", "Janus-Pro", "MAGVIT-v2", "Qwen", "ok", "PDK", "OpenROAD", "Yosys", "JTAG", "BIST", "KV", "Q·K", "lm_head", "RISC-V", "UART", "NVMe", "PCIe", "English", "Español", "llm2chip", "Bits", "Top-1", "MUX", "MLP",
]);

function sources(dir: string): string[] {
  return readdirSync(dir).flatMap((name) => {
    const path = join(dir, name);
    if (statSync(path).isDirectory()) return name === "wasm" ? [] : sources(path);
    return /\.(ts|tsx)$/.test(name) && !/\.test\.tsx?$/.test(name) && !["i18n.ts", "engineText.ts", "paramHelpEs.ts"].includes(name) ? [path] : [];
  });
}

/** A sample of what a template gives: each `${…}` becomes a number, as most of them are. */
const sample = (template: string) => template.replace(/\$\{(?:[^{}]|\{[^{}]*\})*\}/g, "7");

/** Words written the same in Spanish. */
const SAME_WORDS = new Set(["chip", "chips", "bits", "Bits", "MLX", "Booth", "radix", "Die", "die", "dies", "Token", "token", "tokens", "Mbit", "DSP", "mult", "tok", "Bus", "Panel", "Prompt", "instr", "real", "Chips"]);

/** Texts given to the model as they are, not interface: the sample questions. */
const INPUTS = new Set(["What is the capital of France? Answer in one sentence.", "The capital of France is"]);

/** Whether `text` needs no translation: no words, or only names, units and shared words. */
function neutral(text: string): boolean {
  if (SAME.has(text.trim()) || INPUTS.has(text)) return true;
  const words = text.match(/[A-Za-zÀ-ÿ]{3,}/g) ?? [];
  return words.every((w) => SAME.has(w) || SAME_WORDS.has(w));
}

describe("Spanish of the interface", () => {
  const files = sources(ROOT);

  it("covers every text passed to t()", () => {
    const missing: string[] = [];
    for (const file of files) {
      const source = readFileSync(file, "utf8");
      const texts = [
        ...[...source.matchAll(/\bt\("((?:[^"\\]|\\.)*)"\)/g)].map((m) => JSON.parse(`"${m[1]}"`) as string),
        ...[...source.matchAll(/\bt\(`((?:[^`\\]|\\.)*)`\)/g)].map((m) => sample(m[1])),
      ];
      for (const text of texts) {
        if (!neutral(text) && translate("es", text) === text) missing.push(`${relative(ROOT, file)}: ${text}`);
      }
    }
    report([...new Set(missing)].map((m) => "T " + m));
    expect([...new Set(missing)]).toEqual([]);
  });

  it("translates the sentences kept in tables of data", () => {
    const missing: string[] = [];
    // The datasheet and the parameter reference are written in English, as files.
    for (const file of files.filter((f) => !/(datasheet|markdown)\.ts$/.test(f))) {
      const source = readFileSync(file, "utf8")
        .replace(/(^|[\s{(;])\/\*[\s\S]*?\*\//gm, "$1")
        .replace(/^\s*\/\/.*$/gm, "")
        // What only the console, the engine's API or the class names see.
        // The help of each parameter has its Spanish in paramHelpEs.ts.
        .replace(/\bhelp: "(?:[^"\\]|\\.)*"/g, "")
        .replace(/(console\.\w+|new Error|throw|fetch|className=|import .* from|querySelector\w*)\([^)]*\)|className="[^"]*"|from "[^"]*"/g, "");
      for (const m of source.matchAll(/"([A-Z][^"\\]*\b[a-z]{2,}\b[^"\\]*)"/g)) {
        const text = m[1];
        if (!/ /.test(text) || neutral(text) || /^[A-Z_]+$|\.(png|svg|json|zip|v|vhd)$|^https?:|\{|\$|[<>]/.test(text)) continue;
        if (translate("es", text) === text) missing.push(`${relative(ROOT, file)}: ${text}`);
      }
    }
    report([...new Set(missing)].map((m) => "D " + m));
    expect([...new Set(missing)]).toEqual([]);
  });

  it("leaves no text in the markup outside t()", () => {
    const loose: string[] = [];
    for (const file of files.filter((f) => f.endsWith(".tsx"))) {
      const source = readFileSync(file, "utf8").replace(/(^|[\s{(;])\/\*[\s\S]*?\*\//gm, "$1").replace(/^\s*\/\/.*$/gm, "");
      // Text between tags: the closing ">" of a tag, words, then "<" or "{".
      for (const m of source.matchAll(/(?<=[^=\s-]>|^\s+)([^<>{}\n;=()`'"]*[A-Za-z]{3,}[^<>{}\n;=()`'"]*)(?=\s*(?:<\/|\{|<[a-zA-Z]))/gm)) {
        const text = m[1].trim();
        // Code that happens to sit before a "<" or "{": declarations, keys and keywords.
        if (/^(const|let|return|try|interface|export|function|type|if|else|catch|await|async|case|default)\b|^[\w.]+\??:|:\s*$|\b(Record|Partial)$|&&$|\?$|^[a-z]+[A-Z]|^\w+\.\w+/.test(text)) continue;
        if (text && !neutral(text)) loose.push(`${relative(ROOT, file)}: ${text}`);
      }
      // Attributes a person reads.
      for (const m of source.matchAll(/\b(title|placeholder|aria-label|alt|label)="([^"]*[A-Za-z]{3,}[^"]*)"/g)) {
        if (!neutral(m[2])) loose.push(`${relative(ROOT, file)}: ${m[1]}="${m[2]}"`);
      }
    }
    report([...new Set(loose)].map((m) => "L " + m));
    expect([...new Set(loose)]).toEqual([]);
  });
});
