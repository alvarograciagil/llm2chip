import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { translate } from "./i18n";

/** A Rust source of the engine, read as text. */
const rust = (path: string) => readFileSync(new URL(`../../${path}`, import.meta.url), "utf8");

/** A sample of what a format string gives: each `{…}` becomes a number. */
const sample = (text: string) => text.replace(/\{[^{}]*\}/g, "7");

const untranslated = (texts: string[]) => [...new Set(texts)].filter((text) => translate("es", text) === text);

// The viewer and the program views show these texts as the engine writes them, so
// a new one in Rust must also get its Spanish.
describe("the engine's texts in Spanish", () => {
  it("names every operation of the compiled program", () => {
    const source = rust("crates/compiler/src/lib.rs");
    const names = [
      ...[...source.matchAll(/\b(?:e|self)\.(?:op|matvec|norm)\(\s*"([^"]+)"/g)].map((m) => m[1]),
      ...[...source.matchAll(/\bwith_bias\(\s*&?(?:mut\s+)?e,\s*"([^"]+)"/g)].map((m) => m[1]),
    ];
    expect(names.length).toBeGreaterThan(40);
    expect(untranslated(names)).toEqual([]);
  });

  it("titles every trace", () => {
    const source = rust("crates/trace/src/lib.rs");
    const body = source.slice(source.indexOf("pub fn sharing_label"), source.indexOf("pub fn design_title"));
    const labels = [...body.matchAll(/=> (?:format!\()?"([^"]+)"/g)].map((m) => sample(m[1]));
    expect(labels.length).toBe(3);
    const titles = labels.flatMap((label) => [`Small model on 4 dies, ${label}`, `Qwen3.8-27B on 16 dies, ${label}`]);
    expect(untranslated([...labels, ...titles])).toEqual([]);
  });

  it("explains every error a recording or a look inside can give", () => {
    const errors = ["crates/server/src/trace_job.rs", "crates/trace/src/replay.rs"].flatMap((path) =>
      [...rust(path).matchAll(/(?:Err\((?:format!\()?|ok_or\()"([^"]+)"/g)]
        .map((m) => m[1])
        // A request the interface never sends, or a file name the system's own error follows.
        .filter((text) => !/^\{\w+\} is not |^[\w.]+: \{/.test(text))
        .map(sample),
    );
    expect(errors.length).toBeGreaterThan(8);
    expect(untranslated(errors)).toEqual([]);
  });
});
