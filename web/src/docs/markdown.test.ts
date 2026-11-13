import { readFileSync, writeFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { FIELDS } from "../params";
import { parametersMarkdown } from "./markdown";

const path = new URL("../../../docs/parameters.md", import.meta.url);

describe("parameter documentation", () => {
  it("explains every parameter of the panel", () => {
    for (const field of FIELDS) expect(field.help, field.key).toBeTruthy();
  });

  it("is up to date with the application", () => {
    const text = parametersMarkdown();
    if (process.env.UPDATE_DOCS) writeFileSync(path, text);
    expect(readFileSync(path, "utf8")).toBe(text);
  });
});
