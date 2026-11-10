import { readFileSync, writeFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { FIELDS } from "../params";
import { parametersMarkdown } from "./markdown";

const path = new URL("../../../docs/parametros.md", import.meta.url);

describe("documentación de parámetros", () => {
  it("explica cada parámetro del panel", () => {
    for (const field of FIELDS) expect(field.help, field.key).toBeTruthy();
  });

  it("está al día con la aplicación", () => {
    const text = parametersMarkdown();
    if (process.env.UPDATE_DOCS) writeFileSync(path, text);
    expect(readFileSync(path, "utf8")).toBe(text);
  });
});
