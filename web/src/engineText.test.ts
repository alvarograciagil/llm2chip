import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { ENGINE_TEXT, englishOf } from "./engineText";
import { PARAM_HELP_EN } from "./paramHelpEn";
import { FIELDS } from "./params";

/** Strings in the engine's sources that never reach a person: names, links, ids. */
const NOT_TEXT = [/^https:/, /board!\(/, /^Sipeed|^Radiona|^Digilent|^AMD |^Gowin|^Lattice|^Xilinx|^Zynq|^Terasic|^Altera|^Intel |^Cyclone/, /^Cada die ocupa$/];

function engineStrings(file: string): string[] {
  const source = readFileSync(new URL(`../../crates/arch/src/${file}`, import.meta.url), "utf8").split("#[cfg(test)]")[0].replace(/(^|[^:])\/\/[^\n]*/gm, "$1"); // a URL's "://" is not a comment
  return [...source.matchAll(/"((?:[^"\\]|\\.)*)"/g)]
    .map((m) => m[1])
    .filter((s) => s.includes(" ") && /[a-záéíóúñ]/.test(s) && !(s.startsWith("{") && s.endsWith("}") && !s.slice(1).includes("{")))
    .filter((s) => !NOT_TEXT.some((r) => r.test(s)));
}

const shape = (text: string) => text.replace(/\{[^}]*\}/g, "{}");

describe("engine texts in English", () => {
  it("cover every text the engine writes", () => {
    const known = new Set(ENGINE_TEXT.map(([spanish]) => shape(spanish)));
    const missing = ["estimate.rs", "fpga.rs", "feasibility.rs"].flatMap(engineStrings).filter((s) => !known.has(shape(s)));
    expect(missing).toEqual([]);
  });

  it("include the help of every parameter", () => {
    const missing = FIELDS.filter((f) => f.help && !PARAM_HELP_EN[f.key]).map((f) => f.key);
    expect(missing).toEqual([]);
  });

  it("rewrite texts with numbers and nested names", () => {
    expect(englishOf("Cada die ocupa 144 mm² y el máximo es 10 mm².")).toBe("Each die takes 144 mm² and the limit is 10 mm².");
    expect(englishOf("Cada die necesita 9815 mm² aun repartido en 512 chips y el proceso admite 10 mm²: Celdas: memoria de pesos 9531 mm², Unidades vectoriales 252 mm².")).toBe(
      "Each die needs 9815 mm² even spread over 512 chips and the process takes 10 mm²: Cells: weight memory 9531 mm², Vector units 252 mm².",
    );
    expect(englishOf("Cada die necesita 14 mm² y el proceso admite 10 mm²: Memoria de la CPU 5 mm².")).toBe("Each die needs 14 mm² and the process takes 10 mm²: CPU memory 5 mm².");
    expect(englishOf("Proceso GF180 (abierto, fabricable, calibrado con OpenROAD)")).toBe("Process GF180 (open, manufacturable, calibrated with OpenROAD)");
    expect(englishOf("algo que no es del motor")).toBeNull();
  });
});
