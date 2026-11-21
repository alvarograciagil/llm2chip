import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { spanishOf } from "./engineText";
import { spanishWord } from "./wordText";

/** The body of `describe_word`, the function that writes a word's description. */
function describeWord(): string {
  const source = readFileSync(new URL("../../crates/netlist/src/die.rs", import.meta.url), "utf8");
  const start = source.indexOf("pub fn describe_word");
  return source.slice(start, source.indexOf("\n}\n", start));
}

describe("a microcode word in Spanish", () => {
  it("puts each piece of the sentence in Spanish", () => {
    expect(spanishWord("Receives a word from the previous chip; takes it from the return link of the group's last chip.")).toBe(
      "Recibe una palabra del die anterior; la recoge del enlace de vuelta del último die del grupo.",
    );
    expect(spanishWord("Applies the exponential table to r510 on every lane at once and stores it in r57.")).toBe(
      "Aplica la tabla de la exponencial a r510 en todos los carriles a la vez y guarda el resultado en r57.",
    );
    expect(spanishWord("Adds r0[0] + r0[0].")).toBe("Suma r0[0] + r0[0].");
    expect(spanishWord("The cells start a new sum; the cells receive r3 over the activation network, multiply it by their weight and accumulate it; repeats the step 1 more time, moving through the elements.")).toBe(
      "Las celdas empiezan una suma nueva; las celdas reciben r3 por la red de activaciones, lo multiplican por su peso y lo acumulan; repite el paso 1 vez más, recorriendo los elementos.",
    );
    expect(spanishWord("Copies r4; sends it to the next chip; if counter x2 has not reached its limit, goes back to word 17.")).toBe(
      "Copia r4; envía el resultado al die siguiente; si el contador x2 no ha llegado a su límite, vuelve a la palabra 17.",
    );
    expect(spanishWord("End of the routine: the die tells the CPU it has finished.")).toBe("Fin de la rutina: el die avisa a la CPU de que ha terminado.");
  });

  it("reaches the interface through the engine's texts", () => {
    expect(spanishOf("Finds the maximum of r9 on every lane at once.")).toBe("Busca el máximo de r9 en todos los carriles a la vez.");
  });

  it("leaves alone what is not a word's description", () => {
    expect(spanishWord("Adds r1 + r2; something the engine never writes.")).toBeNull();
    expect(spanishWord("Gate projection")).toBeNull();
  });

  it("knows every piece the engine writes", () => {
    const body = describeWord();
    const tables = [...body.slice(body.indexOf("let name = match"), body.indexOf("};", body.indexOf("let name = match"))).matchAll(/=> "([^"]+)"/g)].map((m) => m[1]);
    expect(tables.length).toBeGreaterThan(5);
    const pieces = [...body.matchAll(/"((?:[^"\\]|\\.)*)"/g)]
      .map((m) => m[1])
      .filter((s) => s.includes(" ") && !s.startsWith(" ") && !s.includes("{what}") && s !== "; " && !tables.includes(s));
    const samples = pieces.flatMap((s) => {
      const filled = s.replace("{times} more {}", "4 more times").replace(/\{(a|b|dst)\}/g, "r2[3]").replace(/\{\}/g, "7");
      return filled.includes("{name}") ? tables.map((name) => filled.replace("{name}", name)) : [filled];
    });
    const sentence = (s: string) => (/[.]$/.test(s) ? s : `${s.charAt(0).toUpperCase()}${s.slice(1)}.`);
    const missing = samples.filter((s) => spanishWord(sentence(s)) === null);
    expect(missing).toEqual([]);
  });
});
