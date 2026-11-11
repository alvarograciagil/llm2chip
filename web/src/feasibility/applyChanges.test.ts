import { describe, expect, it } from "vitest";
import type { TechNode } from "../engine";
import { applyChanges } from "./FeasibilityPanel";

const nodes = [{ id: "sky130", label: "SkyWater 130", die_max_mm2: 10 }, { id: "gf180", label: "GF180", die_max_mm2: 20 }] as TechNode[];

describe("applying a suggestion", () => {
  it("drops the checkpoint's own mix when it sets one weight width", () => {
    expect(applyChanges({ optimizations: [], use_model_mix: true }, [{ field: "weight_bits", value: 4, label: "" }], nodes)).toEqual({ weight_bits: 4, use_model_mix: false });
  });

  it("fits the die to the new process", () => {
    const patch = applyChanges({ optimizations: [], die_width_mm: 26, die_height_mm: 33, die_max_mm2: 858 }, [{ field: "node", value: "gf180", label: "" }], nodes);
    expect(patch.node).toBe("gf180");
    expect(Number(patch.die_max_mm2)).toBeLessThanOrEqual(20);
  });

  it("draws the die it asks for", () => {
    expect(applyChanges({ optimizations: [], die_max_mm2: 9 }, [{ field: "die_max_mm2", value: 10, label: "" }], nodes)).toEqual({ die_width_mm: 3.1, die_height_mm: 3.1, die_max_mm2: 10 });
  });
});
