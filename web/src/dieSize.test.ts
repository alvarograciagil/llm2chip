import { describe, expect, it } from "vitest";
import { dieLimit, fitDie } from "./dieSize";

describe("die size", () => {
  it("shrinks a die that does not fit the process, keeping its shape", () => {
    const die = fitDie(28.5, 28.6, 10);
    expect(die.die_max_mm2).toBeLessThanOrEqual(10);
    expect(die.die_max_mm2).toBeGreaterThan(9);
    expect(die.die_width_mm / die.die_height_mm).toBeCloseTo(28.5 / 28.6, 1);
  });

  it("leaves a die that fits as it is", () => {
    expect(fitDie(3, 3, 10)).toEqual({ die_width_mm: 3, die_height_mm: 3, die_max_mm2: 9 });
  });

  it("never goes past the reticle", () => {
    expect(dieLimit(undefined)).toBe(858);
    expect(dieLimit(1200)).toBe(858);
    expect(dieLimit(10)).toBe(10);
  });
});
