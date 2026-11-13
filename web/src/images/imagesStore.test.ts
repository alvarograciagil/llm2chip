import { describe, expect, it } from "vitest";
import { regionBetween, regionIsUsable } from "./imagesStore";

describe("la zona que se retoca", () => {
  it("ordena las esquinas y no sale de la imagen", () => {
    expect(regionBetween([300.4, 20], [-5, 600], 512)).toEqual([0, 20, 300, 512]);
  });

  it("debe cubrir al menos un código de la retícula", () => {
    expect(regionIsUsable([0, 0, 31, 100], 512)).toBe(false);
    expect(regionIsUsable([0, 0, 32, 32], 512)).toBe(true);
    expect(regionIsUsable(null, 512)).toBe(false);
  });
});
