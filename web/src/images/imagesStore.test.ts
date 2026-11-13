import { describe, expect, it } from "vitest";
import { regionBetween, regionIsUsable } from "./imagesStore";

describe("the region being retouched", () => {
  it("orders the corners and stays inside the image", () => {
    expect(regionBetween([300.4, 20], [-5, 600], 512)).toEqual([0, 20, 300, 512]);
  });

  it("must cover at least one code of the grid", () => {
    expect(regionIsUsable([0, 0, 31, 100], 512)).toBe(false);
    expect(regionIsUsable([0, 0, 32, 32], 512)).toBe(true);
    expect(regionIsUsable(null, 512)).toBe(false);
  });
});
