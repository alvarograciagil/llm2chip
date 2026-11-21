import { describe, expect, it } from "vitest";
import { opaque } from "./scene";

describe("the scene's colours", () => {
  it("lays translucent tokens over the canvas and leaves solid ones alone", () => {
    expect(opaque("rgba(255, 255, 255, 0.5)", "#000000")).toBe("rgb(128, 128, 128)");
    expect(opaque("rgba(0, 0, 0, 0.12)", "rgb(255, 255, 255)")).toBe("rgb(224, 224, 224)");
    expect(opaque("#0a84ff", "#000")).toBe("#0a84ff");
  });
});
