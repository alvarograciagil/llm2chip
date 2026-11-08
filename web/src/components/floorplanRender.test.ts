import { describe, expect, it } from "vitest";
import { fitCamera, mix } from "./floorplanRender";

describe("fitCamera", () => {
  const rect = { x: 0, y: 0, w: 100, h: 50 };

  it("waits until the view is larger than its padding", () => {
    expect(fitCamera(rect, 1, 1, 56)).toBeNull();
  });

  it("centres the rectangle inside the padding", () => {
    const cam = fitCamera(rect, 1000, 1000, 50)!;
    expect(cam.k).toBeCloseTo(9);
    expect((rect.x - cam.x) * cam.k).toBeCloseTo(50);
    expect((rect.y - cam.y) * cam.k).toBeCloseTo(275);
  });
});

describe("mix", () => {
  it("interpolates between colours", () => {
    expect(mix("#000000", "#ffffff", 0.5)).toBe("rgb(128,128,128)");
  });
});
