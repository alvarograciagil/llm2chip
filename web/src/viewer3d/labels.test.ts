import { describe, expect, it } from "vitest";
import { placeLabels, type LabelSpot } from "./labels";

const spot = (x: number, y: number, depth: number, visible = true): LabelSpot => ({ x, y, depth, visible, full: { w: 120, h: 40 }, short: { w: 40, h: 18 } });

describe("the dies' labels", () => {
  it("shows every label in full when none covers another", () => {
    expect(placeLabels([spot(100, 100, 0.5), spot(300, 100, 0.6), spot(100, 300, 0.7), spot(300, 300, 0.8)], null)).toEqual(["full", "full", "full", "full"]);
  });

  it("keeps the nearest in full and shortens the one behind it", () => {
    expect(placeLabels([spot(100, 100, 0.9), spot(130, 110, 0.4)], null)).toEqual(["short", "full"]);
  });

  it("always shows the chosen die in full", () => {
    expect(placeLabels([spot(100, 100, 0.9), spot(130, 110, 0.4)], 0)).toEqual(["full", "short"]);
  });

  it("hides the labels of dies behind the camera", () => {
    expect(placeLabels([spot(100, 100, 0.5, false), spot(110, 100, 0.6)], null)).toEqual(["hidden", "full"]);
  });

  it("lets a short label stop a farther one from showing in full", () => {
    // The middle one is shortened by the nearest; its name still keeps the farthest short.
    const forms = placeLabels([spot(100, 100, 0.1), spot(100, 140, 0.2), spot(100, 158, 0.3)], null);
    expect(forms).toEqual(["full", "short", "short"]);
  });
});
