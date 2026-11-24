import { describe, expect, it } from "vitest";
import type { Report, VoiceDiePlan } from "../engine";
import { fitCamera, hitTest, mix, partsLabel } from "./floorplanRender";

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

describe("parts of the chip", () => {
  const es = (text: string) => ({ Text: "Texto", Vision: "Visión", Listening: "Escucha", Speaking: "Habla", "sound decoder": "decodificador de sonido" })[text] ?? text;

  it("names what a die is for, and which stage of speaking a shared die holds", () => {
    expect(partsLabel(["text", "vision"], [])).toBe("Text + Vision");
    expect(partsLabel(["listen", "speak"], ["speech to text", "sound decoder"], es)).toBe("Escucha + Habla (decodificador de sonido)");
    expect(partsLabel(["speak"], ["speech writer", "sound decoder"])).toBe("Speaking");
  });

  it("finds a voice die under the pointer", () => {
    const voice: VoiceDiePlan = { index: 4, rect: { x: 10, y: 50, w: 8, h: 8 }, parts: ["listen"], stages: ["speech to text"], units: 1, lanes: 64, cells: 2048, parts_mm2: [3, 26, 16, 0], clock_mhz: 1000, reference: true };
    const report = { floorplan: { width: 60, height: 60, num_layers: 1, dies: [], voice: [voice], memory: [] }, blocks: [] } as unknown as Report;
    const hit = hitTest(report, 12, 52, 1);
    expect(hit?.kind).toBe("voice");
    expect(hit?.label).toContain("Die 4 · Listening · 45 mm²");
    expect(hit?.label).toContain("reference");
    expect(hitTest(report, 30, 30, 1)).toBeNull();
  });
});
