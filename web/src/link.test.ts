import { describe, expect, it } from "vitest";
import { readLink } from "./link";

describe("links", () => {
  it("open a view, and the simulator with a model", () => {
    expect(readLink("#schematic")).toEqual({ view: "schematic", model: null });
    expect(readLink("#simulator")).toEqual({ view: "simulator", model: null });
    expect(readLink("#simulator/stories260K")).toEqual({ view: "simulator", model: "stories260K" });
    expect(readLink("#simulator/HuggingFaceTB/SmolLM2-135M-Instruct")).toEqual({ view: "simulator", model: "HuggingFaceTB/SmolLM2-135M-Instruct" });
  });

  it("leave a shared design and anything else alone", () => {
    expect(readLink("#d=eyJtb2RlbCI6")).toBeNull();
    expect(readLink("")).toBeNull();
    expect(readLink("#nothing")).toBeNull();
    expect(readLink("#floorplan/stories260K")).toEqual({ view: "floorplan", model: null });
  });
});
