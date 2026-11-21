import { describe, expect, it } from "vitest";
import type { NetlistModule } from "../engine";
import { enginePath, frameAt, scaledActivity, type Frame } from "./inside";

const module = {
  name: "die",
  title: "Die",
  description: "",
  ports: [],
  nets: [],
  instances: [
    { name: "unit0", kind: { kind: "module", of: "unit" }, group: "Units", conns: {} },
    { name: "unit1", kind: { kind: "module", of: "unit" }, group: "Units", conns: {} },
    { name: "seq", kind: { kind: "module", of: "sequencer" }, group: "Control", conns: {} },
  ],
} as unknown as NetlistModule;

describe("inside a die", () => {
  it("names an instance as the engine does", () => {
    expect(enginePath([])).toBe("");
    expect(enginePath(["unit0", "alu"])).toBe("/unit0/alu");
  });

  it("finds the frame of a cycle", () => {
    const frames: Frame[] = [{ cycle: 10, nets: {}, activity: {} }, { cycle: 11, nets: { pc: "1" }, activity: {} }];
    expect(frameAt(frames, 11.6)?.nets.pc).toBe("1");
    expect(frameAt(frames, 12)).toBeNull();
  });

  it("shades the busiest block fully, the others in proportion, and each area by its busiest", () => {
    const frame: Frame = { cycle: 0, nets: {}, activity: { unit0: 0.2, unit1: 0.05, seq: 0 } };
    expect(scaledActivity(module, frame)).toEqual({ unit0: 1, Units: 1, unit1: 0.25 });
    expect(scaledActivity(module, { ...frame, activity: {} })).toEqual({});
  });
});
