import { describe, expect, it } from "vitest";
import { formatNumber, fromLog, toLog } from "./format";

describe("formatNumber", () => {
  it("uses English separators and sensible precision", () => {
    expect(formatNumber(19.523)).toBe("19.5");
    expect(formatNumber(0.537)).toBe("0.54");
    expect(formatNumber(3820.28)).toBe("3,820");
  });
});

describe("log sliders", () => {
  it("round-trips a value", () => {
    expect(fromLog(toLog(8192, 256, 262144), 256, 262144)).toBeCloseTo(8192, 6);
  });
});
