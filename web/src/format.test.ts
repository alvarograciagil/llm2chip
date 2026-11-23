import { describe, expect, it } from "vitest";
import { formatNumber, formatTokenTime, fromLog, toLog } from "./format";

describe("formatNumber", () => {
  it("uses English separators and sensible precision", () => {
    expect(formatNumber(19.523)).toBe("19.5");
    expect(formatNumber(0.537)).toBe("0.54");
    expect(formatNumber(3820.28)).toBe("3,820");
  });
});

describe("formatTokenTime", () => {
  it("shows short times in milliseconds, with the speed they give", () => {
    expect(formatTokenTime(0.001706)).toBe("1.71 ms/token · 586 tok/s");
    expect(formatTokenTime(0.29)).toBe("0.29 s/token · 3.45 tok/s");
    expect(formatTokenTime(21.5)).toBe("21.5 s/token · 0.047 tok/s");
  });
});

describe("log sliders", () => {
  it("round-trips a value", () => {
    expect(fromLog(toLog(8192, 256, 262144), 256, 262144)).toBeCloseTo(8192, 6);
  });
});
