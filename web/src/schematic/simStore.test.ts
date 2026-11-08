import { describe, expect, it } from "vitest";
import { expectedTotal, parseList, toSigned, validateInputs } from "./simStore";
import type { CellInfo } from "../engine";

const info: CellInfo = {
  config: { weight_bits: 4, activation_bits: 8, accumulator_bits: 16, mac_stages: 1, adder_stages: 1, addr_bits: 3, coding: "affine", exponent_bits: 0 },
  latency: 2,
  counts: {},
};

describe("inputs", () => {
  it("parses signed integers with mixed separators", () => {
    expect(parseList("3, -2 5;1")).toEqual([3n, -2n, 5n, 1n]);
    expect(parseList("3, x")).toBeNull();
  });

  it("rejects values that do not fit the hardware", () => {
    expect(validateInputs(info, "8", "1")).toHaveProperty("error");
    expect(validateInputs(info, "-8", "127")).toHaveProperty("weights");
    expect(validateInputs(info, "1,2", "1")).toHaveProperty("error");
    expect(validateInputs(info, "1,1,1,1,1,1,1,1,1", "1,1,1,1,1,1,1,1,1")).toHaveProperty("error");
  });
});

describe("reference arithmetic", () => {
  it("computes a dot product and wraps at the accumulator width", () => {
    expect(expectedTotal([3n, -2n], [100n, -50n], 16)).toBe(400n);
    expect(expectedTotal([127n], [32767n], 16)).toBe(BigInt.asIntN(16, 127n * 32767n));
  });

  it("reads two's complement bit strings", () => {
    expect(toSigned("1111")).toBe(-1n);
    expect(toSigned("0111")).toBe(7n);
  });
});
