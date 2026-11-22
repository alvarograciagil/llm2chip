import { afterEach, describe, expect, it, vi } from "vitest";
import { memoryWarning } from "./browserModels";

const MB = 2 ** 20;
const device = (memory: number | undefined, touch: boolean) => {
  vi.stubGlobal("navigator", { deviceMemory: memory });
  vi.stubGlobal("matchMedia", () => ({ matches: touch }));
};

describe("the memory a model needs", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("warns when a model takes more than a quarter of the device's memory", () => {
    device(8, false);
    expect(memoryWarning(259 * MB)).toBeNull();
    device(2, true);
    expect(memoryWarning(259 * MB)).toBe("It needs about 777 MB of memory, and this device has 2 GB: the page may close.");
  });

  it("warns phones and tablets that do not say how much they have", () => {
    device(undefined, true);
    expect(memoryWarning(MB)).toBeNull();
    expect(memoryWarning(259 * MB)).toBe("It needs about 777 MB of memory: on a phone or tablet the page may close.");
    device(undefined, false);
    expect(memoryWarning(259 * MB)).toBeNull();
  });
});
