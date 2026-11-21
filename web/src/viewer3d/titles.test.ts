import { describe, expect, it } from "vitest";
import { translate } from "../i18n";

describe("trace titles in Spanish", () => {
  it("translates the small model's and a design's", () => {
    expect(translate("es", "Small model on 4 dies, in groups of 2")).toBe("Modelo pequeño en 4 dies, en grupos de 2");
    expect(translate("es", "Qwen3.8-27B on 16 dies, in groups of 8")).toBe("Qwen3.8-27B en 16 dies, en grupos de 8");
  });
});
