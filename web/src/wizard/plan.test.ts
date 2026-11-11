import { describe, expect, it } from "vitest";
import { boardsWithin, DEFAULT_ANSWERS, FPGA_BOARDS, fpgaFit, fpgaParams, modelParams, suggested } from "./plan";

const board = (id: string) => FPGA_BOARDS.find((b) => b.id === id)!;

describe("asistente de proyecto", () => {
  it("pone un modelo diminuto en la memoria de la FPGA", () => {
    // stories260K a 8 bits: unos 0,26 MB de pesos y una caché pequeña.
    const fit = fpgaFit(board("nexys-video"), 0.00026, 0.00005);
    expect(fit.where).toBe("chip");
  });

  it("lleva los pesos a la DDR cuando no caben dentro", () => {
    const fit = fpgaFit(board("tang-mega-138k"), 0.015, 0.001);
    expect(fit.where).toBe("ddr");
    expect(fpgaParams(DEFAULT_ANSWERS, board("tang-mega-138k"), fit).weight_backend).toBe("external_ddr");
  });

  it("dice por qué no cabe", () => {
    const tiny = fpgaFit(board("tang-primer-25k"), 0.0001, 0);
    expect(tiny.where).toBe("none");
    expect(tiny.where === "none" && tiny.reason).toMatch(/lógica/);
    const huge = fpgaFit(board("arty-a7-100t"), 16, 1);
    expect(huge.where === "none" && huge.reason).toMatch(/no cabe/);
  });

  it("deja que el número de chips salga del modelo", () => {
    expect(modelParams({ ...DEFAULT_ANSWERS, chip_split: "tensor" })).toMatchObject({ dies: 0, chip_split: "tensor" });
  });

  it("propone el modelo que pide el objetivo entre los que caben", () => {
    const rows = [
      { id: "pequeño", params_g: 0.0003, tok_s: 900, fits: true },
      { id: "mediano", params_g: 0.015, tok_s: 300, fits: true },
      { id: "enorme", params_g: 27, tok_s: 1, fits: false },
    ];
    expect(suggested(rows, "validate")).toBe("pequeño");
    expect(suggested(rows, "largest")).toBe("mediano");
    expect(suggested(rows, "fastest")).toBe("pequeño");
  });
});

describe("placas por presupuesto", () => {
  it("con unos 250 $, la que más guarda dentro del chip es la Kria KV260", () => {
    expect(boardsWithin(250)[0].id).toBe("kv260");
  });

  it("deja fuera las que no tienen lógica para el chip", () => {
    expect(boardsWithin(1000).map((b) => b.id)).not.toContain("tang-primer-25k");
  });
});
