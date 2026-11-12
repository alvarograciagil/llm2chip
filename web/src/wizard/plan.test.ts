import { describe, expect, it } from "vitest";
import { boardsWithin, DEFAULT_ANSWERS, fpgaParams, makers, modelParams, suggested, type FpgaBoard } from "./plan";

const board = (id: string, maker: string, lut: FpgaBoard["lut"], luts_k: number, bram_mbit: number, price_usd: number): FpgaBoard => ({ id, maker, name: id, fpga: "", url: "", image: null, luts_k, lut, bram_mbit, dsp: 300, ddr_gb: 1, ddr_gbps: 3, clock_mhz: 40, clock_measured: false, price_usd, price: "", tools: "", base_logic_k: lut === "LUT6" ? 25.4 : 28.6 });
const BOARDS = [board("pequeña", "Sipeed", "LUT4", 23, 1, 35), board("mega", "Sipeed", "LUT4", 138, 6.1, 195), board("kv260", "AMD", "LUT6", 117, 23.1, 249), board("alveo", "AMD", "LUT6", 1728, 432, 9000)];

describe("asistente de proyecto", () => {
  it("diseña para la placa: su reloj y su memoria los pone el motor", () => {
    const inside = fpgaParams(DEFAULT_ANSWERS, BOARDS[1], false);
    expect(inside).toMatchObject({ fpga_board: "mega", weight_backend: "sram", dies: 1 });
    expect(fpgaParams(DEFAULT_ANSWERS, BOARDS[1], true).weight_backend).toBe("external_ddr");
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
    expect(boardsWithin(BOARDS, 250)[0].id).toBe("kv260");
  });

  it("deja fuera las que no tienen lógica para el chip, y agrupa por fabricante", () => {
    expect(boardsWithin(BOARDS, 10000).map((b) => b.id)).not.toContain("pequeña");
    expect(makers(BOARDS)).toEqual(["AMD", "Sipeed"]);
  });
});
