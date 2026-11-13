import { describe, expect, it } from "vitest";
import { boardsWithin, DEFAULT_ANSWERS, fpgaParams, makers, modelParams, suggested, type FpgaBoard } from "./plan";

const board = (id: string, maker: string, lut: FpgaBoard["lut"], luts_k: number, bram_mbit: number, price_usd: number): FpgaBoard => ({ id, maker, name: id, fpga: "", url: "", image: null, luts_k, lut, bram_mbit, dsp: 300, ddr_gb: 1, ddr_gbps: 3, clock_mhz: 40, clock_measured: false, price_usd, price: "", tools: "", base_logic_k: lut === "LUT6" ? 12.8 : 17.9 });
const BOARDS = [board("small", "Sipeed", "LUT4", 15, 1, 35), board("mega", "Sipeed", "LUT4", 138, 6.1, 195), board("kv260", "AMD", "LUT6", 117, 23.1, 249), board("alveo", "AMD", "LUT6", 1728, 432, 9000)];

describe("project assistant", () => {
  it("designs for the board: the engine sets its clock and memory", () => {
    const inside = fpgaParams(DEFAULT_ANSWERS, BOARDS[1], false);
    expect(inside).toMatchObject({ fpga_board: "mega", weight_backend: "sram", dies: 1 });
    expect(fpgaParams(DEFAULT_ANSWERS, BOARDS[1], true).weight_backend).toBe("external_ddr");
  });

  it("lets the number of chips follow from the model", () => {
    expect(modelParams({ ...DEFAULT_ANSWERS, chip_split: "tensor" })).toMatchObject({ dies: 0, chip_split: "tensor" });
  });

  it("suggests the model the goal asks for among those that fit", () => {
    const rows = [
      { id: "small", params_g: 0.0003, tok_s: 900, fits: true },
      { id: "medium", params_g: 0.015, tok_s: 300, fits: true },
      { id: "huge", params_g: 27, tok_s: 1, fits: false },
    ];
    expect(suggested(rows, "validate")).toBe("small");
    expect(suggested(rows, "largest")).toBe("medium");
    expect(suggested(rows, "fastest")).toBe("small");
  });
});

describe("boards by budget", () => {
  it("with about $250, the one that keeps the most inside the chip is the Kria KV260", () => {
    expect(boardsWithin(BOARDS, 250)[0].id).toBe("kv260");
  });

  it("leaves out those without logic for the chip, and groups by maker", () => {
    expect(boardsWithin(BOARDS, 10000).map((b) => b.id)).not.toContain("small");
    expect(makers(BOARDS)).toEqual(["AMD", "Sipeed"]);
  });
});
