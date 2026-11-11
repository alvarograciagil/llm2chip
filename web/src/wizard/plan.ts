import type { ChipParams, Report } from "../engine";

/** Where a new project starts from. */
export type Start = "fpga" | "chip" | "model" | "custom";

/** An FPGA board the chip can be tried on. Figures from the makers; prices are approximate
 * and change with the shop and the region. */
export interface FpgaBoard {
  id: string;
  maker: string;
  name: string;
  /** The maker's own page for the board. */
  url: string;
  /** A photo on the maker's site, where one could be checked; otherwise none. */
  image?: string;
  fpga: string;
  /** Look-up tables, in thousands, of the kind the family has: Intel's logic elements
   * count like four-input tables. */
  luts_k: number;
  lut: "LUT4" | "LUT6" | "LE";
  /** Block memory on the FPGA, UltraRAM included. */
  bram_mbit: number;
  dsp: number;
  /** Memory on the board, and roughly what it delivers per second. */
  ddr_gb: number;
  ddr_gbps: number;
  /** A clock the generated logic should reach without hand placement. */
  clock_mhz: number;
  /** Approximate price in dollars, to sort by, and as the shops show it. */
  price_usd: number;
  price: string;
  /** Free tools that can build for it. */
  tools: string;
}

export const FPGA_BOARDS: FpgaBoard[] = [
  { id: "tang-primer-25k", maker: "Sipeed", url: "https://wiki.sipeed.com/hardware/en/tang/tang-primer-25k/primer-25k.html", image: "https://wiki.sipeed.com/hardware/zh/tang/tang-primer-25k/assets/25k_dock_45.jpg", name: "Sipeed Tang Primer 25K", fpga: "Gowin GW5A-25", luts_k: 23, lut: "LUT4", bram_mbit: 1.0, dsp: 28, ddr_gb: 0, ddr_gbps: 0, clock_mhz: 40, price_usd: 35, price: "~35 $", tools: "Gowin EDA" },
  { id: "tang-mega-60k", maker: "Sipeed", url: "https://wiki.sipeed.com/hardware/en/tang/tang-mega-60k/mega-60k.html", image: "https://wiki.sipeed.com/hardware/zh/tang/tang-mega-60k/assets/mega_60k_top.png", name: "Sipeed Tang Mega 60K Dock", fpga: "Gowin GW5AT-60", luts_k: 60, lut: "LUT4", bram_mbit: 2.1, dsp: 118, ddr_gb: 0.25, ddr_gbps: 1.6, clock_mhz: 40, price_usd: 90, price: "~90 $", tools: "Gowin EDA" },
  { id: "tang-mega-138k-dock", maker: "Sipeed", url: "https://wiki.sipeed.com/hardware/en/tang/tang-mega-138k/mega-138k.html", image: "https://wiki.sipeed.com/hardware/zh/tang/tang-mega-138k/assets/mega_138k_top.jpg", name: "Sipeed Tang Mega 138K Dock", fpga: "Gowin GW5AST-138", luts_k: 138, lut: "LUT4", bram_mbit: 6.1, dsp: 298, ddr_gb: 0.5, ddr_gbps: 1.6, clock_mhz: 40, price_usd: 130, price: "~130 $", tools: "Gowin EDA" },
  { id: "ulx3s-85f", maker: "Radiona", url: "https://radiona.org/ulx3s/", name: "Radiona ULX3S 85F", fpga: "Lattice ECP5 LFE5U-85F", luts_k: 84, lut: "LUT4", bram_mbit: 3.7, dsp: 156, ddr_gb: 0.032, ddr_gbps: 0.2, clock_mhz: 40, price_usd: 150, price: "~130-150 $", tools: "Yosys + nextpnr, libres" },
  { id: "atum-a3-nano", maker: "Terasic", url: "https://www.terasic.com.tw/cgi-bin/page/archive.pl?Language=English&CategoryNo=44&No=1373", name: "Terasic Atum A3 Nano", fpga: "Altera Agilex 3 A3CZ135", luts_k: 135, lut: "LE", bram_mbit: 8.3, dsp: 368, ddr_gb: 0.064, ddr_gbps: 0.2, clock_mhz: 80, price_usd: 180, price: "180-245 $", tools: "Quartus Prime Pro (licencia gratuita para Agilex 3)" },
  { id: "tang-mega-138k", maker: "Sipeed", url: "https://wiki.sipeed.com/hardware/en/tang/tang-mega-138k/mega-138k-pro.html", image: "https://wiki.sipeed.com/hardware/en/tang/tang-mega-138k/assets/mega_138k_pro_top.png", name: "Sipeed Tang Mega 138K Pro", fpga: "Gowin GW5AST-138", luts_k: 138, lut: "LUT4", bram_mbit: 6.1, dsp: 298, ddr_gb: 1, ddr_gbps: 3.2, clock_mhz: 40, price_usd: 195, price: "~195 $", tools: "Gowin EDA" },
  { id: "pynq-z2", maker: "TUL", url: "https://www.tulembedded.com/FPGA/ProductsPYNQ-Z2.html", name: "TUL PYNQ-Z2", fpga: "Xilinx Zynq XC7Z020", luts_k: 53, lut: "LUT6", bram_mbit: 4.9, dsp: 220, ddr_gb: 0.5, ddr_gbps: 2.1, clock_mhz: 50, price_usd: 200, price: "200-260 $", tools: "Vivado (edición gratuita)" },
  { id: "de10-nano", maker: "Terasic", url: "https://www.terasic.com.tw/cgi-bin/page/archive.pl?Language=English&No=1046", name: "Terasic DE10-Nano", fpga: "Intel Cyclone V SE 5CSEBA6", luts_k: 110, lut: "LE", bram_mbit: 5.6, dsp: 112, ddr_gb: 1, ddr_gbps: 3.2, clock_mhz: 40, price_usd: 225, price: "~225 $", tools: "Quartus Prime Lite (gratuito)" },
  { id: "kv260", maker: "AMD", url: "https://www.amd.com/en/products/system-on-modules/kria/k26/kv260-vision-starter-kit.html", name: "AMD Kria KV260", fpga: "Zynq UltraScale+ K26 (clase ZU5EV)", luts_k: 117, lut: "LUT6", bram_mbit: 23.1, dsp: 1248, ddr_gb: 4, ddr_gbps: 19, clock_mhz: 100, price_usd: 249, price: "~249 $", tools: "Vivado" },
  { id: "arty-a7-100t", maker: "Digilent", url: "https://digilent.com/shop/arty-a7-100t-artix-7-fpga-development-board/", name: "Digilent Arty A7-100T", fpga: "Xilinx XC7A100T", luts_k: 63, lut: "LUT6", bram_mbit: 4.9, dsp: 240, ddr_gb: 0.25, ddr_gbps: 2.6, clock_mhz: 50, price_usd: 250, price: "~250 $", tools: "Vivado (edición gratuita)" },
  { id: "zybo-z7-20", maker: "Digilent", url: "https://digilent.com/shop/zybo-z7-zynq-7000-arm-fpga-soc-development-board/", name: "Digilent Zybo Z7-20", fpga: "Xilinx Zynq XC7Z020", luts_k: 53, lut: "LUT6", bram_mbit: 4.9, dsp: 220, ddr_gb: 1, ddr_gbps: 4.3, clock_mhz: 50, price_usd: 300, price: "~300 $", tools: "Vivado (edición gratuita)" },
  { id: "kr260", maker: "AMD", url: "https://www.amd.com/en/products/system-on-modules/kria/k26/kr260-robotics-starter-kit.html", name: "AMD Kria KR260", fpga: "Zynq UltraScale+ K26 (clase ZU5EV)", luts_k: 117, lut: "LUT6", bram_mbit: 23.1, dsp: 1248, ddr_gb: 4, ddr_gbps: 19, clock_mhz: 100, price_usd: 349, price: "~349 $", tools: "Vivado" },
  { id: "alinx-ax7202", maker: "ALINX", url: "https://www.en.alinx.com/Product/FPGA-Development-Boards/Artix-7/AX7202.html", name: "ALINX AX7202", fpga: "Xilinx XC7A200T", luts_k: 134, lut: "LUT6", bram_mbit: 13.1, dsp: 740, ddr_gb: 1, ddr_gbps: 6.4, clock_mhz: 50, price_usd: 450, price: "400-500 $", tools: "Vivado (edición gratuita)" },
  { id: "nexys-video", maker: "Digilent", url: "https://digilent.com/shop/nexys-video-artix-7-fpga-trainer-board-for-multimedia-applications/", name: "Digilent Nexys Video", fpga: "Xilinx XC7A200T", luts_k: 134, lut: "LUT6", bram_mbit: 13.1, dsp: 740, ddr_gb: 0.5, ddr_gbps: 3.2, clock_mhz: 50, price_usd: 550, price: "~500 €", tools: "Vivado (edición gratuita)" },
  { id: "genesys-2", maker: "Digilent", url: "https://digilent.com/shop/genesys-2-kintex-7-fpga-development-board/", name: "Digilent Genesys 2", fpga: "Xilinx Kintex-7 XC7K325T", luts_k: 204, lut: "LUT6", bram_mbit: 16, dsp: 840, ddr_gb: 1, ddr_gbps: 7.2, clock_mhz: 80, price_usd: 1000, price: "~1 000 $", tools: "Vivado (licencia del dispositivo incluida)" },
  { id: "alveo-u250", maker: "AMD", url: "https://www.amd.com/en/products/accelerators/alveo/u250/a-u250-a64g-pq-g.html", name: "AMD Alveo U250", fpga: "Xilinx UltraScale+ XCU250", luts_k: 1728, lut: "LUT6", bram_mbit: 432, dsp: 12288, ddr_gb: 64, ddr_gbps: 77, clock_mhz: 200, price_usd: 9000, price: "~9 000 $", tools: "Vivado" },
];

/** The smallest chip the generator makes, measured with Yosys: CPU, die and sampler. */
export const BASE_LOGIC_K = { LUT4: 48.3, LUT6: 45.9, LE: 48.3 };
/** Microcode, CPU memory and function tables of that chip, which any model adds to, with
 * its microcode kept as a dictionary and the embedding left in the image. */
export const BASE_MEMORY_MBIT = 1.95;

/** The makers, in the order the menu shows them. */
export const FPGA_MAKERS: string[] = [...new Set([...FPGA_BOARDS].sort((a, b) => a.maker.localeCompare(b.maker)).map((b) => b.maker))];

/** The boards that cost at most `budget` dollars, with the one that holds the most on
 * chip first: on-chip memory is what lets a model run without waiting on the board's. */
export function boardsWithin(budget: number): FpgaBoard[] {
  return FPGA_BOARDS.filter((b) => b.price_usd <= budget && BASE_LOGIC_K[b.lut] <= b.luts_k).sort((a, b) => b.bram_mbit - a.bram_mbit || b.ddr_gbps - a.ddr_gbps);
}

/** The answers to the wizard's questions. Each start uses only some of them. */
export interface Answers {
  start: Start;
  model: string;
  board: string;
  node: string;
  /** What to favour where there is a choice. */
  goal: "validate" | "largest" | "fastest";
  weight_bits: number;
  context_tokens: number;
  target_tok_s: number;
  die_max_mm2: number;
  weight_backend: "sram" | "rom";
  chip_split: "layers" | "tensor" | "mixed";
}

export const DEFAULT_ANSWERS: Answers = {
  start: "model",
  model: "",
  board: "kv260",
  node: "n6",
  goal: "validate",
  weight_bits: 8,
  context_tokens: 2048,
  target_tok_s: 1000,
  die_max_mm2: 858,
  weight_backend: "sram",
  chip_split: "layers",
};

/** Where a model's weights and cache go on an FPGA board, or why they do not fit. */
export type FpgaFit = { where: "chip" | "ddr"; memory_mbit: number; logic_share: number } | { where: "none"; reason: string };

/**
 * Whether a model fits a board: its weights and cache in the block memory with the
 * generator's own memories, or else in the board's memory, which then sets the speed.
 */
export function fpgaFit(board: FpgaBoard, weightsGb: number, kvGb: number): FpgaFit {
  const logic = BASE_LOGIC_K[board.lut] / board.luts_k;
  if (logic > 1) return { where: "none", reason: `La lógica básica del chip ocupa ${Math.round(BASE_LOGIC_K[board.lut])} K ${board.lut} y la FPGA tiene ${board.luts_k} K.` };
  const memory = (weightsGb + kvGb) * 8000 + BASE_MEMORY_MBIT;
  if (memory <= board.bram_mbit) return { where: "chip", memory_mbit: memory, logic_share: logic };
  if (weightsGb + kvGb <= board.ddr_gb * 0.9) return { where: "ddr", memory_mbit: memory, logic_share: logic };
  const external = board.ddr_gb > 0 ? `ni en los ${board.ddr_gb} GB de la placa` : "y la placa no tiene memoria externa";
  return { where: "none", reason: `Necesita ${formatMbit(memory)} y no cabe en los ${board.bram_mbit} Mbit de la FPGA ${external}.` };
}

export function formatMbit(mbit: number): string {
  return mbit >= 8000 ? `${(mbit / 8000).toFixed(1)} GB` : `${mbit.toFixed(1)} Mbit`;
}

/** The parameters every start shares: the model's limits come from the model itself. */
function common(a: Answers): Partial<ChipParams> {
  return { weight_bits: a.weight_bits, context_tokens: a.context_tokens, use_model_mix: false };
}

/** The design parameters an FPGA board stands for, with the model placed where it fits. */
export function fpgaParams(a: Answers, board: FpgaBoard, fit: FpgaFit): Partial<ChipParams> {
  const outside = fit.where === "ddr";
  return {
    ...common(a),
    node: "fpga_u250",
    dies: 1,
    die_max_mm2: 1200,
    freq_ghz: board.clock_mhz / 1000,
    target_tok_s: 0,
    // The multipliers go in the DSP blocks; the generator's chip uses a few, a large board more.
    macs_per_die: Math.max(8, Math.min(board.dsp, 1024)),
    vector_lanes_per_die: board.dsp >= 1000 ? 64 : 4,
    weight_backend: outside ? "external_ddr" : "sram",
    kv_backend: outside ? "external_ddr" : "sram",
    ddr_gbps: Math.max(board.ddr_gbps, 0.01),
    ddr_capacity_gb: Math.max(board.ddr_gb, 0.004),
    chip_split: "layers",
  };
}

/** One die: the area and process chosen, and as many multipliers as the speed asks. */
export function chipParams(a: Answers): Partial<ChipParams> {
  return { ...common(a), node: a.node, dies: 1, die_max_mm2: a.die_max_mm2, target_tok_s: a.target_tok_s, weight_backend: a.weight_backend, kv_backend: "sram", chip_split: "layers" };
}

/** A model as the base: the number of chips follows from what it needs. */
export function modelParams(a: Answers): Partial<ChipParams> {
  return { ...common(a), node: a.node, dies: 0, die_max_mm2: a.die_max_mm2, target_tok_s: a.target_tok_s, weight_backend: a.weight_backend, kv_backend: "sram", chip_split: a.chip_split };
}

export function metric(report: Report | null, id: string): number {
  return report?.metrics.find((m) => m.id === id)?.value ?? NaN;
}

/** Whether an estimate is a design that can be built: no errors among its checks. */
export function buildable(report: Report | null): boolean {
  return !!report && !report.violations.some((v) => v.severity === "error");
}

/** The model the goal points to among those that fit: the smallest, the largest or the fastest. */
export function suggested<T extends { id: string; params_g: number; tok_s: number; fits: boolean }>(rows: T[], goal: Answers["goal"]): string | null {
  const fitting = rows.filter((r) => r.fits);
  if (fitting.length === 0) return null;
  const by = goal === "validate" ? (r: T) => -r.params_g : goal === "largest" ? (r: T) => r.params_g : (r: T) => r.tok_s;
  return fitting.reduce((best, r) => (by(r) > by(best) ? r : best)).id;
}
