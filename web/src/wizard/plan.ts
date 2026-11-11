import type { ChipParams, Report } from "../engine";

/** Where a new project starts from. */
export type Start = "fpga" | "chip" | "model" | "custom";

/** An FPGA board the chip can be tried on, as the engine describes it. */
export interface FpgaBoard {
  id: string;
  maker: string;
  name: string;
  fpga: string;
  url: string;
  image: string | null;
  luts_k: number;
  lut: "LUT4" | "LUT6" | "LE";
  bram_mbit: number;
  dsp: number;
  ddr_gb: number;
  ddr_gbps: number;
  clock_mhz: number;
  clock_measured: boolean;
  price_usd: number;
  price: string;
  tools: string;
}

/** The makers of `boards`, in alphabetical order. */
export function makers(boards: FpgaBoard[]): string[] {
  return [...new Set(boards.map((b) => b.maker))].sort((a, b) => a.localeCompare(b));
}

/** The boards that cost at most `budget` dollars, with the one that holds the most on
 * chip first: on-chip memory is what lets a model run without waiting on the board's. */
export function boardsWithin(boards: FpgaBoard[], budget: number, baseLogicK: (lut: string) => number): FpgaBoard[] {
  return boards.filter((b) => b.price_usd <= budget && baseLogicK(b.lut) <= b.luts_k).sort((a, b) => b.bram_mbit - a.bram_mbit || b.ddr_gbps - a.ddr_gbps);
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

export function formatMbit(mbit: number): string {
  return mbit >= 8000 ? `${(mbit / 8000).toFixed(1)} GB` : `${mbit.toFixed(1)} Mbit`;
}

/** The parameters every start shares: the model's limits come from the model itself. */
function common(a: Answers): Partial<ChipParams> {
  return { weight_bits: a.weight_bits, context_tokens: a.context_tokens, use_model_mix: false };
}

/** The design for a board: the engine takes its clock and memory from the board. The
 * weights go inside the FPGA, or into the board's memory when `outside`. */
export function fpgaParams(a: Answers, board: FpgaBoard, outside: boolean): Partial<ChipParams> {
  return {
    ...common(a),
    node: "fpga_u250",
    fpga_board: board.id,
    dies: 1,
    die_max_mm2: 1200,
    target_tok_s: 0,
    freq_ghz: 1,
    // The multipliers go in the DSP blocks; the generator's chip uses a few, a large board more.
    macs_per_die: Math.max(8, Math.min(board.dsp - 8, 1024)),
    vector_lanes_per_die: board.dsp >= 1000 ? 4 : 1,
    vector_units_per_die: 1,
    weight_backend: outside ? "external_ddr" : "sram",
    kv_backend: outside ? "external_ddr" : "sram",
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
