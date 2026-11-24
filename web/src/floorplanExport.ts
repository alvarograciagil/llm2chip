import type { Floorplan, Rect, Report, SideDiePlan } from "./engine";

const CATEGORY_FILL: Record<string, string> = {
  memory: "#4f8cff",
  compute: "#ff9f43",
  interconnect: "#a55eea",
  vector: "#26de81",
  state: "#fd79a8",
  system: "#a4b0be",
};

const escape = (text: string) => text.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c] ?? c);

function box(r: Rect, fill: string, title: string): string {
  return `<rect x="${r.x}" y="${r.y}" width="${r.w}" height="${r.h}" fill="${fill}" stroke="#222" stroke-width="0.05"><title>${escape(title)}</title></rect>`;
}

/** The plan as a vector drawing, with one rectangle per block. */
export function floorplanSvg(report: Report): string {
  const plan = report.floorplan;
  const labels = new Map(report.blocks.map((b) => [b.id, b]));
  const parts = [`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${plan.width} ${plan.height}" width="${plan.width * 4}mm" height="${plan.height * 4}mm">`];
  for (const die of plan.dies) {
    parts.push(box(die.rect, "#f5f6f8", `Die ${die.index}`));
    for (const block of die.blocks) {
      const info = labels.get(block.id);
      parts.push(box(block.rect, CATEGORY_FILL[info?.category ?? "system"] ?? "#ccc", info?.label ?? block.id));
    }
  }
  const fill = (v: SideDiePlan) => (v.parts.includes("draw") ? "#b3e5fc" : v.parts.includes("listen") ? "#ffcc80" : "#ff9eb0");
  for (const v of plan.side ?? []) parts.push(box(v.rect, fill(v), `Die ${v.index}: ${v.stages.join(" + ")}${v.reference ? " (reference)" : ""}`));
  for (const memory of plan.memory) parts.push(box(memory.rect, "#dfe4ea", memory.label));
  parts.push("</svg>");
  return parts.join("\n");
}

const databaseUnits = (mm: number) => Math.round(mm * 1000 * 1000);

/** The blocks as a DEF placement, one database unit per nanometre, for layout tools. */
export function floorplanDef(plan: Floorplan, name: string): string {
  const components = [
    ...plan.dies.flatMap((die) => die.blocks.map((block) => ({ name: `die${die.index}_${block.id}`, rect: block.rect }))),
    ...(plan.side ?? []).map((v) => ({ name: `die${v.index}_${v.stages.join("_").replace(/\W+/g, "_")}`, rect: v.rect })),
  ];
  const lines = [
    "VERSION 5.8 ;",
    `DESIGN ${name.replace(/\W+/g, "_")} ;`,
    "UNITS DISTANCE MICRONS 1000 ;",
    `DIEAREA ( 0 0 ) ( ${databaseUnits(plan.width)} ${databaseUnits(plan.height)} ) ;`,
    `COMPONENTS ${components.length} ;`,
    ...components.map((c) => `  - ${c.name} ${c.name}_macro + FIXED ( ${databaseUnits(c.rect.x)} ${databaseUnits(c.rect.y)} ) N ;`),
    "END COMPONENTS",
    "END DESIGN",
    "",
  ];
  return lines.join("\n");
}
