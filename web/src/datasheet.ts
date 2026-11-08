import type { Report } from "./engine";
import { formatNumber } from "./format";

const cell = (text: string) => text.replaceAll("|", "\\|");

/** A one-page description of the current design, in Markdown. */
export function datasheet(name: string, modelName: string, report: Report, params: Record<string, unknown>): string {
  const lines: string[] = [];
  lines.push(`# Hoja de datos: ${name}`, "");
  lines.push(`Modelo: **${modelName}** · Tecnología: **${String(params.node)}** · ${report.dies} dies de ${formatNumber(report.die_area_mm2)} mm², ${formatNumber(report.total_area_mm2)} mm² en total.`, "");
  lines.push("## Medidas", "", "| Grupo | Medida | Valor | Origen |", "|---|---|---|---|");
  for (const m of report.metrics) {
    lines.push(`| ${cell(m.section)} | ${cell(m.label)} | ${formatNumber(m.value)} ${m.unit} | ${m.confidence} |`);
  }
  lines.push("", "## Área por die", "", "| Bloque | mm² | Por qué tiene este tamaño |", "|---|---|---|");
  for (const block of [...report.blocks].sort((a, b) => b.per_die_mm2 - a.per_die_mm2)) {
    lines.push(`| ${cell(block.label)} | ${formatNumber(block.per_die_mm2)} | ${cell(block.why)} |`);
  }
  if (report.violations.length > 0) {
    lines.push("", "## Avisos", "");
    for (const v of report.violations) lines.push(`- **${v.severity === "error" ? "Error" : "Aviso"}:** ${v.message} ${v.fix}`);
  }
  lines.push("", "## Parámetros", "", "```json", JSON.stringify(params, null, 2), "```", "");
  lines.push("Los valores etiquetados como supuestos no están medidos. El área de lógica procede de la netlist generada; las memorias y la energía están calibradas contra chips publicados.", "");
  return lines.join("\n");
}
