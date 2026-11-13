import type { Report } from "./engine";
import { formatNumber } from "./format";

const cell = (text: string) => text.replaceAll("|", "\\|");

/** A one-page description of the current design, in Markdown. */
export function datasheet(name: string, modelName: string, report: Report, params: Record<string, unknown>): string {
  const lines: string[] = [];
  lines.push(`# Datasheet: ${name}`, "");
  lines.push(`Model: **${modelName}** · Technology: **${String(params.node)}** · ${report.dies} dies of ${formatNumber(report.die_area_mm2)} mm², ${formatNumber(report.total_area_mm2)} mm² in total.`, "");
  lines.push("## Measurements", "", "| Group | Measurement | Value | Source |", "|---|---|---|---|");
  for (const m of report.metrics) {
    lines.push(`| ${cell(m.section)} | ${cell(m.label)} | ${formatNumber(m.value)} ${m.unit} | ${m.confidence} |`);
  }
  lines.push("", "## Area per die", "", "| Block | mm² | Why it has this size |", "|---|---|---|");
  for (const block of [...report.blocks].sort((a, b) => b.per_die_mm2 - a.per_die_mm2)) {
    lines.push(`| ${cell(block.label)} | ${formatNumber(block.per_die_mm2)} | ${cell(block.why)} |`);
  }
  if (report.violations.length > 0) {
    lines.push("", "## Warnings", "");
    for (const v of report.violations) lines.push(`- **${v.severity === "error" ? "Error" : "Warning"}:** ${v.message} ${v.fix}`);
  }
  lines.push("", "## Parameters", "", "```json", JSON.stringify(params, null, 2), "```", "");
  lines.push("Values labelled as assumed are not measured. The logic area comes from the generated netlist; memories and energy are calibrated against published chips.", "");
  return lines.join("\n");
}
