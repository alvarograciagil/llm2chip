import { EXTRA_DOCS } from "../paramDocs";
import { PARAM_GROUPS, type Field } from "../params";

function accepted(field: Field): string {
  switch (field.kind) {
    case "range":
      return `From ${field.min} to ${field.max}${field.unit ? ` ${field.unit}` : ""}${field.log ? ", on a logarithmic scale" : ""}.`;
    case "select":
      return `Options: ${field.options.map(([, text]) => text).join(", ")}.`;
    case "toggle":
      return "On or off.";
    case "preset":
      return `Ready-made formats: ${field.sections.map((s) => s.title).join(", ")}.`;
  }
}

/** `docs/parameters.md`, from the same text the app shows. */
export function parametersMarkdown(): string {
  const lines = [
    "# Chip parameters",
    "",
    "<!-- Generated from web/src/params.ts and web/src/paramDocs.ts: UPDATE_DOCS=1 npx vitest run docs -->",
    "",
    "Everything that can be adjusted, what it changes in the circuit and where it shows. Those in the application's panel move the estimator at once; the generator's are derived by the compiler from the model and the microprogram; the generation ones are chosen with each request. In the application, each parameter is explained when the cursor goes over its \u201c?\u201d, and clicking it opens this same documentation.",
    "",
  ];
  for (const group of PARAM_GROUPS) {
    lines.push(`## ${group.title}`, "");
    for (const field of group.fields) {
      lines.push(`### ${field.label} (\`${field.key}\`)`, "", field.help ?? "", "", `*${accepted(field)}*`, "");
    }
  }
  for (const section of EXTRA_DOCS) {
    lines.push(`## ${section.title}`, "", section.intro, "");
    for (const entry of section.entries) {
      lines.push(`### ${entry.name}`, "", entry.text, "");
    }
  }
  return lines.join("\n");
}
