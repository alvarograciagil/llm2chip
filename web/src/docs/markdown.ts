import { EXTRA_DOCS } from "../paramDocs";
import { PARAM_GROUPS, type Field } from "../params";

function accepted(field: Field): string {
  switch (field.kind) {
    case "range":
      return `De ${field.min} a ${field.max}${field.unit ? ` ${field.unit}` : ""}${field.log ? ", en escala logarítmica" : ""}.`;
    case "select":
      return `Opciones: ${field.options.map(([, text]) => text).join(", ")}.`;
    case "toggle":
      return "Activado o desactivado.";
    case "preset":
      return `Formatos ya hechos: ${field.sections.map((s) => s.title).join(", ")}.`;
  }
}

/** `docs/parametros.md`, from the same text the app shows. */
export function parametersMarkdown(): string {
  const lines = [
    "# Parámetros del chip",
    "",
    "<!-- Generado desde web/src/params.ts y web/src/paramDocs.ts: UPDATE_DOCS=1 npx vitest run docs -->",
    "",
    "Todo lo que se puede ajustar, qué cambia en el circuito y en qué se nota. Los del panel de la aplicación mueven el estimador al momento; los del generador los deduce el compilador del modelo y del microprograma; los de generación se eligen en cada petición. En la aplicación, cada parámetro se explica al pasar el cursor por su «?» y abre esta misma documentación al pulsarlo.",
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
