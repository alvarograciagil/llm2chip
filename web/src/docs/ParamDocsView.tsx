import { useEffect } from "react";
import { EXTRA_DOCS } from "../paramDocs";
import { PARAM_GROUPS, type Field } from "../params";
import { useDesign } from "../store";

/** What a field accepts, in words. */
function accepted(field: Field): string {
  switch (field.kind) {
    case "range": {
      const unit = field.unit ? ` ${field.unit}` : "";
      return `De ${field.min.toLocaleString("es")} a ${field.max.toLocaleString("es")}${unit}${field.log ? ", en escala logarítmica" : ""}.`;
    }
    case "select":
      return `Opciones: ${field.options.map(([, text]) => text).join(", ")}.`;
    case "toggle":
      return "Activado o desactivado.";
    case "preset":
      return `${field.sections.length} familias de formatos ya hechos.`;
  }
}

export const anchorOf = (key: string) => `parametro-${key}`;

/** Every parameter of the chip, explained. */
export function ParamDocsView() {
  const focus = useDesign((s) => s.docsFocus);

  useEffect(() => {
    if (focus) document.getElementById(anchorOf(focus))?.scrollIntoView({ block: "start", behavior: "smooth" });
  }, [focus]);

  return (
    <section className="stage docs-stage">
      <article className="docs-column">
        <h1>Parámetros del chip</h1>
        <p className="muted-text">
          Todo lo que se puede ajustar, qué cambia en el circuito y en qué se nota. Los del panel izquierdo mueven el estimador al momento; los del generador los deduce el compilador del modelo y del microprograma; los de generación se eligen en cada petición.
        </p>
        <nav className="docs-index" aria-label="Secciones">
          {PARAM_GROUPS.map((group) => (
            <a key={group.title} href={`#grupo-${group.title}`}>{group.title}</a>
          ))}
          {EXTRA_DOCS.map((section) => (
            <a key={section.id} href={`#${section.id}`}>{section.title}</a>
          ))}
        </nav>
        {PARAM_GROUPS.map((group) => (
          <section key={group.title} id={`grupo-${group.title}`}>
            <h2>{group.title}</h2>
            <dl className="docs-list">
              {group.fields.map((field) => (
                <div key={field.key} id={anchorOf(field.key)} className={focus === field.key ? "docs-entry focused" : "docs-entry"}>
                  <dt>
                    {field.label} <code>{field.key}</code>
                  </dt>
                  <dd>
                    <p>{field.help}</p>
                    <p className="muted-text">{accepted(field)}</p>
                  </dd>
                </div>
              ))}
            </dl>
          </section>
        ))}
        {EXTRA_DOCS.map((section) => (
          <section key={section.id} id={section.id}>
            <h2>{section.title}</h2>
            <p className="muted-text">{section.intro}</p>
            <dl className="docs-list">
              {section.entries.map((entry) => (
                <div key={entry.name} className="docs-entry">
                  <dt>{entry.name}</dt>
                  <dd>
                    <p>{entry.text}</p>
                  </dd>
                </div>
              ))}
            </dl>
          </section>
        ))}
      </article>
    </section>
  );
}
