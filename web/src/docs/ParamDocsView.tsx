import { useEffect } from "react";
import { EXTRA_DOCS } from "../paramDocs";
import { PARAM_GROUPS, type Field } from "../params";
import { useDesign } from "../store";
import { locale } from "../format";

/** What a field accepts, in words. */
function accepted(field: Field): string {
  switch (field.kind) {
    case "range": {
      const unit = field.unit ? ` ${field.unit}` : "";
      return `From ${field.min.toLocaleString(locale())} to ${field.max.toLocaleString(locale())}${unit}${field.log ? ", on a logarithmic scale" : ""}.`;
    }
    case "select":
      return `Options: ${field.options.map(([, text]) => text).join(", ")}.`;
    case "toggle":
      return "On or off.";
    case "preset":
      return `${field.sections.length} families of ready-made formats.`;
  }
}

export const anchorOf = (key: string) => `parameter-${key}`;

/** Every parameter of the chip, explained. */
export function ParamDocsView() {
  const focus = useDesign((s) => s.docsFocus);

  useEffect(() => {
    if (focus) document.getElementById(anchorOf(focus))?.scrollIntoView({ block: "start", behavior: "smooth" });
  }, [focus]);

  return (
    <section className="stage docs-stage">
      <article className="docs-column">
        <h1>Chip parameters</h1>
        <p className="muted-text">
          Everything that can be adjusted, what it changes in the circuit and where it shows. Those in the left panel move the estimator at once; the generator's are derived by the compiler from the model and the microprogram; the generation ones are chosen with each request.
        </p>
        <nav className="docs-index" aria-label="Sections">
          {PARAM_GROUPS.map((group) => (
            <a key={group.title} href={`#group-${group.title}`}>{group.title}</a>
          ))}
          {EXTRA_DOCS.map((section) => (
            <a key={section.id} href={`#${section.id}`}>{section.title}</a>
          ))}
        </nav>
        {PARAM_GROUPS.map((group) => (
          <section key={group.title} id={`group-${group.title}`}>
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
