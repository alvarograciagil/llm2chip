import { useEffect } from "react";
import { EXTRA_DOCS } from "../paramDocs";
import { PARAM_GROUPS, type Field } from "../params";
import { useDesign } from "../store";
import { locale } from "../format";
import { useT } from "../i18n";
import { PARAM_HELP_ES } from "../paramHelpEs";

/** What a field accepts, in words. */
function accepted(field: Field, t: (text: string) => string): string {
  switch (field.kind) {
    case "range": {
      const unit = field.unit ? ` ${t(field.unit)}` : "";
      const range = `${field.min.toLocaleString(locale())} ${t("to")} ${field.max.toLocaleString(locale())}${unit}`;
      return field.log ? t(`From ${range}, on a logarithmic scale.`) : t(`From ${range}.`);
    }
    case "select":
      return `${t("Options")}: ${field.options.map(([, text]) => t(text)).join(", ")}.`;
    case "toggle":
      return t("On or off.");
    case "preset":
      return t(`${field.sections.length} families of ready-made formats.`);
  }
}

export const anchorOf = (key: string) => `parameter-${key}`;

/** Every parameter of the chip, explained. */
export function ParamDocsView() {
  const focus = useDesign((s) => s.docsFocus);
  const language = useDesign((s) => s.language);
  const t = useT();

  useEffect(() => {
    if (focus) document.getElementById(anchorOf(focus))?.scrollIntoView({ block: "start", behavior: "smooth" });
  }, [focus]);

  return (
    <section className="stage docs-stage">
      <article className="docs-column">
        <h1>{t("Chip parameters")}</h1>
        <p className="muted-text">
          {t("Everything that can be adjusted, what it changes in the circuit and where it shows. Those in the left panel move the estimator at once; the generator's are derived by the compiler from the model and the microprogram; the generation ones are chosen with each request.")}
        </p>
        <nav className="docs-index" aria-label={t("Sections")}>
          {PARAM_GROUPS.map((group) => (
            <a key={group.title} href={`#group-${group.title}`}>{t(group.title)}</a>
          ))}
          {EXTRA_DOCS.map((section) => (
            <a key={section.id} href={`#${section.id}`}>{t(section.title)}</a>
          ))}
        </nav>
        {PARAM_GROUPS.map((group) => (
          <section key={group.title} id={`group-${group.title}`}>
            <h2>{t(group.title)}</h2>
            <dl className="docs-list">
              {group.fields.map((field) => (
                <div key={field.key} id={anchorOf(field.key)} className={focus === field.key ? "docs-entry focused" : "docs-entry"}>
                  <dt>
                    {t(field.label)} <code>{field.key}</code>
                  </dt>
                  <dd>
                    <p>{language === "es" ? (PARAM_HELP_ES[field.key] ?? field.help) : field.help}</p>
                    <p className="muted-text">{accepted(field, t)}</p>
                  </dd>
                </div>
              ))}
            </dl>
          </section>
        ))}
        {EXTRA_DOCS.map((section) => (
          <section key={section.id} id={section.id}>
            <h2>{t(section.title)}</h2>
            <p className="muted-text">{t(section.intro)}</p>
            <dl className="docs-list">
              {section.entries.map((entry) => (
                <div key={entry.name} className="docs-entry">
                  <dt>{t(entry.name)}</dt>
                  <dd>
                    <p>{t(entry.text)}</p>
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
