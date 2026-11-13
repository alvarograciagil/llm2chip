import type { ChipParams, TechNode } from "../engine";
import { dieLimit, fitDie } from "../dieSize";
import { formatNumber } from "../format";
import { useT } from "../i18n";
import { Icon } from "../components/icons";
import { useDesign } from "../store";
import type { Advice, Change } from "./useAdvice";

/** The parameters a set of changes leaves, with what each change drags along in the interface. */
export function applyChanges(params: ChipParams, changes: Change[], nodes: TechNode[]): Record<string, unknown> {
  const area = Number(params.die_max_mm2);
  const width = Number(params.die_width_mm) || Math.sqrt(area);
  const height = Number(params.die_height_mm) || area / width;
  const patch: Record<string, unknown> = {};
  for (const change of changes) {
    patch[change.field] = change.value;
    // All the weights at one width: the checkpoint's own mix would keep some wider.
    if (change.field === "weight_bits") patch.use_model_mix = false;
    if (change.field === "node") Object.assign(patch, fitDie(width, height, dieLimit(nodes.find((n) => n.id === change.value)?.die_max_mm2)));
    if (change.field === "die_max_mm2") {
      const side = Math.floor(Math.sqrt(Number(change.value)) * 10) / 10;
      Object.assign(patch, { die_width_mm: side, die_height_mm: side, die_max_mm2: Number(change.value) });
    }
  }
  return patch;
}

/** Whether the design can be built as it stands, why not, and the changes that make it so. */
export function FeasibilityPanel({ advice, checking, searching }: { advice: Advice | null; checking: boolean; searching: boolean }) {
  const t = useT();
  const params = useDesign((s) => s.params);
  const nodes = useDesign((s) => s.nodes);
  const setParams = useDesign((s) => s.setParams);
  if (!advice) return checking ? <p className="muted-text">{t("Checking whether it can be built…")}</p> : null;
  const v = advice.verdict;
  const chips = (n: number) => (n === 1 ? t("1 chip") : `${formatNumber(n)} ${t("chips")}`);

  return (
    <>
      <h3>{t("Feasibility")}</h3>
      <div className={`inset padded feasibility ${v.layout ? "ok" : v.possible ? "partial" : "blocked"}`} aria-busy={checking}>
        <p className="feasibility-verdict">
          <Icon name={v.possible ? "check" : "warning"} size={16} />
          <strong>
            {v.possible
              ? `${t("It can be built")}: ${chips(v.dies)} ${t("of")} ${formatNumber(v.die_area_mm2)} mm²${v.designs > 1 ? `, ${formatNumber(v.designs)} ${t("distinct designs")}` : ""}`
              : t("It cannot be built with these parameters")}
          </strong>
        </p>
        {v.possible && v.layout && <p className="muted-text">{t("And laid out with OpenROAD on this process.")}</p>}
        {v.reasons.length > 0 && (
          <ul className="feasibility-reasons">
            {v.reasons.map((reason) => (
              <li key={reason}>{t(reason)}</li>
            ))}
          </ul>
        )}
        {v.notes.map((note) => (
          <p key={note} className="muted-text">
            {t(note)}
          </p>
        ))}
        {advice.suggestions.length > 0 && (
          <>
            <p className="feasibility-heading">{advice.suggestions[0].cheaper ? t("Cheaper to make") : t("How to make it feasible")}</p>
            {advice.suggestions.map((s, i) => (
              <div key={i} className="suggestion">
                <ul>
                  {s.changes.map((c) => (
                    <li key={c.field}>
                      {t(c.label)}
                      {c.risk && <span className="risk"> · {t(c.risk)}</span>}
                    </li>
                  ))}
                </ul>
                <div className="suggestion-foot">
                  <span>
                    {chips(s.dies)} {t("of")} {formatNumber(s.die_area_mm2)} mm²{s.designs < s.dies ? ` · ${s.designs === 1 ? t("a single design") : `${formatNumber(s.designs)} ${t("distinct designs")}`}` : ""} · {formatNumber(s.tok_s)} tok/s
                  </span>
                  <button className="pill primary-pill" onClick={() => setParams(applyChanges(params, s.changes, nodes))}>
                    {t("Apply")}
                  </button>
                </div>
              </div>
            ))}
          </>
        )}
        {searching && !v.layout && <p className="muted-text">{t("Looking for changes that make it feasible…")}</p>}
        {!searching && !v.layout && advice.suggestions.length === 0 && !v.possible && (
          <p className="muted-text">{t("No change of the parameters makes it feasible on this process: it needs a smaller model or a process with larger dies.")}</p>
        )}
      </div>
    </>
  );
}
