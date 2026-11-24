import { useEffect, useMemo, useRef, useState, type ChangeEvent } from "react";
import { fpgaBoards, type Maturity, type QualityRisk } from "../engine";
import { makers, type FpgaBoard } from "../wizard/plan";
import { formatNumber, fromLog, locale, toLog } from "../format";
import { dieLimit, fitDie } from "../dieSize";
import { matchingPreset, PARAM_GROUPS, type Field } from "../params";
import { useDesign } from "../store";
import { useT } from "../i18n";
import { PARAM_HELP_ES } from "../paramHelpEs";
import { Disclosure, InfoTip, Segmented, Switch } from "./controls";

const LOG_STEPS = 1000;

type RangeField = Extract<Field, { kind: "range" }>;

function ValueInput({ field, value, onCommit }: { field: RangeField; value: number; onCommit: (v: number) => void }) {
  const t = useT();
  const [draft, setDraft] = useState<string | null>(null);
  const commit = () => {
    if (draft === null) return;
    // Thousands separators out, and the decimal mark as JavaScript reads it.
    const parsed = Number(locale() === "es-ES" ? draft.replace(/\./g, "").replace(",", ".") : draft.replace(/,/g, ""));
    if (Number.isFinite(parsed)) {
      const clamped = Math.min(field.max, Math.max(field.min, parsed));
      onCommit(field.integer ? Math.round(clamped) : clamped);
    }
    setDraft(null);
  };
  return (
    <span className="value">
      <input
        type="text"
        inputMode="decimal"
        aria-label={t(`Value of ${t(field.label)}`)}
        value={draft ?? formatNumber(value)}
        onFocus={(e) => {
          setDraft(locale() === "es-ES" ? String(value).replace(".", ",") : String(value));
          // After the draft is drawn, or the new text would clear the selection.
          const input = e.target;
          requestAnimationFrame(() => input.select());
        }}
        onChange={(e) => setDraft(e.target.value)}
        onBlur={commit}
        onKeyDown={(e) => {
          if (e.key === "Enter") e.currentTarget.blur();
          if (e.key === "Escape") {
            setDraft(null);
            e.currentTarget.blur();
          }
        }}
      />
      {field.unit && <small>{t(field.unit)}</small>}
    </span>
  );
}

function Row({ field, children, below }: { field: Field; children: React.ReactNode; below?: React.ReactNode }) {
  const t = useT();
  const setHelp = useDesign((s) => s.setHelp);
  const openDocs = useDesign((s) => s.openDocs);
  const value = useDesign((s) => s.params[field.key]);
  const fallback = useDesign((s) => s.defaults[field.key]);
  const resetParam = useDesign((s) => s.resetParam);
  const revealed = useDesign((s) => s.revealed);
  const language = useDesign((s) => s.language);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (revealed?.key === field.key) ref.current?.scrollIntoView({ block: "center", behavior: "smooth" });
  }, [revealed, field.key]);

  const changed = value !== fallback;
  const help = field.help && (language === "es" ? (PARAM_HELP_ES[field.key] ?? field.help) : field.help);
  return (
    <div
      ref={ref}
      className={revealed?.key === field.key ? "row flash" : "row"}
      onMouseEnter={() => setHelp(help ?? null)}
      onMouseLeave={() => setHelp(null)}
      onFocus={() => setHelp(help ?? null)}
    >
      <div className="row-head">
        <span className="row-label">
          {t(field.label)}
          {help && <InfoTip id={`ayuda-${field.key}`} label={t(field.label)} text={help} onOpen={() => openDocs(field.key)} />}
          {changed && (
            <button className="changed" aria-label={`${t("Reset")} ${t(field.label)}`} title={t("Reset")} onClick={() => resetParam(field.key)} />
          )}
        </span>
        {children}
      </div>
      {below}
    </div>
  );
}

function RangeRow({ field }: { field: RangeField }) {
  const t = useT();
  const value = useDesign((s) => s.params[field.key]) as number;
  const setParam = useDesign((s) => s.setParam);
  const position = field.log ? toLog(Math.max(value, field.min), field.min, field.max) * LOG_STEPS : value;
  const onChange = (e: ChangeEvent<HTMLInputElement>) => {
    const raw = Number(e.target.value);
    const next = field.log ? fromLog(raw / LOG_STEPS, field.min, field.max) : raw;
    setParam(field.key, field.integer ? Math.round(next) : next);
  };
  const fill = field.log ? position / LOG_STEPS : (value - field.min) / (field.max - field.min);
  return (
    <Row
      field={field}
      below={
        <input
          type="range"
          aria-label={t(field.label)}
          style={{ ["--fill" as string]: `${Math.min(1, Math.max(0, fill)) * 100}%` }}
          min={field.log ? 0 : field.min}
          max={field.log ? LOG_STEPS : field.max}
          step={field.log ? 1 : field.step ?? (field.max - field.min) / 200}
          value={position}
          onChange={onChange}
        />
      }
    >
      <ValueInput field={field} value={value} onCommit={(v) => setParam(field.key, v)} />
    </Row>
  );
}

function PresetRow({ field }: { field: Extract<Field, { kind: "preset" }> }) {
  const t = useT();
  const params = useDesign((s) => s.params);
  const setParam = useDesign((s) => s.setParam);
  const current = matchingPreset(params) ?? "";
  const choose = (label: string) => {
    const preset = field.sections.flatMap((section) => section.presets).find((p) => p.label === label);
    if (preset) Object.entries(preset.values).forEach(([key, value]) => setParam(key, value as never));
  };
  return (
    <Row field={field}>
      <select aria-label={t(field.label)} value={current} onChange={(e) => choose(e.target.value)}>
        {current === "" && <option value="">{t("Custom")}</option>}
        {field.sections.map((section) => (
          <optgroup key={section.title} label={t(section.title)}>
            {section.presets.map((preset) => (
              <option key={preset.label} value={preset.label}>
                {t(preset.label)}
              </option>
            ))}
          </optgroup>
        ))}
      </select>
    </Row>
  );
}

/** A text the design carries: kept as it is typed, and given to the design when the field
 * is left or typing pauses. */
function TextRow({ field }: { field: Extract<Field, { kind: "text" }> }) {
  const t = useT();
  const value = String(useDesign((s) => s.params[field.key]) ?? "");
  const setParam = useDesign((s) => s.setParam);
  const [draft, setDraft] = useState(value);
  useEffect(() => setDraft(value), [value]);
  useEffect(() => {
    if (draft === value) return;
    const timer = setTimeout(() => setParam(field.key, draft), 600);
    return () => clearTimeout(timer);
  }, [draft, value, field.key, setParam]);
  return (
    <Row
      field={field}
      below={
        <textarea
          className="param-text"
          aria-label={t(field.label)}
          placeholder={field.placeholder && t(field.placeholder)}
          rows={3}
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onBlur={() => draft !== value && setParam(field.key, draft)}
        />
      }
    >
      {null}
    </Row>
  );
}

function FieldRow({ field }: { field: Field }) {
  const t = useT();
  const value = useDesign((s) => s.params[field.key]);
  const setParam = useDesign((s) => s.setParam);
  if (field.kind === "range") return <RangeRow field={field} />;
  if (field.kind === "preset") return <PresetRow field={field} />;
  if (field.kind === "text") return <TextRow field={field} />;
  if (field.kind === "toggle") {
    return (
      <Row field={field}>
        <Switch checked={value as boolean} label={t(field.label)} onChange={(v) => setParam(field.key, v)} />
      </Row>
    );
  }
  return (
    <Row field={field}>
      <select aria-label={t(field.label)} value={value as string} onChange={(e) => setParam(field.key, e.target.value)}>
        {field.options.map(([id, label]) => (
          <option key={id} value={id}>
            {t(label)}
          </option>
        ))}
      </select>
    </Row>
  );
}

/** What the design is built for: an FPGA board, or chips of its own, with how big each die
 * may be and how many there may be. Everything else follows from it. */
function TargetRows() {
  const t = useT();
  const nodes = useDesign((s) => s.nodes);
  const params = useDesign((s) => s.params);
  const setParam = useDesign((s) => s.setParam);
  const setParams = useDesign((s) => s.setParams);
  const boards = useMemo(() => fpgaBoards() as FpgaBoard[], []);
  const node = String(params.node);
  const onBoard = node.startsWith("fpga");
  const silicon = nodes.filter((n) => !n.id.startsWith("fpga"));
  const dies = Number(params.dies);
  const area = Number(params.die_max_mm2);
  const width = Number(params.die_width_mm) || Math.round(Math.sqrt(area) * 10) / 10;
  const height = Number(params.die_height_mm) || Math.round((area / width) * 10) / 10;
  const limitOf = (id: string) => dieLimit(nodes.find((n) => n.id === id)?.die_max_mm2);
  const limit = limitOf(node);
  // Editing one side keeps the other and stops where the process limit is reached.
  const setWidth = (w: number) => setParams(fitDie(Math.min(w, Math.floor((limit / height) * 10) / 10), height, limit));
  const setHeight = (h: number) => setParams(fitDie(width, Math.min(h, Math.floor((limit / width) * 10) / 10), limit));

  return (
    <>
      <div className="row">
        <div className="row-head">
          <span className="row-label">{t("Target")}</span>
          <Segmented
            label={t("Target")}
            value={onBoard ? "fpga" : "chip"}
            options={[["fpga", t("FPGA")], ["chip", t("Own chip")]]}
            onChange={(target) =>
              setParams(
                target === "fpga"
                  ? { node: "fpga_u250", fpga_board: params.fpga_board || "kv260", dies: 1 }
                  : (() => {
                      const chosen = silicon.some((n) => n.id === params.node) ? node : "n6";
                      // A die is at most the scanner's reticle, 26 × 33 mm, or what the process takes.
                      const fitted = area > 858 ? fitDie(26, 33, limitOf(chosen)) : fitDie(width, height, limitOf(chosen));
                      return { node: chosen, ...fitted };
                    })(),
              )
            }
          />
        </div>
      </div>
      {onBoard ? (
        <div className="row">
          <div className="row-head">
            <span className="row-label">{t("Board")}</span>
            <select aria-label={t("Board")} value={String(params.fpga_board)} onChange={(e) => setParam("fpga_board", e.target.value)}>
              <option value="">{t("No board: generic scale")}</option>
              {makers(boards).map((maker) => (
                <optgroup key={maker} label={maker}>
                  {boards
                    .filter((b) => b.maker === maker)
                    .map((b) => (
                      <option key={b.id} value={b.id}>
                        {t(b.name)}
                      </option>
                    ))}
                </optgroup>
              ))}
            </select>
          </div>
        </div>
      ) : (
        <>
          <div className="row">
            <div className="row-head">
              <span className="row-label">{t("Process")}</span>
              <select aria-label={t("Process")} value={node} onChange={(e) => setParams({ node: e.target.value, ...fitDie(width, height, limitOf(e.target.value)) })}>
                {silicon.map((n) => (
                  <option key={n.id} value={n.id}>
                    {t(n.label)}
                  </option>
                ))}
              </select>
            </div>
          </div>
          <div className="row">
            <div className="row-head">
              <span className="row-label">{t("Die size")}</span>
              <span className="die-size">
                <input type="number" min={1} max={33} step={0.5} aria-label={t("Die width")} value={width} onChange={(e) => setWidth(Number(e.target.value) || 1)} />
                ×
                <input type="number" min={1} max={33} step={0.5} aria-label={t("Die height")} value={height} onChange={(e) => setHeight(Number(e.target.value) || 1)} />
                <small>mm</small>
              </span>
            </div>
            {limit < 858 && <p className="row-note">{t("This process takes dies of up to")} {formatNumber(limit)} mm².</p>}
          </div>
          <div className="row">
            <div className="row-head">
              <span className="row-label">{t("Chips")}</span>
              <span className="die-size">
                <Segmented label={t("Chips")} value={dies === 0 ? "auto" : dies === 1 ? "1" : "n"} options={[["1", "1"], ["n", "N"], ["auto", t("No limit")]]} onChange={(v) => setParam("dies", v === "1" ? 1 : v === "auto" ? 0 : Math.max(2, dies))} />
                {dies > 1 && <input type="number" min={2} max={1024} aria-label={t("Number of chips")} value={dies} onChange={(e) => setParam("dies", Math.max(2, Number(e.target.value) || 2))} />}
              </span>
            </div>
          </div>
        </>
      )}
    </>
  );
}

function ModelSection() {
  const t = useT();
  const models = useDesign((s) => s.models);
  const model = useDesign((s) => s.model);
  const selectModel = useDesign((s) => s.selectModel);
  const addModel = useDesign((s) => s.addModel);
  const setView = useDesign((s) => s.setView);

  const onFile = async (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) addModel(file.name.replace(/\.json$/, ""), await file.text());
    e.target.value = "";
  };

  return (
    <div className="inset standalone">
      <div className="row">
        <div className="row-head">
          <span className="row-label">{t("Model")}</span>
          <select aria-label={t("Model")} value={model?.id} onChange={(e) => selectModel(e.target.value)}>
            {models.map((m) => (
              <option key={m.id} value={m.id}>
                {t(m.name)}
              </option>
            ))}
          </select>
        </div>
      </div>
      <TargetRows />
      <button className="row link-row" onClick={() => setView("models")}>
        {t("Find other models…")}
      </button>
      <label className="row link-row">
        {t("Load another model's config.json…")}
        <input type="file" accept=".json,application/json" onChange={onFile} />
      </label>
    </div>
  );
}

function ParametersTab() {
  const t = useT();
  const params = useDesign((s) => s.params);
  const reset = useDesign((s) => s.reset);
  const revealed = useDesign((s) => s.revealed);
  const [query, setQuery] = useState("");
  const needle = query.trim().toLowerCase();

  const groups = useMemo(
    () =>
      PARAM_GROUPS.map((group) => ({
        ...group,
        fields: group.fields.filter(
          (f) => (!f.when || f.when(params)) && (!needle || f.label.toLowerCase().includes(needle)),
        ),
      })).filter((g) => g.fields.length > 0),
    [params, needle],
  );

  return (
    <>
      <input className="search" type="search" placeholder={t("Search parameter")} aria-label={t("Search parameter")} value={query} onChange={(e) => setQuery(e.target.value)} />
      {!needle && <ModelSection />}
      {groups.map((group) => (
        <Disclosure
          key={group.title}
          title={t(group.title)}
          defaultOpen={group.open}
          forceOpen={needle !== "" || group.fields.some((f) => f.key === revealed?.key)}
          summary={`${group.fields.length}`}
        >
          {group.fields.map((field) => (
            <FieldRow key={field.key} field={field} />
          ))}
        </Disclosure>
      ))}
      {groups.length === 0 && <p className="empty">{t("No parameter matches.")}</p>}
      <button className="plain" onClick={reset}>
        {t("Reset all")}
      </button>
    </>
  );
}

const MATURITY: [Maturity, string, string][] = [
  ["proven", "Proven", "Techniques already used in real chips."],
  ["plausible", "Plausible", "Reasonable; their effect must be measured here."],
  ["experimental", "Experimental", "Open research, to try and compare."],
];

const RISK: Record<QualityRisk, string> = {
  lossless: "Does not change the output",
  measure: "Measure the quality",
  high: "High risk to quality",
};

function OptimizationsTab() {
  const t = useT();
  const catalog = useDesign((s) => s.catalog);
  const active = useDesign((s) => s.params.optimizations);
  const toggle = useDesign((s) => s.toggleOptimization);

  return (
    <>
      {MATURITY.map(([maturity, title, subtitle]) => {
        const items = catalog.filter((o) => o.maturity === maturity);
        return (
          <Disclosure key={maturity} title={t(title)} defaultOpen={true} summary={`${items.filter((o) => active.includes(o.id)).length} / ${items.length}`}>
            <p className="group-note">{t(subtitle)}</p>
            {items.map((o) => (
              <div className="opt" key={o.id} title={t(o.what)}>
                <div className="row-head">
                  <span className="row-label">{t(o.name)}</span>
                  <Switch checked={active.includes(o.id)} label={t(o.name)} onChange={() => toggle(o.id)} />
                </div>
                <p className="opt-effect">{t(o.effect)}</p>
                <p className={`opt-risk risk-${o.quality_risk}`}>{t(RISK[o.quality_risk])}</p>
              </div>
            ))}
          </Disclosure>
        );
      })}
    </>
  );
}

export function Sidebar() {
  const t = useT();
  const tab = useDesign((s) => s.sidebarTab);
  const setTab = useDesign((s) => s.setSidebarTab);
  const active = useDesign((s) => s.params.optimizations.length);

  return (
    <aside className="sidebar">
      <div className="sidebar-tabs">
        <Segmented
          label={t("Panel")}
          value={tab}
          options={[
            ["parameters", t("Parameters")],
            ["optimizations", active > 0 ? `${t("Optimizations")} (${active})` : t("Optimizations")],
          ]}
          onChange={setTab}
        />
      </div>
      <div className="sidebar-body">{tab === "parameters" ? <ParametersTab /> : <OptimizationsTab />}</div>
    </aside>
  );
}
