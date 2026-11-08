import { useEffect, useMemo, useRef, useState, type ChangeEvent } from "react";
import type { Maturity, QualityRisk } from "../engine";
import { formatNumber, fromLog, toLog } from "../format";
import { PARAM_GROUPS, type Field } from "../params";
import { useDesign } from "../store";
import { Disclosure, Segmented, Switch } from "./controls";

const LOG_STEPS = 1000;

type RangeField = Extract<Field, { kind: "range" }>;

function ValueInput({ field, value, onCommit }: { field: RangeField; value: number; onCommit: (v: number) => void }) {
  const [draft, setDraft] = useState<string | null>(null);
  const commit = () => {
    if (draft === null) return;
    const parsed = Number(draft.replace(/\./g, "").replace(",", "."));
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
        aria-label={`Valor de ${field.label}`}
        value={draft ?? formatNumber(value)}
        onFocus={(e) => {
          setDraft(String(value).replace(".", ","));
          e.target.select();
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
      {field.unit && <small>{field.unit}</small>}
    </span>
  );
}

function Row({ field, children, below }: { field: Field; children: React.ReactNode; below?: React.ReactNode }) {
  const setHelp = useDesign((s) => s.setHelp);
  const value = useDesign((s) => s.params[field.key]);
  const fallback = useDesign((s) => s.defaults[field.key]);
  const resetParam = useDesign((s) => s.resetParam);
  const revealed = useDesign((s) => s.revealed);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (revealed?.key === field.key) ref.current?.scrollIntoView({ block: "center", behavior: "smooth" });
  }, [revealed, field.key]);

  const changed = value !== fallback;
  return (
    <div
      ref={ref}
      className={revealed?.key === field.key ? "row flash" : "row"}
      onMouseEnter={() => setHelp(field.help ?? null)}
      onMouseLeave={() => setHelp(null)}
      onFocus={() => setHelp(field.help ?? null)}
    >
      <div className="row-head">
        <span className="row-label">
          {field.label}
          {changed && (
            <button className="changed" aria-label={`Restablecer ${field.label}`} title="Restablecer" onClick={() => resetParam(field.key)} />
          )}
        </span>
        {children}
      </div>
      {below}
    </div>
  );
}

function RangeRow({ field }: { field: RangeField }) {
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
          aria-label={field.label}
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

function FieldRow({ field }: { field: Field }) {
  const value = useDesign((s) => s.params[field.key]);
  const setParam = useDesign((s) => s.setParam);
  if (field.kind === "range") return <RangeRow field={field} />;
  if (field.kind === "toggle") {
    return (
      <Row field={field}>
        <Switch checked={value as boolean} label={field.label} onChange={(v) => setParam(field.key, v)} />
      </Row>
    );
  }
  return (
    <Row field={field}>
      <select aria-label={field.label} value={value as string} onChange={(e) => setParam(field.key, e.target.value)}>
        {field.options.map(([id, label]) => (
          <option key={id} value={id}>
            {label}
          </option>
        ))}
      </select>
    </Row>
  );
}

function ModelSection() {
  const models = useDesign((s) => s.models);
  const model = useDesign((s) => s.model);
  const selectModel = useDesign((s) => s.selectModel);
  const addModel = useDesign((s) => s.addModel);
  const nodes = useDesign((s) => s.nodes);
  const node = useDesign((s) => s.params.node) as string;
  const setParam = useDesign((s) => s.setParam);

  const onFile = async (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) addModel(file.name.replace(/\.json$/, ""), await file.text());
    e.target.value = "";
  };

  return (
    <div className="inset standalone">
      <div className="row">
        <div className="row-head">
          <span className="row-label">Modelo</span>
          <select aria-label="Modelo" value={model?.id} onChange={(e) => selectModel(e.target.value)}>
            {models.map((m) => (
              <option key={m.id} value={m.id}>
                {m.name}
              </option>
            ))}
          </select>
        </div>
      </div>
      <div className="row">
        <div className="row-head">
          <span className="row-label">Tecnología</span>
          <select aria-label="Tecnología" value={node} onChange={(e) => setParam("node", e.target.value)}>
            {nodes.map((n) => (
              <option key={n.id} value={n.id}>
                {n.label}
              </option>
            ))}
          </select>
        </div>
      </div>
      <label className="row link-row">
        Cargar config.json de otro modelo…
        <input type="file" accept=".json,application/json" onChange={onFile} />
      </label>
    </div>
  );
}

function ParametersTab() {
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
      <input className="search" type="search" placeholder="Buscar parámetro" aria-label="Buscar parámetro" value={query} onChange={(e) => setQuery(e.target.value)} />
      {!needle && <ModelSection />}
      {groups.map((group) => (
        <Disclosure
          key={group.title}
          title={group.title}
          defaultOpen={group.open}
          forceOpen={needle !== "" || group.fields.some((f) => f.key === revealed?.key)}
          summary={`${group.fields.length}`}
        >
          {group.fields.map((field) => (
            <FieldRow key={field.key} field={field} />
          ))}
        </Disclosure>
      ))}
      {groups.length === 0 && <p className="empty">Ningún parámetro coincide.</p>}
      <button className="plain" onClick={reset}>
        Restablecer todo
      </button>
    </>
  );
}

const MATURITY: [Maturity, string, string][] = [
  ["probada", "Probadas", "Técnicas ya usadas en chips reales."],
  ["plausible", "Plausibles", "Razonables; hay que medir su efecto aquí."],
  ["experimental", "Experimentales", "Investigación abierta, para probar y comparar."],
];

const RISK: Record<QualityRisk, string> = {
  ninguno: "No altera la salida",
  medir: "Medir la calidad",
  alto: "Riesgo alto de calidad",
};

function OptimizationsTab() {
  const catalog = useDesign((s) => s.catalog);
  const active = useDesign((s) => s.params.optimizations);
  const toggle = useDesign((s) => s.toggleOptimization);
  const setHelp = useDesign((s) => s.setHelp);

  return (
    <>
      {MATURITY.map(([maturity, title, subtitle]) => {
        const items = catalog.filter((o) => o.maturity === maturity);
        return (
          <Disclosure key={maturity} title={title} defaultOpen={true} summary={`${items.filter((o) => active.includes(o.id)).length} de ${items.length}`}>
            <p className="group-note">{subtitle}</p>
            {items.map((o) => (
              <div className="opt" key={o.id} onMouseEnter={() => setHelp(o.what)} onMouseLeave={() => setHelp(null)}>
                <div className="row-head">
                  <span className="row-label">{o.name}</span>
                  <Switch checked={active.includes(o.id)} label={o.name} onChange={() => toggle(o.id)} />
                </div>
                <p className="opt-effect">{o.effect}</p>
                <p className={`opt-risk risk-${o.quality_risk}`}>{RISK[o.quality_risk]}</p>
              </div>
            ))}
          </Disclosure>
        );
      })}
    </>
  );
}

export function Sidebar() {
  const tab = useDesign((s) => s.sidebarTab);
  const setTab = useDesign((s) => s.setSidebarTab);
  const help = useDesign((s) => s.help);
  const active = useDesign((s) => s.params.optimizations.length);

  return (
    <aside className="sidebar">
      <div className="sidebar-tabs">
        <Segmented
          label="Panel"
          value={tab}
          options={[
            ["parametros", "Parámetros"],
            ["optimizaciones", active > 0 ? `Optimizaciones (${active})` : "Optimizaciones"],
          ]}
          onChange={setTab}
        />
      </div>
      <div className="sidebar-body">{tab === "parametros" ? <ParametersTab /> : <OptimizationsTab />}</div>
      <footer className="help" aria-live="polite">
        {help ?? "Pasa el cursor sobre un ajuste para ver qué hace."}
      </footer>
    </aside>
  );
}
