import { useEffect, useMemo, useRef, useState, type ChangeEvent } from "react";
import { createPortal } from "react-dom";
import { fpgaBoards, type Maturity, type QualityRisk } from "../engine";
import { makers, type FpgaBoard } from "../wizard/plan";
import { formatNumber, fromLog, toLog } from "../format";
import { matchingPreset, PARAM_GROUPS, type Field } from "../params";
import { useDesign } from "../store";
import { useT } from "../i18n";
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
      {field.unit && <small>{field.unit}</small>}
    </span>
  );
}

/** A "?" that explains a parameter while the pointer is on it, above everything so no
 * scrolling panel cuts it, and opens its documentation when pressed. */
function InfoTip({ id, label, text, onOpen }: { id: string; label: string; text: string; onOpen: () => void }) {
  const [at, setAt] = useState<{ x: number; y: number; above: boolean } | null>(null);
  const show = (e: { currentTarget: HTMLElement }) => {
    const box = e.currentTarget.getBoundingClientRect();
    const above = box.bottom + 160 > window.innerHeight;
    setAt({ x: Math.max(8, Math.min(box.left - 8, window.innerWidth - 276)), y: above ? box.top - 6 : box.bottom + 6, above });
  };
  return (
    <span className="info">
      <button className="info-button" aria-label={`Qué es ${label}`} aria-describedby={id} onClick={onOpen} onMouseEnter={show} onMouseLeave={() => setAt(null)} onFocus={show} onBlur={() => setAt(null)}>
        ?
      </button>
      {/* The sidebar's blur would make a fixed position its own: the tip lives in the page. */}
      {createPortal(
        <span className={at ? "tooltip shown" : "tooltip"} role="tooltip" id={id} style={at ? { left: at.x, top: at.y, transform: at.above ? "translateY(-100%)" : undefined } : undefined}>
          {text}
          <em>Pulsa para ver la documentación completa.</em>
        </span>,
        document.body,
      )}
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
          {t(field.label)}
          {field.help && <InfoTip id={`ayuda-${field.key}`} label={t(field.label)} text={field.help} onOpen={() => openDocs(field.key)} />}
          {changed && (
            <button className="changed" aria-label={`Restablecer ${t(field.label)}`} title="Restablecer" onClick={() => resetParam(field.key)} />
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

function PresetRow({ field }: { field: Extract<Field, { kind: "preset" }> }) {
  const params = useDesign((s) => s.params);
  const setParam = useDesign((s) => s.setParam);
  const current = matchingPreset(params) ?? "";
  const choose = (label: string) => {
    const preset = field.sections.flatMap((section) => section.presets).find((p) => p.label === label);
    if (preset) Object.entries(preset.values).forEach(([key, value]) => setParam(key, value as never));
  };
  return (
    <Row field={field}>
      <select aria-label={field.label} value={current} onChange={(e) => choose(e.target.value)}>
        {current === "" && <option value="">Personalizado</option>}
        {field.sections.map((section) => (
          <optgroup key={section.title} label={section.title}>
            {section.presets.map((preset) => (
              <option key={preset.label} value={preset.label}>
                {preset.label}
              </option>
            ))}
          </optgroup>
        ))}
      </select>
    </Row>
  );
}

function FieldRow({ field }: { field: Field }) {
  const t = useT();
  const value = useDesign((s) => s.params[field.key]);
  const setParam = useDesign((s) => s.setParam);
  if (field.kind === "range") return <RangeRow field={field} />;
  if (field.kind === "preset") return <PresetRow field={field} />;
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
  const setSize = (w: number, h: number) => setParams({ die_width_mm: w, die_height_mm: h, die_max_mm2: Math.round(w * h) });

  return (
    <>
      <div className="row">
        <div className="row-head">
          <span className="row-label">{t("Destino")}</span>
          <Segmented
            label={t("Destino")}
            value={onBoard ? "fpga" : "chip"}
            options={[["fpga", t("FPGA")], ["chip", t("Chip propio")]]}
            onChange={(target) =>
              setParams(
                target === "fpga"
                  ? { node: "fpga_u250", fpga_board: params.fpga_board || "kv260", dies: 1 }
                  : // A die is at most the scanner's reticle, 26 × 33 mm.
                    { node: silicon.some((n) => n.id === params.node) ? params.node : "n6", ...(area > 858 ? { die_width_mm: 26, die_height_mm: 33, die_max_mm2: 858 } : {}) },
              )
            }
          />
        </div>
      </div>
      {onBoard ? (
        <div className="row">
          <div className="row-head">
            <span className="row-label">{t("Placa")}</span>
            <select aria-label={t("Placa")} value={String(params.fpga_board)} onChange={(e) => setParam("fpga_board", e.target.value)}>
              <option value="">{t("Sin placa: escala genérica")}</option>
              {makers(boards).map((maker) => (
                <optgroup key={maker} label={maker}>
                  {boards
                    .filter((b) => b.maker === maker)
                    .map((b) => (
                      <option key={b.id} value={b.id}>
                        {b.name}
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
              <span className="row-label">{t("Proceso")}</span>
              <select aria-label={t("Proceso")} value={node} onChange={(e) => setParam("node", e.target.value)}>
                {silicon.map((n) => (
                  <option key={n.id} value={n.id}>
                    {n.label}
                  </option>
                ))}
              </select>
            </div>
          </div>
          <div className="row">
            <div className="row-head">
              <span className="row-label">{t("Tamaño del die")}</span>
              <span className="die-size">
                <input type="number" min={1} max={33} step={0.5} aria-label={t("Ancho del die")} value={width} onChange={(e) => setSize(Number(e.target.value) || 1, height)} />
                ×
                <input type="number" min={1} max={33} step={0.5} aria-label={t("Alto del die")} value={height} onChange={(e) => setSize(width, Number(e.target.value) || 1)} />
                <small>mm</small>
              </span>
            </div>
          </div>
          <div className="row">
            <div className="row-head">
              <span className="row-label">{t("Chips")}</span>
              <span className="die-size">
                <Segmented label={t("Chips")} value={dies === 0 ? "auto" : dies === 1 ? "1" : "n"} options={[["1", "1"], ["n", "N"], ["auto", t("Sin límite")]]} onChange={(v) => setParam("dies", v === "1" ? 1 : v === "auto" ? 0 : Math.max(2, dies))} />
                {dies > 1 && <input type="number" min={2} max={1024} aria-label={t("Número de chips")} value={dies} onChange={(e) => setParam("dies", Math.max(2, Number(e.target.value) || 2))} />}
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
          <span className="row-label">{t("Modelo")}</span>
          <select aria-label="Modelo" value={model?.id} onChange={(e) => selectModel(e.target.value)}>
            {models.map((m) => (
              <option key={m.id} value={m.id}>
                {m.name}
              </option>
            ))}
          </select>
        </div>
      </div>
      <TargetRows />
      <button className="row link-row" onClick={() => setView("modelos")}>
        {t("Buscar otros modelos…")}
      </button>
      <label className="row link-row">
        {t("Cargar config.json de otro modelo…")}
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
      <input className="search" type="search" placeholder={t("Buscar parámetro")} aria-label={t("Buscar parámetro")} value={query} onChange={(e) => setQuery(e.target.value)} />
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
      {groups.length === 0 && <p className="empty">{t("Ningún parámetro coincide.")}</p>}
      <button className="plain" onClick={reset}>
        {t("Restablecer todo")}
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

  return (
    <>
      {MATURITY.map(([maturity, title, subtitle]) => {
        const items = catalog.filter((o) => o.maturity === maturity);
        return (
          <Disclosure key={maturity} title={title} defaultOpen={true} summary={`${items.filter((o) => active.includes(o.id)).length} de ${items.length}`}>
            <p className="group-note">{subtitle}</p>
            {items.map((o) => (
              <div className="opt" key={o.id} title={o.what}>
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
  const t = useT();
  const tab = useDesign((s) => s.sidebarTab);
  const setTab = useDesign((s) => s.setSidebarTab);
  const active = useDesign((s) => s.params.optimizations.length);

  return (
    <aside className="sidebar">
      <div className="sidebar-tabs">
        <Segmented
          label="Panel"
          value={tab}
          options={[
            ["parametros", t("Parámetros")],
            ["optimizaciones", active > 0 ? `${t("Optimizaciones")} (${active})` : t("Optimizaciones")],
          ]}
          onChange={setTab}
        />
      </div>
      <div className="sidebar-body">{tab === "parametros" ? <ParametersTab /> : <OptimizationsTab />}</div>
    </aside>
  );
}
