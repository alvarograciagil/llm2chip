import { fpgaBoards, type BlockArea, type Confidence, type DiePlan, type Metric, type Report } from "../engine";
import { formatNumber } from "../format";
import { Icon } from "./icons";
import { useT } from "../i18n";
import { LayoutPanel } from "../layout/LayoutPanel";
import { FeasibilityPanel } from "../feasibility/FeasibilityPanel";
import { useAdvice } from "../feasibility/useAdvice";
import { fieldLabel } from "../params";
import { useDesign, type Selection } from "../store";

const GLYPH: Record<Confidence, string> = { derived: "●", calibrated: "◐", assumed: "○" };
const CONFIDENCE: Record<Confidence, string> = { derived: "Derivada del modelo", calibrated: "Calibrada", assumed: "Supuesta, sin calibrar" };
export const CATEGORY_LABEL: Record<BlockArea["category"], string> = {
  memory: "Memoria de pesos",
  compute: "Cálculo",
  interconnect: "Interconexión",
  vector: "Unidades vectoriales",
  state: "Estado",
  system: "Control y sistema",
};

const HEADLINE = ["tok_s", "dies", "total_area_mm2", "power_w"];
const BOARD_HEADLINE = ["tok_s", "fpga_memory_pct", "fpga_logic_pct", "fpga_dsp_pct"];
const ARRAY_IDS = ["cell_weights", "cell_macs", "scale_units", "activation_noc"];


const MULTIPLIER_NAME = { array: "Array", wallace: "Wallace", booth: "Booth", multiples: "Múltiplos precalculados" };

function bytesLabel(bytes: number) {
  if (bytes >= 1e9) return `${formatNumber(bytes / 1e9)} GB`;
  if (bytes >= 1e6) return `${formatNumber(bytes / 1e6)} MB`;
  if (bytes >= 1e3) return `${formatNumber(bytes / 1e3)} KB`;
  return `${formatNumber(bytes)} B`;
}

function Delta({ current, base, unit }: { current: number; base: number | undefined; unit: string }) {
  if (base === undefined || base === 0) return null;
  const change = (current - base) / Math.abs(base);
  if (Math.abs(change) < 0.0005) return null;
  return (
    <small className="delta" aria-label={`Respecto a la referencia: ${formatNumber(base)} ${unit}`}>
      {change > 0 ? "+" : "−"}
      {formatNumber(Math.abs(change) * 100)} %
    </small>
  );
}

/** A value the estimator offers for `field`, as the person reads it. */
function describeValue(field: string, value: unknown): string {
  if (typeof value === "boolean") return value ? "activar" : "desactivar";
  if (typeof value === "number") return value === 0 ? "automático" : formatNumber(value);
  if (field === "fpga_board") return (fpgaBoards() as { id: string; name: string }[]).find((b) => b.id === value)?.name ?? String(value);
  if (value === "external_ddr") return field === "kv_backend" ? "caché en la memoria de la placa" : "pesos en la memoria de la placa";
  const named: Record<string, string> = { auto: "automático", custom: "a medida", sram: "SRAM", rom: "ROM" };
  return typeof value === "string" ? (named[value] ?? value) : String(value);
}

function Summary({ report }: { report: Report }) {
  const t = useT();
  const baseline = useDesign((s) => s.baseline);
  const reveal = useDesign((s) => s.reveal);
  const setParam = useDesign((s) => s.setParam);
  const setSelection = useDesign((s) => s.setSelection);
  const estimateMs = useDesign((s) => s.estimateMs);
  const base = (id: string) => baseline?.report.metrics.find((m) => m.id === id)?.value;

  // On a board, how much of it the design takes comes first.
  const sections = [...new Set(report.metrics.map((m) => m.section))].sort((a, b) => Number(b === "FPGA") - Number(a === "FPGA"));
  // On an FPGA board what matters is how much of it the design takes, not square millimetres.
  const onBoard = report.metrics.some((m) => m.id === "fpga_memory_pct");
  const headline = (onBoard ? BOARD_HEADLINE : HEADLINE).map((id) => report.metrics.find((m) => m.id === id)).filter((m): m is Metric => !!m);
  const blocks = [...report.blocks].sort((a, b) => b.per_die_mm2 - a.per_die_mm2);
  const largest = blocks[0]?.per_die_mm2 ?? 1;
  const model = useDesign((s) => s.model);
  const params = useDesign((s) => s.params);
  const { advice, checking, searching } = useAdvice(model, params, !onBoard);

  return (
    <>
      <div className="stats">
        {headline.map((m) => (
          <div className="stat" key={m.id} title={t(m.why)}>
            <span className="stat-value">
              {formatNumber(m.value)}
              {m.unit && <small> {m.unit}</small>}
            </span>
            <span className="stat-label">{t(m.label)}</span>
            <Delta current={m.value} base={base(m.id)} unit={m.unit} />
          </div>
        ))}
      </div>
      {baseline && <p className="note">{t("Comparando con")} «{baseline.name}».</p>}

      {!onBoard && <FeasibilityPanel advice={advice} checking={checking} searching={searching} />}

      {report.decisions.length > 0 && (
        <>
          <h3>{t("Circuito elegido para estos parámetros")}</h3>
          <dl className="inset metrics">
            <div><dt>{t("Multiplicadores por die")}</dt><dd>{formatNumber(report.macs_per_die)}</dd></div>
            {report.decisions.map((d) => (
              <div key={d.weight_bits} title={`${formatNumber(d.area_um2)} µm² y ${formatNumber(d.energy_pj)} pJ por multiplicación, hasta ${formatNumber(d.fmax_ghz)} GHz.`}>
                <dt>{t("Pesos de {n} bits").replace("{n}", String(d.weight_bits))} ({formatNumber(d.share * 100)} %)</dt>
                <dd>{t(MULTIPLIER_NAME[d.multiplier])}, {d.mac_stages + d.adder_stages} {t("etapas")}{d.skip_zeros ? `, ${t("salta ceros")}` : ""}</dd>
              </div>
            ))}
          </dl>
        </>
      )}

      {report.violations.length > 0 && (
        <div className="issues">
          {report.violations.map((v, i) => (
            <div key={i} className={`issue ${v.severity}`} role={v.severity === "error" ? "alert" : "status"}>
              <span className="issue-glyph" aria-hidden="true">
                <Icon name={v.severity === "error" ? "warning" : "info"} size={16} />
              </span>
              <div>
                <strong>{t(v.message)}</strong>
                <p>{t(v.fix)}</p>
                {v.field && (
                  <div className="issue-actions">
                    {v.value !== undefined && (
                      <button className="link" onClick={() => setParam(v.field!, v.value)}>
                        {t("Corregir")}: {t(describeValue(v.field!, v.value))}
                      </button>
                    )}
                    <button className="link" onClick={() => reveal(v.field!)}>
                      {t("Mostrar ajuste")}
                    </button>
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      )}

      <LayoutPanel report={report} allowed={checking ? null : (advice?.verdict.layout ?? null)} />

      {!onBoard && <h3>{t("Reparto del área por die")}</h3>}
      <div className="inset" hidden={onBoard}>
        {blocks.map((b) => (
          <button className="area-row" key={b.id} onClick={() => setSelection(ARRAY_IDS.includes(b.id) ? { kind: "array", die: 0 } : { kind: "block", die: 0, id: b.id })}>
            <i className={`swatch cat-${b.category}`} aria-hidden="true" />
            <span className="area-name">{t(b.label)}</span>
            <span className="area-value">{formatNumber(b.per_die_mm2)} mm²</span>
            <span className="area-track" aria-hidden="true">
              <span style={{ width: `${Math.max(0.6, (b.per_die_mm2 / largest) * 100)}%` }} />
            </span>
          </button>
        ))}
      </div>

      {sections.map((section) => (
        <div key={section}>
          <h3>{t(section)}</h3>
          <dl className="inset metrics">
            {report.metrics
              .filter((m) => m.section === section)
              .map((m) => (
                <div key={m.id} title={t(m.why)}>
                  <dt>
                    <i className="glyph" aria-label={t(CONFIDENCE[m.confidence])} title={t(CONFIDENCE[m.confidence])}>
                      {GLYPH[m.confidence]}
                    </i>
                    {t(m.label)}
                  </dt>
                  <dd>
                    {formatNumber(m.value)} {m.unit}
                    <Delta current={m.value} base={base(m.id)} unit={m.unit} />
                  </dd>
                </div>
              ))}
          </dl>
        </div>
      ))}

      <p className="legend-note">
        ● derivada del modelo · ◐ calibrada · ○ supuesta. Recalculado en {estimateMs.toFixed(1)} ms.
      </p>
    </>
  );
}

function Detail({ title, subtitle, children }: { title: string; subtitle?: string; children: React.ReactNode }) {
  const setSelection = useDesign((s) => s.setSelection);
  return (
    <>
      <button className="back" onClick={() => setSelection(null)}>
        ‹ Resumen
      </button>
      <h2 className="detail-title">{title}</h2>
      {subtitle && <p className="detail-sub">{subtitle}</p>}
      {children}
    </>
  );
}

function Facts({ facts }: { facts: [string, string][] }) {
  const t = useT();
  return (
    <dl className="inset metrics">
      {facts.map(([k, v]) => (
        <div key={k}>
          <dt>{t(k)}</dt>
          <dd>{v}</dd>
        </div>
      ))}
    </dl>
  );
}

function layersText(die: DiePlan, t: (text: string) => string) {
  const parts: string[] = [];
  if (die.has_embedding) parts.push("embedding");
  if (die.first_layer !== null) parts.push(die.first_layer === die.last_layer ? `${t("capa")} ${die.first_layer}` : `${t("capas")} ${die.first_layer}–${die.last_layer}`);
  if (die.has_head) parts.push(t("salida de vocabulario"));
  return parts.join(", ");
}

function BlockDetail({ block, report }: { block: BlockArea; report: Report }) {
  const t = useT();
  const reveal = useDesign((s) => s.reveal);
  return (
    <Detail title={t(block.label)} subtitle={t(CATEGORY_LABEL[block.category])}>
      <p className="prose">{t(block.what)}</p>
      <Facts
        facts={[
          ["Área por die", `${formatNumber(block.per_die_mm2)} mm²`],
          ["Área total", `${formatNumber(block.per_die_mm2 * report.dies)} mm²`],
          ["Parte del die", `${formatNumber((block.per_die_mm2 / report.die_area_mm2) * 100)} %`],
          ["Ocupado", `${formatNumber(block.activity * 100)} % ${t("del token")}`],
          ["Energía", `${formatNumber(block.energy_mj)} mJ/token`],
          ["Datos movidos", `${bytesLabel(block.bytes_token)}/token`],
        ]}
      />
      <h3>{t("Por qué tiene este tamaño")}</h3>
      <p className="prose">{t(block.why)}</p>
      {block.driven_by.length > 0 && (
        <>
          <h3>{t("Qué lo cambia")}</h3>
          <div className="chips">
            {block.driven_by.map((key) => (
              <button key={key} className="chip" onClick={() => reveal(key)}>
                {fieldLabel(key)}
              </button>
            ))}
          </div>
        </>
      )}
    </Detail>
  );
}

function SelectionDetail({ selection, report }: { selection: NonNullable<Selection>; report: Report }) {
  const t = useT();
  const setSelection = useDesign((s) => s.setSelection);
  const die = report.floorplan.dies[selection.die] ?? report.floorplan.dies[0];
  const arrayBlocks = report.blocks.filter((b) => ARRAY_IDS.includes(b.id));

  if (selection.kind === "block") {
    const block = report.blocks.find((b) => b.id === selection.id);
    return block ? <BlockDetail block={block} report={report} /> : null;
  }
  if (selection.kind === "die") {
    return (
      <Detail title={`Die ${die.index}`} subtitle={layersText(die, t)}>
        <p className="prose">{t("Cada die ejecuta sus capas y pasa un único vector de activación al siguiente.")}</p>
        <Facts
          facts={[
            ["Área", `${formatNumber(report.die_area_mm2)} mm²`],
            ["Pesos alojados", `${formatNumber(die.weights / 1e9)} G`],
            ["Celdas", formatNumber(die.cells.count)],
          ]}
        />
      </Detail>
    );
  }
  if (selection.kind === "array") {
    return (
      <Detail title={t("Matriz de celdas")} subtitle={`Die ${die.index} · ${formatNumber(die.cells.count)} ${t("celdas")}`}>
        <p className="prose">{t("Cada celda guarda un trozo de cada matriz junto a su multiplicador, así que los pesos nunca viajan: solo se mueven las activaciones.")}</p>
        <h3>{t("Partes")}</h3>
        <div className="inset">
          {arrayBlocks.map((b) => (
            <button className="area-row plain-row" key={b.id} onClick={() => setSelection({ kind: "block", die: die.index, id: b.id })}>
              <i className={`swatch cat-${b.category}`} aria-hidden="true" />
              <span className="area-name">{t(b.label)}</span>
              <span className="area-value">{formatNumber(b.per_die_mm2)} mm²</span>
            </button>
          ))}
        </div>
      </Detail>
    );
  }
  const cells = die.cells;
  return (
    <Detail title={`${t("Celda")} ${selection.col}, ${selection.row}`} subtitle={`Die ${die.index}`}>
      <p className="prose">{t("Todas las celdas son iguales: cambia qué trozo de cada matriz guarda cada una.")}</p>
      <Facts
        facts={[
          ["Memoria de pesos", `${formatNumber(cells.sram_fraction * 100)} %`],
          ["Multiplicador y acumulador", `${formatNumber(cells.mac_fraction * 100)} %`],
          ["Escala y sesgo", `${formatNumber(cells.scale_fraction * 100)} %`],
          ["Red de activaciones", `${formatNumber(cells.noc_fraction * 100)} %`],
        ]}
      />
      <h3>{t("Qué guarda")}</h3>
      <dl className="inset metrics scroll">
        {cells.segments.map((s) => (
          <div key={s.label}>
            <dt>{s.label}</dt>
            <dd>{formatNumber(s.fraction * 100)} %</dd>
          </div>
        ))}
      </dl>
    </Detail>
  );
}

export function Inspector() {
  const report = useDesign((s) => s.report);
  const error = useDesign((s) => s.error);
  const selection = useDesign((s) => s.selection);

  return (
    <aside className="inspector">
      {error && <div className="issue error" role="alert"><span className="issue-icon">!</span><div><strong>{error}</strong></div></div>}
      {!error && report && (selection ? <SelectionDetail selection={selection} report={report} /> : <Summary report={report} />)}
    </aside>
  );
}
