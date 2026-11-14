import { useMemo, useState } from "react";
import { estimateDesign, type ChipParams } from "../engine";
import { useSweep, type Sweep } from "./useSweep";
import { formatNumber } from "../format";
import { matchingPreset, WEIGHT_PRESETS } from "../params";
import { useDesign } from "../store";
import { useT } from "../i18n";
import { FORMAT_QUALITY } from "./formatQuality";

interface Axis {
  key: string;
  label: string;
  unit: string;
  min: number;
  max: number;
  log: boolean;
  integer: boolean;
}

const AXES: Axis[] = [
  { key: "macs_per_die", label: "Multipliers per die", unit: "", min: 8000, max: 2000000, log: true, integer: true },
  { key: "context_tokens", label: "Context", unit: "tokens", min: 512, max: 1048576, log: true, integer: true },
  { key: "max_params_g", label: "Max parameters", unit: "G", min: 8, max: 200, log: true, integer: false },
  { key: "weight_bits", label: "Weight bits", unit: "", min: 3, max: 8, log: false, integer: true },
  { key: "freq_ghz", label: "Frequency", unit: "GHz", min: 0.25, max: 3, log: true, integer: false },
  { key: "vector_lanes_per_die", label: "Vector lanes", unit: "", min: 1024, max: 512000, log: true, integer: true },
];

const METRICS: [string, string, string][] = [
  ["tok_s", "Speed", "tok/s"],
  ["total_area_mm2", "Total area", "mm²"],
  ["dies", "Dies", ""],
  ["power_w", "Power", "W"],
  ["energy_token_mj", "Energy per token", "mJ"],
];

const STEPS = 28;

function samples(axis: Axis): number[] {
  return Array.from({ length: STEPS }, (_, i) => {
    const t = i / (STEPS - 1);
    const v = axis.log ? axis.min * Math.pow(axis.max / axis.min, t) : axis.min + (axis.max - axis.min) * t;
    return axis.integer ? Math.round(v) : v;
  }).filter((v, i, all) => all.indexOf(v) === i);
}

const H = 300;
const W = 760;
const PAD = { l: 66, r: 16, t: 14, b: 40 };

function ticks(min: number, max: number, log: boolean): number[] {
  if (log && min > 0) {
    const first = Math.ceil(Math.log10(min));
    const last = Math.floor(Math.log10(max));
    const list: number[] = [];
    for (let p = first; p <= last; p++) list.push(10 ** p);
    if (list.length >= 2) return list;
  }
  return Array.from({ length: 5 }, (_, i) => min + ((max - min) * i) / 4);
}

interface Point {
  x: number;
  y: number;
  ok: boolean;
  label: string;
}

function Plot({ points, xLabel, yLabel, xLog, yLog, line, onPick }: { points: Point[]; xLabel: string; yLabel: string; xLog: boolean; yLog: boolean; line: boolean; onPick?: (p: Point) => void }) {
  const tr = useT();
  const [hover, setHover] = useState<Point | null>(null);
  const xs = points.map((p) => p.x);
  const ys = points.map((p) => p.y);
  const [xMin, xMax] = [Math.min(...xs), Math.max(...xs)];
  const [yMin, yMax] = [yLog ? Math.min(...ys) : Math.min(0, ...ys), Math.max(...ys)];
  const scale = (v: number, a: number, b: number, log: boolean) => (log && a > 0 ? Math.log(v / a) / Math.log(b / a || 1) : (v - a) / (b - a || 1));
  const px = (v: number) => PAD.l + scale(v, xMin, xMax, xLog) * (W - PAD.l - PAD.r);
  const py = (v: number) => H - PAD.b - scale(v, yMin, yMax, yLog) * (H - PAD.t - PAD.b);
  const yTicks = ticks(yMin, yMax, yLog);
  const xTicks = ticks(xMin, xMax, xLog);
  const path = points.map((p, i) => `${i ? "L" : "M"}${px(p.x).toFixed(1)} ${py(p.y).toFixed(1)}`).join(" ");

  return (
    <svg className="plot" viewBox={`0 0 ${W} ${H}`} role="img" aria-label={tr(`${yLabel} against ${xLabel}`)} onMouseLeave={() => setHover(null)}>
      {yTicks.map((t) => (
        <g key={t}>
          <line className="plot-grid" x1={PAD.l} x2={W - PAD.r} y1={py(t)} y2={py(t)} />
          <text className="plot-tick" x={PAD.l - 8} y={py(t) + 4} textAnchor="end">{formatNumber(t)}</text>
        </g>
      ))}
      {xTicks.map((t) => (
        <text className="plot-tick" key={t} x={px(t)} y={H - PAD.b + 16} textAnchor="middle">{formatNumber(t)}</text>
      ))}
      <text className="plot-axis" x={(PAD.l + W - PAD.r) / 2} y={H - 6} textAnchor="middle">{xLabel}</text>
      {line && <path className="plot-line" d={path} />}
      {points.map((p, i) => (
        <g key={i} onMouseEnter={() => setHover(p)} onClick={() => onPick?.(p)}>
          <circle cx={px(p.x)} cy={py(p.y)} r={12} fill="transparent" />
          <circle className={p.ok ? "plot-dot" : "plot-dot bad"} cx={px(p.x)} cy={py(p.y)} r={line ? 3 : 4.5} />
        </g>
      ))}
      {hover && (
        <text className="plot-hover" x={Math.min(px(hover.x) + 10, W - 230)} y={Math.max(py(hover.y) - 10, 16)}>
          {hover.label}
        </text>
      )}
    </svg>
  );
}

/** Sweeping the multipliers means no longer deriving them from a speed target. */
const MANUAL_MACS = { target_tok_s: 0 };

const PARETO_VARIANTS = [34000, 68000, 136000, 272000, 544000, 1088000].flatMap((macs) => [3, 4, 6, 8].map((bits) => ({ macs_per_die: macs, weight_bits: bits, use_model_mix: false, target_tok_s: 0 })));

const FORMAT_PRESETS = WEIGHT_PRESETS.flatMap((section) => section.presets.map((preset) => ({ ...preset, family: section.title })));
const FORMAT_VARIANTS = FORMAT_PRESETS.map((preset) => preset.values);

function SweepProgress<V extends Record<string, unknown>>({ sweep }: { sweep: Sweep<V> }) {
  const t = useT();
  if (sweep.total === 0 || sweep.done >= sweep.total) return null;
  return <p className="note" role="status">{t(`Generating circuits: ${sweep.done} of ${sweep.total}`)}</p>;
}

export function ExplorerView() {
  const t = useT();
  const model = useDesign((s) => s.model);
  const params = useDesign((s) => s.params);
  const setParam = useDesign((s) => s.setParam);
  const [xKey, setXKey] = useState("macs_per_die");
  const [yKey, setYKey] = useState("tok_s");
  const axis = AXES.find((a) => a.key === xKey)!;
  const metric = METRICS.find((m) => m[0] === yKey)!;

  const curveVariants = useMemo(() => samples(axis).map((v): Record<string, number> => ({ [axis.key]: v, ...(axis.key === "macs_per_die" ? MANUAL_MACS : {}) })), [axis]);
  const curveSweep = useSweep("curve", model, params, curveVariants);
  const curve = useMemo(
    () =>
      (curveSweep.results ?? []).map((r, i): Point => {
        const v = curveSweep.variants[i][axis.key];
        return {
          x: v,
          y: r.metrics[yKey],
          ok: r.errors === 0,
          label: `${t(axis.label)}: ${formatNumber(v)} · ${t(metric[1])}: ${formatNumber(r.metrics[yKey])} ${metric[2]}`,
        };
      }),
    [curveSweep, axis, yKey, metric, t],
  );

  const paretoSweep = useSweep("pareto", model, params, PARETO_VARIANTS);
  const pareto = useMemo(() => {
    const all = (paretoSweep.results ?? [])
      .map((r, i) => ({
        x: r.metrics.total_area_mm2,
        y: r.metrics.tok_s,
        ok: r.errors === 0,
        macs: paretoSweep.variants[i].macs_per_die,
        bits: paretoSweep.variants[i].weight_bits,
        label: t(`${formatNumber(paretoSweep.variants[i].macs_per_die)} mult. · ${paretoSweep.variants[i].weight_bits} bits · ${formatNumber(r.metrics.tok_s)} tok/s`),
      }))
      .filter((p) => p.ok);
    const front = all.filter((p) => !all.some((o) => o !== p && o.x <= p.x && o.y >= p.y && (o.x < p.x || o.y > p.y))).sort((a, b) => a.x - b.x);
    return { all, front };
  }, [paretoSweep, t]);

  return (
    <section className="stage explorer-stage">
      <div className="explorer">
        <div className="card">
          <div className="card-title">
            <h2>{t("Sweep of one parameter")}</h2>
            <div className="plot-controls">
              <select aria-label={t("Parameter")} value={xKey} onChange={(e) => setXKey(e.target.value)}>
                {AXES.map((a) => <option key={a.key} value={a.key}>{t(a.label)}</option>)}
              </select>
              <span aria-hidden="true">→</span>
              <select aria-label={t("Measure")} value={yKey} onChange={(e) => setYKey(e.target.value)}>
                {METRICS.map(([id, name]) => <option key={id} value={id}>{t(name)}</option>)}
              </select>
            </div>
          </div>
          <p className="muted-text plot-sub">{t(metric[1])}{metric[2] ? ` (${metric[2]})` : ""}. {t("Click a point to apply it to the design.")}</p>
          <SweepProgress sweep={curveSweep} />
          {curve.length > 1 && (
            <Plot points={curve} xLabel={t(axis.label)} yLabel={t(metric[1])} xLog={axis.log} yLog={yKey !== "dies"} line onPick={(p) => {
              if (axis.key === "macs_per_die") setParam("target_tok_s", 0);
              setParam(axis.key, p.x);
            }} />
          )}
        </div>
        <div className="card">
          <div className="card-title">
            <h2>{t("Pareto frontier")}</h2>
          </div>
          <p className="muted-text plot-sub">{t("Speed (tok/s) against total area (mm²), varying multipliers and weight bits. Only designs that meet the constraints.")}</p>
          <SweepProgress sweep={paretoSweep} />
          {pareto.all.length > 1 && (
            <Plot points={pareto.all} xLabel={t("Total area (mm²)")} yLabel={t("Speed")} xLog yLog line={false} onPick={(p) => {
              const chosen = pareto.all.find((o) => o.x === p.x && o.y === p.y);
              if (chosen) {
                setParam("target_tok_s", 0);
                setParam("macs_per_die", chosen.macs);
                setParam("weight_bits", chosen.bits);
                setParam("use_model_mix", false);
              }
            }} />
          )}
          <table className="front">
            <thead><tr><th>{t("Multipliers")}</th><th>Bits</th><th>{t("Area")}</th><th>{t("Speed")}</th></tr></thead>
            <tbody>
              {pareto.front.map((p) => (
                <tr key={`${p.macs}-${p.bits}`}>
                  <td>{formatNumber(p.macs)}</td>
                  <td>{p.bits}</td>
                  <td>{formatNumber(p.x)} mm²</td>
                  <td>{formatNumber(p.y)} tok/s</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <FormatComparison />
      </div>
    </section>
  );
}

interface Reference {
  name: string;
  source: string;
  model: string;
  params: Record<string, unknown>;
  checks: [string, string, number, number, string][];
}

const REFERENCES: Reference[] = [
  {
    name: "Taalas HC1",
    source: "Llama 3.1 8B in ROM, 6 nm",
    model: "llama-3.1-8b",
    params: { max_params_g: 8.1, weight_backend: "rom", use_model_mix: false, group_size: 0, weight_bits: 4, context_tokens: 1024, vector_lanes_per_die: 65536, node: "n6" },
    checks: [
      ["die_area_mm2", "Area", 815, 815, "mm²"],
      ["tok_s", "Speed", 14400, 17000, "tok/s"],
      ["power_w", "Power", 200, 250, "W"],
    ],
  },
  {
    name: "d-Matrix Raptor",
    source: "Stacked DRAM, more than 100 TB/s, 4 nm",
    model: "llama-3.1-8b",
    params: { max_params_g: 8.1, weight_backend: "stacked_dram", stacked_dram_tbps: 100, use_model_mix: false, group_size: 0, weight_bits: 4, context_tokens: 2048, node: "n5", macs_per_die: 1000000 },
    checks: [["weight_bandwidth", "Bandwidth used", 0, 100, "TB/s"]],
  },
];

function FormatComparison() {
  const t = useT();
  const model = useDesign((s) => s.model);
  const params = useDesign((s) => s.params);
  const setParam = useDesign((s) => s.setParam);
  const sweep = useSweep("formats", model, params, FORMAT_VARIANTS);
  const rows = useMemo(
    () => (sweep.results ?? []).map((r, i) => ({ preset: FORMAT_PRESETS[i], metrics: r.metrics, errors: r.errors })).sort((a, b) => a.metrics.total_area_mm2 - b.metrics.total_area_mm2),
    [sweep],
  );
  const current = matchingPreset(params as unknown as Record<string, unknown>);
  return (
    <div className="card">
      <div className="card-title">
        <h2>{t("Weight formats")}</h2>
      </div>
      <p className="muted-text plot-sub">{t("Each format's circuit with this design, from smallest to largest area. Click a row to use it. Top-1 is the agreement with floating point over 16 or 64 tokens of the real Qwen3.8. That model already comes quantized to 4 bits, so small formats may come out better than they would with the original weights.")}</p>
      <SweepProgress sweep={sweep} />
      <div className="format-table">
        <table className="front">
          <thead>
            <tr><th>{t("Format")}</th><th>{t("Area")}</th><th>{t("Speed")}</th><th>{t("Power")}</th><th>{t("Energy")}</th><th>Top-1</th></tr>
          </thead>
          <tbody>
            {rows.map(({ preset, metrics, errors }) => {
              const quality = FORMAT_QUALITY[preset.label];
              return (
                <tr
                  key={preset.label}
                  className={preset.label === current ? "selected-row" : ""}
                  tabIndex={0}
                  onClick={() => Object.entries(preset.values).forEach(([key, value]) => setParam(key, value))}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") Object.entries(preset.values).forEach(([key, value]) => setParam(key, value));
                  }}
                >
                  <td>{t(preset.label)}{errors > 0 ? " ⚠" : ""}</td>
                  <td>{formatNumber(metrics.total_area_mm2)} mm²</td>
                  <td>{formatNumber(metrics.tok_s)} tok/s</td>
                  <td>{formatNumber(metrics.power_w)} W</td>
                  <td>{formatNumber(metrics.energy_token_mj)} mJ</td>
                  <td>{quality ? `${formatNumber(quality.top1)} %` : "—"}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export function CalibrationPanel() {
  const t = useT();
  const models = useDesign((s) => s.models);
  const defaults = useDesign((s) => s.defaults);
  const rows = useMemo(
    () =>
      REFERENCES.map((ref) => {
        const model = models.find((m) => m.id === ref.model);
        if (!model) return null;
        const report = estimateDesign(model, { ...defaults, target_tok_s: 0, ...ref.params } as ChipParams);
        return { ref, report };
      }),
    [models, defaults],
  );

  return (
    <aside className="inspector">
      <h2 className="detail-title">{t("Calibration")}</h2>
      <p className="detail-sub">{t("The model against published chips.")}</p>
      {rows.map((row) => {
        if (!row) return null;
        return (
          <div key={row.ref.name}>
            <h3>{row.ref.name}</h3>
            <p className="note">{t(row.ref.source)}</p>
            <dl className="inset metrics">
              {row.ref.checks.map(([id, label, low, high, unit]) => {
                const value = id === "die_area_mm2" ? row.report.die_area_mm2 : row.report.metrics.find((m) => m.id === id)?.value ?? 0;
                const inRange = value >= low * 0.8 && value <= high * 1.2;
                return (
                  <div key={id}>
                    <dt>{t(label)}</dt>
                    <dd>
                      {inRange ? "✓ " : "✕ "}
                      {formatNumber(value)} {unit}
                      <small className="delta">{t("real")} {low === high ? formatNumber(low) : `${formatNumber(low)}–${formatNumber(high)}`}</small>
                    </dd>
                  </div>
                );
              })}
            </dl>
          </div>
        );
      })}
      <h3>{t("What is calibrated")}</h3>
      <div className="inset padded">
        <p className="muted-text">
          {t("The logic area of each cell comes from the real gates of its netlist. ROM density and energy are fitted to the HC1, the only chip of this kind with public figures for area, speed and power. The SRAM, HBM and stacked DRAM memories are still assumptions, best read as orders of magnitude.")}
        </p>
      </div>
    </aside>
  );
}
