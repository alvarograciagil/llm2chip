import { useMemo, useState } from "react";
import { estimateDesign, sweepDesign, type ChipParams, type SweepResult } from "../engine";
import { formatNumber } from "../format";
import { matchingPreset, WEIGHT_PRESETS } from "../params";
import { useDesign } from "../store";
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
  { key: "macs_per_die", label: "Multiplicadores por die", unit: "", min: 8000, max: 2000000, log: true, integer: true },
  { key: "context_tokens", label: "Contexto", unit: "tokens", min: 512, max: 1048576, log: true, integer: true },
  { key: "max_params_g", label: "Parámetros máximos", unit: "G", min: 8, max: 200, log: true, integer: false },
  { key: "weight_bits", label: "Bits de peso", unit: "", min: 3, max: 8, log: false, integer: true },
  { key: "freq_ghz", label: "Frecuencia", unit: "GHz", min: 0.25, max: 3, log: true, integer: false },
  { key: "vector_lanes_per_die", label: "Carriles vectoriales", unit: "", min: 1024, max: 512000, log: true, integer: true },
];

const METRICS: [string, string, string][] = [
  ["tok_s", "Velocidad", "tok/s"],
  ["total_area_mm2", "Área total", "mm²"],
  ["dies", "Dies", ""],
  ["power_w", "Potencia", "W"],
  ["energy_token_mj", "Energía por token", "mJ"],
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
    <svg className="plot" viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`${yLabel} frente a ${xLabel}`} onMouseLeave={() => setHover(null)}>
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

export function ExplorerView() {
  const model = useDesign((s) => s.model);
  const params = useDesign((s) => s.params);
  const setParam = useDesign((s) => s.setParam);
  const [xKey, setXKey] = useState("macs_per_die");
  const [yKey, setYKey] = useState("tok_s");
  const axis = AXES.find((a) => a.key === xKey)!;
  const metric = METRICS.find((m) => m[0] === yKey)!;

  const curve = useMemo(() => {
    if (!model) return [];
    const values = samples(axis);
    const results = sweepDesign(model, params, values.map((v) => ({ [axis.key]: v })));
    return values.map((v, i): Point => ({
      x: v,
      y: results[i].metrics[yKey],
      ok: results[i].errors === 0,
      label: `${axis.label}: ${formatNumber(v)} · ${metric[1]}: ${formatNumber(results[i].metrics[yKey])} ${metric[2]}`,
    }));
  }, [model, params, axis, yKey, metric]);

  const pareto = useMemo(() => {
    if (!model) return { all: [] as (Point & { macs: number; bits: number })[], front: [] as (Point & { macs: number; bits: number })[] };
    const macs = [34000, 68000, 136000, 272000, 544000, 1088000];
    const bits = [3, 4, 6, 8];
    const variants = macs.flatMap((m) => bits.map((b) => ({ macs_per_die: m, weight_bits: b, use_model_mix: false })));
    const results: SweepResult[] = sweepDesign(model, params, variants);
    const all = results
      .map((r, i) => ({
        x: r.metrics.total_area_mm2,
        y: r.metrics.tok_s,
        ok: r.errors === 0,
        macs: variants[i].macs_per_die,
        bits: variants[i].weight_bits,
        label: `${formatNumber(variants[i].macs_per_die)} mult. · ${variants[i].weight_bits} bits · ${formatNumber(r.metrics.tok_s)} tok/s`,
      }))
      .filter((p) => p.ok);
    const front = all.filter((p) => !all.some((o) => o !== p && o.x <= p.x && o.y >= p.y && (o.x < p.x || o.y > p.y))).sort((a, b) => a.x - b.x);
    return { all, front };
  }, [model, params]);

  return (
    <section className="stage explorer-stage">
      <div className="explorer">
        <div className="card">
          <div className="card-title">
            <h2>Barrido de un parámetro</h2>
            <div className="plot-controls">
              <select aria-label="Parámetro" value={xKey} onChange={(e) => setXKey(e.target.value)}>
                {AXES.map((a) => <option key={a.key} value={a.key}>{a.label}</option>)}
              </select>
              <span aria-hidden="true">→</span>
              <select aria-label="Medida" value={yKey} onChange={(e) => setYKey(e.target.value)}>
                {METRICS.map(([id, name]) => <option key={id} value={id}>{name}</option>)}
              </select>
            </div>
          </div>
          <p className="muted-text plot-sub">{metric[1]}{metric[2] ? ` (${metric[2]})` : ""}. Clic en un punto para aplicarlo al diseño.</p>
          {curve.length > 1 && (
            <Plot points={curve} xLabel={axis.label} yLabel={metric[1]} xLog={axis.log} yLog={yKey !== "dies"} line onPick={(p) => setParam(axis.key, p.x)} />
          )}
        </div>
        <div className="card">
          <div className="card-title">
            <h2>Frontera de Pareto</h2>
          </div>
          <p className="muted-text plot-sub">Velocidad (tok/s) frente a área total (mm²), variando multiplicadores y bits de peso. Solo diseños que cumplen las restricciones.</p>
          {pareto.all.length > 1 && (
            <Plot points={pareto.all} xLabel="Área total (mm²)" yLabel="Velocidad" xLog yLog line={false} onPick={(p) => {
              const chosen = pareto.all.find((o) => o.x === p.x && o.y === p.y);
              if (chosen) {
                setParam("macs_per_die", chosen.macs);
                setParam("weight_bits", chosen.bits);
                setParam("use_model_mix", false);
              }
            }} />
          )}
          <table className="front">
            <thead><tr><th>Multiplicadores</th><th>Bits</th><th>Área</th><th>Velocidad</th></tr></thead>
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
    source: "Llama 3.1 8B en ROM, 6 nm",
    model: "llama-3.1-8b",
    params: { max_params_g: 8.1, weight_backend: "rom", use_model_mix: false, group_size: 0, weight_bits: 4, context_tokens: 1024, vector_lanes_per_die: 65536, node: "n6" },
    checks: [
      ["die_area_mm2", "Área", 815, 815, "mm²"],
      ["tok_s", "Velocidad", 14400, 17000, "tok/s"],
      ["power_w", "Potencia", 200, 250, "W"],
    ],
  },
  {
    name: "d-Matrix Raptor",
    source: "DRAM apilada, más de 100 TB/s, 4 nm",
    model: "llama-3.1-8b",
    params: { max_params_g: 8.1, weight_backend: "stacked_dram", stacked_dram_tbps: 100, use_model_mix: false, group_size: 0, weight_bits: 4, context_tokens: 2048, node: "n5", macs_per_die: 1000000 },
    checks: [["weight_bandwidth", "Ancho de banda usado", 0, 100, "TB/s"]],
  },
];

function FormatComparison() {
  const model = useDesign((s) => s.model);
  const params = useDesign((s) => s.params);
  const setParam = useDesign((s) => s.setParam);
  const rows = useMemo(() => {
    if (!model) return [];
    const presets = WEIGHT_PRESETS.flatMap((section) => section.presets.map((preset) => ({ ...preset, family: section.title })));
    const results: SweepResult[] = sweepDesign(model, params, presets.map((p) => p.values));
    return presets.map((preset, i) => ({ preset, metrics: results[i].metrics, errors: results[i].errors })).sort((a, b) => a.metrics.total_area_mm2 - b.metrics.total_area_mm2);
  }, [model, params]);
  const current = matchingPreset(params as unknown as Record<string, unknown>);
  return (
    <div className="card">
      <div className="card-title">
        <h2>Formatos de pesos</h2>
      </div>
      <p className="muted-text plot-sub">El circuito de cada formato con este diseño, de menor a mayor área. Clic en una fila para usarlo. Top-1 es la coincidencia con coma flotante en 16 o 64 tokens de Qwen3.8 real. Ese modelo ya viene cuantizado a 4 bits, así que los formatos pequeños pueden salir mejor de lo que saldrían con pesos originales.</p>
      <div className="format-table">
        <table className="front">
          <thead>
            <tr><th>Formato</th><th>Área</th><th>Velocidad</th><th>Potencia</th><th>Energía</th><th>Top-1</th></tr>
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
                  <td>{preset.label}{errors > 0 ? " ⚠" : ""}</td>
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
  const models = useDesign((s) => s.models);
  const defaults = useDesign((s) => s.defaults);
  const rows = useMemo(
    () =>
      REFERENCES.map((ref) => {
        const model = models.find((m) => m.id === ref.model);
        if (!model) return null;
        const report = estimateDesign(model, { ...defaults, ...ref.params } as ChipParams);
        return { ref, report };
      }),
    [models, defaults],
  );

  return (
    <aside className="inspector">
      <h2 className="detail-title">Calibración</h2>
      <p className="detail-sub">El modelo frente a chips publicados.</p>
      {rows.map((row) => {
        if (!row) return null;
        return (
          <div key={row.ref.name}>
            <h3>{row.ref.name}</h3>
            <p className="note">{row.ref.source}</p>
            <dl className="inset metrics">
              {row.ref.checks.map(([id, label, low, high, unit]) => {
                const value = id === "die_area_mm2" ? row.report.die_area_mm2 : row.report.metrics.find((m) => m.id === id)?.value ?? 0;
                const inRange = value >= low * 0.8 && value <= high * 1.2;
                return (
                  <div key={id}>
                    <dt>{label}</dt>
                    <dd>
                      {inRange ? "✓ " : "✕ "}
                      {formatNumber(value)} {unit}
                      <small className="delta">real {low === high ? formatNumber(low) : `${formatNumber(low)}–${formatNumber(high)}`}</small>
                    </dd>
                  </div>
                );
              })}
            </dl>
          </div>
        );
      })}
      <h3>Qué está calibrado</h3>
      <div className="inset padded">
        <p className="muted-text">
          El área de lógica de cada celda sale de las puertas reales de su netlist. La densidad de ROM y la energía se ajustan al HC1, el único chip de este tipo con cifras públicas de área, velocidad y consumo. Las memorias SRAM, HBM y DRAM apilada siguen siendo supuestos y conviene leerlas como orden de magnitud.
        </p>
      </div>
    </aside>
  );
}
