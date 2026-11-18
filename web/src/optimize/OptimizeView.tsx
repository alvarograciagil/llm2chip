import { useEffect, useMemo, useState } from "react";
import { Segmented } from "../components/controls";
import { Plot } from "../explorer/ExplorerView";
import { formatNumber, locale } from "../format";
import { useT } from "../i18n";
import { useSimulator } from "../simulator/simulatorStore";
import { useDesign } from "../store";
import { shapeOf, useOptimize, type Design, type Effort, type Goal, type Request } from "./optimizeStore";

const GOALS: [Goal, string][] = [
  ["speed", "Speed"],
  ["area", "Area"],
  ["efficiency", "Area per tok/s"],
  ["energy", "Energy per token"],
];

const GOAL_HELP: Record<Goal, string> = {
  speed: "The most tokens per second within the limits.",
  area: "The least silicon that meets the limits.",
  efficiency: "The least silicon for each token per second: the best value.",
  energy: "The least energy for each token, as the estimator puts it at the measured speed.",
};

const ALTERNATIVE: Record<string, string> = {
  fastest: "Fastest",
  smallest: "Smallest",
  efficient: "Least area per tok/s",
  frugal: "Least energy",
};

const LIMIT: Record<string, string> = {
  reticle: "the reticle: a larger die cannot be exposed",
  max_area: "the area limit",
  max_power: "the power limit",
  min_tok_s: "the minimum speed",
  first_token: "the first-token limit",
};

const PHASE: Record<string, string> = {
  pruning: "Ruling out what cannot hold the model",
  screening: "Measuring each way of sharing",
  climbing: "Refining the most promising",
  done: "Done",
};

/** Reads a number field; empty means no limit. */
function NumberField({ label, value, onChange, unit }: { label: string; value: number; onChange: (v: number) => void; unit?: string }) {
  const t = useT();
  return (
    <label className="optimize-field">
      <span>{label}</span>
      <input type="number" min={0} value={value || ""} placeholder={t("No limit")} onChange={(e) => onChange(Number(e.target.value) || 0)} />
      {unit && <span className="muted-text">{unit}</span>}
    </label>
  );
}

/** A range of powers of two, from `low` to `high` (log2), chosen from lists. */
function Span({ label, value, low, high, onChange }: { label: string; value: [number, number]; low: number; high: number; onChange: (v: [number, number]) => void }) {
  const t = useT();
  const bits = Array.from({ length: high - low + 1 }, (_, i) => low + i);
  return (
    <div className="optimize-field">
      <span>{label}</span>
      <select aria-label={`${label} (${t("from")})`} value={value[0]} onChange={(e) => onChange([Number(e.target.value), Math.max(Number(e.target.value), value[1])])}>
        {bits.map((b) => <option key={b} value={b}>{formatNumber(2 ** b)}</option>)}
      </select>
      <span aria-hidden="true">–</span>
      <select aria-label={`${label} (${t("to")})`} value={value[1]} onChange={(e) => onChange([Math.min(value[0], Number(e.target.value)), Number(e.target.value)])}>
        {bits.map((b) => <option key={b} value={b}>{formatNumber(2 ** b)}</option>)}
      </select>
    </div>
  );
}

/** One value from a list. */
function Choose({ label, value, options, onChange, unit }: { label: string; value: number; options: [number, string][]; onChange: (v: number) => void; unit?: string }) {
  return (
    <label className="optimize-field">
      <span>{label}</span>
      <select value={value} onChange={(e) => onChange(Number(e.target.value))}>
        {options.map(([v, name]) => <option key={v} value={v}>{name}</option>)}
      </select>
      {unit && <span className="muted-text">{unit}</span>}
    </label>
  );
}

/** Several values from a list, as boxes to tick. */
function Ticks<V extends string | number>({ label, values, options, onChange }: { label: string; values: V[]; options: [V, string][]; onChange: (v: V[]) => void }) {
  return (
    <fieldset className="optimize-ticks">
      <legend>{label}</legend>
      {options.map(([v, name]) => (
        <label key={String(v)} className="check">
          <input type="checkbox" checked={values.includes(v)} onChange={(e) => onChange(e.target.checked ? [...values, v] : values.filter((x) => x !== v))} />
          {name}
        </label>
      ))}
    </fieldset>
  );
}

const CHIP_CHOICES = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64];
const GROUP_CHOICES = [2, 3, 4, 6, 8, 12, 16];
const CONTEXTS = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072];
const PROMPTS = [32, 128, 512, 1024, 2048, 4096, 8192];
const CLOCKS = [0.5, 1, 1.5, 2, 3, 5];
const FACTOR_STEPS = [1, 2, 4, 8, 16];

/** The ways of sharing as boxes, and as the list the engine reads. */
function sharingsOf(text: string): string[] {
  return text.split(",").map((s) => s.trim()).filter(Boolean);
}

function shape(d: Design): string {
  return `${d.chips} × ${formatNumber(2 ** d.cluster_bits)}×${formatNumber(2 ** d.lane_bits)} · ${formatNumber(2 ** d.cell_bits)}`;
}

function DesignRow({ design, label, onApply }: { design: Design; label: string; onApply: () => void }) {
  const t = useT();
  return (
    <tr>
      <td>{label}</td>
      <td>{shape(design)}</td>
      <td>{design.sharing_name}</td>
      <td>{formatNumber(design.tokens_per_second)}</td>
      <td>{formatNumber(design.area_mm2)}</td>
      <td>{formatNumber(design.die_mm2)}</td>
      <td>{design.area_per_tok_s.toLocaleString(locale(), { maximumSignificantDigits: 3 })}</td>
      <td>{design.energy_mj === null ? "—" : formatNumber(design.energy_mj)}</td>
      <td>{design.power_w === null ? "—" : formatNumber(design.power_w)}</td>
      <td>{design.power_w ? formatNumber(design.tokens_per_second / design.power_w) : "—"}</td>
      <td>
        <button type="button" className="button" onClick={onApply}>{t("Apply")}</button>
      </td>
    </tr>
  );
}

/** The search for the best set of chips for a model, each design measured on its circuit. */
export function OptimizeView() {
  const t = useT();
  const { status, error, start, poll, cancel, draft: request, setDraft: setRequest } = useOptimize();
  const connect = useSimulator((s) => s.connect);
  const online = useSimulator((s) => s.online);
  const models = useSimulator((s) => s.models);
  const designModel = useDesign((s) => s.model);
  const addModel = useDesign((s) => s.addModel);
  const setParams = useDesign((s) => s.setParams);
  const setView = useDesign((s) => s.setView);
  const usable = models.filter((m) => m.supported);
  const [path, setPath] = useState("");
  const [factorSteps, setFactorSteps] = useState(8);
  const running = status.state === "running";

  useEffect(() => {
    void connect();
    void poll();
  }, [connect, poll]);
  useEffect(() => {
    if (!running) return;
    const timer = setInterval(() => void poll(), 1000);
    return () => clearInterval(timer);
  }, [running, poll]);
  // The designer's model first, when the engine has it.
  const chosen = path || usable.find((m) => m.path === designModel?.path)?.path || usable[0]?.path || "";
  const model = usable.find((m) => m.path === chosen);

  const limits = (patch: Partial<Request["limits"]>) => setRequest({ ...request, limits: { ...request.limits, ...patch } });
  const space = (patch: Partial<Request["space"]>) => setRequest({ ...request, space: { ...request.space, ...patch } });
  const run = () => {
    void start(chosen, request, factorSteps);
  };
  // A design applied to the designer, on its model.
  const apply = (design: Design) => {
    if (model && designModel?.path !== model.path) addModel(model.name, model.config, model.path);
    setParams({ ...shapeOf(design), context_tokens: request.context });
    setView("floorplan");
  };

  const report = status.state === "done" ? status.report : null;
  const progress = status.state === "running" ? status.progress : null;
  const points = useMemo(
    () =>
      (report?.frontier ?? []).map((d) => ({
        x: d.area_mm2,
        y: d.tokens_per_second,
        ok: true,
        label: `${shape(d)} · ${d.sharing_name} · ${formatNumber(d.tokens_per_second)} tok/s · ${formatNumber(d.area_mm2)} mm²`,
      })),
    [report],
  );

  return (
    <section className="stage explorer-stage">
      <div className="explorer">
        <div className="card">
          <div className="card-title">
            <h2>{t("Automatic optimisation")}</h2>
          </div>
          <p className="muted-text plot-sub">{t("Say what matters most and the limits it must meet: the engine measures designs on their generated circuits and recommends the best.")}</p>
          {online === false && <p className="placeholder">{t("The local engine is not running. Start it with scripts/dev.sh.")}</p>}
          <div className="optimize-form">
            <label className="optimize-field">
              <span>{t("Model")}</span>
              <select value={chosen} onChange={(e) => setPath(e.target.value)} disabled={running}>
                {usable.map((m) => <option key={m.path} value={m.path}>{m.name}</option>)}
              </select>
            </label>
            <Segmented label={t("What to optimise")} value={request.goal} options={GOALS.map(([g, name]) => [g, t(name)])} onChange={(goal) => setRequest({ ...request, goal })} />
            <p className="muted-text">{t(GOAL_HELP[request.goal])}</p>
            <NumberField label={t("Minimum speed")} unit="tok/s" value={request.limits.min_tok_s} onChange={(v) => limits({ min_tok_s: v })} />
            <NumberField label={t("Longest wait for the first token")} unit="ms" value={request.limits.max_first_token_ms} onChange={(v) => limits({ max_first_token_ms: v })} />
            <Choose label={t("Most chips")} value={request.limits.max_chips} options={[[0, t("No limit")], ...CHIP_CHOICES.map((n): [number, string] => [n, String(n)])]} onChange={(max_chips) => limits({ max_chips })} />
            <NumberField label={t("Most silicon")} unit="mm²" value={request.limits.max_area_mm2} onChange={(v) => limits({ max_area_mm2: v })} />
            <NumberField label={t("Most power")} unit="W" value={request.limits.max_power_w} onChange={(v) => limits({ max_power_w: v })} />
            <details className="optimize-advanced">
              <summary>{t("Advanced options")}</summary>
              <Ticks label={t("Numbers of chips to try")} values={request.space.chips} options={CHIP_CHOICES.map((n) => [n, String(n)])} onChange={(chips) => space({ chips: [...chips].sort((a, b) => a - b) })} />
              <Ticks
                label={t("Ways of sharing")}
                values={sharingsOf(request.space.sharing)}
                options={[
                  ["layers", t("By layers: each chip runs whole layers")],
                  ["rows", t("One group: every chip shares each layer")],
                  ["mixed:auto", t("Groups of every size that divides the chips")],
                  ...GROUP_CHOICES.map((g): [string, string] => [`mixed:${g}`, t(`Groups of ${g}`)]),
                ]}
                onChange={(list) => space({ sharing: list.join(",") })}
              />
              <Span label={t("Units per die")} low={0} high={11} value={request.space.cluster_bits} onChange={(v) => space({ cluster_bits: v })} />
              <Span label={t("Lanes per unit")} low={0} high={8} value={request.space.lane_bits} onChange={(v) => space({ lane_bits: v })} />
              <Span label={t("Cells per die")} low={3} high={26} value={request.space.cell_bits} onChange={(v) => space({ cell_bits: v })} />
              <Choose label={t("Context")} unit="tokens" value={request.context} options={CONTEXTS.map((c) => [c, formatNumber(c)])} onChange={(context) => setRequest({ ...request, context })} />
              <Choose label={t("Prompt for the first token")} unit="tokens" value={request.prompt} options={PROMPTS.map((p) => [p, formatNumber(p)])} onChange={(prompt) => setRequest({ ...request, prompt })} />
              <Choose label={t("Highest clock")} unit="GHz" value={request.max_clock_ghz} options={CLOCKS.map((c) => [c, formatNumber(c)])} onChange={(max_clock_ghz) => setRequest({ ...request, max_clock_ghz })} />
              <Choose label={t("Cycles to apply a group's scale")} value={factorSteps} options={FACTOR_STEPS.map((s) => [s, String(s)])} onChange={setFactorSteps} />
              <Segmented<Effort> label={t("Search")} value={request.effort} options={[["quick", t("Quick")], ["thorough", t("Thorough")]]} onChange={(effort) => setRequest({ ...request, effort })} />
            </details>
            <div className="optimize-actions">
              {running ? (
                <button type="button" className="button" onClick={() => void cancel()}>{t("Stop")}</button>
              ) : (
                <button type="button" className="button primary" disabled={!chosen} onClick={run}>{t("Search")}</button>
              )}
            </div>
          </div>
          {error && <div className="issue error" role="alert">{error}</div>}
        </div>

        {progress && (
          <div className="card" role="status">
            <h2>{t(PHASE[progress.phase] ?? progress.phase)}</h2>
            <progress max={Math.max(progress.planned, 1)} value={progress.done} />
            <p className="muted-text">{t(`Measured ${progress.done} designs of about ${progress.planned}`)}</p>
            {progress.best && <p>{t("Best so far")}: {shape(progress.best)} · {progress.best.sharing_name} · {formatNumber(progress.best.tokens_per_second)} tok/s · {formatNumber(progress.best.area_mm2)} mm²</p>}
          </div>
        )}

        {status.state === "error" && <div className="issue error" role="alert">{status.error}</div>}

        {report && (
          <div className="card">
            <div className="card-title">
              <h2>{t("Recommendation")}</h2>
            </div>
            {report.recommended ? (
              <>
                <p>
                  {t(GOAL_HELP[report.goal])} {report.limiting.length > 0 && <>{t("Held back by")} {report.limiting.map((l) => t(LIMIT[l] ?? l)).join(", ")}.</>}
                </p>
                <table className="front">
                  <thead>
                    <tr>
                      <th></th>
                      <th>{t("Chips × units × lanes · cells")}</th>
                      <th>{t("Sharing")}</th>
                      <th>tok/s</th>
                      <th>{t("Total mm²")}</th>
                      <th>{t("mm² per die")}</th>
                      <th>{t("mm² per tok/s")}</th>
                      <th>{t("Energy per token (mJ)")}</th>
                      <th>W</th>
                      <th>{t("tok/s per W")}</th>
                      <th></th>
                    </tr>
                  </thead>
                  <tbody>
                    <DesignRow design={report.recommended} label={t("Recommended")} onApply={() => apply(report.recommended!)} />
                    {report.alternatives.map((a) => <DesignRow key={a.label} design={a.design} label={t(ALTERNATIVE[a.label] ?? a.label)} onApply={() => apply(a.design)} />)}
                  </tbody>
                </table>
                <p className="muted-text">{t("Speed, clock and silicon are measured on each design's own circuit; energy and power are estimated by the architecture model at that speed.")}</p>
              </>
            ) : (
              <p>{t("No design in the space meets every limit.")} {report.failure && <>{t("One could not be built")}: {report.failure}</>}</p>
            )}
            {report.too_few_chips.length > 0 && <p className="muted-text">{t("Too few chips to hold the model")}: {report.too_few_chips.join(", ")}</p>}
            <p className="muted-text">{t(`Measured ${report.measured} designs; ${report.from_cache} came from earlier searches`)}</p>
            {points.length > 1 && <Plot points={points} xLabel={t("Total area (mm²)")} yLabel={t("Speed")} xLog yLog line onPick={(p) => {
              const picked = report.frontier.find((d) => d.area_mm2 === p.x && d.tokens_per_second === p.y);
              if (picked) apply(picked);
            }} />}
          </div>
        )}
      </div>
    </section>
  );
}

/** How the search works, beside it. */
export function OptimizePanel() {
  const t = useT();
  return (
    <aside className="inspector">
      <h2 className="detail-title">{t("Automatic optimisation")}</h2>
      <p className="detail-sub">{t("How the search finds a design")}</p>
      <div className="inset padded">
        <p>{t("The estimator first rules out numbers of chips whose dies cannot hold the model.")}</p>
        <p>{t("Each way of sharing left gets the number of cells that suits the goal, measured on its circuit; the most promising then try more and fewer units, and every width of lane in a thorough search.")}</p>
        <p>{t("Every measurement is kept, so asking again with another goal or other limits only measures what is new.")}</p>
        <p>{t("Apply takes the chips, the sharing, the units, the lanes, the cells and the clock to the designer, where the rest can still be changed by hand.")}</p>
      </div>
    </aside>
  );
}
