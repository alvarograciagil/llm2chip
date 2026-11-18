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

function Span({ label, value, onChange }: { label: string; value: [number, number]; onChange: (v: [number, number]) => void }) {
  return (
    <label className="optimize-field">
      <span>{label}</span>
      <input type="number" value={value[0]} onChange={(e) => onChange([Number(e.target.value), value[1]])} />
      <span aria-hidden="true">–</span>
      <input type="number" value={value[1]} onChange={(e) => onChange([value[0], Number(e.target.value)])} />
    </label>
  );
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
  const [chipsText, setChipsText] = useState(request.space.chips.join(", "));
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
    const chips = chipsText.split(/[\s,]+/).map(Number).filter((n) => Number.isInteger(n) && n > 0);
    void start(chosen, { ...request, space: { ...request.space, chips } }, factorSteps);
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
            <NumberField label={t("Most chips")} value={request.limits.max_chips} onChange={(v) => limits({ max_chips: v })} />
            <NumberField label={t("Most silicon")} unit="mm²" value={request.limits.max_area_mm2} onChange={(v) => limits({ max_area_mm2: v })} />
            <NumberField label={t("Most power")} unit="W" value={request.limits.max_power_w} onChange={(v) => limits({ max_power_w: v })} />
            <details className="optimize-advanced">
              <summary>{t("Advanced options")}</summary>
              <label className="optimize-field">
                <span>{t("Numbers of chips to try")}</span>
                <input type="text" value={chipsText} onChange={(e) => setChipsText(e.target.value)} />
              </label>
              <label className="optimize-field">
                <span>{t("Ways of sharing")}</span>
                <input type="text" value={request.space.sharing} onChange={(e) => space({ sharing: e.target.value })} />
              </label>
              <p className="muted-text">{t("layers, rows or mixed:N, separated by commas; mixed:auto tries every group size that divides the chips.")}</p>
              <Span label={t("Units per die (log2)")} value={request.space.cluster_bits} onChange={(v) => space({ cluster_bits: v })} />
              <Span label={t("Lanes per unit (log2)")} value={request.space.lane_bits} onChange={(v) => space({ lane_bits: v })} />
              <Span label={t("Cells per die (log2)")} value={request.space.cell_bits} onChange={(v) => space({ cell_bits: v })} />
              <NumberField label={t("Context")} unit="tokens" value={request.context} onChange={(v) => setRequest({ ...request, context: v || 8192 })} />
              <NumberField label={t("Prompt for the first token")} unit="tokens" value={request.prompt} onChange={(v) => setRequest({ ...request, prompt: v || 1024 })} />
              <NumberField label={t("Highest clock")} unit="GHz" value={request.max_clock_ghz} onChange={(v) => setRequest({ ...request, max_clock_ghz: v || 5 })} />
              <NumberField label={t("Cycles to apply a group's scale")} value={factorSteps} onChange={(v) => setFactorSteps(v || 8)} />
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
                      <th>{t("mJ per token")}</th>
                      <th>W</th>
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
