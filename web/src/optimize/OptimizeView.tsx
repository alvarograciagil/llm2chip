import { useEffect, useId, useMemo, useState, type ReactNode } from "react";
import { InfoTip, Switch } from "../components/controls";
import { Icon, type IconName } from "../components/icons";
import { Plot } from "../explorer/ExplorerView";
import { formatNumber, locale } from "../format";
import { useT } from "../i18n";
import { useSimulator } from "../simulator/simulatorStore";
import { matchingPreset } from "../params";
import { useDesign } from "../store";
import { shapeOf, useOptimize, type Design, type Goal, type Request } from "./optimizeStore";

type T = (text: string) => string;

const GOALS: { id: Goal; icon: IconName; title: string; text: string; help: string }[] = [
  { id: "speed", icon: "speed", title: "Fastest", text: "The most tokens per second.", help: "Grows each die to the most cells the limits allow, and keeps the design that generates fastest for one user at the full context." },
  { id: "area", icon: "silicon", title: "Smallest", text: "The least silicon that meets the requirements.", help: "Looks for the fewest chips and cells that still reach the minimum speed and the other requirements. Set a minimum speed, or the smallest design that runs at all wins." },
  { id: "efficiency", icon: "balance", title: "Best value", text: "The least silicon for each token per second.", help: "Divides the silicon by the speed: the design where each token per second costs least. Usually the sensible choice when cost matters as much as speed." },
  { id: "energy", icon: "energy", title: "Least energy", text: "The fewest joules for each token.", help: "The energy each token takes, from the architecture model at the speed measured on the circuit: the multiplications, memory reads and links, plus the leakage while the token is made." },
];

const ALTERNATIVES: Record<string, string> = {
  fastest: "Fastest",
  smallest: "Smallest",
  efficient: "Best value",
  frugal: "Least energy",
};

const LIMITS: Record<string, string> = {
  reticle: "the size of the reticle: a larger die cannot be made in one exposure",
  max_area: "the silicon allowed",
  max_power: "the power allowed",
  min_tok_s: "the minimum speed",
  first_token: "the wait for the first token",
};

const STEPS: [string, string][] = [
  ["pruning", "Rule out"],
  ["screening", "Measure each way of sharing"],
  ["climbing", "Refine the best"],
];

const CHIPS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64];
const GROUPS = [2, 3, 4, 6, 8, 12, 16];
const CONTEXTS = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072];
const PROMPTS = [32, 128, 512, 1024, 2048, 4096, 8192];
const CLOCKS = [0.5, 1, 1.5, 2, 3, 5];
const FACTOR_STEPS = [1, 2, 4, 8, 16];

/** How the chips share the model, in words. */
function sharingWords(name: string, t: T): string {
  if (name === "layers") return t("by layers");
  if (name === "rows") return t("all in one group");
  return t(`in groups of ${name.split(":")[1]}`);
}

/** The whole design in a line: chips and how they share the model. */
function headline(d: Design, t: T): string {
  return `${d.chips} ${t(d.chips === 1 ? "chip" : "chips")} ${sharingWords(d.sharing_name, t)}`;
}

/** Each die in a line. */
function dieLine(d: Design, t: T): string {
  return `${t("dies of")} ${formatNumber(d.die_mm2)} mm² · ${formatNumber(2 ** d.cluster_bits)} ${t("units")} × ${formatNumber(2 ** d.lane_bits)} ${t("lanes")} · ${formatNumber(2 ** d.cell_bits)} ${t("cells")} · ${formatNumber(d.clock_hz / 1e6)} MHz`;
}

/** About how long is left, from how fast the designs measured so far went. */
function remaining(startedAt: number | null, done: number, planned: number, t: T): string {
  if (!startedAt || done < 3 || planned <= done) return "";
  const minutes = Math.ceil(((Date.now() - startedAt) / done) * (planned - done) / 60000);
  return ` · ${minutes <= 1 ? t("about a minute left") : t(`about ${minutes} minutes left`)}`;
}

const significant = (v: number) => v.toLocaleString(locale(), { maximumSignificantDigits: 3 });

function percent(ratio: number): string {
  const p = Math.round((ratio - 1) * 100);
  return `${p > 0 ? "+" : ""}${p} %`;
}

/** A label with its "?" beside it. */
function Labelled({ label, help }: { label: string; help: string }) {
  const id = useId();
  return (
    <span className="opt-label">
      {label}
      <InfoTip id={id} label={label} text={help} />
    </span>
  );
}

/** One requirement: off means no limit. */
function Requirement({ label, help, on, onToggle, children }: { label: string; help: string; on: boolean; onToggle: (on: boolean) => void; children: ReactNode }) {
  const t = useT();
  return (
    <div className={on ? "opt-row" : "opt-row off"}>
      <Labelled label={label} help={help} />
      <div className="opt-control">
        <Switch checked={on} label={label} onChange={onToggle} />
        {on ? children : <span className="muted-text">{t("No limit")}</span>}
      </div>
    </div>
  );
}

function Amount({ value, unit, step, label, onChange }: { value: number; unit: string; step?: number; label: string; onChange: (v: number) => void }) {
  return (
    <span className="opt-amount">
      <input type="number" aria-label={label} min={0} step={step} value={value} onChange={(e) => onChange(Math.max(0, Number(e.target.value)))} />
      <span className="muted-text">{unit}</span>
    </span>
  );
}

function Pick({ value, options, label, onChange }: { value: number; options: [number, string][]; label: string; onChange: (v: number) => void }) {
  return (
    <select aria-label={label} value={value} onChange={(e) => onChange(Number(e.target.value))}>
      {options.map(([v, name]) => <option key={v} value={v}>{name}</option>)}
    </select>
  );
}

/** Buttons that each turn one value on or off. */
function Pills<V extends string | number>({ values, options, onChange }: { values: V[]; options: [V, string][]; onChange: (v: V[]) => void }) {
  return (
    <div className="opt-pills">
      {options.map(([v, name]) => {
        const on = values.includes(v);
        return (
          <button type="button" key={String(v)} className={on ? "opt-pill on" : "opt-pill"} aria-pressed={on} onClick={() => onChange(on ? values.filter((x) => x !== v) : [...values, v])}>
            {name}
          </button>
        );
      })}
    </div>
  );
}

/** A range of powers of two (log2), chosen from two lists. */
function Range({ value, low, high, label, onChange }: { value: [number, number]; low: number; high: number; label: string; onChange: (v: [number, number]) => void }) {
  const t = useT();
  const options = Array.from({ length: high - low + 1 }, (_, i): [number, string] => [low + i, formatNumber(2 ** (low + i))]);
  return (
    <span className="opt-range">
      <Pick label={`${label}: ${t("from")}`} value={value[0]} options={options} onChange={(v) => onChange([v, Math.max(v, value[1])])} />
      <span className="muted-text">{t("to")}</span>
      <Pick label={`${label}: ${t("to")}`} value={value[1]} options={options} onChange={(v) => onChange([Math.min(value[0], v), v])} />
    </span>
  );
}

function Section({ number, title, children }: { number: number; title: string; children: ReactNode }) {
  return (
    <section className="opt-section">
      <h3>
        <span className="opt-step">{number}</span>
        {title}
      </h3>
      {children}
    </section>
  );
}

function Stat({ label, value, unit, help, estimated }: { label: string; value: string; unit?: string; help: string; estimated?: boolean }) {
  const t = useT();
  return (
    <div className="opt-stat">
      <Labelled label={label} help={help} />
      <strong>
        {value}
        {unit && <small> {unit}</small>}
      </strong>
      {estimated && <span className="opt-tag">{t("estimated")}</span>}
    </div>
  );
}

function Alternative({ label, design, best, onApply }: { label: string; design: Design; best: Design; onApply: () => void }) {
  const t = useT();
  return (
    <div className="opt-alt">
      <span className="opt-alt-label">{label}</span>
      <strong>{headline(design, t)}</strong>
      <span className="muted-text">{dieLine(design, t)}</span>
      <span className="opt-deltas">
        <span>
          {formatNumber(design.tokens_per_second)} tok/s <em>{percent(design.tokens_per_second / best.tokens_per_second)}</em>
        </span>
        <span>
          {formatNumber(design.area_mm2)} mm² <em>{percent(design.area_mm2 / best.area_mm2)}</em>
        </span>
      </span>
      <button type="button" onClick={onApply}>{t("Apply")}</button>
    </div>
  );
}

/** The search for the best set of chips for a model, each design measured on its circuit. */
export function OptimizeView() {
  const t = useT();
  const { status, error, start, poll, cancel, startedAt, draft: request, setDraft: setRequest } = useOptimize();
  const connect = useSimulator((s) => s.connect);
  const online = useSimulator((s) => s.online);
  const models = useSimulator((s) => s.models);
  const designModel = useDesign((s) => s.model);
  const designParams = useDesign((s) => s.params);
  const weightFormat = matchingPreset(designParams as unknown as Record<string, unknown>) ?? "custom";
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
  const sharings = request.space.sharing.split(",").map((s) => s.trim()).filter(Boolean);
  const apply = (design: Design) => {
    if (model && designModel?.path !== model.path) addModel(model.name, model.config, model.path);
    setParams({ ...shapeOf(design), context_tokens: request.context });
    setView("floorplan");
  };

  const report = status.state === "done" ? status.report : null;
  const progress = status.state === "running" ? status.progress : null;
  const best = report?.recommended ?? null;
  const points = useMemo(
    () =>
      (report?.frontier ?? []).map((d) => ({
        x: d.area_mm2,
        y: d.tokens_per_second,
        ok: true,
        label: `${headline(d, t)} · ${formatNumber(d.tokens_per_second)} tok/s · ${formatNumber(d.area_mm2)} mm²`,
      })),
    [report, t],
  );
  const range = (bits: [number, number]) => `${formatNumber(2 ** bits[0])}–${formatNumber(2 ** bits[1])}`;
  const summary = [
    `${request.space.chips.length} ${t("chip counts")}`,
    `${sharings.length} ${t("ways of sharing")}`,
    `${range(request.space.cluster_bits)} ${t("units")}`,
    `${range(request.space.lane_bits)} ${t("lanes")}`,
    `${range(request.space.cell_bits)} ${t("cells")}`,
  ].join(" · ");

  return (
    <section className="stage opt-stage">
      <div className="opt">
        <header className="opt-header">
          <h2>{t("Automatic optimisation")}</h2>
          <p>{t("Tell it what matters and what the chip must meet. It builds candidate designs, measures each one on its own generated circuit and recommends the best.")}</p>
        </header>
        {online === false && <div className="issue error" role="alert">{t("The local engine is not running. Start it with scripts/dev.sh.")}</div>}

        <Section number={1} title={t("Model")}>
          <div className="opt-row">
            <Labelled label={t("Model to put on chips")} help={t("A model on this machine, with its weights: the circuits are generated from them. Add folders or download models in the Models view.")} />
            <div className="opt-control">
              <select aria-label={t("Model to put on chips")} value={chosen} disabled={running} onChange={(e) => setPath(e.target.value)}>
                {usable.map((m) => <option key={m.path} value={m.path}>{m.size_gb ? `${m.name} · ${formatNumber(m.size_gb)} GB` : m.name}</option>)}
              </select>
            </div>
          </div>
          <p className="muted-text">{t("With its weights in the design's format")}: {t(weightFormat)}. {t("It is chosen in the sidebar, under Arithmetic.")}</p>
        </Section>

        <Section number={2} title={t("What matters most")}>
          <div className="opt-goals" role="radiogroup" aria-label={t("What matters most")}>
            {GOALS.map((g) => (
              <button type="button" key={g.id} role="radio" aria-checked={request.goal === g.id} className={request.goal === g.id ? "opt-goal on" : "opt-goal"} onClick={() => setRequest({ ...request, goal: g.id })}>
                <Icon name={g.icon} size={22} />
                <strong>{t(g.title)}</strong>
                <span>{t(g.text)}</span>
              </button>
            ))}
          </div>
          <p className="opt-note">{t(GOALS.find((g) => g.id === request.goal)!.help)}</p>
        </Section>

        <Section number={3} title={t("Requirements")}>
          <div className="opt-row">
            <Labelled label={t("Context")} help={t("The longest conversation the chip must hold, in tokens. Every design is measured at its last token, where attention is slowest, so the speed shown is guaranteed for the whole context.")} />
            <div className="opt-control">
              <Pick label={t("Context")} value={request.context} options={CONTEXTS.map((c) => [c, `${formatNumber(c)} ${t("tokens")}`])} onChange={(context) => setRequest({ ...request, context })} />
            </div>
          </div>
          <Requirement label={t("Minimum speed")} help={t("Tokens per second for one user at the full context. Designs slower than this are left out.")} on={request.limits.min_tok_s > 0} onToggle={(on) => limits({ min_tok_s: on ? 1000 : 0 })}>
            <Amount label={t("Minimum speed")} value={request.limits.min_tok_s} unit="tok/s" step={100} onChange={(min_tok_s) => limits({ min_tok_s: min_tok_s || 1 })} />
          </Requirement>
          <Requirement label={t("First token within")} help={t("The wait from sending a prompt to the first word of the answer. The chips read the prompt in a pipeline, so more chips usually answer sooner.")} on={request.limits.max_first_token_ms > 0} onToggle={(on) => limits({ max_first_token_ms: on ? 500 : 0 })}>
            <Amount label={t("First token within")} value={request.limits.max_first_token_ms} unit="ms" step={50} onChange={(v) => limits({ max_first_token_ms: v || 1 })} />
            <Pick label={t("Prompt length")} value={request.prompt} options={PROMPTS.map((p) => [p, `${t("for a prompt of")} ${formatNumber(p)} ${t("tokens")}`])} onChange={(prompt) => setRequest({ ...request, prompt })} />
          </Requirement>
          <Requirement label={t("At most chips")} help={t("How many dies the system may have. More chips hold more of the model and can share each layer, but cost packages, boards and links.")} on={request.limits.max_chips > 0} onToggle={(on) => limits({ max_chips: on ? 16 : 0 })}>
            <Pick label={t("At most chips")} value={request.limits.max_chips} options={CHIPS.map((n) => [n, String(n)])} onChange={(max_chips) => limits({ max_chips })} />
          </Requirement>
          <Requirement label={t("At most silicon")} help={t("The silicon of all the chips together, in mm². Every die must also fit the reticle, about 858 mm² in N6, whatever this says.")} on={request.limits.max_area_mm2 > 0} onToggle={(on) => limits({ max_area_mm2: on ? 10000 : 0 })}>
            <Amount label={t("At most silicon")} value={request.limits.max_area_mm2} unit="mm²" step={500} onChange={(v) => limits({ max_area_mm2: v || 1 })} />
          </Requirement>
          <Requirement label={t("At most power")} help={t("Power while generating, estimated by the architecture model at the measured speed.")} on={request.limits.max_power_w > 0} onToggle={(on) => limits({ max_power_w: on ? 300 : 0 })}>
            <Amount label={t("At most power")} value={request.limits.max_power_w} unit="W" step={10} onChange={(v) => limits({ max_power_w: v || 1 })} />
          </Requirement>
        </Section>

        <Section number={4} title={t("Search")}>
          <div className="opt-goals two" role="radiogroup" aria-label={t("Search")}>
            {(
              [
                ["quick", "Quick", "The three most promising ways of sharing, with the widest lanes. Minutes on a large model."],
                ["thorough", "Thorough", "The eight most promising, and every lane width. Several times longer."],
              ] as const
            ).map(([id, title, text]) => (
              <button type="button" key={id} role="radio" aria-checked={request.effort === id} className={request.effort === id ? "opt-goal on" : "opt-goal"} onClick={() => setRequest({ ...request, effort: id })}>
                <strong>{t(title)}</strong>
                <span>{t(text)}</span>
              </button>
            ))}
          </div>
          <details className="opt-space">
            <summary>
              {t("Which designs are tried")} <span className="muted-text">{summary}</span>
            </summary>
            <div className="opt-row tall">
              <Labelled label={t("Numbers of chips")} help={t("Each number of chips is tried with every way of sharing that suits it. Those whose dies cannot hold the model are ruled out first.")} />
              <div className="opt-control">
                <Pills values={request.space.chips} options={CHIPS.map((n) => [n, String(n)])} onChange={(chips) => space({ chips: [...chips].sort((a, b) => a - b) })} />
              </div>
            </div>
            <div className="opt-row tall">
              <Labelled label={t("How the chips share the model")} help={t("By layers, each chip runs a run of whole layers and only one works at a time. In groups, the chips of a group share every layer, each with a slice of every matrix and head, and the groups run in a chain. One group shares every layer among all the chips. Automatic tries every group size that divides the chips.")} />
              <div className="opt-control">
                <Pills
                  values={sharings}
                  options={[["layers", t("By layers")], ["rows", t("One group")], ["mixed:auto", t("Groups: automatic")], ...GROUPS.map((g): [string, string] => [`mixed:${g}`, t(`Groups of ${g}`)])]}
                  onChange={(list) => space({ sharing: list.join(",") })}
                />
              </div>
            </div>
            <div className="opt-row">
              <Labelled label={t("Vector units per die")} help={t("Units that run the vector work side by side: norms, attention, the delta rule. More units go faster but take logic and slow the clock a little.")} />
              <div className="opt-control">
                <Range label={t("Vector units per die")} value={request.space.cluster_bits} low={0} high={11} onChange={(cluster_bits) => space({ cluster_bits })} />
              </div>
            </div>
            <div className="opt-row">
              <Labelled label={t("Lanes per unit")} help={t("Elements each unit handles at once. Wide lanes suit long vectors; the search tries every width only when it is thorough.")} />
              <div className="opt-control">
                <Range label={t("Lanes per unit")} value={request.space.lane_bits} low={0} high={8} onChange={(lane_bits) => space({ lane_bits })} />
              </div>
            </div>
            <div className="opt-row">
              <Labelled label={t("Compute cells per die")} help={t("Each cell is a multiplier with its share of the weights. More cells multiply faster, but each adds logic: the die grows until the reticle stops it.")} />
              <div className="opt-control">
                <Range label={t("Compute cells per die")} value={request.space.cell_bits} low={3} high={26} onChange={(cell_bits) => space({ cell_bits })} />
              </div>
            </div>
            <div className="opt-row">
              <Labelled label={t("Highest clock")} help={t("Each die runs at its own fastest clock, from its critical path, or at this if it is lower.")} />
              <div className="opt-control">
                <Pick label={t("Highest clock")} value={request.max_clock_ghz} options={CLOCKS.map((c) => [c, `${formatNumber(c)} GHz`])} onChange={(max_clock_ghz) => setRequest({ ...request, max_clock_ghz })} />
              </div>
            </div>
            <div className="opt-row">
              <Labelled label={t("Cycles to scale a group")} help={t("Cycles each cell takes to apply the scale and bias of a group of weights, while it adds up the next group. Fewer cost more logic. A model whose quantisation groups are smaller than this needs fewer.")} />
              <div className="opt-control">
                <Pick label={t("Cycles to scale a group")} value={factorSteps} options={FACTOR_STEPS.map((s) => [s, String(s)])} onChange={setFactorSteps} />
              </div>
            </div>
          </details>
        </Section>

        <div className="opt-actions">
          {running ? (
            <button type="button" onClick={() => void cancel()}>
              <Icon name="stop" /> {t("Stop")}
            </button>
          ) : (
            <button type="button" className="primary" disabled={!chosen || request.space.chips.length === 0 || sharings.length === 0} onClick={() => void start(chosen, request, factorSteps)}>
              <Icon name="optimize" /> {t("Find the best design")}
            </button>
          )}
        </div>
        {error && <div className="issue error" role="alert">{error}</div>}

        {progress && (
          <div className="opt-card" role="status">
            <ol className="opt-phases">
              {STEPS.map(([id, name], i) => {
                const at = STEPS.findIndex(([s]) => s === progress.phase);
                return <li key={id} className={i < at ? "done" : i === at ? "now" : ""}>{t(name)}</li>;
              })}
            </ol>
            <progress max={Math.max(progress.planned, 1)} value={progress.done} />
            <p className="muted-text">
              {t(`Measured ${progress.done} designs of about ${progress.planned}`)}
              {remaining(startedAt, progress.done, progress.planned, t)}
            </p>
            {progress.best && (
              <p>
                <span className="muted-text">{t("Best so far")}</span> <strong>{headline(progress.best, t)}</strong> · {formatNumber(progress.best.tokens_per_second)} tok/s · {formatNumber(progress.best.area_mm2)} mm²
              </p>
            )}
          </div>
        )}

        {status.state === "error" && <div className="issue error" role="alert">{status.error}</div>}

        {report && !best && (
          <div className="opt-card">
            <h3>{t("No design meets every requirement")}</h3>
            <p>{t("Loosen a requirement, or try more chips or larger dies.")}</p>
            {report.failure && <p className="muted-text">{t("One design could not be built")}: {report.failure}</p>}
          </div>
        )}

        {report && best && (
          <>
            <div className="opt-card opt-hero">
              <span className="opt-alt-label">{t("Recommended")}</span>
              <h3>{headline(best, t)}</h3>
              <p className="muted-text">{dieLine(best, t)}</p>
              <div className="opt-stats">
                <Stat label={t("Speed")} value={formatNumber(best.tokens_per_second)} unit="tok/s" help={t("Tokens per second for one user at the last token of the context, from the cycles of each chip's own microprogram and its clock.")} />
                <Stat label={t("Silicon")} value={formatNumber(best.area_mm2)} unit="mm²" help={t("All the dies together: logic, the cells' memories, the vector memories and the microcode.")} />
                <Stat label={t("Silicon per tok/s")} value={significant(best.area_per_tok_s)} unit="mm²" help={t("What each token per second costs in silicon: the lower, the better value.")} />
                <Stat label={t("First token")} value={formatNumber(best.prompt_seconds * 1e3)} unit="ms" help={t("The wait for the first token after the prompt length chosen in the requirements.")} />
                {best.energy_mj !== null && <Stat label={t("Energy per token")} value={significant(best.energy_mj)} unit="mJ" estimated help={t("Estimated by the architecture model at the measured speed. Its inverse is tokens per joule.")} />}
                {best.power_w !== null && <Stat label={t("Power")} value={formatNumber(best.power_w)} unit="W" estimated help={t("While generating, estimated by the architecture model at the measured speed.")} />}
              </div>
              {report.limiting.length > 0 && (
                <p className="opt-note">
                  {t("It cannot go further because of")} {report.limiting.map((l) => t(LIMITS[l] ?? l)).join(", ")}.
                </p>
              )}
              <div className="opt-actions">
                <button type="button" className="primary" onClick={() => apply(best)}>{t("Apply to the design")}</button>
              </div>
            </div>

            {report.alternatives.length > 0 && (
              <>
                <h3 className="opt-subtitle">{t("Other good options")}</h3>
                <div className="opt-alts">
                  {report.alternatives.map((a) => <Alternative key={a.label} label={t(ALTERNATIVES[a.label] ?? a.label)} design={a.design} best={best} onApply={() => apply(a.design)} />)}
                </div>
              </>
            )}

            {points.length > 1 && (
              <div className="opt-card">
                <h3>
                  <Labelled label={t("Speed against silicon")} help={t("Each point is a measured design that no other beats on both speed and silicon. Click one to apply it.")} />
                </h3>
                <Plot points={points} xLabel={t("Total area (mm²)")} yLabel={t("Speed")} xLog yLog line onPick={(p) => {
                  const picked = report.frontier.find((d) => d.area_mm2 === p.x && d.tokens_per_second === p.y);
                  if (picked) apply(picked);
                }} />
                <details className="opt-space">
                  <summary>{t("All the numbers")}</summary>
                  <table className="front">
                    <thead>
                      <tr>
                        <th>{t("Design")}</th>
                        <th>tok/s</th>
                        <th>mm²</th>
                        <th>{t("mm² per tok/s")}</th>
                        <th>{t("mJ per token")}</th>
                        <th>W</th>
                        <th></th>
                      </tr>
                    </thead>
                    <tbody>
                      {report.frontier.map((d, i) => (
                        <tr key={i}>
                          <td>
                            {headline(d, t)}
                            <br />
                            <span className="muted-text">{dieLine(d, t)}</span>
                          </td>
                          <td>{formatNumber(d.tokens_per_second)}</td>
                          <td>{formatNumber(d.area_mm2)}</td>
                          <td>{significant(d.area_per_tok_s)}</td>
                          <td>{d.energy_mj === null ? "—" : significant(d.energy_mj)}</td>
                          <td>{d.power_w === null ? "—" : formatNumber(d.power_w)}</td>
                          <td>
                            <button type="button" onClick={() => apply(d)}>{t("Apply")}</button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </details>
              </div>
            )}
            <p className="opt-footnote">
              {t("Speed, clock and silicon are measured on each design's own circuit; energy and power are estimated by the architecture model at that speed.")} {t(`Measured ${report.measured} designs; ${report.from_cache} came from earlier searches`)}
              {report.too_few_chips.length > 0 && <> {t("Too few chips to hold the model")}: {report.too_few_chips.join(", ")}.</>}
            </p>
          </>
        )}
      </div>
    </section>
  );
}

/** How the search works, and the words it uses. */
export function OptimizePanel() {
  const t = useT();
  return (
    <aside className="inspector">
      <h2 className="detail-title">{t("How it searches")}</h2>
      <p className="detail-sub">{t("Every number it recommends was measured on a generated circuit.")}</p>
      <ol className="opt-how">
        <li>{t("The estimator rules out numbers of chips whose dies cannot hold the model.")}</li>
        <li>{t("Each way of sharing left gets the number of cells that suits the goal, measured on its circuit.")}</li>
        <li>{t("The most promising try more and fewer vector units, and every lane width in a thorough search.")}</li>
        <li>{t("Every measurement is kept, so asking again with another goal or other requirements only measures what is new.")}</li>
      </ol>
      <h3 className="opt-panel-title">{t("Words")}</h3>
      <dl className="opt-glossary">
        <dt>{t("Sharing by layers")}</dt>
        <dd>{t("Each chip runs whole layers; a single user's token goes through them one after another.")}</dd>
        <dt>{t("Groups")}</dt>
        <dd>{t("The chips of a group share every layer at once, each with a slice of every matrix and head; the groups run in a chain.")}</dd>
        <dt>{t("Compute cell")}</dt>
        <dd>{t("A multiplier next to its share of the weights. Cells hold the model and do the matrix products.")}</dd>
        <dt>{t("Vector unit and lanes")}</dt>
        <dd>{t("What does everything else: norms, attention, the delta rule. Lanes are the elements a unit handles at once.")}</dd>
        <dt>{t("Reticle")}</dt>
        <dd>{t("The largest die a lithography machine exposes at once, about 858 mm²: no die may be larger.")}</dd>
      </dl>
    </aside>
  );
}
