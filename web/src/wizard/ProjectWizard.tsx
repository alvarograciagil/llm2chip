import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { createPortal } from "react-dom";
import { estimateDesign, fpgaBoards, type ChipParams, type ModelSource, type Report } from "../engine";
import { formatNumber } from "../format";
import { useDesign } from "../store";
import { Segmented } from "../components/controls";
import { useT } from "../i18n";
import { useOptimize } from "../optimize/optimizeStore";
import { useVoice } from "../voice/voiceStore";
import { buildable, chipParams, DEFAULT_ANSWERS, fpgaParams, makers, metric, modelParams, suggested, type Answers, type FpgaBoard, type Start } from "./plan";

const STARTS: { id: Start; title: string; text: string }[] = [
  { id: "fpga", title: "An FPGA", text: "Pick a board and see which models fit, where their weights go and how fast they would run." },
  { id: "chip", title: "A single chip", text: "Everything on one die: choose the process, the area and the speed, and see which models fit." },
  { id: "model", title: "A model", text: "Choose the LLM; the number of chips and how they share it follow from the speed you ask for." },
  { id: "custom", title: "Custom", text: "The default parameters with the model you choose, to tune everything yourself." },
  { id: "voice", title: "A voice assistant", text: "A chip that listens, answers and speaks: which stages share a die, how much silicon, and how soon the answer sounds." },
];

type Step = "start" | "board" | "questions" | "model" | "name" | "voice";

const STEPS: Record<Start, Step[]> = {
  fpga: ["start", "board", "questions", "model", "name"],
  chip: ["start", "questions", "model", "name"],
  model: ["start", "model", "questions", "name"],
  custom: ["start", "model", "name"],
  voice: ["start", "voice"],
};

const STEP_TITLES: Record<Step, string> = {
  start: "Where do you start from?",
  board: "Which board do you have or want to buy?",
  questions: "A few questions",
  model: "Which model?",
  name: "Summary",
  voice: "A chip that listens, answers and speaks",
};

interface Row {
  model: ModelSource;
  id: string;
  params_g: number;
  tok_s: number;
  fits: boolean;
  note: string;
  params: Partial<ChipParams>;
}

function estimate(model: ModelSource, defaults: ChipParams, params: Partial<ChipParams>): Report | null {
  try {
    return estimateDesign(model, { ...defaults, ...params } as ChipParams);
  } catch {
    return null;
  }
}

/** The design for exactly this model: no more parameters than it has. */
function sized(model: ModelSource, defaults: ChipParams, params: Partial<ChipParams>): { params: Partial<ChipParams>; report: Report | null } {
  const first = estimate(model, defaults, params);
  const size = metric(first, "model_params_g");
  if (!Number.isFinite(size)) return { params, report: first };
  const exact = { ...params, max_params_g: size };
  return { params: exact, report: estimate(model, defaults, exact) };
}

/** Each model, judged for the start chosen: whether it fits and how fast it would go. */
function judge(answers: Answers, models: ModelSource[], defaults: ChipParams, boards: FpgaBoard[]): Row[] {
  const board = boards.find((b) => b.id === answers.board) ?? boards[0];
  return models.map((model) => {
    if (answers.start === "fpga") {
      // Inside the FPGA if it fits there, else with the weights in the board's memory: the
      // engine says which, with the same checks the design will show afterwards.
      const inside = sized(model, defaults, fpgaParams(answers, board, false));
      const placed = buildable(inside.report) ? inside : sized(model, defaults, fpgaParams(answers, board, true));
      const { params, report } = placed;
      const fits = buildable(report);
      const problem = report?.violations.find((v) => v.severity === "error")?.message;
      const where = placed === inside ? `In the FPGA's memory: ${Math.round(metric(report, "fpga_memory_pct"))} % used, logic at ${Math.round(metric(report, "fpga_logic_pct"))} %.` : "Weights in the board's memory: its bandwidth sets the speed.";
      const note = !report ? "The estimator does not take this model." : fits ? where : (problem ?? "It does not fit on this board.");
      return { model, id: model.id, params_g: metric(report, "model_params_g"), tok_s: metric(report, "tok_s"), fits, note, params };
    }
    const { params, report } = sized(model, defaults, answers.start === "chip" ? chipParams(answers) : answers.start === "model" ? modelParams(answers) : {});
    const fits = buildable(report) && (answers.start !== "chip" || report?.dies === 1);
    const problem = report?.violations.find((v) => v.severity === "error")?.message;
    const note = !report ? "The estimator does not take this model." : fits ? `${formatNumber(report.die_area_mm2)} mm² per die${report.dies > 1 ? `, ${report.dies} dies` : ""}.` : (problem ?? "It does not fit in one die.");
    return { model, id: model.id, params_g: metric(report, "model_params_g"), tok_s: metric(report, "tok_s"), fits, note, params };
  });
}

/** A parameter count given in billions, in the unit that reads best. */
function formatParams(billions: number): string {
  if (billions >= 1) return `${formatNumber(billions)} G`;
  if (billions >= 1e-3) return `${formatNumber(billions * 1e3)} M`;
  return `${formatNumber(billions * 1e6)} K`;
}

function Choice({ selected, title, children, onClick, disabled }: { selected: boolean; title: string; children?: ReactNode; onClick: () => void; disabled?: boolean }) {
  return (
    <button className={selected ? "wizard-choice on" : "wizard-choice"} aria-pressed={selected} disabled={disabled} onClick={onClick}>
      <strong>{title}</strong>
      {children}
    </button>
  );
}

function Question({ label, help, children }: { label: string; help: string; children: ReactNode }) {
  const t = useT();
  return (
    <div className="wizard-question">
      <div>
        <strong>{t(label)}</strong>
        <p>{t(help)}</p>
      </div>
      {children}
    </div>
  );
}

const CONTEXTS: [number, string][] = [[64, "64"], [128, "128"], [256, "256"], [512, "512"], [1024, "1 K"], [2048, "2 K"], [4096, "4 K"], [8192, "8 K"], [16384, "16 K"], [32768, "32 K"], [65536, "64 K"], [131072, "128 K"], [262144, "256 K"], [1048576, "1 M"]];
const WEIGHT_BITS: [number, string][] = [[2, "2 bits"], [3, "3 bits"], [4, "4 bits (q4)"], [5, "5 bits"], [6, "6 bits"], [8, "8 bits (q8)"], [16, "16 bits, unquantized"]];
const SPEEDS: [string, string][] = [["100", "100"], ["1000", "1 000"], ["5000", "5 000"], ["20000", "20 000"]];

function Questions({ answers, set }: { answers: Answers; set: (patch: Partial<Answers>) => void }) {
  const t = useT();
  const nodes = useDesign((s) => s.nodes).filter((n) => !n.id.startsWith("fpga"));
  const { start } = answers;
  return (
    <div className="wizard-questions">
      {start !== "model" && (
        <Question label={t("What are you after?")} help={t("It decides which model I suggest among those that fit.")}>
          <Segmented label={t("Goal")} value={answers.goal} options={[["validate", t("Check that it works")], ["largest", t("The largest model")], ["fastest", t("The highest speed")]]} onChange={(goal) => set({ goal })} />
        </Question>
      )}
      <Question label={t("Weight bits")} help={t("With 8 bits the model barely loses quality; with 4 it takes half and models twice as large fit; below 4 quality drops fast. With 16 they are kept as they come.")}>
        <select aria-label={t("Weight bits")} value={answers.weight_bits} onChange={(e) => set({ weight_bits: Number(e.target.value) })}>
          {WEIGHT_BITS.map(([bits, label]) => (
            <option key={bits} value={bits}>
              {label}
            </option>
          ))}
        </select>
      </Question>
      <Question label={t("Context")} help={t("The tokens the chip remembers. The cache grows with it, and it is what needs the most memory after the weights.")}>
        <select aria-label={t("Context")} value={answers.context_tokens} onChange={(e) => set({ context_tokens: Number(e.target.value) })}>
          {CONTEXTS.map(([tokens, label]) => (
            <option key={tokens} value={tokens}>
              {label} tokens
            </option>
          ))}
        </select>
      </Question>
      {start !== "fpga" && (
        <>
          <Question label={t("Target speed")} help={t("Tokens per second for one user, guaranteed with the context full. The multipliers of each die come from here.")}>
            <Segmented label={t("Speed")} value={String(answers.target_tok_s)} options={SPEEDS} onChange={(v) => set({ target_tok_s: Number(v) })} />
          </Question>
          <Question label={t("Process")} help={t("The manufacturing node: it sets the area of each memory bit and each gate, and the clock.")}>
            <select aria-label={t("Process")} value={answers.node} onChange={(e) => set({ node: e.target.value })}>
              {nodes.map((n) => (
                <option key={n.id} value={n.id}>
                  {t(n.label)}
                </option>
              ))}
            </select>
          </Question>
          <Question label={t("Max die area")} help={t("The physical limit is the reticle, about 858 mm². Small dies come out cheaper per wafer.")}>
            <Segmented label={t("Area")} value={String(answers.die_max_mm2)} options={[["100", "100 mm²"], ["300", "300 mm²"], ["600", "600 mm²"], ["858", t("Reticle")]]} onChange={(v) => set({ die_max_mm2: Number(v) })} />
          </Question>
          <Question label={t("Weights")} help={t("In SRAM the chip takes other models; in ROM they are burnt in at manufacturing and take less.")}>
            <Segmented label={t("Weights")} value={answers.weight_backend} options={[["sram", t("SRAM, swappable")], ["rom", t("ROM, burnt in")]]} onChange={(weight_backend) => set({ weight_backend })} />
          </Question>
        </>
      )}
      {start === "model" && (
        <Question label={t("Chip split")} help={t("By layers: each chip does some layers and passes the vector to the next; little communication, but with one user one chip works at a time. By tensor: they all work on each layer at once; more speed for one user, with fast links. Mixed: tensor groups joined in a chain.")}>
          <Segmented label={t("Split")} value={answers.chip_split} options={[["layers", t("By layers")], ["tensor", t("By tensor")], ["mixed", t("Mixed")]]} onChange={(chip_split) => set({ chip_split })} />
        </Question>
      )}
    </div>
  );
}

function ModelList({ rows, answers, choose }: { rows: Row[]; answers: Answers; choose: (id: string) => void }) {
  const t = useT();
  const judged = answers.start === "fpga" || answers.start === "chip";
  const best = judged ? suggested(rows, answers.goal) : null;
  return (
    <ul className="wizard-models" role="listbox" aria-label={t("Models")}>
      {rows.map((r) => (
        <li key={r.id}>
          <button role="option" aria-selected={answers.model === r.id} className={answers.model === r.id ? "on" : ""} disabled={judged && !r.fits} onClick={() => choose(r.id)}>
            <span className="wizard-model-name">
              {t(r.model.name)}
              {r.id === best && <em>{t("Recommended")}</em>}
            </span>
            <span className="wizard-model-size">{Number.isFinite(r.params_g) ? t(`${formatParams(r.params_g)} parameters`) : ""}</span>
            {judged && <span className={r.fits ? "wizard-model-note" : "wizard-model-note bad"}>{t(r.note)}</span>}
            {judged && r.fits && Number.isFinite(r.tok_s) && <span className="wizard-model-speed">{formatNumber(r.tok_s)} tok/s</span>}
          </button>
        </li>
      ))}
    </ul>
  );
}

/** A drawing of a board, for those whose maker's photo could not be checked. */
function BoardDrawing({ board }: { board: FpgaBoard }) {
  const t = useT();
  return (
    <svg className="board-drawing" viewBox="0 0 320 200" role="img" aria-label={t(`Drawing of the ${board.name} board`)}>
      <rect x="6" y="6" width="308" height="188" rx="10" className="pcb" />
      {[18, 302].flatMap((x) => [18, 182].map((y) => <circle key={`${x}-${y}`} cx={x} cy={y} r="5" className="hole" />))}
      <rect x="110" y="56" width="92" height="92" rx="4" className="fpga" />
      <text x="156" y="98" className="label">
        FPGA
      </text>
      <text x="156" y="114" className="sub">
        {board.fpga.replace(/\(.*\)/, "").match(/[A-Z0-9-]*\d[A-Z0-9-]*/g)?.pop() ?? ""}
      </text>
      {board.ddr_gb > 0 && (
        <>
          <rect x="222" y="62" width="62" height="30" rx="3" className="chip" />
          <rect x="222" y="112" width="62" height="30" rx="3" className="chip" />
          <text x="253" y="160" className="sub">
            {board.ddr_gb >= 1 ? `${board.ddr_gb} GB` : `${Math.round(board.ddr_gb * 1000)} MB`}
          </text>
        </>
      )}
      <rect x="26" y="70" width="44" height="64" rx="3" className="chip" />
      <rect x="40" y="170" width="240" height="12" rx="2" className="header" />
      <text x="160" y="30" className="sub">
        {t("A drawing, not a photo")}
      </text>
    </svg>
  );
}

function BoardPicker({ board, choose }: { board: string; choose: (id: string) => void }) {
  const t = useT();
  const boards = useMemo(() => fpgaBoards() as FpgaBoard[], []);
  const current = boards.find((b) => b.id === board) ?? boards[0];
  const [photoFailed, setPhotoFailed] = useState<string | null>(null);
  const models = boards.filter((b) => b.maker === current.maker).sort((a, b) => a.price_usd - b.price_usd);
  const logic = current.base_logic_k / current.luts_k;
  const memory = (mb: number) => (mb >= 1 ? `${mb} GB` : `${Math.round(mb * 1000)} MB`);
  return (
    <div className="board-picker">
      <div className="board-menus">
        <label>
          {t("Maker")}
          <select value={current.maker} onChange={(e) => choose([...boards].filter((b) => b.maker === e.target.value).sort((a, b) => a.price_usd - b.price_usd)[0].id)}>
            {makers(boards).map((m) => (
              <option key={m}>{m}</option>
            ))}
          </select>
        </label>
        <label>
          {t("Model")}
          <select value={current.id} onChange={(e) => choose(e.target.value)}>
            {models.map((b) => (
              <option key={b.id} value={b.id}>
                {t(b.name)}
              </option>
            ))}
          </select>
        </label>
      </div>
      <article className="board-card">
        <div className="board-photo">
          {current.image && photoFailed !== current.id ? <img src={current.image} alt={t(`Photo of the ${current.name} board`)} referrerPolicy="no-referrer" onError={() => setPhotoFailed(current.id)} /> : <BoardDrawing board={current} />}
        </div>
        <div className="board-specs">
          <h3>{t(current.name)}</h3>
          <dl>
            <dt>FPGA</dt>
            <dd>{current.fpga}</dd>
            <dt>{t("Logic")}</dt>
            <dd>
              {current.luts_k} K {current.lut}
              {logic <= 1 ? ` · ${t("the basic chip takes")} ~${Math.round(logic * 100)} %` : ` · ${t("the basic chip does not fit")}`}
            </dd>
            <dt>{t("On-chip memory")}</dt>
            <dd>{current.bram_mbit} Mbit</dd>
            <dt>{t("Multipliers")}</dt>
            <dd>{current.dsp} DSP</dd>
            <dt>{t("On-board memory")}</dt>
            <dd>{current.ddr_gb > 0 ? `${memory(current.ddr_gb)}, ~${current.ddr_gbps} GB/s` : t("None")}</dd>
            <dt>{t("Clock")}</dt>
            <dd>
              {current.clock_mhz} MHz, {current.clock_measured ? t("measured by routing the chip") : t("estimated")}
            </dd>
            <dt>{t("Tools")}</dt>
            <dd>{t(current.tools)}</dd>
            <dt>{t("Price")}</dt>
            <dd>{current.price}</dd>
          </dl>
          <a href={current.url} target="_blank" rel="noopener noreferrer">
            {t("Official site of")} {current.maker} ↗
          </a>
        </div>
      </article>
    </div>
  );
}

function Summary({ report, model, board }: { report: Report | null; model: ModelSource | undefined; board: string | null }) {
  const t = useT();
  if (!report || !model) return <p className="wizard-empty">{t("The estimator cannot handle this combination. Go back and change an answer.")}</p>;
  const problems = report.violations.filter((v) => v.severity === "error");
  return (
    <div className="wizard-summary">
      <dl>
        <dt>{t("Model")}</dt>
        <dd>{t(model.name)}</dd>
        {board && (
          <>
            <dt>{t("Board")}</dt>
            <dd>{t(board)}</dd>
          </>
        )}
        <dt>{t("Speed")}</dt>
        <dd>{formatNumber(metric(report, "tok_s"))} tok/s</dd>
        <dt>{t("Chips")}</dt>
        <dd>
          {report.dies} {report.dies === 1 ? "die" : "dies"} {t("of")} {formatNumber(report.die_area_mm2)} mm²
        </dd>
        <dt>{t("Power")}</dt>
        <dd>{formatNumber(metric(report, "power_w"))} W</dd>
      </dl>
      {problems.length > 0 && (
        <ul className="wizard-problems">
          {problems.slice(0, 3).map((v) => (
            <li key={v.message}>{t(v.message)}</li>
          ))}
        </ul>
      )}
      <p className="wizard-hint">{t("These are the analytical estimator's figures. Everything can be adjusted afterwards in the parameter panel, and the simulator measures the microprogram's real cycles.")}</p>
    </div>
  );
}

/** What a voice chip is planned for: how soon the answer must sound and how long a question
 * can be. The Voice view measures the stages and plans the dies. */
function VoiceQuestions() {
  const t = useT();
  const v = useVoice();
  return (
    <div className="wizard-questions">
      <p>{t("It is planned in the Voice view, with the models on your computer: Moonshine listens, a language model answers, OuteTTS and DAC speak. Each stage is measured on several shapes of die, and stages that never run at once can share one when that saves silicon.")}</p>
      <label>
        {t("First sound within")}
        <span className="opt-amount">
          <input type="number" min={100} step={50} value={v.firstSoundMs} onChange={(e) => v.set({ firstSoundMs: Number(e.target.value) })} />
          <span className="muted-text">ms</span>
        </span>
      </label>
      <label>
        {t("Longest question")}
        <span className="opt-amount">
          <input type="number" min={1} max={30} step={1} value={v.questionSeconds} onChange={(e) => v.set({ questionSeconds: Number(e.target.value) })} />
          <span className="muted-text">s</span>
        </span>
      </label>
    </div>
  );
}

export function ProjectWizard({ onClose }: { onClose: () => void }) {
  const t = useT();
  const models = useDesign((s) => s.models);
  const defaults = useDesign((s) => s.defaults);
  const boards = useMemo(() => fpgaBoards() as FpgaBoard[], []);
  const createProject = useDesign((s) => s.createProject);
  const setView = useDesign((s) => s.setView);
  const draft = useOptimize((s) => s.draft);
  const setDraft = useOptimize((s) => s.setDraft);
  const [answers, setAnswers] = useState<Answers>({ ...DEFAULT_ANSWERS, model: models[0]?.id ?? "" });
  const [step, setStep] = useState(0);
  const [name, setName] = useState("");
  const dialog = useRef<HTMLDivElement>(null);
  const set = (patch: Partial<Answers>) => setAnswers((a) => ({ ...a, ...patch }));

  const steps = STEPS[answers.start];
  const current = steps[step];
  // Only a board or a single die needs every model judged; otherwise only the chosen one
  // is estimated, at the end: a large model with its chips chosen for it takes a while.
  const judged = answers.start === "fpga" || answers.start === "chip";
  // The estimates run after the step is drawn, so the window shows that it is working; and
  // choosing another model only recomputes what depends on it.
  const [rows, setRows] = useState<Row[]>([]);
  const [busy, setBusy] = useState(false);
  const key = JSON.stringify({ ...answers, model: judged ? "" : answers.model });
  useEffect(() => {
    if (current !== "model" && current !== "name") return;
    setBusy(true);
    const timer = setTimeout(() => {
      const listed: Row[] = models.map((model) => ({ model, id: model.id, params_g: NaN, tok_s: NaN, fits: true, note: "", params: {} }));
      setRows(judged ? judge(answers, models, defaults, boards) : current === "name" ? listed.map((r) => (r.id === answers.model ? judge(answers, [r.model], defaults, boards)[0] : r)) : listed);
      setBusy(false);
    }, 30);
    return () => clearTimeout(timer);
    // `key` stands for the answers that change the estimates.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [current, key, judged, models, defaults, boards]);
  const row = rows.find((r) => r.id === answers.model);
  const params = useMemo(() => row?.params ?? {}, [row]);
  const model = models.find((m) => m.id === answers.model);
  const report = useMemo(() => (current === "name" && model ? estimate(model, defaults, params) : null), [current, model, defaults, params]);
  const board = answers.start === "fpga" ? (boards.find((b) => b.id === answers.board)?.name ?? null) : null;

  // On the model step, the suggestion is chosen unless the person picked a model that fits.
  useEffect(() => {
    if (current !== "model" || (answers.start !== "fpga" && answers.start !== "chip")) return;
    if (rows.find((r) => r.id === answers.model)?.fits) return;
    const best = suggested(rows, answers.goal);
    if (best) set({ model: best });
  }, [current, rows, answers.start, answers.goal, answers.model]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    dialog.current?.focus();
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const canGo = current !== "model" || (!!model && (!(answers.start === "fpga" || answers.start === "chip") || !!row?.fits));
  const finish = () => {
    if (!model) return;
    createProject(name.trim() || `${model.name}${board ? ` on ${board}` : ""}`, model.id, params);
    onClose();
  };
  // The project, then a search measured on the generated circuits with the wizard's goal.
  const optimise = () => {
    finish();
    const goal = answers.goal === "fastest" ? "speed" : "area";
    setDraft({ ...draft, goal, limits: { ...draft.limits, min_tok_s: answers.goal === "fastest" ? 0 : answers.target_tok_s }, context: answers.context_tokens });
    setView("optimize");
  };

  // Over the whole window: the toolbar's blur would otherwise hold it.
  return createPortal(
    <div className="wizard-backdrop" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <div className="wizard" role="dialog" aria-modal="true" aria-label={t("New project")} tabIndex={-1} ref={dialog}>
        <header>
          <span className="wizard-steps">
            {t("Step")} {step + 1} / {steps.length}
          </span>
          <h2>{t(STEP_TITLES[current])}</h2>
        </header>
        <div className="wizard-body">
          {current === "start" && (
            <div className="wizard-grid">
              {STARTS.map((s) => (
                <Choice key={s.id} selected={answers.start === s.id} title={t(s.title)} onClick={() => set({ start: s.id, weight_bits: s.id === "fpga" ? 8 : answers.weight_bits, context_tokens: s.id === "fpga" ? 256 : Math.max(answers.context_tokens, 2048) })}>
                  <p>{t(s.text)}</p>
                </Choice>
              ))}
            </div>
          )}
          {current === "voice" && <VoiceQuestions />}
          {current === "board" && <BoardPicker board={answers.board} choose={(board) => set({ board })} />}
          {current === "questions" && <Questions answers={answers} set={set} />}
          {busy && <p className="wizard-empty">{t("Estimating each design…")}</p>}
          {current === "model" && !busy && <ModelList rows={rows} answers={answers} choose={(id) => set({ model: id })} />}
          {current === "name" && (
            <>
              {!busy && <Summary report={report} model={model} board={board} />}
              <label className="wizard-name">
                {t("Project name")}
                <input type="text" value={name} placeholder={model ? `${t(model.name)}${board ? ` ${t("on")} ${t(board)}` : ""}` : ""} onChange={(e) => setName(e.target.value)} onKeyDown={(e) => e.key === "Enter" && finish()} />
              </label>
            </>
          )}
        </div>
        <footer>
          <button onClick={onClose}>{t("Cancel")}</button>
          <span />
          {step > 0 && <button onClick={() => setStep(step - 1)}>{t("Back")}</button>}
          {current === "voice" ? (
            <button
              className="primary"
              onClick={() => {
                useVoice.getState().set({ planOnOpen: true });
                setView("voice");
                onClose();
              }}
            >
              {t("Plan the chip")}
            </button>
          ) : current === "name" ? (
            <>
              {answers.start !== "fpga" && (
                <button disabled={!model} onClick={optimise} title={t("Create the project and search for the best set of chips, measured on the generated circuits")}>
                  {t("Create and optimise")}
                </button>
              )}
              <button className="primary" disabled={!model} onClick={finish}>
                {t("Create project")}
              </button>
            </>
          ) : (
            <button className="primary" disabled={!canGo} onClick={() => setStep(step + 1)}>
              {t("Next")}
            </button>
          )}
        </footer>
      </div>
    </div>,
    document.body,
  );
}
