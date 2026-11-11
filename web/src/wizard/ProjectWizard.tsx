import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { createPortal } from "react-dom";
import { estimateDesign, type ChipParams, type ModelSource, type Report } from "../engine";
import { formatNumber } from "../format";
import { useDesign } from "../store";
import { Segmented } from "../components/controls";
import { BASE_LOGIC_K, buildable, chipParams, DEFAULT_ANSWERS, FPGA_BOARDS, FPGA_MAKERS, fpgaFit, fpgaParams, formatMbit, metric, modelParams, suggested, type Answers, type FpgaBoard, type FpgaFit, type Start } from "./plan";

const STARTS: { id: Start; title: string; text: string }[] = [
  { id: "fpga", title: "Una FPGA", text: "Eliges una placa y te dice qué modelos le caben, dónde van sus pesos y a qué velocidad irían." },
  { id: "chip", title: "Un solo chip", text: "Todo en un die: eliges el proceso, el área y la velocidad, y ves qué modelos caben." },
  { id: "model", title: "Un modelo", text: "Eliges el LLM; el número de chips y su reparto salen de la velocidad que pidas." },
  { id: "custom", title: "A medida", text: "Los parámetros por defecto con el modelo que elijas, para ajustarlo todo tú." },
];

type Step = "start" | "board" | "questions" | "model" | "name";

const STEPS: Record<Start, Step[]> = {
  fpga: ["start", "board", "questions", "model", "name"],
  chip: ["start", "questions", "model", "name"],
  model: ["start", "model", "questions", "name"],
  custom: ["start", "model", "name"],
};

const STEP_TITLES: Record<Step, string> = {
  start: "¿Desde dónde empiezas?",
  board: "¿Qué placa tienes o quieres comprar?",
  questions: "Unas preguntas",
  model: "¿Qué modelo?",
  name: "Resumen",
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
function judge(answers: Answers, models: ModelSource[], defaults: ChipParams): Row[] {
  const board = FPGA_BOARDS.find((b) => b.id === answers.board) ?? FPGA_BOARDS[0];
  return models.map((model) => {
    if (answers.start === "fpga") {
      // First where the weights and the cache would go, then the speed with them there.
      const sizing = estimate(model, defaults, fpgaParams(answers, board, { where: "chip", memory_mbit: 0, logic_share: 0 }));
      const fit: FpgaFit = sizing ? fpgaFit(board, metric(sizing, "model_weights_gb"), metric(sizing, "kv_gb")) : { where: "none", reason: "El estimador no admite este modelo." };
      const { params, report: placed } = sized(model, defaults, fpgaParams(answers, board, fit));
      const report = fit.where === "none" ? null : placed;
      const note = fit.where === "none" ? fit.reason : fit.where === "chip" ? `En la memoria de la FPGA: ${formatMbit(fit.memory_mbit)} de ${board.bram_mbit} Mbit, lógica al ${Math.round(fit.logic_share * 100)} %.` : `Pesos en la memoria de la placa (${formatMbit(fit.memory_mbit)}): su ancho de banda marca la velocidad.`;
      return { model, id: model.id, params_g: metric(sizing, "model_params_g"), tok_s: metric(report, "tok_s"), fits: fit.where !== "none" && !!report, note, params };
    }
    const { params, report } = sized(model, defaults, answers.start === "chip" ? chipParams(answers) : answers.start === "model" ? modelParams(answers) : {});
    const fits = buildable(report) && (answers.start !== "chip" || report?.dies === 1);
    const problem = report?.violations.find((v) => v.severity === "error")?.message;
    const note = !report ? "El estimador no admite este modelo." : fits ? `${formatNumber(report.die_area_mm2)} mm² por die${report.dies > 1 ? `, ${report.dies} dies` : ""}.` : (problem ?? "No cabe en un die.");
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
  return (
    <div className="wizard-question">
      <div>
        <strong>{label}</strong>
        <p>{help}</p>
      </div>
      {children}
    </div>
  );
}

const CONTEXTS: [string, string][] = [["64", "64"], ["256", "256"], ["2048", "2 K"], ["8192", "8 K"], ["32768", "32 K"]];
const SPEEDS: [string, string][] = [["100", "100"], ["1000", "1 000"], ["5000", "5 000"], ["20000", "20 000"]];

function Questions({ answers, set }: { answers: Answers; set: (patch: Partial<Answers>) => void }) {
  const nodes = useDesign((s) => s.nodes).filter((n) => !n.id.startsWith("fpga"));
  const { start } = answers;
  return (
    <div className="wizard-questions">
      {start !== "model" && (
        <Question label="¿Qué buscas?" help="Decide qué modelo te propongo entre los que caben.">
          <Segmented label="Objetivo" value={answers.goal} options={[["validate", "Validar que funciona"], ["largest", "El modelo más grande"], ["fastest", "La mayor velocidad"]]} onChange={(goal) => set({ goal })} />
        </Question>
      )}
      <Question label="Bits de peso" help="Con 8 bits el modelo apenas pierde calidad; con 4 ocupa la mitad y caben modelos el doble de grandes.">
        <Segmented label="Bits de peso" value={String(answers.weight_bits)} options={[["8", "8 bits"], ["4", "4 bits"]]} onChange={(v) => set({ weight_bits: Number(v) })} />
      </Question>
      <Question label="Contexto" help="Los tokens que el chip recuerda. La caché crece con él, y es lo que más memoria pide después de los pesos.">
        <Segmented label="Contexto" value={String(answers.context_tokens)} options={start === "fpga" ? CONTEXTS.slice(0, 4) : CONTEXTS.slice(2)} onChange={(v) => set({ context_tokens: Number(v) })} />
      </Question>
      {start !== "fpga" && (
        <>
          <Question label="Velocidad objetivo" help="Tokens por segundo para un usuario, garantizados con el contexto lleno. De aquí salen los multiplicadores de cada die.">
            <Segmented label="Velocidad" value={String(answers.target_tok_s)} options={SPEEDS} onChange={(v) => set({ target_tok_s: Number(v) })} />
          </Question>
          <Question label="Proceso" help="El nodo de fabricación: fija el área de cada bit de memoria y de cada puerta, y la frecuencia.">
            <select aria-label="Proceso" value={answers.node} onChange={(e) => set({ node: e.target.value })}>
              {nodes.map((n) => (
                <option key={n.id} value={n.id}>
                  {n.label}
                </option>
              ))}
            </select>
          </Question>
          <Question label="Área máxima por die" help="El límite físico es la retícula, unos 858 mm². Los dies pequeños salen más baratos por oblea.">
            <Segmented label="Área" value={String(answers.die_max_mm2)} options={[["100", "100 mm²"], ["300", "300 mm²"], ["600", "600 mm²"], ["858", "Retícula"]]} onChange={(v) => set({ die_max_mm2: Number(v) })} />
          </Question>
          <Question label="Pesos" help="En SRAM el chip admite otros modelos; en ROM se graban al fabricarlo y ocupan menos.">
            <Segmented label="Pesos" value={answers.weight_backend} options={[["sram", "SRAM, intercambiables"], ["rom", "ROM, grabados"]]} onChange={(weight_backend) => set({ weight_backend })} />
          </Question>
        </>
      )}
      {start === "model" && (
        <Question label="Reparto entre chips" help="Por capas: cada chip hace unas capas y pasa el vector al siguiente; poca comunicación, pero con un usuario trabaja un chip cada vez. Por tensor: todos trabajan en cada capa a la vez; más velocidad para un usuario, con enlaces rápidos. Mixto: grupos en tensor unidos en cadena.">
          <Segmented label="Reparto" value={answers.chip_split} options={[["layers", "Por capas"], ["tensor", "Por tensor"], ["mixed", "Mixto"]]} onChange={(chip_split) => set({ chip_split })} />
        </Question>
      )}
    </div>
  );
}

function ModelList({ rows, answers, choose }: { rows: Row[]; answers: Answers; choose: (id: string) => void }) {
  const judged = answers.start === "fpga" || answers.start === "chip";
  const best = judged ? suggested(rows, answers.goal) : null;
  return (
    <ul className="wizard-models" role="listbox" aria-label="Modelos">
      {rows.map((r) => (
        <li key={r.id}>
          <button role="option" aria-selected={answers.model === r.id} className={answers.model === r.id ? "on" : ""} disabled={judged && !r.fits} onClick={() => choose(r.id)}>
            <span className="wizard-model-name">
              {r.model.name}
              {r.id === best && <em>Recomendado</em>}
            </span>
            <span className="wizard-model-size">{Number.isFinite(r.params_g) ? `${formatParams(r.params_g)} parámetros` : ""}</span>
            {judged && <span className={r.fits ? "wizard-model-note" : "wizard-model-note bad"}>{r.note}</span>}
            {judged && r.fits && Number.isFinite(r.tok_s) && <span className="wizard-model-speed">{formatNumber(r.tok_s)} tok/s</span>}
          </button>
        </li>
      ))}
    </ul>
  );
}

/** A drawing of a board, for those whose maker's photo could not be checked. */
function BoardDrawing({ board }: { board: FpgaBoard }) {
  return (
    <svg className="board-drawing" viewBox="0 0 320 200" role="img" aria-label={`Ilustración de la placa ${board.name}`}>
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
        Ilustración, no es una foto
      </text>
    </svg>
  );
}

function BoardPicker({ board, choose }: { board: string; choose: (id: string) => void }) {
  const current = FPGA_BOARDS.find((b) => b.id === board) ?? FPGA_BOARDS[0];
  const [photoFailed, setPhotoFailed] = useState<string | null>(null);
  const models = FPGA_BOARDS.filter((b) => b.maker === current.maker).sort((a, b) => a.price_usd - b.price_usd);
  const logic = BASE_LOGIC_K[current.lut] / current.luts_k;
  const memory = (mb: number) => (mb >= 1 ? `${mb} GB` : `${Math.round(mb * 1000)} MB`);
  return (
    <div className="board-picker">
      <div className="board-menus">
        <label>
          Fabricante
          <select value={current.maker} onChange={(e) => choose([...FPGA_BOARDS].filter((b) => b.maker === e.target.value).sort((a, b) => a.price_usd - b.price_usd)[0].id)}>
            {FPGA_MAKERS.map((m) => (
              <option key={m}>{m}</option>
            ))}
          </select>
        </label>
        <label>
          Modelo
          <select value={current.id} onChange={(e) => choose(e.target.value)}>
            {models.map((b) => (
              <option key={b.id} value={b.id}>
                {b.name} · {b.price}
              </option>
            ))}
          </select>
        </label>
      </div>
      <article className="board-card">
        <div className="board-photo">
          {current.image && photoFailed !== current.id ? <img src={current.image} alt={`Foto de la placa ${current.name}`} referrerPolicy="no-referrer" onError={() => setPhotoFailed(current.id)} /> : <BoardDrawing board={current} />}
        </div>
        <div className="board-specs">
          <h3>{current.name}</h3>
          <dl>
            <dt>FPGA</dt>
            <dd>{current.fpga}</dd>
            <dt>Lógica</dt>
            <dd>
              {current.luts_k} K {current.lut}
              {logic <= 1 ? ` · el chip básico ocupa ~${Math.round(logic * 100)} %` : " · no cabe el chip básico"}
            </dd>
            <dt>Memoria en el chip</dt>
            <dd>{current.bram_mbit} Mbit</dd>
            <dt>Multiplicadores</dt>
            <dd>{current.dsp} DSP</dd>
            <dt>Memoria en la placa</dt>
            <dd>{current.ddr_gb > 0 ? `${memory(current.ddr_gb)}, ~${current.ddr_gbps} GB/s` : "Ninguna"}</dd>
            <dt>Reloj estimado</dt>
            <dd>{current.clock_mhz} MHz</dd>
            <dt>Herramientas</dt>
            <dd>{current.tools}</dd>
            <dt>Precio</dt>
            <dd>{current.price}</dd>
          </dl>
          <a href={current.url} target="_blank" rel="noopener noreferrer">
            Web oficial de {current.maker} ↗
          </a>
        </div>
      </article>
    </div>
  );
}

function Summary({ report, model, board }: { report: Report | null; model: ModelSource | undefined; board: string | null }) {
  if (!report || !model) return <p className="wizard-empty">El estimador no puede con esta combinación. Vuelve atrás y cambia alguna respuesta.</p>;
  const problems = report.violations.filter((v) => v.severity === "error");
  return (
    <div className="wizard-summary">
      <dl>
        <dt>Modelo</dt>
        <dd>{model.name}</dd>
        {board && (
          <>
            <dt>Placa</dt>
            <dd>{board}</dd>
          </>
        )}
        <dt>Velocidad</dt>
        <dd>{formatNumber(metric(report, "tok_s"))} tok/s</dd>
        <dt>Chips</dt>
        <dd>
          {report.dies} {report.dies === 1 ? "die" : "dies"} de {formatNumber(report.die_area_mm2)} mm²
        </dd>
        <dt>Consumo</dt>
        <dd>{formatNumber(metric(report, "power_w"))} W</dd>
      </dl>
      {problems.length > 0 && (
        <ul className="wizard-problems">
          {problems.slice(0, 3).map((v) => (
            <li key={v.message}>{v.message}</li>
          ))}
        </ul>
      )}
      <p className="wizard-hint">Son cifras del estimador analítico. Todo se puede ajustar después en el panel de parámetros, y el simulador mide los ciclos reales del microprograma.</p>
    </div>
  );
}

export function ProjectWizard({ onClose }: { onClose: () => void }) {
  const models = useDesign((s) => s.models);
  const defaults = useDesign((s) => s.defaults);
  const createProject = useDesign((s) => s.createProject);
  const [answers, setAnswers] = useState<Answers>({ ...DEFAULT_ANSWERS, model: models[0]?.id ?? "" });
  const [step, setStep] = useState(0);
  const [name, setName] = useState("");
  const dialog = useRef<HTMLDivElement>(null);
  const set = (patch: Partial<Answers>) => setAnswers((a) => ({ ...a, ...patch }));

  const steps = STEPS[answers.start];
  const current = steps[step];
  const rows = useMemo(() => (current === "model" || current === "name" ? judge(answers, models, defaults) : []), [current, answers, models, defaults]);
  const row = rows.find((r) => r.id === answers.model);
  const params = useMemo(() => row?.params ?? {}, [row]);
  const model = models.find((m) => m.id === answers.model);
  const report = useMemo(() => (current === "name" && model ? estimate(model, defaults, params) : null), [current, model, defaults, params]);
  const board = answers.start === "fpga" ? (FPGA_BOARDS.find((b) => b.id === answers.board)?.name ?? null) : null;

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
    createProject(name.trim() || `${model.name}${board ? ` en ${board}` : ""}`, model.id, params);
    onClose();
  };

  // Over the whole window: the toolbar's blur would otherwise hold it.
  return createPortal(
    <div className="wizard-backdrop" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <div className="wizard" role="dialog" aria-modal="true" aria-label="Nuevo proyecto" tabIndex={-1} ref={dialog}>
        <header>
          <span className="wizard-steps">
            Paso {step + 1} de {steps.length}
          </span>
          <h2>{STEP_TITLES[current]}</h2>
        </header>
        <div className="wizard-body">
          {current === "start" && (
            <div className="wizard-grid">
              {STARTS.map((s) => (
                <Choice key={s.id} selected={answers.start === s.id} title={s.title} onClick={() => set({ start: s.id, weight_bits: s.id === "fpga" ? 8 : answers.weight_bits, context_tokens: s.id === "fpga" ? 256 : Math.max(answers.context_tokens, 2048) })}>
                  <p>{s.text}</p>
                </Choice>
              ))}
            </div>
          )}
          {current === "board" && <BoardPicker board={answers.board} choose={(board) => set({ board })} />}
          {current === "questions" && <Questions answers={answers} set={set} />}
          {current === "model" && <ModelList rows={rows} answers={answers} choose={(id) => set({ model: id })} />}
          {current === "name" && (
            <>
              <Summary report={report} model={model} board={board} />
              <label className="wizard-name">
                Nombre del proyecto
                <input type="text" value={name} placeholder={model ? `${model.name}${board ? ` en ${board}` : ""}` : ""} onChange={(e) => setName(e.target.value)} onKeyDown={(e) => e.key === "Enter" && finish()} />
              </label>
            </>
          )}
        </div>
        <footer>
          <button onClick={onClose}>Cancelar</button>
          <span />
          {step > 0 && <button onClick={() => setStep(step - 1)}>Atrás</button>}
          {current === "name" ? (
            <button className="primary" disabled={!model} onClick={finish}>
              Crear proyecto
            </button>
          ) : (
            <button className="primary" disabled={!canGo} onClick={() => setStep(step + 1)}>
              Siguiente
            </button>
          )}
        </footer>
      </div>
    </div>,
    document.body,
  );
}
