import { useEffect, useMemo, useRef, useState } from "react";
import { Segmented } from "../components/controls";
import { Icon } from "../components/icons";
import { useT } from "../i18n";
import { scheduleDesign, type Schedule, type ScheduledStage } from "../engine";
import { formatNumber } from "../format";
import { useDesign } from "../store";
import { matchingPreset } from "../params";
import { chipDraftDepth, loadSignature, useSimulator } from "./simulatorStore";
import { sameModel } from "./sameModel";
import { ChipActivity } from "./ChipActivity";

const KIND_LABEL: Record<ScheduledStage["kind"], string> = {
  matvec: "Matriz × vector",
  vector: "Unidades vectoriales",
  embedding: "Embedding",
  link: "Enlace entre dies",
};
const KIND_CLASS: Record<ScheduledStage["kind"], string> = {
  matvec: "tl-matvec",
  vector: "tl-vector",
  embedding: "tl-embedding",
  link: "tl-link",
};

/** What each kind of stage does in the chip, for the stage the reader points at. */
const KIND_HELP: Record<ScheduledStage["kind"], string> = {
  matvec: "Las celdas del die multiplican esta matriz por el vector de activación: cada celda lee sus pesos de su SRAM, los multiplica por el valor que le llega por la red de activaciones, y el árbol de sumas junta los productos de cada fila.",
  vector: "Las unidades vectoriales trabajan elemento a elemento, carril a carril: normas, rotaciones, funciones por tabla, atención sobre la caché o la regla delta sobre su estado.",
  embedding: "Se lee la fila del token en la tabla de embedding: es el vector con el que empiezan las capas.",
  link: "El vector oculto pasa al die siguiente por el enlace, y allí siguen las capas.",
};

function Timeline({ schedule, pulse }: { schedule: Schedule; pulse: number }) {
  const t = useT();
  const dies = Math.max(...schedule.stages.map((s) => s.die)) + 1;
  const [hover, setHover] = useState<ScheduledStage | null>(null);
  const [pinned, setPinned] = useState<ScheduledStage | null>(null);
  const playhead = useRef<HTMLDivElement>(null);
  const shown = hover ?? pinned;
  const share = (kind: ScheduledStage["kind"]) => schedule.stages.filter((s) => s.kind === kind).reduce((sum, s) => sum + s.cycles, 0) / schedule.total_cycles;
  const ns = (cycles: number) => (cycles / schedule.total_cycles) * schedule.token_ns;
  const percent = (part: number) => (part > 0 && part < 1e-4 ? `< ${formatNumber(0.01)}` : formatNumber(part * 100));

  useEffect(() => {
    const el = playhead.current;
    if (!el || pulse === 0 || window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    el.animate([{ left: "0%", opacity: 1 }, { left: "100%", opacity: 1 }], { duration: 700, easing: "linear" });
  }, [pulse]);

  return (
    <div className="timeline">
      <div className="timeline-head">
        <strong>{t("Recorrido de un token")}</strong>
        <span>{formatNumber(schedule.total_cycles)} {t("ciclos")} · {formatNumber(schedule.token_ns / 1000)} µs</span>
      </div>
      <div className="timeline-rows" onMouseLeave={() => setHover(null)}>
        <div className="timeline-playhead" ref={playhead} />
        {Array.from({ length: dies }, (_, die) => (
          <div className="timeline-row" key={die}>
            <span className="timeline-label">{die}</span>
            <div className="timeline-track">
              {schedule.stages
                .filter((s) => s.die === die)
                .map((s, i) => (
                  <i
                    key={i}
                    className={s === pinned ? `${KIND_CLASS[s.kind]} pinned` : KIND_CLASS[s.kind]}
                    style={{ left: `${(s.start / schedule.total_cycles) * 100}%`, width: `${Math.max(0.04, (s.cycles / schedule.total_cycles) * 100)}%` }}
                    tabIndex={0}
                    aria-label={`${s.name}${s.layer !== null ? `, ${t("capa")} ${s.layer}` : ""}, ${formatNumber(Math.round(s.cycles))} ${t("ciclos")}`}
                    onMouseEnter={() => setHover(s)}
                    onFocus={() => setHover(s)}
                    onBlur={() => setHover(null)}
                    onClick={() => setPinned(s === pinned ? null : s)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" || e.key === " ") setPinned(s === pinned ? null : s);
                    }}
                  />
                ))}
            </div>
          </div>
        ))}
      </div>
      <div className="legend flat">
        {(Object.keys(KIND_LABEL) as ScheduledStage["kind"][]).map((k) => (
          <span key={k}>
            <i className={`swatch ${KIND_CLASS[k]}`} aria-hidden="true" />
            {t(KIND_LABEL[k])} {percent(share(k))} %
          </span>
        ))}
      </div>
      <div className="stage-detail" aria-live="polite">
        {shown ? (
          <>
            <p>
              <strong>{shown.name}</strong>
              {shown.layer !== null ? ` · ${t("capa")} ${shown.layer}` : ""} · die {shown.die} · {formatNumber(Math.round(shown.cycles))} {t("ciclos")} ({percent(shown.cycles / schedule.total_cycles)} % {t("del token")}, {formatNumber(ns(shown.cycles))} ns) · {t("empieza en el ciclo")} {formatNumber(Math.round(shown.start))}
            </p>
            <p className="muted-text">{t(KIND_HELP[shown.kind])}</p>
          </>
        ) : (
          <p className="muted-text">{t("Pasa por encima de un tramo, o púlsalo para fijarlo, y verás qué hace el die en él.")}</p>
        )}
      </div>
    </div>
  );
}

type Profile = "die" | "chip" | "custom";

const PROFILES: Record<Exclude<Profile, "custom">, { firmware: boolean; samplerUnit: boolean; firmwareTokenizer: boolean; transactionLoad: boolean }> = {
  die: { firmware: false, samplerUnit: false, firmwareTokenizer: false, transactionLoad: false },
  chip: { firmware: true, samplerUnit: true, firmwareTokenizer: true, transactionLoad: true },
};

function profileOf(s: { firmware: boolean; samplerUnit: boolean; firmwareTokenizer: boolean; transactionLoad: boolean }): Profile {
  const same = (p: (typeof PROFILES)[keyof typeof PROFILES]) => p.firmware === s.firmware && p.samplerUnit === s.samplerUnit && p.firmwareTokenizer === s.firmwareTokenizer && p.transactionLoad === s.transactionLoad;
  return (Object.keys(PROFILES) as (keyof typeof PROFILES)[]).find((k) => same(PROFILES[k])) ?? "custom";
}

export function SimulatorView() {
  const t = useT();
  const model = useDesign((s) => s.model);
  const params = useDesign((s) => s.params);
  const output = useSimulator((s) => s.output);
  const prompt = useSimulator((s) => s.prompt);
  const image = useSimulator((s) => s.image);
  const current = useSimulator((s) => s.models.find((m) => m.name === s.loaded));
  const readsImages = current?.images?.reads ?? false;
  const makesImages = current?.images?.makes ?? false;
  const draw = useSimulator((s) => s.draw && makesImages);
  const drawing = useSimulator((s) => s.drawing);
  const picture = useSimulator((s) => s.picture);
  const ending = useSimulator((s) => s.ending);
  const imageReport = useSimulator((s) => s.imageReport);
  const set = useSimulator((s) => s.set);
  const generate = useSimulator((s) => s.generate);
  const stop = useSimulator((s) => s.stop);
  const running = useSimulator((s) => s.running);
  // What the engine holds counts only if it is this design's model, built for its formats.
  const loaded = useSimulator((s) => s.loaded !== null && s.loadedSignature === loadSignature(s.path, params) && s.models.some((m) => m.path === s.path && m.name === s.loaded && model !== null && (m.path === model.path || sameModel(model.config, m.config))));
  const phase = useSimulator((s) => s.phase);
  const prefilled = useSimulator((s) => s.prefilled);
  const promptTokens = useSimulator((s) => s.promptTokens);
  const pulse = useSimulator((s) => s.pulse);
  const error = useSimulator((s) => s.error);

  const schedule = useMemo(() => {
    if (!model) return null;
    try {
      return scheduleDesign(model, params);
    } catch {
      return null;
    }
  }, [model, params]);

  return (
    <section className="stage sim-stage">
      <div className="sim-column">
        <div className="transcript" aria-live="polite">
          {picture && draw ? (
            <figure className="picture">
              <img src={picture.data} width={picture.side} height={picture.side} alt={`Imagen generada: ${prompt}`} />
              <figcaption>{picture.side} × {picture.side} px, en {formatNumber(picture.seconds)} s</figcaption>
            </figure>
          ) : output ? <p>{output}</p> : <p className="placeholder">{loaded ? (draw ? "La imagen aparecerá aquí." : "La respuesta del modelo aparecerá aquí.") : "Carga el chip en el panel de la derecha para simularlo: con el modelo y la aritmética del diseño."}</p>}
          {!running && ending && !draw && (
            <p className="note-warning">
              {ending === "loop" ? "El modelo empezó a repetirse sin escribir su fin de turno y el chip lo detuvo, como hace su firmware." : ending === "context" ? "Se llenó el contexto del chip: no recuerda más tokens. Sube el contexto en el diseño para respuestas más largas." : "El modelo no escribió su fin de turno: se detuvo al llegar al límite de tokens."}
            </p>
          )}
          {running && drawing && (
            <p className="placeholder">
              {drawing.decoding ? "El decodificador convierte los tokens en píxeles…" : `El modelo escribe la imagen: ${drawing.done} de ${drawing.total} tokens visuales (${formatNumber(drawing.seconds)} s)`}
            </p>
          )}
          {running && phase === "Leyendo el prompt" && <p className="placeholder">Leyendo el prompt: {prefilled} de {promptTokens} tokens</p>}
        </div>
        {error && (
          <div className="issue error" role="alert">
            <span className="issue-icon" aria-hidden="true">!</span>
            <div><strong>{error}</strong></div>
          </div>
        )}
        {image && (
          <figure className="attached">
            <img src={image.preview} alt="Imagen adjunta" />
            <figcaption>
              {image.width} × {image.height} px{imageReport ? ` · ${formatNumber(imageReport.patches)} parches → ${formatNumber(imageReport.tokens)} tokens en ${formatNumber(imageReport.seconds)} s` : ""}
              <button className="link inline" onClick={() => set({ image: null, imageReport: null })}>Quitar</button>
            </figcaption>
          </figure>
        )}
        {makesImages && (
          <>
            <Segmented label="Qué hace el modelo" value={draw ? "draw" : "answer"} options={[["answer", t("Responder")], ["draw", t("Dibujar")]]} onChange={(mode) => set({ draw: mode === "draw" })} />
            {draw && <p className="muted-text">El dibujo aún se calcula en coma flotante: el chip no lleva todavía la cabeza de generación de imágenes ni el decodificador.</p>}
          </>
        )}
        <div className="composer">
          <label className={readsImages ? "pill attach" : "pill attach disabled"} title={readsImages ? "Adjuntar una imagen: el codificador del modelo la convierte en tokens que se leen antes de la pregunta" : current?.images?.note ?? "Este modelo no lee imágenes"}>
            Imagen
            <input type="file" accept="image/*" disabled={!readsImages} onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) void readImage(file).then((picture) => set({ image: picture, imageReport: null }));
              e.target.value = "";
            }} />
          </label>
          <textarea
            aria-label="Texto de entrada"
            rows={3}
            value={prompt}
            placeholder="Escribe un texto"
            onChange={(e) => set({ prompt: e.target.value })}
            onKeyDown={(e) => {
              if (e.key === "Enter" && (e.metaKey || e.ctrlKey) && loaded && !running) generate();
            }}
          />
          {running ? (
            <button className="pill" onClick={stop}>Detener</button>
          ) : (
            <button className="pill primary-pill" disabled={!loaded || !prompt.trim()} onClick={generate}>Enviar</button>
          )}
        </div>
        {loaded && <ChipActivity />}
        {schedule && <Timeline schedule={schedule} pulse={pulse} />}
      </div>
    </section>
  );
}

/** Longest side, in pixels, of a picture sent to the encoder: more only costs time. */
const IMAGE_SIDE = 512;

/** A picture's pixels as RGB, scaled down to at most `IMAGE_SIDE` on its longer side. */
async function readImage(file: File): Promise<{ width: number; height: number; rgb: number[]; preview: string }> {
  const bitmap = await createImageBitmap(file);
  const scale = Math.min(1, IMAGE_SIDE / Math.max(bitmap.width, bitmap.height));
  const width = Math.max(1, Math.round(bitmap.width * scale));
  const height = Math.max(1, Math.round(bitmap.height * scale));
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d")!;
  context.drawImage(bitmap, 0, 0, width, height);
  const rgba = context.getImageData(0, 0, width, height).data;
  const rgb: number[] = new Array(width * height * 3);
  for (let i = 0, j = 0; i < rgba.length; i += 4, j += 3) {
    rgb[j] = rgba[i];
    rgb[j + 1] = rgba[i + 1];
    rgb[j + 2] = rgba[i + 2];
  }
  return { width, height, rgb, preview: canvas.toDataURL("image/jpeg", 0.85) };
}

export function SimulatorPanel() {
  const t = useT();
  const state = useSimulator();
  const params = useDesign((s) => s.params);
  const formatLabel = matchingPreset(params) ?? "personalizado";
  const report = useDesign((s) => s.report);
  const { connect, set, load } = state;
  const setView = useDesign((s) => s.setView);

  useEffect(() => {
    void connect();
    const timer = setInterval(() => void connect(), 5000);
    return () => clearInterval(timer);
  }, [connect]);

  const mean = state.seconds.length ? state.seconds.reduce((a, b) => a + b, 0) / state.seconds.length : 0;
  const chipRate = report?.metrics.find((m) => m.id === "tok_s")?.value ?? 0;
  // Only weights of the very model the chip is designed for can run on it.
  const design = useDesign((s) => s.model);
  const candidates = useMemo(() => (design ? state.models.filter((m) => m.path === design.path || sameModel(design.config, m.config)) : []), [design, state.models]);
  const selected = candidates.find((m) => m.path === state.path);
  useEffect(() => {
    if (!selected && candidates.length > 0) set({ path: candidates[0].path });
  }, [selected, candidates, set]);
  const stale = state.loaded !== null && state.loadedSignature !== null && state.loadedSignature !== loadSignature(state.path, params);
  const draftDepth = chipDraftDepth(params);

  if (state.online === false) {
    return (
      <aside className="inspector">
        <h2 className="detail-title">Motor local</h2>
        <div className="issue aviso" role="status">
          <span className="issue-icon" aria-hidden="true">i</span>
          <div>
            <strong>El motor no está en marcha.</strong>
            <p>Ábrelo con este comando en la carpeta del proyecto: scripts/dev.sh</p>
          </div>
        </div>
      </aside>
    );
  }

  return (
    <aside className="inspector">
      <h2 className="detail-title">Simulador</h2>
      <p className="detail-sub">Ejecuta el microprograma con pesos reales.</p>

      <h3>{t("El chip que se simula")}</h3>
      <div className="inset padded form">
        <p className="sim-model">
          <strong>{design?.name ?? "Sin modelo"}</strong>
          <span className="muted-text">El modelo del diseño, con su aritmética: enteros de {String(params.activation_bits)} bits y pesos {formatLabel}.</span>
        </p>
        {candidates.length === 0 ? (
          <>
            <p className="muted-text">No hay pesos de este modelo en el equipo, así que el chip no se puede simular todavía.</p>
            <button className="link-button" onClick={() => setView("modelos")}>
              Descargar sus pesos…
            </button>
          </>
        ) : (
          <>
            {candidates.length > 1 && (
              <label>
                Pesos de este modelo
                <select value={state.path} onChange={(e) => set({ path: e.target.value })}>
                  {candidates.map((m) => (
                    <option key={m.path} value={m.path} disabled={!m.supported}>
                      {m.name}
                      {m.supported ? "" : " (no simulable)"}
                    </option>
                  ))}
                </select>
              </label>
            )}
            {selected && (
              <p className="muted-text">
                {selected.name} · {selected.architecture} · {formatNumber(selected.size_gb)} GB
              </p>
            )}
            <label className="check">
              <input type="checkbox" checked={state.useGpu} onChange={(e) => set({ useGpu: e.target.checked })} />
              {t("Acelerar con la GPU")}
            </label>
            <button className="pill primary-pill" disabled={state.loading || !selected?.supported} onClick={() => void load()}>
              {t(state.loading ? "Cargando…" : state.loaded ? "Recargar" : "Cargar el chip")}
            </button>
            {stale && <p className="note-warning">El diseño cambió desde que se cargó: recarga para simular el chip actual.</p>}
            {state.loaded && <p className="muted-text">Cargado: {state.loaded} · {state.accelerator ?? "CPU"}</p>}
          </>
        )}
      </div>

      <h3>{t("Qué se simula")}</h3>
      <div className="inset padded form">
        <Segmented
          label="Perfil"
          value={profileOf(state)}
          options={[["die", t("El die")], ["chip", t("El chip completo")], ["custom", t("A medida")]]}
          onChange={(profile) => {
            if (profile !== "custom") set(PROFILES[profile]);
          }}
        />
        {(
          [
            ["firmware", "CPU y firmware del chip", "El firmware real corre instrucción a instrucción y gobierna el die por sus registros.", state.firmware, false],
            ["samplerUnit", "Unidad de muestreo del chip", "Elige cada token con la misma lógica que el circuito, top-k incluido. Con la CPU siempre se usa.", state.samplerUnit || state.firmware, state.firmware],
            ["firmwareTokenizer", "Tokenizador en el firmware", "El firmware convierte el texto en tokens y los tokens en bytes, y se compara con la biblioteca de referencia.", state.firmware && state.firmwareTokenizer, !state.firmware],
            ["transactionLoad", "Comprobar la carga de la imagen", "Al cargar, recorre todos los bloques como el controlador real, verifica su CRC y cuenta los ciclos. Hay que recargar el chip.", state.transactionLoad, false],
          ] as const
        ).map(([key, label, help, checked, disabled]) => (
          <label className="check" key={key} title={help}>
            <input type="checkbox" checked={checked} disabled={disabled} onChange={(e) => set({ [key]: e.target.checked })} />
            {t(label)}
            <span className="hint-icon" aria-label={help}>
              <Icon name="info" size={13} />
            </span>
          </label>
        ))}
        <dl className="metrics">
          <div><dt>Aritmética</dt><dd>Enteros de {String(params.activation_bits)} bits, la del chip</dd></div>
          <div><dt>Tokenizador</dt><dd>{state.firmware && state.firmwareTokenizer ? "En el firmware" : "Fuera del chip"}</dd></div>
          <div><dt>Carga de pesos</dt><dd>{params.weight_backend === "rom" ? "Grabados en el chip (ROM)" : state.transactionLoad ? "Por transacciones" : "Directa del checkpoint"}</dd></div>
        </dl>
      </div>

      <h3>{t("Generación")}</h3>
      <div className="inset padded form">
        <label>
          Tokens máximos
          <input type="text" inputMode="numeric" value={state.maxTokens} onChange={(e) => set({ maxTokens: Number(e.target.value) || 1 })} />
        </label>
        <label>
          Temperatura (0 elige siempre el más probable)
          <input type="text" inputMode="decimal" value={state.temperature} onChange={(e) => set({ temperature: Number(e.target.value.replace(",", ".")) || 0 })} />
        </label>
        <label className="check" title="Con la plantilla de chat del modelo, abre su bloque de razonamiento antes de la respuesta. Sin él, el bloque llega cerrado y el modelo contesta directamente. Es texto del prompt: no cambia el chip.">
          <input type="checkbox" checked={state.thinking} onChange={(e) => set({ thinking: e.target.checked })} />
          Razonar antes de responder
        </label>
        {state.thinking && (
          <label title="Cuántos tokens puede razonar el modelo antes de que el firmware del chip cierre el razonamiento por él y pase a la respuesta. Vacío lo deja sin límite; 0 equivale a no razonar.">
            Presupuesto de razonamiento (tokens, vacío sin límite)
            <input type="text" inputMode="numeric" value={state.thinkBudget} placeholder="Sin límite" onChange={(e) => set({ thinkBudget: e.target.value.replace(/[^0-9]/g, "") })} />
          </label>
        )}
        <p className="muted-text">{draftDepth > 0 ? `La cabeza MTP del diseño adivina ${draftDepth} tokens y se miden sus aciertos.` : "El diseño no lleva cabeza MTP."}</p>
      </div>

      <h3>{t("Medidas")}</h3>
      <dl className="inset metrics">
        <div><dt>Fase</dt><dd>{state.phase || "—"}</dd></div>
        <div><dt>Tokens generados</dt><dd>{state.tokens}</dd></div>
        <div><dt>Simulación</dt><dd>{mean ? `${formatNumber(mean)} s/token` : "—"}</dd></div>
        <div><dt>Este chip, estimado</dt><dd>{formatNumber(chipRate)} tok/s</dd></div>
        <div><dt>Más rápido que simularlo</dt><dd>{mean ? `×${formatNumber(chipRate * mean)}` : "—"}</dd></div>
        {state.loadReport && <div><dt>Imagen: bloques</dt><dd>{formatNumber(state.loadReport.segments)}{state.loadReport.created ? " (creada)" : ""}</dd></div>}
        {state.loadReport && <div><dt>Imagen: ciclos de carga</dt><dd>{formatNumber(state.loadReport.cycles)}</dd></div>}
        {state.loadReport && <div><dt>Imagen: errores de CRC</dt><dd>{state.loadReport.crc_errors.length}</dd></div>}
        {state.chip && <div><dt>Ciclos del chip, con firmware</dt><dd>{formatNumber(state.chip.cycles)}</dd></div>}
        {state.chip?.tokenizerMatch != null && <div><dt>Tokenizador frente a la referencia</dt><dd>{state.chip.tokenizerMatch ? "Coincide" : "Difiere"}</dd></div>}
        {state.chip && <div><dt>Fuera del die (CPU y muestreo)</dt><dd>{formatNumber(state.chip.cycles - state.chip.dieCycles)} ciclos</dd></div>}
        {state.acceptance.tries.map((tries, depth) => (
          <div key={depth}><dt>Aciertos MTP a profundidad {depth + 1}</dt><dd>{tries ? `${formatNumber((state.acceptance.hits[depth] / tries) * 100)} %` : "—"}</dd></div>
        ))}
      </dl>
    </aside>
  );
}
