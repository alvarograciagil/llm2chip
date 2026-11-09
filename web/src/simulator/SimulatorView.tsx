import { useEffect, useMemo, useRef, useState } from "react";
import { Segmented } from "../components/controls";
import { scheduleDesign, type Schedule, type ScheduledStage } from "../engine";
import { formatNumber } from "../format";
import { useDesign } from "../store";
import { matchingPreset } from "../params";
import { useSimulator, type WeightFormat } from "./simulatorStore";

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

/** The weight format of the current design, as the engine expects it. */
function designFormat(p: Record<string, unknown>): WeightFormat {
  const coding = String(p.weight_coding);
  const grouped = coding === "affine" || coding === "float";
  return {
    coding,
    bits: Number(p.weight_bits),
    exponent_bits: Number(p.weight_exponent_bits),
    bias: p.weight_bias === true,
    group_size: grouped ? Number(p.group_size) : 0,
    fine_scale: p.weight_fine_scale === true,
    keep_source: coding === "affine" && p.use_model_mix === true,
  };
}

function Timeline({ schedule, pulse }: { schedule: Schedule; pulse: number }) {
  const dies = Math.max(...schedule.stages.map((s) => s.die)) + 1;
  const [hover, setHover] = useState<ScheduledStage | null>(null);
  const playhead = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = playhead.current;
    if (!el || pulse === 0 || window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    el.animate([{ left: "0%", opacity: 1 }, { left: "100%", opacity: 1 }], { duration: 700, easing: "linear" });
  }, [pulse]);

  return (
    <div className="timeline">
      <div className="timeline-head">
        <strong>Recorrido de un token</strong>
        <span>{hover ? `${hover.name}${hover.layer !== null ? ` · capa ${hover.layer}` : ""} · ${formatNumber(hover.cycles)} ciclos` : `${formatNumber(schedule.total_cycles)} ciclos`}</span>
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
                    className={KIND_CLASS[s.kind]}
                    style={{ left: `${(s.start / schedule.total_cycles) * 100}%`, width: `${Math.max(0.04, (s.cycles / schedule.total_cycles) * 100)}%` }}
                    onMouseEnter={() => setHover(s)}
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
            {KIND_LABEL[k]}
          </span>
        ))}
      </div>
    </div>
  );
}

export function SimulatorView() {
  const model = useDesign((s) => s.model);
  const params = useDesign((s) => s.params);
  const output = useSimulator((s) => s.output);
  const prompt = useSimulator((s) => s.prompt);
  const set = useSimulator((s) => s.set);
  const generate = useSimulator((s) => s.generate);
  const stop = useSimulator((s) => s.stop);
  const running = useSimulator((s) => s.running);
  const loaded = useSimulator((s) => s.loaded);
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
          {output ? <p>{output}</p> : <p className="placeholder">{loaded ? "La respuesta del modelo aparecerá aquí." : "Carga un modelo en el panel de la derecha para empezar."}</p>}
          {running && phase === "Leyendo el prompt" && <p className="placeholder">Leyendo el prompt: {prefilled} de {promptTokens} tokens</p>}
        </div>
        {error && (
          <div className="issue error" role="alert">
            <span className="issue-icon" aria-hidden="true">!</span>
            <div><strong>{error}</strong></div>
          </div>
        )}
        <div className="composer">
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
        {schedule && <Timeline schedule={schedule} pulse={pulse} />}
      </div>
    </section>
  );
}

export function SimulatorPanel() {
  const state = useSimulator();
  const params = useDesign((s) => s.params);
  const formatLabel = matchingPreset(params) ?? "personalizado";
  const report = useDesign((s) => s.report);
  const { connect, set, load } = state;

  useEffect(() => {
    void connect();
    const timer = setInterval(() => void connect(), 5000);
    return () => clearInterval(timer);
  }, [connect]);

  const mean = state.seconds.length ? state.seconds.reduce((a, b) => a + b, 0) / state.seconds.length : 0;
  const chipRate = report?.metrics.find((m) => m.id === "tok_s")?.value ?? 0;
  const selected = state.models.find((m) => m.path === state.path);

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

      <h3>Modelo</h3>
      <div className="inset padded form">
        <label>
          Pesos
          <select value={state.path} onChange={(e) => set({ path: e.target.value })}>
            {state.models.map((m) => (
              <option key={m.path} value={m.path} disabled={!m.supported}>
                {m.name}{m.supported ? "" : " (no soportado)"}
              </option>
            ))}
          </select>
        </label>
        {selected && <p className="muted-text">{selected.architecture} · {formatNumber(selected.size_gb)} GB</p>}
        <Segmented
          label="Aritmética"
          value={state.numeric}
          options={[["float", "Coma flotante"], ["fixed", "Exponente por vector"], ["exact", "Enteros exactos"]]}
          onChange={(numeric) => set({ numeric })}
        />
        {state.numeric === "exact" && (
          <div className="fixed-grid">
            <label>
              Bits de vector
              <input type="text" inputMode="numeric" value={state.activationBits} onChange={(e) => set({ activationBits: Number(e.target.value) || 0 })} />
            </label>
          </div>
        )}
        {state.numeric === "fixed" && (
          <div className="fixed-grid">
            {([
              ["Activación, bits", "activationBits"],
              ["Estado, bits", "stateBits"],
              ["Tablas, bits", "tableBits"],
            ] as const).map(([label, key]) => (
              <label key={label + key}>
                {label}
                <input type="text" inputMode="numeric" value={state[key]} onChange={(e) => set({ [key]: Number(e.target.value) || 0 })} />
              </label>
            ))}
          </div>
        )}
        <label className="check">
          <input type="checkbox" checked={state.useDesignFormat} onChange={(e) => set({ useDesignFormat: e.target.checked, designFormat: designFormat(params) })} />
          Convertir los pesos al formato del chip ({formatLabel})
        </label>
        <label className="check">
          <input type="checkbox" checked={state.useGpu} onChange={(e) => set({ useGpu: e.target.checked })} />
          Acelerar con la GPU
        </label>
        <button className="pill primary-pill" disabled={state.loading || !state.path} onClick={() => { set({ designFormat: designFormat(params) }); void load(); }}>
          {state.loading ? "Cargando…" : state.loaded ? "Recargar" : "Cargar modelo"}
        </button>
        {state.loaded && <p className="muted-text">Cargado: {state.loaded} · {state.accelerator ?? "CPU"}</p>}
      </div>

      <h3>Generación</h3>
      <div className="inset padded form">
        <label>
          Tokens máximos
          <input type="text" inputMode="numeric" value={state.maxTokens} onChange={(e) => set({ maxTokens: Number(e.target.value) || 1 })} />
        </label>
        <label>
          Temperatura (0 elige siempre el más probable)
          <input type="text" inputMode="decimal" value={state.temperature} onChange={(e) => set({ temperature: Number(e.target.value.replace(",", ".")) || 0 })} />
        </label>
        <label>
          Medir cabeza MTP (profundidad, 0 para desactivar)
          <input type="text" inputMode="numeric" value={state.mtpDepth} onChange={(e) => set({ mtpDepth: Math.min(3, Number(e.target.value) || 0) })} />
        </label>
      </div>

      <h3>Medidas</h3>
      <dl className="inset metrics">
        <div><dt>Fase</dt><dd>{state.phase || "—"}</dd></div>
        <div><dt>Tokens generados</dt><dd>{state.tokens}</dd></div>
        <div><dt>Simulación</dt><dd>{mean ? `${formatNumber(mean)} s/token` : "—"}</dd></div>
        <div><dt>Este chip, estimado</dt><dd>{formatNumber(chipRate)} tok/s</dd></div>
        <div><dt>Más rápido que simularlo</dt><dd>{mean ? `×${formatNumber(chipRate * mean)}` : "—"}</dd></div>
        {state.acceptance.tries.map((tries, depth) => (
          <div key={depth}><dt>Aciertos MTP a profundidad {depth + 1}</dt><dd>{tries ? `${formatNumber((state.acceptance.hits[depth] / tries) * 100)} %` : "—"}</dd></div>
        ))}
      </dl>
    </aside>
  );
}
