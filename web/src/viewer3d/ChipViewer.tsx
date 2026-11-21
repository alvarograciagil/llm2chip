import { useEffect, useRef, useState, type ChangeEvent } from "react";
import { useT } from "../i18n";
import { useDesign } from "../store";
import { Segmented } from "../components/controls";
import { download } from "../download";
import { ChipScene, readPalette } from "./scene";
import { DEFAULT_RECORD, useViewer, type RecordRequest } from "./viewerStore";
import { dieAt, dieTotals, divergedAt, opKinds, stepAt, totalCycles, type Stride, type Trace, type WordClass } from "./trace";

// The 3D viewer of the chip at work: loaded on demand, so Three.js only downloads when the
// view opens. The stage shows the scene; the panel loads traces and drives playback.

export default function ChipViewer() {
  return (
    <>
      <ChipStage />
      <ChipPanel />
    </>
  );
}

const CLASS_LABEL: Record<WordClass, string> = {
  cells: "Cells multiply and add",
  vector: "Every lane computes",
  scalar: "One element at a time",
  send: "Sends to the next chip",
  take: "Takes from the previous chip",
  halt: "Finished",
};

const CLASS_COLOR: Record<WordClass, string> = {
  cells: "var(--cat-memory)",
  vector: "var(--cat-compute)",
  scalar: "var(--green)",
  send: "var(--cat-interconnect)",
  take: "var(--cat-interconnect)",
  halt: "var(--neutral-fill)",
};

function ChipStage() {
  const t = useT();
  const host = useRef<HTMLDivElement>(null);
  const labels = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<ChipScene | null>(null);
  const trace = useViewer((s) => s.trace);
  const appearance = useDesign((s) => s.appearance);
  const [failed, setFailed] = useState<string | null>(null);

  useEffect(() => {
    if (!host.current) return;
    let scene: ChipScene;
    try {
      scene = new ChipScene(host.current);
    } catch (e) {
      setFailed(e instanceof Error ? e.message : String(e));
      return;
    }
    sceneRef.current = scene;
    let last = performance.now();
    scene.setTick(() => {
      const now = performance.now();
      const viewer = useViewer.getState();
      viewer.advance(Math.min((now - last) / 1000, 0.1));
      last = now;
      const { cycle, selected, speed, playing } = useViewer.getState();
      // Words cross a link in a cycle: show those of the last few frames' worth of cycles.
      scene.show(cycle, selected, playing ? Math.max(16, speed / 4) : 16);
    });
    scene.onPick = (die) => useViewer.getState().select(die);
    scene.onFrame = (positions) => {
      const box = labels.current;
      if (!box) return;
      positions.forEach((p, i) => {
        const label = box.children[i] as HTMLElement | undefined;
        if (!label) return;
        label.style.transform = `translate(${p.x}px, ${p.y}px) translate(-50%, -100%)`;
        label.style.visibility = p.visible ? "visible" : "hidden";
      });
    };
    return () => {
      scene.dispose();
      sceneRef.current = null;
    };
  }, []);

  useEffect(() => {
    if (trace) sceneRef.current?.setTrace(trace);
  }, [trace]);

  // The scene takes its colours from the app's, in light and in dark.
  useEffect(() => {
    const media = window.matchMedia("(prefers-color-scheme: dark)");
    const update = () => host.current && sceneRef.current?.setPalette(readPalette(host.current));
    update();
    media.addEventListener("change", update);
    return () => media.removeEventListener("change", update);
  }, [appearance]);

  return (
    <section className="stage viewer-stage" aria-label={t("The chip at work")}>
      <div ref={host} className="viewer-canvas" />
      <div ref={labels} className="viewer-labels" aria-hidden="true">
        {trace?.dies.map((_, i) => <DieLabel key={i} die={i} trace={trace} />)}
      </div>
      {failed && (
        <div className="viewer-empty">
          <p>{t("This browser cannot draw in 3D here.")}</p>
          <p className="muted-text">{failed}</p>
        </div>
      )}
      {!trace && !failed && (
        <div className="viewer-empty">
          <p>{t("Record a run of a small model on its dies, or open a trace, to watch the chip at work.")}</p>
        </div>
      )}
    </section>
  );
}

function DieLabel({ die, trace }: { die: number; trace: Trace }) {
  const t = useT();
  const cycle = useViewer((s) => Math.floor(s.cycle));
  const state = dieAt(trace, die, cycle);
  const group = trace.dies[die].group;
  const what = state.halted ? t("Finished") : state.stalled ? t("Waiting on a link") : (state.op?.name ?? t("Between operations"));
  const diverged = divergedAt(trace, die, cycle);
  return (
    <div className={`viewer-label${state.stalled || diverged ? " stalled" : ""}`}>
      <strong>{t(`Die ${die}`)}</strong>
      {group && <span className="muted-text"> · {t(`group ${group[0]}`)}</span>}
      <div>{what}</div>
      {diverged && <div>{t(`Differs from the exact machine since ${trace.ops[diverged.op].name}`)}</div>}
    </div>
  );
}

const STRIDES: [Stride, string][] = [
  ["cycle", "Cycle"],
  ["word", "Word"],
  ["op", "Operation"],
  ["layer", "Layer"],
  ["step", "Token"],
];

const SPEEDS = [1, 10, 100, 1_000, 10_000, 100_000];

function ChipPanel() {
  const t = useT();
  const trace = useViewer((s) => s.trace);
  const cycle = useViewer((s) => Math.floor(s.cycle));
  const playing = useViewer((s) => s.playing);
  const speed = useViewer((s) => s.speed);
  const stride = useViewer((s) => s.stride);
  const selected = useViewer((s) => s.selected);
  const error = useViewer((s) => s.error);
  const { setPlaying, setSpeed, setStride, step, setCycle, open } = useViewer.getState();
  const fileInput = useRef<HTMLInputElement>(null);

  const onFile = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) void open(file);
    e.target.value = "";
  };

  const total = trace ? totalCycles(trace) : 0;
  const current = trace ? trace.steps[stepAt(trace, cycle)] : null;
  return (
    <aside className="inspector essential">
      <h2 className="detail-title">{t("The chip at work")}</h2>
      <p className="detail-sub">{trace ? t(trace.title) : t("Every cycle comes from the gates of the dies, as they ran.")}</p>

      <Recorder />
      <div className="sim-actions">
        <button className="pill" onClick={() => fileInput.current?.click()}>{t("Open a trace")}</button>
        {trace && <button className="pill" onClick={() => saveTrace(trace)}>{t("Save the trace")}</button>}
        <input ref={fileInput} type="file" accept=".json,application/json" hidden onChange={onFile} />
      </div>
      {error && <div className="issue error" role="status">{error}</div>}

      {trace && current && (
        <>
          <h3>{t("Playback")}</h3>
          <div className="sim-actions">
            <button className="pill" onClick={() => step(-1)} aria-label={t("Step back")}>◀︎</button>
            <button className="pill primary-pill" onClick={() => setPlaying(!playing)}>{playing ? t("Pause") : t("Play")}</button>
            <button className="pill" onClick={() => step(1)} aria-label={t("Step forward")}>▶︎</button>
          </div>
          <div className="scope viewer-strides">
            <Segmented label={t("Step by")} value={stride} options={STRIDES.map(([id, label]): [Stride, string] => [id, t(label)])} onChange={setStride} />
          </div>
          <label className="viewer-row">
            <span>{t("Speed")}</span>
            <input type="range" min={0} max={SPEEDS.length - 1} step={1} value={Math.max(0, SPEEDS.findIndex((s) => s >= speed))} onChange={(e) => setSpeed(SPEEDS[Number(e.target.value)])} />
            <span className="muted-text">{t(`${speed.toLocaleString()} cycles a second`)}</span>
          </label>
          <label className="viewer-row">
            <span>{t("Cycle")}</span>
            <input type="range" min={0} max={total - 1} value={cycle} onChange={(e) => setCycle(Number(e.target.value))} />
          </label>
          <Timeline trace={trace} cycle={cycle} />
          <dl className="inset metrics">
            <div><dt>{t("Cycle")}</dt><dd>{cycle.toLocaleString()} / {total.toLocaleString()}</dd></div>
            <div><dt>{current.routine === "token" ? t("Token") : t("Logits of the token")}</dt><dd>{t(`${current.token} at position ${current.position}`)}</dd></div>
          </dl>

          <Breaks trace={trace} />
          <Divergences trace={trace} />

          <h3>{t("Each die")}</h3>
          <table className="viewer-dies">
            <tbody>
              {trace.dies.map((_, i) => <DieRow key={i} trace={trace} die={i} cycle={cycle} selected={selected === i} />)}
            </tbody>
          </table>
          {selected !== null && <DieDetail trace={trace} die={selected} cycle={cycle} />}

          <h3>{t("Colours")}</h3>
          <ul className="viewer-legend">
            {(Object.keys(CLASS_LABEL) as WordClass[]).filter((c) => c !== "take").map((c) => (
              <li key={c}><span className="swatch" style={{ background: CLASS_COLOR[c] }} />{t(CLASS_LABEL[c])}</li>
            ))}
            <li><span className="swatch" style={{ background: "var(--red)" }} />{t("Waiting on a link")}</li>
            <li><span className="swatch" style={{ background: "var(--cat-vector)" }} />{t("A token in the KV cache")}</li>
          </ul>
        </>
      )}
    </aside>
  );
}

function Breaks({ trace }: { trace: Trace }) {
  const t = useT();
  const breakpoints = useViewer((s) => s.breakpoints);
  const stopped = useViewer((s) => s.stopped);
  const { setBreakpoints, runToBreak } = useViewer.getState();
  const layers = Array.from({ length: trace.layers }, (_, i) => i);
  return (
    <>
      <h3>{t("Stop at")}</h3>
      <div className="inset padded">
        <label className="viewer-check">
          <input type="checkbox" checked={breakpoints.stall} onChange={(e) => setBreakpoints({ ...breakpoints, stall: e.target.checked })} />
          {t("A die starting to wait on a link")}
        </label>
        <label className="viewer-check">
          <input type="checkbox" checked={breakpoints.divergence} onChange={(e) => setBreakpoints({ ...breakpoints, divergence: e.target.checked })} />
          {t("A result that differs from the exact machine")}
        </label>
        <label className="viewer-row">
          <span>{t("Operation")}</span>
          <select value={breakpoints.kind ?? ""} onChange={(e) => setBreakpoints({ ...breakpoints, kind: e.target.value || null })}>
            <option value="">{t("Any")}</option>
            {opKinds(trace).map((k) => <option key={k} value={k}>{k}</option>)}
          </select>
        </label>
        <label className="viewer-row">
          <span>{t("Layer")}</span>
          <select value={breakpoints.layer ?? ""} onChange={(e) => setBreakpoints({ ...breakpoints, layer: e.target.value === "" ? null : Number(e.target.value) })}>
            <option value="">{t("Any")}</option>
            {layers.map((l) => <option key={l} value={l}>{l}</option>)}
          </select>
        </label>
        <div className="sim-actions">
          <button className="pill" onClick={runToBreak}>{t("Run to the next stop")}</button>
        </div>
        {stopped && <p className="muted-text">{t(stopped)}</p>}
      </div>
    </>
  );
}

function Divergences({ trace }: { trace: Trace }) {
  const t = useT();
  const { setCycle, select } = useViewer.getState();
  return (
    <>
      <h3>{t("Against the exact machine")}</h3>
      {trace.divergences.length === 0 ? (
        <div className="issue ok" role="status">{t("Every die agrees with the exact machine, bit for bit.")}</div>
      ) : (
        <ul className="viewer-divergences">
          {trace.divergences.map((d, i) => {
            const step = trace.steps[d.step];
            return (
              <li key={i}>
                <button className="link" onClick={() => { setCycle(d.cycle); select(d.die); }}>
                  {t(`Die ${d.die}: ${trace.ops[d.op].name}`)}
                </button>
                <span className="muted-text"> · {t(`element ${d.index}: ${d.got} instead of ${d.want}`)} · {step.routine === "token" ? t(`token at position ${step.position}`) : t("the logits")}</span>
              </li>
            );
          })}
        </ul>
      )}
    </>
  );
}

function Recorder() {
  const t = useT();
  const recording = useViewer((s) => s.recording);
  const record = useViewer((s) => s.record);
  const [request, setRequest] = useState<RecordRequest>(DEFAULT_RECORD);
  const running = recording.state === "running";
  return (
    <div className="inset padded">
      <label className="viewer-row">
        <span>{t("Number of dies")}</span>
        <select value={request.chips} onChange={(e) => setRequest({ ...request, chips: Number(e.target.value) })}>
          {[1, 2, 4, 6].map((n) => <option key={n} value={n}>{n}</option>)}
        </select>
      </label>
      <label className="viewer-row">
        <span>{t("Split")}</span>
        <select value={request.sharing} onChange={(e) => setRequest({ ...request, sharing: e.target.value })}>
          <option value="layers">{t("By layers")}</option>
          <option value="rows">{t("By rows")}</option>
          <option value="mixed:2">{t("Groups of two")}</option>
        </select>
      </label>
      <label className="viewer-row">
        <span>{t("Model")}</span>
        <select value={request.model} onChange={(e) => setRequest({ ...request, model: e.target.value as RecordRequest["model"] })}>
          <option value="grouped">{t("Six heads on two key heads")}</option>
          <option value="plain">{t("Two heads on one key head")}</option>
        </select>
      </label>
      <div className="sim-actions">
        <button className="pill primary-pill" disabled={running} onClick={() => void record(request)}>{running ? t("Recording…") : t("Record on the gates")}</button>
      </div>
      {recording.state === "error" && <div className="issue error" role="status">{recording.error}</div>}
    </div>
  );
}

function DieRow({ trace, die, cycle, selected }: { trace: Trace; die: number; cycle: number; selected: boolean }) {
  const t = useT();
  const state = dieAt(trace, die, cycle);
  const colour = state.halted ? "var(--neutral-fill)" : state.stalled ? "var(--red)" : CLASS_COLOR[state.word?.class ?? "halt"];
  const what = state.halted ? t("Finished") : state.stalled ? t("Waiting on a link") : (state.op?.name ?? t("Between operations"));
  const layer = state.op?.layer;
  return (
    <tr className={selected ? "on" : ""} onClick={() => useViewer.getState().select(selected ? null : die)}>
      <td><span className="swatch" style={{ background: colour }} /></td>
      <td>{die}</td>
      <td>{what}</td>
      <td className="muted-text">{layer !== null && layer !== undefined ? t(`layer ${layer}`) : ""}</td>
    </tr>
  );
}

function DieDetail({ trace, die, cycle }: { trace: Trace; die: number; cycle: number }) {
  const t = useT();
  const state = dieAt(trace, die, cycle);
  const d = trace.dies[die];
  const totals = dieTotals(trace, die);
  const all = Math.max(d.pcs.length, 1);
  const percent = (n: number) => `${Math.round((100 * n) / all)} %`;
  return (
    <div className="inset padded">
      <h3>{t(`Die ${die}`)}</h3>
      <dl className="metrics">
        <div><dt>{t("Units")}</dt><dd>{t(`${d.units} of ${d.lanes} lanes`)}</dd></div>
        <div><dt>{t("Cells")}</dt><dd>{d.cells}</dd></div>
        <div><dt>{t("Working")}</dt><dd>{percent(totals.work)}</dd></div>
        <div><dt>{t("Waiting on a link")}</dt><dd>{percent(totals.stalled)}</dd></div>
        <div><dt>{t("Finished, waiting for the others")}</dt><dd>{percent(totals.halted)}</dd></div>
      </dl>
      {state.word && (
        <>
          <p className="muted-text">{t(`Word ${state.pc} of the microprogram`)}</p>
          <p>{t(state.word.text)}</p>
        </>
      )}
    </div>
  );
}

/** The run as a strip: each token's steps, and where any die waited on a link. */
function Timeline({ trace, cycle }: { trace: Trace; cycle: number }) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const total = totalCycles(trace);
  useEffect(() => {
    const el = canvas.current;
    if (!el) return;
    const ratio = window.devicePixelRatio || 1;
    const width = el.clientWidth * ratio;
    const height = el.clientHeight * ratio;
    el.width = width;
    el.height = height;
    const g = el.getContext("2d");
    if (!g) return;
    const style = getComputedStyle(el);
    const rows = trace.dies.length;
    const row = height / rows;
    g.clearRect(0, 0, width, height);
    for (let die = 0; die < rows; die++) {
      g.fillStyle = style.getPropertyValue("--cat-compute").trim() || "#30a46c";
      g.globalAlpha = 0.35;
      g.fillRect(0, die * row + 1, width, row - 2);
      g.globalAlpha = 1;
      g.fillStyle = style.getPropertyValue("--neutral-fill").trim() || "#8e8e93";
      const pcs = trace.dies[die].pcs;
      for (let x = 0; x < width; x++) {
        const at = Math.floor((x / width) * total);
        if (pcs[at] < 0) g.fillRect(x, die * row + 1, 1, row - 2);
      }
      g.fillStyle = style.getPropertyValue("--red").trim() || "#ff3b30";
      for (const [start, length] of trace.dies[die].stalls) {
        g.fillRect((start / total) * width, die * row + 1, Math.max(1, (length / total) * width), row - 2);
      }
    }
    g.fillStyle = style.getPropertyValue("--label").trim() || "#000";
    for (const s of trace.steps) g.fillRect((s.start / total) * width, 0, ratio, height);
  }, [trace, total]);
  return (
    <div className="viewer-timeline">
      <canvas ref={canvas} />
      <div className="viewer-cursor" style={{ left: `${(100 * cycle) / Math.max(total - 1, 1)}%` }} />
    </div>
  );
}

function saveTrace(trace: Trace) {
  download(JSON.stringify(trace), "llm2chip-trace.json", "application/json");
}
