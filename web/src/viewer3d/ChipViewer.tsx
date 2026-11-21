import { useEffect, useRef, useState, type ChangeEvent, type ReactNode } from "react";
import { useT } from "../i18n";
import { useDesign } from "../store";
import { Segmented } from "../components/controls";
import { download } from "../download";
import { ChipScene, readPalette } from "./scene";
import { DEFAULT_RECORD, designRequest, useViewer, type RecordRequest } from "./viewerStore";
import { InsideDie } from "./InsideDie";
import { useInside } from "./inside";
import { operationName } from "../operations";
import { byOperations, compareTraces, cyclesByKind, dieAt, dieShares, divergedAt, opKinds, useShare, stepAt, totalCycles, type Stride, type Trace, type WordClass } from "./trace";

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

/** The scene on show, for recording it. */
let shownScene: ChipScene | null = null;

/** Opens die `die` to its gates, once the camera has flown down onto it. */
function lookInside(die: number) {
  useViewer.getState().select(die);
  useInside.getState().enter(die);
}

const CLASS_LABEL: Record<WordClass, string> = {
  cells: "Cells multiply and add",
  vector: "Every lane computes",
  scalar: "One element at a time",
  send: "Sends to the next die",
  take: "Takes from the previous die",
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
    shownScene = scene;
    let last = performance.now();
    // The heat map is worked out again only when the cycle on show changes.
    let heat: { trace: Trace; cycle: number; shares: number[] } | null = null;
    scene.setTick(() => {
      const now = performance.now();
      const viewer = useViewer.getState();
      viewer.advance(Math.min((now - last) / 1000, 0.1));
      last = now;
      const { trace, cycle, selected, speed, playing, colouring } = useViewer.getState();
      let shares: number[] | null = null;
      if (trace && colouring === "heat") {
        if (!heat || heat.trace !== trace || heat.cycle !== Math.floor(cycle)) heat = { trace, cycle: Math.floor(cycle), shares: useShare(trace, cycle) };
        shares = heat.shares;
      }
      // Words cross a link in a cycle: show those of the last few frames' worth of cycles.
      scene.show(cycle, selected, playing ? Math.max(16, speed / 4) : 16, shares);
    });
    scene.onPick = (die) => useViewer.getState().select(die);
    scene.onOpen = (die) => lookInside(die);
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
      if (shownScene === scene) shownScene = null;
    };
  }, []);

  const touring = useViewer((s) => s.touring);
  useEffect(() => sceneRef.current?.setTour(touring), [touring]);
  // Looking inside a die flies the camera to it; leaving flies it back.
  const inside = useInside((s) => s.die);
  // The schematic comes once the camera has reached the die.
  const [arrived, setArrived] = useState(false);
  useEffect(() => {
    sceneRef.current?.focus(inside);
    setArrived(false);
    if (inside === null) return;
    const timer = setTimeout(() => setArrived(true), 650);
    return () => clearTimeout(timer);
  }, [inside]);
  // A new trace closes whatever die was open.
  useEffect(() => useInside.getState().reset(), [trace]);

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
      {touring && trace && <Caption trace={trace} />}
      {inside !== null && arrived && trace && <InsideDie trace={trace} />}
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

/** For a presentation: where the run is and what the dies are doing, in large type. */
function Caption({ trace }: { trace: Trace }) {
  const t = useT();
  const cycle = useViewer((s) => Math.floor(s.cycle));
  const step = trace.steps[stepAt(trace, cycle)];
  const states = trace.dies.map((_, d) => dieAt(trace, d, cycle));
  const working = states.filter((s) => !s.halted && !s.stalled);
  const waiting = states.filter((s) => s.stalled).length;
  const layers = [...new Set(working.map((s) => s.op?.layer).filter((l): l is number => l !== null && l !== undefined))];
  // Traced by operations, no real token goes through: only where in the context it is.
  const token = byOperations(trace) ? t(`Token at position ${step.position}`) : t(`Token ${step.token} at position ${step.position}`);
  const where = step.routine === "token" ? token : t("The logits");
  return (
    <div className="viewer-caption">
      <strong>{where}</strong>
      <span>{layers.length > 0 ? t(`${working.length} of ${trace.dies.length} dies working, on layer ${layers.join(", ")}`) : t(`${working.length} of ${trace.dies.length} dies working`)}</span>
      {waiting > 0 && <span>{t(`${waiting} waiting`)}</span>}
    </div>
  );
}

function DieLabel({ die, trace }: { die: number; trace: Trace }) {
  const t = useT();
  const cycle = useViewer((s) => Math.floor(s.cycle));
  const state = dieAt(trace, die, cycle);
  const group = trace.dies[die].group;
  const what = describe(trace, state, t);
  const diverged = divergedAt(trace, die, cycle);
  return (
    <div className={`viewer-label${state.stalled || diverged ? " stalled" : ""}`}>
      <strong>{t(`Die ${die}`)}</strong>
      {group && <span className="muted-text"> · {t(`group ${group[0]}`)}</span>}
      <div>{what}</div>
      {diverged && <div>{t(`Differs from the exact reference since ${trace.ops[diverged.op].name}`)}</div>}
    </div>
  );
}

/** What a die is doing, in a few words. Traced by operations, a die that does not run
 * waits for its turn in the chain, and one that waits does so for its group's slowest. */
function describe(trace: Trace, state: ReturnType<typeof dieAt>, t: (text: string) => string): string {
  const timing = byOperations(trace);
  if (state.halted) return timing ? t("Waiting for its turn") : t("Finished");
  if (state.stalled) return timing ? t("Waiting for its group") : t("Waiting on a link");
  return state.op ? t(state.op.name) : t("Between operations");
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
      <p className="detail-sub">{trace ? t(trace.title) : t("Every cycle comes from the logic gates of the dies, as they ran.")}</p>

      <Recorder />
      <div className="sim-actions">
        <button className="pill" onClick={() => fileInput.current?.click()}>{t("Open a trace")}</button>
        {trace && <button className="pill" onClick={() => saveTrace(trace)}>{t("Save the trace")}</button>}
        <input ref={fileInput} type="file" accept=".json,application/json" hidden onChange={onFile} />
      </div>
      {error && <Issue kind="error">{t(error)}</Issue>}

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
            <div><dt>{current.routine === "token" ? t("Token") : t("Logits of the token")}</dt><dd>{byOperations(trace) ? t(`Position ${current.position}`) : t(`${current.token} at position ${current.position}`)}</dd></div>
          </dl>

          <Breaks trace={trace} />
          <Divergences trace={trace} />
          <Uses trace={trace} />
          <Compare trace={trace} />
          <Presentation />

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

function Uses({ trace }: { trace: Trace }) {
  const t = useT();
  const colouring = useViewer((s) => s.colouring);
  const selected = useViewer((s) => s.selected);
  const { setColouring } = useViewer.getState();
  const die = selected ?? 0;
  const by = [...cyclesByKind(trace, die).entries()].sort((a, b) => b[1] - a[1]);
  const total = Math.max(1, by.reduce((sum, [, n]) => sum + n, 0));
  const name = (kind: string) => (kind === "idle" ? t("Idle") : kind === "waiting" ? t("Waiting") : kind === "other" ? t("Between operations") : t(operationName(kind)));
  return (
    <>
      <h3>{t("Where the cycles go")}</h3>
      <div className="scope">
        <Segmented label={t("Colour the lanes by")} value={colouring} options={[["now", t("What they do now")], ["heat", t("Use so far")]]} onChange={setColouring} />
      </div>
      <p className="muted-text">{t(`Die ${die}, the whole run`)}</p>
      <ul className="viewer-bars">
        {by.slice(0, 10).map(([kind, n]) => (
          <li key={kind}>
            <span>{name(kind)}</span>
            <span className="bar"><span style={{ width: `${(100 * n) / total}%` }} /></span>
            <span className="muted-text">{Math.round((100 * n) / total)} %</span>
          </li>
        ))}
      </ul>
    </>
  );
}

/** A routine of a trace, for a person: the token or the logits, and where in the context. */
function stepLabel(label: string, t: (text: string) => string): string {
  const [routine, position] = label.split(" ");
  return routine === "token" ? t(`Token at position ${position}`) : t(`Logits at position ${position}`);
}

function Compare({ trace }: { trace: Trace }) {
  const t = useT();
  const other = useViewer((s) => s.other);
  const { openOther } = useViewer.getState();
  const input = useRef<HTMLInputElement>(null);
  const c = other ? compareTraces(trace, other) : null;
  const percent = (v: number) => `${Math.round(100 * v)} %`;
  return (
    <>
      <h3>{t("Compare")}</h3>
      <div className="sim-actions">
        <button className="pill" onClick={() => input.current?.click()}>{other ? t("Another trace") : t("Open a trace to compare")}</button>
        {other && <button className="pill" onClick={() => void openOther(null)}>{t("Stop comparing")}</button>}
        <input ref={input} type="file" accept=".json,application/json" hidden onChange={(e) => { void openOther(e.target.files?.[0] ?? null); e.target.value = ""; }} />
      </div>
      {other && c && (
        <table className="viewer-compare">
          <thead>
            <tr><th /><th>{t("This trace")}</th><th>{t(other.title)}</th></tr>
          </thead>
          <tbody>
            {c.steps.map((s) => (
              <tr key={s.label}><td>{stepLabel(s.label, t)}</td><td>{s.a?.toLocaleString() ?? "–"}</td><td>{s.b?.toLocaleString() ?? "–"}</td></tr>
            ))}
            <tr><td>{t("Busy")}</td><td>{percent(c.share.a[0])}</td><td>{percent(c.share.b[0])}</td></tr>
            <tr><td>{t("Waiting")}</td><td>{percent(c.share.a[1])}</td><td>{percent(c.share.b[1])}</td></tr>
            <tr><td>{t("Idle")}</td><td>{percent(c.share.a[2])}</td><td>{percent(c.share.b[2])}</td></tr>
          </tbody>
        </table>
      )}
    </>
  );
}

function Presentation() {
  const t = useT();
  const touring = useViewer((s) => s.touring);
  const { setTouring } = useViewer.getState();
  const [recorder, setRecorder] = useState<MediaRecorder | null>(null);
  const [failure, setFailure] = useState<string | null>(null);
  const record = () => {
    if (recorder) {
      recorder.stop();
      return;
    }
    const canvas = shownScene?.canvas();
    if (!canvas || typeof MediaRecorder === "undefined") return setFailure(t("This browser cannot record the view."));
    const type = ["video/webm;codecs=vp9", "video/webm", "video/mp4"].find((m) => MediaRecorder.isTypeSupported(m));
    if (!type) return setFailure(t("This browser cannot record the view."));
    const chunks: Blob[] = [];
    const r = new MediaRecorder(canvas.captureStream(30), { mimeType: type });
    r.ondataavailable = (e) => chunks.push(e.data);
    r.onstop = () => {
      download(new Blob(chunks, { type }), type.startsWith("video/mp4") ? "llm2chip-chip.mp4" : "llm2chip-chip.webm", type);
      setRecorder(null);
    };
    r.start();
    setFailure(null);
    setRecorder(r);
  };
  return (
    <>
      <h3>{t("Presentation")}</h3>
      <div className="sim-actions">
        <button className={touring ? "pill primary-pill" : "pill"} onClick={() => setTouring(!touring)}>{touring ? t("End the tour") : t("Guided tour")}</button>
        <button className={recorder ? "pill primary-pill" : "pill"} onClick={record}>{recorder ? t("Stop and save the video") : t("Record a video")}</button>
      </div>
      <p className="muted-text">{t("The tour turns the camera round the dies and says what each is doing; the video records the view as it plays.")}</p>
      {failure && <Issue kind="error">{failure}</Issue>}
    </>
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
      <h3>{t("Stop when")}</h3>
      <div className="inset padded">
        <label className="viewer-check">
          <input type="checkbox" checked={breakpoints.stall} onChange={(e) => setBreakpoints({ ...breakpoints, stall: e.target.checked })} />
          {t("A die starts waiting on a link")}
        </label>
        <label className="viewer-check">
          <input type="checkbox" checked={breakpoints.divergence} onChange={(e) => setBreakpoints({ ...breakpoints, divergence: e.target.checked })} />
          {t("A result differs from the exact reference")}
        </label>
        <label className="viewer-row">
          <span>{t("An operation starts")}</span>
          <select value={breakpoints.kind ?? ""} onChange={(e) => setBreakpoints({ ...breakpoints, kind: e.target.value || null })}>
            <option value="">{t("Don't stop")}</option>
            {opKinds(trace).map((k) => <option key={k} value={k}>{t(operationName(k))}</option>)}
          </select>
        </label>
        <label className="viewer-row">
          <span>{t("A layer starts")}</span>
          <select value={breakpoints.layer ?? ""} onChange={(e) => setBreakpoints({ ...breakpoints, layer: e.target.value === "" ? null : Number(e.target.value) })}>
            <option value="">{t("Don't stop")}</option>
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
      <h3>{t("Against the exact reference")}</h3>
      {byOperations(trace) ? (
        <p className="muted-text">{t("A trace by operations is not checked against the exact reference: for that, simulate the small model on its logic gates.")}</p>
      ) : trace.divergences.length === 0 ? (
        <Issue kind="ok">{t("Every die agrees bit for bit with the exact reference.")}</Issue>
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

type Source = "gates" | "design";

function Recorder() {
  const t = useT();
  const recording = useViewer((s) => s.recording);
  const record = useViewer((s) => s.record);
  const model = useDesign((s) => s.model);
  const params = useDesign((s) => s.params);
  const [source, setSource] = useState<Source>("gates");
  const [request, setRequest] = useState<RecordRequest>(DEFAULT_RECORD);
  const running = recording.state === "running";
  const design = designRequest(model?.path, params);
  return (
    <div className="inset padded">
      <div className="scope">
        <Segmented label={t("What to trace")} value={source} options={[["gates", t("Small model")], ["design", t("This design")]]} onChange={setSource} />
      </div>
      {source === "gates" ? (
        <>
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
            <span>{t("Query and key heads")}</span>
            <select value={request.model} onChange={(e) => setRequest({ ...request, model: e.target.value as RecordRequest["model"] })}>
              <option value="grouped">{t("6 and 2")}</option>
              <option value="plain">{t("2 and 1")}</option>
            </select>
          </label>
          <p className="muted-text">{t("Every cycle of every die, simulated on its logic gates and checked against the exact reference after each token.")}</p>
          <div className="sim-actions">
            <button className="pill primary-pill" disabled={running} onClick={() => void record(request)}>{running ? t("Simulating…") : t("Simulate and record")}</button>
          </div>
        </>
      ) : (
        <>
          <p className="muted-text">{t("The model and dies of the sidebar, operation by operation from each chip's own microprogram: a token at the start, the middle and the end of the context. Links take no time here, as in the sizing. A large model takes a minute or so.")}</p>
          {typeof design === "string" ? (
            <Issue kind="error">{t(design)}</Issue>
          ) : (
            <div className="sim-actions">
              <button className="pill primary-pill" disabled={running} onClick={() => void record(design)}>{running ? t("Tracing…") : t("Trace this design")}</button>
            </div>
          )}
        </>
      )}
      {recording.state === "error" && <Issue kind="error">{t(recording.error)}</Issue>}
    </div>
  );
}

function DieRow({ trace, die, cycle, selected }: { trace: Trace; die: number; cycle: number; selected: boolean }) {
  const t = useT();
  const state = dieAt(trace, die, cycle);
  const colour = state.halted ? "var(--neutral-fill)" : state.stalled ? "var(--red)" : CLASS_COLOR[state.word?.class ?? "halt"];
  const what = describe(trace, state, t);
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
  const shares = dieShares(trace, die);
  const percent = (n: number) => `${Math.round(100 * n)} %`;
  return (
    <div className="inset padded">
      <h3>{t(`Die ${die}`)}</h3>
      <dl className="metrics">
        <div><dt>{t("Vector units")}</dt><dd>{t(`${d.units} of ${d.lanes} lanes each`)}</dd></div>
        <div><dt>{t("Cells")}</dt><dd>{d.cells}</dd></div>
        <div><dt>{t("Busy")}</dt><dd>{percent(shares.work)}</dd></div>
        <div><dt>{byOperations(trace) ? t("Waiting for its group") : t("Waiting on a link")}</dt><dd>{percent(shares.stalled)}</dd></div>
        <div><dt>{byOperations(trace) ? t("Waiting for its turn") : t("Finished, waiting for the others")}</dt><dd>{percent(shares.halted)}</dd></div>
      </dl>
      {!byOperations(trace) && (
        <div className="sim-actions">
          <button className="pill" onClick={() => lookInside(die)}>{t("Look inside, down to the logic gates")}</button>
        </div>
      )}
      {state.word && (byOperations(trace) && state.op ? (
        // Traced by operations, each "word" is a whole operation of the program.
        <>
          <p className="muted-text">{t(`Operation ${state.pc} of the program`)}</p>
          <p>{t(state.op.name)} ({t(operationName(state.op.kind))})</p>
        </>
      ) : (
        <>
          <p className="muted-text">{t(`Word ${state.pc} of the microprogram`)}</p>
          <p>{t(state.word.text)}</p>
        </>
      ))}
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
      if (pcs.length > 0) {
        for (let x = 0; x < width; x++) {
          const at = Math.floor((x / width) * total);
          if (pcs[at] < 0) g.fillRect(x, die * row + 1, 1, row - 2);
        }
      } else {
        // Traced by operations: idle except where it runs one.
        g.fillRect(0, die * row + 1, width, row - 2);
        g.fillStyle = style.getPropertyValue("--cat-compute").trim() || "#30a46c";
        for (const [start, length] of trace.dies[die].spans) g.fillRect((start / total) * width, die * row + 1, Math.max(1, (length / total) * width), row - 2);
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

function Issue({ kind, children }: { kind: "ok" | "error"; children: ReactNode }) {
  return (
    <div className={`issue ${kind}`} role="status">
      <span className="issue-icon" aria-hidden="true">{kind === "ok" ? "✓" : "!"}</span>
      <span>{children}</span>
    </div>
  );
}

function saveTrace(trace: Trace) {
  download(JSON.stringify(trace), "llm2chip-trace.json", "application/json");
}
