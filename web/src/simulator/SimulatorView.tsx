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
  matvec: "Matrix × vector",
  vector: "Vector units",
  embedding: "Embedding",
  link: "Die-to-die link",
};
const KIND_CLASS: Record<ScheduledStage["kind"], string> = {
  matvec: "tl-matvec",
  vector: "tl-vector",
  embedding: "tl-embedding",
  link: "tl-link",
};

/** What each kind of stage does in the chip, for the stage the reader points at. */
const KIND_HELP: Record<ScheduledStage["kind"], string> = {
  matvec: "The die's cells multiply this matrix by the activation vector: each cell reads its weights from its SRAM, multiplies them by the value the activation network brings it, and the adder tree gathers each row's products.",
  vector: "The vector units work element by element, lane by lane: norms, rotations, table functions, attention over the cache or the delta rule over its state.",
  embedding: "The token's row is read from the embedding table: it is the vector the layers start from.",
  link: "The hidden vector goes to the next die over the link, and the layers carry on there.",
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
        <strong>{t("A token's path")}</strong>
        <span>{formatNumber(schedule.total_cycles)} {t("cycles")} · {formatNumber(schedule.token_ns / 1000)} µs</span>
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
                    aria-label={`${t(s.name)}${s.layer !== null ? `, ${t("layer")} ${s.layer}` : ""}, ${formatNumber(Math.round(s.cycles))} ${t("cycles")}`}
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
              <strong>{t(shown.name)}</strong>
              {shown.layer !== null ? ` · ${t("layer")} ${shown.layer}` : ""} · die {shown.die} · {formatNumber(Math.round(shown.cycles))} {t("cycles")} ({percent(shown.cycles / schedule.total_cycles)} % {t("of the token")}, {formatNumber(ns(shown.cycles))} ns) · {t("starts at cycle")} {formatNumber(Math.round(shown.start))}
            </p>
            <p className="muted-text">{t(KIND_HELP[shown.kind])}</p>
          </>
        ) : (
          <p className="muted-text">{t("Hover over a stretch, or click it to pin it, to see what the die does in it.")}</p>
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
              <img src={picture.data} width={picture.side} height={picture.side} alt={`${t("Generated image")}: ${prompt}`} />
              <figcaption>{picture.side} × {picture.side} px, in {formatNumber(picture.seconds)} s</figcaption>
            </figure>
          ) : output ? <p>{output}</p> : <p className="placeholder">{loaded ? (draw ? t("The image will appear here.") : t("The model's answer will appear here.")) : t("Load the chip in the right panel to simulate it: with the design's model and arithmetic.")}</p>}
          {!running && ending && !draw && (
            <p className="note-warning">
              {ending === "loop" ? t("The model started repeating itself without writing its end of turn and the chip stopped it, as its firmware does.") : ending === "context" ? t("The chip's context filled up: it remembers no more tokens. Raise the context in the design for longer answers.") : t("The model did not write its end of turn: it stopped at the token limit.")}
            </p>
          )}
          {running && drawing && (
            <p className="placeholder">
              {drawing.decoding ? t("The decoder turns the tokens into pixels…") : `${t("The model writes the image")}: ${drawing.done} / ${drawing.total} ${t("visual tokens")} (${formatNumber(drawing.seconds)} s)`}
            </p>
          )}
          {running && phase === "Reading the prompt" && <p className="placeholder">{t("Reading the prompt")}: {prefilled} / {promptTokens} tokens</p>}
        </div>
        {error && (
          <div className="issue error" role="alert">
            <span className="issue-icon" aria-hidden="true">!</span>
            <div><strong>{error}</strong></div>
          </div>
        )}
        {image && (
          <figure className="attached">
            <img src={image.preview} alt={t("Attached image")} />
            <figcaption>
              {image.width} × {image.height} px{imageReport ? ` · ${formatNumber(imageReport.patches)} ${t("patches")} → ${formatNumber(imageReport.tokens)} tokens, ${formatNumber(imageReport.seconds)} s${imageReport.onChip ? `, ${t("encoded by the chip with its arithmetic")}` : `, ${t("encoded by the host in floating point")}`}` : ""}
              <button className="link inline" onClick={() => set({ image: null, imageReport: null })}>{t("Remove")}</button>
            </figcaption>
          </figure>
        )}
        {makesImages && (
          <>
            <Segmented label={t("What the model does")} value={draw ? "draw" : "answer"} options={[["answer", t("Answer")], ["draw", t("Draw")]]} onChange={(mode) => set({ draw: mode === "draw" })} />
            {draw && <p className="muted-text">{t("With the chip's arithmetic, the language model and its image heads run on the die; the guidance mix, the sampling and the pixel decoder run on the host.")}</p>}
          </>
        )}
        <div className="composer">
          <label className={readsImages ? "pill attach" : "pill attach disabled"} title={readsImages ? t("Attach an image: the model's encoder turns it into tokens read before the question") : current?.images?.note ?? t("This model does not read images")}>
            {t("Image")}
            <input type="file" accept="image/*" disabled={!readsImages} onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) void readImage(file).then((picture) => set({ image: picture, imageReport: null }));
              e.target.value = "";
            }} />
          </label>
          <textarea
            aria-label={t("Input text")}
            rows={3}
            value={prompt}
            placeholder={t("Type a text")}
            onChange={(e) => set({ prompt: e.target.value })}
            onKeyDown={(e) => {
              if (e.key === "Enter" && (e.metaKey || e.ctrlKey) && loaded && !running) generate();
            }}
          />
          {running ? (
            <button className="pill" onClick={stop}>{t("Stop")}</button>
          ) : (
            <button className="pill primary-pill" disabled={!loaded || !prompt.trim()} onClick={generate}>{t("Send")}</button>
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
  const formatLabel = matchingPreset(params) ?? t("custom");
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
        <h2 className="detail-title">{t("Local engine")}</h2>
        <div className="issue warning" role="status">
          <span className="issue-icon" aria-hidden="true">i</span>
          <div>
            <strong>{t("The engine is not running.")}</strong>
            <p>{t("Start it with this command in the project folder")}: scripts/dev.sh</p>
          </div>
        </div>
      </aside>
    );
  }

  return (
    <aside className="inspector">
      <h2 className="detail-title">{t("Simulator")}</h2>
      <p className="detail-sub">{t("Runs the microprogram with real weights.")}</p>

      <h3>{t("The chip being simulated")}</h3>
      <div className="inset padded form">
        <p className="sim-model">
          <strong>{design?.name ?? t("No model")}</strong>
          <span className="muted-text">{t(`The design's model, with its arithmetic: ${String(params.activation_bits)}-bit integers and weights ${t(formatLabel)}.`)}</span>
        </p>
        {candidates.length === 0 ? (
          <>
            <p className="muted-text">{t("There are no weights of this model on this machine, so the chip cannot be simulated yet.")}</p>
            <button className="link-button" onClick={() => setView("models")}>
              {t("Download its weights…")}
            </button>
          </>
        ) : (
          <>
            {candidates.length > 1 && (
              <label>
                {t("Weights of this model")}
                <select value={state.path} onChange={(e) => set({ path: e.target.value })}>
                  {candidates.map((m) => (
                    <option key={m.path} value={m.path} disabled={!m.supported}>
                      {m.name}
                      {m.supported ? "" : ` (${t("not runnable")})`}
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
              {t("Use the GPU")}
            </label>
            <button className="pill primary-pill" disabled={state.loading || !selected?.supported} onClick={() => void load()}>
              {t(state.loading ? "Loading…" : state.loaded ? "Reload" : "Load the chip")}
            </button>
            {stale && <p className="note-warning">{t("The design changed since it was loaded: reload to simulate the current chip.")}</p>}
            {state.loaded && <p className="muted-text">{t("Loaded")}: {state.loaded} · {state.accelerator ?? "CPU"}</p>}
          </>
        )}
      </div>

      <h3>{t("What is simulated")}</h3>
      <div className="inset padded form">
        <Segmented
          label={t("Profile")}
          value={profileOf(state)}
          options={[["die", t("The die")], ["chip", t("The whole chip")], ["custom", t("Custom")]]}
          onChange={(profile) => {
            if (profile !== "custom") set(PROFILES[profile]);
          }}
        />
        {(
          [
            ["firmware", "Chip CPU and firmware", "The real firmware runs instruction by instruction and drives the die through its registers.", state.firmware, false],
            ["samplerUnit", "Chip sampling unit", "Picks each token with the same logic as the circuit, top-k included. Always used with the CPU.", state.samplerUnit || state.firmware, state.firmware],
            ["firmwareTokenizer", "Tokenizer in firmware", "The firmware turns the text into tokens and the tokens into bytes, and is compared with the reference library.", state.firmware && state.firmwareTokenizer, !state.firmware],
            ["transactionLoad", "Check the image load", "On loading, it goes through every block like the real controller, checks its CRC and counts the cycles. The chip must be reloaded.", state.transactionLoad, false],
          ] as const
        ).map(([key, label, help, checked, disabled]) => (
          <label className="check" key={key} title={t(help)}>
            <input type="checkbox" checked={checked} disabled={disabled} onChange={(e) => set({ [key]: e.target.checked })} />
            {t(label)}
            <span className="hint-icon" aria-label={t(help)}>
              <Icon name="info" size={13} />
            </span>
          </label>
        ))}
        <dl className="metrics">
          <div><dt>{t("Arithmetic")}</dt><dd>{String(params.activation_bits)}-bit {t("integers, the chip's")}</dd></div>
          <div><dt>{t("Tokenizer")}</dt><dd>{t(state.firmware && state.firmwareTokenizer ? "In the firmware" : "Outside the chip")}</dd></div>
          <div><dt>{t("Weight loading")}</dt><dd>{t(params.weight_backend === "rom" ? "Burnt into the chip (ROM)" : state.transactionLoad ? "By transactions" : "Straight from the checkpoint")}</dd></div>
        </dl>
      </div>

      <h3>{t("Generation")}</h3>
      <div className="inset padded form">
        <label>
          {t("Max tokens")}
          <input type="text" inputMode="numeric" value={state.maxTokens} onChange={(e) => set({ maxTokens: Number(e.target.value) || 1 })} />
        </label>
        <label>
          {t("Temperature (0 always picks the most likely)")}
          <input type="text" inputMode="decimal" value={state.temperature} onChange={(e) => set({ temperature: Number(e.target.value.replace(",", ".")) || 0 })} />
        </label>
        <label className="check" title={t("With the model's chat template, opens its reasoning block before the answer. Without it, the block arrives closed and the model answers directly. It is prompt text: it does not change the chip.")}>
          <input type="checkbox" checked={state.thinking} onChange={(e) => set({ thinking: e.target.checked })} />
          {t("Think before answering")}
        </label>
        {state.thinking && (
          <label title={t("How many tokens the model may reason before the chip's firmware closes the reasoning for it and moves to the answer. Empty means no limit; 0 means no reasoning.")}>
            {t("Reasoning budget (tokens, empty for no limit)")}
            <input type="text" inputMode="numeric" value={state.thinkBudget} placeholder={t("No limit")} onChange={(e) => set({ thinkBudget: e.target.value.replace(/[^0-9]/g, "") })} />
          </label>
        )}
        <p className="muted-text">{draftDepth > 0 ? `${t("The design's MTP head guesses tokens ahead")}: ${draftDepth}.` : t("The design has no MTP head.")}</p>
      </div>

      <h3>{t("Measurements")}</h3>
      <dl className="inset metrics">
        <div><dt>{t("Phase")}</dt><dd>{state.phase ? t(state.phase) : "—"}</dd></div>
        <div><dt>{t("Tokens generated")}</dt><dd>{state.tokens}</dd></div>
        <div><dt>{t("Simulation")}</dt><dd>{mean ? `${formatNumber(mean)} s/token` : "—"}</dd></div>
        <div><dt>{t("This chip, estimated")}</dt><dd>{formatNumber(chipRate)} tok/s</dd></div>
        <div><dt>{t("Faster than simulating it")}</dt><dd>{mean ? `×${formatNumber(chipRate * mean)}` : "—"}</dd></div>
        {state.loadReport && <div><dt>{t("Image: blocks")}</dt><dd>{formatNumber(state.loadReport.segments)}{state.loadReport.created ? ` (${t("created")})` : ""}</dd></div>}
        {state.loadReport && <div><dt>{t("Image: load cycles")}</dt><dd>{formatNumber(state.loadReport.cycles)}</dd></div>}
        {state.loadReport && <div><dt>{t("Image: CRC errors")}</dt><dd>{state.loadReport.crc_errors.length}</dd></div>}
        {state.chip && <div><dt>{t("Chip cycles, with firmware")}</dt><dd>{formatNumber(state.chip.cycles)}</dd></div>}
        {state.chip?.tokenizerMatch != null && <div><dt>{t("Tokenizer against the reference")}</dt><dd>{t(state.chip.tokenizerMatch ? "Matches" : "Differs")}</dd></div>}
        {state.chip && <div><dt>{t("Outside the die (CPU and sampling)")}</dt><dd>{formatNumber(state.chip.cycles - state.chip.dieCycles)} {t("cycles")}</dd></div>}
        {state.acceptance.tries.map((tries, depth) => (
          <div key={depth}><dt>{t("MTP hits at depth")} {depth + 1}</dt><dd>{tries ? `${formatNumber((state.acceptance.hits[depth] / tries) * 100)} %` : "—"}</dd></div>
        ))}
      </dl>
    </aside>
  );
}
