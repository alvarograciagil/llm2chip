import { EngineMissing } from "../components/EngineMissing";
import { useEffect } from "react";
import { formatNumber } from "../format";
import { useDesign } from "../store";
import { useT } from "../i18n";
import type { ServerModel } from "../simulator/simulatorStore";
import { useVoice, type Arithmetic, type SystemPlan } from "./voiceStore";

const PHASES: Record<string, string> = { listening: "Listening…", thinking: "Thinking of an answer…", speaking: "Speaking…" };

export function VoiceView() {
  const t = useT();
  const s = useVoice();
  const ready = s.question !== null && s.listen !== "" && s.think !== "" && s.speak !== "" && s.sound !== "" && !s.running && !s.recording;
  const stage = (label: string, text: string | undefined, seconds: number | undefined) => (
    <div className="voice-stage">
      <h3>{t(label)}</h3>
      <p className={text ? "" : "muted-text"}>{text ? `“${text}”` : " "}</p>
      {seconds !== undefined && <span className="muted-text">{formatNumber(seconds)} s</span>}
    </div>
  );
  return (
    <section className="stage sim-stage">
      <div className="images-column">
        <p className="muted-text">{t("A spoken turn through three programs, as three dies would run them: speech to text, an answer, text to speech. Only text passes from one to the next.")}</p>
        {s.history.length > 0 && (
          <ol className="voice-dies muted-text">
            {s.history.map((turn, i) => (
              <li key={i}>
                “{turn.heard}” → “{turn.answer}”
              </li>
            ))}
          </ol>
        )}
        {stage("1. Heard", s.result?.heard?.text, s.result?.heard?.seconds)}
        {stage("2. Answer", s.result?.answer?.text, s.result?.answer?.seconds)}
        <div className="voice-stage">
          <h3>{t("3. Said")}</h3>
          {s.result?.spoken ? (
            <>
              <audio controls autoPlay src={s.result.spoken.audio} />
              <span className="muted-text">
                {formatNumber(s.result.spoken.duration)} s {t("of speech")}, {s.result.spoken.tokens} {t("tokens")}, {formatNumber(s.result.spoken.seconds)} s
              </span>
            </>
          ) : (
            <p className="muted-text">{s.running ? t(PHASES[s.phase] ?? "Working…") : " "}</p>
          )}
        </div>
        {s.plan && <SystemPlanView plan={s.plan} />}
        {s.error && (
          <div className="issue error" role="alert">
            <span className="issue-icon" aria-hidden="true">!</span>
            <div><strong>{t(s.error)}</strong></div>
          </div>
        )}
        <div className="composer">
          {s.recording ? (
            <button className="pill primary-pill" onClick={() => s.stop()}>
              {t("Stop recording")}
            </button>
          ) : (
            <button className="pill" disabled={s.running} onClick={() => void s.record().catch(() => s.set({ error: "The microphone is not available." }))}>
              {t("Record")}
            </button>
          )}
          <label className={s.running ? "pill attach disabled" : "pill attach"} title={t("Open a recording instead: WAV, MP3 or anything the browser plays")}>
            {t("Open audio")}
            <input
              type="file"
              accept="audio/*"
              disabled={s.running}
              onChange={(e) => {
                const input = e.target;
                const file = input.files?.[0];
                // The file is read before the input lets it go, so the same one can be opened again.
                if (file)
                  void s
                    .open(file)
                    .catch(() => s.set({ error: "The file could not be read as audio." }))
                    .finally(() => (input.value = ""));
              }}
            />
          </label>
          <span className="muted-text">{s.question ? `${t("Question")}: ${formatNumber(s.question.seconds)} s` : t("Record a question or open one.")}</span>
          <button className="pill primary-pill" disabled={!ready} onClick={() => void s.start()}>
            {t(s.fresh ? "Ask" : "Ask again")}
          </button>
          {!s.fresh && !s.running && (
            <button className="pill" title={t("Forget the conversation: the next question starts a new one")} onClick={() => s.set({ fresh: true, history: [], result: null })}>
              {t("New conversation")}
            </button>
          )}
        </div>
      </div>
    </section>
  );
}

/** What the planner chose: the dies and their stages, each stage's timing, and the
 * groupings weighed with why the others were not chosen. */
function SystemPlanView({ plan }: { plan: SystemPlan }) {
  const t = useT();
  const chosen = plan.plan;
  return (
    <div className="voice-stage">
      <h3>{t("The chip, planned")}</h3>
      {chosen ? (
        <>
          <p>
            {chosen.dies.reduce((n, d) => n + d.count, 0)} {t("dies")}, {formatNumber(Math.round(chosen.area_mm2))} mm², {t("first sound after")} {formatNumber(Math.round(chosen.first_sound_ms))} ms
          </p>
          <ul className="voice-dies">
            {chosen.dies.map((d, i) => (
              <li key={i}>
                <strong>{d.stages.map((n) => t(n)).join(" + ")}</strong>: {d.count > 1 ? `${d.count} × ` : ""}
                {d.units} × {d.lanes}, {formatNumber(d.cells)} {t("cells")}, {formatNumber(Math.round(d.mm2))} mm², {formatNumber(Math.round(d.clock_mhz))} MHz
              </li>
            ))}
          </ul>
          <ul className="voice-dies muted-text">
            {chosen.stages.map((st) => (
              <li key={st.name}>
                {t(st.name)}: {t("first output after")} {formatNumber(st.first_ms)} ms, {formatNumber(Math.round(st.per_second))} {t("a second")}
                {st.variant ? ` (${t(st.variant)})` : ""}
              </li>
            ))}
          </ul>
        </>
      ) : (
        <p className="muted-text">{t("No way of putting the stages on dies meets what was asked.")}</p>
      )}
      <details>
        <summary>{t("Every grouping weighed")}</summary>
        <ul className="voice-dies muted-text">
          {plan.weighed.map((w, i) => (
            <li key={i}>
              {w.groups.map((g) => `[${g.map((n) => t(n)).join(" + ")}]`).join(" ")}: {w.area_mm2 !== null ? `${formatNumber(Math.round(w.area_mm2))} mm², ${formatNumber(Math.round(w.first_sound_ms ?? 0))} ms` : t(w.reason ?? "")}
            </li>
          ))}
        </ul>
      </details>
    </div>
  );
}

function Choose({ label, value, models, onChange, disabled }: { label: string; value: string; models: ServerModel[]; onChange: (path: string) => void; disabled: boolean }) {
  const t = useT();
  return (
    <label>
      {label}
      {models.length === 0 ? (
        <span className="muted-text">{t("None in the model folders")}</span>
      ) : (
        <select value={value} disabled={disabled} onChange={(e) => onChange(e.target.value)}>
          {models.map((m) => (
            <option key={m.path} value={m.path}>
              {m.name}
            </option>
          ))}
        </select>
      )}
    </label>
  );
}

export function VoicePanel() {
  const t = useT();
  const s = useVoice();
  const setView = useDesign((d) => d.setView);
  const { connect } = s;
  useEffect(() => {
    void connect();
  }, [connect]);
  // The project wizard asks for a plan: made as soon as the four models are known, whether
  // this panel was open already or not.
  const { planOnOpen, listen, think, speak, sound, planning, startPlan } = s;
  useEffect(() => {
    if (planOnOpen && listen && think && speak && sound && !planning) {
      useVoice.getState().set({ planOnOpen: false });
      void startPlan();
    }
  }, [planOnOpen, listen, think, speak, sound, planning, startPlan]);
  const of = (role: string) => s.models.filter((m) => m.voice === role);
  const missing = of("listens").length === 0 || of("speaks").length === 0 || of("sounds").length === 0;
  return (
    <aside className="inspector">
      <h2 className="detail-title">{t("Voice")}</h2>
      <p className="detail-sub">{t("Ask aloud and hear the answer: Moonshine listens, a language model answers, OuteTTS and DAC speak.")}</p>
      {s.online === false ? (
        <EngineMissing />
      ) : (
        <>
          <h3>{t("Models")}</h3>
          <div className="inset padded form">
            <Choose label={t("Listens")} value={s.listen} models={of("listens")} disabled={s.running} onChange={(listen) => s.set({ listen })} />
            <Choose label={t("Answers")} value={s.think} models={s.models.filter((m) => m.supported && !m.voice)} disabled={s.running} onChange={(think) => s.set({ think })} />
            <Choose label={t("Writes the speech")} value={s.speak} models={of("speaks")} disabled={s.running} onChange={(speak) => s.set({ speak })} />
            <Choose label={t("Makes the sound")} value={s.sound} models={of("sounds")} disabled={s.running} onChange={(sound) => s.set({ sound })} />
            {missing && (
              <>
                <p className="muted-text">{t("It needs UsefulSensors/moonshine-tiny, OuteAI/OuteTTS-1.0-0.6B and DAC.speech's decoder converted with tools/convert/dac.py: see docs/voice.md.")}</p>
                <button className="link-button" onClick={() => setView("models")}>
                  {t("See the models…")}
                </button>
              </>
            )}
          </div>
          <h3>{t("Arithmetic")}</h3>
          <div className="inset padded form">
            <label>
              {t("Every stage in")}
              <select value={s.arithmetic} disabled={s.running} onChange={(e) => s.set({ arithmetic: e.target.value as Arithmetic })}>
                <option value="float">{t("floating point")}</option>
                <option value="16">{t("the die's arithmetic, 16-bit vectors")}</option>
                <option value="20">{t("the die's arithmetic, 20-bit vectors")}</option>
                <option value="24">{t("the die's arithmetic, 24-bit vectors")}</option>
              </select>
            </label>
            <p className="muted-text">{t(s.arithmetic === "float" ? "About two minutes for a short answer." : "About five minutes: the simulation goes integer by integer. The sound decoder needs 20 bits or more to be understood.")}</p>
            <label className="check" title={t("OuteTTS reads a sentence said in that voice, with its codes, before each of its own: about 1,800 tokens more, which the chip reads before its first sound.")}>
              <input type="checkbox" checked={s.speaker} disabled={s.running || s.planning} onChange={(e) => s.set({ speaker: e.target.checked })} />
              {t("In the voice of the speaker profile")}
            </label>
          </div>
          <h3>{t("The chip")}</h3>
          <div className="inset padded form">
            <p className="muted-text">{t("Measures each stage on several shapes of die and chooses which share a die, for the least silicon that gives the first sound in time.")}</p>
            <label>
              {t("First sound within")}
              <span className="opt-amount">
                <input type="number" min={100} step={50} value={s.firstSoundMs} disabled={s.planning} onChange={(e) => s.set({ firstSoundMs: Number(e.target.value) })} />
                <span className="muted-text">ms</span>
              </span>
            </label>
            <label>
              {t("Longest question")}
              <span className="opt-amount">
                <input type="number" min={1} max={30} step={1} value={s.questionSeconds} disabled={s.planning} onChange={(e) => s.set({ questionSeconds: Number(e.target.value) })} />
                <span className="muted-text">s</span>
              </span>
            </label>
            <label>
              {t("Most silicon")}
              <span className="opt-amount">
                <input type="number" min={0} step={50} value={s.areaMm2} placeholder={t("no limit")} disabled={s.planning} onChange={(e) => s.set({ areaMm2: e.target.value })} />
                <span className="muted-text">mm²</span>
              </span>
            </label>
            <button className="pill primary-pill" disabled={s.planning || missing || !s.think} onClick={() => void s.startPlan()}>
              {t(s.planning ? "Planning…" : "Plan the chip")}
            </button>
            {s.planning && <p className="muted-text">{t("Measuring")} {t(s.planDoing)}…</p>}
            {s.planError && <p className="muted-text">{t(s.planError)}</p>}
          </div>
        </>
      )}
    </aside>
  );
}
