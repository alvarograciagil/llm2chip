import { useEffect, useRef, useState } from "react";
import { useT } from "../i18n";
import { formatNumber } from "../format";
import { busiestWork, cpuCycles, cpuInstructions, describeStep, GROUPS, groupCycles, ROUTINE_LABEL, TASK_LABEL, type ChipStep, type CpuTask, type StepActivity } from "./activity";
import { useSimulator } from "./simulatorStore";

const GROUP = Object.fromEntries(GROUPS.map((g) => [g.id, g]));

function bytes(n: number): string {
  return n >= 1 << 20 ? `${formatNumber(n / (1 << 20))} MB` : `${formatNumber(n / 1024)} KB`;
}

/** The CPU's cycles by group as one bar, each part with what it was in its tooltip; with
 * `work`, only what the CPU did itself, without its waits. */
function TaskBar({ activity, label, work = false }: { activity: StepActivity; label: string; work?: boolean }) {
  const t = useT();
  const parts = groupCycles(activity).filter((g) => g.cycles > 0 && !(work && g.id === "wait"));
  const total = parts.reduce((sum, g) => sum + g.cycles, 0);
  const describe = parts.map((g) => `${t(GROUP[g.id].label)}: ${formatNumber(g.cycles)}`).join(", ");
  return (
    <div className="task-bar" role="img" aria-label={`${label}: ${describe}`}>
      {parts.map((g) => (
        <i key={g.id} className={g.id === "wait" ? "waiting" : undefined} style={{ flexGrow: g.cycles, background: GROUP[g.id].color }} title={`${t(GROUP[g.id].label)}: ${formatNumber(g.cycles)} ${t("ciclos")} (${formatNumber((g.cycles / total) * 100)} %)`} />
      ))}
    </div>
  );
}

function StepRow({ step, open, onToggle, tokenizer }: { step: ChipStep; open: boolean; onToggle: () => void; tokenizer: boolean }) {
  const t = useT();
  const total = cpuCycles(step.activity);
  const routines = Object.entries(step.activity.routines);
  const work = busiestWork(step.activity);
  return (
    <li className={open ? "step open" : "step"}>
      <button className="step-head" onClick={onToggle} aria-expanded={open}>
        <span className="step-name">{step.kind === "prefill" ? `${t("Prompt")} ${step.index}` : `${t("Respuesta")} ${step.index}`}</span>
        <span className="step-text">{step.kind === "token" ? (step.text.trim() ? `«${step.text}»` : t("(sin texto aún)")) : routines.map(([name]) => ROUTINE_LABEL[name] ?? name).join(" + ")}</span>
        <TaskBar activity={step.activity} label={t("Ciclos de la CPU en este paso")} />
        <span className="step-cycles">{formatNumber(total)}</span>
      </button>
      {open && (
        <div className="step-detail">
          <ol>
            {describeStep(step, formatNumber, tokenizer).map((line, i) => (
              <li key={i}>{line}</li>
            ))}
          </ol>
          <p className="muted-text">
            {formatNumber(cpuInstructions(step.activity))} {t("instrucciones de la CPU")}
            {work ? ` · ${t("lo que más le ocupó")}: ${t(GROUP[work].label).toLowerCase()}` : ""}
          </p>
        </div>
      )}
    </li>
  );
}

/** What the chip did, step by step, with what its CPU spent the cycles on. */
export function ChipActivity() {
  const t = useT();
  const steps = useSimulator((s) => s.steps);
  const totals = useSimulator((s) => s.totals);
  const cpu = useSimulator((s) => s.cpuInfo);
  const running = useSimulator((s) => s.running);
  const firmware = useSimulator((s) => s.firmware);
  const [open, setOpen] = useState<number | null>(null);
  const list = useRef<HTMLOListElement>(null);

  // The list follows the newest step unless the reader has scrolled up to look at one.
  useEffect(() => {
    const el = list.current;
    if (el && running && el.scrollHeight - el.scrollTop - el.clientHeight < 60) el.scrollTop = el.scrollHeight;
  }, [steps.length, running]);

  if (!cpu || !totals) {
    return (
      <div className="activity">
        <div className="timeline-head">
          <strong>{t("Qué hace el chip en cada paso")}</strong>
        </div>
        <p className="muted-text">
          {firmware
            ? t("Envía un texto para ver, paso a paso, qué hacen la CPU, el die y la unidad de muestreo.")
            : t("Con el perfil «El die» no hay CPU: el host da los tokens al die directamente. Elige «El chip completo» en el panel para ver qué hace la CPU en cada paso.")}
        </p>
      </div>
    );
  }

  const all = cpuCycles(totals);
  const waiting = groupCycles(totals).find((g) => g.id === "wait")?.cycles ?? 0;
  const working = all - waiting;
  const die = Object.values(totals.routines).reduce((sum, [, cycles]) => sum + cycles, 0);
  const last = steps.at(-1);
  const now = running && last ? busiestWork(last.activity) : null;
  const tasks = (Object.entries(totals.cpu) as [CpuTask, [number, number]][]).filter(([, [cycles]]) => cycles > 0).sort((a, b) => b[1][0] - a[1][0]);

  return (
    <div className="activity">
      <div className="timeline-head">
        <strong>{t("La CPU del chip")}</strong>
        <span>{now ? `${t("Ahora")}: ${t(GROUP[now].label).toLowerCase()}` : `${formatNumber(all)} ${t("ciclos")}`}</span>
      </div>
      <p className="muted-text">
        {cpu.isa} · {t("programa")} {bytes(cpu.programBytes)} ({t("firmware de")} {bytes(cpu.firmwareBytes)}) · {t("datos")} {bytes(cpu.dataBytes)}{cpu.tokenizer ? ` ${t("con las tablas del tokenizador")}` : ""}.{" "}
        {cpu.tokenizer
          ? t("Recibe el texto del host, lo convierte en tokens, da al die cada token y su posición, espera mientras calcula, prepara el muestreo y convierte cada token elegido en texto para el host.")
          : t("Recibe los tokens del host, da al die cada token y su posición, espera mientras calcula, prepara el muestreo y envía al host cada token elegido.")}
      </p>
      <p className="activity-wait">
        {t("Espera al die o al muestreo el")} <strong>{formatNumber(all ? (waiting / all) * 100 : 0)} %</strong> {t("del tiempo. Su propio trabajo se reparte así:")}
      </p>
      <TaskBar activity={totals} work label={t("Trabajo de la CPU en toda la simulación")} />
      <div className="legend flat">
        {GROUPS.filter((g) => g.id !== "wait").map((g) => {
          const cycles = groupCycles(totals).find((c) => c.id === g.id)?.cycles ?? 0;
          return (
            <span key={g.id} title={g.tasks.map((task) => `${t(TASK_LABEL[task])}: ${formatNumber(totals.cpu[task]?.[0] ?? 0)}`).join("\n")}>
              <i className="swatch" style={{ background: g.color }} aria-hidden="true" />
              {t(g.label)} {working ? `${formatNumber((cycles / working) * 100)} %` : ""}
            </span>
          );
        })}
      </div>
      <dl className="metrics activity-metrics">
        <div><dt>{t("Ciclos del chip")}</dt><dd>{formatNumber(all)}</dd></div>
        <div><dt>{t("Die ocupado (estimado por el diseño)")}</dt><dd>{formatNumber(die)}</dd></div>
        <div><dt>{t("CPU trabajando")}</dt><dd>{formatNumber(working)} ({formatNumber(all ? (working / all) * 100 : 0)} %)</dd></div>
        <div><dt>{t("Instrucciones")}</dt><dd>{formatNumber(cpuInstructions(totals))}</dd></div>
        <div><dt>{t("Tokens muestreados")}</dt><dd>{formatNumber(totals.samples[0])} · {formatNumber(totals.samples[1])} {t("ciclos")}</dd></div>
      </dl>
      <details className="task-table">
        <summary>{t("Todas las tareas de la CPU")}</summary>
        <table>
          <tbody>
            {tasks.map(([task, [cycles, instructions]]) => (
              <tr key={task}>
                <td>{t(TASK_LABEL[task])}</td>
                <td>{formatNumber(cycles)} {t("ciclos")}</td>
                <td>{formatNumber(instructions)} {t("instr.")}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </details>

      <div className="timeline-head steps-head">
        <strong>{t("Paso a paso")}</strong>
        <span>{t("Pulsa un paso para ver qué hizo el chip")}</span>
      </div>
      <div className="legend flat steps-legend">
        <span>{t("Cada barra es el tiempo del paso:")}</span>
        <span>
          <i className="swatch waiting" style={{ background: GROUP.wait.color }} aria-hidden="true" />
          {t("la CPU esperando")}
        </span>
        <span>{t("y en color, su trabajo")}</span>
      </div>
      <ol className="steps" ref={list}>
        {steps.map((step, i) => (
          <StepRow key={i} step={step} open={open === i} onToggle={() => setOpen(open === i ? null : i)} tokenizer={cpu.tokenizer} />
        ))}
      </ol>
    </div>
  );
}
