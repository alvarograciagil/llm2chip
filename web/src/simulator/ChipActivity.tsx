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
        <i key={g.id} className={g.id === "wait" ? "waiting" : undefined} style={{ flexGrow: g.cycles, background: GROUP[g.id].color }} title={`${t(GROUP[g.id].label)}: ${formatNumber(g.cycles)} ${t("cycles")} (${formatNumber((g.cycles / total) * 100)} %)`} />
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
        <span className="step-name">{step.kind === "prefill" ? `${t("Prompt")} ${step.index}` : `${t("Answer")} ${step.index}`}</span>
        <span className="step-text">{step.kind === "token" ? (step.text.trim() ? `«${step.text}»` : t("(no text yet)")) : routines.map(([name]) => t(ROUTINE_LABEL[name] ?? name)).join(" + ")}</span>
        <TaskBar activity={step.activity} label={t("CPU cycles in this step")} />
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
            {formatNumber(cpuInstructions(step.activity))} {t("CPU instructions")}
            {work ? ` · ${t("what took it most")}: ${t(GROUP[work].label).toLowerCase()}` : ""}
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
          <strong>{t("What the chip does in each step")}</strong>
        </div>
        <p className="muted-text">
          {firmware
            ? t("Send a text to see, step by step, what the CPU, the die and the sampling unit do.")
            : t("With the “The die” profile there is no CPU: the host gives the tokens to the die directly. Choose “The whole chip” in the panel to see what the CPU does in each step.")}
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
        <strong>{t("The chip's CPU")}</strong>
        <span>{now ? `${t("Now")}: ${t(GROUP[now].label).toLowerCase()}` : `${formatNumber(all)} ${t("cycles")}`}</span>
      </div>
      <p className="muted-text">
        {cpu.isa} · {t("program")} {bytes(cpu.programBytes)} ({t("firmware of")} {bytes(cpu.firmwareBytes)}) · {t("data")} {bytes(cpu.dataBytes)}{cpu.tokenizer ? ` ${t("with the tokenizer's tables")}` : ""}.{" "}
        {cpu.tokenizer
          ? t("It receives the text from the host, turns it into tokens, gives the die each token and its position, waits while it computes, sets up sampling and turns each chosen token into text for the host.")
          : t("It receives the tokens from the host, gives the die each token and its position, waits while it computes, sets up sampling and sends each chosen token to the host.")}
      </p>
      <p className="activity-wait">
        {t("It waits for the die or the sampler")} <strong>{formatNumber(all ? (waiting / all) * 100 : 0)} %</strong> {t("of the time. Its own work splits like this:")}
      </p>
      <TaskBar activity={totals} work label={t("The CPU's work in the whole simulation")} />
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
        <div><dt>{t("Chip cycles")}</dt><dd>{formatNumber(all)}</dd></div>
        <div><dt>{t("Die busy (estimated from the design)")}</dt><dd>{formatNumber(die)}</dd></div>
        <div><dt>{t("CPU working")}</dt><dd>{formatNumber(working)} ({formatNumber(all ? (working / all) * 100 : 0)} %)</dd></div>
        <div><dt>{t("Instructions")}</dt><dd>{formatNumber(cpuInstructions(totals))}</dd></div>
        <div><dt>{t("Sampled tokens")}</dt><dd>{formatNumber(totals.samples[0])} · {formatNumber(totals.samples[1])} {t("cycles")}</dd></div>
      </dl>
      <details className="task-table">
        <summary>{t("All the CPU's tasks")}</summary>
        <table>
          <tbody>
            {tasks.map(([task, [cycles, instructions]]) => (
              <tr key={task}>
                <td>{t(TASK_LABEL[task])}</td>
                <td>{formatNumber(cycles)} {t("cycles")}</td>
                <td>{formatNumber(instructions)} {t("instr.")}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </details>

      <div className="timeline-head steps-head">
        <strong>{t("Step by step")}</strong>
        <span>{t("Click a step to see what the chip did")}</span>
      </div>
      <div className="legend flat steps-legend">
        <span>{t("Each bar is the step's time:")}</span>
        <span>
          <i className="swatch waiting" style={{ background: GROUP.wait.color }} aria-hidden="true" />
          {t("the CPU waiting")}
        </span>
        <span>{t("and in colour, its work")}</span>
      </div>
      <ol className="steps" ref={list}>
        {steps.map((step, i) => (
          <StepRow key={i} step={step} open={open === i} onToggle={() => setOpen(open === i ? null : i)} tokenizer={cpu.tokenizer} />
        ))}
      </ol>
    </div>
  );
}
