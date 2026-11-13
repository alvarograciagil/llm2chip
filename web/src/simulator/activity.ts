// What the chip did in each step of a simulation, as the engine reports it with the CPU and
// its firmware: the CPU's cycles by task, the die's routines and the sampling unit.

export type CpuTask = "start" | "receive" | "tokenize" | "load" | "order" | "wait_die" | "sample" | "wait_sampler" | "detokenize" | "send" | "control";

/** Per step: cycles and instructions of each CPU task, runs and cycles of each routine of
 * the die, and tokens and cycles of the sampling unit. */
export interface StepActivity {
  cpu: Partial<Record<CpuTask, [number, number]>>;
  routines: Record<string, [number, number]>;
  samples: [number, number];
}

export interface ChipStep {
  kind: "prefill" | "token";
  /** Position of the token in the prompt or in the answer, from 1. */
  index: number;
  text: string;
  /** The chip's clock when the step ended. */
  cycles: number;
  activity: StepActivity;
}

/** What the CPU is and where it runs the firmware from. */
export interface CpuInfo {
  isa: string;
  programBytes: number;
  dataBytes: number;
  firmwareBytes: number;
  imageBytes: number;
  tokenizer: boolean;
}

export type GroupId = "host" | "text" | "die" | "sampler" | "control" | "wait";

/** The CPU's tasks in a few groups, in a fixed order with fixed colours. Waiting is grey:
 * it is the CPU idle while the die or the sampling unit works. */
export const GROUPS: { id: GroupId; label: string; tasks: CpuTask[]; color: string }[] = [
  { id: "host", label: "Talking to the host", tasks: ["receive", "send"], color: "var(--cat-interconnect)" },
  { id: "text", label: "Text and tokens", tasks: ["tokenize", "detokenize"], color: "var(--cat-compute)" },
  { id: "die", label: "Orders to the die", tasks: ["order", "load"], color: "var(--cat-memory)" },
  { id: "sampler", label: "Sampling", tasks: ["sample"], color: "var(--cat-vector)" },
  { id: "control", label: "Start-up and control", tasks: ["start", "control"], color: "var(--cat-state)" },
  { id: "wait", label: "Waiting for the die or the sampler", tasks: ["wait_die", "wait_sampler"], color: "var(--neutral-fill)" },
];

export const TASK_LABEL: Record<CpuTask, string> = {
  start: "Start-up",
  receive: "Receive from the host",
  tokenize: "Turn text into tokens",
  load: "Fetch rows from the image",
  order: "Give the die its parameters and orders",
  wait_die: "Wait for the die",
  sample: "Set up sampling and read the token",
  wait_sampler: "Wait for the sampler",
  detokenize: "Turn the token into text",
  send: "Send to the host",
  control: "Firmware rules",
};

export const ROUTINE_LABEL: Record<string, string> = {
  token: "pass a token through every layer",
  logits: "compute the logits",
  feed_from_main: "give the MTP head the model's state",
  feed_from_draft: "give the MTP head its own state",
  draft: "guess a token with the MTP head",
  draft_logits: "the MTP head's logits",
};

export const EMPTY_ACTIVITY: StepActivity = { cpu: {}, routines: {}, samples: [0, 0] };

/** Reads an activity as the engine sends it, tolerating what is missing. */
export function readActivity(raw: unknown): StepActivity {
  const a = (raw ?? {}) as Partial<StepActivity>;
  return { cpu: a.cpu ?? {}, routines: a.routines ?? {}, samples: a.samples ?? [0, 0] };
}

export function readCpuInfo(raw: Record<string, unknown>): CpuInfo {
  return {
    isa: String(raw.isa ?? "RV32IM"),
    programBytes: Number(raw.program_bytes) || 0,
    dataBytes: Number(raw.data_bytes) || 0,
    firmwareBytes: Number(raw.firmware_bytes) || 0,
    imageBytes: Number(raw.image_bytes) || 0,
    tokenizer: raw.tokenizer === true,
  };
}

export function addActivity(a: StepActivity, b: StepActivity): StepActivity {
  const cpu: StepActivity["cpu"] = { ...a.cpu };
  for (const [task, [cycles, instructions]] of Object.entries(b.cpu) as [CpuTask, [number, number]][]) {
    const was = cpu[task] ?? [0, 0];
    cpu[task] = [was[0] + cycles, was[1] + instructions];
  }
  const routines = { ...a.routines };
  for (const [name, [runs, cycles]] of Object.entries(b.routines)) {
    const was = routines[name] ?? [0, 0];
    routines[name] = [was[0] + runs, was[1] + cycles];
  }
  return { cpu, routines, samples: [a.samples[0] + b.samples[0], a.samples[1] + b.samples[1]] };
}

export function taskCycles(a: StepActivity, task: CpuTask): number {
  return a.cpu[task]?.[0] ?? 0;
}

/** The CPU's cycles in each group, in the order of `GROUPS`. */
export function groupCycles(a: StepActivity): { id: GroupId; cycles: number }[] {
  return GROUPS.map((g) => ({ id: g.id, cycles: g.tasks.reduce((sum, task) => sum + taskCycles(a, task), 0) }));
}

export function cpuCycles(a: StepActivity): number {
  return Object.values(a.cpu).reduce((sum, v) => sum + (v?.[0] ?? 0), 0);
}

export function cpuInstructions(a: StepActivity): number {
  return Object.values(a.cpu).reduce((sum, v) => sum + (v?.[1] ?? 0), 0);
}

/** The group that took most of the CPU's own work in a step, waiting aside. */
export function busiestWork(a: StepActivity): GroupId | null {
  const work = groupCycles(a).filter((g) => g.id !== "wait" && g.cycles > 0);
  return work.length ? work.reduce((best, g) => (g.cycles > best.cycles ? g : best)).id : null;
}

/** What happened in a step, one sentence per part, in the order the chip does them. */
export function describeStep(step: ChipStep, format: (n: number) => string, tokenizer: boolean): string[] {
  const a = step.activity;
  const lines: string[] = [];
  const cycles = (task: CpuTask) => taskCycles(a, task);
  if (cycles("start")) lines.push(`The CPU starts: it clears its memory and reads the chip's description the image left (${format(cycles("start"))} cycles).`);
  if (cycles("receive")) lines.push(`It receives ${tokenizer ? "the text" : "the token numbers"} from the host byte by byte (${format(cycles("receive"))} cycles).`);
  if (cycles("tokenize")) lines.push(`It turns the text into tokens with the tokenizer's tables: it normalizes, splits and merges pairs of symbols (${format(cycles("tokenize"))} cycles).`);
  if (cycles("load")) lines.push(`It fetches this token's embedding row from the image's storage (${format(cycles("load"))} cycles).`);
  if (cycles("order")) lines.push(`It writes the token and the position into the die's parameter registers and gives it its orders (${format(cycles("order"))} cycles).`);
  for (const [name, [runs, routineCycles]] of Object.entries(a.routines)) {
    const what = ROUTINE_LABEL[name] ?? name;
    lines.push(`The die runs the routine “${what}”${runs > 1 ? ` ${runs} times` : ""}${routineCycles ? `: ${format(routineCycles)} cycles` : ""}.`);
  }
  if (cycles("wait_die")) lines.push(`Meanwhile, the CPU waits, reading the die's status (${format(cycles("wait_die"))} cycles).`);
  if (a.samples[0]) lines.push(`The sampling unit goes through the logits and picks the token (${format(a.samples[1])} cycles); the CPU gives it its settings and reads the result (${format(cycles("sample") + cycles("wait_sampler"))} cycles).`);
  if (cycles("detokenize")) lines.push(`It turns the token into text (${format(cycles("detokenize"))} cycles).`);
  if (cycles("send")) lines.push(`It sends ${step.text ? `“${step.text}”` : "the result"} to the host (${format(cycles("send"))} cycles).`);
  if (cycles("control")) lines.push(`It applies its rules: end of turn, repetitions and reasoning budget (${format(cycles("control"))} cycles).`);
  return lines;
}
