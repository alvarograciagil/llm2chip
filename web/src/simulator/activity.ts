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
  { id: "host", label: "Comunicación con el host", tasks: ["receive", "send"], color: "var(--cat-interconexion)" },
  { id: "text", label: "Texto y tokens", tasks: ["tokenize", "detokenize"], color: "var(--cat-calculo)" },
  { id: "die", label: "Órdenes al die", tasks: ["order", "load"], color: "var(--cat-memoria)" },
  { id: "sampler", label: "Muestreo", tasks: ["sample"], color: "var(--cat-vectorial)" },
  { id: "control", label: "Arranque y control", tasks: ["start", "control"], color: "var(--cat-estado)" },
  { id: "wait", label: "Esperando al die o al muestreo", tasks: ["wait_die", "wait_sampler"], color: "var(--neutral-fill)" },
];

export const TASK_LABEL: Record<CpuTask, string> = {
  start: "Arranque",
  receive: "Recibir del host",
  tokenize: "Convertir texto en tokens",
  load: "Traer filas de la imagen",
  order: "Dar parámetros y órdenes al die",
  wait_die: "Esperar al die",
  sample: "Preparar el muestreo y leer el token",
  wait_sampler: "Esperar al muestreo",
  detokenize: "Convertir el token en texto",
  send: "Enviar al host",
  control: "Reglas del firmware",
};

export const ROUTINE_LABEL: Record<string, string> = {
  token: "pasar un token por todas las capas",
  logits: "calcular los logits",
  feed_from_main: "dar a la cabeza MTP el estado del modelo",
  feed_from_draft: "dar a la cabeza MTP su propio estado",
  draft: "adivinar un token con la cabeza MTP",
  draft_logits: "logits de la cabeza MTP",
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
  if (cycles("start")) lines.push(`La CPU arranca: pone a cero su memoria y lee la descripción del chip que dejó la imagen (${format(cycles("start"))} ciclos).`);
  if (cycles("receive")) lines.push(`Recibe del host ${tokenizer ? "el texto" : "los números de los tokens"} byte a byte (${format(cycles("receive"))} ciclos).`);
  if (cycles("tokenize")) lines.push(`Convierte el texto en tokens con las tablas del tokenizador: normaliza, separa y une pares de símbolos (${format(cycles("tokenize"))} ciclos).`);
  if (cycles("load")) lines.push(`Trae del almacenamiento de la imagen la fila del embedding de este token (${format(cycles("load"))} ciclos).`);
  if (cycles("order")) lines.push(`Escribe el token y la posición en los registros de parámetros del die y le da las órdenes (${format(cycles("order"))} ciclos).`);
  for (const [name, [runs, routineCycles]] of Object.entries(a.routines)) {
    const what = ROUTINE_LABEL[name] ?? name;
    lines.push(`El die ejecuta la rutina «${what}»${runs > 1 ? ` ${runs} veces` : ""}${routineCycles ? `: ${format(routineCycles)} ciclos` : ""}.`);
  }
  if (cycles("wait_die")) lines.push(`Mientras, la CPU espera leyendo el estado del die (${format(cycles("wait_die"))} ciclos).`);
  if (a.samples[0]) lines.push(`La unidad de muestreo recorre los logits y elige el token (${format(a.samples[1])} ciclos); la CPU le da los ajustes y lee el resultado (${format(cycles("sample") + cycles("wait_sampler"))} ciclos).`);
  if (cycles("detokenize")) lines.push(`Convierte el token en texto (${format(cycles("detokenize"))} ciclos).`);
  if (cycles("send")) lines.push(`Envía al host ${step.text ? `«${step.text}»` : "el resultado"} (${format(cycles("send"))} ciclos).`);
  if (cycles("control")) lines.push(`Aplica sus reglas: fin de turno, repeticiones y presupuesto de razonamiento (${format(cycles("control"))} ciclos).`);
  return lines;
}
