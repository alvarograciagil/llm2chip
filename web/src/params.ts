type Params = Record<string, unknown>;

interface Base {
  key: string;
  label: string;
  help?: string;
  /** Hidden while it would have no effect on the design. */
  when?: (p: Params) => boolean;
}

export type Field = Base &
  (
    | { kind: "range"; min: number; max: number; step?: number; log?: boolean; unit?: string; integer?: boolean }
    | { kind: "select"; options: [string, string][] }
    | { kind: "preset"; sections: PresetSection[] }
    | { kind: "toggle" }
  );

export interface PresetSection {
  title: string;
  presets: { label: string; values: Params }[];
}

export interface Group {
  title: string;
  open: boolean;
  fields: Field[];
}

const range = (from: number, to: number) => Array.from({ length: to - from + 1 }, (_, i) => from + i);
const fixed = (bits: number): Params => ({ weight_coding: "fixed", weight_bits: bits, use_model_mix: false, group_size: 0 });
const float = (bits: number, exponent: number, group = 0, fine = false): Params => ({
  weight_coding: "float",
  weight_bits: bits,
  weight_exponent_bits: exponent,
  use_model_mix: false,
  group_size: group,
  weight_bias: false,
  weight_fine_scale: fine,
  ...(group > 0 ? { scale_bits: 8 } : {}),
});
const affine = (bits: number, group: number, bias: boolean): Params => ({ weight_coding: "affine", weight_bits: bits, use_model_mix: false, group_size: group, weight_bias: bias });

export const WEIGHT_PRESETS: PresetSection[] = [
  {
    title: "Afín por grupos (MLX)",
    presets: [
      { label: "Mezcla de 4 y 8 bits (Qwen real)", values: { weight_coding: "affine", weight_bits: 4, use_model_mix: true, group_size: 32, weight_bias: true } },
      ...range(2, 8).map((bits) => ({ label: `MLX ${bits} bits`, values: affine(bits, bits <= 5 ? 32 : 64, true) })),
    ],
  },
  {
    title: "GGUF",
    presets: [
      { label: "Q2_K", values: affine(2, 16, false) },
      { label: "Q3_K", values: affine(3, 16, false) },
      { label: "Q4_0", values: affine(4, 32, false) },
      { label: "Q4_1", values: affine(4, 32, true) },
      { label: "Q5_0", values: affine(5, 32, false) },
      { label: "Q5_1", values: affine(5, 32, true) },
      { label: "Q6_K", values: affine(6, 16, false) },
      { label: "Q8_0", values: affine(8, 32, false) },
    ],
  },
  { title: "Punto fijo (Q)", presets: range(2, 32).map((bits) => ({ label: `Q${bits}`, values: fixed(bits) })) },
  {
    title: "Coma flotante",
    presets: [
      { label: "FP8 (E4M3)", values: float(8, 4) },
      { label: "FP8 (E5M2)", values: float(8, 5) },
      { label: "FP16", values: float(16, 5) },
      { label: "BF16", values: float(16, 8) },
      { label: "FP24", values: float(24, 7) },
      { label: "FP32", values: float(32, 8) },
    ],
  },
  {
    title: "Microescalado",
    presets: [
      { label: "MXFP4 (E2M1 / 32)", values: float(4, 2, 32) },
      { label: "NVFP4 (E2M1 / 16, escala fina)", values: float(4, 2, 16, true) },
      { label: "MXFP6 (E2M3 / 32)", values: float(6, 2, 32) },
      { label: "MXFP6 (E3M2 / 32)", values: float(6, 3, 32) },
      { label: "MXFP8 (E4M3 / 32)", values: float(8, 4, 32) },
      { label: "MXFP8 (E5M2 / 32)", values: float(8, 5, 32) },
    ],
  },
  {
    title: "Posit",
    presets: [
      { label: "Posit8 es1", values: { ...float(8, 1), weight_coding: "posit" } },
      { label: "Posit8 es2", values: { ...float(8, 2), weight_coding: "posit" } },
      { label: "Posit16 es1", values: { ...float(16, 1), weight_coding: "posit" } },
      { label: "Posit16 es2", values: { ...float(16, 2), weight_coding: "posit" } },
      { label: "Posit32 es2", values: { ...float(32, 2), weight_coding: "posit" } },
    ],
  },
  {
    title: "Logarítmico",
    presets: [
      { label: "LNS 8 (4 de fracción)", values: { ...float(8, 4), weight_coding: "log" } },
      { label: "LNS 12 (6 de fracción)", values: { ...float(12, 6), weight_coding: "log" } },
      { label: "LNS 16 (8 de fracción)", values: { ...float(16, 8), weight_coding: "log" } },
    ],
  },
  {
    title: "Libro de códigos",
    presets: [2, 3, 4, 6, 8].map((bits) => ({ label: `${bits} bits (${2 ** bits} valores)`, values: { ...fixed(bits), weight_coding: "codebook" } })),
  },
];

export function matchingPreset(params: Params): string | null {
  for (const section of WEIGHT_PRESETS) {
    for (const preset of section.presets) {
      if (Object.entries(preset.values).every(([key, value]) => params[key] === value)) return preset.label;
    }
  }
  return null;
}

const usesHbm = (p: Params) => p.weight_backend === "hbm4" || p.kv_backend === "hbm4";
const usesExternal = (p: Params) => p.weight_backend === "external_ddr" || p.kv_backend === "external_ddr";

export const PARAM_GROUPS: Group[] = [
  {
    title: "Objetivo",
    open: true,
    fields: [
      { kind: "range", key: "max_params_g", label: "Parámetros máximos", min: 0.0001, max: 200, log: true, unit: "G", help: "Tamaño del mayor modelo que el chip puede alojar. Dimensiona la memoria de pesos." },
      { kind: "range", key: "context_tokens", label: "Contexto", min: 128, max: 1048576, log: true, integer: true, unit: "tokens", help: "Hasta 1 M de tokens. La caché KV crece linealmente con el contexto y es el bloque que más limita la velocidad; a partir de unos 32 K conviene llevarla a HBM4." },
      { kind: "toggle", key: "mtp_enabled", label: "Soporte MTP", help: "Añade la cabeza que especula los siguientes tokens y aprovecha los dies inactivos." },
      { kind: "range", key: "mtp_depth", label: "Profundidad MTP", min: 1, max: 3, step: 1, integer: true, help: "Tokens especulados por ronda.", when: (p) => p.mtp_enabled === true },
      { kind: "toggle", key: "include_vision", label: "Incluir visión", help: "Reserva memoria para el codificador de imágenes." },
    ],
  },
  {
    title: "Organización",
    open: true,
    fields: [
      { kind: "range", key: "macs_per_die", label: "Multiplicadores por die", min: 2, max: 2000000, log: true, integer: true, help: "Con un solo usuario solo trabaja un die cada vez, así que este valor marca la velocidad." },
      { kind: "range", key: "freq_ghz", label: "Frecuencia", min: 0.02, max: 3.4, log: true, unit: "GHz", help: "Limitada por las etapas del multiplicador y por el nodo." },
      { kind: "range", key: "dies", label: "Dies", min: 0, max: 32, step: 1, integer: true, help: "0 elige automáticamente el mínimo que cumple el área máxima." },
      { kind: "range", key: "die_max_mm2", label: "Área máxima por die", min: 10, max: 858, step: 1, integer: true, unit: "mm²", help: "El límite físico de la litografía ronda los 858 mm²." },
    ],
  },
  {
    title: "Memoria",
    open: true,
    fields: [
      { kind: "select", key: "weight_backend", label: "Pesos", options: [["sram", "SRAM cargable"], ["rom", "ROM fija"], ["stacked_dram", "DRAM apilada"], ["hbm4", "HBM4"], ["external_ddr", "Memoria externa"]], help: "Con SRAM el chip admite otros modelos. Con ROM queda fijado a uno." },
      { kind: "select", key: "kv_backend", label: "Caché y estado", options: [["sram", "SRAM"], ["hbm4", "HBM4"], ["external_ddr", "Memoria externa"]], help: "Dónde viven la caché KV y el estado recurrente." },
      { kind: "range", key: "hbm_stacks", label: "Pilas HBM4", min: 1, max: 16, step: 1, integer: true, when: usesHbm },
      { kind: "range", key: "stacked_dram_tbps", label: "Ancho de banda DRAM", min: 10, max: 400, log: true, unit: "TB/s", when: (p) => p.weight_backend === "stacked_dram" },
      { kind: "range", key: "ddr_gbps", label: "Ancho de banda externo", min: 0.01, max: 400, log: true, unit: "GB/s", when: usesExternal },
      { kind: "range", key: "ddr_capacity_gb", label: "Capacidad externa", min: 0.004, max: 256, log: true, unit: "GB", when: usesExternal },
    ],
  },
  {
    title: "Aritmética",
    open: true,
    fields: [
      { kind: "preset", key: "weight_coding_preset", label: "Formato de pesos", sections: WEIGHT_PRESETS, help: "El circuito se adapta: cambia la memoria, el multiplicador y el decodificador de cada celda." },
      { kind: "select", key: "weight_coding", label: "Codificación", options: [["affine", "Afín por grupos"], ["fixed", "Punto fijo"], ["float", "Coma flotante"], ["posit", "Posit"], ["log", "Logarítmico"], ["codebook", "Libro de códigos"]], help: "Cómo se convierte cada código guardado en un número. Cada familia genera su propio decodificador." },
      { kind: "range", key: "weight_bits", label: "Bits de peso", min: 2, max: 32, step: 1, integer: true, help: "Bits guardados por peso, de Q2 a Q32 y cualquier ancho intermedio." },
      { kind: "range", key: "weight_exponent_bits", label: "Bits de exponente", min: 1, max: 15, step: 1, integer: true, help: "En coma flotante, los bits de exponente. En posit, «es». En logarítmico, los bits de fracción.", when: (p) => ["float", "posit", "log"].includes(String(p.weight_coding)) },
      { kind: "toggle", key: "use_model_mix", label: "Mantener tensores de 8 bits", help: "Reproduce la mezcla de 4 y 8 bits de los pesos reales de Qwen.", when: (p) => p.weight_coding === "affine" },
      { kind: "toggle", key: "weight_bias", label: "Sesgo por grupo", help: "MLX guarda escala y sesgo por grupo; GGUF Q4_0 solo escala.", when: (p) => p.weight_coding === "affine" },
      { kind: "toggle", key: "weight_fine_scale", label: "Escala fina por grupo", help: "Una escala en coma flotante como NVFP4 necesita un multiplicador. Una potencia de dos como MXFP4 solo un desplazador, pero pierde precisión.", when: (p) => p.weight_coding === "float" && Number(p.group_size) > 0 },
      { kind: "range", key: "group_size", label: "Pesos por grupo", min: 0, max: 256, step: 8, integer: true, help: "Comparten escala y sesgo. 0 los elimina.", when: (p) => p.weight_coding === "affine" || p.weight_coding === "float" },
      { kind: "range", key: "scale_bits", label: "Bits de escala y sesgo", min: 8, max: 32, step: 1, integer: true, when: (p) => Number(p.group_size) > 0 && (p.weight_coding === "affine" || p.weight_coding === "float") },
      { kind: "range", key: "activation_bits", label: "Bits de activación", min: 8, max: 32, step: 1, integer: true },
      { kind: "range", key: "accumulator_bits", label: "Bits de acumulador", min: 16, max: 128, step: 1, integer: true, help: "Si el formato necesita más, el chip usa el ancho necesario." },
      { kind: "range", key: "delta_state_bits", label: "Bits de estado DeltaNet", min: 8, max: 32, step: 1, integer: true, help: "El modelo de referencia usa 32. Reducirlo es el mayor riesgo numérico." },
      { kind: "range", key: "kv_bits", label: "Bits de caché KV", min: 4, max: 32, step: 1, integer: true },
    ],
  },
  {
    title: "Etapas",
    open: false,
    fields: [
      { kind: "select", key: "multiplier_kind", label: "Tipo de multiplicador", options: [["wallace", "Wallace"], ["booth", "Booth radix-4"], ["array", "Array"]], help: "Wallace y Booth reducen los productos con sumadores sin propagación y son mucho más rápidos. El array es más simple y compacto a baja frecuencia." },
      { kind: "range", key: "mac_stages", label: "Multiplicador", min: 1, max: 6, step: 1, integer: true, help: "Más etapas permiten mayor frecuencia a cambio de área y latencia." },
      { kind: "range", key: "adder_stages", label: "Árbol de suma", min: 1, max: 6, step: 1, integer: true },
      { kind: "range", key: "noc_stages", label: "Red de activaciones", min: 1, max: 4, step: 1, integer: true },
    ],
  },
  {
    title: "Unidades vectoriales",
    open: false,
    fields: [
      { kind: "range", key: "vector_lanes_per_die", label: "Carriles por die", min: 1, max: 512000, log: true, integer: true, help: "Calculan atención, regla delta y puertas, que multiplican activaciones entre sí." },
      { kind: "range", key: "vector_bits", label: "Bits por carril", min: 8, max: 32, step: 1, integer: true },
      { kind: "range", key: "function_units_per_die", label: "Unidades de función", min: 1, max: 128, step: 1, integer: true },
    ],
  },
  {
    title: "Sistema",
    open: false,
    fields: [
      { kind: "select", key: "link", label: "Enlace entre dies", options: [["ucie", "UCIe"], ["serdes", "SerDes en placa"]] },
      { kind: "select", key: "load_interface", label: "Carga de pesos", options: [["nvme", "NVMe"], ["pcie5", "PCIe 5.0 x16"], ["uart", "UART"]], help: "Solo afecta al tiempo de arranque." },
    ],
  },
];

export const FIELDS: Field[] = PARAM_GROUPS.flatMap((g) => g.fields);
export const fieldLabel = (key: string) => FIELDS.find((f) => f.key === key)?.label ?? key;
