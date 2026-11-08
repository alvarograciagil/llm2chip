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
    | { kind: "toggle" }
  );

export interface Group {
  title: string;
  open: boolean;
  fields: Field[];
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
    open: false,
    fields: [
      { kind: "range", key: "weight_bits", label: "Bits de peso", min: 2, max: 16, step: 1, integer: true, help: "Menos bits reducen memoria y área, pero pueden degradar la calidad." },
      { kind: "toggle", key: "use_model_mix", label: "Mantener tensores de 8 bits", help: "Reproduce la mezcla de 4 y 8 bits de los pesos reales de Qwen." },
      { kind: "range", key: "group_size", label: "Pesos por grupo", min: 0, max: 256, step: 8, integer: true, help: "Comparten escala y sesgo. 0 los elimina." },
      { kind: "range", key: "scale_bits", label: "Bits de escala y sesgo", min: 8, max: 32, step: 1, integer: true, when: (p) => Number(p.group_size) > 0 },
      { kind: "range", key: "activation_bits", label: "Bits de activación", min: 8, max: 32, step: 1, integer: true },
      { kind: "range", key: "accumulator_bits", label: "Bits de acumulador", min: 16, max: 64, step: 1, integer: true },
      { kind: "range", key: "delta_state_bits", label: "Bits de estado DeltaNet", min: 8, max: 32, step: 1, integer: true, help: "El modelo de referencia usa 32. Reducirlo es el mayor riesgo numérico." },
      { kind: "range", key: "kv_bits", label: "Bits de caché KV", min: 4, max: 32, step: 1, integer: true },
    ],
  },
  {
    title: "Etapas",
    open: false,
    fields: [
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
