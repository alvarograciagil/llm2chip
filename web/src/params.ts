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
      { kind: "range", key: "max_params_g", label: "Parámetros máximos", min: 0.0001, max: 100000, log: true, unit: "G", help: "Tamaño del mayor modelo que el chip puede alojar, hasta 100 billones de parámetros (100.000 G). Dimensiona la memoria de pesos." },
      { kind: "range", key: "context_tokens", label: "Contexto", min: 128, max: 1048576, log: true, integer: true, unit: "tokens", help: "Hasta 1 M de tokens. La caché KV crece linealmente con el contexto y es el bloque que más limita la velocidad; a partir de unos 32 K conviene llevarla a HBM4." },
      { kind: "toggle", key: "mtp_enabled", label: "Soporte MTP", help: "Añade la cabeza de predicción múltiple (MTP): una capa extra que adivina los siguientes tokens mientras el modelo principal comprueba los anteriores. Con un solo usuario los dies esperan mucho, y este trabajo aprovecha ese hueco: si acierta, salen varios tokens por pasada." },
      { kind: "range", key: "mtp_depth", label: "Profundidad MTP", min: 1, max: 3, step: 1, integer: true, help: "Cuántos tokens adelanta la cabeza MTP en cada ronda. Más profundidad da más tokens por pasada cuando acierta, pero cada nivel acierta menos (en Qwen, en torno al 94 %, 86 % y 74 %) y exige guardar estado para deshacer.", when: (p) => p.mtp_enabled === true },
      { kind: "toggle", key: "include_vision", label: "Incluir visión", help: "Reserva área para el codificador de imágenes (un ViT de unos 400 M de parámetros en Qwen) que convierte cada imagen en tokens para el modelo de lenguaje." },
    ],
  },
  {
    title: "Organización",
    open: true,
    fields: [
      { kind: "range", key: "target_tok_s", label: "Velocidad objetivo", min: 0, max: 40000, step: 100, integer: true, unit: "tok/s", help: "Con el tamaño del modelo y la frecuencia, determina cuántos multiplicadores hacen falta. Con 0 los fijas a mano." },
      { kind: "range", key: "macs_per_die", when: (p) => !Number(p.target_tok_s), label: "Multiplicadores por die", min: 2, max: 2000000, log: true, integer: true, help: "Con un solo usuario solo trabaja un die cada vez, así que este valor marca la velocidad." },
      { kind: "range", key: "freq_ghz", label: "Frecuencia", min: 0.02, max: 3.4, log: true, unit: "GHz", help: "Reloj del die. Lo limita su camino más lento entre registros: el multiplicador y el árbol de suma (más etapas lo acortan) y el retardo de las puertas del nodo. Más frecuencia da más tok/s con la misma lógica, pero más consumo." },
      { kind: "select", key: "die_mode", label: "Tipo de dies", options: [["custom", "A medida de este modelo"], ["identical", "Todos iguales (ampliable)"]], help: "En los dos casos es el mismo chip repetido, que carga su parte del modelo desde la imagen. A medida dimensiona el die para la parte de este modelo; ampliable reserva sitio (capas, ancho, vocabulario) para conectar más chips con modelos mayores. Solo con los pesos grabados en ROM es distinto cada die." },
      { kind: "range", key: "max_layers", label: "Capas máximas", min: 1, max: 1024, log: true, integer: true, help: "Las capas salen de cada modelo: Qwen3.8 tiene 64 y Llama 3.1 8B, 32. Este límite reserva su estado en cada die.", when: (p) => p.die_mode === "identical" },
      { kind: "range", key: "max_hidden_size", label: "Ancho interno máximo", min: 64, max: 65536, log: true, integer: true, help: "Tamaño máximo del vector que recorre las capas (5.120 en Qwen3.8-27B). Con dies iguales, fija el ancho de búferes y enlaces para cualquier modelo que se cargue.", when: (p) => p.die_mode === "identical" },
      { kind: "range", key: "max_vocabulary", label: "Vocabulario máximo", min: 256, max: 4194304, log: true, integer: true, help: "Tokens distintos que admite el modelo mayor (248.320 en Qwen). Dimensiona lm_head, la unidad de muestreo y las tablas del tokenizador.", when: (p) => p.die_mode === "identical" },
      { kind: "range", key: "dies", label: "Dies", min: 0, max: 32, step: 1, integer: true, help: "En cuántos dies se reparte el modelo. Con 0 se elige el mínimo con el que cada die cabe en el área máxima. Con un solo usuario los dies trabajan por turnos, así que más dies dan más memoria pero no más velocidad por sí solos." },
      { kind: "select", key: "chip_split", label: "Reparto entre chips", options: [["layers", "Por capas, en cadena"], ["tensor", "Por tensor, todos a la vez"], ["mixed", "Mixto: grupos en tensor, en cadena"]], help: "Cómo se reparten el modelo varios chips. Por capas: cada chip guarda unas capas enteras y pasa el vector oculto al siguiente; poca comunicación, pero con un solo usuario trabaja un chip cada vez. Por tensor: cada chip guarda una parte de todas las capas y todos calculan cada capa a la vez; mucho más rápido para un usuario, pero suman sus resultados dos veces por capa y necesitan enlaces rápidos, dentro del encapsulado. Mixto: grupos de chips en tensor unidos en cadena, para que cada grupo quepa en un encapsulado.", when: (p) => Number(p.dies) !== 1 },
      { kind: "range", key: "tensor_group", label: "Chips por grupo", min: 1, max: 32, step: 1, integer: true, help: "En el reparto mixto, cuántos chips comparten cada capa. Los grupos se unen en cadena. Con 1 es el reparto por capas; con todos los chips, por tensor. Conviene que un grupo quepa en un encapsulado, unido por UCIe.", when: (p) => p.chip_split === "mixed" },
      { kind: "range", key: "die_max_mm2", label: "Área máxima por die", min: 10, max: 858, step: 1, integer: true, unit: "mm²", help: "Área máxima de cada die. El límite físico es la retícula del escáner (26 × 33 mm, unos 858 mm²). Los dies grandes rinden menos por oblea, así que a veces conviene un límite menor." },
    ],
  },
  {
    title: "Memoria",
    open: true,
    fields: [
      { kind: "select", key: "weight_backend", label: "Pesos", options: [["sram", "SRAM (modelo intercambiable)"], ["rom", "ROM (modelo grabado)"], ["stacked_dram", "DRAM apilada"], ["hbm4", "HBM4"], ["external_ddr", "Memoria externa"]], help: "Con SRAM el chip admite otros modelos. Con ROM los pesos se graban al fabricarlo: desaparecen la escritura de pesos y su tiempo de carga, pero cambiar de modelo exige otras máscaras." },
      { kind: "select", key: "kv_backend", label: "Caché y estado", options: [["sram", "SRAM"], ["hbm4", "HBM4"], ["external_ddr", "Memoria externa"]], help: "Dónde viven la caché KV de la atención y el estado recurrente de DeltaNet. En SRAM son lo más rápido pero ocupan área que crece con el contexto; fuera, el ancho de banda de esa memoria limita la velocidad." },
      { kind: "range", key: "hbm_stacks", label: "Pilas HBM4", min: 1, max: 16, step: 1, integer: true, help: "Cada pila de HBM4 suma capacidad y ancho de banda, y un PHY de unos 12 mm² en el die que la conecta.", when: usesHbm },
      { kind: "range", key: "stacked_dram_tbps", label: "Ancho de banda DRAM", min: 10, max: 400, log: true, unit: "TB/s", help: "Lo que la DRAM apilada sobre el die entrega por segundo. Con los pesos fuera de la SRAM, marca la velocidad: cada token lee todos los pesos una vez.", when: (p) => p.weight_backend === "stacked_dram" },
      { kind: "range", key: "ddr_gbps", label: "Ancho de banda externo", min: 0.01, max: 400, log: true, unit: "GB/s", help: "Lo que la memoria externa (DDR, LPDDR) entrega por segundo. Con los pesos fuera, los tok/s no pasan de este valor dividido por el tamaño del modelo.", when: usesExternal },
      { kind: "range", key: "ddr_capacity_gb", label: "Capacidad externa", min: 0.004, max: 256, log: true, unit: "GB", help: "Memoria externa disponible: debe alojar lo que se lleva fuera del chip (pesos, caché KV o ambos).", when: usesExternal },
    ],
  },
  {
    title: "Aritmética",
    open: true,
    fields: [
      { kind: "preset", key: "weight_coding_preset", label: "Formato de pesos", sections: WEIGHT_PRESETS, help: "El circuito se adapta: cambia la memoria, el multiplicador y el decodificador de cada celda." },
      { kind: "select", key: "weight_coding", label: "Codificación", options: [["affine", "Afín por grupos"], ["fixed", "Punto fijo"], ["float", "Coma flotante"], ["posit", "Posit"], ["log", "Logarítmico"], ["codebook", "Libro de códigos"]], help: "Cómo se convierte cada código guardado en un número. Cada familia genera su propio decodificador." },
      { kind: "range", key: "weight_bits", label: "Bits de peso", min: 2, max: 32, step: 1, integer: true, help: "Bits guardados por peso, de 2 a 32. Fija la memoria de pesos, que domina el área, y el tamaño del multiplicador de cada celda. Con menos bits cabe más modelo pero se pierde precisión." },
      { kind: "range", key: "weight_exponent_bits", label: "Bits de exponente", min: 1, max: 15, step: 1, integer: true, help: "En coma flotante, los bits de exponente. En posit, «es». En logarítmico, los bits de fracción.", when: (p) => ["float", "posit", "log"].includes(String(p.weight_coding)) },
      { kind: "toggle", key: "use_model_mix", label: "Mantener tensores de 8 bits", help: "Reproduce la mezcla de los pesos reales de Qwen en MLX: la mayoría a 4 bits y embed_tokens, lm_head, out_proj y la MLP de las últimas capas a 8. Las celdas guardan dos códigos de 4 bits en cada palabra de 8.", when: (p) => p.weight_coding === "affine" },
      { kind: "toggle", key: "weight_bias", label: "Sesgo por grupo", help: "Si cada grupo guarda también un sesgo (peso = código · escala + sesgo). MLX y Q4_1 lo hacen; Q4_0 solo guarda escala. El sesgo cuesta memoria y una suma por grupo.", when: (p) => p.weight_coding === "affine" },
      { kind: "toggle", key: "weight_fine_scale", label: "Escala fina por grupo", help: "Una escala en coma flotante como NVFP4 necesita un multiplicador. Una potencia de dos como MXFP4 solo un desplazador, pero pierde precisión.", when: (p) => p.weight_coding === "float" && Number(p.group_size) > 0 },
      { kind: "range", key: "group_size", label: "Pesos por grupo", min: 0, max: 256, step: 8, integer: true, help: "Pesos consecutivos que comparten escala y sesgo (32 en el Qwen de MLX). Grupos pequeños dan más precisión pero más factores que guardar y aplicar. Con 0 no hay grupos.", when: (p) => p.weight_coding === "affine" || p.weight_coding === "float" },
      { kind: "range", key: "scale_bits", label: "Bits de escala y sesgo", min: 8, max: 32, step: 1, integer: true, help: "Precisión con que cada celda guarda la escala y el sesgo de cada grupo. Más bits: menos error y más memoria y multiplicador de escala.", when: (p) => Number(p.group_size) > 0 && (p.weight_coding === "affine" || p.weight_coding === "float") },
      { kind: "range", key: "activation_bits", label: "Bits de activación", min: 8, max: 32, step: 1, integer: true, help: "Ancho de los valores que recorren el chip entre operaciones. Fija el ancho de los multiplicadores, de la red de activaciones y de la memoria vector: es de lo que más pesa en área y en precisión." },
      { kind: "range", key: "accumulator_bits", label: "Bits de acumulador", min: 16, max: 128, step: 1, integer: true, help: "Ancho del acumulador de cada celda, que suma productos de peso por activación. Si es corto, las sumas se desbordan; si el formato necesita más, el chip usa el ancho necesario." },
      { kind: "range", key: "delta_state_bits", label: "Bits de estado DeltaNet", min: 8, max: 32, step: 1, integer: true, help: "Ancho del estado recurrente de DeltaNet, una matriz por cabeza que se actualiza en cada token. La referencia usa 32 bits; reducirlo ahorra mucha memoria, pero el error se acumula a lo largo del contexto y es el mayor riesgo numérico." },
      { kind: "range", key: "kv_bits", label: "Bits de caché KV", min: 4, max: 32, step: 1, integer: true, help: "Ancho de cada clave y valor guardado por token. La caché crece con el contexto, así que con contextos largos es la memoria que más ocupa después de los pesos." },
    ],
  },
  {
    title: "Etapas",
    open: false,
    fields: [
      { kind: "toggle", key: "auto_circuit", label: "Circuito a medida", help: "El generador prueba cada forma de construir la celda para estos parámetros, mide su circuito real y se queda con la mejor." },
      { kind: "select", key: "circuit_goal", label: "Priorizar", options: [["balanced", "Área y energía"], ["area", "Área"], ["energy", "Energía"]], help: "Qué se minimiza entre las formas de celda que alcanzan la frecuencia pedida: área, energía por token o un equilibrio de ambas.", when: (p) => Boolean(p.auto_circuit) },
      { kind: "select", key: "multiplier_kind", when: (p) => !p.auto_circuit, label: "Tipo de multiplicador", options: [["wallace", "Wallace"], ["booth", "Booth radix-4"], ["array", "Array"], ["multiples", "Múltiplos precalculados"]], help: "Wallace y Booth reducen los productos con sumadores sin propagación y son mucho más rápidos. El array es más simple y compacto a baja frecuencia." },
      { kind: "range", key: "mac_stages", when: (p) => !p.auto_circuit, label: "Multiplicador", min: 1, max: 6, step: 1, integer: true, help: "Más etapas permiten mayor frecuencia a cambio de área y latencia." },
      { kind: "range", key: "factor_steps", label: "Ciclos de escala y sesgo", min: 1, max: 16, step: 1, integer: true, help: "Cada celda aplica la escala y el sesgo de un grupo en tantos ciclos, unos pocos bits por ciclo, mientras ya suma el grupo siguiente. Con 1 hace falta un multiplicador entero por celda; con 8 basta uno ocho veces menor, sin perder velocidad si el grupo tiene al menos 8 pesos.", when: (p) => Number(p.group_size) > 0 },
      { kind: "range", key: "adder_stages", when: (p) => !p.auto_circuit, label: "Árbol de suma", min: 1, max: 6, step: 1, integer: true, help: "Registros dentro del árbol que suma los productos de las celdas. Más etapas permiten más frecuencia a cambio de latencia y biestables." },
      { kind: "range", key: "noc_stages", label: "Red de activaciones", min: 1, max: 4, step: 1, integer: true, help: "Registros en la red que reparte cada activación a todas las celdas. En dies grandes los cables son largos: más etapas evitan que limiten el reloj." },
    ],
  },
  {
    title: "Unidades vectoriales",
    open: false,
    fields: [
      { kind: "range", key: "vector_lanes_per_die", label: "Carriles por die", min: 1, max: 512000, log: true, integer: true, help: "Carriles de las unidades vectoriales, que multiplican activación por activación: atención Q·K, regla delta, puertas y normas. Pocos carriles alargan las partes que no son producto de matriz." },
      { kind: "range", key: "vector_units_per_die", label: "Unidades vectoriales por die", min: 1, max: 256, log: true, integer: true, help: "Copias de la unidad vector: cada una calcula una cabeza de atención o DeltaNet a la vez." },
      { kind: "range", key: "sampler_lanes", label: "Logits por ciclo al muestrear", min: 1, max: 4096, log: true, integer: true, help: "Cuántos logits compara a la vez la unidad de muestreo. Si son pocos, elegir el token retrasa al chip." },
      { kind: "range", key: "sampler_entries", label: "Candidatos del muestreo", min: 1, max: 1024, log: true, integer: true, help: "Cuántos de los mejores tokens guarda la unidad de muestreo: el mayor top-k que admite. Cada candidato cuesta registros y ciclos al elegir." },
      { kind: "range", key: "vector_bits", label: "Bits por carril", min: 8, max: 32, step: 1, integer: true, help: "Ancho de los multiplicadores de las unidades vectoriales, que operan activación por activación (atención, regla delta, normas). El área crece con su cuadrado." },
      { kind: "range", key: "function_units_per_die", label: "Unidades de función", min: 1, max: 128, step: 1, integer: true, help: "Tablas que calculan funciones no lineales (exp, SiLU, sigmoide, raíz inversa…) consultando e interpolando. Más unidades aplican la función a más elementos a la vez." },
      { kind: "range", key: "function_table_bits", label: "Resolución de las tablas", min: 4, max: 20, step: 1, integer: true, help: "Entradas de cada tabla de función, en potencias de dos: 12 son 4.096. Más entradas reducen el error de exp, SiLU o la raíz inversa a cambio de memoria en cada unidad." },
    ],
  },
  {
    title: "Sistema",
    open: false,
    fields: [
      { kind: "select", key: "link", label: "Enlace entre dies", options: [["auto", "Automático según encapsulado"], ["ucie", "UCIe"], ["serdes", "SerDes en placa"], ["pins", "Pines de E/S"]], help: "UCIe une dies dentro de un mismo encapsulado: rápido (unos 10 ns por salto) y pequeño. SerDes los une por la placa: lento (unos 150 ns) y mayor. Los pines de E/S son lo único que tienen los chips de una lanzadera abierta, cada uno en su encapsulado: 16 pines por sentido a 50 MHz y unos 60 ns por salto. En automático se llenan encapsulados y se unen entre sí por SerDes, salvo en una lanzadera abierta, que usa los pines." },
      { kind: "range", key: "link_words", label: "Ancho del enlace", min: 1, max: 4096, log: true, integer: true, unit: "activaciones/ciclo", help: "Cuántas activaciones pasa cada enlace por ciclo. El vector oculto cruza cada frontera entre dies en cada token: un enlace estrecho añade miles de ciclos por frontera; uno ancho necesita más módulos UCIe o SerDes." },
      { kind: "select", key: "load_interface", label: "Carga de pesos", options: [["nvme", "NVMe"], ["pcie5", "PCIe 5.0 x16"], ["uart", "UART"]], help: "Por dónde llegan los pesos al encender el chip: NVMe, PCIe 5.0 o UART. Solo afecta al tiempo de arranque, no a la velocidad de generación.", when: (p) => p.weight_backend !== "rom" },
    ],
  },
];

export const FIELDS: Field[] = PARAM_GROUPS.flatMap((g) => g.fields);
export const fieldLabel = (key: string) => FIELDS.find((f) => f.key === key)?.label ?? key;
