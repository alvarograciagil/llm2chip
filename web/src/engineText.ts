// What the engine writes, in English: its texts are built in Spanish, numbers included, so
// each is recognised by its shape and rewritten. `{n}` is a number, `{s}` a text that is
// itself translated (a block's name inside a warning, a list of them). A test keeps this in
// step with the engine's sources.

export const ENGINE_TEXT: [string, string][] = [
  // Formats.
  ["punto fijo", "fixed point"],
  ["formato afín", "affine format"],
  ["coma flotante", "floating point"],
  ["formato logarítmico", "logarithmic format"],
  ["libro de códigos", "codebook"],
  ["ancho interno", "hidden size"],

  // Blocks of the die.
  ["Celdas: memoria de pesos", "Cells: weight memory"],
  ["Cada celda guarda un trozo de cada matriz junto a su multiplicador, para que los pesos nunca viajen.", "Each cell keeps a piece of every matrix next to its multiplier, so weights never travel."],
  ["Celdas: corrección ECC", "Cells: ECC correction"],
  ["Comprueba cada palabra de pesos y corrige el peso que se está leyendo.", "Checks each word of weights and corrects the weight being read."],
  ["Celdas: multiplicadores", "Cells: multipliers"],
  ["Multiplican peso por activación y acumulan la suma de cada fila.", "Multiply weight by activation and accumulate each row's sum."],
  ["Tablas de pesos", "Weight tables"],
  ["Cada celda guarda los valores reales a los que apuntan sus códigos.", "Each cell keeps the real values its codes point to."],
  ["Escala y sesgo por grupo", "Scale and bias per group"],
  ["Aplican la escala y el sesgo que comparten los pesos de cada grupo.", "Apply the scale and bias shared by the weights of each group."],
  ["Red de activaciones", "Activation network"],
  ["Reparte cada vector de entrada a todas las celdas y recoge las sumas parciales.", "Sends each input vector to every cell and gathers the partial sums."],
  ["Unidades vectoriales", "Vector units"],
  ["Multiplican activación por activación o estado: atención Q·K, regla delta y puertas.", "Multiply activation by activation or state: attention Q·K, delta rule and gates."],
  ["Unidades de función", "Function units"],
  ["Tablas para SiLU, sigmoide, softplus, exp, raíz inversa y softmax.", "Tables for SiLU, sigmoid, softplus, exp, inverse square root and softmax."],
  ["Búferes de activación", "Activation buffers"],
  ["Guardan los vectores intermedios entre etapas.", "Hold the intermediate vectors between stages."],
  ["Caché KV de atención", "Attention KV cache"],
  ["Claves y valores de cada token del contexto. Crece con el contexto.", "Keys and values of every token in the context. Grows with the context."],
  ["Estado de DeltaNet", "DeltaNet state"],
  ["Matrices recurrentes de las capas lineales. No crece con el contexto.", "Recurrent matrices of the linear layers. Does not grow with the context."],
  ["MTP: copia de estado", "MTP: state copy"],
  ["Permite deshacer el estado cuando se rechaza un token especulado.", "Lets the state be undone when a speculated token is rejected."],
  ["Unidad de muestreo", "Sampling unit"],
  ["Conserva los mejores logits y elige el siguiente token, con top-k y temperatura.", "Keeps the best logits and picks the next token, with top-k and temperature."],
  ["Tablas del tokenizador", "Tokenizer tables"],
  ["Vocabulario y fusiones que el firmware usa para convertir texto en tokens y volver.", "Vocabulary and merges the firmware uses to turn text into tokens and back."],
  ["Memoria de la CPU", "CPU memory"],
  ["Programa del firmware y memoria donde el tokenizador trabaja con el texto de entrada.", "The firmware's program and the memory where the tokenizer works on the input text."],
  ["Secuenciador y microprograma", "Sequencer and microprogram"],
  ["Ejecuta el microprograma que dirige cada etapa.", "Runs the microprogram that drives each stage."],
  ["CPU de control y carga", "Control and load CPU"],
  ["Tokenizador, muestreo y carga del microprograma. Los pesos ya vienen grabados.", "Tokenizer, sampling and loading of the microprogram. The weights come already burnt in."],
  ["Tokenizador, muestreo y volcado de pesos desde la imagen.", "Tokenizer, sampling and loading of the weights from the image."],
  ["Enlaces entre dies", "Die-to-die links"],
  ["Pasan un vector de activación entre capas consecutivas.", "Pass an activation vector between consecutive layers."],
  ["Controlador de DRAM", "DRAM controller"],
  ["Acceso a la DRAM externa.", "Access to the external DRAM."],
  ["PHY de HBM4", "HBM4 PHY"],
  ["Interfaz con las pilas de memoria HBM4.", "Interface to the HBM4 memory stacks."],
  ["Reloj, alimentación y test", "Clock, power and test"],
  ["Infraestructura necesaria en cada die.", "Infrastructure every die needs."],

  // Warnings and their fixes.
  ["Con ROM cada die lleva grabados pesos distintos, así que no pueden ser todos iguales.", "With ROM each die has different weights burnt in, so they cannot all be the same."],
  ["Usa SRAM o genera cada die a medida.", "Use SRAM or build each die to measure."],
  ["El modelo supera el límite de {s} de este diseño.", "The model exceeds this design's limit of {0}."],
  ["capas", "layers"],
  ["vocabulario", "vocabulary"],
  ["Sube ese límite.", "Raise that limit."],
  ["Optimización desconocida: {s}", "Unknown optimization: {0}"],
  ["Quítala del diseño.", "Remove it from the design."],
  ["«{s}» solo tiene sentido con pesos en SRAM.", "“{0}” only makes sense with weights in SRAM."],
  ["Cambia la memoria de pesos a SRAM o desactiva la optimización.", "Change the weight memory to SRAM or turn the optimization off."],
  ["«{s}» y «{s}» son incompatibles.", "“{0}” and “{1}” are incompatible."],
  ["Elige una de las dos.", "Choose one of the two."],
  ["«multiplier_free» solo rinde lo anunciado con pesos de 4 bits o menos.", "“multiplier_free” only delivers what it promises with weights of 4 bits or fewer."],
  ["Baja los bits de peso y desactiva la mezcla de 8 bits.", "Lower the weight bits and turn off the 8-bit mix."],
  ["Este formato de pesos necesita un acumulador de {n} bits; se usa ese ancho en lugar de {n}.", "This weight format needs a {0}-bit accumulator; that width is used instead of {1}."],
  ["Sube los bits de acumulador o usa un formato más estrecho.", "Raise the accumulator bits or use a narrower format."],
  ["Los pesos CSD solo ahorran área si están grabados en ROM: con memoria cargable la celda necesita un multiplicador general.", "CSD weights only save area when burnt into ROM: with loadable memory the cell needs a general multiplier."],
  ["Elige ROM como memoria de pesos.", "Choose ROM as the weight memory."],
  ["No se alcanza la velocidad objetivo solo con más multiplicadores.", "The target speed cannot be reached with more multipliers alone."],
  ["Sube los carriles vectoriales, la frecuencia o el ancho de banda de memoria.", "Raise the vector lanes, the clock or the memory bandwidth."],
  ["nodo tecnológico desconocido: {s}", "unknown technology node: {0}"],
  ["El modelo no cabe: tiene {n} G de parámetros y el chip admite {n} G.", "The model does not fit: it has {0} G parameters and the chip takes {1} G."],
  ["Sube los parámetros máximos.", "Raise the maximum parameters."],
  ["El modelo declara {n} tokens de contexto nativos; por encima puede perder calidad.", "The model declares {0} tokens of native context; beyond that it may lose quality."],
  ["Reduce el contexto si la calidad cae.", "Reduce the context if quality drops."],
  ["Este modelo no tiene cabeza MTP.", "This model has no MTP head."],
  ["Desactiva MTP.", "Turn MTP off."],
  ["Los pesos ocupan {n} GB y la memoria externa tiene {n} GB.", "The weights take {0} GB and the external memory has {1} GB."],
  ["Amplía la memoria externa o reduce los bits de peso.", "Enlarge the external memory or reduce the weight bits."],
  ["Cada die ocupa {n} mm² y el máximo es {n} mm².", "Each die takes {0} mm² and the limit is {1} mm²."],
  ["Añade dies, reduce multiplicadores o contexto, o activa optimizaciones de memoria.", "Add dies, reduce multipliers or context, or turn on memory optimizations."],
  ["el circuito de la celda no se puede simular", "the cell's circuit cannot be simulated"],
  ["Con {n} etapas de multiplicador el máximo son {n} GHz; se simula a esa frecuencia.", "With {0} multiplier stages the maximum is {1} GHz; it is simulated at that clock."],
  ["Añade etapas o baja la frecuencia.", "Add stages or lower the clock."],
  ["{n} mm² de silicio no caben en un encapsulado de {n} mm².", "{0} mm² of silicon do not fit in a {1} mm² package."],
  ["Elige el enlace automático, que reparte los dies en varios encapsulados unidos por SerDes.", "Choose the automatic link, which spreads the dies over several packages joined by SerDes."],
  ["Los grupos de {n} chips que comparten cada capa no caben en un encapsulado: sus sumas cruzarían la placa, mucho más lenta.", "The groups of {0} chips that share each layer do not fit in a package: their sums would cross the board, much slower."],
  ["Usa grupos más pequeños (reparto mixto) o dies menores para que cada grupo quepa en un encapsulado.", "Use smaller groups (mixed split) or smaller dies so each group fits in a package."],

  // Why each block is its size.
  ["{n} KB por celda con {n} celdas por die, a unos {n} Mbit/mm².", "{0} KB per cell with {1} cells per die, at about {2} Mbit/mm²."],
  ["{n} multiplicadores de {n} × {n} bits para pesos de {n} bits en {s}, de {n} µm² cada uno.", "{0} multipliers of {1} × {2} bits for {3}-bit weights in {4}, {5} µm² each."],
  ["{n} Kbit de tabla por celda.", "{0} Kbit of table per cell."],
  ["Una unidad por celda, usada una vez cada {n} pesos.", "One unit per cell, used once every {0} weights."],
  ["Árbol de {n} niveles sobre {n} celdas.", "Tree of {0} levels over {1} cells."],
  ["{n} carriles de {n} bits.", "{0} lanes of {1} bits."],
  ["{n} unidades con tablas de 64 Kbit.", "{0} units with 64 Kbit tables."],
  ["Ocho vectores del tamaño oculto y dos del ancho intermedio.", "Eight vectors of the hidden size and two of the intermediate width."],
  ["{n} GB para {n} tokens, repartidos entre {n} dies.", "{0} GB for {1} tokens, spread over {2} dies."],
  ["{n} MB a {n} bits, repartidos entre {n} dies.", "{0} MB at {1} bits, spread over {2} dies."],
  ["{n} copias del estado recurrente.", "{0} copies of the recurrent state."],
  ["Área fija estimada por die.", "Estimated fixed area per die."],
  ["{n} Mbit para un vocabulario de {n} tokens.", "{0} Mbit for a vocabulary of {1} tokens."],
  ["Para prompts de hasta {n} tokens.", "For prompts of up to {0} tokens."],
  ["{n} carriles de filtro y {n} candidatos.", "{0} filter lanes and {1} candidates."],
  ["{n} enlaces por pines de E/S, uno por encapsulado.", "{0} links over I/O pins, one per package."],
  ["{n} enlaces UCIe y {n} SerDes, en {n} encapsulados.", "{0} UCIe and {1} SerDes links, in {2} packages."],
  ["Área fija estimada.", "Estimated fixed area."],

  // Metrics.
  ["Parámetros del modelo cargado", "Parameters of the loaded model"],
  ["Suma de los tensores del config.json con las opciones activas.", "Sum of the config.json tensors with the active options."],
  ["Pesos del modelo en memoria", "Model weights in memory"],
  ["Bits por peso, escalas y sesgos por grupo y optimizaciones de almacenamiento.", "Bits per weight, scales and biases per group, and storage optimizations."],
  ["Capacidad de pesos del chip", "Chip weight capacity"],
  ["El chip se dimensiona para el máximo de parámetros, no solo para este modelo.", "The chip is sized for the maximum parameters, not just for this model."],
  ["Tiempo de carga de la imagen", "Image load time"],
  ["Los pesos vienen grabados en el chip y no se cargan.", "The weights come burnt into the chip and are not loaded."],
  ["Tamaño de la imagen dividido por la interfaz de carga.", "Image size divided by the load interface."],
  ["El mínimo que cumple el área máxima por die.", "The fewest that meet the maximum area per die."],
  ["Hasta {n} mm² de silicio en cada uno.", "Up to {0} mm² of silicon in each."],
  ["Área por die", "Area per die"],
  ["Suma de los bloques del die.", "Sum of the die's blocks."],
  ["Área total", "Total area"],
  ["Área por die por número de dies.", "Area per die times the number of dies."],
  ["Multiplicadores de pesos en total", "Weight multipliers in total"],
  ["Con un solo usuario solo trabajan los de un die cada vez.", "With a single user only one die's work at a time."],
  ["Memoria de pesos por celda", "Weight memory per cell"],
  ["Capacidad total repartida entre todas las celdas.", "Total capacity spread over all the cells."],
  ["Generación con un usuario", "Generation with one user"],
  ["Incluye MTP: {n} tokens por ronda con las tasas de aceptación medidas por MTPLX.", "Includes MTP: {0} tokens per round with the acceptance rates MTPLX measured."],
  ["Un token recorre todas las etapas antes de empezar el siguiente.", "A token goes through every stage before the next one starts."],
  ["Muestreo por token", "Sampling per token"],
  ["Filtra los logits mientras salen de lm_head; solo cuenta lo que no se solapa con el cálculo.", "Filters the logits as they leave lm_head; only what does not overlap the computation counts."],
  ["Sin especulación", "Without speculation"],
  ["Inverso del tiempo de un token.", "Inverse of the time of one token."],
  ["Lectura de prompt", "Prompt reading"],
  ["Los tokens ya se conocen, así que todos los dies trabajan a la vez.", "The tokens are already known, so every die works at once."],
  ["Imagen de 512×512", "512×512 image"],
  ["{n} G multiplicaciones del codificador por imagen, en los multiplicadores de un die.", "{0} G encoder multiplications per image, on one die's multipliers."],
  ["Frecuencia efectiva", "Effective clock"],
  ["Limitada a {n} GHz por las etapas del multiplicador.", "Limited to {0} GHz by the multiplier stages."],
  ["Ciclos por token", "Cycles per token"],
  ["Pesos {n}, vectores {n}, latencia {n}, enlaces {n}.", "Weights {0}, vectors {1}, latency {2}, links {3}."],
  ["Ancho de banda", "Bandwidth"],
  ["Pesos leídos por token", "Weights read per token"],
  ["Cada peso de las matrices se usa una vez por token.", "Each weight of the matrices is used once per token."],
  ["Ancho de banda de pesos necesario", "Weight bandwidth needed"],
  ["Pesos por token multiplicados por la velocidad.", "Weights per token times the speed."],
  ["Tráfico entre dies", "Die-to-die traffic"],
  ["Solo viaja el vector de activación entre capas.", "Only the activation vector travels between layers."],
  ["Caché KV", "KV cache"],
  ["Valores por token de contexto por bits y por tokens de contexto.", "Values per context token times bits times context tokens."],
  ["Estado de DeltaNet y convolución", "DeltaNet and convolution state"],
  ["Matrices recurrentes de las capas lineales e historia de las convoluciones.", "Recurrent matrices of the linear layers and the convolutions' history."],
  ["KV por token de contexto", "KV per context token"],
  ["Crece linealmente con el contexto.", "Grows linearly with the context."],
  ["Energía por token", "Energy per token"],
  ["Multiplicar, leer pesos, unidades vectoriales, estado y red.", "Multiplying, reading weights, vector units, state and network."],
  ["Potencia generando", "Power while generating"],
  ["Dinámica {n} W y fugas {n} W.", "Dynamic {0} W and leakage {1} W."],

  // FPGA boards.
  ["Memoria interna usada", "Internal memory used"],
  ["{n} de {n} Mbit: {n} Mbit del chip básico medidos en síntesis, más los pesos y cachés que viven dentro.", "{0} of {1} Mbit: {2} Mbit of the basic chip measured in synthesis, plus the weights and caches that live inside."],
  ["Lógica usada", "Logic used"],
  ["{n} K {s} del chip básico sintetizado con Yosys, de {n} K. Con más carriles y unidades crece; mídelo con la síntesis real.", "{0} K {1} of the basic chip synthesised with Yosys, of {2} K. It grows with more lanes and units; measure it with real synthesis."],
  ["Multiplicadores usados", "Multipliers used"],
  ["{n} productos por ciclo, cada uno en un bloque DSP, de {n}.", "{0} products per cycle, each in a DSP block, of {1}."],
  ["Reloj en esta placa", "Clock on this board"],
  ["Medido colocando y rutando el chip con nextpnr en esta familia.", "Measured by placing and routing the chip with nextpnr on this family."],
  ["Sin colocar y rutar aún en esta familia: una cifra prudente para lógica de FPGA sin optimizar a mano.", "Not yet placed and routed on this family: a cautious figure for FPGA logic not optimised by hand."],
  ["El chip básico ocupa {n} K {s} y esta FPGA tiene {n} K.", "The basic chip takes {0} K {1} and this FPGA has {2} K."],
  ["Elige una FPGA mayor.", "Choose a larger FPGA."],
  ["Lleva los pesos a la memoria de la placa: más lento, pero cabe.", "Move the weights to the board's memory: slower, but it fits."],
  ["Lleva la caché a la memoria de la placa, o reduce el contexto.", "Move the cache to the board's memory, or reduce the context."],
  ["Elige una FPGA con más memoria interna, o reduce bits de peso o contexto.", "Choose an FPGA with more internal memory, or reduce weight bits or context."],
  ["Ninguna placa del catálogo guarda tanto dentro: reduce bits de peso o contexto.", "No board in the catalogue holds that much inside: reduce weight bits or context."],
  ["Hacen falta {n} Mbit de memoria interna y esta FPGA tiene {n}.", "It needs {0} Mbit of internal memory and this FPGA has {1}."],
  ["Hay {n} productos por ciclo y {n} bloques DSP: el resto se hace con LUT, mucho más grandes.", "There are {0} products per cycle and {1} DSP blocks: the rest is done with LUTs, much larger."],
  ["Baja los multiplicadores por die.", "Lower the multipliers per die."],

  ["Placa con Cyclone IV EP4CE10", "Cyclone IV EP4CE10 board"],
  ["Quartus Prime Lite (gratuito)", "Quartus Prime Lite (free)"],
  ["Yosys + nextpnr, libres", "Yosys + nextpnr, open source"],
  ["Quartus Prime Pro (licencia gratuita para Agilex 3)", "Quartus Prime Pro (free licence for Agilex 3)"],
  ["Vivado (edición gratuita)", "Vivado (free edition)"],
  ["Vivado (licencia del dispositivo incluida)", "Vivado (device licence included)"],

  // Feasibility.
  ["DRAM apilada", "stacked DRAM"],
  ["memoria externa DDR", "external DDR memory"],
  ["{n} placas unidas por cable", "{0} boards joined by cable"],
  ["una placa", "one board"],
  ["Una lanzadera abierta entrega cada die en su propio encapsulado: {n} chips sobre {s}, unidos por sus pines de E/S. Cada token pasa {n} µs solo en los saltos, más lo que tarda el vector en cruzar los pines.", "An open shuttle delivers each die in its own package: {0} chips on {1}, joined by their I/O pins. Each token spends {2} µs on the hops alone, plus the time the vector takes to cross the pins."],
  ["{n} chips en {n} encapsulados no caben en una placa (unos {n} encapsulados por placa): harían falta {n} placas unidas por cable. Funciona, pero cada token cruza todos los enlaces: {n} µs por token solo en ellos.", "{0} chips in {1} packages do not fit on one board (about {2} packages per board): it would take {3} boards joined by cable. It works, but each token crosses every link: {4} µs per token in them alone."],
  ["{n} chips en {n} encapsulados sobre una placa, unidos por SerDes: {n} µs por token en enlaces.", "{0} chips in {1} packages on one board, joined by SerDes: {2} µs per token in links."],
  ["{n} chips en un solo encapsulado, unidos por UCIe.", "{0} chips in a single package, joined by UCIe."],
  ["Con los pesos grabados en ROM, cada uno de los {n} dies es un diseño distinto: en una lanzadera abierta ocupa una plaza propia. Con «Todos iguales» basta un diseño fabricado {n} veces.", "With the weights burnt into ROM, each of the {0} dies is a different design: on an open shuttle it takes a place of its own. With “All identical” one design made {1} times is enough."],
  ["{s} {n} mm²", "{0} {1} mm²"],
  [" aun repartido en {n} chips", " even spread over {0} chips"],
  ["Cada die necesita {n} mm²{s} y el proceso admite {n} mm²: {s}.", "Each die needs {0} mm²{1} and the process takes {2} mm²: {3}."],
  ["{s} no tiene un PDK abierto: se estima, pero no se puede hacer su layout. Para eso, SkyWater 130, GF180 o ASAP7.", "{0} has no open PDK: it is estimated, but it cannot be laid out. For that, SkyWater 130, GF180 or ASAP7."],
  ["Los pesos van en {s}, que el PDK abierto no ofrece: solo tiene macros de SRAM.", "The weights go in {0}, which the open PDK does not offer: it only has SRAM macros."],
  ["La caché KV va fuera del chip, y el PDK abierto no tiene la interfaz para esa memoria.", "The KV cache is off the chip, and the open PDK has no interface for that memory."],
  ["{s} es un proceso predictivo: su layout enseña cómo sería, pero ninguna fábrica lo produce.", "{0} is a predictive process: its layout shows what it would be like, but no foundry makes it."],
  ["Repartir el modelo en los chips que hagan falta", "Spread the model over as many chips as it takes"],
  ["Dimensionar para este modelo, {n} G parámetros", "Size it for this model, {0} G parameters"],
  ["El contexto nativo del modelo, {n} tokens", "The model's native context, {0} tokens"],
  ["Usar todo el die que admite el proceso, {n} mm²", "Use the whole die the process takes, {0} mm²"],
  ["calidad: pérdida pequeña", "quality: small loss"],
  ["calidad: pérdida apreciable sin cuantización cuidada", "quality: noticeable loss without careful quantization"],
  ["calidad: pérdida grande", "quality: large loss"],
  ["Pesos de {n} bits", "{0}-bit weights"],
  ["Contexto de {n} tokens", "Context of {0} tokens"],
  ["conversaciones más cortas", "shorter conversations"],
  ["Caché KV de 8 bits", "8-bit KV cache"],
  ["calidad: pérdida pequeña en contextos largos", "quality: small loss in long contexts"],
  ["Una sola unidad de función por die", "A single function unit per die"],
  ["Tablas de función de {n} bits de entrada", "Function tables of {0} input bits"],
  ["precisión de SiLU, exp y raíz algo menor", "somewhat less precise SiLU, exp and root"],
  ["{n} multiplicadores por die", "{0} multipliers per die"],
  ["Velocidad objetivo de {n} tok/s", "Target speed of {0} tok/s"],
  ["menos tok/s", "fewer tok/s"],
  ["{n} carriles vectoriales por die", "{0} vector lanes per die"],
  ["Pesos en SRAM, la memoria del PDK abierto", "Weights in SRAM, the open PDK's memory"],
  ["Caché KV en SRAM dentro del chip", "KV cache in SRAM on the chip"],
  ["Proceso {s}", "Process {0}"],
  ["Todos los dies iguales: un diseño fabricado {n} veces", "All dies identical: one design made {0} times"],
  ["algo más de área por die", "a little more area per die"],

  // Process names, as the engine's tech files give them.
  ["SkyWater 130 (abierto, fabricable, calibrado con OpenROAD)", "SkyWater 130 (open, manufacturable, calibrated with OpenROAD)"],
  ["GF180 (abierto, fabricable, calibrado con OpenROAD)", "GF180 (open, manufacturable, calibrated with OpenROAD)"],
  ["ASAP7 (predictivo, abierto, calibrado con OpenROAD)", "ASAP7 (predictive, open, calibrated with OpenROAD)"],
  ["TSMC N6 (estimado)", "TSMC N6 (estimated)"],
  ["TSMC N5 (estimado)", "TSMC N5 (estimated)"],
  ["TSMC N3 (estimado)", "TSMC N3 (estimated)"],
];

const NUMBER = "(-?[\\d.,]+)";
const TEXT = "(.*?)";

interface Pattern {
  regex: RegExp;
  kinds: ("n" | "s")[];
  english: string;
}

const escape = (text: string) => text.replace(/[.*+?^$()|[\]\\]/g, "\\$&");

const PATTERNS: Pattern[] = [];
const FIXED = new Map<string, string>();
for (const [spanish, english] of ENGINE_TEXT) {
  if (!/\{[ns]\}/.test(spanish)) {
    FIXED.set(spanish, english);
    continue;
  }
  const kinds: ("n" | "s")[] = [];
  const source = spanish
    .split(/(\{[ns]\})/)
    .map((piece) => {
      if (piece === "{n}" || piece === "{s}") {
        kinds.push(piece[1] as "n" | "s");
        return piece === "{n}" ? NUMBER : TEXT;
      }
      return escape(piece);
    })
    .join("");
  PATTERNS.push({ regex: new RegExp(`^${source}$`), kinds, english });
}
// The more fixed text a pattern has, the more surely it is the one meant.
PATTERNS.sort((a, b) => b.regex.source.replace(/\(.*?\)/g, "").length - a.regex.source.replace(/\(.*?\)/g, "").length);

/** The English of an engine text, or null if it is not one of the engine's. */
export function englishOf(text: string): string | null {
  const fixed = FIXED.get(text);
  if (fixed !== undefined) return fixed;
  for (const pattern of PATTERNS) {
    const match = pattern.regex.exec(text);
    if (!match) continue;
    const values = match.slice(1).map((value, i) => (pattern.kinds[i] === "s" ? englishOfPart(value) : value));
    return pattern.english.replace(/\{(\d+)\}/g, (_, i: string) => values[Number(i)] ?? "");
  }
  return null;
}

/** A piece inside a text: one of the engine's, a list of them, or kept as it is. */
function englishOfPart(text: string): string {
  const whole = englishOf(text);
  if (whole !== null) return whole;
  if (text.includes(", ")) return text.split(", ").map((part) => englishOf(part) ?? part).join(", ");
  return text;
}
