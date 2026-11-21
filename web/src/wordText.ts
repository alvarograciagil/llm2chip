// What a microcode word makes the die do, in Spanish. The engine (`describe_word` in
// crates/netlist/src/die.rs) writes it in English as pieces joined by "; ", so each piece is
// recognised on its own and the sentence put back together. A test keeps this in step with
// the engine's source.

/** A register, maybe with an element: `r12` or `r3[4]`. */
const R = String.raw`(r\d+(?:\[\d+\])?)`;
const N = String.raw`(\d+)`;

/** Words that are a whole sentence of their own. */
const WHOLE: Record<string, string> = {
  "End of the routine: the die tells the CPU it has finished.": "Fin de la rutina: el die avisa a la CPU de que ha terminado.",
  "Wait: lets a cycle go by while an earlier result finishes.": "Espera: deja pasar un ciclo mientras termina un resultado anterior.",
};

/** The tables of the function unit, as `applies the … table` names them. */
const TABLES: Record<string, string> = {
  exponential: "de la exponencial",
  sigmoid: "de la sigmoide",
  reciprocal: "del inverso",
  "inverse square root": "de la raíz cuadrada inversa",
  softplus: "de softplus",
  GELU: "de GELU",
  "hyperbolic tangent": "de la tangente hiperbólica",
  function: "de funciones",
};

type Rule = [RegExp, (m: RegExpExecArray) => string];
const rule = (source: string, spanish: (m: RegExpExecArray) => string): Rule => [new RegExp(`^${source}$`), spanish];

/** What the word computes, before where it goes. */
const WHAT: Rule[] = [
  rule(`adds ${R} \\+ ${R}`, (m) => `suma ${m[1]} + ${m[2]}`),
  rule(`subtracts ${R} − ${R}`, (m) => `resta ${m[1]} − ${m[2]}`),
  rule(`multiplies ${R} × ${R}`, (m) => `multiplica ${m[1]} × ${m[2]}`),
  rule(`applies the (.+) table to ${R}`, (m) => (m[1] in TABLES ? `aplica la tabla ${TABLES[m[1]]} a ${m[2]}` : "")),
  rule(`collects the cells' result`, () => "recoge el resultado de las celdas"),
  rule(`finds the maximum of ${R}`, (m) => `busca el máximo de ${m[1]}`),
  rule(`counts the leading zeros of ${R} to normalize it`, (m) => `cuenta los ceros a la izquierda de ${m[1]} para normalizarlo`),
  rule(`copies ${R}`, (m) => `copia ${m[1]}`),
  rule(`shifts ${R} left`, (m) => `desplaza ${m[1]} a la izquierda`),
  rule(`reads the unit's accumulator`, () => "lee el acumulador de la unidad"),
  rule(`takes a constant from the instruction itself`, () => "toma una constante de la propia instrucción"),
  rule(`computes the parity of ${R}`, (m) => `calcula la paridad de ${m[1]}`),
  rule(`receives a word from the previous chip`, () => "recibe una palabra del die anterior"),
  rule(`reads the wide memory and rounds it`, () => "lee la memoria ancha y la redondea"),
  rule(`compares ${R} with ${R}`, (m) => `compara ${m[1]} con ${m[2]}`),
];

/** The other pieces of a word. */
const PIECES: Rule[] = [
  rule(`the cells move to the start of a matrix \\(weights at word ${N}\\)`, (m) => `las celdas van al principio de una matriz (pesos en la palabra ${m[1]})`),
  rule(`the cells receive ${R} over the activation network, multiply it by their weight and accumulate it`, (m) => `las celdas reciben ${m[1]} por la red de activaciones, lo multiplican por su peso y lo acumulan`),
  rule(`the cells start a new sum`, () => "las celdas empiezan una suma nueva"),
  rule(`the sums move to each unit's wide memory`, () => "las sumas pasan a la memoria ancha de cada unidad"),
  rule(`adds it to the accumulator`, () => "suma el resultado al acumulador"),
  rule(`sends it back to the group's first chip`, () => "devuelve el resultado al primer die del grupo"),
  rule(`sends it to the next chip`, () => "envía el resultado al die siguiente"),
  rule(`takes it from the return link of the group's last chip`, () => "la recoge del enlace de vuelta del último die del grupo"),
  rule(`repeats the step ${N} more times?, moving through the elements`, (m) => `repite el paso ${m[1]} ${m[1] === "1" ? "vez" : "veces"} más, recorriendo los elementos`),
  rule(`if counter x${N} has not reached its limit, goes back to word ${N}`, (m) => `si el contador x${m[1]} no ha llegado a su límite, vuelve a la palabra ${m[2]}`),
];

function apply(rules: Rule[], text: string): string | null {
  for (const [regex, spanish] of rules) {
    const m = regex.exec(text);
    if (m) return spanish(m) || null;
  }
  return null;
}

/** One piece: an operation, maybe on every lane and stored somewhere, or one of the others. */
function piece(text: string): string | null {
  const other = apply(PIECES, text);
  if (other) return other;
  const stored = / and stores it in (r\d+(?:\[\d+\])?)$/.exec(text);
  let rest = stored ? text.slice(0, stored.index) : text;
  const lanes = rest.endsWith(" on every lane at once");
  if (lanes) rest = rest.slice(0, -" on every lane at once".length);
  const what = apply(WHAT, rest);
  if (!what) return null;
  return what + (lanes ? " en todos los carriles a la vez" : "") + (stored ? ` y guarda el resultado en ${stored[1]}` : "");
}

/** The Spanish of a word's description, or null if the text is not one. */
export function spanishWord(text: string): string | null {
  if (text in WHOLE) return WHOLE[text];
  if (!text.endsWith(".")) return null;
  const pieces = text.slice(0, -1).split("; ");
  pieces[0] = pieces[0].charAt(0).toLowerCase() + pieces[0].slice(1);
  const spanish = pieces.map(piece);
  if (spanish.some((p) => p === null)) return null;
  const sentence = spanish.join("; ");
  return sentence.charAt(0).toUpperCase() + sentence.slice(1) + ".";
}
