/** Parameters that are not sliders of the designer: what the gate generator derives on its
 * own, what the sizing search takes, and what each generation is asked for. */
export interface DocEntry {
  name: string;
  text: string;
}

export interface DocSection {
  id: string;
  title: string;
  intro: string;
  entries: DocEntry[];
}

export const EXTRA_DOCS: DocSection[] = [
  {
    id: "generador",
    title: "Generador de puertas",
    intro:
      "El circuito que se simula puerta a puerta y se exporta a Verilog sale de estos parámetros. La mayoría no se eligen a mano: el generador los deduce del modelo y del microprograma para que el die tenga justo lo que necesita.",
    entries: [
      { name: "Celdas por die", text: "Cada celda tiene su memoria de pesos y su multiplicador. Más celdas reparten cada producto matriz-vector entre más multiplicadores: menos ciclos por token y más área. Debe haber al menos tantas celdas por unidad como carriles." },
      { name: "Carriles por unidad", text: "Elementos que una unidad vectorial procesa por ciclo en las operaciones elemento a elemento (sumas, normas, funciones). En potencias de dos." },
      { name: "Unidades vectoriales", text: "Copias de la unidad vectorial que ejecutan las mismas instrucciones sobre datos distintos: cada una toma una cabeza de atención o de DeltaNet, o un trozo de un vector. Se unen por una red en árbol que suma y busca máximos." },
      { name: "Profundidad de la memoria de cada celda", text: "Palabras de pesos y grupos de escala y sesgo que guarda cada celda. Sale de repartir todas las matrices del chip entre sus celdas." },
      { name: "Códigos por palabra", text: "La palabra de la celda es tan ancha como el código más ancho del modelo (8 bits en el Qwen de MLX). Cada matriz guarda 1, 2 o 4 códigos por palabra según su propio ancho (q8, q4, q2), y el microcódigo le dice a la celda cuántos al apuntarla. Se desactiva con ECC, que protege palabras enteras." },
      { name: "Ciclos de escala y sesgo (K)", text: "Cada celda aplica la escala y el sesgo de un grupo en K ciclos, unos pocos bits por ciclo, mientras suma el grupo siguiente. Reduce el multiplicador de escala a 1/K de su tamaño; solo cuesta ciclos si el grupo tiene menos de K pesos." },
      { name: "Multiplicadores", text: "Las celdas usan Booth radix-4 (menos productos parciales) y los carriles Wallace (árbol sin propagación de acarreo). También hay array (compacto, lento) y múltiplos precalculados. Los sumadores de 8 bits o más son de prefijo Brent-Kung." },
      { name: "Registros vectoriales e índice", text: "El microprograma nombra cada dato por registro e índice. Una tabla, cargada con la imagen, da a cada registro su base en la memoria vectorial y el paso de los bucles que recorren registros por número; el compilador calcula hasta dónde llega cada registro en sus rutinas, así que cada uno ocupa solo lo que usa. Sin la tabla, cada registro reservaría tanto como el más largo." },
      { name: "Microcódigo", text: "El programa fijo del die, con un punto de entrada por rutina (token, logits, cabeza MTP…). Cada campo de la palabra de control solo tiene los bits que el programa usa, y cada operación que no aparece no se genera." },
      { name: "Columnas de claves", text: "Con varias unidades, las claves de la caché KV se guardan en columnas de 2^n tokens para que cada unidad puntúe su parte de los tokens a la vez." },
      { name: "Tablas de función", text: "Bits de índice de cada tabla (resolución) y cuántas tablas hay (exp, SiLU, sigmoide, softplus, raíz inversa, tanh…)." },
      { name: "Pesos en ROM", text: "Con el modelo grabado en fábrica desaparecen la escritura de pesos y su carga, y la memoria es más densa, pero cambiar de modelo exige otras máscaras." },
      { name: "ECC en celdas", text: "Corrige un bit y detecta dos por palabra de pesos. Cuesta unos 430 equivalentes de puerta por celda y bits de control." },
      { name: "Saltar ceros", text: "Congela la entrada del multiplicador cuando el peso es cero para que no conmute: ahorra energía, no área." },
      { name: "CPU de control", text: "Un RISC-V RV32IM con su memoria de programa y datos: arranca el chip, carga la imagen con CRC, tokeniza, gobierna el die por sus registros, muestrea y aplica la lista de parada y el presupuesto de razonamiento." },
      { name: "Unidad de muestreo", text: "Logits que compara por ciclo y candidatos que guarda (el mayor top-k). Elige el token con temperatura usando una tabla de exponenciales." },
      { name: "Enlace entre chips", text: "Cada palabra viaja con número de secuencia y código corrector; el receptor tiene una cola y pide repeticiones, así que la placa puede retrasar palabras sin romper nada." },
      { name: "Enlace ancho", text: "Con varios chips, el enlace tiene un carril propio por cada carril vectorial, cada uno con su secuencia, su corrección de errores y su cola, como los carriles de UCIe: un bloque entero cruza en un ciclo. El vector oculto pasa de un chip al siguiente en decenas de ciclos en lugar de miles. Se genera solo si el microprograma envía bloques." },
    ],
  },
  {
    id: "reparto",
    title: "Reparto entre chips",
    intro:
      "Cuando el modelo no cabe en un chip, o se quiere más velocidad, hay tres formas de repartirlo. Se elige con «Reparto entre chips» y, en el mixto, «Chips por grupo».",
    entries: [
      { name: "Por capas, en cadena", text: "Cada chip guarda un tramo de capas seguidas. Para cada token, el primero hace sus capas y pasa el vector oculto (unos miles de valores) al segundo, y así hasta el último, que lo devuelve al primero para los logits. Ventajas: poca comunicación, que tolera enlaces lentos por la placa; sin sincronización dentro de una capa; cada chip guarda solo los pesos y la caché de sus capas; al leer un prompt todos trabajan a la vez, cada uno en un token. Inconveniente: generando para un solo usuario solo trabaja un chip cada vez, así que más chips dan más memoria, no más velocidad. El generador de puertas lo implementa y lo prueba en puertas con dos chips sobre un enlace ruidoso." },
      { name: "Por tensor, todos a la vez", text: "Cada chip guarda una parte de cada matriz y de cada grupo de cabezas, y todos calculan cada capa a la vez. Tras la atención y tras la MLP de cada capa suman sus resultados parciales en anillo: 2·(N−1) pasos, cada uno con una parte N-ésima del vector. Ventaja: con un usuario la velocidad crece casi con el número de chips. Inconvenientes: muchas sincronizaciones por token, que exigen enlaces rápidos y de poca latencia, en la práctica dentro del mismo encapsulado; un prompt no se lee en cadena. El estimador modela la variante más eficiente, que reparte también las cabezas de atención. El generador de puertas implementa hoy una variante más sencilla, «por filas»: cada chip guarda una franja igual de filas de cada matriz y, tras cada producto, los chips se pasan sus franjas en anillo hasta que todos tienen el vector entero. Antes de estrechar cada producto acuerdan el desplazamiento por el enlace (dos palabras por chip), así que el resultado es idéntico bit a bit al de un solo chip; la prueba en puertas lo comprueba con dos chips y enlaces ruidosos. La atención, DeltaNet y las normas las calcula entera cada chip: los pesos se reparten, pero la caché KV y el estado se repiten en todos, y se mueve más por el enlace que en la variante eficiente." },
      { name: "Mixto", text: "Grupos de chips que comparten cada capa en tensor, unidos en cadena. Es lo que usan los sistemas grandes: cada grupo dentro de un encapsulado, con UCIe, y los grupos por la placa. Con un grupo de 1 chip es el reparto por capas; con todos en un grupo, el reparto por tensor. El estimador lo modela; el generador de puertas aún no: una imagen de arranque pedida en modo mixto se genera por capas." },
    ],
  },
  {
    id: "dimensionado",
    title: "Dimensionado",
    intro:
      "La búsqueda prueba formas de die (celdas, carriles, unidades, número de chips), genera para cada una su microprograma real y cuenta sus ciclos en la última posición del contexto, así que los tok/s salen del circuito y están garantizados con el contexto lleno.",
    entries: [
      { name: "Tok/s objetivo", text: "La velocidad mínima con el contexto lleno. La búsqueda se queda con el die de menos silicio que la alcanza." },
      { name: "Tiempo al primer token", text: "El máximo tras un prompt de cierta longitud. Con varios chips, el prompt avanza en cadena: cada chip trabaja en un token distinto a la vez." },
      { name: "Contexto", text: "Tokens que caben a la vez. Fija la caché KV, la tabla RoPE y la posición donde se miden los tok/s." },
      { name: "Chips", text: "En cuántos chips iguales se reparten las capas. Se cortan para que el chip que más memoria necesita tenga la menor posible: el primero también guarda lm_head, así que lleva menos capas." },
      { name: "Reloj máximo", text: "Cada die va a su frecuencia más alta según su camino crítico medido, o a esta si es menor." },
      { name: "Proceso", text: "Retardo de un sumador completo, área de una puerta NAND2 con su cableado, área de un bit de SRAM y su periferia, y tamaño máximo de die (retícula: 858 mm² en N6). Un die mayor no se puede fabricar y se descarta." },
      { name: "Más eficiente y equilibrio", text: "De la frontera velocidad-área se proponen el punto con menos área por tok/s y el último en que cada tok/s extra no cuesta más de cierto múltiplo del área media por tok/s." },
    ],
  },
  {
    id: "generacion",
    title: "Generación",
    intro: "Se eligen en cada petición y no cambian el hardware.",
    entries: [
      { name: "Tokens máximos", text: "Cuántos tokens genera como mucho; también para antes en cualquier token de la lista de parada del modelo." },
      { name: "Temperatura", text: "0 elige siempre el más probable. Más alta reparte la probabilidad entre más candidatos." },
      { name: "Top-k", text: "Entre cuántos de los mejores candidatos se sortea; no puede pasar de los candidatos de la unidad de muestreo." },
      { name: "Razonar antes de responder", text: "Con la plantilla de chat del modelo, deja abierto su bloque de razonamiento; sin él, el bloque llega cerrado y contesta directamente. Es texto del prompt." },
      { name: "Presupuesto de razonamiento", text: "Cuántos tokens puede razonar antes de que el firmware cierre el razonamiento por él. Los tokens de apertura y cierre salen del tokenizador de cada modelo." },
      { name: "Profundidad MTP", text: "Cuántos tokens adelanta la cabeza de predicción múltiple; se mide cuántos acierta." },
      { name: "Fidelidad", text: "Cuánto del chip se simula: firmware real instrucción a instrucción, unidad de muestreo, tokenizador en el firmware y carga de la imagen por transacciones." },
    ],
  },
];
