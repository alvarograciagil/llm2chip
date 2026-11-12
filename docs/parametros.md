# Parámetros del chip

<!-- Generado desde web/src/params.ts y web/src/paramDocs.ts: UPDATE_DOCS=1 npx vitest run docs -->

Todo lo que se puede ajustar, qué cambia en el circuito y en qué se nota. Los del panel de la aplicación mueven el estimador al momento; los del generador los deduce el compilador del modelo y del microprograma; los de generación se eligen en cada petición. En la aplicación, cada parámetro se explica al pasar el cursor por su «?» y abre esta misma documentación al pulsarlo.

## Objetivo

### Parámetros máximos (`max_params_g`)

Tamaño del mayor modelo que el chip puede alojar, hasta 100 billones de parámetros (100.000 G). Dimensiona la memoria de pesos.

*De 0.0001 a 100000 G, en escala logarítmica.*

### Contexto (`context_tokens`)

Hasta 1 M de tokens. La caché KV crece linealmente con el contexto y es el bloque que más limita la velocidad; a partir de unos 32 K conviene llevarla a HBM4.

*De 128 a 1048576 tokens, en escala logarítmica.*

### Soporte MTP (`mtp_enabled`)

Añade la cabeza de predicción múltiple (MTP): una capa extra que adivina los siguientes tokens mientras el modelo principal comprueba los anteriores. Con un solo usuario los dies esperan mucho, y este trabajo aprovecha ese hueco: si acierta, salen varios tokens por pasada.

*Activado o desactivado.*

### Profundidad MTP (`mtp_depth`)

Cuántos tokens adelanta la cabeza MTP en cada ronda. Más profundidad da más tokens por pasada cuando acierta, pero cada nivel acierta menos (en Qwen, en torno al 94 %, 86 % y 74 %) y exige guardar estado para deshacer.

*De 1 a 3.*

### Incluir visión (`include_vision`)

Reserva área para el codificador de imágenes (un ViT de unos 400 M de parámetros en Qwen) que convierte cada imagen en tokens para el modelo de lenguaje.

*Activado o desactivado.*

## Organización

### Velocidad objetivo (`target_tok_s`)

Con el tamaño del modelo y la frecuencia, determina cuántos multiplicadores hacen falta. Con 0 los fijas a mano.

*De 0 a 40000 tok/s.*

### Multiplicadores por die (`macs_per_die`)

Con un solo usuario solo trabaja un die cada vez, así que este valor marca la velocidad.

*De 2 a 2000000, en escala logarítmica.*

### Frecuencia (`freq_ghz`)

Reloj del die. Lo limita su camino más lento entre registros: el multiplicador y el árbol de suma (más etapas lo acortan) y el retardo de las puertas del nodo. Más frecuencia da más tok/s con la misma lógica, pero más consumo.

*De 0.02 a 3.4 GHz, en escala logarítmica.*

### Tipo de dies (`die_mode`)

En los dos casos es el mismo chip repetido, que carga su parte del modelo desde la imagen. A medida dimensiona el die para la parte de este modelo; ampliable reserva sitio (capas, ancho, vocabulario) para conectar más chips con modelos mayores. Solo con los pesos grabados en ROM es distinto cada die.

*Opciones: A medida de este modelo, Todos iguales (ampliable).*

### Capas máximas (`max_layers`)

Las capas salen de cada modelo: Qwen3.8 tiene 64 y Llama 3.1 8B, 32. Este límite reserva su estado en cada die.

*De 1 a 1024, en escala logarítmica.*

### Ancho interno máximo (`max_hidden_size`)

Tamaño máximo del vector que recorre las capas (5.120 en Qwen3.8-27B). Con dies iguales, fija el ancho de búferes y enlaces para cualquier modelo que se cargue.

*De 64 a 65536, en escala logarítmica.*

### Vocabulario máximo (`max_vocabulary`)

Tokens distintos que admite el modelo mayor (248.320 en Qwen). Dimensiona lm_head, la unidad de muestreo y las tablas del tokenizador.

*De 256 a 4194304, en escala logarítmica.*

### Dies (`dies`)

En cuántos dies se reparte el modelo. Con 0 se elige el mínimo con el que cada die cabe en el área máxima. Con un solo usuario los dies trabajan por turnos, así que más dies dan más memoria pero no más velocidad por sí solos.

*De 0 a 32.*

### Reparto entre chips (`chip_split`)

Cómo se reparten el modelo varios chips. Por capas: cada chip guarda unas capas enteras y pasa el vector oculto al siguiente; poca comunicación, pero con un solo usuario trabaja un chip cada vez. Por tensor: cada chip guarda una parte de todas las capas y todos calculan cada capa a la vez; mucho más rápido para un usuario, pero suman sus resultados dos veces por capa y necesitan enlaces rápidos, dentro del encapsulado. Mixto: grupos de chips en tensor unidos en cadena, para que cada grupo quepa en un encapsulado.

*Opciones: Por capas, en cadena, Por tensor, todos a la vez, Mixto: grupos en tensor, en cadena.*

### Chips por grupo (`tensor_group`)

En el reparto mixto, cuántos chips comparten cada capa. Los grupos se unen en cadena. Con 1 es el reparto por capas; con todos los chips, por tensor. Conviene que un grupo quepa en un encapsulado, unido por UCIe.

*De 1 a 32.*

### Área máxima por die (`die_max_mm2`)

Área máxima de cada die. El límite físico es la retícula del escáner (26 × 33 mm, unos 858 mm²). Los dies grandes rinden menos por oblea, así que a veces conviene un límite menor.

*De 10 a 858 mm².*

## Memoria

### Pesos (`weight_backend`)

Con SRAM el chip admite otros modelos. Con ROM los pesos se graban al fabricarlo: desaparecen la escritura de pesos y su tiempo de carga, pero cambiar de modelo exige otras máscaras.

*Opciones: SRAM (modelo intercambiable), ROM (modelo grabado), DRAM apilada, HBM4, Memoria externa.*

### Caché y estado (`kv_backend`)

Dónde viven la caché KV de la atención y el estado recurrente de DeltaNet. En SRAM son lo más rápido pero ocupan área que crece con el contexto; fuera, el ancho de banda de esa memoria limita la velocidad.

*Opciones: SRAM, HBM4, Memoria externa.*

### Pilas HBM4 (`hbm_stacks`)

Cada pila de HBM4 suma capacidad y ancho de banda, y un PHY de unos 12 mm² en el die que la conecta.

*De 1 a 16.*

### Ancho de banda DRAM (`stacked_dram_tbps`)

Lo que la DRAM apilada sobre el die entrega por segundo. Con los pesos fuera de la SRAM, marca la velocidad: cada token lee todos los pesos una vez.

*De 10 a 400 TB/s, en escala logarítmica.*

### Ancho de banda externo (`ddr_gbps`)

Lo que la memoria externa (DDR, LPDDR) entrega por segundo. Con los pesos fuera, los tok/s no pasan de este valor dividido por el tamaño del modelo.

*De 0.01 a 400 GB/s, en escala logarítmica.*

### Capacidad externa (`ddr_capacity_gb`)

Memoria externa disponible: debe alojar lo que se lleva fuera del chip (pesos, caché KV o ambos).

*De 0.004 a 256 GB, en escala logarítmica.*

## Aritmética

### Formato de pesos (`weight_coding_preset`)

El circuito se adapta: cambia la memoria, el multiplicador y el decodificador de cada celda.

*Formatos ya hechos: Afín por grupos (MLX), GGUF, Punto fijo (Q), Coma flotante, Microescalado, Posit, Logarítmico, Libro de códigos.*

### Codificación (`weight_coding`)

Cómo se convierte cada código guardado en un número. Cada familia genera su propio decodificador.

*Opciones: Afín por grupos, Punto fijo, Coma flotante, Posit, Logarítmico, Libro de códigos.*

### Bits de peso (`weight_bits`)

Bits guardados por peso, de 2 a 32. Fija la memoria de pesos, que domina el área, y el tamaño del multiplicador de cada celda. Con menos bits cabe más modelo pero se pierde precisión.

*De 2 a 32.*

### Bits de exponente (`weight_exponent_bits`)

En coma flotante, los bits de exponente. En posit, «es». En logarítmico, los bits de fracción.

*De 1 a 15.*

### Mantener tensores de 8 bits (`use_model_mix`)

Reproduce la mezcla de los pesos reales de Qwen en MLX: la mayoría a 4 bits y embed_tokens, lm_head, out_proj y la MLP de las últimas capas a 8. Las celdas guardan dos códigos de 4 bits en cada palabra de 8.

*Activado o desactivado.*

### Sesgo por grupo (`weight_bias`)

Si cada grupo guarda también un sesgo (peso = código · escala + sesgo). MLX y Q4_1 lo hacen; Q4_0 solo guarda escala. El sesgo cuesta memoria y una suma por grupo.

*Activado o desactivado.*

### Escala fina por grupo (`weight_fine_scale`)

Una escala en coma flotante como NVFP4 necesita un multiplicador. Una potencia de dos como MXFP4 solo un desplazador, pero pierde precisión.

*Activado o desactivado.*

### Pesos por grupo (`group_size`)

Pesos consecutivos que comparten escala y sesgo (32 en el Qwen de MLX). Grupos pequeños dan más precisión pero más factores que guardar y aplicar. Con 0 no hay grupos.

*De 0 a 256.*

### Bits de escala y sesgo (`scale_bits`)

Precisión con que cada celda guarda la escala y el sesgo de cada grupo. Más bits: menos error y más memoria y multiplicador de escala.

*De 8 a 32.*

### Bits de activación (`activation_bits`)

Ancho de los valores que recorren el chip entre operaciones. Fija el ancho de los multiplicadores, de la red de activaciones y de la memoria vectorial: es de lo que más pesa en área y en precisión.

*De 8 a 32.*

### Bits de acumulador (`accumulator_bits`)

Ancho del acumulador de cada celda, que suma productos de peso por activación. Si es corto, las sumas se desbordan; si el formato necesita más, el chip usa el ancho necesario.

*De 16 a 128.*

### Bits de estado DeltaNet (`delta_state_bits`)

Ancho del estado recurrente de DeltaNet, una matriz por cabeza que se actualiza en cada token. La referencia usa 32 bits; reducirlo ahorra mucha memoria, pero el error se acumula a lo largo del contexto y es el mayor riesgo numérico.

*De 8 a 32.*

### Bits de caché KV (`kv_bits`)

Ancho de cada clave y valor guardado por token. La caché crece con el contexto, así que con contextos largos es la memoria que más ocupa después de los pesos.

*De 4 a 32.*

## Etapas

### Circuito a medida (`auto_circuit`)

El generador prueba cada forma de construir la celda para estos parámetros, mide su circuito real y se queda con la mejor.

*Activado o desactivado.*

### Priorizar (`circuit_goal`)

Qué se minimiza entre las formas de celda que alcanzan la frecuencia pedida: área, energía por token o un equilibrio de ambas.

*Opciones: Área y energía, Área, Energía.*

### Tipo de multiplicador (`multiplier_kind`)

Wallace y Booth reducen los productos con sumadores sin propagación y son mucho más rápidos. El array es más simple y compacto a baja frecuencia.

*Opciones: Wallace, Booth radix-4, Array, Múltiplos precalculados.*

### Multiplicador (`mac_stages`)

Más etapas permiten mayor frecuencia a cambio de área y latencia.

*De 1 a 6.*

### Ciclos de escala y sesgo (`factor_steps`)

Cada celda aplica la escala y el sesgo de un grupo en tantos ciclos, unos pocos bits por ciclo, mientras ya suma el grupo siguiente. Con 1 hace falta un multiplicador entero por celda; con 8 basta uno ocho veces menor, sin perder velocidad si el grupo tiene al menos 8 pesos.

*De 1 a 16.*

### Árbol de suma (`adder_stages`)

Registros dentro del árbol que suma los productos de las celdas. Más etapas permiten más frecuencia a cambio de latencia y biestables.

*De 1 a 6.*

### Red de activaciones (`noc_stages`)

Registros en la red que reparte cada activación a todas las celdas. En dies grandes los cables son largos: más etapas evitan que limiten el reloj.

*De 1 a 4.*

## Unidades vectoriales

### Carriles por die (`vector_lanes_per_die`)

Carriles de las unidades vectoriales, que multiplican activación por activación: atención Q·K, regla delta, puertas y normas. Pocos carriles alargan las partes que no son producto de matriz.

*De 1 a 512000, en escala logarítmica.*

### Unidades vectoriales por die (`vector_units_per_die`)

Copias de la unidad vectorial: cada una calcula una cabeza de atención o DeltaNet a la vez.

*De 1 a 256, en escala logarítmica.*

### Logits por ciclo al muestrear (`sampler_lanes`)

Cuántos logits compara a la vez la unidad de muestreo. Si son pocos, elegir el token retrasa al chip.

*De 1 a 4096, en escala logarítmica.*

### Candidatos del muestreo (`sampler_entries`)

Cuántos de los mejores tokens guarda la unidad de muestreo: el mayor top-k que admite. Cada candidato cuesta registros y ciclos al elegir.

*De 1 a 1024, en escala logarítmica.*

### Bits por carril (`vector_bits`)

Ancho de los multiplicadores de las unidades vectoriales, que operan activación por activación (atención, regla delta, normas). El área crece con su cuadrado.

*De 8 a 32.*

### Unidades de función (`function_units_per_die`)

Tablas que calculan funciones no lineales (exp, SiLU, sigmoide, raíz inversa…) consultando e interpolando. Más unidades aplican la función a más elementos a la vez.

*De 1 a 128.*

### Resolución de las tablas (`function_table_bits`)

Entradas de cada tabla de función, en potencias de dos: 12 son 4.096. Más entradas reducen el error de exp, SiLU o la raíz inversa a cambio de memoria en cada unidad.

*De 4 a 20.*

## Sistema

### Enlace entre dies (`link`)

UCIe une dies dentro de un mismo encapsulado: rápido (unos 10 ns por salto) y pequeño. SerDes los une por la placa: lento (unos 150 ns) y mayor. Los pines de E/S son lo único que tienen los chips de una lanzadera abierta, cada uno en su encapsulado: 16 pines por sentido a 50 MHz y unos 60 ns por salto. En automático se llenan encapsulados y se unen entre sí por SerDes, salvo en una lanzadera abierta, que usa los pines.

*Opciones: Automático según encapsulado, UCIe, SerDes en placa, Pines de E/S.*

### Ancho del enlace (`link_words`)

Cuántas activaciones pasa cada enlace por ciclo. El vector oculto cruza cada frontera entre dies en cada token: un enlace estrecho añade miles de ciclos por frontera; uno ancho necesita más módulos UCIe o SerDes.

*De 1 a 4096 activaciones/ciclo, en escala logarítmica.*

### Carga de pesos (`load_interface`)

Por dónde llegan los pesos al encender el chip: NVMe, PCIe 5.0 o UART. Solo afecta al tiempo de arranque, no a la velocidad de generación.

*Opciones: NVMe, PCIe 5.0 x16, UART.*

## Generador de puertas

El circuito que se simula puerta a puerta y se exporta a Verilog sale de estos parámetros. La mayoría no se eligen a mano: el generador los deduce del modelo y del microprograma para que el die tenga justo lo que necesita.

### Celdas por die

Cada celda tiene su memoria de pesos y su multiplicador. Más celdas reparten cada producto matriz-vector entre más multiplicadores: menos ciclos por token y más área. Debe haber al menos tantas celdas por unidad como carriles.

### Carriles por unidad

Elementos que una unidad vectorial procesa por ciclo en las operaciones elemento a elemento (sumas, normas, funciones). En potencias de dos.

### Unidades vectoriales

Copias de la unidad vectorial que ejecutan las mismas instrucciones sobre datos distintos: cada una toma una cabeza de atención o de DeltaNet, o un trozo de un vector. Se unen por una red en árbol que suma y busca máximos.

### Profundidad de la memoria de cada celda

Palabras de pesos y grupos de escala y sesgo que guarda cada celda. Sale de repartir todas las matrices del chip entre sus celdas.

### Códigos por palabra

La palabra de la celda es tan ancha como el código más ancho del modelo (8 bits en el Qwen de MLX). Cada matriz guarda 1, 2 o 4 códigos por palabra según su propio ancho (q8, q4, q2), y el microcódigo le dice a la celda cuántos al apuntarla. Se desactiva con ECC, que protege palabras enteras.

### Ciclos de escala y sesgo (K)

Cada celda aplica la escala y el sesgo de un grupo en K ciclos, unos pocos bits por ciclo, mientras suma el grupo siguiente. Reduce el multiplicador de escala a 1/K de su tamaño; solo cuesta ciclos si el grupo tiene menos de K pesos.

### Multiplicadores

Las celdas usan Booth radix-4 (menos productos parciales) y los carriles Wallace (árbol sin propagación de acarreo). También hay array (compacto, lento) y múltiplos precalculados. Los sumadores de 8 bits o más son de prefijo Brent-Kung.

### Registros vectoriales e índice

El microprograma nombra cada dato por registro e índice. Una tabla, cargada con la imagen, da a cada registro su base en la memoria vectorial y el paso de los bucles que recorren registros por número; el compilador calcula hasta dónde llega cada registro en sus rutinas, así que cada uno ocupa solo lo que usa. Sin la tabla, cada registro reservaría tanto como el más largo.

### Microcódigo

El programa fijo del die, con un punto de entrada por rutina (token, logits, cabeza MTP…). Cada campo de la palabra de control solo tiene los bits que el programa usa, y cada operación que no aparece no se genera.

### Columnas de claves

Con varias unidades, las claves de la caché KV se guardan en columnas de 2^n tokens para que cada unidad puntúe su parte de los tokens a la vez.

### Tablas de función

Bits de índice de cada tabla (resolución) y cuántas tablas hay (exp, SiLU, sigmoide, softplus, raíz inversa, tanh…).

### Pesos en ROM

Con el modelo grabado en fábrica desaparecen la escritura de pesos y su carga, y la memoria es más densa, pero cambiar de modelo exige otras máscaras.

### ECC en celdas

Corrige un bit y detecta dos por palabra de pesos. Cuesta unos 430 equivalentes de puerta por celda y bits de control.

### Saltar ceros

Congela la entrada del multiplicador cuando el peso es cero para que no conmute: ahorra energía, no área.

### CPU de control

Un RISC-V RV32IM con su memoria de programa y datos: arranca el chip, carga la imagen con CRC, tokeniza, gobierna el die por sus registros, muestrea y aplica la lista de parada y el presupuesto de razonamiento.

### Unidad de muestreo

Logits que compara por ciclo y candidatos que guarda (el mayor top-k). Elige el token con temperatura usando una tabla de exponenciales.

### Enlace entre chips

Cada palabra viaja con número de secuencia y código corrector; el receptor tiene una cola y pide repeticiones, así que la placa puede retrasar palabras sin romper nada.

### Enlace ancho

Con varios chips, el enlace tiene un carril propio por cada carril vectorial, cada uno con su secuencia, su corrección de errores y su cola, como los carriles de UCIe: un bloque entero cruza en un ciclo. El vector oculto pasa de un chip al siguiente en decenas de ciclos en lugar de miles. Se genera solo si el microprograma envía bloques.

## Reparto entre chips

Cuando el modelo no cabe en un chip, o se quiere más velocidad, hay tres formas de repartirlo. Se elige con «Reparto entre chips» y, en el mixto, «Chips por grupo».

### Por capas, en cadena

Cada chip guarda un tramo de capas seguidas. Para cada token, el primero hace sus capas y pasa el vector oculto (unos miles de valores) al segundo, y así hasta el último, que lo devuelve al primero para los logits. Ventajas: poca comunicación, que tolera enlaces lentos por la placa; sin sincronización dentro de una capa; cada chip guarda solo los pesos y la caché de sus capas; al leer un prompt todos trabajan a la vez, cada uno en un token. Inconveniente: generando para un solo usuario solo trabaja un chip cada vez, así que más chips dan más memoria, no más velocidad. El generador de puertas lo implementa y lo prueba en puertas con dos chips sobre un enlace ruidoso.

### Por tensor, todos a la vez

Cada chip guarda una parte de cada matriz y de cada grupo de cabezas, y todos calculan cada capa a la vez. Tras la atención y tras la MLP de cada capa suman sus resultados parciales en anillo: 2·(N−1) pasos, cada uno con una parte N-ésima del vector. Ventaja: con un usuario la velocidad crece casi con el número de chips. Inconvenientes: muchas sincronizaciones por token, que exigen enlaces rápidos y de poca latencia, en la práctica dentro del mismo encapsulado; un prompt no se lee en cadena. El estimador modela la variante más eficiente, que reparte también las cabezas de atención. El generador de puertas implementa hoy una variante más sencilla, «por filas»: cada chip guarda una franja igual de filas de cada matriz y, tras cada producto, los chips se pasan sus franjas en anillo hasta que todos tienen el vector entero. Antes de estrechar cada producto acuerdan el desplazamiento por el enlace (dos palabras por chip), así que el resultado es idéntico bit a bit al de un solo chip; la prueba en puertas lo comprueba con dos chips y enlaces ruidosos. La atención, DeltaNet y las normas las calcula entera cada chip: los pesos se reparten, pero la caché KV y el estado se repiten en todos, y se mueve más por el enlace que en la variante eficiente.

### Mixto

Grupos de chips que comparten cada capa en tensor, unidos en cadena. Es lo que usan los sistemas grandes: cada grupo dentro de un encapsulado, con UCIe, y los grupos por la placa. Con un grupo de 1 chip es el reparto por capas; con todos en un grupo, el reparto por tensor. En puertas, el último chip de cada grupo cierra el anillo del grupo con un enlace de vuelta al primero, y el vector oculto pasa al grupo siguiente por el enlace de siempre. Una imagen de arranque lleva un plan por cada número de chips: los que no forman grupos enteros se reparten por capas.

## Dimensionado

La búsqueda prueba formas de die (celdas, carriles, unidades, número de chips), genera para cada una su microprograma real y cuenta sus ciclos en la última posición del contexto, así que los tok/s salen del circuito y están garantizados con el contexto lleno.

### Tok/s objetivo

La velocidad mínima con el contexto lleno. La búsqueda se queda con el die de menos silicio que la alcanza.

### Tiempo al primer token

El máximo tras un prompt de cierta longitud. Con varios chips, el prompt avanza en cadena: cada chip trabaja en un token distinto a la vez.

### Contexto

Tokens que caben a la vez. Fija la caché KV, la tabla RoPE y la posición donde se miden los tok/s.

### Chips

En cuántos chips iguales se reparten las capas. Se cortan para que el chip que más memoria necesita tenga la menor posible: el primero también guarda lm_head, así que lleva menos capas.

### Reloj máximo

Cada die va a su frecuencia más alta según su camino crítico medido, o a esta si es menor.

### Proceso

Retardo de un sumador completo, área de una puerta NAND2 con su cableado, área de un bit de SRAM y su periferia, y tamaño máximo de die (retícula: 858 mm² en N6). Un die mayor no se puede fabricar y se descarta.

### Más eficiente y equilibrio

De la frontera velocidad-área se proponen el punto con menos área por tok/s y el último en que cada tok/s extra no cuesta más de cierto múltiplo del área media por tok/s.

## Generación

Se eligen en cada petición y no cambian el hardware.

### Tokens máximos

Cuántos tokens genera como mucho; también para antes en cualquier token de la lista de parada del modelo.

### Temperatura

0 elige siempre el más probable. Más alta reparte la probabilidad entre más candidatos.

### Top-k

Entre cuántos de los mejores candidatos se sortea; no puede pasar de los candidatos de la unidad de muestreo.

### Razonar antes de responder

Con la plantilla de chat del modelo, deja abierto su bloque de razonamiento; sin él, el bloque llega cerrado y contesta directamente. Es texto del prompt.

### Presupuesto de razonamiento

Cuántos tokens puede razonar antes de que el firmware cierre el razonamiento por él. Los tokens de apertura y cierre salen del tokenizador de cada modelo.

### Profundidad MTP

Cuántos tokens adelanta la cabeza de predicción múltiple; se mide cuántos acierta.

### Fidelidad

Cuánto del chip se simula: firmware real instrucción a instrucción, unidad de muestreo, tokenizador en el firmware y carga de la imagen por transacciones.
