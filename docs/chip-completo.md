# Chip completo: diseño y simulación

Objetivo: que el diseño y la simulación incluyan todo lo que llevaría el chip real. La única diferencia entre simular y ejecutar en silicio será cómo entran y salen los datos en el puerto de host.

## Frontera entre el chip y el mundo

El chip real tendría un PHY PCIe o USB. En simulación, ese PHY se sustituye por un puerto digital sencillo. Todo lo que queda a este lado del puerto es igual en ambos casos.

| Señal | Sentido | Uso |
|---|---|---|
| `host_rx_*` | Entrada, flujo de bytes | Comandos y texto UTF-8 |
| `host_tx_*` | Salida, flujo de bytes | Texto UTF-8 y estado |
| `image_req_*`, `image_data_*` | Peticiones de bloque de la imagen | Lectura del modelo desde el host |

El chip recibe bytes de texto y devuelve bytes de texto. El tokenizado, la inferencia, el muestreo y el destokenizado ocurren dentro.

## Bloques

| Bloque | Función | Hoy |
|---|---|---|
| Celdas MAC + SRAM de pesos | Producto matriz-vector | En puertas |
| Red de activaciones | Distribución y suma | En puertas |
| Carriles vectoriales | Atención, delta, puertas de la MLP | En puertas, un carril |
| Tablas de función | SiLU, sigmoide, exp, raíz | En puertas |
| Registros vectoriales y microsecuenciador | Control de cada etapa, bucles y parámetros | En puertas, con microprograma fijo |
| Enlace entre dies | Paso de activaciones | En puertas |
| SRAM KV y estado DeltaNet | Memoria del modelo | Modelo funcional y área |
| Bus interno del sistema | Une CPU, memorias y periféricos | En puertas |
| Puerto de host y FIFO | Entrada y salida de bytes | En puertas |
| CPU de control RV32IM | Firmware | En puertas e ISS, ciclo a ciclo |
| SRAM de instrucciones y datos | Firmware y vocabulario | En puertas |
| Controlador de carga y DMA | Imagen hacia celdas, tablas y microcódigo, con arranque propio y CRC | En puertas e ISS, conectado al die |
| Red de escritura de pesos | Lleva cada palabra a su celda | Solo puertos de carga directos |
| ECC, CRC, BIST y reparación | Fiabilidad y celdas defectuosas | ECC en puertas en la celda, CRC en la carga; autotest y filas de reserva verificados como bloque, sin conectar a todas las memorias |
| Unidad de muestreo y RNG | Top-k y temperatura sobre los logits, con filtro de varios carriles | En puertas y modelo, conectada al die |
| Tokenizador en firmware | Sigue las reglas del `tokenizer.json` de cada modelo mediante tablas: tokens añadidos, normalización, expresiones regulares, BPE y plantilla de tokens especiales | Comparado con la biblioteca `tokenizers` en estilos sintéticos y modelos reales; en ISS con tablas reales y en puertas con tablas diminutas. Su memoria de trabajo se dimensiona con el contexto del diseño y el estimador la cuenta |
| Planificador de tokens y MTP | Decide qué ejecuta cada die, y deshace | En software |
| Reloj, reset y gestión de energía | Secuencia de reset, puertas de reloj | Reset en puertas; el reloj y la energía solo se estiman, porque el simulador tiene un único reloj |
| Depuración y contadores | Contadores y puerto de lectura; falta JTAG estándar | En puertas |

Muestrear en la CPU no sirve en los diseños rápidos. Leer 250.000 logits cuesta unos 250 µs a 1 GHz, y a 15.000 tok/s un token dura 66 µs. Por eso el muestreo tiene unidad propia y la CPU solo recibe el resultado.

Los bloques analógicos (PLL, PHY, sensores) solo tendrán modelo de comportamiento, porque no son lógica digital. El PHY queda fuera de la frontera; la PLL y los sensores se modelan como fuentes de reloj y de valores.

## Niveles de simulación por bloque

Cada bloque se configura por separado desde la simulación. Así se elige entre pruebas lentas y realistas o rápidas y concretas.

| Bloque | Omitir | Funcional | Instrucción o transacción | Puertas |
|---|---|---|---|---|
| Arranque y BIST | Sí | | ISS | CPU en puertas |
| Carga de pesos | Pesos directos del checkpoint | | Controlador real, imagen, ECC y ciclos | Muestra o modelo diminuto |
| Tokenizador | | En Rust, fuera del chip | Firmware en ISS | Firmware en puertas, con tablas diminutas |
| Núcleo de inferencia | | Coma flotante (CPU o GPU) | Enteros exactos (CPU o GPU) | Modelo diminuto |
| Muestreo | | En Rust, fuera del chip | Unidad funcional | Puertas |
| Destokenizador | | En Rust, fuera del chip | Firmware en ISS | Firmware en puertas |

Perfiles de partida:

- **Rápida:** todo funcional, coma flotante y GPU. Para probar modelos y formatos.
- **Aritmética real:** enteros exactos y GPU, con el resto funcional.
- **Flujo real:** arranque, carga, firmware y muestreo con sus niveles realistas, y enteros exactos.
- **Puertas completas:** todo en puertas con un modelo diminuto, de los bytes de entrada a los de salida.

Cada ciclo que muestra la app indica si se ha simulado o estimado, según lo que se haya omitido.

## Chip completo en puertas

`chip_design` une la CPU, las memorias, el controlador de carga, el die y la unidad de muestreo. Con un modelo diminuto, el firmware real recibe bytes, carga la imagen, ejecuta los tokens y devuelve bytes. El resultado y los ciclos coinciden con el nivel de instrucción (CPU en ISS y die en la máquina exacta): 525.668 ciclos en los dos.

## Coste del muestreo

Leer los 248.000 logits de Qwen3.8 de uno en uno costaba casi tanto como inferir el token. La unidad compara ahora varios logits por ciclo con el peor candidato de su lista, y solo los que lo superan entran en ella. Son unas 500 inserciones por token, así que casi ningún grupo se detiene.

Con Qwen3.8 en el flujo real y 64 carriles, los ciclos fuera del die bajan de 1.502.415 a 37.162 para 17 tokens de prompt y 6 generados, con el mismo texto. El estimador supone además que los logits se filtran mientras salen de `lm_head`, y cuenta 787 ciclos por token. El die reducido solo entrega un logit por ciclo, así que en el chip completo en puertas se usa un carril; varios carriles se verifican en la unidad por separado.

## Circuito a medida

Con «Circuito a medida», cada combinación de parámetros genera su propia celda. El generador construye en puertas cada variante (array, Wallace, Booth y múltiplos precalculados, con sus etapas y con o sin aislamiento de ceros), mide su área, su conmutación y su camino crítico, y se queda con la mejor que alcanza la frecuencia pedida. Los anchos distintos de un modelo mixto reciben celdas distintas. Con «Velocidad objetivo», los multiplicadores por die salen de esa velocidad y del tamaño del modelo.

Los múltiplos precalculados salen perdiendo en los casos medidos: quitan el multiplicador, pero multiplican por 5 los bits que hay que repartir a cada celda. Saltar ceros tampoco compensa con los ceros reales de Qwen, un 10,6 % a 4 bits y un 1,6 % a 8.

## Varios chips

Todos los chips salen de la misma netlist y leen la misma imagen. Cada uno lleva CPU, memorias, controlador de carga, die y muestreo. Un solo pin, `follower`, decide en la placa cuál manda: habla con el host, tokeniza y muestrea.

Al arrancar, el maestro envía un contador por la cadena de control. Cada chip toma su número, lo incrementa y lo pasa; al volver al maestro, ese valor es el total, que vuelve a dar la vuelta. Con el total y su número, cada chip busca en la imagen su microprograma y sus pesos. Si la imagen no contempla ese número de chips, el maestro lo avisa por el host. Un chip cuya cadena no tiene vecino funciona solo.

| Pines | Función |
|---|---|
| `link_tx_*`, `link_rx_*` | Datos entre dies, en anillo: activaciones y, del último al primero, los logits |
| `chain_tx_*`, `chain_rx_*` | Recuento al arrancar y órdenes del maestro, en anillo |
| `image_req_*`, `image_resp_*` | Cada chip pide su parte de la misma imagen |
| `follower` | A cero en el chip que manda |
| `reset`, `host_*`, `debug_*` | Reset, host (solo se usa en el maestro) y depuración |

Cada palabra lleva un código que corrige un bit y detecta dos, y una señal de válido y otra de listo. No hay reintento: un error doble se cuenta, pero el dato llega mal. El reloj sigue siendo común en la simulación, así que falta el cruce entre dominios de reloj. La imagen debe traer el reparto de capas de cada número de chips admitido.

## Pesos grabados

El parámetro «Pesos» decide si el modelo se puede cambiar. Con ROM, los pesos, las escalas y los sesgos de cada celda se fijan al fabricar: la celda pierde los puertos de carga y el selector de direcciones, el die pierde la red que escribe en las celdas, y el arranque ya no los transfiere. El microprograma, las tablas y el firmware siguen siendo cargables, así que el chip se puede corregir sin máscaras nuevas, pero no cambiar de modelo.

El mayor ahorro no viene de la lógica sino de la densidad de la ROM frente a la SRAM. Para sintetizarlo en una tecnología real hay que sustituir `llm2chip_rom` por la macro de ROM de la fundición, con el contenido de cada celda.

## Por qué no todo en puertas con Qwen

Qwen3.8 tiene 27.000 millones de pesos, y simularlos en puertas no es viable. La estrategia es la misma que ya funciona para el núcleo:

1. Con un modelo diminuto, todo en puertas: bytes de entrada, carga, firmware, inferencia, muestreo y bytes de salida. Se exporta a Verilog y se comprueba con Icarus.
2. Cada nivel rápido se contrasta bit a bit con las puertas en ese modelo.
3. Con Qwen real, la lógica de control puede ir en puertas o en ISS, y el núcleo de inferencia en la máquina exacta ya verificada. Se comunican por las mismas interfaces de registros que en el chip.

## Carga del modelo en simulación profunda

Hoy los pesos se escriben directamente en la memoria simulada. El cambio central es que la simulación lea de lo que el chip ha cargado.

**Imagen del chip.** Firmware, microcódigo, tablas y tokenizador forman la imagen que recorre el controlador. Para el modelo real, un manifiesto en `~/.llm2chip/images` guarda el destino y el CRC de cada tensor, y los pesos se leen del checkpoint para no duplicar sus 19 GB. Falta escribir una imagen de pesos por celda en el formato del chip.

**Tres niveles para la misma carga:**

| Nivel | Qué ocurre | Alcance |
|---|---|---|
| Directo | Se abre el checkpoint, como ahora | Pruebas rápidas |
| Transacción | El controlador funcional recorre la imagen entera, comprueba CRC y ECC, aplica la reparación, escribe la memoria de cada celda y cuenta los ciclos | Modelo real completo |
| Puertas | El controlador y la red de escritura reales mueven los datos por el puerto de carga | Modelo diminuto completo y muestras del real |

**Muestras con el modelo real.** En puertas se carga un bloque real de pesos, elegido por el usuario o al azar, por el controlador y la red de escritura, y las palabras resultantes deben coincidir bit a bit con las del nivel de transacción. El resto de celdas se precarga directamente en la memoria simulada, como hace un banco de pruebas de RTL.

**Inyección de fallos.** La simulación podrá invertir bits y marcar celdas defectuosas, para comprobar que ECC, CRC y reparación actúan como deben.

## Microprograma fijo

El microprograma se carga una sola vez desde la imagen y no depende del token ni de la posición. El secuenciador tiene registros de índice que suman un desplazamiento a cada operando, y un salto condicional para los bucles. La CPU escribe los parámetros (fila del embedding, posición en la caché, tokens de contexto) y arranca una de las rutinas: token, logits, y las de la cabeza MTP.

Los senos y cosenos de RoPE pasan a una tabla en memoria con una fila por posición. Es la parte que más crece con el contexto (cada posición ocupa 3·h+1 valores), así que en contextos largos convendrá sustituirla por dos tablas y suma de ángulos.

La lógica por cabeza y por capa sigue desenrollada: el programa crece con el número de cabezas, aunque ya no con el de elementos. Plegar las cabezas en bucles es la mejora pendiente para que quepa Qwen3.8 completo.

## Fases

1. Bus del sistema, mapa de memoria y puerto de host.
2. Núcleo RV32IM en puertas e ISS, verificados entre sí.
3. Firmware en Rust `no_std`: arranque, tokenizador, bucle principal.
4. Imagen del chip y controlador de carga, con los tres niveles.
5. Unidad de muestreo y RNG.
6. ECC, CRC, BIST y reparación, con inyección de fallos.
7. Reloj, reset, energía, depuración y contadores.
8. Integración: chip completo en esquemático, estimador desde la netlist, configuración de simulación por bloque, consola y línea de tiempo de arranque.
