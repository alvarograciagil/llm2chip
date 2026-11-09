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
| ECC, CRC, BIST y reparación | Fiabilidad y celdas defectuosas | Solo área |
| Unidad de muestreo y RNG | Top-k y temperatura sobre los logits | En puertas y modelo, conectada al die |
| Planificador de tokens y MTP | Decide qué ejecuta cada die, y deshace | En software |
| Reloj, reset y gestión de energía | Secuencia de reset, puertas de reloj | Falta |
| Depuración y contadores | JTAG, trazas, rendimiento | Falta |

Muestrear en la CPU no sirve en los diseños rápidos. Leer 250.000 logits cuesta unos 250 µs a 1 GHz, y a 15.000 tok/s un token dura 66 µs. Por eso el muestreo tiene unidad propia y la CPU solo recibe el resultado.

Los bloques analógicos (PLL, PHY, sensores) solo tendrán modelo de comportamiento, porque no son lógica digital. El PHY queda fuera de la frontera; la PLL y los sensores se modelan como fuentes de reloj y de valores.

## Niveles de simulación por bloque

Cada bloque se configura por separado desde la simulación. Así se elige entre pruebas lentas y realistas o rápidas y concretas.

| Bloque | Omitir | Funcional | Instrucción o transacción | Puertas |
|---|---|---|---|---|
| Arranque y BIST | Sí | | ISS | CPU en puertas |
| Carga de pesos | Pesos directos del checkpoint | | Controlador real, imagen, ECC y ciclos | Muestra o modelo diminuto |
| Tokenizador | | En Rust, fuera del chip (hoy) | Firmware en ISS (pendiente) | Firmware en puertas (pendiente) |
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

## Coste del muestreo medido

Con Qwen3.8 en el flujo real, 17 tokens de prompt y 6 generados suman 7,5 millones de ciclos, y 1,5 millones quedan fuera del die. Casi todo ese sobrecoste es leer los 248.000 logits de uno en uno: unos 250.000 ciclos por token generado, frente a los 262.000 que tarda el die en inferirlo. La unidad actual casi duplica el tiempo por token.

La propuesta es filtrar en paralelo: cada carril compara sus logits con el peor candidato de la lista y solo los que lo superan pasan a la lista ordenada. En 248.000 logits eso son unas 500 inserciones, muy pocas para frenar el flujo. Además, los logits pueden leerse mientras salen de `lm_head`, sin una pasada aparte.

## Por qué no todo en puertas con Qwen

Qwen3.8 tiene 27.000 millones de pesos, y simularlos en puertas no es viable. La estrategia es la misma que ya funciona para el núcleo:

1. Con un modelo diminuto, todo en puertas: bytes de entrada, carga, firmware, inferencia, muestreo y bytes de salida. Se exporta a Verilog y se comprueba con Icarus.
2. Cada nivel rápido se contrasta bit a bit con las puertas en ese modelo.
3. Con Qwen real, la lógica de control puede ir en puertas o en ISS, y el núcleo de inferencia en la máquina exacta ya verificada. Se comunican por las mismas interfaces de registros que en el chip.

## Carga del modelo en simulación profunda

Hoy los pesos se escriben directamente en la memoria simulada. El cambio central es que la simulación lea de lo que el chip ha cargado.

**Imagen del chip (`.chipimg`).** El compilador genera un archivo con cabecera y versión, mapa de memoria, firmware, microcódigo, tablas de función, tokenizador, y pesos por celda en el formato del chip, con ECC y CRC por segmento. Los pesos quedan alineados, lo que permite usarlos desde la GPU sin copias, y un formato recodificado se calcula una sola vez y se guarda.

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
