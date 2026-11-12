# Probar el chip en una FPGA

El objetivo es validar el proyecto en hardware real: el mismo circuito que se simula puerta a puerta, con un modelo diminuto y a poca velocidad, generando texto en una placa asequible.

## El chip que se lleva a la FPGA

`cargo run --release -p sim --example fpga -- chip.v --dsp` escribe en Verilog el chip completo para el modelo de las pruebas (un Qwen3.5 de juguete: 2 capas, una DeltaNet y una de atención, ancho 8, vocabulario de 8 letras):

- CPU RISC-V RV32IM con su firmware, que arranca, carga la imagen comprobando su CRC, tokeniza, gobierna el die y muestrea.
- Die con 8 celdas, una unidad vectorial de un carril, y solo la lógica que su microprograma usa.
- Unidad de muestreo, enlaces con corrección de errores y JTAG.

Es exactamente el circuito de la prueba `the_chip_sized_for_an_fpga_says_the_same_in_gates`, que lo ejecuta en puertas: escribe lo mismo que el simulador de instrucciones, en los mismos ciclos (243 871 para «hoy» → «hf»).

| Recurso | Cantidad |
|---|---|
| Lógica | ≈ 468 000 equivalentes NAND2 |
| Memoria | 1,95 Mbit: CPU 0,8; tablas de función 0,5; microcódigo 0,5 (con diccionario; 1,3 sin él); banco vectorial 0,07, con la tabla de registros que da a cada uno solo lo que usa (sin ella serían 2,1) |
| Reloj | El camino crítico ronda 24 retardos de sumador completo: en una FPGA de gama baja, entre 20 y 50 MHz |

## Síntesis medida con Yosys

Con `--dsp`, cada multiplicador de un ciclo se escribe como un producto con signo (`$signed(weight) * $signed(x)`) y la herramienta lo pone en sus bloques multiplicadores. Es la misma aritmética: las pruebas comparan cada tipo de multiplicador generado con el producto exacto, y Yosys demuestra formalmente la equivalencia del de las celdas (`equiv_make` + `equiv_induct`). El resto del chip sigue puerta a puerta.

| Familia | LUT | Biestables | Memoria de bloque | Multiplicadores |
|---|---|---|---|---|
| Xilinx serie 7 | ≈ 45 900 LUT6 | ≈ 14 200 | 75 BRAM36 + 31 BRAM18 | 73 DSP48 |
| Gowin | ≈ 48 300 LUT4 + 2 272 ALU | ≈ 14 800 | 190 BSRAM de 18 Kb | 41 MULT18×18 + 32 MULT9×9 |

La tabla de registros cuesta lógica (unas 22 000 LUT, porque se describe con biestables y se lee tres veces por ciclo) y ahorra unas 120 BSRAM; en una FPGA podría ir en memoria distribuida y costar mucho menos.

Sin `--dsp`, en Xilinx hacen falta unas 110 600 LUT6 y ningún DSP: los multiplicadores se convierten en puertas.

`synth_gowin` de Yosys mapea a la familia GW1N; la GW5A de la Tang Mega tiene bloques del mismo tamaño, pero su herramienta oficial puede empaquetarlos algo distinto. Es una estimación sólida, no un informe de colocación y rutado.

## Placas

El asistente de proyecto («+» en la barra, «Una FPGA») tiene estas placas y dice, para cada modelo, si cabe en la memoria del chip, si sus pesos tienen que ir a la memoria de la placa o si no cabe. Precios orientativos de noviembre de 2026; cambian según tienda y región.

| Placa | FPGA | Lógica | Memoria en el chip | DSP | Memoria en la placa | Precio |
|---|---|---|---|---|---|---|
| Placa con Cyclone IV EP4CE10 | Altera Cyclone IV E | 10,3 K LE | 0,41 Mbit | 23 | 32 MB SDRAM (las habituales) | ~25-40 $ |
| Sipeed Tang Primer 25K | Gowin GW5A-25 | 23 K LUT4 | 1 Mbit | 28 | — | ~35 $ |
| Sipeed Tang Mega 60K Dock | Gowin GW5AT-60 | 60 K LUT4 | 2,1 Mbit | 118 | 256 MB DDR3 | ~90 $ |
| Sipeed Tang Mega 138K Dock | Gowin GW5AST-138 | 138 K LUT4 | 6,1 Mbit | 298 | DDR3 | ~130 $ |
| Radiona ULX3S 85F | Lattice ECP5-85F | 84 K LUT4 | 3,7 Mbit | 156 | 32 MB SDRAM | ~130-150 $ |
| Terasic Atum A3 Nano | Altera Agilex 3 | 135 K LE | 8,3 Mbit | 368 | 64 MB SDRAM | 180-245 $ |
| Sipeed Tang Mega 138K Pro | Gowin GW5AST-138 | 138 K LUT4 | 6,1 Mbit | 298 | 1 GB DDR3 | ~195 $ |
| TUL PYNQ-Z2 | Zynq XC7Z020 | 53 K LUT6 | 4,9 Mbit | 220 | 512 MB DDR3 | 200-260 $ |
| Terasic DE10-Nano | Cyclone V SE | 110 K LE | 5,6 Mbit | 112 | 1 GB DDR3 | ~225 $ |
| **AMD Kria KV260** | Zynq UltraScale+ K26 | 117 K LUT6 | **23,1 Mbit** (BRAM + UltraRAM) | 1 248 | **4 GB DDR4, 64 bits** | ~249 $ |
| Digilent Arty A7-100T | XC7A100T | 63 K LUT6 | 4,9 Mbit | 240 | 256 MB DDR3 | ~250 $ |
| Digilent Zybo Z7-20 | Zynq XC7Z020 | 53 K LUT6 | 4,9 Mbit | 220 | 1 GB DDR3 | ~300 $ |
| AMD Kria KR260 | Zynq UltraScale+ K26 | 117 K LUT6 | 23,1 Mbit | 1 248 | 4 GB DDR4 | ~349 $ |
| ALINX AX7202 | XC7A200T | 134 K LUT6 | 13,1 Mbit | 740 | 1 GB DDR3 | 400-500 $ |
| Digilent Nexys Video | XC7A200T | 134 K LUT6 | 13,1 Mbit | 740 | 512 MB DDR3 | ~500 € |
| Digilent Genesys 2 | Kintex-7 XC7K325T | 204 K LUT6 | 16 Mbit | 840 | 1 GB DDR3 | ~1 000 $ |
| AMD Alveo U250 | UltraScale+ XCU250 | 1 728 K LUT6 | 432 Mbit | 12 288 | 64 GB DDR4 | ~9 000 $ |

La Cyclone IV EP4CE10 es la más pequeña del catálogo: el chip más pequeño que genera el proyecto necesita unos 48 000 elementos lógicos y 1,95 Mbit de memoria interna, frente a sus 10 300 y 0,41, aunque los pesos vayan a la SDRAM de la placa. Sirve para probar piezas sueltas, como una celda de cálculo (`cargo run -p netlist --example export`) o un extremo de enlace, pero no un chip completo.

### Recomendación por unos 200 $: Kria KV260

Por unos 50 $ más que la Tang Mega 138K Pro, la **KV260** tiene casi cuatro veces más memoria dentro del chip (23,1 Mbit frente a 6,1), cuatro veces más multiplicadores y 4 GB de DDR4 de 64 bits, varias veces más rápida que la DDR3 de las placas baratas.

- **stories260K cabe entero dentro del chip** en todas las variantes medidas (5,4-9,0 Mbit), con sitio para más carriles y celdas. En la Tang Mega solo cabe la variante más pequeña, la de un carril.
- La lógica básica del chip ocupa alrededor del 39 % de sus LUT.
- Con los pesos en la DDR4 caben modelos de cientos de millones de parámetros, como stories110M o SmolLM2-135M, a la velocidad que dé la memoria.
- Se programa con Vivado. Conviene confirmar antes de comprar que la edición gratuita de la versión que uses incluye el K26.

Si el límite de 200 $ es estricto, la mejor opción es la **Tang Mega 138K Pro**: cabe la lógica con margen y, con las optimizaciones de memoria, stories260K de un carril entra en su memoria interna (5,37 de 6,1 Mbit, a falta de cómo empaquete los bloques la herramienta de Gowin). La **Atum A3 Nano** tiene más memoria dentro del chip (8,3 Mbit), pero su memoria externa es una SDRAM de 64 MB, lenta.

Fuentes: [Kria KV260](https://www.amd.com/en/products/system-on-modules/kria/k26/kv260-vision-starter-kit.html), [renovación de la KV260 en 2025](https://www.amd.com/en/blogs/2025/kria-kv260-vision-ai-starter-kit-refresh.html), [Tang Mega 138K Pro](https://en.wiki.sipeed.com/hardware/en/tang/tang-mega-138k/mega-138k-pro.html), [Tang Mega 138K Dock](https://www.cnx-software.com/2024/06/28/sipeed-tang-mega-138k-dock-is-a-lower-cost-gowin-gw5ast-fpga-risc-v-development-board/), [Atum A3 Nano](https://www.cnx-software.com/2025/05/30/atum-a3-nano-development-board-combines-agilex-3-fpga-and-usb-blaster-iii-programmer/), [PYNQ-Z2](https://www.dfrobot.com/product-1838.html), [Genesys 2](https://www.digikey.in/en/product-highlight/d/digilent/genesys-2-kintex-7-fpga-development-board), [ALINX AX7202](https://botland.store/fpga-boards-and-kits/29031-ax7202-fpga-development-board-amd-artix-7-xc7a200t-5904422389024.html).

## Velocidad esperada

El die gasta 11 103 ciclos por token generado con el modelo de juguete (medido en su microprograma, en la última posición del contexto); con el firmware, unos 12 000-13 000.

### Reloj medido colocando y rutando

Con OSS CAD Suite (Yosys y nextpnr) el chip de juguete se coloca y se ruta en una Lattice ECP5-85F (encapsulado CABGA756, que tiene pines para todos sus puertos):

| Versión del Verilog para FPGA | Frecuencia máxima | LUT4 | Bloques de memoria | Multiplicadores |
|---|---|---|---|---|
| Productos en DSP, el resto en puertas | 15,4 MHz (estimada tras colocar) | 45 391 | 190 DP16KD | 89 |
| Además, los sumadores como sumas (cadenas de acarreo) y el microcódigo con diccionario | 18,2 MHz | 44 748 | 142 DP16KD | 89 |
| Además, el multiplicador del muestreo en un bloque DSP | 20,4 MHz | 43 084 | 142 DP16KD | 91 |
| Además, escala y sesgo en dos ciclos (`--factor-steps 2`) | 22,1 MHz | 46 699 | — | 59 |
| Además, las órdenes de la CPU llegan al die desde registros | **26,5 MHz** | 48 501 | 142 DP16KD | 59 |

El camino crítico era la unidad de muestreo: un multiplicador de desplazamiento y suma de 16 filas, en un ciclo, que alimentaba la dirección de la tabla exponencial. Después pasó a las celdas del die: lectura del sesgo, producto por la escala (repartido en varios bloques DSP encadenados) y acumulador en un mismo ciclo, 49 ns. Con escala y sesgo en dos ciclos, el factor sale de un registro y cada ciclo multiplica la mitad de sus bits: el die gasta 11 172 ciclos por token en vez de 11 103 (un 0,6 % más) y el reloj sube a 22,1 MHz, así que los diseños para FPGA lo usan por defecto. Después fue el bus de la CPU al die: una escritura de la CPU se decodificaba, arrancaba el die y llevaba una dirección a su memoria vectorial en el mismo ciclo. Con las órdenes registradas, el reloj sube a 26,5 MHz. El camino crítico es ahora el del propio secuenciador: la palabra de microcódigo recién leída decide en el mismo ciclo la dirección de la memoria vectorial, con 27 de sus 38 ns en rutado. Es la familia más lenta de las del catálogo; en una Xilinx serie 7 o UltraScale+ el mismo circuito debería ir bastante más rápido.

Con 26,5 MHz, el modelo de juguete daría unos 2 200-2 400 tok/s, sin valor práctico más allá de validar que el chip funciona.

### Con un modelo que escribe cuentos: stories260K

[stories260K](https://huggingface.co/karpathy/tinyllamas) es un Llama de 260 000 parámetros (5 capas, ancho 64, vocabulario de 512) que escribe cuentos cortos en inglés. `tools/convert/llama2c.py` convierte su `.bin` y su tokenizador a una carpeta de Hugging Face, en float32 o con `--affine 8` en el formato afín de MLX (8 bits, escala y sesgo por grupo), que es el que lleva el chip.

`cargo run --release -p sim --example fpga -- chip.v --dsp --model ~/.llm2chip/models/stories260K-q8 --context 64 --cells C --lanes L --memories` dimensiona el chip para ese modelo, cuenta los ciclos del microprograma en la última posición del contexto y desglosa la memoria:

| Celdas × carriles | Ciclos por token | tok/s a 25-50 MHz | Memoria |
|---|---|---|---|
| 8 × 1 | 211 500 | 120-240 | 5,37 Mbit |
| 8 × 4 | 104 800 | 240-480 | 7,33 Mbit |
| 16 × 4 | 87 500 | 290-570 | 7,33 Mbit |
| 32 × 8 | 52 700 | 470-950 | 9,04 Mbit |

La variante de 8 × 1 cabe en los 6,1 Mbit de la Tang Mega 138K; todas caben en una XC7A200T (13,1 Mbit) y en la Kria KV260 (23,1 Mbit). Con más carriles crece sobre todo la tabla de funciones, una por carril.

### Optimizaciones de memoria

Antes de ellas, la variante de 8 × 1 necesitaba unos 10,2 Mbit. Ninguna cambia un solo ciclo, y las pruebas en puertas comprueban que el chip escribe lo mismo, en los mismos ciclos, que el simulador de instrucciones.

| Cambio | Ahorro en 8 × 1 |
|---|---|
| **El embedding se queda en la imagen.** La tabla no ocupa la memoria vectorial: antes de cada token, el firmware trae su fila con el mismo cargador del arranque y la deja en un registro de una fila. La memoria vectorial pasa de 2^16 a 2^15 palabras. | ~1 Mbit |
| **Memoria ancha a medida.** Las sumas de las celdas solo necesitan tantas entradas como filas tiene una pasada o una suma, no tantas como un registro: 2^9 entradas en vez de 2^15. | 1,7 Mbit |
| **Microcódigo con diccionario.** De 9 813 palabras solo 2 018 son distintas, porque cada capa repite casi el mismo código. Cada palabra distinta se guarda una vez y, por cada PC, un índice que dice dónde están la siguiente palabra y el destino de su salto, así que el diccionario se lee en el mismo ciclo que una memoria normal y no se pierde ninguno. | 2,1 Mbit |

`--table-inside` y `--no-dictionary` desactivan la primera y la tercera para comparar.

## Pasos para la prueba

1. Generar el Verilog con el ejemplo `fpga` y la imagen con el mismo die.
2. Sustituir las memorias genéricas (`llm2chip_sram`, `llm2chip_ram2p`, `llm2chip_rom`) por bloques de la FPGA; Yosys ya las infiere.
3. Conectar el puerto de host a una UART (por USB) y la petición de imagen a la flash de la placa o a la propia UART.
4. Enviar un prompt y comparar la salida con la simulación: deben coincidir byte a byte.

Fuentes de precios y especificaciones: [Tang Mega 138K Pro](https://en.wiki.sipeed.com/hardware/en/tang/tang-mega-138k/mega-138k-pro.html), [CNX sobre la Tang Mega](https://cnx-software.com/2023/10/01/sipeed-tang-mega-138k-pro-dock-features-gowin-gw5ast-fpga-risc-v-soc), [Tang Primer 25K](https://cnx-software.com/?p=123307), [ALINX AX7202](https://boards.fpgadeveloper.com/boards/AX7202), [Nexys Video](https://www.digikey.co.uk/en/products/detail/digilent-inc/410-316/5456481), [Arty A7](https://pcbsync.com/xilinx-fpga-buying-guide/). Los precios cambian según tienda y región: son orientativos.
