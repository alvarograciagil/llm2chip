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
| Memoria | 2,7 Mbit: microcódigo 1,3; CPU 0,8; tablas de función 0,5; banco vectorial 0,07, con la tabla de registros que da a cada uno solo lo que usa (sin ella serían 2,1) |
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

| Placa | FPGA | Lógica y memoria | Precio aproximado | ¿Cabe? |
|---|---|---|---|---|
| Sipeed Tang Mega 138K Pro | Gowin GW5AST-138 | 138 K LUT4, 340 BSRAM (6,1 Mbit), 298 DSP, 1 GB DDR3 | ~200 $ (dock) | Sí: lógica al 35 %, memoria al 56 % |
| ALINX AX7202 / AX7A200B | Xilinx XC7A200T | 134 K LUT6, 365 BRAM36, 740 DSP | 400-700 $ | Sí, con mucho margen |
| Digilent Nexys Video | Xilinx XC7A200T | igual | ~500 € | Sí |
| Digilent Arty A7-100T | Xilinx XC7A100T | 63 K LUT6, 135 BRAM36, 240 DSP | ~250 $ | Sí: LUT al 72 %, BRAM al 67 % |
| Sipeed Tang Primer 25K | Gowin GW5A-25 | 23 K LUT4, 56 BSRAM | ~35 $ | No |

Recomendación: la **Tang Mega 138K Pro**. Cabe con margen en lógica y memoria, cuesta unos 200 $ y deja sitio para modelos algo mayores. La **Arty A7-100T** también cabe; una placa con **XC7A200T** da más margen con herramientas de Xilinx.

## Velocidad esperada

El die gasta 11 103 ciclos por token generado con el modelo de juguete (medido en su microprograma, en la última posición del contexto); con el firmware, unos 12 000-13 000. Sin colocar y rutar aún, el reloj en una GW5A rondaría 25-50 MHz: unos 2 000-4 000 tok/s, sin valor práctico más allá de validar que el chip funciona. Con stories260K y las mismas 8 celdas serían unos 50 000-80 000 ciclos por token, unos 300-1 000 tok/s.

## Pasos para la prueba

1. Generar el Verilog con el ejemplo `fpga` y la imagen con el mismo die.
2. Sustituir las memorias genéricas (`llm2chip_sram`, `llm2chip_ram2p`, `llm2chip_rom`) por bloques de la FPGA; Yosys ya las infiere.
3. Conectar el puerto de host a una UART (por USB) y la petición de imagen a la flash de la placa o a la propia UART.
4. Enviar un prompt y comparar la salida con la simulación: deben coincidir byte a byte.

Fuentes de precios y especificaciones: [Tang Mega 138K Pro](https://en.wiki.sipeed.com/hardware/en/tang/tang-mega-138k/mega-138k-pro.html), [CNX sobre la Tang Mega](https://cnx-software.com/2023/10/01/sipeed-tang-mega-138k-pro-dock-features-gowin-gw5ast-fpga-risc-v-soc), [Tang Primer 25K](https://cnx-software.com/?p=123307), [ALINX AX7202](https://boards.fpgadeveloper.com/boards/AX7202), [Nexys Video](https://www.digikey.co.uk/en/products/detail/digilent-inc/410-316/5456481), [Arty A7](https://pcbsync.com/xilinx-fpga-buying-guide/). Los precios cambian según tienda y región: son orientativos.
