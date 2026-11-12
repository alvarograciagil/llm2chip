# Layout de chip con OpenROAD

El Verilog que genera llm2chip se puede llevar hasta un layout fabricable (GDSII) con herramientas abiertas: síntesis con Yosys y floorplan, colocación, árbol de reloj, rutado y comprobaciones con OpenROAD, sobre PDK abiertos que se fabrican de verdad en lanzaderas multiproyecto.

| PDK | Qué es | Se fabrica |
|---|---|---|
| SkyWater 130 (`sky130hd`) | 130 nm, 5 metales | Sí: lanzaderas abiertas como las de Efabless y Tiny Tapeout |
| GlobalFoundries 180 (`gf180`) | 180 nm, 5 V | Sí, en lanzaderas abiertas |
| IHP SG13G2 (`ihp-sg13g2`) | 130 nm BiCMOS | Sí, en las lanzaderas de IHP |
| ASAP7 (`asap7`) | 7 nm predictivo | No: sirve para ver cómo sería en un nodo avanzado |

Los nodos N6, N5 y N3 no tienen PDK abierto: el estimador los modela, pero no se puede sacar su layout.

## Cómo se lanza

Hace falta Docker con la imagen `openroad/orfs` (OpenROAD-flow-scripts, unos 6,5 GB; en un Mac con Apple Silicon corre emulada, más despacio).

```bash
tools/asic/flow.sh diseño.v MODULO sky130hd 10
```

Deja en `~/.llm2chip/asic/MODULO-plataforma/` el GDS, las imágenes del layout y `resumen.json` con el área, la frecuencia máxima, la potencia y las violaciones de reglas.

## Primera medida: la celda de cálculo

La celda que repite el die miles de veces (producto de un peso afín de 8 bits por una activación de 16, multiplicador Wallace, escala, sesgo y acumulador), exportada con `cargo run -p netlist --example export -- affine 8 0 wallace`:

| PDK | Área de celdas | Frecuencia máxima | Potencia | Violaciones |
|---|---|---|---|---|
| SkyWater 130 | 8 911 µm² (1 110 celdas) | 119 MHz | 1,19 mW a 100 MHz | 0 |
| GF180 | 41 614 µm² (1 101 celdas) | 125 MHz | 26,6 mW a 67 MHz | 0 |

## Calibración del estimador

`tools/asic/calibrate.sh` hace el layout de cinco celdas distintas (las de `tools/asic/cells.json`: afín de 8 bits con Wallace sin segmentar y segmentado, con Booth segmentado, entera de 4 bits con multiplicador en array, y coma flotante de 8 bits) en los tres procesos abiertos, dos a la vez, y guarda lo medido en `tests/golden/asic.json`.

Cómo se mide:

- Cada celda lleva 4 palabras de pesos en biestables, sin ECC.
- El reloj objetivo es fijo por proceso (4 ns en SkyWater 130, 6 ns en GF180, 500 ps en ASAP7, cuyas bibliotecas cuentan picosegundos), cerca de lo que el proceso da de sí, para que OpenROAD optimice la temporización sin inflar el área.
- La frecuencia máxima es la que OpenROAD calcula tras rutar, con el margen que sobra o falta.

`cargo run -p arch --example calibrate` compara cada medida con lo que espera el estimador de esa misma celda, contando sus 4 palabras como biestables con su multiplexor, y ajusta por mínimos cuadrados el retardo de un sumador completo y el del biestable de cada proceso:

| Celda | Proceso | Área medida µm² | Estimada | Relación | Reloj medido MHz | Estimado | Relación |
|---|---|---|---|---|---|---|---|
| affine8-wallace-1+1 | sky130hd | 11830 | 11048 | 0.93 | 149 | 141 | 0.95 |
| affine8-wallace-1+1 | gf180 | 49127 | 40586 | 0.83 | 145 | 130 | 0.90 |
| affine8-wallace-1+1 | asap7 | 152 | 155 | 1.02 | 1458 | 1354 | 0.93 |
| affine8-booth-3+2 | sky130hd | 10514 | 11802 | 1.12 | 283 | 279 | 0.99 |
| affine8-booth-3+2 | gf180 | 51658 | 43774 | 0.85 | 224 | 226 | 1.01 |
| affine8-booth-3+2 | asap7 | 165 | 167 | 1.01 | 2654 | 2410 | 0.91 |
| affine8-wallace-3+2 | sky130hd | 11788 | 13047 | 1.11 | 367 | 337 | 0.92 |
| affine8-wallace-3+2 | gf180 | 54464 | 48477 | 0.89 | 288 | 259 | 0.90 |
| affine8-wallace-3+2 | asap7 | 180 | 184 | 1.02 | 2843 | 2793 | 0.98 |
| fixed4-array-2+1 | sky130hd | 8877 | 7317 | 0.82 | 161 | 161 | 1.00 |
| fixed4-array-2+1 | gf180 | 37112 | 27055 | 0.73 | 134 | 146 | 1.09 |
| fixed4-array-2+1 | asap7 | 119 | 103 | 0.87 | 1492 | 1519 | 1.02 |
| float8-wallace-2+1 | sky130hd | 14874 | 12224 | 0.82 | 171 | 189 | 1.11 |
| float8-wallace-2+1 | gf180 | 71537 | 44567 | 0.62 | 152 | 166 | 1.10 |
| float8-wallace-2+1 | asap7 | 212 | 169 | 0.79 | 1560 | 1746 | 1.12 |

| Proceso | Sumador completo | Biestable | Tope de reloj | Áreas de puerta |
|---|---|---|---|---|
| SkyWater 130 | 0,41 ns | 0,51 ns | 400 MHz | `sky130_fd_sc_hd` |
| GF180 | 0,38 ns | 1,58 ns | 300 MHz | `gf180mcu_fd_sc_mcu9t5v0`, la de 9 pistas que usa OpenROAD (antes se tomaba otra, con la mitad de área) |
| ASAP7 | 0,038 ns | 0,13 ns | 3 GHz | `asap7sc7p5t` |

El reloj queda a menos del 12 % en las quince celdas. El área queda entre el 62 % y el 112 % de la medida; la coma flotante es la que más se aleja, porque el factor de síntesis de Yosys se midió sobre multiplicadores enteros. El test `crates/arch/tests/asic.rs` comprueba esos márgenes con las medidas guardadas, sin Docker.

Las celdas cuyo reloj objetivo no se alcanza terminan con unas decenas de violaciones de reglas en el rutado detallado; las que lo alcanzan, con ninguna. El área y la frecuencia no cambian por ello.

## La corrección de errores en el camino de lectura

Con SRAM, cada celda guarda sus pesos en palabras de unos 64 bits con bits de comprobación, y corrige el peso que lee. El estimador medía el reloj de la celda con una memoria de una sola palabra, sin corrector, y el layout también. Con el corrector, el camino crítico pasaba de unos 8 retardos de sumador completo a 35: una celda de 16 bits en N6 no superaba 0,62 GHz, frente a los más de 2 GHz estimados.

Tres cambios lo arreglan:

- Las paridades, el recuento de bits del síndrome y la comprobación de carril eran cadenas lineales de puertas; ahora son árboles equilibrados, con la misma lógica.
- El peso corregido se registra antes del multiplicador, y la activación con él: la corrección tiene su propio ciclo, y la latencia de la celda sube en uno. Sin ese registro, la celda sería entre un 35 y un 55 % más lenta.
- El estimador mide el reloj, y el layout se hace, con la memoria más pequeña que conserva la corrección de la celda real (`timed_cell`). Por encima de ese tamaño el camino ya no crece.

Con todo ello, el camino de la celda con corrección vuelve a 7,4-8,3 retardos, lo mismo que sin ella.

## Memorias

Las memorias son macros de SRAM, y en los procesos abiertos solo existen pequeñas. Las de SkyWater 130 que trae OpenROAD (`sky130_sram_1rw1r_*`, de OpenRAM) llegan a 32 Kbit y ocupan 35,5 µm² por bit en 5 Kbit, 25,6 en 16 Kbit y 22,0 en 32 Kbit; el modelo las reproduce con 13,17 µm² por bit y periferia 3,79, y una memoria mayor se construye con varias de 32 Kbit. GF180 usa las de GlobalFoundries, de 4 Kbit y unos 51 µm² por bit. Antes el modelo suponía macros de cualquier tamaño, con densidades 5 veces mejores que las reales.

## Una tesela con una macro SRAM real

`tools/asic/tile.sh` hace el layout de una celda con su memoria de pesos en una macro SRAM real de SkyWater 130, no en biestables: la celda afín de 8 bits con Wallace en 3 + 2 etapas guarda 4096 pesos (32 Kbit) en una macro `sky130_sram_1rw1r_128x256_8` de OpenRAM. Cada palabra de 128 bits de la macro lleva 16 pesos: los bits altos de la dirección eligen la palabra, los bajos, registrados un ciclo, el peso, y una escritura solo toca su byte con la máscara de la macro.

| Medida | Valor |
|---|---|
| Macro | 0,722 mm² (el estimador da lo mismo, dentro del 5 %) |
| Lógica | 0,031 mm² |
| Die | 1,84 mm² |
| Reloj máximo | 211 MHz |
| Potencia | 38,7 mW |
| Violaciones tras rutar | 0 |

El reloj lo pone la macro, no la lógica: entrega el dato en el flanco de bajada, así que la lectura tiene medio ciclo, y la celda sola llegaba a 367 MHz. La tecnología `sky130` lleva ahora ese periodo mínimo (`sram_min_period_ns`, 4,74 ns), y el estimador limita a él el reloj de los diseños con pesos en SRAM.

Para integrarla hicieron falta tres cosas:

- Conectar sus pines de alimentación, que OpenRAM llama `vdd` y `gnd`.
- Cortar las filas de celdas a 10 µm de la macro, para que las vías de sus raíles no caigan sobre la macro.
- Darle a la macro su propia rejilla de alimentación: bandas de met4 sobre ella, con vías a sus miles de islas de alimentación en met3 y a met5, ceñidas a su contorno.

El análisis de la red de alimentación de OpenROAD se deja fuera: el modelo abstracto de esta macro tiene, en dos columnas, rectángulos de `vdd` y de `gnd` solapados en met3 (por ejemplo, `vdd` en y = 109,815-110,305 y `gnd` en 109,875-110,365, en x = 214,22), y el análisis los da por cortocircuitos.

El DRC de KLayout sobre el GDS final (`make drc`) da 522 617 violaciones:

- 521 993 dentro de las celdas de bit de la macro, que siguen reglas especiales de SRAM que el juego estándar no exceptúa. Una celda sin macro de la calibración da 0.
- 582 en la celda superior, sobre la macro: espaciados en met1-met4 entre lo que dibuja el flujo (las bandas de met4 de su rejilla, las vías a sus islas de met3 y las conexiones a sus pines) y metal del dibujo de la macro que su modelo abstracto no declara; este dice que met4 está libre.
- 42 fuera de la macro, en vías de la red de alimentación.

El LVS no se pasa: necesita la netlist CDL de la macro, que el flujo no encuentra.

Así que esta macro, la que trae la imagen de OpenROAD y ningún diseño de ejemplo usa, no sirve para fabricar: su modelo abstracto está incompleto. Las cifras de área y reloj valen como medida, pero un chip real necesitaría las macros de OpenRAM para SkyWater 130 publicadas con anillos de alimentación en met4 y modelos abstractos completos, que habría que descargar aparte, y pasar el DRC y el LVS sobre ellas.

## Viabilidad antes del layout

Antes de hacer ningún layout, el inspector dice si el diseño se puede fabricar en el proceso elegido (`arch::feasibility`):

- **Qué lo impide**, con cifras: cuánto necesita cada die frente a lo que admite el proceso, y los tres bloques que más ocupan; los errores del estimador; y, para el layout, si el proceso tiene PDK abierto y si las memorias que usa el diseño existen en él (solo SRAM).
- **Cómo hacerlo viable**: prueba cambios de parámetros (repartir en más chips, dimensionar para el modelo, contexto nativo, menos bits de peso o de caché KV, menos multiplicadores o carriles, tablas de función más pequeñas, todo el die del proceso, otro proceso abierto que se fabrique) y solo ofrece los que, estimados de nuevo, pasan la comprobación. Si ninguno basta solo, busca una combinación cambio a cambio, y después otra con el menor número de chips sin aceptar pérdidas grandes de calidad. Cada propuesta indica sus chips, su área, sus tok/s y lo que arriesga.
- **Qué significa el número de chips**: cuántos encapsulados y placas hacen falta (unos 16 encapsulados por placa para chips grandes, 64 para chips pequeños de lanzadera), el tiempo que cada token pasa en los enlaces, y que los chips son una misma netlist repetida que carga su parte del modelo desde la imagen, así que con pesos en SRAM basta un diseño; solo con pesos grabados en ROM cada die es distinto. Cada chip lleva su CPU, la memoria de la CPU, las tablas del tokenizador y el muestreo, aunque solo el primero los use.
- **Enlaces de una lanzadera abierta**: cada die llega en su propio encapsulado y solo tiene sus pines de E/S para hablar con los demás. En automático, SkyWater 130 y GF180 usan enlaces por pines (16 pines por sentido a 50 MHz, unos 60 ns por salto), y los tok/s cuentan lo que tarda el vector en cruzarlos.

La búsqueda hace unas cien estimaciones, y tarda menos de dos segundos en nativo: las celdas generadas, sus caminos críticos y su conmutación se guardan, porque se repiten de una estimación a otra.

Al cambiar de modelo, el diseño se dimensiona para él: sus límites (parámetros, capas, ancho, vocabulario) pasan a ser los del modelo, el contexto no supera el nativo, y los multiplicadores, carriles vectoriales y unidades de función siguen la proporción del diseño por defecto, o la velocidad pedida si la hay.

El botón de layout queda desactivado mientras el diseño no sea viable.

Ejemplo: stories260K en SkyWater 130 no cabe en un die de 10 mm² (la memoria de pesos sola ocupa más de 45 mm²). La propuesta con menos chips lo reparte en 42 dies de 10 mm² con contexto de 512 tokens y caché KV de 8 bits.

## Desde la aplicación

Con el destino «Chip propio» en SkyWater 130, GF180 o ASAP7, el inspector tiene el apartado **Layout real de la celda**. «Hacer el layout» toma la celda que el generador eligió para el diseño (codificación, bits, multiplicador y etapas), la pasa por OpenROAD en segundo plano y muestra la imagen del layout con su área, frecuencia máxima, potencia, número de celdas y violaciones de reglas, al lado de lo que estimaba el modelo. El trabajo queda en `~/.llm2chip/asic/`.

«Usar las cifras medidas» guarda en el diseño el área y la frecuencia medidas (`measured_cell`, `measured_cell_um2`, `measured_cell_fmax_ghz`). El estimador las usa en lugar de las suyas solo para esa misma celda en ese mismo proceso: si cambian el circuito o el proceso, vuelve a la estimación. Las memorias, la red de activaciones y el resto del die siguen estimándose.

La celda que se mide es la que eligió el estimador, con todo lo que cambia sus puertas (codificación, bits, acumulador, multiplicador, etapas, ECC, salto de ceros); solo la profundidad de la memoria se reduce a 4 palabras.

## Lo siguiente

- El die completo, con sus memorias como macros de SRAM: con las cifras de la tesela, stories260K en SkyWater 130 no cabe en una lanzadera (ver la viabilidad).
- Afinar el factor de síntesis de la coma flotante.
