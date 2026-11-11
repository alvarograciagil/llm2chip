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

Las memorias siguen siendo una estimación: las macros abiertas de SRAM para SkyWater 130 (OpenRAM) ocupan unos 23 µm² por bit en 8 Kbit, y el modelo usa esa cifra para los bloques pequeños.

## Desde la aplicación

Con el destino «Chip propio» en SkyWater 130, GF180 o ASAP7, el inspector tiene el apartado **Layout real de la celda**. «Hacer el layout» toma la celda que el generador eligió para el diseño (codificación, bits, multiplicador y etapas), la pasa por OpenROAD en segundo plano y muestra la imagen del layout con su área, frecuencia máxima, potencia, número de celdas y violaciones de reglas, al lado de lo que estimaba el modelo. El trabajo queda en `~/.llm2chip/asic/`.

«Usar las cifras medidas» guarda en el diseño el área y la frecuencia medidas (`measured_cell`, `measured_cell_um2`, `measured_cell_fmax_ghz`). El estimador las usa en lugar de las suyas solo para esa misma celda en ese mismo proceso: si cambian el circuito o el proceso, vuelve a la estimación. Las memorias, la red de activaciones y el resto del die siguen estimándose.

La celda que se mide es la que eligió el estimador, con todo lo que cambia sus puertas (codificación, bits, acumulador, multiplicador, etapas, ECC, salto de ceros); solo la profundidad de la memoria se reduce a 4 palabras.

## Lo siguiente

- El die completo, con sus memorias como macros de SRAM en vez de biestables.
- Afinar el factor de síntesis de la coma flotante.
