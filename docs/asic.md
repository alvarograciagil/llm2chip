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

Con esa medida, la tecnología `sky130` del estimador toma:

- las áreas de puerta de la propia biblioteca de celdas (`sky130_fd_sc_hd`: and2 6,26 µm², xor2 8,76 µm², mux2 11,26 µm², biestable 20,02 µm²);
- el retardo de un sumador completo, 0,49 ns, de ajustar el camino crítico que predice el modelo de la celda (16 sumadores) a los 8,38 ns que mide OpenROAD tras rutar, más 0,6 ns del biestable.

Las memorias siguen siendo una estimación: las macros abiertas de SRAM para SkyWater 130 (OpenRAM) ocupan unos 23 µm² por bit en 8 Kbit, y el modelo usa esa cifra para los bloques pequeños.

## Desde la aplicación

Con el destino «Chip propio» en SkyWater 130, GF180 o ASAP7, el inspector tiene el apartado **Layout real de la celda**. «Hacer el layout» toma la celda que el generador eligió para el diseño (codificación, bits, multiplicador y etapas), la pasa por OpenROAD en segundo plano y muestra la imagen del layout con su área, frecuencia máxima, potencia, número de celdas y violaciones de reglas, al lado de lo que estimaba el modelo. El trabajo queda en `~/.llm2chip/asic/`.

«Usar las cifras medidas» guarda en el diseño el área y la frecuencia medidas (`measured_cell`, `measured_cell_um2`, `measured_cell_fmax_ghz`). El estimador las usa en lugar de las suyas solo para esa misma celda en ese mismo proceso: si cambian el circuito o el proceso, vuelve a la estimación. Las memorias, la red de activaciones y el resto del die siguen estimándose.

Primera medida desde la aplicación (stories260K, Booth con 3 etapas de multiplicador y 2 de suma, SkyWater 130): 10 471 µm² y 264 MHz, frente a 14 955 µm² y 200 MHz estimados, con 3,98 mW y 0 violaciones. Por eso el tope de frecuencia de `sky130` sube de 0,2 a 0,35 GHz.

## Lo siguiente

- El die completo, con sus memorias como macros de SRAM en vez de biestables.
- Calibrar el resto de tecnologías del mismo modo.
