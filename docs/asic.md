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

## Lo siguiente

- El die completo, con sus memorias como macros de SRAM en vez de biestables.
- Lanzarlo desde la aplicación para el destino «Chip propio», con el layout y sus cifras a la vista.
- Calibrar el resto de tecnologías del mismo modo.
