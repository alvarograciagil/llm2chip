# llm2chip

Convierte un modelo de lenguaje en el diseño de un chip a medida: lo dimensiona, genera su circuito, lo simula y exporta el hardware.

## Arrancar

```bash
scripts/dev.sh
```

Abre `http://localhost:5173`. Para la app de Mac:

```bash
scripts/build-app.sh
```

Comprobaciones completas: `scripts/check.sh`.

## Pantallas

- **Plano:** áreas del chip con zoom desde el sistema hasta la celda.
- **Esquemático:** el circuito real por áreas, hasta la puerta lógica, simulable paso a paso y exportable a Verilog y VHDL.
- **Simulador:** escribe un texto y ejecuta el microprograma con pesos reales, en coma flotante o con los enteros exactos del chip, convirtiendo los pesos al formato del diseño.
- **Explorador:** barridos, frontera de Pareto, comparación de formatos de peso y calibración.

## Formatos de peso

El formato es un parámetro del chip y cada uno genera su propio circuito: punto fijo de Q2 a Q32, afín por grupos (MLX, GGUF), coma flotante de FP8 a FP32, microescalado (MXFP, NVFP4), posit, logarítmico y libro de códigos. Hay 72 plantillas y también se puede configurar cada familia a mano.

## Exportación

La celda se exporta como proyecto completo: Verilog, VHDL, banco de pruebas, restricciones, guiones de Yosys y Vivado y un Makefile. La barra superior genera además una hoja de datos del diseño.

## Estructura

| Carpeta | Contenido |
|---|---|
| `crates/model-ir` | Modelo independiente del hardware, desde `config.json` |
| `crates/arch` | Parámetros, tecnología, estimador y plano |
| `crates/compiler` | Microprograma y calendario |
| `crates/sim` | Ejecutor sobre pesos reales, máquina de enteros exactos y bajada a puertas |
| `crates/netlist` | Generador de circuitos, simulador por puertas, Verilog y VHDL |
| `crates/server` | Motor local |
| `crates/wasm` | Enlace con el navegador |
| `web` | Interfaz |
| `app` | Empaquetado para macOS, Windows y Linux |
| `tests/golden` | Mediciones sobre pesos reales |
| `tools/synth` | Comparación de la netlist generada con Yosys y ABC |
| `docs` | Notas de investigación y estado del arte |

## Qué está verificado

- Qwen3.8-27B genera los mismos tokens que MLX en la prueba de referencia, con el microprograma.
- Un modelo Qwen3.5 completo de tamaño diminuto se ejecuta entero en puertas y sus logits coinciden con la coma flotante. También uno con mezcla de expertos.
- La máquina de enteros exactos coincide bit a bit con las puertas en todos los búferes, y en Qwen3.8-27B real da el mismo token que la coma flotante en 29 de 30 casos.
- Los decodificadores de cada formato y los multiplicadores array, Wallace y Booth se comprueban contra la aritmética de referencia en todos los anchos.
- El Verilog y el VHDL exportados se comprueban con Icarus y nvc.

## Límites

Es un diseño de arquitectura y lógica, no un layout fabricable. Los modelos grandes se simulan con la máquina de enteros: su circuito completo no se simula puerta a puerta. Esa máquina usa mantisas de 16 bits, así que no refleja formatos de peso más anchos. El área de la lógica usa celdas ASAP7 reales sin optimizar por síntesis; las memorias y la energía están calibradas solo contra el HC1 de Taalas.
