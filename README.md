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
- **Simulador:** escribe un texto y ejecuta el microprograma con pesos reales.
- **Explorador:** barridos, frontera de Pareto y calibración.

## Estructura

| Carpeta | Contenido |
|---|---|
| `crates/model-ir` | Modelo independiente del hardware, desde `config.json` |
| `crates/arch` | Parámetros, tecnología, estimador y plano |
| `crates/compiler` | Microprograma y calendario |
| `crates/sim` | Ejecutor sobre pesos reales y bajada a puertas |
| `crates/netlist` | Generador de circuitos, simulador por puertas, Verilog y VHDL |
| `crates/server` | Motor local |
| `crates/wasm` | Enlace con el navegador |
| `web` | Interfaz |
| `app` | Empaquetado para macOS |
| `tests/golden` | Mediciones sobre pesos reales |

## Qué está verificado

- Qwen3.8-27B genera los mismos tokens que MLX en la prueba de referencia, con el microprograma.
- Un modelo Qwen3.5 completo de tamaño diminuto se ejecuta entero en puertas y sus logits coinciden con la coma flotante.
- El Verilog y el VHDL exportados se comprueban con Icarus y nvc.

## Límites

Es un diseño de arquitectura y lógica, no un layout fabricable. Los modelos grandes se simulan con el microprograma: su circuito completo no se simula puerta a puerta. El área de la lógica usa celdas ASAP7 reales sin optimizar por síntesis; las memorias y la energía están calibradas solo contra el HC1 de Taalas.
