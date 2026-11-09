# llm2chip

**Da a llm2chip un modelo de lenguaje y unos objetivos, y genera el circuito de un chip hecho a su medida.** Permite ver ese circuito, cambiar sus parámetros, ejecutar el modelo sobre él y exportarlo a Verilog y VHDL.

La idea de fondo es llevar la especialización hasta el final: en vez de un acelerador genérico que mueve pesos desde una memoria externa, cada multiplicador tiene al lado los pesos que usa, y todo el chip se dimensiona para un modelo o una familia de modelos concreta.

## Para qué sirve

- **Explorar el diseño de un chip para LLM** sin escribir RTL: cuánta área, energía, potencia y velocidad cuesta un modelo con cada decisión.
- **Comparar decisiones de arquitectura** con cifras: formato de los pesos, SRAM, ROM o DRAM apilada, número de multiplicadores, etapas de pipeline, chips iguales o a medida.
- **Medir qué le pasa al modelo real** al usar la aritmética del chip: ejecuta pesos reales con los enteros exactos que usaría el silicio y compara el resultado con la coma flotante.
- **Entender cómo funciona por dentro** un chip así, desde las capas del modelo hasta cada puerta lógica.
- **Tener un punto de partida en RTL** para síntesis, FPGA o investigación.

## Cómo funciona

```
config.json + pesos          parámetros y objetivos
        │                             │
        ▼                             ▼
   grafo del modelo ──────────► generador del circuito
                                      │  construye y mide cada variante
                                      ▼
              ┌──────────────┬────────┴───────┬──────────────┐
              ▼              ▼                ▼              ▼
          plano y       microprograma      netlist en     Verilog
         estimación      y firmware         puertas      y VHDL
                             │                │
                             ▼                ▼
                     simulación con      simulación
                       pesos reales     puerta a puerta
```

Lo que dibuja el esquemático es la misma netlist que se simula y se exporta: no hay bloques decorativos.

**El circuito cambia con los parámetros.** Para cada combinación, el generador construye en puertas cada forma posible de la celda de cálculo, mide su área, su conmutación y su camino crítico, y se queda con la mejor que alcanza la frecuencia pedida. Con una velocidad objetivo, el número de multiplicadores sale del tamaño del modelo.

**Vale para un modelo o para muchos.** Se puede generar cada die a medida de un modelo concreto, con la máxima eficiencia, o repetir un único die y conectar más para admitir modelos mayores, hasta unos límites de capas, ancho y vocabulario. Las capas no son un valor fijo: salen del `config.json` de cada modelo.

## Qué lleva el chip

| Parte | Función |
|---|---|
| Celdas de cálculo | Memoria de pesos junto a su multiplicador, con ECC o en ROM |
| Red de activaciones | Reparte cada vector a las celdas y recoge las sumas |
| Unidades vectoriales y de función | Atención, regla delta, SiLU, sigmoide, exponencial |
| Secuenciador | Un microprograma fijo, con bucles y parámetros, para cualquier token |
| CPU RV32IM | Firmware real en Rust: arranque, tokenizador y control |
| Controlador de carga | Lleva la imagen a cada memoria y comprueba su CRC |
| Unidad de muestreo | Top-k y temperatura, con filtro de logits en paralelo |
| Enlaces entre chips | Datos y órdenes con corrección de errores y control de flujo |
| Fiabilidad y depuración | Autotest de memoria, filas de reserva, contadores y puerto de depuración |

Todos los chips de un sistema salen de la misma netlist. Una identidad fijada en la placa decide cuál actúa de maestro.

## La aplicación

- **Plano:** áreas del chip, con zoom desde el sistema hasta la celda.
- **Funcionamiento:** qué operaciones hace el modelo, capa a capa.
- **Esquemático:** el circuito real por áreas, hasta la puerta lógica, simulable paso a paso.
- **Simulador:** escribe un texto y ejecútalo sobre el chip, eligiendo cuánto simular de cada bloque.
- **Explorador:** barridos, frontera de Pareto, comparación de formatos y calibración.

### Cuánto simular

| Perfil | Núcleo de inferencia | Resto del chip | Uso |
|---|---|---|---|
| Rápida | Coma flotante, con GPU | Software | Probar modelos y formatos |
| Aritmética real | Enteros exactos, con GPU | Software | Medir la calidad del diseño |
| Flujo real | Enteros exactos, con GPU | Firmware real, unidad de muestreo y comprobación de la imagen | Ver el chip completo |

La aceleración usa Metal en macOS y Vulkan en Windows y Linux. La simulación puerta a puerta completa se reserva para modelos diminutos.

## Empezar

Hace falta Rust con `wasm32-unknown-unknown` y `riscv32im-unknown-none-elf`, `wasm-pack` y Node. Icarus Verilog y Yosys son opcionales.

```bash
scripts/dev.sh
```

Abre `http://localhost:5173`. El motor local solo escucha en `127.0.0.1`. Para empaquetar la aplicación de escritorio:

```bash
scripts/build-app.sh
```

Para pasar todas las comprobaciones:

```bash
scripts/check.sh
```

Windows tiene los equivalentes `.ps1`. Los pesos no se incluyen: el simulador busca modelos en `~/.mtplx/models` y en la caché de Hugging Face.

## Qué está verificado

- La netlist de un chip completo, de bytes de entrada a bytes de salida, coincide ciclo a ciclo con la simulación rápida en un modelo Qwen3.5 diminuto.
- Dos chips idénticos en una placa simulada, con retardos y bits invertidos en los enlaces, generan el mismo texto y los mismos logits que uno solo.
- La máquina de enteros exactos coincide bit a bit con las puertas en todos los búferes de esos modelos.
- Qwen3.8-27B real se ejecuta de principio a fin con el firmware del chip. Su tokenizador coincide con la biblioteca de referencia en los textos de prueba.
- Cada formato de peso y cada multiplicador se comprueban contra la aritmética de referencia, y el Verilog y el VHDL exportados se ejecutan en simuladores externos.

Los detalles y mediciones están en [`docs/`](docs) y [`tests/golden/`](tests/golden).

## Límites

Es un diseño de arquitectura y lógica, no un layout fabricable: no hay colocación ni rutado, ni firma de temporización, ni PDK comercial. En concreto:

- Área, energía y velocidad son estimaciones, parte derivadas del circuito generado y parte calibradas con chips publicados. La aplicación indica el origen de cada cifra.
- Los modelos reales no se simulan puerta a puerta, sino con una máquina verificada contra las puertas en modelos diminutos.
- El enlace entre chips corrige errores simples, pero no retransmite; el reloj se modela común a todos los chips y falta JTAG estándar.
- Los paquetes de Windows y Linux y el backend de Vulkan todavía no se han ejecutado en esos sistemas.

## Estructura

| Carpeta | Contenido |
|---|---|
| `crates/model-ir` | Modelo independiente del hardware, a partir de `config.json` |
| `crates/arch` | Parámetros, tecnología, estimador y plano |
| `crates/compiler` | Programa del modelo y calendario |
| `crates/netlist` | Generador de circuitos, simulador de puertas, Verilog y VHDL |
| `crates/sim` | Ejecución sobre pesos reales, máquina exacta y GPU |
| `crates/server` | Motor local |
| `crates/wasm` | Enlace con el navegador |
| `firmware` | Firmware de la CPU del chip |
| `web` | Interfaz |
| `app` | Aplicación de escritorio |
| `tech`, `models`, `presets` | Nodos tecnológicos, modelos y diseños de partida |
| `tools` | Referencia en Python y comparación con Yosys |
