# How the simulation works

llm2chip does not simulate a separate model of the chip. Everything starts from the circuit the generator builds, and each faster level of simulation is checked against the slower one under it, bit for bit and cycle for cycle, on models small enough to run in gates. Large models then run on the fastest level that is still exact.

```
   gates (netlist)  ◄── same netlist ──►  Verilog / VHDL ──► Icarus Verilog, Yosys, nextpnr, OpenROAD
        ▲  bit and cycle exact, tiny models
        │
   instruction level: RV32IM simulator + exact die machine + sampling unit model
        ▲  bit exact, any model
        │
   exact integer machine (CPU or GPU): the die's arithmetic, one token at a time
        ▲  measured: top-1 agreement, KL divergence
        │
   floating point (CPU or GPU): the reference, checked against mlx-lm
```

## The levels

| Level | Code | What runs | Exact to | Reach |
|---|---|---|---|---|
| Floating point | `sim::Engine`, `sim::accel` (Metal, Vulkan) | The model's operations in `f32`, with the real weights | `mlx-lm`, token by token | Any supported model |
| Exact machine | `sim::exact`, `sim::vm` | The compiled microprogram in the die's block floating point: integer codes, per-group scale and bias, exponents per block, function tables | The gates, bit for bit | Any supported model, including Qwen3.8-27B and the MoE |
| Instruction level | `sim::chip`, `netlist::rv32` | The real firmware on an RV32IM instruction-set simulator, driving the exact machine through the same registers the die exposes, plus the sampling unit and the load controller | The gates, cycle for cycle | Any model whose firmware tables fit |
| Gates | `netlist::sim` | The generated netlist: AND, OR, XOR, NOT, multiplexers, flip-flops and memories | The exported Verilog in Icarus | Tiny models; samples of large ones |

### Floating point

The reference. Each architecture is written once from its original code (Qwen3.5 with Gated DeltaNet, Llama, Mistral, Qwen2/3, Gemma 1-3, DeepSeek V3 with latent attention and MoE routing, Janus-Pro, Show-o with MAGVIT-v2, the Qwen3.5 vision encoder). Every architecture with tiny random weights gives the same logits as `mlx-lm` (`tests/golden/plain`), and real Qwen3-0.6B matches 24 of 24 greedy tokens. Large matrix products run on the GPU: Metal on macOS, Vulkan on Windows and Linux.

DeltaNet always uses the recurrent step, one token at a time, also while reading the prompt. The chunked form some implementations use is not what the hardware does; a test checks that both give the same result.

### The exact machine

The compiler turns the model into a program for the die: matrix-vector products in the cells, element-wise operations, norms, RoPE, attention and the delta rule in the vector lanes, and functions (SiLU, sigmoid, exponential, softplus, reciprocal square root) through tables. `sim::lower` expresses that program in the die's arithmetic, and the same lowering drives both the microcode for the gates and the exact virtual machine, so they cannot drift apart.

Vectors are stored in block floating point: one shared exponent per block of lanes and a signed mantissa per element (16 bits by default, configurable from 8 to 24). Weights stay as the integer codes of their format (for example MLX's 4-bit affine codes, `q·scale + bias` per group of 32 or 64) and are never expanded to floats; the cell adds the codes and applies scale and bias per group. Weights that arrive in floating point are first rewritten into the chosen format, the same way.

The exact machine runs at software speed and can use the GPU. What it gives is the real output of the chip, so the quality loss of a design (format, mantissa bits, state bits, table resolution) is measured, not estimated. See [benchmarks.md](benchmarks.md#quality-of-the-chips-arithmetic).

### Instruction level

The chip's CPU is an RV32IM core. Its firmware (`firmware/`, Rust, `no_std`) boots, checks the image, tokenizes, orders the die's routines, waits for them, starts the sampling unit, turns tokens into bytes and talks to the host. At instruction level, the firmware runs on an instruction-set simulator that matches the gate-level CPU cycle for cycle (fetch, decode, execute; loads one more cycle; multiply and divide 33 more). The die is the exact machine, reached through the same memory-mapped registers.

A die whose program is a set of routines (a speech recogniser, a sound decoder) is ordered by a schedule in the chip's image, which the firmware follows: the same schedule can be run on the host against the exact machine (`sim::chip::follow_schedule`), so the firmware's order is checked against the host's. Where the die has a command queue, the instruction level models it cycle for cycle ([voice.md](voice.md#the-chip-runs-its-programs-by-itself)).

This level gives the real flow: the bytes that come out of the chip, the cycles the CPU spends on each task (`netlist::activity`: receiving, tokenizing, fetching rows from the image, commanding the die, waiting, sampling, detokenizing, sending) and the cycles of each die routine.

### Gates

`netlist::sim` simulates the netlist the schematic view draws: two-state, cycle based, event driven. Only gates whose inputs changed are evaluated again, in topological order, from a bitmap of pending gates. That brought the full test suite from about 30 to about 5 minutes; a two-chip board went from 225 to 16 s.

What runs in gates:

- **The whole chip with a tiny Qwen3.5:** CPU, firmware, load controller, die and sampling unit, from input bytes to output bytes, in the same cycles as the instruction level.
- **stories260K through the whole chip:** the same text and cycles as the instruction level, and the same logits as the exact machine.
- **Several chips on a board:** two or four chips with delays and flipped bits on their links, split by layers, by rows or both, give the same logits as one die.
- **Images:** a toy Show-o filling an image through the firmware (12.45 M cycles), the MAGVIT-v2 toy encoder and decoder (2.65 M cycles for the decoder), and the Qwen-style vision encoder, patch by patch.
- **Voice:** toy Moonshine (280,614 cycles), streaming Moonshine (900,912) and DAC (70,624) on the die bit for bit, Moonshine's whole chip following its schedule through a command queue (1,192,263 cycles, the same bytes as the instruction level), and one layer of the real Moonshine-tiny on the die (13.6 M cycles, `LLM2CHIP_REAL_MOONSHINE=1`).
- **Reliability:** ECC with injected faults, spare rows, the memory self-test, JTAG, scan chains with ATPG patterns.

Gates are also exported to Verilog and VHDL, and the exported chip runs in Icarus Verilog with its firmware, giving the same bytes and cycles (`LLM2CHIP_ICARUS=1`, close to an hour).

## Why large models are not simulated in gates

A Qwen3.8-27B die has around 136,000 cells, each with its multiplier and its weight memory, and one token takes hundreds of thousands of cycles. Simulating that gate by gate would take far longer than a token is worth. Instead:

1. The gates and the exact machine are compared bit for bit on tiny models, in every buffer, at several vector widths (12, 16, 18 and 20 bits).
2. The exact machine runs the real model, with the same lowering and the same microprogram generator.
3. Samples of the real model (a block of real weights through the load controller and the cells) go through gates and must match the exact machine word for word.

## Where the cycles come from

Speed figures come from two places, and the application says which one a number comes from:

- **The estimator** (`arch`) computes area, energy and speed analytically from the model's shape and the parameters, in milliseconds, so every slider responds instantly. Its multipliers, paths and switching come from cells generated and measured in gates, and its areas are calibrated against OpenROAD layouts and published chips.
- **The circuit** (`sim::sizing`) generates the real microprogram for a die shape and follows its jumps to the last position of the context, so the cycles are those the circuit spends with the context full. The clock comes from the critical path of the generated circuit. This is what the sizing search in the application uses when a target speed is given.

## Profiles in the application

The Simulator view chooses a level per block:

| Profile | Inference core | Rest of the chip | Use |
|---|---|---|---|
| Fast | Floating point, GPU | Software | Trying models and prompts |
| Real arithmetic | Exact machine, GPU | Software | Measuring a design's quality |
| Real flow | Exact machine, GPU | Real firmware at instruction level, sampling unit, image checks | Seeing the whole chip at work, with the CPU's activity per token |

For each token it shows the simulated cycles, the time the chip would take and what the CPU, the die and the sampling unit did, in order. Every cycle shown says whether it was simulated or estimated.

### A conversation in the simulator

The simulator holds a conversation as the chip does ([full-chip.md](full-chip.md#a-conversation-and-the-clear-pin)). Each question goes on over the cache the earlier ones left. On top of it goes only what the chat template writes after the last answer and the new question (`PromptStyle::next_turn`). The view shows the turns, and how much of the chip's context they take.

- **New conversation** empties the cache, as the clear pin does. The button tells how long the pin takes in the design.
- **Through the real firmware**, the simulated chip stays on between questions with the engine inside it. A new conversation on the same chip raises its clear pin. The view shows the cycles the firmware took to handle it. There the engine stands in for the cache, so emptying the cache is not counted, and the view shows the design's estimate beside them.
- **What ends a conversation:**
  - loading a model;
  - changing the chip, the system prompt or, through the firmware, the generation settings;
  - a question about a picture, or drawing one;
  - stopping an answer halfway.
- **A full context** ends the answer. A question that does not fit is refused until a new conversation starts.
- **In the browser** (the online demo) it works the same, on the die alone.

`a_conversation_goes_on_over_the_cache_in_the_browser` (in `crates/wasm-sim/tests/parity.rs`) checks the browser's simulator with SmolLM2-135M. A second question answers as the whole conversation fed at once does, and a new conversation answers the first question as the first time.

## Verification at a glance

| Check | Where |
|---|---|
| Floating point against `mlx-lm` on every architecture | `crates/sim/tests/plain_families.rs`, `tests/golden/plain` |
| Exact machine against gates, every buffer | `crates/sim/tests/tiny_model.rs` |
| Firmware at instruction level against gates, same bytes and cycles | `crates/sim/tests/chip.rs`, `multichip.rs` |
| Exported Verilog in Icarus (close to an hour) | `LLM2CHIP_ICARUS=1 cargo test -p sim --release --test chip -- --ignored` |
| Images in chip arithmetic and in gates | `vision_chip.rs`, `showo_chip.rs`, `magvit_chip.rs` |
| Voice in chip arithmetic, in gates and through the firmware; the command queue in gates | `moonshine.rs`, `moonshine_stream.rs`, `dac.rs`, `conversation.rs`, `chip.rs` ([voice.md](voice.md#tests)) |
| Speech models against Transformers | `tools/golden/*.py`, `listen`, `speak`, `dac_check` |
| Tokenizer against the reference library | `crates/sim/tests/tokenizer.rs` |
| Every weight format and multiplier against reference arithmetic | `crates/netlist/tests` |
| Estimator against OpenROAD layouts and published chips | `tests/golden/asic.json`, `docs/asic.md`, `docs/research.md` |

`scripts/check.sh` runs the quick suite in about a minute and a half; `scripts/check.sh --full` adds the whole chips in gates (about 5 minutes more).
