# Chip layout with OpenROAD

The Verilog that llm2chip generates can be taken all the way to a manufacturable layout (GDSII) with open tools: synthesis with Yosys, and floorplan, placement, clock tree, routing and checks with OpenROAD, on open PDKs that are really manufactured on multi-project shuttles.

| PDK | What it is | Manufactured |
|---|---|---|
| SkyWater 130 (`sky130hd`) | 130 nm, 5 metal layers | Yes: open shuttles such as Tiny Tapeout (and formerly Efabless) |
| GlobalFoundries 180 (`gf180`) | 180 nm, 5 V | Yes, on open shuttles |
| IHP SG13G2 (`ihp-sg13g2`) | 130 nm BiCMOS | Yes, on IHP's shuttles |
| ASAP7 (`asap7`) | 7 nm predictive | No: it shows what the design would look like in an advanced node |

The N6, N5 and N3 nodes have no open PDK: the estimator models them, but no layout can be made for them.

## How to run it

You need Docker with the `openroad/orfs` image (OpenROAD-flow-scripts, about 6.5 GB; on an Apple Silicon Mac it runs emulated, more slowly).

```bash
tools/asic/flow.sh design.v MODULE sky130hd 10
```

It leaves in `~/.llm2chip/asic/MODULE-platform/` the GDS, images of the layout and `summary.json` with the area, maximum frequency, power and rule violations.

## First measurement: the compute cell

The cell the die repeats thousands of times (an 8-bit affine weight times a 16-bit activation, Wallace multiplier, scale, bias and accumulator), exported with `cargo run -p netlist --example export -- affine 8 0 wallace`:

| PDK | Cell area | Maximum frequency | Power | Violations |
|---|---|---|---|---|
| SkyWater 130 | 8,911 µm² (1,110 cells) | 119 MHz | 1.19 mW at 100 MHz | 0 |
| GF180 | 41,614 µm² (1,101 cells) | 125 MHz | 26.6 mW at 67 MHz | 0 |

## Calibrating the estimator

`tools/asic/calibrate.sh` lays out five different cells (those in `tools/asic/cells.json`: 8-bit affine with an unpipelined and a pipelined Wallace multiplier and with a pipelined Booth multiplier, 4-bit integer with an array multiplier, and 8-bit floating point) on the three open processes, two at a time, and stores what it measures in `tests/golden/asic.json`.

How it measures:

- Each cell holds 4 words of weights in flip-flops, without ECC.
- The target clock is fixed per process (4 ns on SkyWater 130, 6 ns on GF180, 500 ps on ASAP7, whose libraries count picoseconds), close to what the process can do, so OpenROAD optimizes timing without inflating the area.
- The maximum frequency is the one OpenROAD computes after routing, with the slack left over or missing.

`cargo run -p arch --example calibrate` compares each measurement with what the estimator expects for that same cell, counting its 4 words as flip-flops with their multiplexer, and fits by least squares the delay of a full adder and of a flip-flop for each process:

| Cell | Process | Measured area µm² | Estimated | Ratio | Measured clock MHz | Estimated | Ratio |
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

| Process | Full adder | Flip-flop | Clock ceiling | Gate areas |
|---|---|---|---|---|
| SkyWater 130 | 0.41 ns | 0.51 ns | 400 MHz | `sky130_fd_sc_hd` |
| GF180 | 0.38 ns | 1.58 ns | 300 MHz | `gf180mcu_fd_sc_mcu9t5v0`, the 9-track library OpenROAD uses (another one, with half the area, was used before) |
| ASAP7 | 0.038 ns | 0.13 ns | 3 GHz | `asap7sc7p5t` |

The clock is within 12 % on all fifteen cells. The area is between 62 % and 112 % of the measurement; floating point strays the most, because Yosys' synthesis factor was measured on integer multipliers. The test `crates/arch/tests/asic.rs` checks those margins against the stored measurements, without Docker.

Cells whose target clock is not met end up with a few dozen rule violations in detailed routing; those that meet it, with none. Area and frequency do not change because of it.

## Error correction on the read path

With SRAM, each cell keeps its weights in words of about 64 bits with check bits, and corrects the weight it reads. The estimator measured the cell's clock with a single-word memory, without a corrector, and so did the layout. With the corrector, the critical path went from about 8 full-adder delays to 35: a 16-bit cell in N6 could not go above 0.62 GHz, against the more than 2 GHz estimated.

Three changes fix it:

- The parities, the bit count of the syndrome and the lane check were linear chains of gates; now they are balanced trees, with the same logic.
- The corrected weight is registered before the multiplier, and the activation with it: correction gets its own cycle, and the cell's latency goes up by one. Without that register, the cell would be between 35 and 55 % slower.
- The estimator measures the clock, and the layout is made, with the smallest memory that keeps the real cell's correction (`timed_cell`). Above that size the path no longer grows.

With all of it, the path of the cell with correction is back to 7.4-8.3 delays, the same as without it.

## Memories

Memories are SRAM macros, and on the open processes only small ones exist. The SkyWater 130 ones that come with OpenROAD (`sky130_sram_1rw1r_*`, from OpenRAM) reach 32 Kbit and take 35.5 µm² per bit at 5 Kbit, 25.6 at 16 Kbit and 22.0 at 32 Kbit; the model reproduces them with 13.17 µm² per bit and a periphery of 3.79, and a larger memory is built from several 32 Kbit ones. GF180 uses GlobalFoundries' own, of 4 Kbit and about 51 µm² per bit. Before, the model assumed macros of any size, with densities 5 times better than the real ones.

## A tile with a real SRAM macro

`tools/asic/tile.sh` lays out a cell with its weight memory in a real SkyWater 130 SRAM macro, not in flip-flops: the 8-bit affine cell with a Wallace multiplier in 3 + 2 stages keeps 4096 weights (32 Kbit) in an OpenRAM `sky130_sram_1rw1r_128x256_8` macro. Each 128-bit word of the macro holds 16 weights: the high address bits choose the word, the low ones, registered for a cycle, the weight, and a write only touches its byte through the macro's mask.

| Measurement | Value |
|---|---|
| Macro | 0.722 mm² (the estimator gives the same, within 5 %) |
| Logic | 0.031 mm² |
| Die | 1.84 mm² |
| Maximum clock | 211 MHz |
| Power | 38.7 mW |
| Violations after routing | 0 |

The clock is set by the macro, not the logic: it delivers the data on the falling edge, so the read gets half a cycle, and the cell alone reached 367 MHz. The `sky130` technology now carries that minimum period (`sram_min_period_ns`, 4.74 ns), and the estimator limits the clock of designs with weights in SRAM to it.

Integrating it took three things:

- Connecting its power pins, which OpenRAM calls `vdd` and `gnd`.
- Cutting the rows of cells 10 µm away from the macro, so the vias of their rails do not land on the macro.
- Giving the macro its own power grid: met4 stripes over it, with vias to its thousands of met3 power islands and to met5, kept to its outline.

OpenROAD's power grid analysis is left out: this macro's abstract model has, in two columns, overlapping `vdd` and `gnd` rectangles in met3 (for instance `vdd` at y = 109.815-110.305 and `gnd` at 109.875-110.365, at x = 214.22), and the analysis takes them for shorts.

KLayout's DRC on the final GDS (`make drc`) gives 522,617 violations:

- 521,993 inside the macro's bit cells, which follow special SRAM rules the standard deck does not exempt. A calibration cell with no macro gives 0.
- 582 in the top cell, over the macro: met1-met4 spacings between what the flow draws (its grid's met4 stripes, the vias to its met3 islands and the connections to its pins) and metal in the macro's drawing that its abstract model does not declare; the model says met4 is free.
- 42 outside the macro, on vias of the power network.

LVS is not run: it needs the macro's CDL netlist, which the flow does not find.

So this macro, the one that comes with the OpenROAD image and that no example design uses, is not fit for manufacturing: its abstract model is incomplete. Its area and clock figures are valid as measurements, but a real chip would need the OpenRAM macros for SkyWater 130 published with met4 power rings and complete abstract models, downloaded separately, and would have to pass DRC and LVS with them.

## Feasibility before the layout

Before any layout, the inspector says whether the design can be manufactured on the chosen process (`arch::feasibility`):

- **What stops it,** with figures: how much each die needs against what the process allows, and the three largest blocks; the estimator's errors; and, for the layout, whether the process has an open PDK and whether the memories the design uses exist on it (SRAM only).
- **How to make it feasible:** it tries parameter changes (splitting across more chips, sizing for the model, the native context, fewer weight or KV cache bits, fewer multipliers or lanes, smaller function tables, the whole die the process allows, another open process that is manufactured) and only offers those that, estimated again, pass the check. If no single change is enough, it looks for a combination change by change, and then for another with the fewest chips without accepting large losses of quality. Each proposal shows its chips, its area, its tok/s and what it risks.
- **What the number of chips means:** how many packages and boards are needed (about 16 packages per board for large chips, 64 for small shuttle chips), the time each token spends on the links, and that the chips are one netlist repeated that loads its part of the model from the image, so with weights in SRAM one design is enough; only with weights burnt into ROM is each die different. Every chip has its CPU, the CPU's memory, the tokenizer tables and sampling, even though only the first one uses them.
- **Links on an open shuttle:** each die arrives in its own package and only has its I/O pins to talk to the others. In automatic mode, SkyWater 130 and GF180 use pin links (16 pins per direction at 50 MHz, about 60 ns per hop), and the tok/s count the time the vector takes to cross them.

The search makes about a hundred estimates and takes under two seconds natively: the generated cells, their critical paths and their switching are cached, because they repeat from one estimate to the next.

When the model changes, the design is sized for it: its limits (parameters, layers, width, vocabulary) become the model's, the context does not exceed the native one, and the multipliers, vector lanes and function units follow the default design's proportion, or the requested speed if there is one.

The layout button stays disabled while the design is not feasible.

Example: stories260K on SkyWater 130 does not fit in a 10 mm² die (the weight memory alone takes more than 45 mm²). The proposal with the fewest chips splits it into 42 dies of 10 mm² with a 512-token context and an 8-bit KV cache.

## From the application

With the "Custom chip" target on SkyWater 130, GF180 or ASAP7, the inspector has a **Real layout of the cell** section. "Make the layout" takes the cell the generator chose for the design (coding, bits, multiplier and stages), runs it through OpenROAD in the background and shows the image of the layout with its area, maximum frequency, power, number of cells and rule violations, next to what the model estimated. The work stays in `~/.llm2chip/asic/`.

"Use the measured figures" stores the measured area and frequency in the design (`measured_cell`, `measured_cell_um2`, `measured_cell_fmax_ghz`). The estimator uses them instead of its own only for that same cell on that same process: if the circuit or the process changes, it goes back to estimating. The memories, the activation network and the rest of the die are still estimated.

The cell measured is the one the estimator chose, with everything that changes its gates (coding, bits, accumulator, multiplier, stages, ECC, zero skipping); only the depth of the memory is reduced to 4 words.

## Next

- The whole die, with its memories as SRAM macros: with the tile's figures, stories260K on SkyWater 130 does not fit on a shuttle (see feasibility).
- Refining the synthesis factor for floating point.
