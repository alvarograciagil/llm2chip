# Benchmarks and results

What llm2chip measures, and what it gives today. Every figure says where it comes from:

- **Measured:** from a run of the circuit, the exact machine or the gates, with the command to repeat it ([examples.md](examples.md)).
- **Estimated:** from the analytic estimator, which is calibrated against OpenROAD layouts and published chips but is not a sign-off.

Nothing here has run on silicon or on an FPGA board. Simulation times are from an Apple M2 Max with 64 GB of memory.

## Qwen3.8-27B on a chip

The main target: Qwen3.8-27B (27 billion parameters, 64 layers mixing Gated DeltaNet and full attention, with an MTP head and a vision encoder), from the MLX checkpoint `Qwen3.8-27B-MTPLX-Optimized-Speed-FP16`. Its weights are mixed: 4-bit affine codes in groups of 32, and 8-bit in groups of 64 for the embedding, the output head, DeltaNet's output projection and the last eight MLPs. The chip keeps them as they are.

### The default design, estimated

`cargo run --release -p arch --example report` with the application's default parameters: weights in SRAM next to their multipliers, TSMC N6, 1 GHz, 16-bit activations, one user.

| | |
|---|---|
| Parameters | 26.9 G |
| Weights in memory | 19.5 GB (chip capacity 25.4 GB) |
| Dies | 13 of 806.5 mm² (reticle limit 815 mm²), in 3 packages |
| Total silicon | 10,485 mm² |
| Weight multipliers | 1,768,000 (136,000 per die), 14.4 KB of weights each |
| Generation, one user | **3,755 tok/s** (265,500 cycles per token) |
| Prompt reading | 48,960 tok/s |
| Energy | 56 mJ per token, **240 W** while generating |
| Weights read per token | 18.2 GB, 68 TB/s from the cells' own memories |
| KV cache | 0.54 GB, 64 KiB per context token |
| DeltaNet and convolution state | 154 MB |
| Die-to-die traffic | 38 MB/s |
| Time to load the image | 1.4 s |

Where each die's area goes:

| Block | mm² per die |
|---|---|
| Cells: weight memory | 717.2 |
| Cells: multipliers | 24.3 |
| Scale and bias per group | 17.1 |
| Attention KV cache | 12.0 |
| Cells: ECC correction | 7.2 |
| Control and load CPU | 6.0 |
| Clock, power and test | 6.0 |
| Tokenizer tables | 3.9 |
| DeltaNet state | 3.4 |
| Die-to-die links | 2.9 |
| Vector units | 2.7 |
| Activation network | 2.5 |
| Sequencer, CPU memory, function units, buffers | 1.4 |

Weight memory takes 89 % of the silicon: this kind of chip is a memory with multipliers inside, and SRAM density, which has barely improved since N5, sets its size and cost. That is why weight format matters more than anything else (see below), and why the estimator offers ROM (weights fixed at manufacture) and stacked DRAM as alternatives.

For reference, the same estimator gives 787 mm² and 15,234 tok/s for the Taalas HC1 preset (Llama 3.1 8B in mask ROM, N6), against the 815 mm² and more than 15,000 tok/s published for that chip ([research.md](research.md)).

### Speed from the circuit

The estimator assumes the cells stream weights at full rate. `sim::sizing` checks that against the real circuit: for each die shape it generates the model's microprogram, follows its loops and jumps to the last position of the context, and takes the clock from the generated circuit's critical path. That figure is guaranteed with the context full.

One token at the last position of an 8,192-token context, on a die with 128 vector units of 128 lanes and 262,144 cells, following the real microprogram's jumps (`kinds` example):

| Operation | Cycles | Share |
|---|---|---|
| Attention sum (scores × values) | 181,376 | 34 % |
| Matrix products in the cells | 157,708 | 30 % |
| DeltaNet delta rule | 73,056 | 14 % |
| RMSNorm | 35,238 | 7 % |
| SiLU, element-wise products, RoPE, gated norms, the rest | 86,454 | 16 % |
| **Total** | **533,832** | |

The generated die's critical path is about 25 full-adder delays (measured on small dies and extrapolated by the sizing search), which in N6 (0.04 ns per full adder, estimated) allows about 1 GHz: **about 1,870 tok/s** for one user with the context full, half of what the estimator gives. Matrix products take only 30 % of the circuit's cycles; attention at full context and the element-wise work in the vector lanes take the rest, which is where the estimator is optimistic. Attention grows with the context, so these figures are the slowest case.

That die is a speed reference, not one that fits: with 262,144 cells it is far larger than the reticle.

### Dies sized from the circuit

`frontier` builds the real microprogram of every chip for each die shape, counts the words each vector unit's memory needs, adds the cells' memory, the microcode memory and the logic, and keeps the shapes whose dies fit the reticle (858 mm²). For Qwen3.8-27B with an 8,192-token context and a 1,024-token prompt (N6, estimated areas; 8, 12 or 16 chips, the default of `CHIPS`). Each row is a point of the speed-for-area frontier.

Split by layers (`SHARING=layers`): each chip runs whole layers and passes one vector to the next.

| Chips | Units × lanes | Cells per die | tok/s | Clock | First token | Total area | Per chip |
|---|---|---|---|---|---|---|---|
| 8 | 64 × 64 | 8,192 | 203 | 994 MHz | 608 ms | 6,295 mm² | 787 mm² |
| 8 | 64 × 128 | 16,384 | 419 | 1,010 MHz | 297 ms | 6,689 mm² | 836 mm² |
| 12 | 64 × 128 | 32,768 | 669 | 1,010 MHz | 117 ms | 7,680 mm² | 640 mm² |
| 12 | 64 × 128 | 65,536 | 954 | 1,010 MHz | 76 ms | 8,629 mm² | 719 mm² |
| 12 | 128 × 128 | 65,536 | 1,057 | 989 MHz | 76 ms | 9,856 mm² | 821 mm² |
| 16 | 64 × 128 | 131,072 | 1,187 | 1,010 MHz | 49 ms | 13,235 mm² | 827 mm² |

Mixed split in groups of four (`SHARING=mixed:4`): the four chips of a group share every layer of their part, each with a quarter of each matrix's rows and a quarter of the attention and DeltaNet heads, and the groups are chained.

| Chips | Units × lanes | Cells per die | tok/s | Clock | First token | Total area | Per chip |
|---|---|---|---|---|---|---|---|
| 12 | 32 × 64 | 8,192 | 709 | 1,014 MHz | 423 ms | 5,693 mm² | 474 mm² |
| 12 | 32 × 128 | 8,192 | 884 | 1,031 MHz | 369 ms | 5,779 mm² | 482 mm² |
| 12 | 32 × 64 | 16,384 | 985 | 1,014 MHz | 279 ms | 6,373 mm² | 531 mm² |
| 12 | 32 × 128 | 16,384 | 1,347 | 1,031 MHz | 227 ms | 6,459 mm² | 538 mm² |
| 12 | 64 × 128 | 16,384 | 1,507 | 1,023 MHz | 219 ms | 6,779 mm² | 565 mm² |
| 12 | 64 × 64 | 32,768 | 1,612 | 1,007 MHz | 177 ms | 7,520 mm² | 627 mm² |
| 12 | 64 × 128 | 32,768 | 2,214 | 1,023 MHz | 141 ms | 7,684 mm² | 640 mm² |
| 12 | 128 × 128 | 32,768 | 2,314 | 1,016 MHz | 144 ms | 8,847 mm² | 737 mm² |
| 12 | 64 × 128 | 65,536 | 2,744 | 1,023 MHz | 110 ms | 9,538 mm² | 795 mm² |
| 12 | 128 × 128 | 65,536 | **3,236** | 1,016 MHz | 100 ms | 10,143 mm² | 845 mm² |

So the circuit as generated today reaches **about 3,240 tok/s within the reticle** on 12 dies of 128 × 128 units in groups of four, which is also the point with the least area for each token per second (3.1 mm² per tok/s). Sixteen dies are no longer on the frontier: twelve are now both faster and smaller. Of each fastest die's 845 mm², 555 are cell memories, 190 vector memories, 100 logic and 0.1 microcode. The mixed split needs a quarter of the cells per die for the same speed, because each chip of a group runs a quarter of each layer; its first token is slower than with layers, because the prompt crosses the groups one after the other.

Most of the gain over the first mixed frontier (1,468 tok/s on 13,578 mm²) came from the microprogram and the links (November 2026):
- each chip of a group runs SiLU, sigmoid and element-wise products on its own slice, and DeltaNet's projection, convolution and SiLU on its own heads' channels, so a vector is gathered only where something reads it whole;
- products that cannot all be shared (DeltaNet's 48-row `in_proj_a` and `in_proj_b`) take a pass of their own, so the large projections in the same pass are shared too, which also cuts each cell's memory by a quarter;
- the vector units narrow a slice each of a product shared by rows;
- exchanges between chips go as streams, 16 blocks a chunk, and a chip passes each block on in the cycle it arrives; the dies hold their read addresses while a link waits, so a stream loses nothing;
- chips agree on a largest value by passing every value round at once;
- long single steps share cycles with their neighbours, short loops are laid out turn after turn, and no normalisation is spent just before a table that aligns its input.

A later round cut the single-cycle bookkeeping, from 370,627 to 334,195 cycles a token on 16 dies of 64 × 128 units:
- a narrowed product takes its normalisation from the shift the chips already agreed on, so it needs no second scan and no second round of the ring;
- the constants of the exponent arithmetic come in the image instead of being written before each use;
- the next pass of the cells starts feeding while the last one settles, its totals loaded and cleared among the new pass's words;
- the delta rule shared among units updates its state in place and reads its rows of the key and query where they are;
- streams clear their counters in the word before, and attention does not normalise exponentials that are normalised already.

A third round, measured on that fastest shape (12 dies of 128 × 128 units, 65,536 cells), took a token from 327,511 to 318,637 cycles with the same clock and area:
- a stream starts reading in the last words of the one before, which only write;
- a step can scan what it writes, a cycle later from the flops that keep each lane's write, so a normalisation that follows needs no scan of its own. Scanning in the same cycle cost the clock an eighth, which is why it waits a cycle.

A fourth round added two control bits, each a parallel path and a multiplexer, and took the same shape from 318,637 to 313,906 cycles a token with the same clock and 9 mm² more over the 12 dies:
- the wide output of a narrowing shifts left as well as right, so the normalisation after narrowing a product rides in the same pass;
- a left shift can take the A operand rounded right, so aligning a vector to an exponent takes one pass instead of two.

With 64 × 128 units a token went from 687,745 cycles on 65,536 cells to 334,195 on 131,072. The rest of the gap to the estimator's 3,750 tok/s is mostly attention at full context (about a quarter of the cycles: six heads use 48 of the 64 units, and each head's softmax is repeated in the eight units that share it) and the delta rule's short steps.

What each unit's vector memory holds on the busiest chip of 16 with 64 × 64 units and 32,768 cells (`memory` example):

| | Before | Now |
|---|---|---|
| Keys | 2,097,152 words | 360,448 |
| Values | 2,097,152 | 360,448 |
| Rotation table | 794,624 | 128 |
| Working registers, constants and recurrent state | 582,080 | 582,144 |
| **Total** | **5,571,008** | **1,303,168** |

Two hardware changes made the difference (November 2026):
- **The query heads of a key head keep its cache once between them.** The units that run them form a group: each keeps a share of the key head's columns, and all read each share from its unit in turn, through a selector on the vector unit's second operand. Before, each of the six query heads of a group kept a whole copy.
- **The rotation table stays in the image.** Before each routine that rotates, the firmware brings the row of the token's position, so each unit holds one row instead of one per position of the context.

Before those, three fixes to the compact vector memory had already brought the busiest chip from about 33 M words per unit, when no die fitted, to 5.6 M: each unit keeps only its own heads' columns of the KV cache, a unit with one DeltaNet head keeps its state in one register rather than one per head, and each chip loads only its own layers' norms, convolution taps and decay rates. On dies with fewer units than heads each unit still keeps a copy of its heads' cache for every round of heads, which is why the frontier's dies have a unit for every head.

## Quality of the chip's arithmetic

All measured on the real weights, against floating point.

### The exact integer machine

The most likely token, chip arithmetic against floating point, over a generated text (`exact` example; `tests/golden/exact.md`):

| Model | Vector bits | Same top-1 token |
|---|---|---|
| Qwen3.8-27B | 16 | 29 of 30 |
| Qwen3.8-27B | 14 | 11 of 12 |
| Qwen3.6-35B-A3B (MoE) | 16 | 11 of 12 |
| Qwen3.6-35B-A3B (MoE) | 18 | 11 of 12 |

The gates match the exact machine bit for bit on tiny models at 12, 16, 18 and 20 bits, so these are the chip's own outputs.

### How many bits the activations, the state and the tables need

Qwen3.8-27B with teacher forcing (`numeric` example; `tests/golden/numeric_*.txt`). Block floating point, with one exponent per block of lanes:

| Activations | DeltaNet state | Function tables | Top-1 agreement | Mean KL |
|---|---|---|---|---|
| 20 bits | 32 bits | — | 100 % | 0.0000 |
| 16 bits | 16 bits | — | 100 % | 0.0008 |
| 16 bits | 12 bits | — | 100 % | 0.0017 |
| 14 bits | 32 bits | — | 100 % | 0.0034 |
| 14 bits | 14 bits | 10 bits | 100 % | 0.0060 |
| 12 bits | 32 bits | — | 97.5 % | 0.0146 |
| 16 bits | 16 bits | 8 bits | 90.0 % | 0.0411 |
| 10 bits | 32 bits | — | 87.5 % | 0.1880 |
| 8 bits | 32 bits | — | 0 % | 12.47 |

Plain fixed point fails: with Q24.12 activations and a Q32.20 state, top-1 agreement is 90 %, and with 16-bit fixed point it falls to 20 % with hundreds of saturations. That result is why the die uses block floating point. DeltaNet's recurrent state, the main numeric risk in the original plan, holds at 16 and even 12 bits.

### Weight formats

Each tensor requantized to the format and the rest run in floating point, 64 generated tokens (`formats` example; `tests/golden/format_quality.md`):

| Format | Top-1 | Mean KL |
|---|---|---|
| fp8 | 95.3 % | 0.0052 |
| mlx4 (4-bit affine, the checkpoint's own) | 93.8 % | 0.0053 |
| posit8 | 93.8 % | 0.0099 |
| q8 | 93.8 % | 0.0265 |
| nvfp4 | 90.6 % | 0.0510 |
| mxfp4 | 93.8 % | 0.1073 |
| q6 | 65.6 % | 0.6837 |
| book8 (codebook) | 73.4 % | 0.8838 |
| book4 (codebook) | 57.8 % | 1.3339 |

The checkpoint is already quantized to 4 bits, so narrower formats start from an existing error and the 4-bit floating-point formats cannot be fully judged. Formats with a scale per group (mlx4, nvfp4, mxfp4) keep 4-bit quality at a quarter of the memory of 16 bits; plain integers without groups need 7 or 8 bits.

### Speculative decoding with the MTP head

How often Qwen3.8-27B's own MTP head guesses the following tokens, in floating point (`mtp` example; `tests/golden/mtp_acceptance.txt`):

| Depth | Accepted |
|---|---|
| 1 | 90.0 % (54 of 60) |
| 2 | 74.6 % (44 of 59) |
| 3 | 67.2 % (39 of 58) |

MTPLX reports 93.8, 85.5 and 74.3 % for the same checkpoint; the estimator uses those rates for the speed with speculation.

## Images

| What | Result | Command |
|---|---|---|
| Qwen3.8-27B vision encoder (460 M parameters, 27 blocks) in the die's arithmetic | 1.5 % from floating point at 16 bits; the whole model in chip arithmetic recognises a red disc and a square | `see_exact`, `see ... chip` |
| Qwen3.8-27B with images, floating point | Names the colour of drawn shapes: "Red", "Blue", "Green" | `see` |
| Janus-Pro-1B drawing "a red apple on a wooden table" | Floating point 387 s, chip arithmetic 543 s; both draw a red apple on wood | `draw`, `EXACT=16` |
| Show-o transformer, one pass, 8-bit affine cells on the addressed die | 61 of 64 masked places pick the same best code as floating point, about 46 s per pass (210 s with 16-bit mantissas) | `showo_addressed ... 4096 pass` |
| Show-o by weight format | Mean KL 0.001 nats with 8-bit affine weights, 0.066 with 4 bits | `showo_formats` |
| Show-o retouch as on the chip (integer MaskGIT, guidance on the die, 8-bit cells) | 4 guided steps in 360 s and 4.3 GB | `showo_fill_chip` |
| MAGVIT-v2 decoder in chip arithmetic | 49.6 dB PSNR from floating point (195 s) | |
| MAGVIT-v2 encoder in chip arithmetic | 99.3 % of the bits equal (254 s) | |
| MAGVIT-v2 round trip in floating point | About 30 dB PSNR | `inpaint ... reconstruct` |

Times are simulation times on the Mac, not chip times.

**Integer sampling against floating point.** Retouching a red disc into "a yellow lemon" (guidance 1.75, 4 steps), measuring how yellow the square comes out (`min(R, G) − B`):

| Way | Seeds | Yellow (mean ± error) |
|---|---|---|
| Floating point | 24 | 95.1 ± 4.1 |
| Firmware's integer MaskGIT, model in floating point | 24 | 86.9 ± 2.3 |
| The chip: integer MaskGIT, transformer on the die, 8-bit cells | 4 | 84.5 ± 5.6 |

The chip follows the integer sampler seed by seed. The gap to floating point is within the spread between seeds, and three other texts ("a green apple", "an orange", "a blue ball") show no systematic difference ([vision.md](vision.md#quality-of-the-integer-fill)).

## Gate-level runs

Bit and cycle exact against the instruction level and the exact machine ([simulation.md](simulation.md)):

| Run | Cycles |
|---|---|
| Toy Show-o filling an image through the whole chip (CPU, firmware, die) | 12.45 M |
| Toy Show-o transformer on the die | 3.1 M |
| Toy MAGVIT-v2 decoder, 664 routines | 2.65 M |
| Toy Show-o fill through the firmware at instruction level | 23.8 M |
| Whole chip in Icarus Verilog with its image | 1,296,777 |
| stories260K through the whole chip | Same text (", there was") and cycles as the instruction level |

The event-driven gate simulator runs the full suite of whole chips in gates in about 5 minutes (it was 30 with a plain cycle simulator).

## FPGA

The smallest complete chip (CPU, die, sampling unit), synthesized with Yosys (`fpga` example, `tools/synth/families.sh`; [fpga.md](fpga.md)):

| Family | Logic | Flip-flops | Block memory | Multipliers |
|---|---|---|---|---|
| Xilinx 7 series | 11,166 LUT6 | 5,443 | 42 BRAM36 + 14 BRAM18 | 42 DSP48 |
| Gowin | 17,961 LUT + 2,214 ALU | 5,519 | 125 BSRAM | 60 |
| Lattice ECP5 | 17,872 LUT4 | 5,527 | 125 DP16KD | 59 |
| Cyclone IV E | 53,603 LE | 9,017 | — | — |

Placed and routed with nextpnr on an ECP5-85F: **35.4 MHz**, up from 15.4 MHz through the changes listed in fpga.md.

stories260K (a 260,000-parameter Llama that writes short stories), cycles measured on its microprogram at the end of a 64-token context:

| Cells × lanes | Cycles per token | tok/s at 25-50 MHz | Memory |
|---|---|---|---|
| 8 × 1 | 211,500 | 120-240 | 4.63 Mbit |
| 8 × 4 | 104,800 | 240-480 | 5.62 Mbit |
| 16 × 4 | 87,500 | 290-570 | 5.62 Mbit |
| 32 × 8 | 52,700 | 470-950 | 6.47 Mbit |

## Physical design

The compute cell through OpenROAD on open PDKs ([asic.md](asic.md)):

| PDK | Area | Clock | Power | Violations |
|---|---|---|---|---|
| SkyWater 130 | 8,911 µm² | 119 MHz | 1.19 mW at 100 MHz | 0 |
| GF180 | 41,614 µm² | 125 MHz | 26.6 mW at 67 MHz | 0 |

A tile with 4,096 weights in a real 32 Kbit OpenRAM macro on SkyWater 130: 0.722 mm² of macro, as estimated, at 211 MHz. Fifteen layouts (five cells on three processes) calibrate the estimator to within 12 % on clock and 62-112 % on area (`tests/golden/asic.json`).

Factory test coverage with the generated scan chains and ATPG patterns: cell 100 %, CPU 99.4 %, sampling unit 97.7 %.

## Multipliers

Each multiplier and format is generated, checked against exact products and synthesized with Yosys and ABC (`tests/golden/synthesis.md`). For 8-bit fixed point:

| Multiplier | Area after ABC | Path in full adders |
|---|---|---|
| Array | 117.8 µm² | 25.9 |
| Wallace | 131.4 µm² | 16.0 |
| Booth | 124.8 µm² | 21.0 |

The generator builds every option for each design and keeps the smallest that meets the clock.
