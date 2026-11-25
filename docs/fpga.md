# Testing the chip on an FPGA

The goal is to validate the project on real hardware: the same circuit that is simulated gate by gate, with a tiny model and at low speed, generating text on an affordable board.

## The chip that goes to the FPGA

`cargo run --release -p sim --example fpga -- chip.v --dsp --factor-steps 2` writes in Verilog the whole chip for the test model (a toy Qwen3.5: 2 layers, one DeltaNet and one attention, width 8, a vocabulary of 8 letters):

- A RISC-V RV32IM CPU with its firmware, which boots, loads the image checking its CRC, tokenizes, drives the die and samples.
- A die with 8 cells, a one-lane vector unit, and only the logic its microprogram uses.
- The sampling unit.

By default it is the chip for **a single board**: no JTAG port, no factory memory self-test and no command chain towards other chips, which a single board does not use. It keeps the same pins, tied to zero, and the firmware finds itself alone and carries on unchanged. `--chain` generates the complete chip, the one that is repeated when several boards are joined. The test `a_chip_alone_on_one_board_says_the_same_without_tests_or_chain` runs the single-board chip in gates, and `the_chip_sized_for_an_fpga_says_the_same_in_gates` does the same with the complete one: both write the same as the instruction-level simulator, in the same cycles.

| Resource | Amount |
|---|---|
| Logic | ≈ 358,000 NAND2 equivalents (with scale and bias in two cycles, `--factor-steps 2`; about 494,000 in one) |
| Memory | 1.81 Mbit: CPU 0.66 (48 KB of program and 32 KB of data); function tables 0.26; microcode 0.5 (with a dictionary; 1.3 without); vector bank 0.07, with the register table giving each register only what it uses (2.1 without it) |
| Clock | The critical path is about 24 full-adder delays: on a low-end FPGA, between 20 and 50 MHz |

## Synthesis measured with Yosys

With `--dsp`, each single-cycle multiplier is written as a signed product (`$signed(weight) * $signed(x)`) and the tool puts it in its multiplier blocks. It is the same arithmetic: the tests compare each kind of generated multiplier with the exact product, and Yosys formally proves the cells' one equivalent (`equiv_make` + `equiv_induct`). The rest of the chip stays gate by gate.

`tools/synth/families.sh` synthesizes the chip for each family, each run with a time limit, and the slow test `the_catalog_knows_how_much_logic_the_smallest_chip_takes` does the same and checks the board catalogue stays within 10 % of the result. With `--dsp --factor-steps 2`, the single-board chip:

| Family | LUT | Flip-flops | Block memory | Multipliers |
|---|---|---|---|---|
| Xilinx 7 series | 11,166 LUT6 + 403 of distributed memory | 5,443 | 42 BRAM36 + 14 BRAM18 | 42 DSP48 |
| Gowin | 17,961 LUT + 2,214 ALU | 5,519 | 125 BSRAM | 42 MULT18×18 + 18 MULT9×9 |
| Lattice ECP5 | 17,872 LUT4, with distributed memory | 5,527 | 125 DP16KD | 59 MULT18×18 |
| Altera Cyclone IV E (memories apart) | 53,603 LE | 9,017 | — | — |

Before trimming the register table, removing what a single board does not use and fitting the function tables and the CPU's memory, it took 23,500 LUT6 on Xilinx, 24,400 LUT4 on Gowin and 28,610 LUT4 on ECP5.

The register table says where each register starts in the vector memory and saves about 120 BSRAM. In gates it is 512 entries of 16 bits in flip-flops, read three times per cycle: 39,700 gates and 8,192 flip-flops. In the Verilog for FPGA it is written as an asynchronous-read memory, which the tools map to their distributed memory, one copy per read: the chip's logic halves (from about 46,000 to 24,000 LUT4 on ECP5) and 8,000 flip-flops disappear. A test compares that memory, in iverilog, with the gates.

Without `--dsp`, Xilinx needs about 110,600 LUT6 and no DSP: the multipliers turn into gates.

Yosys' `synth_gowin` maps to the GW1N family; the Tang Mega's GW5A has blocks of the same size, but its official tool may pack them somewhat differently. It is a solid estimate, not a place-and-route report.

## Boards

The project assistant ("+" in the toolbar, "An FPGA") has these boards and says, for each model, whether it fits in the chip's memory, whether its weights have to go to the board's memory, or whether it does not fit. Prices are rough, as of November 2026; they vary by shop and region.

| Board | FPGA | Logic | On-chip memory | DSP | On-board memory | Price |
|---|---|---|---|---|---|---|
| Cyclone IV EP4CE10 board | Altera Cyclone IV E | 10.3 K LE | 0.41 Mbit | 23 | 8 MB SDRAM (the OMDAZZ, with a 16 Mbit EPCS16); others have 32 MB | ~$25-40 |
| Sipeed Tang Primer 25K | Gowin GW5A-25 | 23 K LUT4 | 1 Mbit | 28 | — | ~$35 |
| Sipeed Tang Mega 60K Dock | Gowin GW5AT-60 | 60 K LUT4 | 2.1 Mbit | 118 | 256 MB DDR3 | ~$90 |
| Sipeed Tang Mega 138K Dock | Gowin GW5AST-138 | 138 K LUT4 | 6.1 Mbit | 298 | DDR3 | ~$130 |
| Radiona ULX3S 85F | Lattice ECP5-85F | 84 K LUT4 | 3.7 Mbit | 156 | 32 MB SDRAM | ~$130-150 |
| Terasic Atum A3 Nano | Altera Agilex 3 | 135 K LE | 8.3 Mbit | 368 | 64 MB SDRAM | $180-245 |
| Sipeed Tang Mega 138K Pro | Gowin GW5AST-138 | 138 K LUT4 | 6.1 Mbit | 298 | 1 GB DDR3 | ~$195 |
| TUL PYNQ-Z2 | Zynq XC7Z020 | 53 K LUT6 | 4.9 Mbit | 220 | 512 MB DDR3 | $200-260 |
| Terasic DE10-Nano | Cyclone V SE | 110 K LE | 5.6 Mbit | 112 | 1 GB DDR3 | ~$225 |
| **AMD Kria KV260** | Zynq UltraScale+ K26 | 117 K LUT6 | **23.1 Mbit** (BRAM + UltraRAM) | 1,248 | **4 GB DDR4, 64-bit** | ~$249 |
| Digilent Arty A7-100T | XC7A100T | 63 K LUT6 | 4.9 Mbit | 240 | 256 MB DDR3 | ~$250 |
| Digilent Zybo Z7-20 | Zynq XC7Z020 | 53 K LUT6 | 4.9 Mbit | 220 | 1 GB DDR3 | ~$300 |
| AMD Kria KR260 | Zynq UltraScale+ K26 | 117 K LUT6 | 23.1 Mbit | 1,248 | 4 GB DDR4 | ~$349 |
| ALINX AX7202 | XC7A200T | 134 K LUT6 | 13.1 Mbit | 740 | 1 GB DDR3 | $400-500 |
| Digilent Nexys Video | XC7A200T | 134 K LUT6 | 13.1 Mbit | 740 | 512 MB DDR3 | ~€500 |
| Digilent Genesys 2 | Kintex-7 XC7K325T | 204 K LUT6 | 16 Mbit | 840 | 1 GB DDR3 | ~$1,000 |
| AMD Alveo U250 | UltraScale+ XCU250 | 1,728 K LUT6 | 432 Mbit | 12,288 | 64 GB DDR4 | ~$9,000 |

The Cyclone IV EP4CE10 is the smallest in the catalogue: the smallest chip the project generates needs about 54,000 logic elements (measured with Yosys, `synth_intel`, memories not counted) and 1.68 Mbit of internal memory, against its 10,300 and 0.41 (the Cyclone IV has no distributed memory, so its register table stays in flip-flops, and the microcode alone takes 0.5 Mbit), even with the weights in the board's SDRAM. Its SDRAM is not usable for the weights today either: the chip keeps them in each cell's memory and has no path to read them from outside.

Voice models do not fit either, in this board or in the largest Cyclone IV: Moonshine-tiny's 27 million weights need about 108 Mbit even in 4-bit codes, against the EP4CE115's 3.9 Mbit (see [voice](voice.md#the-demo)).

What does fit in it is one of the chip's compute cells behind the serial port: `cargo run -p netlist --example board -- DIR` writes the project, with its testbench, the Quartus project and a host script that loads a matrix, multiplies it by a vector and compares each row with the exact computation. It takes about 790 LE and 280 flip-flops (10 % of the EP4CE10) and 32 Kbit of memory. The test `the_cell_behind_the_serial_port_multiplies_a_matrix_exactly` runs it in iverilog. The project's pins are the usual ones for these boards and are marked as unverified: check them against the board's schematic.

### Recommendation for about $200: Kria KV260

For about $50 more than the Tang Mega 138K Pro, the **KV260** has almost four times more on-chip memory (23.1 Mbit against 6.1), four times more multipliers and 4 GB of 64-bit DDR4, several times faster than the cheap boards' DDR3.

- **stories260K fits entirely inside the chip** in every variant measured (5.4-9.0 Mbit), with room for more lanes and cells. On the Tang Mega only the smallest variant fits, the one-lane one.
- The chip's basic logic takes about 39 % of its LUTs.
- With the weights in the DDR4, models of hundreds of millions of parameters fit, such as stories110M or SmolLM2-135M, at whatever speed the memory gives.
- It is programmed with Vivado. Before buying, check that the free edition of the version you use includes the K26.

If the $200 limit is strict, the best option is the **Tang Mega 138K Pro**: the logic fits with margin and, with the memory optimizations, one-lane stories260K fits in its internal memory (5.37 of 6.1 Mbit, pending how Gowin's tool packs the blocks). The **Atum A3 Nano** has more on-chip memory (8.3 Mbit), but its external memory is a 64 MB SDRAM, which is slow.

Sources: [Kria KV260](https://www.amd.com/en/products/system-on-modules/kria/k26/kv260-vision-starter-kit.html), [KV260 refresh in 2025](https://www.amd.com/en/blogs/2025/kria-kv260-vision-ai-starter-kit-refresh.html), [Tang Mega 138K Pro](https://en.wiki.sipeed.com/hardware/en/tang/tang-mega-138k/mega-138k-pro.html), [Tang Mega 138K Dock](https://www.cnx-software.com/2024/06/28/sipeed-tang-mega-138k-dock-is-a-lower-cost-gowin-gw5ast-fpga-risc-v-development-board/), [Atum A3 Nano](https://www.cnx-software.com/2025/05/30/atum-a3-nano-development-board-combines-agilex-3-fpga-and-usb-blaster-iii-programmer/), [PYNQ-Z2](https://www.dfrobot.com/product-1838.html), [Genesys 2](https://www.digikey.in/en/product-highlight/d/digilent/genesys-2-kintex-7-fpga-development-board), [ALINX AX7202](https://botland.store/fpga-boards-and-kits/29031-ax7202-fpga-development-board-amd-artix-7-xc7a200t-5904422389024.html).

## Expected speed

The die spends 7,762 cycles per generated token with the toy model, with scale and bias in two cycles (7,720 in one), measured on its microprogram at the last position of the context; the firmware adds its own. When the clocks in the table were measured it was 11,172: the microprogram has shortened since.

### Clock measured after place and route

With OSS CAD Suite (Yosys and nextpnr), the toy chip is placed and routed on a Lattice ECP5-85F (CABGA756 package, which has pins for all its ports):

| Version of the Verilog for FPGA | Maximum frequency | LUT4 | Memory blocks | Multipliers |
|---|---|---|---|---|
| Products in DSP, the rest in gates | 15.4 MHz (estimated after placement) | 45,391 | 190 DP16KD | 89 |
| Also, adders as sums (carry chains) and the microcode with a dictionary | 18.2 MHz | 44,748 | 142 DP16KD | 89 |
| Also, the sampler's multiplier in a DSP block | 20.4 MHz | 43,084 | 142 DP16KD | 91 |
| Also, scale and bias in two cycles (`--factor-steps 2`) | 22.1 MHz | 46,699 | — | 59 |
| Also, the CPU's commands reach the die from registers | 26.5 MHz | 48,501 | 142 DP16KD | 59 |
| Also, the register table in distributed memory and the vector address sums in carry chains | 32.1 MHz | 28,610 | 142 DP16KD | 59 |
| For a single board, with registers, function tables and CPU memory fitted | 30.7 MHz | 18,486 | 126 DP16KD | 59 |
| Also, the sampling unit's sums and comparisons in carry chains | **35.4 MHz** | 18,440 | 126 DP16KD | 59 |
| Also, registers shared between buffers that are never alive at once | 34.5 MHz | 17,872 | 125 DP16KD | 59 |
| Also, 48 KB of program memory and the serial port with flow control (`--uart 16`) | 34.2 MHz | 17,923 | 129 DP16KD | 59 |

The critical path was the sampling unit: a 16-row shift-and-add multiplier, in one cycle, feeding the address of the exponential table. Then it moved to the die's cells: reading the bias, the product by the scale (split across several chained DSP blocks) and the accumulator in the same cycle, 49 ns. With scale and bias in two cycles, the factor comes from a register and each cycle multiplies half of its bits: the die spends 11,172 cycles per token instead of 11,103 (0.6 % more) and the clock rises to 22.1 MHz, so designs for FPGA use it by default. Next was the bus from the CPU to the die: a CPU write was decoded, started the die and took an address to its vector memory in the same cycle. With registered commands, the clock rises to 26.5 MHz. Next was the vector memory address: the microcode word just read chose an entry of the register table out of 512, shifted it and added twice, gate by gate, in the same cycle, 38 ns with 27 in routing. With the table in distributed memory and the sums in carry chains, the clock rises to 32.1 MHz with half the logic. The critical path then moved to the sequencer's loop: the word just read compares its repeat count, and that chooses the address of the next one in the dictionary, 31 ns with 21 in routing. Computing the PC's sum earlier does not improve it (30.5 MHz, within placement noise). With the single-board chip it was the sampling unit again, 32.6 ns: the subtraction giving the logit difference went gate by gate and the tool turned it into a chain of 16 LUTs before the multiplier. With the sampling unit's sums as sums, in carry chains, the clock rises to 35.4 MHz, with the same cycles. Then the cell rules: scale times product in two chained DSP blocks, about 28 ns with 18 in routing. With the 48 KB program memory and the serial port it is the sequencer's loop again, 29.2 ns with 19.3 in routing: a microcode word read from block memory chooses the address of the next one. The new memory and the link cost four memory blocks and about 50 LUT4, and leave the clock within placement noise.

In the sequencer, measured in gates, the longest path to the address of the next word is not the repeat count but the index register: choosing it, adding the step and comparing it with the limit decide whether to jump. Shortening it without changing a single cycle needs reading the dictionary and the index through two ports at once, that is, duplicating those memories, which is just what small FPGAs lack; the other way is reading the word one cycle earlier with a one-cycle delay on jumps, which changes the compiler, the cycle model and every test that counts cycles. This is the slowest family in the catalogue; on a Xilinx 7 series or UltraScale+ the same circuit should run quite a bit faster.

At 34.2 MHz the die's cycles alone allow about 4,400 tok/s with the toy model, somewhat fewer with the firmware's, of no practical value beyond showing the chip works. Without `--factor-steps 2` the same chip takes 15,038 LUT4 and 91 multipliers but only reaches 25.4 MHz: reading the bias, the product by the scale and the sum fall in one cycle again.

### With a model that writes stories: stories260K

[stories260K](https://huggingface.co/karpathy/tinyllamas) is a Llama of 260,000 parameters (5 layers, width 64, a vocabulary of 512) that writes short stories in English. `tools/convert/llama2c.py` converts its `.bin` and its tokenizer to a Hugging Face folder, in float32 or, with `--affine 8`, in MLX's affine format (8 bits, scale and bias per group), which is the one the chip uses.

`cargo run --release -p sim --example fpga -- chip.v --dsp --model ~/.llm2chip/models/stories260K-q8 --context 64 --cells C --lanes L --memories` sizes the chip for that model, counts the microprogram's cycles at the last position of the context and breaks down the memory:

| Cells × lanes | Cycles per token | tok/s at 25-50 MHz | Memory |
|---|---|---|---|
| 8 × 1 | 211,500 | 120-240 | 4.63 Mbit |
| 8 × 4 | 104,800 | 240-480 | 5.62 Mbit |
| 16 × 4 | 87,500 | 290-570 | 5.62 Mbit |
| 32 × 8 | 52,700 | 470-950 | 6.47 Mbit |

The 8 × 1, 8 × 4 and 16 × 4 variants fit in the Tang Mega 138K's 6.1 Mbit; all of them fit in an XC7A200T (13.1 Mbit) and in the Kria KV260 (23.1 Mbit). With more lanes it is mostly the function table that grows, one per lane.

### Memory optimizations

Before them, the 8 × 1 variant needed about 10.2 Mbit. None changes a single cycle, and the tests in gates check the chip writes the same, in the same cycles, as the instruction-level simulator.

| Change | Saving on 8 × 1 |
|---|---|
| **The embedding stays in the image.** The table does not take vector memory: before each token, the firmware brings its row with the same loader used at boot and leaves it in a one-row register. The vector memory goes from 2^16 to 2^15 words. | ~1 Mbit |
| **A wide memory sized to fit.** The cells' sums only need as many entries as a pass or a sum has rows, not as many as a register: 2^9 entries instead of 2^15. | 1.7 Mbit |
| **Just the function tables needed.** The chip only carries the tables up to the highest function its microprogram uses: a Llama uses four of the eight (exponential, sigmoid, inverse and inverse square root). | 0.26 Mbit per lane |
| **CPU memory sized to fit.** The firmware's data, its stack and the chip's description fit in 32 KB; it was 64 KB. With `--tokenizer DIR/tokenizer.json` the CPU tokenizes on the chip and its tables are added (with stories260K, 128 KB in all). | 0.26 Mbit |
| **Microcode with a dictionary.** Of 9,813 words only 2,018 are distinct, because each layer repeats almost the same code. Each distinct word is stored once and, for each PC, an index says where the next word and its jump target are, so the dictionary is read in the same cycle as a plain memory and none is lost. | 2.1 Mbit |

`--table-inside` and `--no-dictionary` turn off the embedding and the dictionary optimizations, for comparison.

### Logic: registers sized to fit

stories260K used 464 registers, one per buffer, one per matrix and a fixed reserve of 320 working ones. Now it uses 118, without changing a single cycle:

- The working registers go at the end and the die only has those its routines reach.
- Two buffers that are never alive at the same time within a token share a register. Each lives from when it is written until its last read; chained products write all their outputs into the first, so their outputs live from there. The logits, the final state, whatever the MTP and image heads touch, and whatever is read before being written (what passes from one token to the next) get their own register.
- The cells' matrices only keep their exponent in the register: those with the same one share it.

The register table goes from 2^9 to 2^7 entries and the microcode's register fields narrow: about 37,000 NAND2 equivalents and 0.2 Mbit of microcode less in stories260K (from 558,000 to 521,000); in the test model, from 2^9 to 2^8. If a microprogram ever reached a register the die does not have, assembly fails with an error instead of generating a chip that computes wrongly. The exact engine, the one in the numerics lab, keeps one register per buffer so it can read them all at the end of a token.

All of this follows the microprogram, which changes with the model: the chip for FPGA is generated for the models it will run, as its operations, control fields and vector memory already were. An image for several chips is sized for what any of them uses, in any of the splits it holds.

### One chip for several models

The microcode, the dictionary, the register table, the function tables and the weights arrive with the image at boot, so the same chip can run several models if it is built for all of them: with `--model` repeated, the `fpga` example gathers the logic any of their microprograms uses and gives each memory what the largest needs. Each model then boots from its own image, with the same cycles as on a chip of its own. With stories260K in floating point and in 8 bits, the chip for both is the first one's (518,000 NAND2 and 6.73 Mbit), against 478,000 and 4.63 Mbit for the 8-bit one alone. The test `one_chip_runs_either_of_two_models_from_its_image` runs in gates two versions of the test model, with 4-bit and 2-bit codes, on the same chip.

## Steps for the test

1. Generate the Verilog with the `fpga` example and the image with the same die.
2. Replace the generic memories (`llm2chip_sram`, `llm2chip_ram2p`, `llm2chip_rom`) with the FPGA's blocks; Yosys already infers them.
3. Generate the chip with a serial port (`--uart CLOCKS_A_BIT`: the chip's clock over the baud rate, rounded, for instance 32 MHz / 2 Mbaud = 16) and connect `uart_rx`, `uart_tx` and `uart_cts_n` (to the adapter's CTS) to the board's USB serial port, choosing a divider within the tolerance in [host-links.md](host-links.md#the-serial-port-uart). Connect the image request to the board's flash. A quad SPI link (`--qspi`) also works, with a microcontroller as host, and so does I2C (`--i2c 42`, the address in hexadecimal), with the data pin as a tristate that only drives low, a pull-up, and `chip_host.py --i2c` on a Raspberry Pi.
4. Send a question with `tools/host/chip_host.py PORT ask "Once upon a time"` and compare the answer with the simulation's: the messages must match byte for byte.

Sources for prices and specifications: [Tang Mega 138K Pro](https://en.wiki.sipeed.com/hardware/en/tang/tang-mega-138k/mega-138k-pro.html), [CNX on the Tang Mega](https://cnx-software.com/2023/10/01/sipeed-tang-mega-138k-pro-dock-features-gowin-gw5ast-fpga-risc-v-soc), [Tang Primer 25K](https://cnx-software.com/?p=123307), [ALINX AX7202](https://boards.fpgadeveloper.com/boards/AX7202), [Nexys Video](https://www.digikey.co.uk/en/products/detail/digilent-inc/410-316/5456481), [Arty A7](https://pcbsync.com/xilinx-fpga-buying-guide/). Prices vary by shop and region: they are only a guide.
