# The whole chip: design and simulation

Goal: the design and the simulation include everything the real chip would have. The only difference between simulating and running in silicon is how data gets in and out through the host port.

## The boundary between the chip and the world

The real chip would have a PCIe or USB PHY. In simulation, that PHY is replaced by a simple digital port. Everything on this side of the port is the same in both cases.

| Signal | Direction | Use |
|---|---|---|
| `host_rx_*` | Input, byte stream | Commands and UTF-8 text |
| `host_tx_*` | Output, byte stream | UTF-8 text and status |
| `image_req_*`, `image_data_*` | Requests for image blocks | Reading the model from the host |

The chip receives bytes of text and returns bytes of text. Tokenizing, inference, sampling and detokenizing happen inside.

## Blocks

| Block | Role | Today |
|---|---|---|
| MAC cells + weight SRAM | Matrix-vector product | In gates |
| Activation network | Distribution and summing | In gates |
| Vector lanes | Attention, delta rule, MLP gates | In gates, with banked registers and one ALU per lane |
| Function tables | SiLU, sigmoid, exp, root | In gates |
| Vector registers and micro-sequencer | Control of each stage, loops and parameters | In gates, with a fixed microprogram |
| Die-to-die link | Passing activations, with error correction, retries of damaged words and clock crossing | In gates |
| KV SRAM and DeltaNet state | The model's memory | Functional model and area |
| Internal system bus | Joins the CPU, memories and peripherals | In gates |
| Host port and FIFO | Bytes in and out | In gates |
| RV32IM control CPU | Firmware | In gates and ISS, cycle by cycle |
| Instruction and data SRAM | Firmware and vocabulary | In gates |
| Load controller and DMA | Image into cells, tables and microcode, with its own boot and CRC | In gates and ISS, connected to the die |
| Weight write network | Takes each word to its cell | Direct load ports only |
| ECC, CRC, BIST and repair | Reliability and defective cells | Per-lane ECC in the cell and CRC on loading, in gates; march self-test on every read-write memory, reachable through JTAG; spare rows verified as a block |
| Sampling unit and RNG | Top-k and temperature on the logits, with a multi-lane filter | In gates and as a model, connected to the die |
| Firmware tokenizer | Follows the rules of each model's `tokenizer.json` through tables: added tokens, normalization, regular expressions, BPE and special-token template | Compared with the `tokenizers` library on synthetic styles and real models; in the ISS with real tables and in gates with tiny ones. Its working memory is sized from the design's context, and the estimator counts it |
| Token scheduler and MTP | Decides what each die runs, and undoes | In firmware, checked against the software loop |
| Clock, reset and power management | Reset sequence, clock gates | Reset in gates; clock and power are only estimated, because the simulator has a single clock |
| Debug and counters | Counters, read port and standard JTAG | In gates |

Sampling on the CPU does not work for fast designs. Reading 250,000 logits takes about 250 µs at 1 GHz, and at 15,000 tok/s a token lasts 66 µs. That is why sampling has its own unit and the CPU only receives the result.

Analog blocks (PLL, PHY, sensors) only get a behavioural model, as they are not digital logic. The PHY stays outside the boundary; the PLL and sensors are modelled as sources of clock and values.

## Simulation levels per block

Each block is configured separately from the simulation, to choose between slow, realistic runs and fast, focused ones.

| Block | Skip | Functional | Instruction or transaction | Gates |
|---|---|---|---|---|
| Boot and BIST | Yes | | ISS | CPU in gates |
| Weight loading | Weights straight from the checkpoint | | Real controller, image, ECC and cycles | A sample, or a tiny model |
| Tokenizer | | In Rust, outside the chip | Firmware in the ISS | Firmware in gates, with tiny tables |
| Inference core | | Floating point (CPU or GPU) | Exact integers (CPU or GPU) | Tiny model |
| Sampling | | In Rust, outside the chip | Functional unit | Gates |
| Detokenizer | | In Rust, outside the chip | Firmware in the ISS | Firmware in gates |

Starting profiles:

- **Fast:** everything functional, floating point and GPU. For trying models and formats.
- **Real arithmetic:** exact integers and GPU, the rest functional.
- **Real flow:** boot, loading, firmware and sampling at their realistic levels, and exact integers.
- **Full gates:** everything in gates with a tiny model, from input bytes to output bytes.

Every cycle the application shows says whether it was simulated or estimated, depending on what was skipped.

## The whole chip in gates

`chip_design` joins the CPU, the memories, the load controller, the die and the sampling unit. With a tiny model, the real firmware receives bytes, loads the image, runs the tokens and returns bytes. The result and the cycles match the instruction level (CPU in the ISS and die on the exact machine): 525,668 cycles in both.

## The cost of sampling

Reading Qwen3.8's 248,000 logits one by one cost almost as much as inferring the token. The unit now compares several logits per cycle with the worst candidate on its list, and only those that beat it get in. That is about 500 insertions per token, so almost no group stalls.

With Qwen3.8 in the real flow and 64 lanes, the cycles outside the die drop from 1,502,415 to 37,162 for 17 prompt tokens and 6 generated ones, with the same text. The estimator also assumes the logits are filtered as they come out of `lm_head`, and counts 787 cycles per token. The reduced die delivers only one logit per cycle, so the whole chip in gates uses one lane; several lanes are verified on the unit separately.

## Custom circuit

With "Custom circuit", each combination of parameters generates its own cell. The generator builds every variant in gates (array, Wallace, Booth and precomputed multiples, with their stages and with or without zero gating), measures its area, switching and critical path, and keeps the best one that meets the requested clock. The different widths of a mixed model get different cells. With "Target speed", the multipliers per die follow from that speed and the size of the model.

Precomputed multiples lose in the cases measured: they remove the multiplier, but multiply by 5 the bits that must be sent to each cell. Skipping zeros does not pay either with Qwen's real zeros, 10.6 % at 4 bits and 1.6 % at 8.

## Several chips

All the chips come from the same netlist and read the same image. Each has a CPU, memories, a load controller, a die and sampling. A single pin, `follower`, decides on the board which one leads: it talks to the host, tokenizes and samples.

At boot, the leader sends a counter along the control chain. Each chip takes its number, increments it and passes it on; when it gets back to the leader, that value is the total, which goes round once more. With the total and its own number, each chip looks up its microprogram and weights in the image. If the image does not cover that number of chips, the leader reports it to the host. A chip whose chain has no neighbour works alone.

| Pins | Role |
|---|---|
| `link_tx_*`, `link_rx_*` | Data between dies, in a ring: activations and, from the last to the first, the logits |
| `loop_tx_*`, `loop_rx_*` | In the mixed split, a link back from the last chip of each group to the first, closing the group's ring |
| `chain_tx_*`, `chain_rx_*` | Counting at boot and the leader's commands, in a ring |
| `image_req_*`, `image_resp_*` | Each chip asks for its part of the same image |
| `follower` | Low on the chip that leads |
| `reset`, `host_*`, `debug_*` | Reset, host (only used on the leader) and debug |

### Splits

- **By layers:** each chip runs some whole layers and passes the hidden vector to the next over `link`.
- **By rows:** every chip keeps a slice of the rows of each matrix, they compute each product at once and pass their slices round a ring over `link`, the last joined to the first.
- **Mixed:** groups of chips split by rows, joined in a chain by layers. Within a group the ring runs over `link` and `loop` closes it, from the group's last chip to its first; `link` also carries the hidden vector from the last chip of one group to the first of the next, which passes it to the others in its group. Each microcode word carries two bits, one for the port it sends on and one for the port it takes from, because one word can send for one instruction and take for another. The logits' matrices are not split: only the first chip computes them, from the final state the last one sends it. `two_groups_of_two_dies_run_a_token_over_their_links` runs a token on four dies in gates joined this way, and checks the first one's logits are bit for bit those of a single die.

An image carries a plan for each number of chips: in the mixed split, numbers that do not form whole groups are split by layers, and those that fit in a single group, by rows.

Each word carries a sequence number and a code that corrects one bit and detects two, with a valid and a ready signal. The receiver corrects single errors and, on a double one, asks for the word again over the acknowledgement signals; the sender keeps unacknowledged words and repeats them, and if the request is lost, asks again after a while. The receiving half runs on the sender's clock and hands words to the chip's clock through a queue with Gray pointers, so the two chips do not need the same clock (`words_cross_between_two_unrelated_clocks`). On the simulated board the chips share a clock; the board tests inject single and double flipped bits, and require every single error to be logged as corrected and every double one as repeated. The image must carry the layer split for every number of chips it supports.

## Burnt-in weights

The "Weights" parameter decides whether the model can be changed. With ROM, each cell's weights, scales and biases are fixed at manufacturing: the cell loses its load ports and address selector, the die loses the network that writes into the cells, and boot no longer transfers them. The microprogram, tables and firmware are still loadable, so the chip can be corrected without new masks, but cannot change model.

The biggest saving does not come from the logic but from the density of ROM against SRAM. To synthesize it in a real technology, `llm2chip_rom` must be replaced by the foundry's ROM macro, with each cell's contents.

## Why not everything in gates with Qwen

Qwen3.8 has 27 billion weights, and simulating them in gates is not feasible. The strategy is the one that already works for the core:

1. With a tiny model, everything in gates: input bytes, loading, firmware, inference, sampling and output bytes. It is exported to Verilog and checked with Icarus.
2. Each fast level is compared bit for bit with the gates on that model.
3. With the real Qwen, the control logic can run in gates or in the ISS, and the inference core on the already verified exact machine. They talk through the same register interfaces as on the chip.

## Loading the model in deep simulation

Today the weights are written straight into the simulated memory. The central change is for the simulation to read from what the chip has loaded.

**The chip's image.** Firmware, microcode, tables and tokenizer make up the image the controller walks through. For the real model, a manifest in `~/.llm2chip/images` keeps the destination and CRC of each tensor, and the weights are read from the checkpoint so as not to duplicate its 19 GB. Writing an image of weights per cell in the chip's format is still to be done.

**Three levels for the same load:**

| Level | What happens | Reach |
|---|---|---|
| Direct | The checkpoint is opened, as now | Quick tests |
| Transaction | The functional controller walks the whole image, checks CRC and ECC, applies repair, writes each cell's memory and counts the cycles | The whole real model |
| Gates | The real controller and write network move the data through the load port | A whole tiny model, and samples of the real one |

**Samples with the real model.** In gates, a real block of weights, chosen by the user or at random, is loaded through the controller and write network, and the resulting words must match those of the transaction level bit for bit. The other cells are preloaded straight into the simulated memory, as an RTL testbench does.

**Fault injection.** The simulation will be able to flip bits and mark cells as defective, to check that ECC, CRC and repair work as they should.

## A fixed microprogram

The microprogram is loaded once from the image and does not depend on the token or the position. The sequencer has index registers that add an offset to each operand, and a conditional jump for loops. The CPU writes the parameters (embedding row, cache position, context tokens) and starts one of the routines: token, logits, and those of the MTP head.

RoPE's sines and cosines form a table with one row per position, 3·h+1 values each, the part of the chip's data that grows most with the context. On a chip whose embedding table stays in the image, the rotation table stays there too: before each routine that rotates, the firmware has the load controller bring the row of the token's position into the table's register, which holds that one row. Models whose image tokens take rotations written by the host (M-RoPE, Show-o) or that have two rotation tables keep the whole table in the die.

The program grows with the layers each chip holds and, on a die with fewer vector units than heads, with the rounds of heads: each round repeats a loop's body for its own heads. It no longer grows with the number of elements, and a die with a unit for every head runs each loop over heads once. For Qwen3.8-27B (`memory` example):

| Chips | Units | Words per chip | Microcode memory |
|---|---|---|---|
| 16 | 64 | 3,849 to 17,566 | 1.2 Mbit |
| 16 | 4 | 17,396 to 75,359 | 18 Mbit |
| 1 | 64 | 222,802 | 55 Mbit |

The memory is the fewer bits of the words one after another and of a dictionary of distinct words with an index, which only works up to 2^16 words. Even all of Qwen3.8 on one chip takes a millimetre or two in N6, far below what its vector memory takes, and the sizing now counts it. Folding the layers into one loop would make the program as small as one layer, which matters on an FPGA, whose block RAM is small; it needs every layer's registers and weights laid out the same way, and is still to be done.

## Phases

1. System bus, memory map and host port.
2. RV32IM core in gates and in the ISS, checked against each other.
3. Firmware in `no_std` Rust: boot, tokenizer, main loop.
4. The chip's image and load controller, with the three levels.
5. Sampling unit and RNG.
6. ECC, CRC, BIST and repair, with fault injection.
7. Clock, reset, power, debug and counters.
8. Integration: whole chip in the schematic, estimator from the netlist, simulation settings per block, console and boot timeline.
