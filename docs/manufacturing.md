# Towards manufacturing

What is verified about the chip, what rules the board must follow, and what is still missing to manufacture it.

## What is verified

| What | How | Test |
|---|---|---|
| The whole chip (CPU, firmware, loader, die, sampler) writes the same thing, in the same cycles, at instruction level and in gates | Toy models, text only and with an image | `crates/sim/tests/chip.rs`, `crates/sim/tests/vision_chip.rs` |
| An outside simulator, Icarus Verilog, gives the same as ours with the exported netlist | The whole chip with its firmware, gate by gate; with `LLM2CHIP_ICARUS=1` | `text_goes_through_the_whole_chip_in_icarus_verilog` (226,632 cycles, about 40 min), `the_whole_chip_reads_a_picture_in_icarus_verilog` (1,296,777 cycles, about 4 h) |
| A real model, stories260K, through the whole chip, at instruction level and in gates | The firmware tokenizes "Once upon a time" and writes ", there was", the same as the die's machine driven from the host and as floating point; the gates, in the same 1,909,169 cycles | `a_real_model_writes_the_same_text_through_the_whole_chip` (about 4 min), and the die alone in `a_real_model_s_token_on_the_die_matches_the_exact_machine` |
| Blocks, die and cells in Verilog against the gate simulator | Icarus Verilog, cycle by cycle | `crates/netlist/tests` |
| Synthesis and routing for FPGA | Yosys and nextpnr (ECP5); Gowin and Cyclone IV by synthesis | `docs/fpga.md` |
| Scan chain and factory test patterns | The ATPG patterns, applied in the gate simulator as a tester would, give what was predicted; with an injected fault, they fail if the ATPG said they would find it and do not fail if it said it was undetectable. The Verilog with scan gives the same in Icarus. The STIL file, read back, does the same. The chip with its chains works as the one without them | `crates/netlist/tests/scan.rs` |

### A bug that only showed up with a real model

stories260K was the first model whose microprogram goes past 2^13 words (9,813), and in gates its die computed zeros.

The cause was in how an immediate travels:
- **Where it goes:** a constant of the instruction travels in the jump target field, and the die extends its sign from the last bit of that field.
- **Why it broke:** the microprogram was assembled for a test die with 13-bit targets and loaded into the final die, with 14. A −15 stored in 13 bits (0x1FF1) was read in 14 as +8,177.
- **What exposed it:** each head's RoPE subtracts 15 from the exponent, so the keys' exponent came out 8,192 higher and the cache stored them as zeros.

Now the immediate is stored with its full sign and each die takes the bits its field has (`a_negative_immediate_keeps_its_sign_on_a_die_with_wider_jump_targets`).

Along the way another bug came up, this one in the instruction-level simulator: with the embedding table in the image, it always read token 0's row.

## Rules for the board

- **A single clock.** The host port's byte port, the image port and the chip-to-chip links are synchronous: valid and ready are read on the same edge of the chip's clock. Whatever sits on the other side must use that clock, or a copy in phase, for instance an FPGA bridge or a microcontroller fed from the board's clock output. A device with its own clock needs a dual-clock FIFO outside the chip.
- **Reset.** The pin can be released at any time: it goes through two flip-flops before use, and then the chip stays in reset for 15 more cycles (`RESET_RELEASE` = 17 cycles from the pin). Wait for `booting` to go low before sending bytes to the host port.
- **The host port's serial links** have no clock in common with the chip. The serial port's line, the quad SPI's clock, select and data lines and I2C's clock each go through two flip-flops (I2C's data line through three), so a host on its own clock needs nothing more. The quad SPI clock may be up to an eighth of the chip's, and I2C's a twentieth. I2C's data pin must be open drain, pulled down by `i2c_sda_low` (see [full-chip.md](full-chip.md#the-host-ports-links)).
- **JTAG.** TCK, TMS and TDI each go through two flip-flops. The chip's clock must run at least 4 times faster than TCK, with each half period of TCK lasting 4 cycles or more.
- **`follower` pin.** It is static: set it on the board and do not change it while the chip runs.

## Factory tests: scan and ATPG

Every chip that comes off the wafer has to be tested: a defect leaves a wire stuck at 0 or at 1. That takes two pieces:
- **The scan chain** (`netlist::scan::insert_scan`): joins every flip-flop into a shift register, from `scan_in` to `scan_out`.
  - With `scan_enable`, each cycle shifts the chain one place: the tester loads any state and reads what the chip holds.
  - Without it, a clock cycle captures what the logic computes.
  - `test_mode` puts each memory behind a shadow: the logic reads flip-flops of the chain instead of the memory, and those flip-flops capture a mix of the address, data and write enable going into it. The memories' contents are tested by the march self-test.
  - With all three signals at 0, the circuit works just as it does without a chain (checked cycle by cycle), with one more multiplexer in front of each flip-flop.
- **The patterns** (`netlist::atpg::generate`): with the chain, testing the sequential circuit comes down to testing the logic between flip-flops.
  1. First, random patterns, 64 at a time, simulating every fault.
  2. For what is left, PODEM searches for a pattern fault by fault. It picks its objectives with SCOAP measures and drops early the paths along which the difference can no longer get out.
  3. If the search runs out, the fault cannot be detected: the logic at that point is redundant.

  Values have three states (0, 1 and unknown) for what comes out of the memories, which the test does not control.

The fault list is the classic one: each signal stuck at its source and each gate input where the signal fans out. Faults on the shift path are found by the chain test, which passes a `0011…` through it.

`cargo run --release -p netlist --example atpg` gives this table:

| Block | Flip-flops in the chain | Faults | Detected | By the chain | Undetectable | Aborted | Patterns | Fault coverage | Test coverage | Seconds |
|---|---|---|---|---|---|---|---|---|---|---|
| cell | 132 | 7134 | 5436 | 405 | 1293 | 0 | 113 + 7 | 81.88 % | 100.00 % | 0.0 |
| cpu | 236 | 24596 | 21874 | 711 | 1866 | 145 | 272 + 361 | 91.82 % | 99.36 % | 21.7 |
| sampler | 363 | 48226 | 32872 | 1090 | 13459 | 805 | 418 + 866 | 70.42 % | 97.68 % | 147.0 |

The blocks are the weight cell with 4-bit weights, 16-bit activations and 32 weights, the RV32 CPU, and the 16-bit logit sampler with 8 candidates. A whole small die (three cells, the sequencer, the vector unit and the tables) takes much longer with the current search, which simulates the whole circuit again at every decision: `cargo run --release -p netlist --example atpg -- die`.

- **Test coverage** is the usual measure: the faults found among those some pattern can find.
- **Fault coverage** also counts the undetectable ones, of which there are many here. They are not shortcomings of the test but redundant logic in the netlist, which is not optimized:
  - the multiplier's first row adds a zero;
  - many adders' carry out is unused;
  - the sampler's comparisons use a whole adder and only read its carry.

  Synthesis with Yosys flattens the circuit and removes that logic. Measured with the memories as black boxes (`synth -flatten`, `abc -g AND,OR,XOR,MUX`):

  | Block | Our cells | After Yosys | Difference | Undetectable faults |
  |---|---|---|---|---|
  | cell | 1,380 | 972 | −30 % | 22 % |
  | cpu | 4,151 | 3,459 | −17 % | 7 % |
  | sampler | 9,121 | 5,698 | −38 % | 25 % |

  The reduction follows the share of undetectable faults. Part of it is something else: Yosys folds the enable multiplexer into the flip-flop. Coverage on the synthesized netlist is still to be measured, which needs reading the Verilog that Yosys writes.

A scan test writes into the memories while shifting, as the march self-test does, so it runs before the image is loaded.

**The whole die** is not tested flattened into a single fault list: its cells are identical and are tested at once (`scan::insert_scan_parallel`).
- **Testable cell:** each cell becomes a version whose inputs, in test mode, come from a wrapper: one register per vector unit, shared by its cells, with as many flip-flops as the cell has input bits.
- **Parallel chains:** they all start from `scan_cells_in`, after the wrapper, so the same shift loads every cell alike.
  - `scan_cells_out` gives the first cell's chain.
  - `scan_cells_differ` rises if any other cell's differs from it.
- **The cell's output** is already a register (`total_reg`) inside its chain: it needs no output wrapper.
- **The patterns** are generated on a slice: the wrapper and the testable cell, chained as in the die.

On the test die (three cells):

| | |
|---|---|
| Main chain (the rest of the die) | 1,304 flip-flops |
| Each cell's chain | 367 flip-flops, 51 of them in the wrapper |
| Cell faults | 101,486, 98.0 % test coverage with 571 patterns |

Checked in the gate simulator, with the cell's patterns applied to all three at once:
- with no faults, the first cell gives what was predicted and none differs;
- with a fault in the third cell, the difference rises;
- with a fault in the first, its output is also not the predicted one.

With the chains off, the die does the same thing cycle by cycle.

**The cost:**
- **Per cell:** one multiplexer per input bit (51) and one per flip-flop in the chain. No new flip-flops in the cell, apart from its memories' shadows.
- **Per vector unit:** the wrapper's 51 flip-flops.
- **The test time of each pattern** is what the longest chain takes to shift: about 1,300 cycles here, however many cells there are. With a single serial chain, a die of 100,000 cells of about 300 flip-flops each would need about 30 million cycles per pattern.

**The whole chip** (`chip::chip_design_for_test`) carries the chains: the die's cells side by side as above, and one main chain through everything else, from the die's sequencer and vector units to the CPU, the memories' shadows and the sampler (`scan::insert_scan_around`). On the small test chip the main chain has 3,981 flip-flops and each cell's 299. With the test pins at zero the chip does what `chip::chip_design` does, cycle by cycle, and in test mode each chain carries a pattern from its input to its output (`crates/netlist/tests/scan.rs`). The FPGA builds go without it: an FPGA is tested by its maker and does not need it.

The test signals are pins of their own. The alternative that saves pins is driving the chain from the JTAG TAP with internal scan instructions.

### The tester's file: STIL

`netlist::stil::write` gives the test of a design with one chain in STIL (IEEE 1450), the format testers read. Each test cycle drives the inputs at its start, strobes the outputs at 40 ns and pulses `clk` from 50 to 80 ns. The file has the chain test (`0011…` shifted through and read back), then each pattern: its state shifted in while the previous one's result shifts out (`load_unload`), and one cycle that sets the input pins, reads the output pins and captures (`capture`).

`netlist::stil::read` takes that file back and expands it into test cycles, and `netlist::stil::run` applies them on the gate simulator as a tester would. The cell's file runs without a mismatch on the fault-free circuit, and fails with a fault the patterns find and with one on the shift path (`crates/netlist/tests/scan.rs`). `STIL=folder cargo run --release -p netlist --example atpg` writes each block's file.

## What manufacturing still needs

- **A PDK and its libraries:** standard cells instead of our generic gates, and SRAMs from the foundry's memory compiler, or open generators such as OpenRAM.
- **Placement, clock tree and routing,** timing analysis at every process corner, power and IR drop, and DRC and LVS checks down to GDSII. With an open PDK (SKY130, GF180 or IHP SG13G2), OpenROAD does this.
- **A pad ring** with electrostatic discharge protection, and a package.
- **Factory tests:**
  - there is a memory self-test, JTAG, scan chains in the chip for manufacturing, ATPG patterns per block and their STIL files;
  - the patterns of the chip's main chain are not generated whole, only per block: the current search is too slow for a whole die;
  - the chains have pins of their own, not the JTAG TAP;
  - at-speed delay (transition) faults are missing.
- **Silicon:** nothing has been tested on a physical FPGA or in a chip.
