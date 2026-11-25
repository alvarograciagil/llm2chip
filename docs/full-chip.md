# The whole chip: design and simulation

Goal: the design and the simulation include everything the real chip would have. The only difference between simulating and running in silicon is how data gets in and out through the host port.

## The boundary between the chip and the world

The real chip would have a PCIe or USB PHY. In simulation, that PHY is replaced by a simple digital port. Everything on this side of the port is the same in both cases.

| Signal | Direction | Use |
|---|---|---|
| `host_rx_*` | Input, byte stream | Messages from the host: turns, control (see [below](#the-host-port-messages)) |
| `host_tx_*` | Output, byte stream | Messages to the host: answers, status, errors |
| `image_req_*`, `image_data_*` | Requests for image blocks | Reading the model from the host |
| `reset` | Input | Starts again from the image: reloads everything, weights included |
| `clear` | Input | Clears the conversation without reloading the weights (see [below](#a-conversation-and-the-clear-pin)) |

The chip receives text and returns text, in messages. Tokenizing, inference, sampling and detokenizing happen inside. A chip run by a schedule (a speech recogniser or a sound decoder) exchanges streams of 32-bit words instead: samples or codes in, tokens or samples out ([voice.md](voice.md#what-the-host-sends-and-gets-back)).

## The host port: messages

Everything that crosses the host port is a message, both ways, so a host can ask and steer the chip while it works, not only send text. The format is SLIP's: a message is its **kind** (a byte), its **bytes**, and an **end** byte, `0xC0`. Inside a message, `0xC0` goes as `0xDB 0xDC` and `0xDB` as `0xDB 0xDD`. There is no length field: the end marks the boundary, and a byte lost on the way spoils one message only, since the next end starts afresh. `netlist::port` writes and reads them for every host: the tests, the engine's simulator, the gates in the browser.

| From the host | Kind | Bytes | The chip answers |
|---|---|---|---|
| A turn as text | `0x01` | UTF-8 | Its answer as text, a message a token, then the end |
| A turn as tokens | `0x02` | 32-bit tokens | Its answer as tokens, a message a token, then the end |
| A picture | `0x03` | The patches, as the chip's encoder takes them | Nothing: the turn after it places it |
| A stream | `0x04` | 32-bit words, for a chip run by a schedule or Show-o's passes | A stream of words |
| Status | `0x10` | | Whether it is answering, the positions the conversation takes and its capacity |
| Options | `0x11` | | Each option's number and value |
| Set an option | `0x12` | Its number and its value | The option with its new value, or an error |
| Stop | `0x13` | | The answer ends there |
| Clear | `0x14` | | As the clear pin |

| From the chip | Kind | Bytes |
|---|---|---|
| Boot | `0x20` | A byte a segment loaded: `.` for a good checksum, `E` for one that failed |
| Ready | `0x21` | The conversation's capacity in positions, zero for a chip that answers once: it waits for a first turn |
| End of an answer | `0x22` | Why (end token, limit, loop, full context, stopped, cleared), the positions the conversation takes and its capacity |
| Error | `0x2F` | A code (no plan for this many chips, an unknown message, one the chip does not take now, an unknown option, a text too long) and a detail |

**What the host can see and change while the chip runs.** The status gives the context in use, so the host knows the percentage without counting. The options are the generation's settings in the chip's description: the most tokens an answer takes, greedy or not, top-k, the temperature, the seed and the reasoning budget. Setting one writes it in data memory, and the next token follows it.

**Between tokens.** Before each token it writes, the firmware looks at the port's status, as it did for the clear pin. A message that is not a turn is handled there and answered at once, so a status asked in the middle of an answer comes back between two tokens. A turn sent during an answer waits for the answer's end. A message the chip answers in the middle of one of its own, such as a stream of words, ends that one first and starts it again after.

**The port's hardware does the framing.** As bytes arrive it takes the escapes out and keeps each byte's place beside it in the receive queue, as a two-bit mark: a message's start (its kind), a byte inside, or its end. The status register tells the CPU what is at the head of the queue: a payload byte (the usual case), a start, an end, or the clear pin. A payload byte costs the firmware what a plain byte port did. On the way out the firmware writes a start (`HOST_TX_START`), the bytes and an end (`HOST_TX_END`), and the hardware puts in the escapes. Reading the head without taking it (`HOST_RX_PEEK`) lets the firmware leave a turn waiting. All this is about a hundred flip-flops and the queues' two extra bits.

**Integrity** is left to the physical link. The framing limits a lost or corrupted byte to one message, and a host can always clear the conversation and start again.

### The host port's links

Besides its byte port (`host_*`: valid, ready and a byte each way, on the chip's clock), a chip can have physical links, chosen in the design (`host_uart`, `uart_baud`, `host_qspi`; `netlist::soc::HostLinks`). Each carries the same messages:

| Link | Pins | Carries | Costs |
|---|---|---|---|
| Byte port | `host_rx_*`, `host_tx_*` | A byte a cycle each way; the chip holds the host back when its queue is full | Always there |
| Serial port (UART) | `uart_rx`, `uart_tx` | 8N1, a byte in ten bits at the chosen rate | 234 gates, 48 flip-flops |
| Quad SPI target | `qspi_sck`, `qspi_cs_n`, `qspi_in` (4), `qspi_out` (4), `qspi_oe`, `qspi_irq` | Four bits each clock edge, the clock up to an eighth of the chip's: about 60 MB/s at 1 GHz, 2 MB/s at 35 MHz | 227 gates, 42 flip-flops |

**Which link answers.** The chip answers on the link the last message came from. Before any message it speaks on the first serial link it has (the quad SPI one, then the serial port), or the byte port. What waits in the send queue goes out on whichever link the next message names, so a host that speaks first gets the boot report too.

**One link at a time.** A message that starts on one link holds the port until its end. In the meantime the byte port waits, but a serial link cannot be held back, so its bytes are dropped and counted, as are a serial link's bytes that come with the receive queue full. `debug_select` = 6 (`DEBUG_HOST_DROPPED`) reads the count.

**The quad SPI target** works in SPI mode 0, the high nibble first. After the select falls, the first byte is a command:

| Command | Then |
|---|---|
| `0x01` write | The host's bytes, into the receive queue |
| `0x02` read | A dummy byte while the lines turn round, then the chip's bytes; with nothing to send it sends `0xC0`, which a host's reader skips |
| `0x05` status | A dummy byte, then two bytes: the room in the receive queue and the bytes waiting to be read |

The interrupt line is high while the chip has something to send on this link. The host should read the status before writing more than the room it gives.

**Timing.** The serial port's line and the quad SPI's pins go through two flip-flops into the chip's clock. A read's byte is taken from the send queue a few cycles after the last one went out, which is why the quad SPI clock may be up to an eighth of the chip's.

**Speech** goes over the host port as words: about 64 KB/s to listen and 192 KB/s to speak. A serial port slower than that would make the chip wait for it, and the estimate warns about it.

**A host for a board.** `tools/host/chip_host.py` frames what you ask over a board's serial port and prints what the chip answers: a question as text or tokens, the status (with the context in use as a percentage), the options, setting one, a stop and a clear.

**Checked:** `each_link_is_answered_on_its_own` (in `crates/netlist/tests/links.rs`) runs the control system with the real firmware in gates, with a serial port and a quad SPI target beside its byte port. Before any message the boot report comes out on the quad SPI link, and its interrupt line goes down once it is read. Then a status asked on the serial port is answered there, an option asked on the byte port is answered there, and one written over the quad SPI link is read back from it, escapes and all. A serial byte sent while the byte port is inside a message is dropped and counted.

**Checked:** `the_host_asks_and_steers_the_chip_in_messages` (in `chip.rs`) asks a chip its options, sets one and sees the next answer follow it, asks for its status while it waits and in the middle of an answer, stops an answer, asks for a clear as a message, and gets the errors of what the chip does not take. Every test of the whole chip in gates checks the framing both ways, byte for byte and cycle for cycle against the instruction level.

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
| Routine schedules | Orders a program of routines by itself (a speech recogniser, a sound decoder): lengths, loops, sums over rows, tokens, samples back to the host; see [voice.md](voice.md) | In firmware, checked against the host's order and at instruction level |
| Command queue before the die | The CPU's orders to the die wait there until the die is free, so the CPU orders the next routine while the die runs this one; built only where asked (`DieFeatures::queue_bits`) | In gates and ISS, cycle for cycle |
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

`chip_design` joins the CPU, the memories, the load controller, the die and the sampling unit. With a tiny model, the real firmware receives bytes, loads the image, runs the tokens and returns bytes. The result and the cycles match the instruction level (CPU in the ISS and die on the exact machine): 230,144 cycles in both (`text_goes_through_the_whole_chip_in_gates`). With a command queue before the die, 230,439 (`text_goes_through_a_chip_with_a_command_queue_in_gates`); a toy Moonshine's whole chip following its schedule through the queue, 1,192,263.

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
| `reset`, `clear`, `host_*`, `debug_*` | Reset, clearing the conversation, host (only used on the leader) and debug |

### Splits

- **By layers:** each chip runs some whole layers and passes the hidden vector to the next over `link`.
- **By rows:** every chip keeps a slice of the rows of each matrix, they compute each product at once and pass their slices round a ring over `link`, the last joined to the first.
- **Mixed:** groups of chips split by rows, joined in a chain by layers. Within a group the ring runs over `link` and `loop` closes it, from the group's last chip to its first; `link` also carries the hidden vector from the last chip of one group to the first of the next, which passes it to the others in its group. Each microcode word carries two bits, one for the port it sends on and one for the port it takes from, because one word can send for one instruction and take for another. The logits' matrices are not split: only the first chip computes them, from the final state the last one sends it. `two_groups_of_two_dies_run_a_token_over_their_links` runs a token on four dies in gates joined this way, and checks the first one's logits are bit for bit those of a single die.

An image carries a plan for each number of chips: in the mixed split, numbers that do not form whole groups are split by layers, and those that fit in a single group, by rows.

Each word carries a sequence number and a code that corrects one bit and detects two, with a valid and a ready signal. The receiver corrects single errors and, on a double one, asks for the word again over the acknowledgement signals; the sender keeps unacknowledged words and repeats them, and if the request is lost, asks again after a while. The receiving half runs on the sender's clock and hands words to the chip's clock through a queue with Gray pointers, so the two chips do not need the same clock (`words_cross_between_two_unrelated_clocks`). On the simulated board the chips share a clock; the board tests inject single and double flipped bits, and require every single error to be logged as corrected and every double one as repeated. The image must carry the layer split for every number of chips it supports.

## Burnt-in weights

The "Weights" parameter decides whether the model can be changed. With ROM, each cell's weights, scales and biases are fixed at manufacturing: the cell loses its load ports and address selector, the die loses the network that writes into the cells, and boot no longer transfers them. The microprogram, tables and firmware are still loadable, so the chip can be corrected without new masks, but cannot change model.

The biggest saving does not come from the logic but from the density of ROM against SRAM. To synthesize it in a real technology, `llm2chip_rom` must be replaced by the foundry's ROM macro, with each cell's contents.

## A fixed system prompt

A project can give the chip a **system prompt** (`system_prompt`): a fixed text it runs before every question, such as how to answer. A voice assistant needs one to answer in a short sentence.

**Where it lives.** Its tokens go in the chip's description (`netlist::chip::add_system_prompt`): `INFO_SYSTEM` points at a word with their count, followed by the tokens. The description is part of the image, so the system prompt sits wherever the image does:
- With the weights in ROM, it is burnt in with them, and changing it takes a new chip.
- Otherwise it is loaded at power-up with the weights, and writing a new image changes it.

The description has 1,008 words. After its own words, that leaves room for roughly 850 tokens, and longer prompts are refused.

**How it runs.** At the start of a conversation, before reading the host's prompt, the firmware hands each of these tokens to the same path as the prompt's tokens. Their positions, the reasoning budget and the draft head therefore treat them exactly as text the host sent. A chip without one pays a single branch, and the firmware grew by 108 bytes. Chips run by schedules (speech, sound) and Show-o's chips do not read a prompt, so they ignore it.

**What the host sends.** The host frames the conversation with the model's chat template, system prompt first (`PromptStyle::render_with_system`), and splits it there (`split_system`). The system prompt's framed tokens go in the chip, and the rest goes over the host port. The model's special tokens, such as BOS, go with the system prompt, so a chip that tokenizes on the CPU no longer adds them to the question.

A template that does not put the system prompt ahead of the question is refused, rather than sent as different text. Llama 2's, which folds it into the first user turn, is one example. So is a split whose tokens differ from the whole conversation's.

The simulator (local and in the browser), the Listen and Speak views, the voice planner and written chip images all take the project's system prompt.

**What it costs.** A chip that holds a conversation (next section) reads the system prompt once a conversation starts, after power-up or the clear pin, while it waits for the first question. The estimate shows:
- its tokens (about four characters each, as the browser has no tokenizer);
- the time it takes at the prompt reading speed;
- that it takes that much of the context.

The voice planner counts its tokens in the language model's context but not before the first output. A chip that answers once reads it before its only question.

`the_firmware_runs_the_chips_system_prompt_before_the_hosts` checks it at instruction level with the tiny model. The system prompt goes in the description and the question over the port, and the chip writes the same tokens as the software reading the whole conversation.

## A conversation and the clear pin

A chip can **hold a conversation**: after each answer it sends the answer's end and waits for the next turn. The turn goes on over the same cache, without reading the earlier turns again. Two pins start again:

| Pin | What it does | Takes |
|---|---|---|
| `reset` | Starts from the image: the firmware, the weights, everything | As long as loading the image |
| `clear` | Clears the conversation: zeros over the cache and the recurrent state, and the vector memory's first contents again; then it reads the system prompt and waits | Milliseconds |

**The answer's end** is a message of its own, with why it ended and the positions the conversation takes. The host sends the next turn as its chat template writes it after the answer, as `voice::Conversation` does. A cache that fills up takes nothing more: the chip ends the turn and waits for a clear.

**When the context fills,** the chip forgets nothing by itself. It takes no more tokens until the clear pin: an answer that fills the cache ends there, and the next question has nowhere to go. Each key in the cache carries its position's rotation, and each token sits in the place its position gives it. Dropping the oldest tokens is therefore not free. These are the ways to go on:
- **From the host, with the chip as it is.** Before the cache fills, the host raises the clear pin and sends again the system prompt and the latest turns that fit. It can also send a summary of the conversation instead of the turns, which compresses the context. The chip's own model can write the summary as one more turn, if it is asked while the cache still has room for it. It costs reading those turns again, or writing the summary and reading it, once at each compression.
- **In the chip, not built.**
  - **A ring with attention sinks**, as in StreamingLLM: the first tokens stay, and the newest overwrite the oldest. It needs the firmware to place the cache's tokens and rotations otherwise, and it is still to be seen whether the die's addressing takes it without new logic.
  - **A cache in fewer bits,** 8 or 4 a value: more tokens fit in the same memory. This is a choice of the design.
- **DeltaNet's layers** do not fill: their recurrent state has a fixed size and fades the old by itself. In a hybrid model, such as Qwen3.5, only the attention layers' cache fills.

**What the clear zeros.** It zeros what a conversation leaves in the die, and nothing else:
- **The cache.** On a die that keeps each token's row of keys and values together, only the rows the conversation used are zeroed: what the time takes grows with the conversation, not with the context. A die that keeps each column's tokens together zeros its cache whole.
- **The recurrent state** (DeltaNet's and the convolutions'). Zeroing it is needed for correctness, not only for privacy: without it, the next conversation starts from the old one's state.
- **The vector memory's first contents.** The firmware loads every vector segment of the boot again, so anything a conversation rewrote, such as rotation rows for an image, is as it was.

At power-up the firmware zeros the whole cache, since it cannot know what the memory held.

**How it works.**
- **The pin:** it stays requested until the firmware writes the host port's status register, and the firmware sees it as bit 1 of that status (`CLEAR_REQUESTED`). Raising it drops the bytes the host had sent and the chip had not read. The firmware watches it while it waits for bytes and before each token it writes, so it can cut an answer short. A host can ask the same in a message.
- **The load controller** writes the zeros. While its register 9 (`LOADER_FILL`) holds 1, each element's words are zeros and the image is not read. The timing is the same as a read the image answers at once: three cycles an element.
- **The conversation table** sits in the chip's description (`netlist::chip::add_conversation`). It holds the cache's capacity in positions, then entries of five words: a first address, how many registers from it and the step between them, how far each reaches, and the elements of one position's row (zero when the register is zeroed whole). `Layout::conversation_state` makes it from the die's own layout.
- **A chain of chips:** only the leader's pin is wired. The leader sends the clear down the chain as a command with the positions used, and each chip zeros its own layers' part.
- **A software die**, as the simulator runs a model through the real firmware, keeps its cache in its engine. Its table has one entry, a register that stands for that cache (`sim::chip::runner_conversation`), and the load controller's fill of it empties the engine.

The estimate shows how long clearing a conversation that filled the context takes. The chip images the application writes hold a conversation, and so does the Simulator view (see [simulation.md](simulation.md#a-conversation-in-the-simulator)). A chip whose description has no table answers once and stops, as before.

**Checked:**
- `a_conversation_goes_on_over_its_cache_until_the_clear_pin` and `a_die_with_the_cache_by_rows_clears_only_the_rows_used` run at instruction level on the die's exact machine, on a die that keeps its cache by columns and one that keeps it by rows. A second turn writes what one question with the whole conversation writes. After a clear, raised while the chip waits or in the middle of an answer, the cache and the state read zeros and the same question gets the first answer again. Without the table's entries, the recurrent state is left behind.
- `the_clear_pin_works_in_gates_as_at_instruction_level` runs the whole chip in gates: an answer, the clear while it waits, and the same question again. It writes the same bytes, and between the answers it takes the same cycles.
- `the_leading_chips_clear_pin_clears_every_chip_of_the_chain` (in `multichip.rs`, with the slow tests) runs two chips in gates on a board, each with its own layers. After an answer each holds its part of the conversation. The leader's pin zeros both parts, in some 10,600 cycles, and the same question gets the first answer again.
- `the_firmware_holds_a_conversation_on_a_kept_engine_until_the_clear_pin` (in `tiny_model.rs`) runs the firmware at instruction level on a software die that keeps its engine, as the simulator does: two turns as the whole conversation, then the clear and the first answer again.

**The firmware** takes 35,704 bytes of a program memory of 48 KB, in banks of 32 and 16 KB (`netlist::soc::PROGRAM_WORDS`): the host port's messages took it past 32 KB. It is built with less inlining (`-inline-threshold=25`, in `firmware/.cargo/config.toml`), which keeps it fast at decoding sound. Reading a word from the host calls nothing in the usual case, so the function saves no registers: what is not a payload byte goes to another function as its last step. With it the toy recogniser's CPU takes 2.06 million cycles, against 2.28 million before the messages.

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

## Programs of routines

A program that is a set of routines rather than a token's pass (an image or speech encoder, a sound decoder) is ordered by a **schedule**: records the firmware follows (`netlist::schedule`), kept at word 0x2000 of the data memory, where a tokenizer's tables would be. Words of the chip's description say where it is and how deep the command queue is:

| Word | What |
|---|---|
| `INFO_SCHEDULE` (29) | The schedule's word in data memory; zero for a chip that reads a prompt |
| `INFO_QUEUE` (30) | The command queue's orders; zero without one |
| `INFO_SYSTEM` (31) | Where the system prompt starts in the description; zero without one (see below) |
| `INFO_CONVERSATION` (32) | Where the conversation table starts in the description; zero for a chip that answers once (see below) |
| `INFO_PARAMETERS` (33 on) | The parameter registers, seven words each |

The **command queue** (`DieFeatures::queue_bits`) sits between the CPU's writes and the die: writes to a parameter register or to `REG_ENTRY` wait in it, and the die takes one a cycle whenever it is free, an entry starting it. `REG_QUEUE_ROOM` (49) says how many more it takes, and `REG_STATUS` says free once it is empty and the die idle. It is built only on the dies that ask for it, such as voice's, so the CPU orders the next routine while the die runs this one. [voice.md](voice.md#the-chip-runs-its-programs-by-itself) has the records, what each chip's host sends and what the firmware's work costs.

## Phases

1. System bus, memory map and host port.
2. RV32IM core in gates and in the ISS, checked against each other.
3. Firmware in `no_std` Rust: boot, tokenizer, main loop.
4. The chip's image and load controller, with the three levels.
5. Sampling unit and RNG.
6. ECC, CRC, BIST and repair, with fault injection.
7. Clock, reset, power, debug and counters.
8. Integration: whole chip in the schematic, estimator from the netlist, simulation settings per block, console and boot timeline.
