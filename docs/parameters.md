# Chip parameters

<!-- Generated from web/src/params.ts and web/src/paramDocs.ts: UPDATE_DOCS=1 npx vitest run docs -->

Everything that can be adjusted, what it changes in the circuit and where it shows. Those in the application's panel move the estimator at once; the generator's are derived by the compiler from the model and the microprogram; the generation ones are chosen with each request. In the application, each parameter is explained when the cursor goes over its “?”, and clicking it opens this same documentation.

## Goal

### Max parameters (`max_params_g`)

Size of the largest model the chip can hold, up to 100 trillion parameters (100,000 G). Sizes the weight memory.

*From 0.0001 to 100000 G, on a logarithmic scale.*

### Context (`context_tokens`)

Up to 1 M tokens. The KV cache grows linearly with the context and is the block that most limits speed; from about 32 K it is worth moving it to HBM4.

*From 128 to 1048576 tokens, on a logarithmic scale.*

### MTP support (`mtp_enabled`)

Adds the multi-token prediction (MTP) head: an extra layer that guesses the next tokens while the main model checks the previous ones. With a single user the dies wait a lot, and this work fills that gap: when it guesses right, several tokens come out per pass.

*On or off.*

### MTP depth (`mtp_depth`)

How many tokens the MTP head runs ahead each round. More depth gives more tokens per pass when it guesses right, but each level guesses less often (in Qwen, around 94 %, 86 % and 74 %) and needs state kept to undo.

*From 1 to 3.*

### Vision (image analysis) (`include_vision`)

The chip reads images: reserves area for the image encoder (a ViT of about 400 M parameters in Qwen) that turns each image into tokens for the language model, so it can describe them or answer about them. It does not draw: that is Draws images.

*On or off.*

### Draws images (text to image) (`draw_images`)

The chip also draws: a model (Show-o) draws and retouches pictures from a text, on a die of its own. It is not vision, which analyses images. The floorplan and the estimate count it: a reference die measured with Show-o's transformer, its embedding table in external memory and MAGVIT-v2 on the host. Off, nothing of drawing is in the chip or the app.

*On or off.*

### Listens (speech to text) (`speech_in`)

The chip also listens: a speech recogniser (Moonshine) turns a spoken question into text for the language model, on dies of its own. The floorplan and the estimate count them: reference dies until they are planned with your models in the Listen view. Off, nothing of listening is in the chip or the app.

*On or off.*

### Speaks (text to speech) (`speech_out`)

The chip also speaks: a speech writer (OuteTTS) and a sound decoder (DAC) say the answer aloud, on dies of their own. The floorplan and the estimate count them: reference dies until they are planned with your models in the Speak view. Off, nothing of speaking is in the chip or the app.

*On or off.*

## Organization

### Target speed (`target_tok_s`)

With the model's size and the clock, sets how many multipliers are needed. With 0 you set them by hand.

*From 0 to 40000 tok/s.*

### Multipliers per die (`macs_per_die`)

With a single user only one die works at a time, so this value sets the speed.

*From 2 to 2000000, on a logarithmic scale.*

### Frequency (`freq_ghz`)

The die's clock. Its slowest path between registers limits it: the multiplier and the adder tree (more stages shorten them) and the node's gate delay. A faster clock gives more tok/s with the same logic, but more power.

*From 0.02 to 3.4 GHz, on a logarithmic scale.*

### Die type (`die_mode`)

Either way it is the same chip repeated, loading its part of the model from the image. To measure sizes the die for this model's part; expandable reserves room (layers, width, vocabulary) to connect more chips for larger models. Only with the weights burnt into ROM is each die different.

*Options: Tailored to this model, All alike (expandable).*

### Max layers (`max_layers`)

The layers come from each model: Qwen3.8 has 64 and Llama 3.1 8B, 32. This limit reserves their state in each die.

*From 1 to 1024, on a logarithmic scale.*

### Max hidden size (`max_hidden_size`)

Largest width of the vector that runs through the layers (5,120 in Qwen3.8-27B). With identical dies, it sets the width of buffers and links for any model loaded.

*From 64 to 65536, on a logarithmic scale.*

### Max vocabulary (`max_vocabulary`)

Distinct tokens the largest model takes (248,320 in Qwen). Sizes lm_head, the sampling unit and the tokenizer tables.

*From 256 to 4194304, on a logarithmic scale.*

### Dies (`dies`)

How many dies the model is spread over. With 0 the fewest is chosen with which each die fits the maximum area. With a single user the dies work in turns, so more dies give more memory but not more speed on their own.

*From 0 to 32.*

### Chip split (`chip_split`)

How several chips share the model. By layers: each chip keeps whole layers and passes the hidden vector to the next; little communication, but with a single user one chip works at a time. By tensor: each chip keeps a part of every layer and all of them compute each layer at once; much faster for one user, but they add up their results twice per layer and need fast links, inside the package. Mixed: groups of chips in tensor joined in a chain, so each group fits in a package.

*Options: By layers, in a chain, By tensor, all at once, Mixed: tensor groups in a chain.*

### Chips per group (`tensor_group`)

In the mixed split, how many chips share each layer. The groups are joined in a chain. With 1 it is the split by layers; with all the chips, by tensor. A group should fit in a package, joined by UCIe.

*From 1 to 32.*

### Max die area (`die_max_mm2`)

Maximum area of each die. The physical limit is the scanner's reticle (26 × 33 mm, about 858 mm²). Large dies yield less per wafer, so a smaller limit sometimes pays.

*From 10 to 858 mm².*

## Memory

### Weights (`weight_backend`)

With SRAM the chip takes other models. With ROM the weights are burnt in at manufacture: weight writing and its load time go away, but changing model needs new masks.

*Options: SRAM (swappable model), ROM (model etched in), Stacked DRAM, HBM4, External memory.*

### Cache and state (`kv_backend`)

Where the attention's KV cache and DeltaNet's recurrent state live. In SRAM they are fastest but take area that grows with the context; outside, that memory's bandwidth limits speed.

*Options: SRAM, HBM4, External memory.*

### HBM4 stacks (`hbm_stacks`)

Each HBM4 stack adds capacity and bandwidth, and a PHY of about 12 mm² on the die that connects it.

*From 1 to 16.*

### DRAM bandwidth (`stacked_dram_tbps`)

What the DRAM stacked on the die delivers per second. With the weights out of SRAM, it sets the speed: each token reads every weight once.

*From 10 to 400 TB/s, on a logarithmic scale.*

### External bandwidth (`ddr_gbps`)

What the external memory (DDR, LPDDR) delivers per second. With the weights outside, tok/s cannot exceed this value divided by the model's size.

*From 0.01 to 400 GB/s, on a logarithmic scale.*

### External capacity (`ddr_capacity_gb`)

External memory available: it must hold what is taken off the chip (weights, KV cache or both).

*From 0.004 to 256 GB, on a logarithmic scale.*

## Arithmetic

### Weight format (`weight_coding_preset`)

The circuit adapts: the memory, the multiplier and the decoder of each cell change.

*Ready-made formats: Grouped affine (MLX), GGUF, Fixed point (Q), Floating point, Microscaling, Posit, Logarithmic, Codebook.*

### Coding (`weight_coding`)

How each stored code is turned into a number. Each family generates its own decoder.

*Options: Grouped affine, Fixed point, Floating point, Posit, Logarithmic, Codebook.*

### Weight bits (`weight_bits`)

Bits stored per weight, from 2 to 32. Sets the weight memory, which dominates the area, and the size of each cell's multiplier. With fewer bits more model fits but precision is lost.

*From 2 to 32.*

### Exponent bits (`weight_exponent_bits`)

In floating point, the exponent bits. In posit, “es”. In logarithmic, the fraction bits.

*From 1 to 15.*

### Keep 8-bit tensors (`use_model_mix`)

Keeps the embeddings and the output head at 8 bits and the rest at 4. For Qwen it reproduces the mix of its released weights, which also keeps out_proj and the MLP of the last layers at 8. The cells keep two 4-bit codes in each 8-bit word.

*On or off.*

### Bias per group (`weight_bias`)

Whether each group also keeps a bias (weight = code · scale + bias). MLX and Q4_1 do; Q4_0 only keeps a scale. The bias costs memory and an addition per group.

*On or off.*

### Fine scale per group (`weight_fine_scale`)

A floating-point scale like NVFP4 needs a multiplier. A power of two like MXFP4 only a shifter, but it loses precision.

*On or off.*

### Weights per group (`group_size`)

Consecutive weights that share scale and bias (32 in MLX's Qwen). Small groups give more precision but more factors to keep and apply. With 0 there are no groups.

*From 0 to 256.*

### Scale and bias bits (`scale_bits`)

Precision with which each cell keeps the scale and bias of each group. More bits: less error, and more memory and scale multiplier.

*From 8 to 32.*

### Activation bits (`activation_bits`)

Width of the values that travel through the chip between operations. Sets the width of the multipliers, the activation network and the vector memory: among what weighs most in area and precision.

*From 8 to 32.*

### Accumulator bits (`accumulator_bits`)

Width of each cell's accumulator, which adds the products of weight and activation. If short, the sums overflow; if the format needs more, the chip uses the width needed.

*From 16 to 128.*

### DeltaNet state bits (`delta_state_bits`)

Width of DeltaNet's recurrent state, a matrix per head updated every token. The reference uses 32 bits; reducing it saves a lot of memory, but the error builds up over the context and it is the largest numerical risk.

*From 8 to 32.*

### KV cache bits (`kv_bits`)

Width of each key and value stored per token. The cache grows with the context, so with long contexts it is the memory that takes most after the weights.

*From 4 to 32.*

## Pipeline stages

### Custom circuit (`auto_circuit`)

The generator tries every way of building the cell for these parameters, measures its real circuit and keeps the best.

*On or off.*

### Prioritize (`circuit_goal`)

What is minimised among the cell designs that reach the clock asked for: area, energy per token or a balance of both.

*Options: Area and energy, Area, Energy.*

### Multiplier type (`multiplier_kind`)

Wallace and Booth reduce the products with carry-free adders and are much faster. The array is simpler and more compact at low clocks.

*Options: Wallace, Booth radix-4, Array, Precomputed multiples.*

### Multiplier (`mac_stages`)

More stages allow a higher clock in exchange for area and latency.

*From 1 to 6.*

### Scale and bias cycles (`factor_steps`)

Each cell applies a group's scale and bias over this many cycles, a few bits a cycle, while it already adds up the next group. With 1 each cell needs a whole multiplier; with 8 one eight times smaller is enough, with no loss of speed if the group has at least 8 weights.

*From 1 to 16.*

### Adder tree (`adder_stages`)

Registers inside the tree that adds the cells' products. More stages allow a higher clock in exchange for latency and flip-flops.

*From 1 to 6.*

### Activation network (`noc_stages`)

Registers in the network that sends each activation to every cell. In large dies the wires are long: more stages keep them from limiting the clock.

*From 1 to 4.*

## Vector units

### Lanes per die (`vector_lanes_per_die`)

Lanes of the vector units, which multiply activation by activation: attention Q·K, delta rule, gates and norms. Few lanes lengthen the parts that are not matrix products.

*From 1 to 512000, on a logarithmic scale.*

### Vector units per die (`vector_units_per_die`)

Copies of the vector unit: each computes one attention or DeltaNet head at a time.

*From 1 to 256, on a logarithmic scale.*

### Logits per cycle when sampling (`sampler_lanes`)

How many logits the sampling unit compares at once. With few, picking the token holds the chip back.

*From 1 to 4096, on a logarithmic scale.*

### Sampling candidates (`sampler_entries`)

How many of the best tokens the sampling unit keeps: the largest top-k it takes. Each candidate costs registers and cycles when picking.

*From 1 to 1024, on a logarithmic scale.*

### Bits per lane (`vector_bits`)

Width of the vector units' multipliers, which work activation by activation (attention, delta rule, norms). Their area grows with its square.

*From 8 to 32.*

### Function units (`function_units_per_die`)

Tables that compute non-linear functions (exp, SiLU, sigmoid, inverse square root…) by looking up and interpolating. More units apply the function to more elements at once.

*From 1 to 128.*

### Table resolution (`function_table_bits`)

Entries of each function table, in powers of two: 12 is 4,096. More entries reduce the error of exp, SiLU or the inverse square root in exchange for memory in each unit.

*From 4 to 20.*

## System

### Die-to-die link (`link`)

UCIe joins dies inside one package: fast (about 10 ns per hop) and small. SerDes joins them across the board: slow (about 150 ns) and larger. I/O pins are all that chips from an open shuttle have, each in its own package: 16 pins each way at 50 MHz and about 60 ns per hop. Automatic fills packages and joins them with SerDes, except on an open shuttle, which uses the pins.

*Options: Automatic by package, UCIe, Board SerDes, I/O pins.*

### Link width (`link_words`)

How many activations each link passes per cycle. The hidden vector crosses every boundary between dies on every token: a narrow link adds thousands of cycles per boundary; a wide one needs more UCIe or SerDes modules.

*From 1 to 4096 activations/cycle, on a logarithmic scale.*

### Weight loading (`load_interface`)

Where the weights arrive from when the chip powers up: NVMe, PCIe 5.0 or UART. It only affects start-up time, not generation speed.

*Options: NVMe, PCIe 5.0 x16, UART.*

## Gate generator

The circuit that is simulated gate by gate and exported to Verilog comes from these parameters. Most are not chosen by hand: the generator derives them from the model and the microprogram so the die has just what it needs.

### Cells per die

Each cell has its weight memory and its multiplier. More cells split each matrix-vector product among more multipliers: fewer cycles per token and more area. There must be at least as many cells per unit as lanes.

### Lanes per unit

Elements a vector unit processes per cycle in element-wise operations (sums, norms, functions). In powers of two.

### Vector units

Copies of the vector unit that run the same instructions on different data: each takes an attention or DeltaNet head, or a piece of a vector. They are joined by a tree network that adds and finds maxima.

### Depth of each cell's memory

Weight words and scale-and-bias groups each cell keeps. It comes from spreading all the chip's matrices over its cells.

### Codes per word

The cell's word is as wide as the model's widest code (8 bits in MLX's Qwen). Each matrix keeps 1, 2 or 4 codes per word according to its own width (q8, q4, q2), and the microcode tells the cell how many when it points at it. It is turned off with ECC, which protects whole words.

### Scale and bias cycles (K)

Each cell applies a group's scale and bias in K cycles, a few bits per cycle, while it adds up the next group. It shrinks the scale multiplier to 1/K of its size; it only costs cycles if the group has fewer than K weights.

### Multipliers

The cells use radix-4 Booth (fewer partial products) and the lanes Wallace (a tree without carry propagation). There are also array (compact, slow) and precomputed multiples. Adders of 8 bits or more are Brent-Kung prefix adders.

### Vector registers and index

The microprogram names each piece of data by register and index. A table, loaded with the image, gives each register its base in the vector memory and the step of the loops that go through registers by number; the compiler works out how far each register reaches in its routines, so each one takes only what it uses. Without the table, each register would reserve as much as the longest.

### Microcode

The die's fixed program, with one entry point per routine (token, logits, MTP head…). Each field of the control word has only the bits the program uses, and each operation that does not appear is not generated.

### Key columns

With several units, the KV cache's keys are kept in columns of 2^n tokens so each unit scores its share of the tokens at once.

### Function tables

Index bits of each table (resolution) and how many tables there are (exp, SiLU, sigmoid, softplus, inverse square root, tanh…).

### Weights in ROM

With the model burnt in at the factory, weight writing and loading disappear, and the memory is denser, but changing the model takes other masks.

### ECC in cells

Corrects one bit and detects two per weight word. It costs about 430 gate equivalents per cell, plus check bits.

### Skip zeros

Freezes the multiplier's input when the weight is zero so it does not switch: saves energy, not area.

### Control CPU

A RISC-V RV32IM with its program and data memory: boots the chip, loads the image with CRC, tokenizes, drives the die through its registers, samples, and applies the stop list and the reasoning budget.

### Sampling unit

Logits it compares per cycle and candidates it keeps (the largest top-k). It picks the token with temperature using a table of exponentials.

### Chip-to-chip link

Each word travels with a sequence number and an error-correcting code; the receiver has a queue and asks for retries, so the board can delay words without breaking anything.

### Wide link

With several chips, the link has a lane of its own for each vector lane, each with its sequence, its error correction and its queue, like UCIe lanes: a whole block crosses in one cycle. The hidden vector goes from one chip to the next in tens of cycles instead of thousands. It is only generated if the microprogram sends blocks.

## Chip split

When the model does not fit in one chip, or more speed is wanted, there are three ways to split it. It is chosen with “Chip split” and, in the mixed split, “Chips per group”.

### By layers, in a chain

Each chip keeps a stretch of consecutive layers. For each token, the first does its layers and passes the hidden vector (a few thousand values) to the second, and so on to the last, which returns it to the first for the logits. Advantages: little communication, which tolerates slow links across the board; no synchronization within a layer; each chip keeps only its layers' weights and cache; reading a prompt, they all work at once, each on a token. Drawback: generating for a single user only one chip works at a time, so more chips give more memory, not more speed. The gate generator implements it and tests it in gates with two chips over a noisy link.

### By tensor, all at once

Each chip keeps part of every matrix and of every group of heads, and they all compute each layer at once. After the attention and after the MLP of each layer they add up their partial results in a ring: 2·(N−1) steps, each with an N-th of the vector. Advantage: with one user the speed grows almost with the number of chips. Drawbacks: many synchronizations per token, which need fast, low-latency links, in practice inside the same package; a prompt is not read in a chain. The estimator models the most efficient variant, which also splits the attention heads. The gate generator implements a simpler variant today, “by rows”: each chip keeps an equal band of rows of each matrix and, after each product, the chips pass their bands round a ring until all have the whole vector. Before narrowing each product they agree on the shift over the link (two words per chip), so the result is bit for bit that of a single chip; the test in gates checks it with two chips and noisy links. Attention, DeltaNet and the norms are computed whole by each chip: the weights are split, but the KV cache and the state are repeated in all of them, and more moves over the link than in the efficient variant.

### Mixed

Groups of chips that share each layer by tensor, joined in a chain. It is what large systems use: each group inside a package, with UCIe, and the groups across the board. With a group of 1 chip it is the split by layers; with all in one group, the split by tensor. In gates, the last chip of each group closes the group's ring with a link back to the first, and the hidden vector passes to the next group over the usual link. A boot image carries a plan for each number of chips: those that do not form whole groups are split by layers.

## Sizing

The search tries die shapes (cells, lanes, units, number of chips), generates the real microprogram for each one and counts its cycles at the last position of the context, so the tok/s come from the circuit and are guaranteed with the context full.

### Target tok/s

The minimum speed with the context full. The search keeps the die with the least silicon that reaches it.

### Time to first token

The maximum after a prompt of a given length. With several chips, the prompt advances in a chain: each chip works on a different token at once.

### Context

Tokens that fit at once. It sets the KV cache, the RoPE table and the position where the tok/s are measured.

### Chips

How many identical chips the layers are split over. They are cut so the chip that needs the most memory has as little as possible: the first one also keeps lm_head, so it carries fewer layers.

### Maximum clock

Each die runs at its highest frequency according to its measured critical path, or at this one if it is lower.

### Process

Delay of a full adder, area of a NAND2 gate with its wiring, area of an SRAM bit and its periphery, and maximum die size (reticle: 858 mm² in N6). A larger die cannot be made and is discarded.

### Most efficient and balance

From the speed-area frontier, the point with the least area per tok/s is proposed, and the last one where each extra tok/s costs no more than a given multiple of the mean area per tok/s.

## Generation

Chosen with each request; they do not change the hardware.

### Max tokens

How many tokens it generates at most; it also stops at any token on the model's stop list.

### Temperature

0 always picks the most likely. Higher spreads the probability over more candidates.

### Top-k

Among how many of the best candidates it draws; it cannot exceed the sampling unit's candidates.

### Think before answering

With the model's chat template, leaves its reasoning block open; without it, the block arrives closed and it answers directly. It is prompt text.

### Reasoning budget

How many tokens it may reason before the firmware closes the reasoning for it. The opening and closing tokens come from each model's tokenizer.

### MTP depth

How many tokens the multi-token prediction head runs ahead; how many it gets right is measured.

### Fidelity

How much of the chip is simulated: real firmware instruction by instruction, sampling unit, tokenizer in the firmware and image loading by transactions.
