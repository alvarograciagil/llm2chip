# Images on the chip

Qwen3.5 reads images with an encoder of its own (a ViT) that turns each image into tokens the language model reads in place of `<|image_pad|>`. This document describes that encoder as llm2chip reproduces it, what already works and how it is taken to the circuit. It also covers the two image models that run on the chip: Janus-Pro, which draws, and Show-o, which draws and retouches.

## The Qwen3.8-27B encoder

It comes from `vision_config` and `model-vision.safetensors` (BF16 weights, about 460 M parameters).

| Step | What it does | Size in Qwen3.8-27B |
|---|---|---|
| Preprocessing | Resizes to sides that are multiples of 32 px (between 256×256 and 16 M pixels), normalizes to [−1, 1] and cuts 16×16 patches in two identical frames | 1,536 values per patch |
| Patch embedding | A 3D convolution equivalent to one matrix per patch, with a bias | 1,536 → 1,152 |
| Learned position | A 48×48 table interpolated bilinearly to the image's grid | 1,152 per patch |
| 27 blocks | LayerNorm, bidirectional attention with 2D RoPE (half the frequencies follow the row and the other half the column), LayerNorm and an MLP with tanh GELU; all with biases | 16 heads of 72 |
| Merger | LayerNorm and, for each 2×2 group of neighbouring patches, 4,608 → 4,608 → 5,120 with GELU | 1 token per 4 patches |

Patches are ordered in 2×2 blocks, so each token of the language model is the concatenation of four consecutive patches.

In the language model each token has three positions (time, row, column) for RoPE: of the 32 rotated frequencies, those with index 1, 4, 7… (up to 33) follow the row, those with index 2, 5, 8… (up to 30) the column, and the rest the time (`mrope_section` [11, 11, 10], interleaved). Text has all three equal, which is why models without images do not notice the difference. After an image of h×w tokens starting at position p, all its tokens have time p, rows p…p+h−1 and columns p…p+w−1, and the text carries on at p + max(h, w).

## What already works

- **Floating-point reference** (`crates/sim/src/vision.rs`): preprocessing, the whole encoder and the merger, against the real weights.
- **Language model with images** (`Engine::step_embedding`, `set_mrope`, `continue_text_at`): image tokens go in as vectors with their three positions, and the following text carries on where it should.
- **With the chip's arithmetic** (`ExactEngine`, the same): each vector is loaded in the die's format where the embedding row would go, and its rotation is a row of its own in the RoPE table for that cache position, composed from its three positions (time, row and column) according to the M-RoPE sections. On the chip, the host writes those rows into the table the die already reads by position, just as it brings the embedding rows; the text following the image is rotated from its own position. A test compares this with floating point on a toy model with the whole head rotated.
- **Example** `cargo run --release -p sim --example see -- <model> red`: draws a coloured shape, shows it to the real Qwen and asks what colour it is.
- **Estimator:** with "Vision (image analysis)", the area counts all the encoder's weights (blocks, embeddings and merger), and the time to process a 512×512 image on one die's multipliers is shown.

## How it is taken to the circuit

Almost everything is built from operations the die already has:

1. **Patch embedding and merger:** matrix-vector products in the cells, with their biases (`AddWeight`) and GELU from a table (`FN_GELU_TANH`, `FN_GELU`).
2. **LayerNorm:** the mean is a sum with the register of ones, scaled; then the same norm as RMSNorm on the centred vector, and the bias.
3. **Bidirectional attention:** the same attention as the language model's over a store of the image's keys and values, but without a causal mask. It requires going through the encoder layer by layer: first all the keys and values of a block, then all the queries. The firmware orders it with two routines per block (`VIT_QKV` and `VIT_REST`) and a patch index, and keeps each patch's state between blocks in a store like the KV cache.
4. **2D RoPE:** the firmware writes each patch's row and column; the two tables of cosines and sines already exist for the text's RoPE.
5. **M-RoPE in the text:** done in the chip's arithmetic, with rotation rows written by the host for the image's positions (see above).
6. **Image input:** the language model's token routine takes a register as input instead of the embedding, just as it already takes the MTP head's output.

## Status, stage by stage

Each stage is checked against the previous one on a toy encoder (`make_vision_checkpoint` in the tests: width 8 in 2 heads of 4, 2 blocks, 2×2-pixel patches in 2 frames, 2×2 patches per token of width 8, that of the toy language model), with the tests in `crates/sim/tests/vision.rs`.

| Stage | What | Status |
|---|---|---|
| 0 | Toy encoder; the floating-point reference runs it | Done |
| 1 | `compile_vision`: the encoder as a program with its routines (patch embedding, the two of each block and the merger) and three new operations: `LayerNorm`, and reading and storing a patch's row (`LoadRow`, `StoreRow`) | Done |
| 2 | `ExactVision`: the routines in the die's exact arithmetic, in the firmware's order; it differs from floating point by less than 0.5 % and the language model picks the same token with its tokens as with the reference's | Done |
| 3 | Each routine in gates, bit for bit the same as the exact machine: a microprogram with one entry per routine (`routines_microprogram`) and the host writing pixels, positions and rotation rows (`rotation_writes`) | Done |
| 4 | The firmware orders them; instruction-level simulator against gates, from the image to the text | Done: see "The firmware with an image" |
| 5 | The language model's token routine takes the image as input, end to end | On the exact machine: `attach_vision` puts the encoder in the model's program (its buffers and key slots alongside the model's, its own rotation table, number 2), and a microcode entry reads the token from the encoder's output instead of the embedding. It gives the same logits as giving the model the encoder's tokens separately. In gates, with the host ordering the routines, one die encodes the image and the model reads it, with logits bit for bit equal to the exact machine |

How it runs: the host writes, for each patch, its pixels and its learned position already interpolated, and the rotation of its row and column into the table the die reads by position. The embedding routine keeps each patch in a row store. Each block goes twice over all the patches: the first pass stores keys, values and the query; the second attends over all of them (with the position at the last patch) and carries on with the rest of the block. The last block already applies the merger's LayerNorm, which is per patch, and the merger reads four consecutive rows of the store and joins them into one vector.

### Exponents per row

In the real encoder, the patches' state grows from a few units to almost 300 from block 10 on, and after the last block reaches 4,828 in one channel: the ViTs' outlier channel. Queries, keys and values stay below 14. A single exponent for the whole store does not work: below 64 it saturates the outlier channel (440 % error), and below 8,192 it takes away the precision of the other rows (51 % at 16 bits).

So each row of a store keeps its own exponent, as each register does: `StoreRow` copies the mantissas as they are and notes the vector's exponent in a separate register, at the row's position; `LoadRow` copies them back and restores the exponent. These are two new operations of the instruction set (`save_exponent` and `restore_exponent`), a move of the scalar ALU between the exponent register and an element, with no new hardware. When the merger joins four rows, they are aligned to the largest of their exponents; that is why the merger's LayerNorm comes earlier, in the last block: that way a row with the outlier channel does not drown its neighbours' small values.

With the real Qwen3.8-27B encoder and a 256×256 image (256 patches, 64 tokens), the die's tokens differ from floating point by 1.5 % with 16-bit vectors and by 0.9 % with 20 bits (`cargo run --release -p sim --example see_exact -- <model> [bits]`). With everything in the die's arithmetic, encoder and language model, Qwen3.8-27B sees the red disc and answers "Red" (`--example see -- <model> red "<question>" 3 chip`, about 8 minutes).

### In the application

With "Die arithmetic", a model with an encoder carries it inside the chip (`attach_vision`, up to 256 patches; a larger image is reduced until it fits). The image goes through the encoder in that arithmetic and the model reads its tokens from the encoder's output; the application says where it was encoded. In floating point, the host encodes it as before. With "The whole chip", the real firmware receives the image through the host port and orders its routines (see below); the die, which the application simulates as a software engine, encodes and reads the image as in "Die arithmetic". Images are not accepted with the MTP head.

## The firmware with an image

A chip that carries the encoder says so in its description (`INFO_VISION`, `add_vision`): how many patches it takes, the token the image replaces, the blocks, where the pixels, the position and the rotation row of each patch go, and where the text's rotation table is. The CPU can write the vector memory: it writes the address to `REG_VECTOR_ADDRESS` and the value to `REG_VECTOR_DATA`, which goes in through the die's load port one cycle later, with the die stopped and no logits on their way. Only a chip with vision has it (`DieFeatures::host_writes`); text-only chips do not change.

The protocol, before the prompt (`vision_chip::picture_bytes`, each number in 4 little-endian bytes):

1. Number of patches (0 if there is no image), rows and columns of the grid.
2. For each patch: the exponent and the elements of its pixels, already normalized and in the die's format; the same for its learned position, already interpolated; and its row of the encoder's rotation table.

This is the host's part, like a camera driver's. The firmware writes each patch and runs its embedding, then each block twice over all the patches, and waits for the prompt. When the image's token arrives, for each image token it runs the merger, composes its M-RoPE rotation and reads it instead of an embedding.

**Rotations without floating point on the CPU.** The rotation row of an image token at (time p, row p + r, column p + c) takes each frequency from the table row of the position it needs: that of p, of p + r or of p + c, according to the M-RoPE sections. The firmware composes it by copying those elements with the load controller from the chip's image, where the table is whole and never written. The following text goes max(h, w) positions after the start of the image, not h × w: before each token in cache slot s, the firmware brings from the image row s − (h × w − max(h, w)). As the source is always the image, rewriting rows of the die's table does not spoil those read later.

It is checked in `crates/sim/tests/vision_chip.rs`, with the toy model and an image of 8 × 4 patches (2 × 4 tokens, rows different from columns), text before and after and three generated tokens:

- The firmware, on the die's exact machine, writes the same tokens as the host ordering the same routines and writing the rotations computed from the three positions.
- The whole chip in gates gives the same bytes in the same cycles as the instruction-level simulator (slow test, 1.3 million cycles in a minute and a half).
- The firmware, on the software die the application uses, writes the same tokens with the same logits as the host calling `step_image` with the three positions.

The tests catch a badly composed M-RoPE rotation or text that is not shifted: the logits change.

**Unquantized weights in the cells.** The real encoder's weights come in BF16 and the toy one's in floating point. Without a weight format, such a matrix goes into the cells as one mantissa of the vectors' width per weight (16 bits), and the cells must hold weights that wide. Before, 8-bit cells truncated them silently and only the exact machine multiplied correctly; now the `Layout` gives an error asking for a weight format or wider cells. With an affine format, the cells keep the codes, as described under "The real Show-o on the addressed die". The language model does not notice, as it arrives already quantized.

## Drawing with Janus-Pro

Janus-Pro-1B also generates images: its language model writes 576 tokens from a codebook (a 24 × 24 grid), guided away from what it would draw without the text, and a convolutional decoder turns them into pixels. `cargo run --release -p sim --example draw -- <folder> "<description>" output.bmp [temperature] [seed]`.

With `EXACT=16` (and in the application, with the model loaded in the chip's arithmetic), the two copies of the language model, with and without text, and their two image heads run in the die's integer arithmetic. The compiler adds to Janus' program an image head (a `CodeHead`, the codes being the image's) with two lists of operations:

- **Scores:** from the final state, `proj_out` with its bias, GELU and `vision_head` with its bias, one score for each of the codebook's 16,384 codes.
- **Input:** the code read from `generation_embeddings` like an embedding row, `fc1` with its bias, GELU and `hidden_layers.0` with its bias: the vector that goes in instead of the next token's embedding. It is the same for both copies, so it is computed once.

The host keeps the guidance mix (two contexts, two caches), sampling and the pixel decoder. A test with a tiny Llama and random heads checks that the scores and inputs come out of the chip as in floating point, within 2 %.

With "a red apple on a wooden table", seed 7 and temperature 1, floating point takes 387 s and the chip's arithmetic 543 s, and both draw a red apple on wood. They are not the same picture: at temperature 1, a minimal difference in one logit changes which token is chosen, and from there the drawing takes another path. A test checks that the hidden state the image head reads comes out of the chip as in floating point, within 2 %.

## Retouching images with Show-o

Janus-Pro draws from a text, but does not edit. Show-o (`showlab/show-o`, 1,300 M parameters) does: it is a single transformer, Phi-1.5, that reads text and fills in images made of MAGVIT-v2 codes (`showlab/magvitv2`), a convolutional tokenizer. To retouch a region, its codes are masked and the transformer fills them in over a few steps. At each one it keeps the codes it is surest of, guided by the text and away from what it would do without it.

`crates/sim/src/showo.rs` reproduces it in floating point, as the original code runs it.

- **MAGVIT-v2:** a convolutional encoder (residual blocks with GroupNorm and swish, four halvings and attention in the middle) takes a 256 × 256 image to 16 × 16 vectors of 13 values. Each value gives one bit by its sign, and the 13 bits make a code out of 8,192. The decoder goes the other way. Encoding and decoding an image gives about 30 dB of PSNR (`crates/sim/tests/showo.rs`).
- **Phi-1.5 with Show-o's changes:**
  - attention and MLP in parallel on the same normalized input;
  - LayerNorm on each query and key head before rotating;
  - rotation on half of each head;
  - mixed attention: the text, causal; the image, whole in both directions and over all the text.

  The text is computed once and its cache serves every step.
- **Filling in:**
  1. At each step, each unknown code is drawn from the model's probabilities, with the guidance `(1 + g)·with text − g·without text`.
  2. The least sure stay masked, with Gumbel noise whose temperature decreases step by step. Their number follows a cosine.
  3. The region to retouch is taken to the code grid with bicubic sampling at their centres and a threshold of 0.5, as in the original.

With 16 steps, an image takes about 4 minutes on the processor:

- from "a red apple on a wooden table", Show-o draws the apple on the table;
- retouching the apple's region with "a green apple on a wooden table" and guidance 5, the apple turns green and the table and background stay as they were.

```
cargo run --release -p sim --example inpaint -- <folder> retouch <image.bmp|.ppm> x0,y0,x1,y1 "<text>" output.bmp [steps] [guidance] [seed]
cargo run --release -p sim --example inpaint -- <folder> draw "<text>" output.bmp [steps] [guidance] [seed]
cargo run --release -p sim --example inpaint -- <folder> reconstruct <image> output.bmp
```

The folder holds Show-o's `pytorch_model.safetensors` and `config.json`. Inside go `magvitv2` (its weights and configuration) and `phi`, with Phi-1.5's tokenizer.

### In the application

The **Images** screen uses the Show-o it finds in the model folders:

- **With no image,** it draws a new one from the text.
- **With an image,** its central square is cropped and you drag over it to mark the region that changes. The region must cover at least one code, 1/16 of the side.
- **Keep retouching** takes the result as the new starting image, to change another region.

The panel sets the steps, the guidance and the seed; the same seed, text and region give the same result.

The engine runs it in the background (`/api/retouch`), one job at a time, with its progress step by step, and it can be stopped between steps.

### In the floorplan

Drawing is a part of the chip of its own, apart from vision: vision *reads* images (the encoder above, the *Vision (image analysis)* option), drawing *makes* them. The *Draws images (text to image)* option (`draw_images`, off by default) turns it on. With it off, nothing of drawing is in the project: the Images screen is not shown and no die is counted.

With it on, the floorplan draws a die of its own in the row below the language model's, after the voice dies, coloured *Image drawing* in the *Part* layer and dashed as a reference die. The estimate adds it to the *Total area*, with *Drawing* apart, and its leakage as a metric of its own (not in the power while generating text). The reference die (`arch::drawing`) is Show-o's transformer, guided in the die, on the die its firmware addresses (see "The real Show-o on the addressed die"). It has 8-bit affine weights in groups of 64, on 16 units of 64 lanes with 4,096 cells, measured in N6 as the voice planner measures its stages:

| | mm² |
|---|---|
| Logic | 6.3 |
| Cell memory | 314.4 |
| Vector memory | 109.6 |
| Microcode | 1.8 |
| **Die** | **432.1**, at 1 GHz |

Two things stay off the die, and the estimate says so:
- **The embedding table,** 120 M elements (240 MB at the die's 16 bits). It sits in external memory, which the floorplan draws as *External DRAM*, and the loader brings each token's row.
- **MAGVIT-v2** runs on the host. It keeps whole maps of the picture in the vector memory: about 1,500 mm² for the encoder and 1,750 mm² for the decoder, more than a reticle (858 mm²). On the die it would need its maps in external memory, a tile at a time.

`cargo run --release -p sim --example draw_dies -- <Show-o folder> [shapes]` measures both on other shapes.

### In the chip's arithmetic

`crates/sim/src/showo_chip.rs` takes Show-o's transformer to the chip's operations and runs it on the die, in the order the host or the firmware gives them:

- **The text** goes in as normal tokens, each seeing the previous ones. The padding on the left is not sent: nothing reads it.
- **The image** is seen whole in both directions, so it goes through each layer in two passes over its rows, like the vision encoder:
  1. each row stores its key and value in the layer's cache, after the text's, and its query;
  2. each row attends over the text and all the image's rows and does the rest of the layer.

  Attention and the MLP run in parallel on the same normalized input, which is recomputed in the second pass.
- **Phi's pieces** are operations the chip already had:
  - LayerNorm with a bias, also per head on queries and keys;
  - RoPE on half of each head;
  - GELU with tanh;
  - biases on every projection.
- **The scores** come from the output's rows for the codes, with their biases.

Checked with one pass over the same masked image, against floating point:

| Model | Largest difference | Same best code |
|---|---|---|
| Toy Show-o (2 layers) | 0.06 % | 8 of 8 |
| Real Show-o, 16-bit vectors | 1.04 % | 61 of 64 |

With the real model, by weight format (`cargo run --release -p sim --example showo_formats -- <folder>`):

| Weights | Largest difference | Same best code | Mean KL (nats) | Probability of the chip's best, against the best | Seconds per pass |
|---|---|---|---|---|---|
| 16-bit integers | 1.04 % | 61 of 64 | 0.0007 | 99.9 % | 156 |
| 8-bit affine, groups of 64 | 1.32 % | 61 of 64 | 0.0010 | 99.9 % | 77 |
| 4-bit affine, groups of 32 | 7.36 % | 37 of 64 | 0.066 | 90.5 % | 73 |

In floating point a pass takes 6 s. The sampler cares about the whole distribution, not just the best code, and the KL divergence says it barely moves with 8 bits. With 4 bits it shows.

These run in `crates/sim/tests/showo_chip.rs`. The test with the real model is slow and runs separately. With the same seed, MaskGIT's sampling diverges as soon as one code differs, so the comparison is made pass by pass and not on the final image. (The seconds per pass in this table were measured on the exact machine with 16-bit mantissas per weight; with the weights in 8-bit cells, a pass takes about 42 s.)

On the Images screen, the "As on the chip" option retouches as the chip would: the addressed die (see "The real Show-o on the addressed die"), with the 8-bit affine weights in its cells, and the firmware's loop. With guidance there are two passes per step, about 42 s each, plus about 25 s to prepare the die: a 4-step retouch takes 360 s and uses 4.3 GB of memory (`showo_fill_chip`).

### MAGVIT-v2 in the chip's arithmetic

`crates/sim/src/magvit_chip.rs` takes MAGVIT-v2's encoder and decoder to the chip's operations, also on the exact machine driven from the host.

- **Feature maps** are row stores, one row per pixel with all its channels, with a border of zero rows. That way a pixel's eight neighbours are rows at fixed distances, whatever the pixel. A 3 × 3 convolution is a product of the nine rows laid in a row (`LoadRows`) by the kernel reordered the same way. Halving reads every other pixel's neighbours, and the border does the padding.
- **GroupNorm** needs the mean and spread of each group over the whole map:
  1. each pixel writes its sums per group and those of their squares, with a product by a matrix of ones in the cells' wide accumulators;
  2. those rows are added in pairs, one level at a time, until one holds the totals;
  3. the `GroupNorm` operation normalizes each pixel with them.
- **The attention in the middle** goes in two passes over the pixels, like Show-o's image.
- **Doubling the side** copies each row to its four places.

| Piece | Toy | Real MAGVIT-v2 | Real time |
|---|---|---|---|
| Decoder | 62.5 dB PSNR against floating point | 49.6 dB | 195 s |
| Encoder | 208 of 208 bits equal | 3,306 of 3,328 bits (99.3 %), 236 of 256 codes | 254 s |

In the encoder, values close to zero can come out with the other sign, so some bits change.

The decoder reads each code's ±1 values from a table, like an embedding, so all its work is the chip's routines. Its schedule (`decoder_calls`) is followed alike by the exact machine and the gates. The toy one runs on the die's gates and gives the same colours as the exact machine, bit for bit: 664 routines and 2.65 million cycles (`magvit_s_decoder_on_the_die_s_gates_gives_the_exact_machine_s_pixels`). That checks `LoadRows`, `GroupNorm`, the attention, the upsampling and the shortcut in gates. The routines are ordered by the test itself, acting as the firmware. The toy encoder also gives in gates the same values as the exact machine, halving convolution included: 2.36 million cycles (`magvit_s_encoder_on_the_die_s_gates_gives_the_exact_machine_s_values`). The host writes each pixel's colours into the die, idle between routines, as the firmware does with an image's patches.

On the Images screen, "Also MAGVIT-v2 on the chip" does the whole retouch in the chip's arithmetic. A one-step retouch without guidance took 599 s: encoding, one pass of the transformer and decoding.

### On the die's gates

The toy Show-o runs on the die's gates, routine by routine, and gives the same scores as the exact machine of the same die, bit for bit: 3.1 million cycles (`show_o_s_routines_on_the_die_s_gates_give_the_exact_machine_s_scores`).

The scope of this test:
- **Who orders the routines:** the test itself, acting as the firmware. The RV32 does not take part.
- **The text fills exactly its maximum length,** so each token goes in the cache slot matching its RoPE position. With a shorter text, the positions shift and the rotation rows must be rewritten: the exact machine does it with `place_rotation`, and on the chip the firmware does it, as it does with images' M-RoPE.

Testing it found a bug: the embedding table was sized from the output's vocabulary. In Show-o the output has 8,192 codes and the table 58,498 rows. Now the program states its table's rows (`embedding_rows`).

### With the chip's firmware

The firmware has a Show-o mode. The chip's image carries its description (`ShowOInfo`): the image's rows, the layers, where the scores and the rotation table are. The host sends the text, the position of its first token and the image's tokens, and the firmware:
1. puts each token's rotation in its place in the cache, copying from the image the row of its real position: a text shorter than the maximum starts further on;
2. runs the text, the two passes of each layer and the scoring of each row;
3. returns, for each place in the image, its exponent and the mantissas of its scores.

On the CPU at instruction level, with the die in software and a short text, it returns the same scores as the host ordering the same routines on the same machine: 3.66 million CPU cycles (`the_firmware_runs_show_o_s_pass_as_the_host_ordered_it`).

### MaskGIT on the chip's CPU

The chip also fills in the image by itself. The firmware runs MaskGIT's steps with `maskgit-rt`, a library without dynamic memory that the host uses too, so both give the same codes. Everything is in integers:
- **Each masked code's draw** follows the softmax of its scores: weights of `2^x` in fixed point, from a table, and a xorshift generator.
- **Its confidence** is its log-probability, with `log2` from a table, plus Gumbel noise that cools down step by step.
- **The least sure** stay masked. The schedule of how many remain at each step and the temperature are sent by the host, computed as in Show-o.
- **The die does the guidance,** with two more routines:
  1. in the pass with the text, one stores each row's final state, already normalized;
  2. in the pass without the text, another combines it, `(1 + g)·with text − g·without text`, and scores the result.

  Since the code head is affine, this is the same as combining the scores: `(1 + g)·(W·a + b) − g·(W·c + b) = W·((1 + g)·a − g·c) + b`. A row's state takes a quarter of its scores (2,048 against 8,192 in the real Show-o), and that is what lets the store fit in the die's addresses. The CPU only reads the combined scores.

Checked:
- **Against floating point:** confidences within 0.02 nats and draw frequencies within 1 % (`crates/sim/tests/maskgit_rt.rs`).
- **On the chip at instruction level:** the toy Show-o fills in an image of 16 places, half of them known, with guidance and 4 steps, in 24.1 million CPU cycles. It gives the same codes as the host ordering the same machine (`the_firmware_fills_an_image_as_the_host_ordered_it`).
- **The whole chip in gates,** CPU, firmware, loader and die, fills in the toy image with two steps. It sends the same bytes in the same cycles as at instruction level: 12.45 million cycles, 1,156 s (`the_whole_chip_fills_an_image_in_gates_as_at_instruction_level`).
- **With the real Show-o:** the same loop (`fill_on`), on the addressed die with the guidance in the die and 8-bit affine weights in its cells, retouches the red disc with "a yellow lemon" in 4 steps and 360 s (`cargo run --release -p sim --example showo_fill_chip`). A lemon comes out, with its texture.

### The real Show-o on the addressed die

The firmware addresses the die's memory with 32 bits: the register on top and the element below. At first the real Show-o did not fit, for three reasons, now solved:
- **The embedding table** has 120 million elements (27 bits of element). Now it stays in the image from the start (`Layout::with_cells_table_outside`, `VmEngine::new_table_outside`). Before each routine that reads it, the loader brings the token's row, as the firmware already did with stories260K. In Show-o both the text and the input of each image row read it, so the firmware also loads it before that routine. The host does the same in `fill_on_die`.
- **The image's row stores** were assumed to be as long as the vocabulary, because they are indexed with the same field as the token. With the table outside, that field only reaches the image's rows.
- **The guidance store** kept each row's scores (258 × 8,192, 22 bits). Now it keeps the final state (258 × 2,048, 20 bits); it gives the same, see above.

`showo_chip::addressed_die` sizes the die: each memory as large as the routines use, and the depth of the cells from their number. If register and element do not fit in 32 bits, it gives an error. With 4,096 cells the real Show-o takes 11 bits of register and 20 of element, 31 in all; with 1,024 cells, exactly 32 (`the_real_show_o_fits_the_addressed_die_with_its_table_in_the_image`).

Checked:
- **Toy Show-o:** with the table in the image, the firmware fills in the image with the same codes as the host (`the_firmware_fills_an_image_with_the_table_in_the_image`). With the previous firmware, which did not load the image's row, the test fails.
- **Real Show-o:** in one pass over the addressed die with 4,096 cells, the best code matches floating point in 61 of the 64 masked places (`cargo run --release -p sim --example showo_addressed -- <folder> 4096 pass`).

The cells keep the weights in the design's format even when they come in floating point, as Show-o's do. The recoder rewrites them: for each weight, its 8-bit code, and for each group, its scale and bias, just as with a checkpoint that comes quantized. The gates give the same as the exact machine, bit for bit, with the toy Show-o in 8-bit cells (`floating_point_weights_in_an_8_bit_format_go_into_8_bit_cells`). With the real Show-o, the pass over 8-bit cells gets the same 61 of 64 best codes and takes about 46 s, against about 210 s with a 16-bit mantissa per weight: the exact machine adds the integer codes per group instead of each whole weight.

### Quality of the integer fill

`showo_fill_compare` makes the same retouch (the red disc, guidance 1.75, 4 steps) in three ways:
1. all in floating point;
2. the model in floating point with the firmware's integer MaskGIT, the scores rounded to 16-bit mantissas;
3. the chip: integer MaskGIT and the transformer in the chip's arithmetic, with 8-bit affine weights.

It measures the colour inside the retouched square. With "a yellow lemon" and 24 seeds:

| Way | Seeds | Yellow, `min(R, G) − B` (mean ± error) | Median | Saturation |
|---|---|---|---|---|
| Floating point | 24 | 95.1 ± 4.1 | 91.3 | 0.635 |
| Integer MaskGIT | 24 | 86.9 ± 2.3 | 86.3 | 0.619 |
| Chip | 4 | 84.5 ± 5.6 | 81.2 | 0.622 |

With three other texts, 8 seeds each (mean colour ± error):

| Text | Way | R | G | B | Saturation |
|---|---|---|---|---|---|
| "a green apple" | Floating point | 213.6 ± 3.5 | 187.1 ± 7.2 | 99.3 ± 8.5 | 0.601 |
| | Integer MaskGIT | 208.6 ± 4.2 | 182.3 ± 8.5 | 99.1 ± 5.8 | 0.601 |
| "an orange" | Floating point | 251.3 ± 0.6 | 164.5 ± 2.2 | 97.1 ± 1.5 | 0.616 |
| | Integer MaskGIT | 251.0 ± 0.8 | 160.7 ± 2.3 | 89.3 ± 2.8 | 0.647 |
| "a blue ball" | Floating point | 146.0 ± 4.1 | 139.9 ± 3.8 | 175.7 ± 4.8 | 0.575 |
| | Integer MaskGIT | 145.8 ± 3.7 | 142.9 ± 2.2 | 182.6 ± 4.1 | 0.602 |

What the figures say:
- **The chip follows the integer MaskGIT seed by seed** (100.7 against 93.4; 80.8 against 77.2; 75.1 against 76.9; 81.6 against 79.2). The 8-bit weights and the die's arithmetic do not change the result appreciably.
- **Between the integer and the floating-point MaskGIT,** the difference is within the seeds' noise, and saturation comes out equal or slightly higher with integers. With the first 8 lemon seeds it looked clear (105.6 against 84.7), but with 16 more it is 89.9 against 88.0. What does show is more spread in floating point: a few seeds come out very yellow, up to 143.9. The first impression that the chip came out paler came from the seed chosen.
- **Looking for the cause turned up a real bug:** with flat distributions, the draw overflowed when multiplying in 64 bits and never chose the upper half of the 8,192 codes. With guidance the scores are sharp and it did not happen, so the figures do not change. Now it multiplies in 128 bits, and `a_flat_draw_over_many_codes_is_even` checks it.

Still to do:
- measure the quality on real photographs, not only on a synthetic disc.
