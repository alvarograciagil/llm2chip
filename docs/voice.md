# Voice on the chip

A chip that listens and speaks: speech goes in, the language model answers, and the answer comes out as speech. This document describes how llm2chip takes that on, which models it uses, what each part needs from the die, and where every piece lives in the code. It also says what is done and what is still to do.

The guiding rule is the same as for images: **the die does not know what voice is.** It gains general operations and capabilities that any model may use. Speech recognition and speech synthesis are just more models compiled onto dies, and the pipeline that joins them is a general way of running several programs on several dies. Removing voice from the project should only remove files.

## The pipeline

```
 microphone ──► speech to text ──► language model ──► text to speech ──► speaker
  16 kHz         (Moonshine)        (any model of      (OuteTTS, a Qwen3     24 kHz
  samples        text tokens        the catalogue)      that writes audio
                                    text tokens         codes, + DAC decoder)
```

Each stage is a program of its own on dies of its own. Between stages only text travels, as bytes: a few dozen a second, far below what any link carries. Each chip already reads and writes UTF-8 bytes through its host port, with its own tokenizer, so a recogniser and a language model with different vocabularies join without translating token ids. Splitting the stages over dies costs nothing measurable in latency, and each stage's dies can have their own shape.

In a conversation what matters is not tokens per second but **how long until the first sound**, which should stay below about half a second. It adds up three things:

1. **The end of the utterance:** detecting silence, then the transcription. Moonshine's encoder attends over the whole clip and its GroupNorm averages over all of it, so it starts when the utterance ends; its cost follows the clip's length instead of a fixed 30-second window.
2. **The first token of the language model:** the prompt's prefill. The optimiser already has a limit for it; in a voice design it becomes the main goal.
3. **The first sentence of speech:** synthesis can start as soon as the model writes its first sentence, while it keeps writing.

Speech needs few tokens: DAC codes come at 75 frames per second (the 24 kHz rate over the decoder's upsampling of 8 × 5 × 4 × 2 = 320), two codebooks per frame at 1.5 kbps. That is 150 codes per second, against thousands of tokens per second on the dies. A voice die is sized for latency, not for speed.

## The models

| Stage | Model | What it is | Size | Licence |
|---|---|---|---|---|
| Speech to text | [Moonshine](https://huggingface.co/UsefulSensors/moonshine-tiny) tiny / base | Encoder–decoder transformer over the raw 16 kHz waveform | 27 M / 61 M parameters (108 / 246 MB in F32) | MIT (tiny's card) |
| Text to speech | [OuteTTS-1.0-0.6B](https://huggingface.co/OuteAI/OuteTTS-1.0-0.6B) | A Qwen3-0.6B (`model_type: qwen3`, 28 layers, hidden 1,024, vocabulary 157,760) that writes the codec's codes as tokens | 0.6 B parameters (BF16) | Apache-2.0 |
| Codes to sound | [DAC.speech v1.0](https://huggingface.co/ibm-research/DAC.speech.v1.0) | The decoder of the Descript Audio Codec, 24 kHz, fine-tuned for speech | 296 MB per checkpoint (`.pth`) | CDLA-Permissive-2.0 |

Why these:

- **Moonshine takes the waveform, not a spectrogram.** Whisper needs a log-mel spectrogram, that is, an FFT on the host or in the firmware, and always processes 30-second windows. Moonshine's front end is three convolutions over the samples, which the die already does, and its length follows the audio. Its blocks are the usual ones: LayerNorm without bias, attention with partial RoPE (90 % of each head in tiny, 62 % in base), GELU in the encoder's MLP and a gated SiLU in the decoder's. Eight heads, all with their own keys and values.
- **OuteTTS writes speech as ordinary tokens.** Its language model is a plain Qwen3, a family the compiler already supports, and the codes are rows of its vocabulary: `<|c1_N|>` and `<|c2_N|>` for the two codebooks (`<|c1_0|>` is token 151,669), a pair per frame, grouped by word with a few tokens of their own (the word's duration, energy and pitch). The language model needs no new head and no new field: only the decoder that turns codes into sound is new.
- **DAC's decoder is only convolutions.** No attention and no recurrence: a convolution in, four blocks that each upsample (a transposed convolution) and refine (three residual units with dilations 1, 3 and 9), and a convolution out with tanh. That is the same kind of work as MAGVIT-v2's decoder, which the die already runs in gates, but in one dimension.

OuteTTS uses DAC.speech's 1.5 kbps checkpoint (`weights_24khz_1.5kbps_v1.0.pth`), with two codebooks at 75 frames a second. Its weights come as a PyTorch pickle and must be converted to safetensors by a script in `tools/convert`, as `llama2c.py` does for llama2.c models, loading it with `weights_only` so that no code in the file runs.

## What each stage needs from the die

### Moonshine's encoder

The encoder is a store of rows, like the image encoder: one row per frame, routines that go through it block by block, the keys and values of a block kept in a store and attended bidirectionally. The host writes the samples, 64 to a row, so the front end is:

| Layer | In Moonshine | On the die |
|---|---|---|
| `conv1`: kernel 127, stride 64, 1 → hidden, tanh | One output per 64 samples, reading 127 | `LoadRows` of rows `t` and `t + 1` (128 samples, the last weight zero), a product, `SoftCap`'s tanh |
| `GroupNorm` with one group | Mean and variance over all frames and channels | `GroupNorm`, as in MAGVIT-v2 |
| `conv2`: kernel 7, stride 3, hidden → 2·hidden, GELU | | `LoadRows` with offsets 0…6, the routine called on every third row |
| `conv3`: kernel 3, stride 2, 2·hidden → hidden, GELU | | The same, offsets 0…2, every other row |

A strided one-dimensional convolution is the strided two-dimensional one of MAGVIT-v2 with a single row of pixels: the firmware passes the input row as the token and stores the output at its own index. Together the strides make one frame every 384 samples, about 42 per second.

### Moonshine's decoder

The decoder is a causal language model with one more attention per layer: **cross-attention**, whose queries come from the token and whose keys and values come from the encoder's frames. It needs no new operation and no new field:

- **The keys and values:** each encoder frame, once the encoder is done, goes through every layer's cross-attention key and value projections, and `KvAppend` stores them in a key slot of their own, at the frame's index. The routine is called once per frame, as the image encoder's are.
- **The attention:** `AttnSum` over that slot, in a routine of its own that the host runs with the position at the last frame. An attention reads as far as its position, so it reads every frame, as the image encoder's second pass already does. Each decoder layer is then two routines: self-attention at the token's position, and cross-attention with the MLP at the last frame's.
- **Positions:** RoPE in self-attention only, partial, as Qwen3.5 already has (`rotary` smaller than the head). Moonshine rotates adjacent pairs and the die the two halves of the rotated part: the rows of every query and key projection are reordered alike as they are read, which leaves their products unchanged.
- **The gated MLP:** `fc1` makes both halves in one product, `Slice` separates them, then SiLU and the product, as the other models' MLP.
- **LayerNorm without bias:** a bias of zeros is added as the weights are read.

**One limit:** the GroupNorm of the front end divides by how many values the first convolution makes, a constant of the program. A chip is built for clips of one length, as MAGVIT-v2's is for pictures of one side. Taking any length would need the host to write that count, a small change left for later.

### OuteTTS's language model

Nothing new: it is a Qwen3 and compiles as one. The code tokens are rows of its vocabulary, so it samples them with its own head. The host or the firmware recognises them by their range and passes them to the decoder.

### DAC's decoder

Sound is kept in stores of rows, one per time step with all its channels, each with a border of 27 zero rows on each side: as far as a residual unit's widest convolution reaches (three taps of dilation 9). That way every convolution, padded or dilated, reads its taps at fixed offsets.

| Layer | On the die |
|---|---|
| The codes' vectors: each codebook's 8-value row, projected to 1,024 by a 1×1 convolution, the codebooks added | A routine per codebook, with the code as its token: `EmbRead`, a product, `Add` |
| Convolution with kernel 7 and padding 3 | `LoadRows` with offsets 0…6 from three rows before, a product with the kernel laid out the same way; a second routine stores the result at its own row |
| Transposed convolution with rate s, kernel 2s and padding p = ⌈s/2⌉ | Output row n comes from input rows q − 1 and q, q = (n + p) div s, through a kernel that depends only on the phase (n + p) mod s: a routine per phase, each an ordinary convolution of two rows |
| Dilated convolution (1, 3, 9) | `LoadRows` with offsets spaced by the dilation |
| Snake: `x + sin²(αx) / α`, α per channel | `MulWeight` by α, `Sin`, its square, `MulWeight` by 1/α, `Add` |
| Final tanh | `SoftCap`'s tanh |

Weight normalisation (`weight_norm`) is folded into plain weights when the checkpoint is converted (`tools/convert/dac.py`, which reads the pickle with `weights_only`, so no code in it runs).

**The sine.** In the real decoder Snake's argument `αx` reaches 108 radians on a short sentence: seventeen turns. The die's function tables take inputs in [−8, 8), and the alignment that brings a value to that range saturates. Two small additions fix it, both built only on dies whose microprogram uses them:

- **`FN_SIN`, a ninth table:** `sin(π·x/8)`, one period over the table's whole range of 16. `Sin` scales radians by 8/π first. The function field becomes one bit wider and the table memory twice as large on the dies that use it.
- **`wrap_left`, one control bit:** a mux beside the left shifter's saturation that keeps the shift's low bits instead. With it, the alignment before `FN_SIN` drops whole periods as it goes, which is exactly the argument's place in the period, with every fraction bit kept.

Language-model dies do not use either, so their microprograms, gates and clock do not change. Doing the same with the die as it was would need `round(αx / 2π)` and a subtraction, and block floating point subtracts at the larger exponent, the integers', which loses the fraction.

Two new operations go with it, microcode only: `Sin` (any argument, in radians) and `MulWeight` (a product by a stored vector, element by element, as `AddWeight` adds one).

**How wide the vectors must be.** Snake runs 29 times between the codes and the sound, and each needs the fraction of the period of an argument as large as 108, while the row shares one exponent across up to 1,536 channels. On a short sentence ("Hello, I am a chip."), against floating point:

| Vectors | Difference below the signal | What Moonshine hears in it |
|---|---|---|
| 16 bits | −4.7 dB (as loud as the speech) | "Hello, I am a chiff." |
| 20 bits | 27.0 dB | "Hello, I am a chip." |
| 24 bits | 32.2 dB (the 12-bit tables' limit) | "Hello, I am a chip." |

A die that speaks needs vectors of 20 bits or more; one that listens does well with 16.

## Several programs on several dies

Until now all the dies of a design ran parts of one program: layers in sequence, heads in parallel, or both. Voice needs dies that run **different programs**: some the speech recogniser, some the language model, some the synthesiser. That capability is general:

- **A system is a list of stages** (`sim::system`), each one a program with its own dies, shape and split. Its area is the sum; its speed is each stage's; its latency is the sum along the chain of how long each stage takes to give the next what it needs to start: the whole utterance for the recogniser, the first sentence for the synthesiser (`System::first_output_seconds`).
- **A stage's cycles come from its microprogram.** A language model's stage comes from its sizing (`Stage::from_candidate`), and the recogniser's from the cycles of its own routines (`MoonshineChip::stage`), which on the toy checkpoint are the gates' cycle for cycle.
- **Between stages only text travels, as bytes.** In the simulation the host passes them on. In hardware, one chip's output port feeds the next one's input port: the chips already speak bytes, so the handoff needs no new port, and the vectors never leave a stage.
- **Other uses:** speculative decoding with a small model drafting tokens that a large model checks (the large one checks several at once, so the answer comes faster); a model that reads images whose encoder runs on dies of its own; or several small models in a chain.

Speculative decoding with a separate draft model is not built yet. The MTP head (`DraftProgram`) is its cousin inside one model.

## Where each piece lives

Voice adds files; the core only gains general things.

| Piece | Where | Kind |
|---|---|---|
| `FN_SIN` table and `wrap_left` bit | `crates/netlist` (die and assembler) | General: any periodic function, built only where used |
| `Sin` and `MulWeight` operations | `crates/compiler`, `crates/sim/src/lower.rs`, `crates/sim/src/exec.rs` | General |
| Programs with parts (`Program::parts`) | `crates/compiler` | Refactor: what were `image` and `vision` |
| Systems of stages | `crates/sim/src/system.rs` | General: several programs on several dies |
| Moonshine: floating-point reference | `crates/sim/src/moonshine.rs` | Voice |
| Moonshine on the die: its program, the host's routines, the exact machine | `crates/sim/src/moonshine_chip.rs` | Voice |
| Sound in and out, WAV | `crates/sim/src/audio.rs` | Voice |
| Moonshine against Transformers, and on the chip | `tools/golden/moonshine.py`, `crates/sim/examples/listen.rs` | Voice |
| Moonshine's tests | `crates/sim/tests/moonshine.rs` | Voice |
| Systems' test | in `crates/sim/src/system.rs` | General |
| DAC: reference and on the die | `crates/sim/src/dac.rs`, `crates/sim/src/dac_chip.rs`, `crates/sim/tests/dac.rs` | Voice |
| DAC weights to safetensors | `tools/convert/dac.py` | Voice |
| OuteTTS and DAC in Transformers, and DAC against it | `tools/golden/speak.py`, `crates/sim/examples/dac_check.rs` | Voice |
| Demo, a WAV in and a WAV out (planned) | `crates/sim/examples/talk.rs` | Voice |

## The system, sized

`cargo run --release -p sim --example voice_system -- MOONSHINE LLM OUTETTS DAC` sizes the four stages in N6 with the optimiser's own counts, each on a die of its own, every number from the stage's microprogram: a three-second question, an answer whose first sentence is 12 tokens, and the speech decoded half a second at a time. Weights are 8-bit codes.

| Stage | Die | Silicon | First output | Then |
|---|---|---|---|---|
| Speech to text (Moonshine-tiny) | 1 unit × 32 lanes, 2,048 cells | 88 mm² | 16.7 ms after the question | 6,118 tokens/s |
| Language model (Qwen3-0.6B) | 16 units × 64 lanes, 65,536 cells | 508 mm² | 5.0 ms after a 40-token prompt | 4,068 tokens/s |
| Speech writer (OuteTTS-1.0-0.6B) | the same | 510 mm² | 5.0 ms | 4,063 tokens/s |
| Sound decoder (DAC, 20-bit vectors) | 1 unit × 64 lanes, 2,048 cells | 224 mm² | 233 ms for half a second of sound | about 4 such chunks a second |

**Four dies and 1,330 mm²; the first sound comes 286 ms after the question ends**, well inside the half second a conversation needs, and every stage keeps ahead of the speech it feeds: the writer needs 180 tokens a second for real time and makes 4,000.

The two speech dies are routines over rows, which keep whole maps in vector memory (59 and 153 mm² of them) and are sized here with one unit: on a die of several, every unit holds a copy. Decoding the sound a few frames at a time instead of half a second at a time would shrink the decoder's maps; it is not done. The sine's table and `wrap_left` are left out of the decoder's logic count (a 4,096-word table and a mux).

## The demo

```bash
cargo run --release -p sim --example talk -- MOONSHINE_DIR LLM_DIR OUTETTS_DIR DAC_DIR question.wav answer.wav [die 20]
```

The four folders are `UsefulSensors/moonshine-tiny`, any language model of the catalogue (Qwen3-0.6B here), `OuteAI/OuteTTS-1.0-0.6B`, and DAC.speech converted with `uv run tools/convert/dac.py weights_24khz_1.5kbps_v1.0.pth DAC_DIR`. With macOS's `say` asking "What is the capital of France?", in floating point:

1. Moonshine hears "What is the capital of France?" (0.8 s).
2. Qwen3-0.6B answers "The capital of France is Paris." (7 tokens).
3. OuteTTS writes 524 tokens, 3.04 seconds of speech, and DAC decodes them; Moonshine, listening to the answer, hears "The capital of France is Paris."

With `die 20` every stage runs in the die's arithmetic with 20-bit vectors, and the turn goes the same way: the same question heard, the same answer, 530 tokens of speech (3.08 seconds) that Moonshine hears as "The capital of France is Paris." On a computer that takes about five minutes, since the simulation goes integer by integer; on the chip each stage is a program on dies of its own (see the system, sized, above).

**On an FPGA.** None of the voice models fits one today. The chip keeps its weights in its cells' memory, and Moonshine-tiny alone has 27 million (about 108 Mbit in 4-bit codes), against 3.9 Mbit of internal memory in the largest Cyclone IV (EP4CE115) and 0.41 in the catalogue's EP4CE10, whose logic is also too small for the smallest chip (54,000 logic elements against 10,300). Larger boards fall short too: the Kria KV260 has 23.1 Mbit. Only a toy recogniser like the tests' fits.

## Programs with parts

Before voice, `Program` had two fields named after images: `image` (Janus's heads, which write and read the codes of a second codebook) and `vision` (routines that go through a store of rows: the image encoder, Show-o's image half and MAGVIT-v2). Neither is about images: a speech decoder's codes are a second codebook, and a speech encoder's frames are a store of rows. They are now **parts** of a program, with names that say what they do:

- `CodeHead`: a second codebook, with the ops that score its codes and the ones that turn a code into the model's input.
- `RowPart`: routines over a store of rows, called by the host or the firmware in their order, each with a row as the token.

`program.codes()` and `program.rows()` find them. Nothing else changed: the lowered microcode of every model is the same, word for word.

## Status

| Step | What | Status |
|---|---|---|
| 1 | This document | Done |
| 2 | Programs with parts, with identical microcode | Done: the 890 microprograms the quick tests lower hash the same before and after |
| 3 | Cross-attention and LayerNorm without bias | Done, with no change to the die or the compiler: see above |
| 4 | One-dimensional strided convolution | Done: `LoadRows`, with the host giving the first row read as the token |
| 5 | Systems of stages | Done for sizing and latency (`sim::system`). The optimiser and the application do not offer it yet |
| 6 | Moonshine on a toy checkpoint: floating-point reference, exact machine, gates | Done: the exact machine's scores are within 2.6 % of floating point (the die's function tables; the products agree to millionths), and the gates give the exact machine's scores bit for bit: 280 routines, 280,586 cycles for the clip and three tokens (`crates/sim/tests/moonshine.rs`) |
| 7 | Moonshine with the real weights | Done: on a sentence spoken by macOS's `say`, the reference's frames are within 0.0002 % of Transformers' and its scores within 0.0001 %, the same tokens; the die's arithmetic with 16-bit vectors writes the same text, "Hello, this chip can listen and speak." (`--example listen`) |
| 8 | `FN_SIN`, `wrap_left` and the DAC decoder | Done: the reference gives Transformers' sound sample for sample; on toy decoders the exact machine is 36 to 62 dB below floating point with 20-bit vectors, and the gates give its sound bit for bit (322 routines, 70,624 cycles, the microprogram's count the gates' cycle for cycle); the real decoder as in the table above |
| 9 | OuteTTS with the real weights | Done: in floating point the same prompt and the same 317 tokens as Transformers for "Hello, I am a chip.", scores within 0.0005 %; in the die's arithmetic with 20-bit vectors the first 91 tokens match, and the sound the die's DAC makes of them is heard as the same sentence (`--example speak`) |
| 10 | The demo: a question spoken in, the answer spoken out | Done in floating point and in the die's arithmetic (`--example talk`, below) |
| 11 | The system sized: dies, silicon and the time to the first sound | Done from the command line (`--example voice_system`, above) |
| 12 | Voice in the optimiser and the application | Not done |
