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
| Speech to text while it is spoken | [Moonshine Streaming](https://huggingface.co/moonshine-ai/moonshine-streaming-tiny) tiny | The same decoder; an encoder over 20 ms frames whose attention sees a window of frames | 44 M parameters (176 MB in F32) | MIT |
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

**Any length:** a chip is built for clips of up to some length and takes any shorter one. Its stores are sized for the longest, the host runs as many rows as the clip has, and the front end's GroupNorm, whose mean is over every value of the clip, has its totals multiplied by `1 / count`, which the host writes for each clip, before a GroupNorm of one value. A chip for longer clips scores a shorter one bit for bit as one built for exactly that clip, in the exact machine and in gates (`crates/sim/tests/moonshine.rs`).

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

**A window at a time.** A frame's codes reach 10.4 frames of sound on each side (the first convolution's three, then each block's transposed convolution and residual units at its rate, `DacConfig::reach_frames`, rounded up to 11; measured, 9.6). So the decoder takes the sound a window at a time: each window is the frames it keeps plus 11 on each side, and keeps only its own frames' samples. A chip then holds a window, not the whole sentence, and the first sound needs only the first window. Window by window gives the whole decode's sound: in floating point the same samples to the last bit on the real DAC (windows of 38 frames keeping 16), and in the die's arithmetic the same mantissas, since each row keeps its own exponent (`crates/sim/tests/dac.rs`). The cost is the context: a window of 38 frames for 16 kept does 2.4 times the work, which the system's sizing weighs against the silicon it saves. The speech writer, likewise, takes the answer a sentence at a time (`sim::voice::sentences`).

**The sine.** In the real decoder Snake's argument `αx` reaches 108 radians on a short sentence: seventeen turns. The die's function tables take inputs in [−8, 8), and the alignment that brings a value to that range saturates. Two small additions fix it, both built only on dies whose microprogram uses them:

- **`FN_SIN`, a ninth table:** `sin(π·x/8)`, one period over the table's whole range of 16. `Sin` scales radians by 8/π first. The function field becomes one bit wider and the table memory twice as large on the dies that use it.
- **`wrap_left`, one control bit:** a mux beside the left shifter's saturation that keeps the shift's low bits instead. With it, the alignment before `FN_SIN` drops whole periods as it goes, which is exactly the argument's place in the period, with every fraction bit kept.

Language-model dies do not use either, so their microprograms, gates and clock do not change. Doing the same with the die as it was would need `round(αx / 2π)` and a subtraction, and block floating point subtracts at the larger exponent, the integers', which loses the fraction.

Two new operations go with it, microcode only: `Sin` (any argument, in radians) and `MulWeight` (a product by a stored vector, element by element, as `AddWeight` adds one).

**How wide the vectors must be.** Snake runs 29 times between the codes and the sound, and each needs the fraction of the period of an argument as large as 108, while the row shares one exponent across up to 1,536 channels. On a short sentence ("Hello, I am a chip."), against floating point:

| Vectors | Difference below the signal | What Moonshine hears in it |
|---|---|---|
| 16 bits | 20.2 dB | "Hello, I am a chiff." |
| 20 bits | 27.0 dB | "Hello, I am a chip." |
| 24 bits | 32.2 dB (the 12-bit tables' limit) | "Hello, I am a chip." |

A die that speaks needs vectors of 20 bits or more; one that listens does well with 16. Where the 16-bit die loses is not one place: in floating point with only Snake's argument taken as the die takes it, the sound stays 72.6 dB below the signal, and with every row held in the die's format (one exponent per row, 16-bit mantissas) 70.7 dB (`Dac::sine_bits`, `Dac::row_bits`, `--example dac_check`); a wider accumulator changes nothing. It builds up over the decoder's 29 Snakes and convolutions. (A first measurement gave −4.7 dB at 16 bits: the stores were reserved for as many rows as the token's range, and the decoder's addresses overran the index; see "Stores of rows" below.)

**Stores of rows** take only their own rows of vector memory: a routine over rows takes the row as its token, and a store's register was reserved for as many rows as the token can reach (the vocabulary's, the longest map's). Held to its own size, Moonshine-tiny's die went from 88 to 45 mm² and DAC's from 224 to 90, with every program over rows (vision, MAGVIT-v2, Show-o, Moonshine, DAC) giving the same results in gates and the language models' dies unchanged.

## Several programs on several dies

Until now all the dies of a design ran parts of one program: layers in sequence, heads in parallel, or both. Voice needs dies that run **different programs**: some the speech recogniser, some the language model, some the synthesiser. That capability is general:

- **A system is a list of stages** (`sim::system`), each one a program with its own dies, shape and split. Its area is the sum; its speed is each stage's; its latency is the sum along the chain of how long each stage takes to give the next what it needs to start: the whole utterance for the recogniser, the first sentence for the synthesiser (`System::first_output_seconds`).
- **A stage's cycles come from its microprogram.** A language model's stage comes from its sizing (`Stage::from_candidate`), and the recogniser's from the cycles of its own routines (`MoonshineChip::stage`), which on the toy checkpoint are the gates' cycle for cycle.
- **Between stages only text travels, as bytes.** One chip's output port feeds the next one's input port: the chips already speak bytes, so the handoff needs no new port, and the vectors never leave a stage. It is checked in gates with two whole chips of the test model, simulated cycle by cycle at once: the first's `host_tx_valid` and `host_tx_data` drive the second's `host_rx_valid` and `host_rx_data`, and the second's `host_rx_ready` holds the first back. Only the first's boot report (a dot per segment loaded) is left out, and a zero closes the second's prompt once the first halts. The second writes what it writes when a host gives it the first's answer (`two_chips_in_a_chain_pass_their_text_on_their_pins` in `crates/sim/tests/chip.rs`). Moonshine and DAC now run on the chip's own firmware (see "The firmware's schedule"); the chain of the voice dies is still joined by the host in the simulation.
- **Other uses:** speculative decoding with a small model drafting tokens that a large model checks (the large one checks several at once, so the answer comes faster); a model that reads images whose encoder runs on dies of its own; or several small models in a chain.

Speculative decoding with a separate draft model is not built yet. The MTP head (`DraftProgram`) is its cousin inside one model.

## Listening while the question is spoken

Moonshine reads the whole clip before it writes; its streaming model (`moonshine-ai/moonshine-streaming-tiny`, 44 M parameters) is built to read it as it comes. Its front end takes 80 samples (5 ms) at a time: normalised to zero mean and unit RMS, compressed with `asinh(k·x)`, projected and bent with SiLU; two causal convolutions of stride 2 leave a frame every 20 ms. Its encoder has no RoPE, and each layer's attention sees the 15 frames before and, in the first two and last two layers, the 3 after. The decoder is Moonshine's, the frames given learned positions first. `sim::moonshine` reads it (`MoonshineConfig::streaming`): on "What is the capital of France?" from macOS's `say`, its frames are within 0.0003 % of Transformers' and its scores within 0.0001 % (`tools/golden/moonshine_streaming.py`, `--example listen`).

On the chip (`sim::moonshine_stream_chip`) it listens as the question comes:

- **What the die gains:** a tenth function table, `asinh` (`FN_ASINH`), on dies with sixteen table places, as the sine's: within 0.0014 of floating point over the front end's range. CMVN is a LayerNorm whose weight is `k`, the samples scaled by √10 first so that the die's epsilon (10⁻⁵) acts as the model's (10⁻⁶).
- **A layer that sees `a` frames ahead runs `a` frames behind the one before.** Every 20 ms the host sends the samples; the firmware runs the front end, and each layer as far as its window lets it: Moonshine-tiny's last layer runs 12 frames (240 ms) behind the speech. When the question ends only those frames are left, then the text.
- **Keys and values go into rings** as wide as each layer's window (`KvAppend` with a window), so attention at the window's last frame sees exactly the window. The clip's last frames see fewer ahead: a layer that sees 3 ahead also keeps rings 1, 2 and 3 frames narrower, and frame `last − j` attends at the last frame in the ring `3 − j` narrower, which holds exactly its keys.
- **Learned positions** are a table of 4,096 rows read a row at a time: a program of routines now says how many rows each of its own tables has (`RowPart::tables`), so the die keeps them and not the vocabulary's 32,768.

The die's arithmetic with 16-bit vectors writes what Transformers writes: "What is the capital of France?" and "Hello, this chip can listen and speak." (`--example listen`). On a toy checkpoint the exact machine's scores are within 0.2 % of floating point for every kind of window (none ahead, some ahead, one wider than the clip, no layers), the gates give the exact machine's scores bit for bit (900,912 cycles), and the firmware, at instruction level, writes the host's tokens (`crates/sim/tests/moonshine_stream.rs`).

In the plan, the recogniser's first token comes **0.4 ms** after the question ends instead of 25.7 ms, and each 20 ms takes much less than 20 ms on any die tried. With the rest as before the system is three dies, 930 mm², and the first sound comes in **458 ms** instead of 483. Moonshine-tiny stays the default; the streaming model is used wherever its folder is given as the recogniser.

## The firmware's schedule

A die whose program is a set of routines (an encoder that runs once per clip, a decoder that writes tokens, a decoder of sound over rows) needs someone to order them. Until now the host did. Now the chip's description can carry a **schedule** (`netlist::schedule`) and the firmware follows it, so the chip takes its input and gives its output by itself. The schedule knows nothing of a model; it is a list of records of eight words:

| Record | What the firmware does |
|---|---|
| `RECEIVE var` | A word from the host into a variable |
| `COMPUTE` | A variable from another: `(src · mul + pre) / div + post`, at most `max` |
| `VECTORS` | Vectors from the host (exponent, then elements) into the die, a routine after each |
| `REPEAT` | The next records, a number of times given by a variable, with the index in a variable |
| `CALL` | A routine, its token and position each `a · index + b + variable` |
| `HALVE` | Rows added pairwise until one is left (a sum over a length known only at run time) |
| `GENERATE` | Up to some tokens: the next records for each, then the sampler; a stop token ends it |
| `SEND` | A die element and its exponent to the host |
| `STREAM` | The next records for as long as the host sends a word other than zero first: input that comes while it is worked on |

`MoonshineChip::schedule` writes Moonshine's (63 records for Moonshine-tiny, 39 for the tests' toy): the host sends the clip's length, `1 / count` for its GroupNorm and the rows of samples; the firmware works out the lengths as `Lengths::of` does, runs the encoder and writes the tokens greedily until the end token. `DacChip::schedule` writes DAC's (172 records for DAC.speech, 76 for the toy): the host sends each window's codes and which of its samples to keep, and the chip sends back only those, so it skips computing the overlap's. A schedule can be longer than the chip's description has room for (1,008 words), so it goes where a tokenizer's tables would, at word 0x2000 of the data memory: a chip run by a schedule exchanges words, not text, and has none. DAC.speech's takes 1,384 words there, Moonshine-tiny's 512 (`the_real_decoder_s_schedule_fits_its_chip_image`, `the_real_recogniser_s_schedule_fits_its_chip_image`). `netlist::schedule::expand` reads a schedule as the firmware does, so a schedule is checked against the host's own order of routines (for four clip lengths). At instruction level, the firmware driving a die run by the exact machine writes the host's tokens and the host's sound, mantissa for mantissa (`the_firmware_transcribes_a_clip_as_the_host_ordering_the_routines`, `the_firmware_decodes_window_by_window_as_the_host_does`).

**The CPU's time counts.** Starting a routine costs the control CPU (three cycles an instruction, 33 more for a multiplication) about 930 cycles in the general path. DAC's routines are short (225 die cycles on average on the toy) and many: the real decoder runs **360,000 a window of 54 frames**. So the firmware has a lean path for a repeat made only of calls (up to 16 a repeat): it reads the records once, works out each parameter register's first value and its step, and each iteration only writes, adds and starts the die; about **270 cycles a routine**. The planner counts the CPU's cycles on top of the die's (`sim::chip::firmware_cycles`): 300 a routine on the lean path (the encoder's and the decoder's passes), 1,000 a routine started alone (rows of samples, the pairwise sums, each token's routines, each frame's codes, each sample sent back) and 200 a word received or sent (a three-second clip is 48,000 samples). Both firmware tests check the CPU's measured cycles against that count: on the toy decoder, 26 M cycles of the CPU's own (30 M counted) beside 21 M of the die's. The firmware now fills 32.1 of its 32 KB of program memory.

A die that ran a repeated routine itself, its token stepping in the die's controller, would take this off the CPU. It would need a counter and an adder per parameter register in every die, so it is not built.

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
| A spoken turn's stages, shared by the demo and the engine | `crates/sim/src/voice.rs` | Voice |
| Demo: a WAV in and a WAV out | `crates/sim/examples/talk.rs` | Voice |
| The system planned | `crates/sim/src/system.rs` (`plan`), `sizing::routines_die`; `crates/sim/src/voice_plan.rs`, `crates/sim/examples/voice_system.rs` | General; voice |
| The engine's job and the application's view | `crates/server/src/voice_job.rs`, `web/src/voice/` | Voice |
| Schedules the firmware follows | `crates/netlist/src/schedule.rs`, `firmware/src/main.rs` (`follow`, `repeat_calls`), `sim::chip::RoutinesDie` | General: any program of routines |
| Moonshine's and DAC's schedules | `MoonshineChip::schedule`, `DacChip::schedule`, `dac_chip::schedule_input` | Voice |
| `FN_ASINH` and `Asinh` | `crates/netlist/src/block.rs`, `crates/compiler`, `crates/sim/src/lower.rs`, `crates/sim/src/exec.rs` | General: built only where used |
| Tables of a program's own rows (`RowPart::tables`) | `crates/compiler` | General |
| Streaming Moonshine: reference, on the die, tests | `crates/sim/src/moonshine.rs`, `crates/sim/src/moonshine_stream_chip.rs`, `crates/sim/tests/moonshine_stream.rs`, `tools/golden/moonshine_streaming.py` | Voice |

## The system, planned

`cargo run --release -p sim --example voice_system -- MOONSHINE LLM OUTETTS DAC [FIRST_SOUND_MS]` plans the chip as a system. Each stage is measured on a few shapes of die (1 or 16 units of 64 lanes, 2,048 to 65,536 cells), from its own microprogram and with the optimiser's counts in N6. The decoder is also measured with windows keeping 16, 32 or 64 frames. Then `system::plan` weighs every way of grouping the stages on dies (15 for four stages):

- **A shared die** has its logic once, the cells and microcode of every stage on it, the vector memory of the one that needs most, and the slowest clock.
- **Stages that run at once never share a die.** Within a turn the recogniser runs alone, before the rest; the language model, the speech writer and the decoder run at once, each taking the one before's first sentence or window while it goes on. So only the recogniser may share, with any one of the others.
- **Each die must fit the reticle, each stage must keep up** (the writer 180 tokens a second, the decoder 75 frames: real time), and the first sound must come in time.
- **The recogniser's and the decoder's time is the die's and the firmware's:** the CPU's cycles to order their routines and exchange their words come on top of the die's (see "The firmware's schedule").
- **Of what meets all that, the least silicon wins**, then the fewest dies, then the soonest sound.

For a three-second question, a first sentence of 12 tokens and the first sound within 500 ms:

| Grouping | Silicon | First sound |
|---|---|---|
| Each stage on its own die | 956 mm² | 491 ms |
| Recogniser with the language model | 1,025 mm² | 480 ms |
| Recogniser with the speech writer | 1,025 mm² | 480 ms |
| **Recogniser with the sound decoder** | **927 mm²** | **483 ms** |
| The other eleven | — | stages that run at once |

**Chosen: three dies and 927 mm²**:

- **Moonshine and DAC on one die** of 16 units, 2,048 cells and 20-bit vectors (481 mm²), the decoder with windows of 54 frames keeping 32.
- **Qwen3-0.6B and OuteTTS on a die each** of 16 units and 2,048 cells (222 and 224 mm²): enough for 1,800 tokens a second, so more cells would only add silicon.

Before the firmware ordered the routines, and its cycles were left out, the same plan was 569 mm² with Moonshine and DAC on a die of one unit (123 mm²). With the CPU's 122 M cycles a window (108 M of them for the passes' 360,000 routines), that die no longer gives the first sound in time, so the planner takes 16 units, whose maps the units each keep a copy of (436 mm² of vector memory). The planner merges what pays and keeps apart what would wait on itself; with a tighter time or other models it chooses differently.

**Energy of a turn** (`voice_plan::turn_energy`, in the Voice view too). Each stage's work is counted as the estimator counts a token's: every product of a weight by a value in the cell (its gates' switching, simulated) and the weight's read from its memory, for the products its routines or its program actually do. Every die leaks the whole turn, none powered off. For a three-second question and an answer of 12 tokens (about four seconds of speech), with the three-word profile:

| Stage | Work |
|---|---|
| Speech to text | 1.4 mJ |
| Language model | 29.6 mJ |
| Speech writer | 540 mJ |
| Sound decoder | 342 mJ |
| Every die's leakage, 7.5 s | 8.15 J |
| **The turn** | **9.07 J, 1.21 W on average** |

Leakage is most of it: the weights' memories leak for the whole turn while each die works for a fraction of it. Holding the weights in ROM or powering idle dies off would matter far more than any saving in the work; neither is counted here, nor the control CPUs' own work.

## A larger language model

When the language model fits no single die within the reticle, the planner measures it on the sets of dies the frontier found for Qwen3.8-27B (`VoiceAsk::large`): 12 dies of 64 × 128 units and 32,768 cells in groups of four, and the fastest ones, 12 dies of 128 × 128 units and 16 of them in groups of eight. Such a stage is a die of its own kind repeated, and never shares a die with another stage. A model whose weights could not fit one die even at four bits (one weight for each product of a token, in the process's densest memory) is not measured on single dies at all. A checkpoint already quantized keeps its own codes, as the frontier keeps them; the others go to 8 bits.

With Qwen3.8-27B (4-bit, 8-bit in places) in place of Qwen3-0.6B, for the same turn:

| Dies | What | Silicon |
|---|---|---|
| 1 of 16 × 64 units, 2,048 cells | Moonshine and DAC | 481 mm² |
| **12 of 64 × 128 units, 32,768 cells** | **Qwen3.8-27B, groups of four, 2,518 tokens a second** | **12 × 615 mm²** |
| 1 of 16 × 64 units, 2,048 cells | OuteTTS | 224 mm² |
| **14 dies** | **First sound in 474 ms** | **8,082 mm²** |

The smallest set is enough: the faster ones only add silicon, since the answer's first sentence is ready long before the sound needs it. Each of these dies is 615 mm² for a context of 1,024 tokens, sized to the prompt; the frontier's 641 mm² are for 8,192 tokens. A turn costs 134 J, almost all of it leakage (132 J): 14 dies of memory leaking for 7.5 s.

Two simplifications: a shared die's cell memories are added as each stage's were measured, though one memory of their sum would have a little less periphery; and a recogniser measured with 16-bit vectors keeps its cycles on the decoder's 20-bit die.

The speech dies are routines over rows: each unit of a die keeps its own copy of the maps. Splitting them among the units would need writes only the owning unit makes and reads from a unit chosen as it runs, new datapaths across units, and would not make the products faster, since they run in the cells; so the planner weighs the copies against the speed instead.

## Speaking more naturally

- **Sampling.** OuteTTS draws its tokens as its authors recommend: temperature 0.4, top-k 40, top-p 0.9, min-p 0.05, after a repetition penalty of 1.1 over the last 64 tokens (`speak::Sampling::outetts`, from a seed). Greedy stays for comparisons with Transformers, which it matches token for token.
- **A voice profile.** OuteTTS speaks more steadily in a given voice: its prompt starts with a sentence said in that voice, word by word with its duration, loudness, brightness, pitch and codes (`SpeechVocabulary::prompt_with`; the profile is `en-female-1-neutral.json` from OuteTTS's package, kept in the model's `speakers/`). The prompt is OuteTTS's to the token: 1,840 tokens for "Hello, I am a chip.", the same as Transformers' tokenizer gives. It costs those tokens before each sentence: with the whole profile no system gives the first sound within 500 ms (within 1,500 ms the planner chooses three dies and 720 mm², and puts the recogniser with the speech writer instead: first sound in 1,310 ms). So the writer reads only the profile's first three words (`voice::PROFILE_WORDS`, `Speaker::first_words`): 197 tokens instead of 1,840, and Moonshine still hears "Hello, I am a chip." With it the plan is three dies, 949 mm², and the first sound in 472 ms, the writer's die taking 8,192 cells to read its prompt sooner.
- **A conversation.** The language model keeps its cache between questions (`voice::Conversation`): on top of it goes what the chat template writes after the last answer (its end) and the new question, so each question costs only its own tokens. With Qwen3-0.6B, "What is the capital of France?" and then "And of Italy?" gives "The capital of Italy is Rome." from the cache, and the scores after it are those of the whole conversation fed from the start (`crates/sim/tests/conversation.rs`). A template that does not write the last answer as it was makes it start again from the whole conversation.

## In the application

The Voice view records a question with the microphone or opens a recording, runs the turn in the engine with every stage in floating point or in the die's arithmetic, and plays the answer; the conversation goes on until "New conversation". Its panel plans the chip (`/api/voice/plan`): the first sound's limit, the longest question and the most silicon, and it shows the dies chosen and every grouping weighed, with why. The project wizard's "A voice assistant" leads there. The Models list says which folders listen, speak and make the sound.

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
| 11 | The system planned: which stages share a die, at which shape, and the time to the first sound | Done (`sim::system::plan`, `sim::voice_plan`, `--example voice_system`, above) |
| 12 | Voice in the application | Done: the Voice view (a turn, a conversation, the chip planned) and the project wizard's "A voice assistant" (see above) |
| 13 | OuteTTS's sampling, a voice profile, a conversation | Done (see "Speaking more naturally") |
| 14 | Moonshine and DAC on their own firmware | Done: schedules the firmware follows, bit for bit the host's at instruction level, and their CPU cycles counted in the plan (see "The firmware's schedule") |
| 15 | One real layer of Moonshine-tiny in gates | Done: the stem, the first encoder and decoder layers and 64 rows of the vocabulary with the model's own weights, a two-frame clip and two tokens: 160 routines, 13,568,264 cycles, the gates' scores the exact machine's bit for bit (20 minutes; `LLM2CHIP_REAL_MOONSHINE=1`) |
| 16 | A language model of several dies in the plan | Done (see "A larger language model") |
| 17 | Recognising while the question is spoken | Done: streaming Moonshine, the `asinh` table and the `STREAM` record (see "Listening while the question is spoken") |
