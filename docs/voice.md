# Voice on the chip

A chip that listens and speaks: speech goes in, the language model answers, and the answer comes out as speech. This document describes how llm2chip does it: which models it uses, what each part needs from the die, how the chip runs those programs by itself, how a voice system is planned, how to run all of it, how it is tested, and what is still missing.

**Listening and speaking are two separate parts,** and both are optional: a chip may listen (speech to text, answering in text), speak (text to speech, taking its question as text) or both, and a chip that does neither has nothing of voice (see [Two parts, each optional](#two-parts-each-optional)).

The guiding rule is the same as for images: **the die does not know what voice is.** It gains general operations and capabilities that any model may use. Speech recognition and speech synthesis are just more models compiled onto dies, and the pipeline that joins them is a general way of running several programs on several dies. Removing voice from the project should only remove files (see [Where each piece lives](#where-each-piece-lives)).

**Contents**

- [Two parts, each optional](#two-parts-each-optional)
- [The pipeline](#the-pipeline)
- [The models](#the-models)
- [What each stage needs from the die](#what-each-stage-needs-from-the-die)
- [The chip runs its programs by itself](#the-chip-runs-its-programs-by-itself)
- [Several programs on several dies](#several-programs-on-several-dies)
- [The system, planned](#the-system-planned)
- [Speaking more naturally](#speaking-more-naturally)
- [Using it](#using-it)
- [Tests](#tests)
- [Programs with parts](#programs-with-parts)
- [Where each piece lives](#where-each-piece-lives)
- [Limits and what is next](#limits-and-what-is-next)
- [History](#history)

## Two parts, each optional

| Part | What it adds | Project option | Build feature | View |
|---|---|---|---|---|
| **Listening** | A recogniser (Moonshine or streaming Moonshine) that turns a spoken question into text for the language model | *Listens* (`speech_in`) | `listen` | Listen |
| **Speaking** | A speech writer (OuteTTS) and a sound decoder (DAC) that say the answer, or a text | *Speaks* (`speech_out`) | `speak` | Speak |

**In a project** both options are off by default, next to *Vision (image analysis)*. With both off nothing of voice shows: no Listen or Speak view, no voice planning, no voice dies. The project wizard's *A voice assistant* asks which of the two the chip has. The estimate and the floorplan count the voice dies, as dies of their own beside the language model's.

**A system prompt.** The project's fixed system prompt (`system_prompt`, see [full-chip.md](full-chip.md#a-fixed-system-prompt)) is the first thing a spoken conversation reads, so the cache it keeps starts with it. A chip that holds a conversation reads it when the conversation starts, while it waits, so the voice planner counts its tokens in the language model's context but not in the time to the first output. A voice project starts with the system prompt "You are a voice assistant. Answer in one short sentence." when it has none: that is what keeps answers short, and it can be changed in the panel.

**What each die is for.** The floorplan's *Part* colour layer, and the caption above every die, say what each die is for: *Text* (the language model), *Vision* (the image encoder, on die 0 with the embedding when *Vision (image analysis)* is on; its share of the die's cells is drawn as a band), *Listening* and *Speaking* (and *Image drawing*, see [vision.md](vision.md)). Voice dies sit in a row of their own below the language model's, numbered after them. A die two parts share is split side by side, and its caption names the stages it holds: with both parts the reference plan puts the recogniser with the sound decoder, *Listening + Speaking (sound decoder)*, and the speech writer on its own, *Speaking (speech writer)*. In the *Function* layer a voice die shows its logic, cell memory, vector memory and microcode. The estimate's *Total area* adds the voice dies to the language model's, with *Text (language model)* and *Listening and speaking* apart, and their leakage as a metric of its own (not in the power while generating text, since they only work during a spoken turn).

**Reference dies until the project plans its own.** Until the Listen or Speak view plans the chip, the voice dies are those of the plans measured here with Moonshine-tiny, Qwen3-0.6B, OuteTTS and DAC (below), drawn dashed and marked as reference. Planning in the view keeps the chosen dies with the project (`voice_plan` in its parameters: the parts it was planned for and each die's stages, shape, area by part and clock); they count while the project has the same parts, and the reference dies come back when a part is turned on or off. A project's plan never puts the recogniser on the language model's die (`VoiceAsk::text_apart`): the language model's dies are the estimator's, so the voice dies hold only voice stages.

**In the planner and the engine** the stages come from the parts the chip has (`voice_plan::Role`): listening alone is the recogniser and the language model, and its target is the answer's first token; speaking alone is the language model, the writer and the decoder, its question taken as text, and its target is the first sound; both is the whole spoken turn. A turn in the engine takes its question as a clip or as text, and the language model, the writer and the decoder are each optional, so the Speak view can also read a text aloud without answering it.

**In the code** each part is a folder of its own and a Cargo feature of its own: `sim::listen` (Moonshine, its chip, the streaming chip) and `sim::speak` (OuteTTS, DAC, DAC's chip), the features `listen` and `speak` of `sim` and `server`, both on by default. Built without one, nothing of it is compiled: its modules, the turn's and the planner's stage, the catalogue's role, its tests and examples; built with neither, the engine has no voice routes at all. The browser build never carries voice. `scripts/check.sh` builds and lints the engine with neither, with listening alone and with speaking alone, so the separation cannot rot.

```bash
cargo build --release -p server --no-default-features --features listen    # an engine that only listens
cargo build --release -p server --no-default-features                      # one with no voice at all
```

**What stays in every chip** is general, not voice: the die's sine and `asinh` tables, `wrap_left` and the command queue are built only on dies whose programs use them, but the firmware's schedule interpreter (the records and their lean path, about 2 KB of its program memory) is in every chip's firmware, since any program of routines can use it, and the chip's description keeps its two words for a schedule and a queue.

## The pipeline

```
 microphone ──► speech to text ──► language model ──► text to speech ──► speaker
  16 kHz         (Moonshine)        (any model of      (OuteTTS, a Qwen3     24 kHz
  samples        text tokens        the catalogue)      that writes audio
                                    text tokens         codes, + DAC decoder)
```

Each stage is a program of its own on dies of its own. Between stages only text travels, as bytes: a few dozen a second, far below what any link carries. Each chip already reads and writes UTF-8 bytes through its host port, with its own tokenizer, so a recogniser and a language model with different vocabularies join without translating token ids. Splitting the stages over dies costs nothing measurable in latency, and each stage's dies can have their own shape.

In a conversation what matters is not tokens per second but **how long until the first sound**, which should stay below about half a second. It adds up three things:

1. **The end of the utterance:** detecting silence, then the transcription. Moonshine's encoder attends over the whole clip and its GroupNorm averages over all of it, so it starts when the utterance ends; its cost follows the clip's length instead of a fixed 30-second window. Streaming Moonshine encodes while the question is spoken and has almost nothing left at its end.
2. **The first token of the language model:** the prompt's prefill.
3. **The first sentence of speech:** synthesis can start as soon as the model writes its first sentence, while it keeps writing; the sound decoder starts with the first window of codes.

Speech needs few tokens: DAC codes come at 75 frames per second (the 24 kHz rate over the decoder's upsampling of 8 × 5 × 4 × 2 = 320), two codebooks per frame at 1.5 kbps. That is 150 codes per second, against thousands of tokens per second on the dies. A voice die is sized for latency, not for speed.

## The models

| Stage | Model | What it is | Size | Licence |
|---|---|---|---|---|
| Speech to text | [Moonshine](https://huggingface.co/UsefulSensors/moonshine-tiny) tiny / base | Encoder–decoder transformer over the raw 16 kHz waveform | 27 M / 61 M parameters (108 / 246 MB in F32) | MIT (tiny's card) |
| Speech to text while it is spoken | [Moonshine Streaming](https://huggingface.co/moonshine-ai/moonshine-streaming-tiny) tiny | The same decoder; an encoder over 20 ms frames whose attention sees a window of frames | 44 M parameters (176 MB in F32) | MIT |
| Text to speech | [OuteTTS-1.0-0.6B](https://huggingface.co/OuteAI/OuteTTS-1.0-0.6B) | A Qwen3-0.6B (`model_type: qwen3`, 28 layers, hidden 1,024, vocabulary 157,760) that writes the codec's codes as tokens | 0.6 B parameters (BF16) | Apache-2.0 |
| Codes to sound | [DAC.speech v1.0](https://huggingface.co/ibm-research/DAC.speech.v1.0) | The decoder of the Descript Audio Codec, 24 kHz, fine-tuned for speech | 296 MB per checkpoint (`.pth`) | CDLA-Permissive-2.0 |

Any language model of the catalogue answers; Qwen3-0.6B is the one measured here, and Qwen3.8-27B the large one.

Why these:

- **Moonshine takes the waveform, not a spectrogram.** Whisper needs a log-mel spectrogram, that is, an FFT on the host or in the firmware, and always processes 30-second windows. Moonshine's front end is three convolutions over the samples, which the die already does, and its length follows the audio. Its blocks are the usual ones: LayerNorm without bias, attention with partial RoPE (90 % of each head in tiny, 62 % in base), GELU in the encoder's MLP and a gated SiLU in the decoder's. Eight heads, all with their own keys and values.
- **OuteTTS writes speech as ordinary tokens.** Its language model is a plain Qwen3, a family the compiler already supports, and the codes are rows of its vocabulary: `<|c1_N|>` and `<|c2_N|>` for the two codebooks (`<|c1_0|>` is token 151,669), a pair per frame, grouped by word with a few tokens of their own (the word's duration, energy and pitch). The language model needs no new head and no new field: only the decoder that turns codes into sound is new.
- **DAC's decoder is only convolutions.** No attention and no recurrence: a convolution in, four blocks that each upsample (a transposed convolution) and refine (three residual units with dilations 1, 3 and 9), and a convolution out with tanh. That is the same kind of work as MAGVIT-v2's decoder, which the die already runs in gates, but in one dimension.

### Getting them

The engine finds models in `~/.llm2chip/models` and the other folders the Models view lists ([install.md](install.md#a-first-model)). Moonshine, streaming Moonshine and OuteTTS download from the Models view or Hugging Face as they are (safetensors). Two need a step more:

- **OuteTTS's voice profile:** `en-female-1-neutral.json` from OuteTTS's Python package, kept in the model's folder as `speakers/en-female-1-neutral.json`. Without it the writer speaks without a profile.
- **DAC.speech** comes as a PyTorch pickle. `tools/convert/dac.py` turns the 1.5 kbps checkpoint into safetensors and a `config.json`, reading the pickle with `weights_only` so that no code in it runs, and folding weight normalisation into plain weights:

```bash
uv run tools/convert/dac.py weights_24khz_1.5kbps_v1.0.pth ~/.llm2chip/models/dac-speech-24khz-1.5kbps
```

PyTorch is needed only for that conversion and for the references against Transformers (`tools/golden/`), each a script with its dependencies inline for `uv`.

## What each stage needs from the die

### Moonshine's encoder

The encoder is a store of rows, like the image encoder: one row per frame, routines that go through it block by block, the keys and values of a block kept in a store and attended bidirectionally. The host writes the samples, 64 to a row, so the front end is:

| Layer | In Moonshine | On the die |
|---|---|---|
| `conv1`: kernel 127, stride 64, 1 → hidden, tanh | One output per 64 samples, reading 127 | `LoadRows` of rows `t` and `t + 1` (128 samples, the last weight zero), a product, `SoftCap`'s tanh |
| `GroupNorm` with one group | Mean and variance over all frames and channels | `GroupNorm`, as in MAGVIT-v2 |
| `conv2`: kernel 7, stride 3, hidden → 2·hidden, GELU | | `LoadRows` with offsets 0…6, the routine called on every third row |
| `conv3`: kernel 3, stride 2, 2·hidden → hidden, GELU | | The same, offsets 0…2, every other row |

A strided one-dimensional convolution is the strided two-dimensional one of MAGVIT-v2 with a single row of pixels: the firmware passes the input row as the token and stores the output at its own index. Together the strides make one frame every 384 samples, about 42 per second; a clip needs at least 1,151 samples for one frame.

**Any length:** a chip is built for clips of up to some length and takes any shorter one. Its stores are sized for the longest, the host runs as many rows as the clip has, and the front end's GroupNorm, whose mean is over every value of the clip, has its totals multiplied by `1 / count`, which the host writes for each clip, before a GroupNorm of one value. A chip for longer clips scores a shorter one bit for bit as one built for exactly that clip, in the exact machine and in gates.

### Moonshine's decoder

The decoder is a causal language model with one more attention per layer: **cross-attention**, whose queries come from the token and whose keys and values come from the encoder's frames. It needs no new operation and no new field:

- **The keys and values:** each encoder frame, once the encoder is done, goes through every layer's cross-attention key and value projections, and `KvAppend` stores them in a key slot of their own, at the frame's index. The routine is called once per frame, as the image encoder's are.
- **The attention:** `AttnSum` over that slot, in a routine of its own run with the position at the last frame. An attention reads as far as its position, so it reads every frame, as the image encoder's second pass already does. Each decoder layer is then two routines: self-attention at the token's position, and cross-attention with the MLP at the last frame's.
- **Positions:** RoPE in self-attention only, partial, as Qwen3.5 already has (`rotary` smaller than the head). Moonshine rotates adjacent pairs and the die the two halves of the rotated part: the rows of every query and key projection are reordered alike as they are read, which leaves their products unchanged.
- **The gated MLP:** `fc1` makes both halves in one product, `Slice` separates them, then SiLU and the product, as the other models' MLP.
- **LayerNorm without bias:** a bias of zeros is added as the weights are read.

The decoder's routines are shared by both recognisers (`moonshine_chip::decoder_routines`).

### Streaming Moonshine's encoder

Its front end takes 80 samples (5 ms) at a time: normalised to zero mean and unit RMS (CMVN), compressed with `asinh(k·x)`, projected and bent with SiLU; two causal convolutions of kernel 5 and stride 2 leave a frame every 20 ms. Its encoder has no RoPE, and each layer's attention sees the 15 frames before and, in the first two and last two layers, the 3 after; its LayerNorms keep their weight less one (`gamma + 1`). The decoder is Moonshine's, the frames given learned positions first, with a head of its own. On the die (`sim::listen::moonshine_stream_chip`):

- **A tenth function table, `asinh` (`FN_ASINH`),** on dies with sixteen table places, as the sine's: within 0.0014 of floating point over the front end's range (a frame at unit RMS reaches √79 ≈ 8.9, times the compression's 0.64, inside the table's ±8). CMVN is a LayerNorm whose weight is `k`, the samples scaled by √10 first so that the die's epsilon (10⁻⁵) acts as the model's (10⁻⁶).
- **Causal convolutions** read five rows from a store with four rows of zeros before; each is a routine that reads and one that stores.
- **A layer that sees `a` frames ahead runs `a` frames behind the one before.** Every 20 ms the firmware runs the front end and moves each layer as far as its window lets it: Moonshine-tiny's last layer runs 12 frames (240 ms) behind the speech. When the question ends only those frames are left, then the text.
- **Keys and values go into rings** as wide as each layer's window (`KvAppend` with a window), so attention at the window's last frame sees exactly the window. The clip's last frames see fewer ahead: a layer that sees 3 ahead also keeps rings 1, 2 and 3 frames narrower, and frame `last − j` attends at the last frame in the ring `3 − j` narrower, which holds exactly its keys.
- **Learned positions** are a table of 4,096 rows read a row at a time: a program of routines says how many rows each of its own tables has (`RowPart::tables`), so the die keeps them and not the vocabulary's 32,768.

### OuteTTS's language model

Nothing new: it is a Qwen3 and compiles as one. The code tokens are rows of its vocabulary, so it samples them with its own head. The host or the firmware recognises them by their range and passes them to the decoder.

### DAC's decoder

Sound is kept in stores of rows, one per time step with all its channels, each with a border of 27 zero rows on each side: as far as a residual unit's widest convolution reaches (three taps of dilation 9). That way every convolution, padded or dilated, reads its taps at fixed offsets.

| Layer | On the die |
|---|---|
| The codes' vectors: each codebook's 8-value row, projected to 1,024 by a 1×1 convolution, the codebooks added | A routine per codebook, with the code as its token: `EmbRead`, a product, `Add` |
| Convolution with kernel 7 and padding 3 | `LoadRows` with offsets 0…6 from three rows before, a product with the kernel laid out the same way; a second routine stores the result at its own row |
| Transposed convolution with rate s, kernel 2s and padding p = ⌈s/2⌉ | Output row n = s·q + phase − p reads input rows q − 1 and q through a kernel that depends only on the phase: a routine per phase, each an ordinary convolution of two rows. A run of q's that give every phase is one pass |
| Dilated convolution (1, 3, 9) | `LoadRows` with offsets spaced by the dilation |
| Snake: `x + sin²(αx) / α`, α per channel | `MulWeight` by α, `Sin`, its square, `MulWeight` by 1/α, `Add` |
| Final tanh | `SoftCap`'s tanh |

**A window at a time.** A frame's codes reach 10.4 frames of sound on each side (the first convolution's three, then each block's transposed convolution and residual units at its rate, `DacConfig::reach_frames`, rounded up to 11; measured, 9.6). So the decoder takes the sound a window at a time: each window is the frames it keeps plus 11 on each side, and keeps only its own frames' samples. A chip then holds a window, not the whole sentence, and the first sound needs only the first window. Window by window gives the whole decode's sound: in floating point the same samples to the last bit on the real DAC (windows of 38 frames keeping 16), and in the die's arithmetic the same mantissas, since each row keeps its own exponent. The cost is the context: a window of 38 frames for 16 kept does 2.4 times the work, which the system's sizing weighs against the silicon it saves. The speech writer, likewise, takes the answer a sentence at a time (`sim::voice::sentences`).

**The sine.** In the real decoder Snake's argument `αx` reaches 108 radians on a short sentence: seventeen turns. The die's function tables take inputs in [−8, 8), and the alignment that brings a value to that range saturates. Two small additions fix it, both built only on dies whose microprogram uses them:

- **`FN_SIN`, a ninth table:** `sin(π·x/8)`, one period over the table's whole range of 16. `Sin` scales radians by 8/π first. The function field becomes one bit wider and the table memory twice as large on the dies that use it.
- **`wrap_left`, one control bit:** a mux beside the left shifter's saturation that keeps the shift's low bits instead. With it, the alignment before `FN_SIN` drops whole periods as it goes, which is exactly the argument's place in the period, with every fraction bit kept.

Language-model dies use neither, so their microprograms, gates and clock do not change (their control word stays within the 224 bits the load bus carries). Doing the same with the die as it was would need `round(αx / 2π)` and a subtraction, and block floating point subtracts at the larger exponent, the integers', which loses the fraction. Two new operations go with it, microcode only: `Sin` (any argument, in radians) and `MulWeight` (a product by a stored vector, element by element, as `AddWeight` adds one).

**How wide the vectors must be.** Snake runs 29 times between the codes and the sound, and each needs the fraction of the period of an argument as large as 108, while the row shares one exponent across up to 1,536 channels. On a short sentence ("Hello, I am a chip."), against floating point:

| Vectors | Difference below the signal | What Moonshine hears in it |
|---|---|---|
| 16 bits | 20.2 dB | "Hello, I am a chiff." |
| 20 bits | 27.0 dB | "Hello, I am a chip." |
| 24 bits | 32.2 dB (the 12-bit tables' limit) | "Hello, I am a chip." |

A die that speaks needs vectors of 20 bits or more; one that listens does well with 16. Where the 16-bit die loses is not one place: in floating point with only Snake's argument taken as the die takes it, the sound stays 72.6 dB below the signal, and with every row held in the die's format (one exponent per row, 16-bit mantissas) 70.7 dB (`Dac::sine_bits`, `Dac::row_bits`, `--example dac_check`); a wider accumulator changes nothing. It builds up over the decoder's 29 Snakes and convolutions.

**Stores of rows** take only their own rows of vector memory: a routine over rows takes the row as its token, and a store's register was reserved for as many rows as the token can reach (the vocabulary's, the longest map's). Held to its own size, Moonshine-tiny's die went from 88 to 45 mm² and DAC's from 224 to 90, with every program over rows (vision, MAGVIT-v2, Show-o, Moonshine, DAC) giving the same results in gates and the language models' dies unchanged. (A first measurement gave −4.7 dB at 16 bits because of it: the decoder's addresses overran the index.)

## The chip runs its programs by itself

A die whose program is a set of routines (an encoder that runs once per clip, a decoder that writes tokens, a decoder of sound over rows) needs someone to order them. The chip's own firmware does it, following a **schedule** carried in the chip's image, so the chip takes its input and gives its output with no host deciding anything.

### The schedule

`netlist::schedule` defines it: a list of records of eight words, which know nothing of a model.

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

Variable 0 is always zero; 1 and 2 are `GENERATE`'s token and position, 3 the innermost repeat's index, the rest the schedule's own. `netlist::schedule::expand` reads a schedule as the firmware does (`firmware/src/main.rs`, `follow`), so the host can check it against its own order of routines and `sim::chip::follow_schedule` can run it on a die simulated by the exact machine.

A schedule can be longer than the chip's description has room for (1,008 words), so it goes where a tokenizer's tables would, at word 0x2000 of the data memory (`chip::add_schedule`): a chip run by a schedule exchanges words, not text, and has none. Description word `INFO_SCHEDULE` (29) gives its address, `INFO_QUEUE` (30) the command queue's size, and the parameters start at word 31.

| Chip | Records | Words |
|---|---|---|
| Moonshine-tiny (`MoonshineChip::schedule`) | 63 | 512 |
| Streaming Moonshine (`StreamingChip::schedule`) | 83 on the tests' toy | |
| DAC.speech (`DacChip::schedule`) | 172 | 1,384 |

### What the host sends and gets back

Every word is 32 bits, little-endian; a vector is its exponent then its elements, in the die's format (`netlist::block::encode_vector`).

- **Moonshine** (`MoonshineChip::schedule_input`): the clip's length in samples, then `1 / count` for its GroupNorm (a vector of one), then the rows of 64 samples. The chip works out the lengths as `Lengths::of` does, runs the encoder and sends back the text's tokens, greedily, up to the end token or six a second of audio.
- **Streaming Moonshine** (`StreamingChip::schedule_input`): for each 20 ms, a word other than zero and four vectors of 80 samples; a zero when the speech ends. Speech past the last whole 20 ms is left out, and the speech must last longer than the layers run behind (240 ms for Moonshine-tiny's). The chip sends back the tokens.
- **DAC** (`dac_chip::schedule_input`): the number of windows; for each, its codes frame by frame and codebook by codebook, then the first sample it keeps and how many. The chip sends back each kept sample as its mantissa and exponent, and skips computing the overlap's.

At instruction level the firmware, driving a die simulated by the exact machine, writes the host's tokens and the host's sound, mantissa for mantissa.

### What the CPU costs

Starting a routine costs the control CPU (an RV32IM core at three cycles an instruction, 33 more for a multiplication) about 930 cycles in the general path. DAC's routines are many and short: the real decoder runs **360,000 a window of 54 frames**, about 860 die cycles each on a die of one unit. So the firmware has a **lean path** for a repeat made only of calls (up to 16 a repeat): it reads the records once, works out each parameter register's first value and its step, and each iteration only writes the parameters, starts the die and steps the next values while the die runs: 120 to 250 cycles a routine. The parameters cannot be written while the die runs a routine, though, so the writes still wait for it: for the real decoder that was some 50 M cycles a window, too many for a die of one unit to give the first sound in time.

### The command queue

**A command queue before the die** (`DieFeatures::queue_bits`, built only where asked) takes that wait away. The CPU's writes to the parameters and to the entry wait in it, and the die takes them one a cycle whenever it is free, an entry starting it: the CPU orders the next routine while the die runs this one, and waits only when the queue has no room for a whole routine's orders (`REG_QUEUE_ROOM`, register 49). `REG_STATUS` says free once the queue is empty and the die idle, so everything else in the firmware works as before. Sixteen orders are 16 × 38 flip-flops and two pointers; the dies that run language models have none.

The gates take the orders as the instruction-level model (`ChipModel::with_queue`) does, cycle for cycle: the test model's whole chip (230,439 cycles) and the toy Moonshine's whole chip following its schedule (1,192,263 cycles), the same bytes. On a toy decoder whose routines are as short as the CPU's work, the CPU's own cycles fall from 20.4 M to 12.6 M; with routines longer than it, as the real decoder's, from 23.6 M to 8.5 M, most of the rest being the routines started alone and the words exchanged.

### How the planner counts the firmware

The system's planner adds the CPU's cycles to the die's (`sim::chip::firmware_cycles`, `lean_cycles`):

| What | CPU cycles |
|---|---|
| A routine on the lean path, without a queue | 250 |
| A routine on the lean path, with a queue | The die's wait between routines (one a parameter and 20 more), or what of the CPU's 300 the routine does not hide, whichever is more |
| A routine started alone (rows of samples, pairwise sums, each token's routines, each frame's codes, each sample sent back) | 1,000 |
| A word received or sent | 200 |

Both firmware tests check the measured cycles against that count, with the queue and without, and DAC's with routines shorter and longer than the CPU's work. The firmware takes 32,332 of its 32,768 bytes of program memory: 436 are left.

## Several programs on several dies

Until voice, all the dies of a design ran parts of one program: layers in sequence, heads in parallel, or both. Voice needs dies that run **different programs**: some the speech recogniser, some the language model, some the synthesiser. That capability is general:

- **A system is a list of stages** (`sim::system`), each one a program with its own dies, shape and split. Its area is the sum; its speed is each stage's; its latency is the sum along the chain of how long each stage takes to give the next what it needs to start: the whole utterance for the recogniser, the first sentence for the synthesiser (`System::first_output_seconds`).
- **A stage's cycles come from its microprogram.** A language model's stage comes from its sizing, and a program of routines' from the cycles of its own routines and its firmware's, which on the toy checkpoints are the gates' cycle for cycle.
- **Between stages only text travels, as bytes.** One chip's output port feeds the next one's input port: the chips already speak bytes, so the handoff needs no new port, and the vectors never leave a stage. It is checked in gates with two whole chips of the test model, simulated cycle by cycle at once: the first's `host_tx_valid` and `host_tx_data` drive the second's `host_rx_valid` and `host_rx_data`, and the second's `host_rx_ready` holds the first back. The second writes what it writes when a host gives it the first's answer (`two_chips_in_a_chain_pass_their_text_on_their_pins`).
- **Other uses:** speculative decoding with a small model drafting tokens that a large model checks; a model that reads images whose encoder runs on dies of its own; or several small models in a chain.

## The system, planned

`voice_plan::plan_voice` (the `voice_system` example, the Listen and Speak views' panel, `/api/voice/plan`) plans the chip as a system, with the stages of the parts it has:

1. **Each stage is measured on a few shapes of die** (1 or 16 units of 64 lanes, 2,048 to 65,536 cells), from its own microprogram, its firmware's cycles and the optimiser's counts in N6. The decoder is also measured with windows keeping 16, 32 or 64 frames. Dies run by routines get a command queue of 16 orders.
2. **`system::plan` weighs every way of grouping the stages on dies** (15 for four stages):
   - **A shared die** has its logic once, the cells and microcode of every stage on it, the vector memory of the one that needs most, and the slowest clock.
   - **Stages that run at once never share a die.** Within a turn the recogniser runs alone, before the rest; the language model, the speech writer and the decoder run at once, each taking the one before's first sentence or window while it goes on. So only the recogniser may share, with any one of the others.
   - **Each die must fit the reticle, each stage must keep up** (the writer 180 tokens a second, the decoder 75 frames: real time; a streaming recogniser each 20 ms in less than 20 ms), and the first sound must come in time.
   - **Of what meets all that, the least silicon wins**, then the fewest dies, then the soonest sound.

### What can be asked

| Option | Default | Where |
|---|---|---|
| The longest wait from the question's end to the first sound | 500 ms | `FIRST_SOUND_MS` in the example, `first_sound_ms` in the API |
| The longest question | 3 s | `SECONDS`, `seconds` (0.5 to 30) |
| Tokens of the answer's first sentence | 12 | `WORDS` |
| The most silicon in all | unlimited | `area_mm2` |
| The writer reads a speaker's profile | no | `voice`, `speaker` |
| Shapes tried, windows tried, sets of dies for a large language model | as above | `VoiceAsk` |

Plans are kept on disk by their question (`plan_voice_cached`), so asking again costs nothing; a version number in the key drops plans measured before the way of measuring changed.

### The plan

For a three-second question, a first sentence of 12 tokens and the first sound within 500 ms, with Moonshine-tiny, Qwen3-0.6B, OuteTTS and DAC:

| Grouping | Silicon | First sound |
|---|---|---|
| Each stage on its own die | 610 mm² | 472 ms |
| Recogniser with the language model | 679 mm² | 461 ms |
| Recogniser with the speech writer | 679 mm² | 491 ms |
| **Recogniser with the sound decoder** | **591 mm²** | **474 ms** |
| The other eleven | — | stages that run at once |

**Chosen: three dies and 591 mm²**:

- **Moonshine and DAC on one die** of one unit, 2,048 cells and 20-bit vectors (123 mm²), with a command queue, the decoder with windows of 54 frames keeping 32.
- **Qwen3-0.6B on a die** of 16 units and 2,048 cells (222 mm²): enough for 1,800 tokens a second, so more cells would only add silicon.
- **OuteTTS on a die** of 16 units and 8,192 cells (245 mm²), which reads the sentence sooner and leaves the decoder its time.

The planner merges what pays and keeps apart what would wait on itself; with a tighter time or other models it chooses differently:

| Parts | Models | Dies | Silicon | First output | A turn |
|---|---|---|---|---|---|
| Listening and speaking | Moonshine-tiny, Qwen3-0.6B, OuteTTS, DAC | 3 | 591 mm² | 474 ms (sound) | 6.8 J |
| The same, the writer reading a three-word profile | | 3 | 612 mm² | 493 ms (sound) | 8.1 J |
| Listening and speaking | Streaming Moonshine, Qwen3-0.6B, OuteTTS, DAC | 3 | 586 mm² | 482 ms (sound) | 5.7 J |
| Listening and speaking | Streaming Moonshine, Qwen3.8-27B, OuteTTS, DAC | 14 | 7,741 mm² | 472 ms (sound) | 132 J |
| Listening only | Moonshine-tiny, Qwen3-0.6B | 2 | 267 mm² | 46 ms (first token) | 1.0 J |
| Speaking only | Qwen3-0.6B, OuteTTS, DAC | 3 | 543 mm² | 479 ms (sound) | 3.6 J |

With the streaming recogniser, its work after the question ends takes 1.1 ms instead of Moonshine-tiny's 33.2 ms on the same die. The planner spends that margin on silicon, the writer keeping 2,048 cells, rather than on an earlier sound.

How the plan got here: before the firmware ordered the routines, and its cycles were left out, it was 569 mm². Counting them without a command queue, the CPU's 122 M cycles a window (108 M for the passes' 360,000 routines) kept the one-unit die from giving the first sound in time, and the planner took 16 units, whose maps each unit keeps a copy of: 927 mm². The queue brings it back to 591.

### Energy of a turn

`voice_plan::turn_energy`, shown in the Voice view too. Each stage's work is counted as the estimator counts a token's: every product of a weight by a value in the cell (its gates' switching, simulated) and the weight's read from its memory, for the products its routines or its program actually do. Every die leaks the whole turn, none powered off. For a three-second question and an answer of 12 tokens (about four seconds of speech), with the three-word profile:

| Stage | Work |
|---|---|
| Speech to text | 1.4 mJ |
| Language model | 29.6 mJ |
| Speech writer | 540 mJ |
| Sound decoder | 342 mJ |
| Every die's leakage, 7.5 s | 7.16 J |
| **The turn** | **8.07 J, 1.08 W on average** |

Leakage is most of it: the weights' memories leak for the whole turn while each die works for a fraction of it. Holding the weights in ROM or powering idle dies off would matter far more than any saving in the work; neither is counted here, nor the control CPUs' own work.

### A larger language model

When the language model fits no single die within the reticle, the planner measures it on the sets of dies the frontier found for Qwen3.8-27B (`VoiceAsk::large`): 12 dies of 64 × 128 units and 32,768 cells in groups of four, and the fastest ones, 12 dies of 128 × 128 units and 16 in groups of eight. Such a stage is a die of its own kind repeated, and never shares a die with another stage. A model whose weights could not fit one die even at four bits (one weight for each product of a token, in the process's densest memory) is not measured on single dies at all. A checkpoint already quantized keeps its own codes, as the frontier keeps them; the others go to 8 bits.

With Qwen3.8-27B (4-bit, 8-bit in places) and the streaming recogniser, for the same turn:

| Dies | What | Silicon |
|---|---|---|
| 1 of 1 × 64 units, 2,048 cells | Streaming Moonshine and DAC | 140 mm² |
| **12 of 64 × 128 units, 32,768 cells** | **Qwen3.8-27B, groups of four, 2,518 tokens a second** | **12 × 615 mm²** |
| 1 of 16 × 64 units, 2,048 cells | OuteTTS | 224 mm² |
| **14 dies** | **First sound in 472 ms** | **7,741 mm²** |

The smallest set is enough: the faster ones only add silicon, since the answer's first sentence is ready long before the sound needs it. Each of these dies is 615 mm² for a context of 1,024 tokens, sized to the prompt; the frontier's 641 mm² are for 8,192 tokens. A turn costs 132 J, almost all of it leakage (130 J): 14 dies of memory leaking for 7.5 s.

### Simplifications

- A shared die's cell memories are added as each stage's were measured, though one memory of their sum would have a little less periphery.
- A recogniser measured with 16-bit vectors keeps its cycles on the decoder's 20-bit die.
- The speech dies are routines over rows, and each unit of a die keeps its own copy of the maps. Splitting them among the units would need writes only the owning unit makes and reads from a unit chosen as it runs: new datapaths across units, which would not make the products faster, since they run in the cells. So the planner weighs the copies against the speed instead.
- The command queue's flip-flops and the CPU's energy are not in the area and energy counted.

## Speaking more naturally

- **Sampling.** OuteTTS draws its tokens as its authors recommend: temperature 0.4, top-k 40, top-p 0.9, min-p 0.05, after a repetition penalty of 1.1 over the last 64 tokens (`speak::Sampling::outetts`, from a seed). Greedy stays for comparisons with Transformers, which it matches token for token.
- **A voice profile.** OuteTTS speaks more steadily in a given voice: its prompt starts with a sentence said in that voice, word by word with its duration, loudness, brightness, pitch and codes (`SpeechVocabulary::prompt_with`). The prompt is OuteTTS's to the token: 1,840 tokens for "Hello, I am a chip.", the same as Transformers' tokenizer gives. It costs those tokens before each sentence: with the whole profile no system gives the first sound within 500 ms (within 1,500 ms the planner chooses three dies and 720 mm², and the first sound in 1,310 ms). So the writer reads only the profile's first three words (`voice::PROFILE_WORDS`, `Speaker::first_words`): 197 tokens instead of 1,840, and Moonshine still hears "Hello, I am a chip." With it the plan is three dies, 612 mm², and the first sound in 493 ms.
- **A conversation.** The language model keeps its cache between questions (`voice::Conversation`): on top of it goes what the chat template writes after the last answer (its end) and the new question (`PromptStyle::next_turn`), as a chip that holds a conversation is sent it, so each question costs only its own tokens. With Qwen3-0.6B, "What is the capital of France?" and then "And of Italy?" gives "The capital of Italy is Rome." from the cache, and the scores after it are those of the whole conversation fed from the start. A template that does not write an earlier answer as it was cannot go on over a cache, and is refused.

## Using it

### The demo

A question spoken in, the answer spoken out, every stage in turn:

```bash
cargo run --release -p sim --example talk -- MOONSHINE_DIR LLM_DIR OUTETTS_DIR DAC_DIR question.wav answer.wav [die 20] [sample SEED] [voice]
```

`die BITS` runs every stage in the die's arithmetic with vectors of that width (20 or more, for the decoder); `sample SEED` draws the speech as OuteTTS does instead of greedily; `voice` speaks in the profile kept in OuteTTS's `speakers/`. Any recogniser folder works, streaming Moonshine's too. With macOS's `say` asking "What is the capital of France?", in floating point:

1. Moonshine hears "What is the capital of France?" (0.8 s).
2. Qwen3-0.6B answers "The capital of France is Paris." (7 tokens).
3. OuteTTS writes 524 tokens, 3.04 seconds of speech, and DAC decodes them; Moonshine, listening to the answer, hears "The capital of France is Paris."

With `die 20` the turn goes the same way: the same question heard, the same answer, 530 tokens of speech (3.08 seconds) that Moonshine hears as "The capital of France is Paris." On a computer that takes about five minutes, since the simulation goes integer by integer.

### The other examples

| Command | What it does |
|---|---|
| `listen MODEL_DIR clip.wav [golden.json] [bits]` | Speech to text in floating point and in the die's arithmetic (a streaming model as its firmware would run it), checked against a reference from `tools/golden` |
| `speak OUTETTS_DIR DAC_DIR "Text." out.wav [golden.json] [die BITS] [speaker FILE [words N]]` | Text to speech, optionally against Transformers' tokens and scores; `PROMPT_IDS=file` writes the prompt's tokens |
| `dac_check DAC_DIR golden.json [bits] [out.wav]` | DAC's decoder against Transformers' sound, and in the die's arithmetic (`ACCUMULATOR=bits` for another accumulator) |
| `voice_system MOONSHINE LLM OUTETTS DAC [FIRST_OUTPUT_MS] [SECONDS] [WORDS] [voice]` | The system planned, with every grouping weighed and the energy of a turn; `-` for Moonshine plans a chip that only speaks, `-` for OuteTTS and DAC one that only listens |

Each runs as `cargo run --release -p sim --example NAME -- ARGS`.

### References against Transformers

| Script | What it writes |
|---|---|
| `tools/golden/moonshine.py MODEL_DIR clip.wav out.json` | Moonshine's frames, tokens and first scores |
| `tools/golden/moonshine_streaming.py MODEL_DIR clip.wav out.json` | The same for streaming Moonshine (Transformers 5.24.0), the clip cut to whole 20 ms |
| `tools/golden/speak.py OUTETTS_DIR DAC_DIR "text" out.json out.wav` | OuteTTS's prompt, tokens and scores, and DAC's sound |
| `tools/golden/speaker_prompt.py OUTETTS_DIR speaker.json "Text."` | OuteTTS's prompt with a speaker's profile, as its own package writes it |
| `tools/convert/dac.py checkpoint.pth OUT_DIR` | DAC.speech in safetensors |

Each runs with `uv run`, which installs its dependencies (PyTorch, Transformers) apart from the project.

### In the application

Each part has a view, shown only when the project's chip has it:

- **Listen** records a question with the microphone or opens a recording, and shows what Moonshine heard and the language model's answer; when the chip also speaks, the answer is said too, a whole spoken turn.
- **Speak** takes a text, and says it, or first has the language model answer it and says the answer.

Each runs its stages in floating point or in the die's arithmetic, and the conversation goes on until "New conversation". The language model is the project's own when its weights are on this computer. The side panel chooses the models of the parts the chip has and plans the chip: the limit on the first output (the first sound, or the answer's first token when it only listens), the longest question and the most silicon; it shows the dies chosen, each stage's timing, the energy of a turn and every grouping weighed, with why. The project wizard's "A voice assistant" asks which parts the chip has and leads there. The Models list says which folders listen, speak and make the sound.

The engine's API behind it:

| Endpoint | Body | Gives |
|---|---|---|
| `POST /api/voice` | The question as `wav` (base64, with the recogniser `listen`) or as `text`; `think` (the language model, if it answers), `speak` and `sound` (both, if the answer is said); `bits` (or null for floating point), `fresh` (a new conversation), `speaker` | Starts a turn; one at a time |
| `GET /api/voice` | | `state` (`running` with `phase` and what each stage gave so far, `done` with the result, `error`); when idle, which parts the engine was built with (`listen`, `speak`) |
| `POST /api/voice/plan` | `think`, and `listen` and/or `speak` with `sound`; `first_output_ms`, `seconds`, `area_mm2`, `speaker` | Starts a plan |
| `GET /api/voice/plan` | | `state`, and when done the plan (`dies`, `stages`, `energy`, `first_output_ms`), every grouping weighed and which parts it has |

A folder must be in the catalogue and do its part of the turn: listen (Moonshine), speak (a language model that writes speech codes) or sound (a converted DAC). A turn that cannot be (a clip with no recogniser, a writer without a decoder, a written question with nothing to say, a part the engine was built without) is refused with why. The routes exist only in an engine built with listening, speaking or both.

**On an FPGA** none of the voice models fits today. The chip keeps its weights in its cells' memory, and Moonshine-tiny alone has 27 million (about 108 Mbit in 4-bit codes), against 3.9 Mbit of internal memory in the largest Cyclone IV (EP4CE115) and 0.41 in the catalogue's EP4CE10, whose logic is also too small for the smallest chip (54,000 logic elements against 10,300). Larger boards fall short too: the Kria KV260 has 23.1 Mbit. Only a toy recogniser like the tests' fits.

## Tests

Each test file needs the part it tests (`#![cfg(feature = "listen")]` or `"speak"`). The quick suite (`scripts/check.sh`) runs the unmarked ones and builds the engine with neither part, with listening alone and with speaking alone; `--full` adds those marked slow; two need a switch as well. Tests that read a real model skip themselves when it is not in `~/.llm2chip/models`.

| Test | What it checks | When |
|---|---|---|
| `moonshine.rs`: `the_toy_recogniser_makes_a_frame_every_384_samples_and_scores_every_token`, `moonshine_in_the_die_s_arithmetic_scores_as_floating_point_does`, `a_chip_for_longer_clips_scores_a_shorter_one_as_one_built_for_it` | The toy recogniser's reference, the exact machine within 2.6 % of it, any clip length | Quick |
| `moonshine.rs`: `moonshine_on_the_die_s_gates_gives_the_exact_machine_s_scores`, `a_chip_for_longer_clips_takes_a_shorter_one_on_its_gates` | The toy on the gates, bit for bit and cycle for cycle (280 routines, 280,614 cycles) | Full |
| `moonshine.rs`: `the_firmware_s_schedule_runs_what_the_host_runs`, `the_firmware_transcribes_a_clip_as_the_host_ordering_the_routines`, `the_real_recogniser_s_schedule_fits_its_chip_image` | The schedule against the host's order for four clip lengths; the firmware at instruction level; Moonshine-tiny's image | Quick |
| `moonshine.rs`: `the_whole_chip_with_a_command_queue_transcribes_in_gates_as_at_instruction_level` | The whole toy chip with its queue, CPU and die in gates: the same bytes in 1,192,263 cycles | Full |
| `moonshine.rs`: `one_real_moonshine_layer_on_the_die_s_gates_gives_the_exact_machine_s_scores` | One real layer of Moonshine-tiny in gates (13.6 M cycles, about 20 minutes) | `LLM2CHIP_REAL_MOONSHINE=1` |
| `moonshine_stream.rs`: `a_streaming_window_sees_its_frames_and_no_more`, `streaming_moonshine_in_the_die_s_arithmetic_scores_as_floating_point_does`, `every_kind_of_window_scores_as_floating_point_does`, `the_firmware_listens_while_the_speech_comes` | Windows; the exact machine within 0.2 % for every kind of window; the firmware as the host | Quick |
| `moonshine_stream.rs`: `streaming_moonshine_on_the_die_s_gates_gives_the_exact_machine_s_scores` | The toy on the gates in step with the exact machine (900,912 cycles) | Full |
| `dac.rs`: `the_die_s_arithmetic_decodes_codes_as_floating_point_does`, `decoding_window_by_window_gives_the_whole_sound`, `the_firmware_decodes_window_by_window_as_the_host_does`, `the_real_decoder_s_schedule_fits_its_chip_image` | Toy decoders 36 to 62 dB below floating point; windows give the whole sound; the firmware with and without a queue, its cycles against the count; DAC.speech's image | Quick |
| `dac.rs`: `dac_s_decoder_on_the_die_s_gates_gives_the_exact_machine_s_sound` | The toy decoder on the gates (322 routines, 70,624 cycles) | Full |
| `chip.rs`: `text_goes_through_a_chip_with_a_command_queue_in_gates`, `two_chips_in_a_chain_pass_their_text_on_their_pins` | The queue in gates on the test model's whole chip; two chips in a chain | Quick; full |
| `conversation.rs`: `a_conversation_goes_on_from_its_cache` | Qwen3-0.6B's conversation from its cache, against the whole conversation | Full |
| In the crates: `system.rs` (five), `speak.rs`, `voice.rs`, `netlist::schedule`, `dac_chip` (the sine), `moonshine_stream_chip` (asinh) | The planner's rules, sampling, sentences, the schedule's records, the tables | Quick |

## Programs with parts

Before voice, `Program` had two fields named after images: `image` (Janus's heads, which write and read the codes of a second codebook) and `vision` (routines that go through a store of rows: the image encoder, Show-o's image half and MAGVIT-v2). Neither is about images: a speech decoder's codes are a second codebook, and a speech encoder's frames are a store of rows. They are now **parts** of a program, with names that say what they do:

- `CodeHead`: a second codebook, with the ops that score its codes and the ones that turn a code into the model's input.
- `RowPart`: routines over a store of rows, called by the host or the firmware in their order, each with a row as the token; and the rows of any table of its own (`tables`).

`program.codes()` and `program.rows()` find them. Nothing else changed: the lowered microcode of every model is the same, word for word.

## Where each piece lives

Voice adds files; the core only gains general things.

| Piece | Where | Kind |
|---|---|---|
| `FN_SIN` table and `wrap_left` bit | `crates/netlist` (die and assembler) | General: any periodic function, built only where used |
| `FN_ASINH` and `Asinh` | `crates/netlist/src/block.rs`, `crates/compiler`, `crates/sim/src/lower.rs`, `crates/sim/src/exec.rs` | General: built only where used |
| `Sin` and `MulWeight` operations | `crates/compiler`, `crates/sim/src/lower.rs`, `crates/sim/src/exec.rs` | General |
| Programs with parts (`Program::parts`), tables of a program's own rows (`RowPart::tables`) | `crates/compiler` | General |
| Schedules the firmware follows | `crates/netlist/src/schedule.rs`, `firmware/src/main.rs` (`follow`, `repeat_calls`), `sim::chip::{RoutinesDie, follow_schedule}` | General: any program of routines |
| Command queue before the die | `crates/netlist/src/chip.rs` (gates and `ChipModel::with_queue`), `DieFeatures::queue_bits`, `firmware/src/main.rs` | General: built only where asked |
| The firmware's cycles | `sim::chip::{firmware_cycles, lean_cycles}` | General |
| Systems of stages and their plan | `crates/sim/src/system.rs`, `sizing::routines_die` | General: several programs on several dies |
| Moonshine: floating-point reference (both kinds) | `crates/sim/src/listen/moonshine.rs` | Voice |
| Moonshine on the die, its schedule | `crates/sim/src/listen/moonshine_chip.rs` | Voice |
| Streaming Moonshine on the die, its schedule | `crates/sim/src/listen/moonshine_stream_chip.rs` | Voice |
| DAC: reference, on the die, its schedule | `crates/sim/src/speak/dac.rs`, `crates/sim/src/speak/dac_chip.rs` | Voice |
| OuteTTS's prompt, codes and sampling | `crates/sim/src/speak/outetts.rs` | Voice |
| Sound in and out, WAV | `crates/sim/src/audio.rs` | Voice |
| A spoken turn's stages, the conversation | `crates/sim/src/voice.rs` | Voice |
| The voice system planned | `crates/sim/src/voice_plan.rs` | Voice |
| Examples | `crates/sim/examples/{talk, listen, speak, dac_check, voice_system}.rs`, each with the features it needs (`required-features`) | Voice |
| Tests | `crates/sim/tests/{moonshine, moonshine_stream, dac, conversation}.rs` | Voice |
| References and conversion | `tools/golden/{moonshine, moonshine_streaming, speak, speaker_prompt}.py`, `tools/convert/dac.py` | Voice |
| The engine's jobs and the application's views | `crates/server/src/voice_job.rs`, `crates/server/src/catalog.rs` (`voice_role`), `web/src/voice/` (Listen and Speak) | Voice |
| The project's options | `ChipParams::{speech_in, speech_out}` (`crates/arch`), `web/src/params.ts`, `viewShown` in `web/src/store.ts` | Voice |
| Build features | `[features]` of `crates/sim` and `crates/server`; the combinations in `scripts/check.sh` | Voice |

## Limits and what is next

- **The four voice chips are joined by the host in the simulation.** Each runs by itself on its firmware, and two chips in a chain are checked in gates, but the recogniser, the language model, the writer and the decoder have not been chained pin to pin.
- **The speech dies keep a copy of their maps in every unit** (see [Simplifications](#simplifications)); splitting them needs datapaths across units and is not built.
- **The firmware has 436 bytes of program memory left.** Another feature means trimming the code or a larger program memory, which changes every chip's area.
- **What is not counted:** the command queue's flip-flops in the area, the control CPUs' energy, and any power saving from ROM weights or idle dies.
- **A die that speaks needs 20-bit vectors;** at 16 the sound is heard wrongly.
- **Only English models** have been tried, and only Moonshine-tiny and its streaming model, OuteTTS-1.0-0.6B and DAC.speech's 1.5 kbps decoder.
- **Speculative decoding with a separate draft model** uses the same systems of stages and is not built yet; the MTP head (`DraftProgram`) is its cousin inside one model.
- **The optimiser and the application plan general systems only for voice:** `system::plan` takes any stages, but only the voice view asks it.
- **No voice model fits an FPGA** (see above).

## History

| Step | What | Status |
|---|---|---|
| 1 | This document | Done |
| 2 | Programs with parts, with identical microcode | Done: the 890 microprograms the quick tests lower hash the same before and after |
| 3 | Cross-attention and LayerNorm without bias | Done, with no change to the die or the compiler |
| 4 | One-dimensional strided convolution | Done: `LoadRows`, with the first row read as the token |
| 5 | Systems of stages | Done (`sim::system`) |
| 6 | Moonshine on a toy checkpoint: floating-point reference, exact machine, gates | Done: within 2.6 % of floating point (the die's function tables; the products agree to millionths), the gates bit for bit |
| 7 | Moonshine with the real weights | Done: frames within 0.0002 % of Transformers' and scores within 0.0001 %, the same tokens; with 16-bit vectors the same text, "Hello, this chip can listen and speak." |
| 8 | `FN_SIN`, `wrap_left` and the DAC decoder | Done: Transformers' sound sample for sample; toy decoders 36 to 62 dB below floating point at 20 bits, the gates bit for bit |
| 9 | OuteTTS with the real weights | Done: the same prompt and 317 tokens as Transformers for "Hello, I am a chip.", scores within 0.0005 %; with 20-bit vectors the first 91 tokens match, and the die's sound is heard as the same sentence |
| 10 | The demo | Done in floating point and in the die's arithmetic |
| 11 | The system planned | Done |
| 12 | Voice in the application | Done: the Voice view and the wizard's "A voice assistant" |
| 13 | OuteTTS's sampling, a voice profile, a conversation | Done |
| 14 | Moonshine and DAC on their own firmware | Done: schedules, bit for bit the host's at instruction level, their CPU cycles in the plan |
| 15 | One real layer of Moonshine-tiny in gates | Done: the stem, the first encoder and decoder layers and 64 rows of the vocabulary with the model's own weights, a two-frame clip and two tokens, 160 routines, 13,568,264 cycles, bit for bit |
| 16 | A language model of several dies in the plan | Done |
| 17 | Recognising while the question is spoken | Done: streaming Moonshine, the `asinh` table and the `STREAM` record; with 16-bit vectors it writes what Transformers writes |
| 18 | A command queue before the die | Done: in gates and at instruction level cycle for cycle; the plan back to 591 mm² |
| 19 | Listening and speaking, two optional parts | Done: project options, views, planner, API and build features of their own; a chip with neither has nothing of voice |
