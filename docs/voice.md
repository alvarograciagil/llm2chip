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

Each stage is a program of its own on dies of its own. Between stages only token ids travel: a few bytes per second, far below what the links between dies already carry. So splitting the stages over dies costs nothing measurable in latency, and each stage's dies can have their own shape.

In a conversation what matters is not tokens per second but **how long until the first sound**, which should stay below about half a second. It adds up three things:

1. **The end of the utterance:** detecting silence and finishing the transcription. Moonshine processes audio of any length, so it can run while the person speaks.
2. **The first token of the language model:** the prompt's prefill. The optimiser already has a limit for it; in a voice design it becomes the main goal.
3. **The first sentence of speech:** synthesis can start as soon as the model writes its first sentence, while it keeps writing.

Speech needs few tokens: DAC codes come at 75 frames per second (the 24 kHz rate over the decoder's upsampling of 8 × 5 × 4 × 2 = 320), two codebooks per frame at 1.5 kbps. That is 150 codes per second, against thousands of tokens per second on the dies. A voice die is sized for latency, not for speed.

## The models

| Stage | Model | What it is | Size | Licence |
|---|---|---|---|---|
| Speech to text | [Moonshine](https://huggingface.co/UsefulSensors/moonshine-tiny) tiny / base | Encoder–decoder transformer over the raw 16 kHz waveform | 27 M / 61 M parameters (108 / 246 MB in F32) | MIT |
| Text to speech | [OuteTTS-1.0-0.6B](https://huggingface.co/OuteAI/OuteTTS-1.0-0.6B) | A Qwen3-0.6B (`model_type: qwen3`, 28 layers, hidden 1,024, vocabulary 157,760) that writes the codec's codes as tokens of its vocabulary | 0.6 B parameters (BF16) | Apache-2.0 |
| Codes to sound | [DAC.speech v1.0](https://huggingface.co/ibm-research/DAC.speech.v1.0) | The decoder of the Descript Audio Codec, 24 kHz, fine-tuned for speech | 296 MB per checkpoint (`.pth`) | CDLA-Permissive-2.0 |

Why these:

- **Moonshine takes the waveform, not a spectrogram.** Whisper needs a log-mel spectrogram, that is, an FFT on the host or in the firmware, and always processes 30-second windows. Moonshine's front end is three convolutions over the samples, which the die already does, and its length follows the audio. Its blocks are the usual ones: LayerNorm without bias, attention with partial RoPE (90 % of each head in tiny, 62 % in base), GELU in the encoder's MLP and a gated SiLU in the decoder's. Eight heads, all with their own keys and values.
- **OuteTTS writes speech as ordinary tokens.** Its language model is a plain Qwen3, a family the compiler already supports, with the audio codes as extra rows of its vocabulary. The language model needs no new head and no new field: only the decoder that turns codes into sound is new.
- **DAC's decoder is only convolutions.** No attention and no recurrence: a convolution in, four blocks that each upsample (a transposed convolution) and refine (three residual units with dilations 1, 3 and 9), and a convolution out with tanh. That is the same kind of work as MAGVIT-v2's decoder, which the die already runs in gates, but in one dimension.

Facts still to confirm with the real files: how OuteTTS names its code tokens and how many codebooks it uses (the 1.5 kbps checkpoint has two), and the exact layout of DAC.speech's weights, which come as a PyTorch pickle and must be converted to safetensors by a script in `tools/convert`, as `llama2c.py` does for llama2.c models.

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

The decoder is a causal language model with one more attention per layer: **cross-attention**, whose queries come from the token and whose keys and values come from the encoder's frames. This needs no new operation:

- **The keys and values:** each encoder frame, once the encoder is done, goes through every layer's cross-attention key and value projections, and `KvAppend` stores them in a key slot of their own, at the frame's index. The routine is called once per frame, as the image encoder's are.
- **The attention:** `AttnSum` over that slot. The only difference is how far it reads: up to the number of frames, not up to the token's position. That is a new field of `AttnSum` that says where its length comes from (`Span`): the position, a sliding window, or the rows a store holds.
- **Positions:** RoPE in self-attention only, partial, as Qwen3.5 already has (`rotary` smaller than the head).
- **The gated MLP:** `fc1` makes both halves in one product, `Slice` separates them, then SiLU and the product, as the other models' MLP.
- **LayerNorm without bias:** `LayerNorm` reads `{weight}.bias` only if the checkpoint has it.

### OuteTTS's language model

Nothing new: it is a Qwen3 and compiles as one. The code tokens are rows of its vocabulary, so it samples them with its own head. The host or the firmware recognises them by their range and passes them to the decoder.

### DAC's decoder

| Layer | On the die |
|---|---|
| The codes' vectors: each codebook's 8-value row, projected to 1,024 by a 1×1 convolution, the codebooks added | `EmbRead` from each codebook, a product each, `Add` |
| Convolution with kernel 7 | `LoadRows` with offsets 0…6 over a store with a border of three zero rows on each side |
| Transposed convolution with stride s and kernel 2s | Split by phase: each of the s output rows of an input row is an ordinary convolution of its neighbours with a kernel of its own. A routine per phase, storing at row `s · t + phase` |
| Dilated convolution (1, 3, 9) | `LoadRows` with offsets spaced by the dilation |
| Snake: `x + sin²(αx) / α`, α per channel | `MulElem` by α, a sine from a function table, its square, `MulElem` by 1/α, `Add`. The one piece of hardware it needs: a new function table, `FN_SIN` |
| Final tanh | `SoftCap`'s tanh |

Weight normalisation (`weight_norm`) is folded into plain weights when the checkpoint is converted.

`FN_SIN` is the only change the decoder needs in gates. Like the other tables it is built only when a microprogram uses it (`DieFeatures::ops`), so dies without a decoder do not change. Its cost in area and clock has to be measured with the one-point frontier before it goes up, following the rule that changes must be a win everywhere.

## Several programs on several dies

Until now all the dies of a design ran parts of one program: layers in sequence, heads in parallel, or both. Voice needs dies that run **different programs**: some the speech recogniser, some the language model, some the synthesiser. That capability is general:

- **A system is a list of stages**, each one a program with its own dies, shape and split. Its area is the sum; its speed is each stage's; its latency is the sum along the path of one answer.
- **Between stages only tokens travel.** In this first step the host moves them, as it moves the prompt and the generated text today. A hardware handoff would need a token port between dies of different stages: a small queue of token ids on the link, and the receiving stage's firmware reading it as if it were the host's input. Nothing more, because the vectors never leave a stage.
- **Other uses:** speculative decoding with a small model drafting tokens that a large model checks (the large one checks several at once, so the answer comes faster); a model that reads images whose encoder runs on dies of its own; or several small models in a chain.

Speculative decoding with a separate draft model is not built yet. The MTP head (`DraftProgram`) is its cousin inside one model.

## Where each piece lives

Voice adds files; the core only gains general things.

| Piece | Where | Kind |
|---|---|---|
| `Span` in `AttnSum` (cross-attention) | `crates/compiler`, `crates/sim/src/lower.rs` | General: any encoder–decoder |
| `LayerNorm` with no bias | `crates/compiler`, `crates/sim/src/lower.rs` | General |
| `FN_SIN` table | `crates/netlist`, `crates/sim/src/lower.rs` | General: Snake, periodic activations |
| Programs with parts (`Program::parts`) | `crates/compiler` | Refactor: what were `image` and `vision` |
| Systems of stages | `crates/sim/src/system.rs` | General: several programs on several dies |
| Moonshine: floating-point reference | `crates/sim/src/moonshine.rs` | Voice |
| Moonshine on the die | `crates/sim/src/moonshine_chip.rs` | Voice |
| DAC: reference and on the die | `crates/sim/src/dac.rs`, `crates/sim/src/dac_chip.rs` | Voice |
| DAC weights to safetensors | `tools/convert/dac.py` | Voice |
| Demo: a WAV in, a WAV out | `crates/sim/examples/talk.rs` | Voice |
| Tests | `crates/sim/tests/moonshine.rs`, `dac.rs`, `system.rs` | Voice and general |

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
| 3 | Cross-attention (`Span`), LayerNorm without bias; exact machine and gates bit for bit on a toy model | Planned |
| 4 | One-dimensional strided convolution checked with `LoadRows` | Planned |
| 5 | Systems of stages: sizing, optimiser, host moving tokens | Planned |
| 6 | Moonshine: compiler, reference, exact machine and gates on a toy model | Planned |
| 7 | Moonshine with the real weights | Needs a download (108 MB) |
| 8 | `FN_SIN` and the DAC decoder on a toy model | Planned |
| 9 | OuteTTS and DAC with the real weights | Needs downloads (about 1.5 GB, and PyTorch for the conversion) |
| 10 | The demo, a WAV in and a WAV out | After 7 and 9 |
