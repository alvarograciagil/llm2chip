//! Streaming Moonshine as the chip's program: a speech recogniser that listens while the
//! question is spoken. Every 20 ms of speech (four frames of the front end) the host sends
//! the samples, and the chip takes them through the front end and moves every encoder layer
//! one frame on, each as far as its window lets it: a layer that sees 3 frames ahead runs
//! 3 frames behind the one before. When the question ends only the last frames' layers are
//! left, then the decoder writes the text.
//!
//! - **Front end, a frame of 80 samples at a time:** CMVN as a LayerNorm whose weight is the
//!   compression's factor (the samples scaled first so that the die's epsilon is the
//!   model's), the `asinh` table, a product and SiLU, stored with four rows of zeros before
//!   for the causal convolutions; each convolution a routine that reads five rows from the
//!   first and one that stores.
//! - **Encoder layers:** keys and values go into rings as wide as the layer's window
//!   (`KvAppend` with a window), so attention at the window's last frame sees exactly the
//!   window. The clip's last frames see fewer frames ahead: a layer that sees `a` ahead also
//!   keeps rings `1 … a` frames narrower, and frame `last − j` attends at the last frame in
//!   the ring `a − j` narrower, which holds exactly its keys.
//! - **Decoder:** Moonshine's, with each frame's learned position added before its keys.
//!
//! The chip follows a schedule (`schedule`): it works out how far behind each layer runs
//! and orders every routine, so the host only sends samples and reads the text.

use crate::chip::RoutinesDie;
use crate::lower::Layout;
use crate::listen::moonshine::{MoonshineConfig, Streaming, CMVN_EPSILON};
use crate::listen::moonshine_chip::{decoder_routines, Builder, DecoderRoutines, Routine};
use crate::vm::BlockVm;
use crate::weights::Weights;
use compiler::{Buf, KvShape, Op, Part, Program, RowPart};
use netlist::block::{encode_vector, exponent_register};
use netlist::schedule::{Record, FIRST_FREE, NO_ROUTINE, POSITION, TOKEN, ZERO};
use netlist::DieConfig;
use std::ops::Range;
use std::path::Path;
use std::sync::Arc;

/// The die's LayerNorm epsilon (Moonshine's) over the front end's (CMVN's): samples scaled
/// by the square root of this give CMVN's result with the die's epsilon.
const EPSILON_RATIO: f32 = 1e-5 / CMVN_EPSILON;

/// A routine that reads and one that stores.
type Pair = (Range<usize>, Range<usize>);

/// An encoder layer's routines.
#[derive(Clone, Debug)]
struct Layer {
    /// Its keys, values and queries for a frame, at that frame.
    project: Range<usize>,
    /// The rest of it for a frame, at the last frame of its window.
    rest: Range<usize>,
    /// The rest for the clip's last frames: for frame `last − j`, `tail[j]`, at the last frame.
    tail: Vec<Range<usize>>,
    /// The frames it sees ahead.
    ahead: usize,
}

/// What following the schedule costs (`StreamingChip::timing`): die and firmware cycles.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Timing {
    /// The longest 20 ms while the speech comes in: it must take less than 20 ms.
    pub step: u64,
    /// From the speech's end to the first token's scores.
    pub after: u64,
    /// Each token after.
    pub token: u64,
    /// Products of a weight by a value up to the first token, and for each token after.
    pub clip_macs: u64,
    pub token_macs: u64,
}

/// Streaming Moonshine as the chip's program, for clips of up to some number of frames.
pub struct StreamingChip {
    pub config: MoonshineConfig,
    streaming: Streaming,
    program: Program,
    layout: Layout,
    vm: BlockVm,
    bits: u32,
    weights: Arc<Weights>,
    /// The most frames of 20 ms it takes.
    frames: usize,
    input: Buf,
    embed: Range<usize>,
    conv1: Pair,
    conv2: Pair,
    layers: Vec<Layer>,
    cross_keys: Range<usize>,
    decoder: DecoderRoutines,
    logits: Buf,
}

impl StreamingChip {
    /// The checkpoint in `dir` for clips of up to `frames` frames of 20 ms, on a die like
    /// `die` (it gets the `asinh` table); with the weights in its cells and caches for
    /// `tokens` of text when `tokens` is given, as a chip holds them.
    pub fn new(dir: &Path, frames: usize, die: DieConfig, tokens: Option<usize>) -> Result<Self, String> {
        Self::from_parts(MoonshineConfig::read(dir)?, Weights::open(dir)?, frames, die, tokens)
    }

    /// The exact machine's die with vectors of `bits` bits and cells as wide. The host and
    /// the firmware address its vector memory in 32 bits, the register above the element:
    /// elements enough for the longest store, registers in what is left.
    pub fn exact(dir: &Path, frames: usize, bits: u32) -> Result<Self, String> {
        let base = crate::exact::exact_config(bits);
        let config = MoonshineConfig::read(dir)?;
        let longest = (4 + 4 * frames) * config.hidden;
        let index_bits = (usize::BITS - longest.leading_zeros()).max(16);
        let die = DieConfig { cell: netlist::CellConfig { weight_bits: bits, accumulator_bits: 2 * bits + 8, ..base.cell }, index_bits, reg_bits: 32 - index_bits, ..base };
        Self::new(dir, frames, die, None)
    }

    pub fn from_parts(config: MoonshineConfig, weights: Weights, frames: usize, die: DieConfig, tokens: Option<usize>) -> Result<Self, String> {
        let streaming = config.streaming.clone().ok_or("not a streaming Moonshine")?;
        if frames == 0 {
            return Err("a chip for no frames".into());
        }
        let die = DieConfig { function_select_bits: die.function_select_bits.max(4), ..die };
        let (h, heads, head_dim) = (config.hidden, config.heads, config.head_dim);
        let width = streaming.frame;
        let mut b = Builder { weights: &weights, buffers: Vec::new(), ops: Vec::new(), derived: Vec::new() };
        let p = |n: &str| format!("model.encoder.embedder.{n}");

        // The front end, a frame of samples at a time.
        let k = b.floats(&p("comp.log_k"))?.first().ok_or("log_k is empty")?.exp();
        let scale = b.derive("moonshine.cmvn.scale".into(), vec![width], vec![EPSILON_RATIO.sqrt(); width]);
        b.derive("moonshine.cmvn.weight".into(), vec![width], vec![k; width]);
        let input = b.buf(width);
        let embedded = b.buf((4 + 4 * frames) * h);
        let embed = b.routine(|b| {
            let scaled = b.buf(width);
            b.op("scale", Op::MulWeight { src: input, weight: scale, dst: scaled });
            let normed = b.layer_norm("moonshine.cmvn", scaled, width);
            let squashed = b.buf(width);
            b.op("asinh", Op::Asinh { src: normed, dst: squashed });
            let y = b.product(&p("linear"), squashed, h, None);
            let bent = b.buf(h);
            b.op("SiLU", Op::Silu { src: y, dst: bent });
            b.op("store", Op::StoreRow { src: bent, store: embedded });
            Ok(())
        })?;
        // The causal convolutions: one routine reads five rows from the first, one stores.
        let causal = |b: &mut Builder, name: &str, src: Buf, inputs: usize, rows: usize, silu: bool| -> Result<(Buf, usize, Pair), String> {
            let (matrix, outputs) = b.kernel(name, 5, 5)?;
            let near = b.buf(5 * inputs);
            let result = b.buf(outputs);
            let dst = b.buf(rows * outputs);
            let read = b.routine(|b| {
                b.op("read rows", Op::LoadRows { store: src, dst: near, offsets: (0..5).collect() });
                let product = b.product(&matrix, near, outputs, Some(format!("{name}.bias")));
                if silu {
                    b.op("SiLU", Op::Silu { src: product, dst: result });
                } else {
                    b.op("keep", Op::Copy { src: product, dst: result });
                }
                Ok(())
            })?;
            let store = b.routine(|b| {
                b.op("store", Op::StoreRow { src: result, store: dst });
                Ok(())
            })?;
            Ok((dst, outputs, (read, store)))
        };
        let (convolved, wide, conv1) = causal(&mut b, &p("conv1"), embedded, h, 4 + 2 * frames, true)?;
        let (x_store, _, conv2) = causal(&mut b, &p("conv2"), convolved, wide, frames, false)?;

        // The encoder's layers, each with its rings.
        let unit = |b: &mut Builder, name: &str| -> Result<String, String> {
            let gamma = b.floats(&format!("{name}.gamma"))?;
            let weight = gamma.iter().map(|g| g + 1.0).collect();
            b.derive(format!("{name}.unit.weight"), vec![gamma.len()], weight);
            Ok(format!("{name}.unit"))
        };
        let queries = b.buf(frames * h);
        let mut slots = 0;
        let mut layers = Vec::new();
        for layer in 0..config.encoder_layers {
            let l = |n: &str| format!("model.encoder.layers.{layer}.{n}");
            let (before, ahead) = (streaming.windows[layer].0, streaming.ahead(layer));
            // Rings from the window's width down to `before`: the narrower ones for the last frames.
            let rings: Vec<(usize, usize)> = (0..=ahead).map(|j| (slots + j, before + ahead - j)).collect();
            slots += rings.len();
            let (first, second) = (unit(&mut b, &l("input_layernorm"))?, unit(&mut b, &l("post_attention_layernorm"))?);
            let project = b.routine(|b| {
                let x = b.buf(h);
                b.op("read frame", Op::LoadRow { store: x_store, dst: x });
                let normed = b.layer_norm(&first, x, h);
                let q = b.linear(&l("self_attn.q_proj"), normed, h);
                let k = b.linear(&l("self_attn.k_proj"), normed, h);
                let v = b.linear(&l("self_attn.v_proj"), normed, h);
                for (slot, window) in &rings {
                    b.op("store key and value", Op::KvAppend { slot: *slot, key: k, value: v, window: Some(*window) });
                }
                b.op("store query", Op::StoreRow { src: q, store: queries });
                Ok(())
            })?;
            let rest = |b: &mut Builder, (slot, window): (usize, usize)| {
                b.routine(|b| {
                    let q = b.buf(h);
                    b.op("read query", Op::LoadRow { store: queries, dst: q });
                    let attended = b.buf(h);
                    b.op("attention", Op::AttnSum { slot, query: q, head_dim, scale_dim: head_dim, heads, kv_heads: heads, window: Some(window), softcap: None, dst: attended });
                    let out = b.linear(&l("self_attn.o_proj"), attended, h);
                    let x = b.buf(h);
                    b.op("read frame", Op::LoadRow { store: x_store, dst: x });
                    let x1 = b.buf(h);
                    b.op("residual", Op::Add { a: x, b: out, dst: x1 });
                    let normed = b.layer_norm(&second, x1, h);
                    let inner = b.linear(&l("mlp.fc1"), normed, config.intermediate);
                    let bent = b.buf(config.intermediate);
                    b.op("GELU", Op::Gelu { src: inner, approximate: false, dst: bent });
                    let out = b.linear(&l("mlp.fc2"), bent, h);
                    let x2 = b.buf(h);
                    b.op("residual", Op::Add { a: x1, b: out, dst: x2 });
                    b.op("store frame", Op::StoreRow { src: x2, store: x_store });
                    Ok(())
                })
            };
            let main = rest(&mut b, rings[0])?;
            let tail = (0..ahead).map(|j| rest(&mut b, rings[ahead - j])).collect::<Result<Vec<_>, _>>()?;
            layers.push(Layer { project, rest: main, tail, ahead });
        }

        // Every frame's key and value of each decoder layer's cross-attention, at its position.
        let cross_slot = |layer: usize| slots + layer;
        let own_slot = |layer: usize| slots + config.decoder_layers + layer;
        let last = unit(&mut b, "model.encoder.final_norm")?;
        let cross_keys = b.routine(|b| {
            let x = b.buf(h);
            b.op("read frame", Op::LoadRow { store: x_store, dst: x });
            let normed = b.layer_norm(&last, x, h);
            let place = b.buf(h);
            b.op("read position", Op::EmbRead { table: "model.decoder.pos_emb".into(), dst: place });
            let frame = b.buf(h);
            b.op("add position", Op::Add { a: normed, b: place, dst: frame });
            for layer in 0..config.decoder_layers {
                let l = |n: &str| format!("model.decoder.layers.{layer}.encoder_attn.{n}");
                let k = b.linear(&l("k_proj"), frame, h);
                let v = b.linear(&l("v_proj"), frame, h);
                b.op("store key and value", Op::KvAppend { slot: cross_slot(layer), key: k, value: v, window: None });
            }
            Ok(())
        })?;
        let (decoder, logits, x) = decoder_routines(&mut b, &config, &own_slot, &cross_slot)?;

        let positions = weights.shape("model.decoder.pos_emb.weight").map_or(0, |s| s[0]);
        if positions < frames {
            return Err(format!("the model places {positions} frames and the chip takes {frames}"));
        }
        let Builder { buffers, ops, derived, .. } = b;
        let reach = (4 + 4 * frames).max(tokens.unwrap_or(1));
        let kv_slots = (0..slots + 2 * config.decoder_layers).map(|_| KvShape { head_dim, heads }).collect();
        let program = Program {
            family: "moonshine".into(),
            vocab: config.vocab,
            hidden: h,
            num_layers: 0,
            buffers,
            ops: Vec::new(),
            logits_from: 0,
            logits,
            kv_slots,
            delta_slots: Vec::new(),
            conv_slots: Vec::new(),
            rope_theta: config.rope_theta,
            rope_scaling: None,
            local_rope_theta: None,
            rms_epsilon: 1e-5,
            final_hidden: x,
            embedding_rows: None,
            draft: None,
            parts: vec![Part::Rows(RowPart { ops, embed: 0..0, blocks: Vec::new(), merge: 0..0, pixels: input, position: input, output: logits, head_dim, patches: reach, extra: Vec::new(), tables: vec![("model.decoder.pos_emb".into(), positions)] })],
        };
        let weights = Arc::new(weights.with_floats(derived)?);
        // Every frame has a place in the cross-attention's caches, as every token in its own.
        let layout = match tokens {
            Some(tokens) => Layout::with_cells(&program, &weights, die, tokens.max(frames))?,
            None => Layout::new(&program, &weights, die, None)?,
        };
        let mut vm = BlockVm::new(&die, layout.registers_used() as usize + 8, Some(weights.clone()));
        vm.compute_in_cells();
        Ok(Self { config, streaming, program, layout, vm, bits: die.cell.activation_bits, weights, frames, input, embed, conv1, conv2, layers, cross_keys, decoder, logits })
    }

    /// The most frames of 20 ms it takes.
    pub fn frames(&self) -> usize {
        self.frames
    }

    /// The program, its layout and its weights, to run it elsewhere.
    pub fn parts(&self) -> (&Program, &Layout, &Weights) {
        (&self.program, &self.layout, &self.weights)
    }

    pub fn logits_register(&self) -> u32 {
        self.layout.register_of(self.logits)
    }

    /// How many frames behind the conv2 output each layer's projections and the rest of it
    /// run: a layer starts where the one before finished, and finishes as many frames later
    /// as it sees ahead.
    pub fn delays(&self) -> Vec<(usize, usize)> {
        let mut at = 0;
        self.layers.iter().map(|l| {
            let delay = (at, at + l.ahead);
            at += l.ahead;
            delay
        }).collect()
    }

    /// The most tokens it writes for `frames` frames: six a second, as `Moonshine::transcribe`.
    pub fn limit(&self, frames: usize) -> usize {
        (frames * self.streaming.samples_per_frame() * 6 / 16_000 + 1).min(self.config.max_tokens - 1)
    }

    /// Every routine it runs, by number, with whether each rotates, and the microprogram.
    pub fn routines(&self) -> Result<(Vec<Routine>, netlist::Microprogram), String> {
        let mut ranges = vec![self.embed.clone(), self.conv1.0.clone(), self.conv1.1.clone(), self.conv2.0.clone(), self.conv2.1.clone()];
        for layer in &self.layers {
            ranges.push(layer.project.clone());
            ranges.push(layer.rest.clone());
            ranges.extend(layer.tail.iter().cloned());
        }
        ranges.push(self.cross_keys.clone());
        ranges.push(self.decoder.embed.clone());
        let rotating: Vec<Range<usize>> = self.decoder.own.clone();
        ranges.extend(self.decoder.own.iter().cloned());
        ranges.extend(self.decoder.cross.iter().cloned());
        ranges.push(self.decoder.logits.clone());
        let routines: Vec<Routine> = ranges.into_iter().map(|r| { let rope = rotating.contains(&r); (r, rope) }).collect();
        let part = self.program.rows().ok_or("the routines are missing")?;
        let lists: Vec<(&[compiler::Inst], bool)> = routines.iter().map(|(r, rope)| (&part.ops[r.clone()], *rope)).collect();
        let code = self.layout.routines_microprogram(&self.program, &lists)?;
        Ok((routines, code))
    }

    /// The schedule its firmware follows to transcribe speech as it comes, with the routines
    /// numbered as `routines` gives them. The host sends a word that is not zero and the 320
    /// samples of each 20 ms (four vectors, each its exponent and 80 elements), then a zero
    /// when the speech ends; the chip sends back the text's tokens. The speech must last
    /// longer than the layers run behind (`delays`), 240 ms for Moonshine-tiny's.
    pub fn schedule(&self) -> Result<Vec<Record>, String> {
        let (routines, _) = self.routines()?;
        let index = |r: &Range<usize>| routines.iter().position(|(x, _)| x == r).map(|i| i as u32).ok_or("a routine the schedule needs is missing");
        let ib = self.layout.config().index_bits;
        let register = self.layout.register_of(self.input);
        let (address, exponent) = (register << ib, exponent_register(&self.layout.config()) << ib | register);
        let [steps, frames, last, limit, word] = std::array::from_fn(|i| FIRST_FREE + i as u32);
        let neg = |n: usize| (n as u32).wrapping_neg();
        let call = |r: &Range<usize>, token: (u32, u32, u32), position: (u32, u32, u32)| -> Result<Record, String> { Ok(Record::Call { routine: index(r)?, token, position }) };
        let delays = self.delays();
        let behind = delays.last().map_or(0, |d| d.1);

        // One step at frame `s = i + from`: the samples, the front end, every layer as far
        // as it can go. A call `delay` frames behind runs only once `s` reaches it.
        let step = |from: usize| -> Result<Vec<Record>, String> {
            let s = |times: u32, plus: usize, minus: usize| (times, (times as usize * from + plus).wrapping_sub(minus) as u32, ZERO);
            let mut body = Vec::new();
            for f in 0..4 {
                body.push(Record::Vectors { count: ZERO, add: 1, address, exponent, size: self.streaming.frame as u32, routine: NO_ROUTINE });
                body.push(call(&self.embed, s(4, f + 4, 0), (0, 0, ZERO))?);
            }
            for u in 0..2 {
                body.push(call(&self.conv1.0, s(4, 2 * u, 0), (0, 0, ZERO))?);
                body.push(call(&self.conv1.1, s(2, u + 4, 0), (0, 0, ZERO))?);
            }
            body.push(call(&self.conv2.0, s(2, 0, 0), (0, 0, ZERO))?);
            body.push(call(&self.conv2.1, s(1, 0, 0), (0, 0, ZERO))?);
            for (layer, &(start, end)) in self.layers.iter().zip(&delays) {
                if from >= start {
                    body.push(call(&layer.project, s(1, 0, start), s(1, 0, start))?);
                }
                if from >= end {
                    body.push(call(&layer.rest, s(1, 0, end), s(1, 0, start))?);
                }
            }
            if from >= behind {
                body.push(call(&self.cross_keys, s(1, 0, behind), s(1, 0, behind))?);
            }
            Ok(body)
        };
        let repeat = |count: u32, add: u32, body: Vec<Record>| [vec![Record::Repeat { body: body.len() as u32, count, add }], body].concat();

        // The first steps, until every layer runs, each with what can run.
        let mut starts: Vec<usize> = delays.iter().flat_map(|d| [d.0, d.1]).chain([0, behind]).collect();
        starts.sort_unstable();
        starts.dedup();
        let mut s = Vec::new();
        for pair in starts.windows(2) {
            // Each 20 ms starts with the host's word saying it comes.
            let body = [vec![Record::Receive { var: word }], step(pair[0])?].concat();
            s.extend(repeat(ZERO, (pair[1] - pair[0]) as u32, body));
        }
        // Then a step for each 20 ms the host sends, as long as it sends them.
        let body = step(behind)?;
        s.push(Record::Stream { body: body.len() as u32, count: steps });
        s.extend(body);
        s.push(Record::Compute { dst: frames, src: steps, mul: 1, pre: behind as i32, div: 1, post: 0, max: self.frames as i32 });
        s.push(Record::Compute { dst: last, src: frames, mul: 1, pre: -1, div: 1, post: 0, max: i32::MAX });
        let per_frame = self.streaming.samples_per_frame() as i32;
        s.push(Record::Compute { dst: limit, src: frames, mul: 6 * per_frame, pre: 0, div: 16_000, post: 1, max: self.config.max_tokens as i32 - 1 });

        // The last frames, a layer at a time: the projections and rests still to run, then
        // the rests of the last frames in the narrower rings.
        let back = |minus: usize| (1, neg(minus), frames);
        for (layer, &(start, end)) in self.layers.iter().zip(&delays) {
            s.extend(repeat(ZERO, start as u32, vec![call(&layer.project, back(start), back(start))?, call(&layer.rest, back(end), back(start))?]));
            for (j, tail) in layer.tail.iter().enumerate().rev() {
                s.push(call(tail, (0, neg(j), last), (0, 0, last))?);
            }
        }
        s.extend(repeat(ZERO, behind as u32, vec![call(&self.cross_keys, back(behind), back(behind))?]));

        // The text, a token at a time.
        let d = &self.decoder;
        let mut body = vec![call(&d.embed, (0, 0, TOKEN), (0, 0, POSITION))?];
        for (own, cross) in d.own.iter().zip(&d.cross) {
            body.push(call(own, (0, 0, ZERO), (0, 0, POSITION))?);
            body.push(call(cross, (0, 0, ZERO), (0, 0, last))?);
        }
        body.push(call(&d.logits, (0, 0, ZERO), (0, 0, POSITION))?);
        s.push(Record::Generate { body: body.len() as u32, limit, start: self.config.start });
        s.extend(body);
        Ok(s)
    }

    /// What the host sends a chip that follows `schedule` for `samples`: for each 20 ms, a
    /// one and its four vectors; a zero at the end. Samples past the last whole 20 ms are left.
    pub fn schedule_input(&self, samples: &[f32]) -> Result<Vec<u32>, String> {
        let per = self.streaming.samples_per_frame();
        let frames = samples.len() / per;
        if frames > self.frames {
            return Err(format!("this chip takes up to {} frames of 20 ms", self.frames));
        }
        let mut words = Vec::new();
        for frame in samples[..frames * per].chunks(self.streaming.frame) .collect::<Vec<_>>().chunks(4) {
            words.push(1);
            for part in frame {
                let (mantissas, exponent) = encode_vector(&part.iter().map(|v| *v as f64).collect::<Vec<_>>(), self.bits);
                words.push(exponent as u32);
                words.extend(mantissas.iter().map(|m| *m as u32));
            }
        }
        words.push(0);
        Ok(words)
    }

    /// What following the schedule costs for a clip of `frames` frames, with the routines
    /// numbered as in `code`: the die's cycles and the firmware's (each routine started
    /// alone, each word exchanged; `crate::chip::firmware_cycles`), and the products of
    /// weights by values.
    pub fn timing(&self, frames: usize, code: &netlist::Microprogram) -> Result<Timing, String> {
        struct Count<'a> {
            code: &'a netlist::Microprogram,
            config: DieConfig,
            costs: Vec<u64>,
            input: Vec<u32>,
            handed: usize,
            step_words: usize,
            frames: usize,
            cycles: u64,
            macs: u64,
            step_start: u64,
            after_start: Option<u64>,
            samples: Vec<(u64, u64)>,
            timing: Timing,
        }
        impl netlist::schedule::Driver for Count<'_> {
            fn receive(&mut self) -> u32 {
                let at = self.handed;
                self.handed += 1;
                self.cycles += crate::chip::firmware_cycles(0, 0, 1);
                if at.is_multiple_of(self.step_words) && at <= self.frames * self.step_words {
                    // A new 20 ms, or the end of the speech.
                    if at > 0 {
                        self.timing.step = self.timing.step.max(self.cycles - self.step_start);
                    }
                    self.step_start = self.cycles;
                    if at == self.frames * self.step_words {
                        self.after_start = Some(self.cycles);
                        self.timing.clip_macs = self.macs;
                    }
                }
                self.input.get(at).copied().unwrap_or(0)
            }
            fn write(&mut self, _: u32, _: u32) {}
            fn read(&mut self, _: u32) -> u32 {
                0
            }
            fn run(&mut self, routine: u32, token: u32, position: u32) {
                self.cycles += self.code.cycles(&self.config, routine as usize, token, position) + crate::chip::firmware_cycles(0, 1, 0);
                self.macs += self.costs[routine as usize];
            }
            fn sample(&mut self) -> u32 {
                self.samples.push((self.cycles, self.macs));
                3
            }
            fn stops(&self, _: u32) -> bool {
                false
            }
            fn send(&mut self, _: u32) {
                self.cycles += crate::chip::firmware_cycles(0, 0, 1);
            }
        }
        let (routines, _) = self.routines()?;
        let part = self.program.rows().ok_or("the routines are missing")?;
        let costs = routines.iter().map(|(r, _)| crate::system::macs(&part.ops[r.clone()], &self.program.buffers)).collect();
        let input = self.schedule_input(&vec![0.0; frames * self.streaming.samples_per_frame()])?;
        let mut count = Count { code, config: self.layout.config(), costs, input, handed: 0, step_words: 1 + 4 * (1 + self.streaming.frame), frames, cycles: 0, macs: 0, step_start: 0, after_start: None, samples: Vec::new(), timing: Timing::default() };
        netlist::schedule::expand(&netlist::schedule::words(&self.schedule()?), &mut count)?;
        let start = count.after_start.ok_or("the speech never ended")?;
        let [(first, first_macs), (second, second_macs), ..] = count.samples[..] else {
            return Err("fewer than two tokens: a longer clip is needed".into());
        };
        count.timing.after = first - start;
        count.timing.token = second - first;
        count.timing.token_macs = second_macs - first_macs;
        count.timing.clip_macs = first_macs;
        Ok(count.timing)
    }

    /// The die as its firmware sees it, run by the exact machine.
    pub fn die_engine(&mut self, routines: &[Routine], code: netlist::Microprogram) -> RoutinesDie<'_> {
        self.vm.clear();
        crate::chip::load_initial(&mut self.vm, &self.layout.setup().initial, exponent_register(&self.layout.config()));
        let part = self.program.rows().expect("the routines");
        let lists = routines.iter().map(|(r, rope)| (part.ops[r.clone()].to_vec(), *rope)).collect();
        RoutinesDie::new(&self.layout, &self.program, &mut self.vm, lists, code)
    }

    /// The text of `samples`, greedily, the schedule followed on the exact machine; and the
    /// scores of each token as values.
    pub fn transcribe(&mut self, samples: &[f32]) -> Result<(Vec<u32>, Vec<Vec<f32>>), String> {
        let schedule = self.schedule()?;
        let input = self.schedule_input(samples)?;
        let (routines, code) = self.routines()?;
        let (vocab, end) = (self.config.vocab as u32, self.config.end);
        let register = self.logits_register();
        let ib = self.layout.config().index_bits;
        let exponents = exponent_register(&self.layout.config());
        let bits = self.bits;
        let mut die = self.die_engine(&routines, code);
        crate::chip::follow_schedule(&mut die, &schedule, &input, (register << ib, exponents << ib | register, vocab), bits, &[end])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use compiler::Inst;
    use netlist::block::BlockOps;

    /// The die's `asinh` against the floating-point one over the front end's whole range:
    /// a frame normalised to unit RMS reaches √79 ≈ 8.9 at most, times the compression's
    /// factor (0.64 for Moonshine-tiny's), within the table's ±8.
    #[test]
    fn the_die_s_asinh_follows_the_front_end_s_range() {
        let values: Vec<f64> = (0..64).map(|i| (i as f64 - 31.5) * 0.24).collect();
        let program = Program {
            family: "asinh".into(), vocab: 1, hidden: 64, num_layers: 0, buffers: vec![64, 64], ops: Vec::new(), logits_from: 0, logits: 1,
            kv_slots: Vec::new(), delta_slots: Vec::new(), conv_slots: Vec::new(), rope_theta: 1.0, rope_scaling: None, local_rope_theta: None, rms_epsilon: 1e-6,
            final_hidden: 1, embedding_rows: None, draft: None,
            parts: vec![Part::Rows(RowPart { ops: vec![Inst { op: Op::Asinh { src: 0, dst: 1 }, layer: None, name: "asinh" }], embed: 0..0, blocks: Vec::new(), merge: 0..0, pixels: 0, position: 0, output: 1, head_dim: 1, patches: 1, extra: Vec::new(), tables: Vec::new() })],
        };
        let header = b"{}";
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend(header);
        let weights = Arc::new(Weights::from_buffers(vec![file.into_boxed_slice()]).unwrap());
        for bits in [16, 20] {
            let die = DieConfig { function_select_bits: 4, ..crate::exact::exact_config(bits) };
            let layout = Layout::new(&program, &weights, die, None).unwrap();
            let mut vm = BlockVm::new(&die, layout.registers_used() as usize + 8, Some(weights.clone()));
            crate::chip::load_initial(&mut vm, &layout.setup().initial, exponent_register(&die));
            let (mantissas, exponent) = encode_vector(&values, bits);
            vm.load_mantissas(layout.register_of(0), &mantissas, exponent);
            layout.run_ops(&mut vm, &program, &program.rows().unwrap().ops, 0, 0, false).unwrap();
            let got = vm.floats(layout.register_of(1), 64);
            let worst = values.iter().zip(&got).map(|(x, y)| (x.asinh() as f32 - y).abs()).fold(0f32, f32::max);
            eprintln!("{bits} bits: worst {worst}");
            assert!(worst < 0.01, "{bits} bits: {worst}");
        }
    }
}
