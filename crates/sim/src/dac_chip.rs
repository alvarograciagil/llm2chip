//! DAC's decoder in the die's arithmetic: codes to sound with the chip's operations, run
//! in the order the host gives them, as Moonshine and MAGVIT-v2's decoder are.
//!
//! Sound is kept in stores of rows, one per time step with all its channels, each with a
//! border of zero rows on both sides as wide as the furthest any reader reaches:
//!
//! - **A convolution** with padding and dilation reads its taps' rows side by side
//!   (`LoadRows` at offsets `k·d`, from the first row it reaches) and multiplies them by its
//!   kernel laid out the same way; a second routine stores the result at its own row.
//! - **A transposed convolution** of rate `s` and kernel `2s` gives each output row from
//!   two input rows, `q − 1` and `q`, with `q = (n + p) div s`, through a kernel that
//!   depends only on the phase `(n + p) mod s`: a routine per phase, each an ordinary
//!   convolution of two rows.
//! - **Snake**, `x + sin²(αx) / α`, is a product by α (`MulWeight`), a sine (`Sin`), its
//!   square, a product by 1 / α and a sum. The sine's argument reaches about a hundred
//!   radians: its table holds one period over its whole range, and the alignment before it
//!   wraps around instead of saturating, which leaves the argument's place in the period.
//!   That is the die's one addition for speech, built only where a microprogram uses it.
//! - **The codes** come in as tokens: each picks its codebook's row (`EmbRead`), projected
//!   to the latent's width; the codebooks add up.

use crate::dac::DacConfig;
use crate::lower::Layout;
use crate::vm::BlockVm;
use crate::weights::{Tensor, Weights};
use compiler::{Buf, Inst, KvShape, Op, Part, Program, RowPart};
use netlist::block::exponent_register;
use netlist::DieConfig;
use std::ops::Range;
use crate::chip::RoutinesDie;
use netlist::schedule::{Record, FIRST_FREE, ZERO};
use std::path::Path;
use std::sync::Arc;

/// One routine as the host runs it, its token, and whether the output is read after it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Call {
    pub routine: usize,
    pub token: u32,
    pub read: bool,
}

/// A store of `rows` rows of `width` with `border` zero rows on each side.
#[derive(Clone, Copy, Debug)]
struct Store {
    buf: Buf,
    border: usize,
}

struct Builder<'a> {
    weights: &'a Weights,
    buffers: Vec<usize>,
    ops: Vec<Inst>,
    routines: Vec<Range<usize>>,
    derived: Vec<Tensor>,
}

impl Builder<'_> {
    fn buf(&mut self, len: usize) -> Buf {
        self.buffers.push(len);
        self.buffers.len() - 1
    }

    fn op(&mut self, name: &'static str, op: Op) {
        self.ops.push(Inst { op, layer: None, name });
    }

    /// The operations `body` adds, as a routine; gives its number.
    fn routine(&mut self, body: impl FnOnce(&mut Self) -> Result<(), String>) -> Result<usize, String> {
        let start = self.ops.len();
        body(self)?;
        self.routines.push(start..self.ops.len());
        Ok(self.routines.len() - 1)
    }

    fn floats(&self, name: &str) -> Result<Vec<f32>, String> {
        self.weights.floats(name)
    }

    fn shape(&self, name: &str) -> Result<Vec<usize>, String> {
        self.weights.shape(name).map(<[usize]>::to_vec).ok_or(format!("{name} is missing"))
    }

    fn derive(&mut self, name: String, shape: Vec<usize>, values: Vec<f32>) -> String {
        self.derived.push((format!("{name}.weight"), shape, values));
        name
    }

    fn store(&mut self, rows: usize, width: usize, border: usize) -> Store {
        Store { buf: self.buf((rows + 2 * border) * width), border }
    }

    /// `matrix · src + bias` into a new buffer.
    fn product(&mut self, matrix: &str, src: Buf, outputs: usize, bias: &str) -> Buf {
        let (out, biased) = (self.buf(outputs), self.buf(outputs));
        self.op("product", Op::Matvec { matrix: matrix.into(), src, dst: out });
        self.op("bias", Op::AddWeight { src: out, weight: bias.into(), dst: biased });
        biased
    }

    /// Snake with the α of `name`, on `x` of `width`, into a new buffer.
    fn snake(&mut self, name: &str, x: Buf, width: usize) -> Result<Buf, String> {
        let alpha = self.floats(&format!("{name}.alpha"))?;
        let inverse = alpha.iter().map(|a| 1.0 / (a + 1e-9)).collect();
        self.derived.push((format!("{name}.alpha"), vec![width], alpha));
        self.derived.push((format!("{name}.inverse"), vec![width], inverse));
        let [a, s, s2, t, y] = [0; 5].map(|_| self.buf(width));
        self.op("α·x", Op::MulWeight { src: x, weight: format!("{name}.alpha"), dst: a });
        self.op("sine", Op::Sin { src: a, dst: s });
        self.op("square", Op::MulElem { a: s, b: s, dst: s2 });
        self.op("over α", Op::MulWeight { src: s2, weight: format!("{name}.inverse"), dst: t });
        self.op("Snake", Op::Add { a: x, b: t, dst: y });
        Ok(y)
    }

    /// A convolution's kernel `[out, in, kernel]` as one matrix over its taps side by side.
    fn kernel(&mut self, name: &str) -> Result<(String, usize, usize, usize), String> {
        let shape = self.shape(&format!("{name}.weight"))?;
        let (outputs, inputs, kernel) = (shape[0], shape[1], shape[2]);
        let weight = self.floats(&format!("{name}.weight"))?;
        let mut laid = vec![0f32; outputs * kernel * inputs];
        for o in 0..outputs {
            for c in 0..inputs {
                for k in 0..kernel {
                    laid[(o * kernel + k) * inputs + c] = weight[(o * inputs + c) * kernel + k];
                }
            }
        }
        Ok((self.derive(format!("{name}.rows"), vec![outputs, kernel * inputs], laid), outputs, inputs, kernel))
    }
}

/// What the host runs over the rows of one map, in order: for each row, routines with the
/// token each takes from the row.
#[derive(Clone, Debug)]
struct Pass {
    rows: usize,
    /// `(routine, first, step)`: the token is `first + step · row`.
    steps: Vec<(usize, usize, usize)>,
}

/// DAC's decoder as the chip's program, for a fixed number of frames, and what the host runs.
pub struct DacChip {
    pub config: DacConfig,
    program: Program,
    layout: Layout,
    vm: BlockVm,
    exponents: u32,
    weights: Arc<Weights>,
    frames: usize,
    samples: usize,
    codes_in: Vec<usize>,
    keep_latent: usize,
    passes: Vec<Pass>,
    /// For each output row, the phase routine of each transposed convolution, by block.
    output: (usize, Buf),
    lists: Vec<Vec<Inst>>,
}

/// The widest any convolution of a residual unit reaches: three taps of dilation 9.
const BORDER: usize = 27;

impl DacChip {
    /// The decoder in `dir` for `frames` frames, on a die like `die` with vectors of its
    /// activation bits (it gets the ninth function table); with the weights in its cells,
    /// as a chip holds them, when `cells` is true.
    pub fn new(dir: &Path, frames: usize, die: DieConfig, cells: bool) -> Result<Self, String> {
        Self::from_parts(DacConfig::read(dir)?, Weights::open(dir)?, frames, die, cells)
    }

    /// The exact machine's die with vectors of `bits` bits and cells as wide.
    pub fn exact(dir: &Path, frames: usize, bits: u32) -> Result<Self, String> {
        Self::exact_summing(dir, frames, bits, 2 * bits + 8)
    }

    /// The same with sums of `accumulator_bits` bits in the cells.
    pub fn exact_summing(dir: &Path, frames: usize, bits: u32, accumulator_bits: u32) -> Result<Self, String> {
        let base = crate::exact::exact_config(bits);
        let die = DieConfig { cell: netlist::CellConfig { weight_bits: bits, accumulator_bits, ..base.cell }, ..base };
        Self::new(dir, frames, die, false)
    }

    pub fn from_parts(config: DacConfig, weights: Weights, frames: usize, die: DieConfig, cells: bool) -> Result<Self, String> {
        let die = DieConfig { function_select_bits: die.function_select_bits.max(4), ..die };
        let mut b = Builder { weights: &weights, buffers: Vec::new(), ops: Vec::new(), routines: Vec::new(), derived: Vec::new() };
        let mut passes = Vec::new();

        // The codes: each codebook's row through its projection, added up, stored.
        let latent = config.latent_dim;
        let z = b.store(frames, latent, 3);
        let mut codes_in = Vec::new();
        let mut sum: Option<Buf> = None;
        for book in 0..config.codebooks {
            let p = |n: &str| format!("quantizer.quantizers.{book}.{n}");
            let projection = b.floats(&p("out_proj.weight"))?;
            let matrix = b.derive(p("out_proj.flat"), vec![latent, config.codebook_dim], projection);
            codes_in.push(b.routine(|b| {
                let row = b.buf(config.codebook_dim);
                b.op("code's row", Op::EmbRead { table: p("codebook"), dst: row });
                let projected = b.product(&matrix, row, latent, &p("out_proj.bias"));
                sum = Some(match sum {
                    None => projected,
                    Some(before) => {
                        let added = b.buf(latent);
                        b.op("add codebooks", Op::Add { a: before, b: projected, dst: added });
                        added
                    }
                });
                Ok(())
            })?);
        }
        let sum = sum.ok_or("no codebooks")?;
        let keep_latent = b.routine(|b| {
            b.op("store", Op::StoreRow { src: sum, store: z.buf });
            Ok(())
        })?;

        // A convolution of `src`, `rows` output rows, with padding `pad` and dilation `d`:
        // the routine that reads its taps from the first row it reaches, and what happens to
        // its result after, which leaves it in the buffer it gives.
        type After<'f> = &'f mut dyn FnMut(&mut Builder, Buf, usize) -> Result<Buf, String>;
        let convolution = |b: &mut Builder, name: &str, src: Store, pad: usize, d: usize, after: After| -> Result<((usize, usize, usize), Buf), String> {
            let (matrix, outputs, inputs, kernel) = b.kernel(name)?;
            let near = b.buf(kernel * inputs);
            let mut result = 0;
            let read = b.routine(|b| {
                b.op("read taps", Op::LoadRows { store: src.buf, dst: near, offsets: (0..kernel).map(|k| k * d).collect() });
                let out = b.product(&matrix, near, outputs, &format!("{name}.bias"));
                result = after(b, out, outputs)?;
                Ok(())
            })?;
            Ok(((read, src.border - pad, 1), result))
        };
        let store = |b: &mut Builder, src: Buf, dst: Store| -> Result<(usize, usize, usize), String> {
            let routine = b.routine(|b| {
                b.op("store", Op::StoreRow { src, store: dst.buf });
                Ok(())
            })?;
            Ok((routine, dst.border, 1))
        };
        // Snake over every row of `src` into `dst`, laid out alike: the same border.
        let snake_pass = |b: &mut Builder, name: &str, src: Store, dst: Store, rows: usize, width: usize| -> Result<Pass, String> {
            let routine = b.routine(|b| {
                let x = b.buf(width);
                b.op("read", Op::LoadRow { store: src.buf, dst: x });
                let y = b.snake(name, x, width)?;
                b.op("store", Op::StoreRow { src: y, store: dst.buf });
                Ok(())
            })?;
            Ok(Pass { rows, steps: vec![(routine, src.border, 1)] })
        };

        let mut width = config.decoder_dim;
        let mut length = frames;
        let mut x = b.store(length, width, BORDER);
        let (read, result) = convolution(&mut b, "decoder.model.0", z, 3, 1, &mut |_, out, _| Ok(out))?;
        passes.push(Pass { rows: length, steps: vec![read, store(&mut b, result, x)?] });
        for (block, &s) in config.rates.iter().enumerate() {
            let p = |n: &str| format!("decoder.model.{}.block.{n}", block + 1);
            let snaked = b.store(length, width, BORDER);
            passes.push(snake_pass(&mut b, &p("0"), x, snaked, length, width)?);
            // The transposed convolution, a routine per phase.
            let name = p("1");
            let shape = b.shape(&format!("{name}.weight"))?;
            let (inputs, outputs) = (shape[0], shape[1]);
            let weight = b.floats(&format!("{name}.weight"))?;
            let pad = s.div_ceil(2);
            let next = s * length + s - 2 * pad;
            let out = b.store(next, outputs, BORDER);
            let near = b.buf(2 * inputs);
            let result = b.buf(outputs);
            let mut phases = Vec::new();
            for phase in 0..s {
                let mut laid = vec![0f32; outputs * 2 * inputs];
                for o in 0..outputs {
                    for (j, k) in [phase + s, phase].into_iter().enumerate() {
                        for c in 0..inputs {
                            laid[o * 2 * inputs + j * inputs + c] = weight[(c * outputs + o) * 2 * s + k];
                        }
                    }
                }
                let matrix = b.derive(format!("{name}.phase{phase}"), vec![outputs, 2 * inputs], laid);
                phases.push(b.routine(|b| {
                    b.op("read two rows", Op::LoadRows { store: snaked.buf, dst: near, offsets: vec![0, 1] });
                    let y = b.product(&matrix, near, outputs, &format!("{name}.bias"));
                    b.op("keep", Op::Copy { src: y, dst: result });
                    Ok(())
                })?);
            }
            let store = b.routine(|b| {
                b.op("store", Op::StoreRow { src: result, store: out.buf });
                Ok(())
            })?;
            // Output row n = s·q + phase − pad reads rows q − 1 and q of the input with the
            // phase's kernel. Each q gives a row for each phase that falls in the output, in
            // order; a run of q's that give all `s` is one pass, the others a pass each.
            let phases_of = |q: usize| (0..s).filter(|p| (s * q + p).checked_sub(pad).is_some_and(|n| n < next)).collect::<Vec<_>>();
            let step = |q: usize, p: usize, rows: usize| [(phases[p], q + snaked.border - 1, rows.min(1)), (store, s * q + p - pad + out.border, s * rows.min(1))];
            let last = (next - 1 + pad) / s;
            let mut q = 0;
            while q <= last {
                let here = phases_of(q);
                let mut end = q;
                while here.len() == s && end < last && phases_of(end + 1).len() == s {
                    end += 1;
                }
                let rows = end - q + 1;
                let repeated = usize::from(here.len() == s && rows > 1);
                passes.push(Pass { rows, steps: here.iter().flat_map(|p| step(q, *p, repeated)).collect() });
                q = end + 1;
            }
            (x, width, length) = (out, outputs, next);
            // The residual units: Snake, the dilated convolution, Snake, the 1 × 1 one, plus x.
            for (unit, d) in [1, 3, 9].into_iter().enumerate() {
                let q = |n: &str| p(&format!("{}.block.{n}", unit + 2));
                let snaked = b.store(length, width, BORDER);
                passes.push(snake_pass(&mut b, &q("0"), x, snaked, length, width)?);
                let (pointwise, _, _, _) = b.kernel(&q("3"))?;
                let second = q("2");
                let bias = format!("{}.bias", q("3"));
                let (read, refined) = convolution(&mut b, &q("1"), snaked, 3 * d, d, &mut |b, out, width| {
                    let bent = b.snake(&second, out, width)?;
                    Ok(b.product(&pointwise, bent, width, &bias))
                })?;
                // The unit's input added, at its own row.
                let add = b.routine(|b| {
                    let input = b.buf(width);
                    b.op("read input", Op::LoadRow { store: x.buf, dst: input });
                    let total = b.buf(width);
                    b.op("residual", Op::Add { a: input, b: refined, dst: total });
                    b.op("store", Op::StoreRow { src: total, store: x.buf });
                    Ok(())
                })?;
                passes.push(Pass { rows: length, steps: vec![read, (add, x.border, 1)] });
            }
        }
        let last = config.rates.len() + 1;
        let snaked = b.store(length, width, BORDER);
        passes.push(snake_pass(&mut b, &format!("decoder.model.{last}"), x, snaked, length, width)?);
        let (matrix, _, inputs, kernel) = b.kernel(&format!("decoder.model.{}", last + 1))?;
        let near = b.buf(kernel * inputs);
        let mut sound = 0;
        let output_routine = b.routine(|b| {
            b.op("read taps", Op::LoadRows { store: snaked.buf, dst: near, offsets: (0..kernel).collect() });
            let out = b.product(&matrix, near, 1, &format!("decoder.model.{}.bias", last + 1));
            sound = b.buf(1);
            b.op("tanh", Op::SoftCap { src: out, cap: 1.0, dst: sound });
            Ok(())
        })?;

        let Builder { buffers, ops, routines, derived, .. } = b;
        let reach = length + 2 * BORDER;
        let program = Program {
            family: "dac".into(),
            vocab: config.codebook_size,
            hidden: config.latent_dim,
            num_layers: 0,
            buffers,
            ops: Vec::new(),
            logits_from: 0,
            logits: sound,
            kv_slots: Vec::<KvShape>::new(),
            delta_slots: Vec::new(),
            conv_slots: Vec::new(),
            rope_theta: 10_000.0,
            rope_scaling: None,
            local_rope_theta: None,
            rms_epsilon: 1e-6,
            final_hidden: sound,
            embedding_rows: Some(config.codebook_size),
            draft: None,
            parts: vec![Part::Rows(RowPart { ops, embed: 0..0, blocks: Vec::new(), merge: 0..0, pixels: sound, position: sound, output: sound, head_dim: 1, patches: reach, extra: Vec::new() })],
        };
        let lists = routines.iter().map(|r| program.rows().expect("the routines").ops[r.clone()].to_vec()).collect();
        let weights = Arc::new(weights.with_floats(derived)?);
        let layout = if cells { Layout::with_cells(&program, &weights, die, 1)? } else { Layout::new(&program, &weights, die, None)? };
        let mut vm = BlockVm::new(&die, layout.registers_used() as usize + 8, Some(weights.clone()));
        vm.compute_in_cells();
        Ok(Self { exponents: exponent_register(&die), program, layout, vm, weights, frames, samples: length, codes_in, keep_latent, passes, output: (output_routine, sound), lists, config })
    }

    pub fn frames(&self) -> usize {
        self.frames
    }

    pub fn samples(&self) -> usize {
        self.samples
    }

    /// The program, its layout and its weights, to run it elsewhere, such as on gates.
    pub fn parts(&self) -> (&Program, &Layout, &Weights) {
        (&self.program, &self.layout, &self.weights)
    }

    /// Every routine's operations, by number.
    pub fn routines(&self) -> &[Vec<Inst>] {
        &self.lists
    }

    pub fn output_register(&self) -> u32 {
        self.layout.register_of(self.output.1)
    }

    /// Everything the host runs for `codes[book][frame]`, the output read after each sample.
    pub fn calls(&self, codes: &[Vec<u32>]) -> Result<Vec<Call>, String> {
        if codes.len() != self.codes_in.len() || codes.iter().any(|c| c.len() != self.frames) {
            return Err(format!("this chip takes {} codebooks of {} frames", self.codes_in.len(), self.frames));
        }
        let mut calls = Vec::new();
        for frame in 0..self.frames {
            for (list, routine) in codes.iter().zip(&self.codes_in) {
                calls.push(Call { routine: *routine, token: list[frame], read: false });
            }
            calls.push(Call { routine: self.keep_latent, token: (frame + 3) as u32, read: false });
        }
        for pass in &self.passes {
            for row in 0..pass.rows {
                for (routine, first, step) in &pass.steps {
                    calls.push(Call { routine: *routine, token: (first + step * row) as u32, read: false });
                }
            }
        }
        // The last convolution reads seven rows from three before each, with padding 3.
        calls.extend((0..self.samples).map(|t| Call { routine: self.output.0, token: (t + BORDER - 3) as u32, read: true }));
        Ok(calls)
    }

    fn reset(&mut self) {
        self.vm.clear();
        crate::chip::load_initial(&mut self.vm, &self.layout.setup().initial, self.exponents);
    }

    /// The sound of the codes, as values and as the die's mantissas.
    pub fn decode(&mut self, codes: &[Vec<u32>]) -> Result<(Vec<f32>, Vec<i64>), String> {
        let calls = self.calls(codes)?;
        self.reset();
        let register = self.output_register();
        let (mut values, mut mantissas) = (Vec::new(), Vec::new());
        for call in &calls {
            self.layout.run_ops(&mut self.vm, &self.program, &self.lists[call.routine], call.token, 0, false)?;
            if call.read {
                values.push(self.vm.floats(register, 1)[0]);
                mantissas.push(self.vm.mantissa(register, 0));
            }
        }
        Ok((values, mantissas))
    }

    /// The microprogram with an entry for each routine, by number.
    pub fn microprogram(&self) -> Result<netlist::Microprogram, String> {
        let lists: Vec<(&[Inst], bool)> = self.lists.iter().map(|l| (l.as_slice(), false)).collect();
        self.layout.routines_microprogram(&self.program, &lists)
    }

    /// The schedule its firmware follows to decode a sound by itself, a window of this
    /// chip's frames at a time, with the microprogram it numbers the routines by: the host
    /// sends how many windows, and for each its codes, frame by frame and codebook by
    /// codebook, then the first sample it keeps and how many (see `schedule_input`); the
    /// chip sends back each sample kept, its mantissa and its exponent.
    pub fn schedule(&self) -> Result<(netlist::Microprogram, Vec<Record>), String> {
        let code = self.microprogram()?;
        let [windows, code_var, from, count] = std::array::from_fn(|i| FIRST_FREE + i as u32);
        let routine = |r: usize| r as u32;
        let call = |r: usize, token: (u32, u32, u32)| Record::Call { routine: routine(r), token, position: (0, 0, ZERO) };
        let repeat = |count: u32, add: u32, body: Vec<Record>| [vec![Record::Repeat { body: body.len() as u32, count, add }], body].concat();
        // As `calls`, for one window.
        let mut frame = Vec::new();
        for r in &self.codes_in {
            frame.push(Record::Receive { var: code_var });
            frame.push(call(*r, (0, 0, code_var)));
        }
        frame.push(call(self.keep_latent, (1, 3, ZERO)));
        let mut window = repeat(ZERO, self.frames as u32, frame);
        for pass in &self.passes {
            window.extend(repeat(ZERO, pass.rows as u32, pass.steps.iter().map(|(r, first, step)| call(*r, (*step as u32, *first as u32, ZERO))).collect()));
        }
        window.push(Record::Receive { var: from });
        window.push(Record::Receive { var: count });
        let (ib, register) = (self.layout.config().index_bits, self.output_register());
        let send = Record::Send { address: register << ib, exponent: self.exponents << ib | register };
        window.extend(repeat(count, 0, vec![call(self.output.0, (1, (BORDER - 3) as u32, from)), send]));
        Ok((code, [vec![Record::Receive { var: windows }], repeat(windows, 0, window)].concat()))
    }

    /// The control CPU's cycles beyond the die's to follow `schedule` for a window keeping
    /// `kept` samples (`crate::chip::firmware_cycles`): each frame's routines start alone
    /// with their codes received, the passes' on the lean path, and each sample kept alone,
    /// sent back with its exponent.
    pub fn firmware_cycles(&self, kept: usize) -> u64 {
        let frames = self.frames as u64;
        let books = self.codes_in.len() as u64;
        let passes: u64 = self.passes.iter().map(|p| (p.rows * p.steps.len()) as u64).sum();
        crate::chip::firmware_cycles(passes, frames * (books + 1) + kept as u64, frames * books + 2 + 2 * kept as u64)
    }

    /// The die as its firmware sees it, run by the exact machine, routines numbered as in
    /// the microprogram `code`.
    pub fn die_engine(&mut self, code: netlist::Microprogram) -> RoutinesDie<'_> {
        self.reset();
        let lists = self.lists.iter().map(|l| (l.clone(), false)).collect();
        RoutinesDie::new(&self.layout, &self.program, &mut self.vm, lists, code)
    }

    /// A sample the chip sends, from its mantissa and exponent.
    pub fn value(&self, mantissa: i64, exponent: i64) -> f32 {
        mantissa as f32 * 2f32.powi(exponent as i32 - self.layout.config().cell.activation_bits as i32 + 1)
    }

    /// The die's cycles for `calls`, counted from the microprogram.
    pub fn cycles(&self, calls: &[Call]) -> Result<u64, String> {
        let code = self.microprogram()?;
        let config = self.layout.config();
        Ok(calls.iter().map(|call| code.cycles(&config, call.routine, call.token, 0)).sum())
    }
}

/// What the host sends a chip of `width` frames that follows `DacChip::schedule` for
/// `codes[book][frame]` in windows keeping `chunk` frames each.
pub fn schedule_input(config: &DacConfig, codes: &[Vec<u32>], chunk: usize) -> Vec<u32> {
    let frames = codes.first().map_or(0, Vec::len);
    let (hop, width) = (config.hop(), config.window_frames(frames, chunk));
    let windows = config.windows(frames, chunk);
    let mut words = vec![windows.len() as u32];
    for (start, owned) in windows {
        for frame in start..start + width {
            words.extend(codes.iter().map(|book| book[frame]));
        }
        let from = (owned.start - start) * hop;
        let to = if owned.end == frames { config.samples(width) } else { (owned.end - start) * hop };
        words.extend([from as u32, (to - from) as u32]);
    }
    words
}

/// Decodes `codes[book][frame]` a window at a time (`DacConfig::windows`, `chunk` frames
/// kept from each) on the exact machine's die with `bits`-bit vectors, built for one window:
/// what a chip that holds a window and not the whole sound does. The sound and the die's
/// mantissas.
pub fn decode_chunked(dir: &Path, codes: &[Vec<u32>], chunk: usize, bits: u32) -> Result<(Vec<f32>, Vec<i64>), String> {
    let config = DacConfig::read(dir)?;
    let frames = codes.first().map_or(0, Vec::len);
    let width = config.window_frames(frames, chunk);
    let mut chip = DacChip::exact(dir, width, bits)?;
    let mut mantissas = Vec::new();
    let sound = crate::dac::stitch(&config, frames, chunk, &mut |start, width| {
        let (values, raw) = chip.decode(&crate::dac::window(codes, start, width))?;
        mantissas.push((start, raw));
        Ok(values)
    })?;
    // The same samples as the values kept, from each window's mantissas.
    let hop = config.hop();
    let mut kept = Vec::with_capacity(sound.len());
    for ((start, raw), (_, owned)) in mantissas.iter().zip(config.windows(frames, chunk)) {
        let from = (owned.start - start) * hop;
        let to = if owned.end == frames { raw.len() } else { (owned.end - start) * hop };
        kept.extend_from_slice(&raw[from..to]);
    }
    Ok((sound, kept))
}

#[cfg(test)]
mod tests {
    use super::*;
    use netlist::block::{encode_vector, BlockOps};

    /// The die's sine against the floating-point one, from small arguments to ones of many
    /// periods, where the alignment before the table must wrap around.
    #[test]
    fn the_die_s_sine_follows_any_argument_around_its_period() {
        let values: Vec<f64> = (0..64).map(|i| (i as f64 - 31.5) * 3.77).collect();
        let program = Program {
            family: "sine".into(), vocab: 1, hidden: 64, num_layers: 0, buffers: vec![64, 64], ops: Vec::new(), logits_from: 0, logits: 1,
            kv_slots: Vec::new(), delta_slots: Vec::new(), conv_slots: Vec::new(), rope_theta: 1.0, rope_scaling: None, local_rope_theta: None, rms_epsilon: 1e-6,
            final_hidden: 1, embedding_rows: None, draft: None,
            parts: vec![Part::Rows(RowPart { ops: vec![Inst { op: Op::Sin { src: 0, dst: 1 }, layer: None, name: "sine" }], embed: 0..0, blocks: Vec::new(), merge: 0..0, pixels: 0, position: 0, output: 1, head_dim: 1, patches: 1, extra: Vec::new() })],
        };
        let header = b"{}";
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend(header);
        let weights = Arc::new(Weights::from_buffers(vec![file.into_boxed_slice()]).unwrap());
        for bits in [16, 20] {
            let base = crate::exact::exact_config(bits);
            let die = DieConfig { function_select_bits: 4, ..base };
            let layout = Layout::new(&program, &weights, die, None).unwrap();
            let mut vm = BlockVm::new(&die, layout.registers_used() as usize + 8, Some(weights.clone()));
            crate::chip::load_initial(&mut vm, &layout.setup().initial, exponent_register(&die));
            let (mantissas, exponent) = encode_vector(&values, bits);
            vm.load_mantissas(layout.register_of(0), &mantissas, exponent);
            layout.run_ops(&mut vm, &program, &program.rows().unwrap().ops, 0, 0, false).unwrap();
            let got = vm.floats(layout.register_of(1), 64);
            let worst = values.iter().zip(&got).map(|(x, y)| (x.sin() as f32 - y).abs()).fold(0f32, f32::max);
            eprintln!("{bits} bits: worst {worst}");
            assert!(worst < if bits == 16 { 0.01 } else { 0.002 }, "{bits} bits: {worst}");
        }
    }
}
