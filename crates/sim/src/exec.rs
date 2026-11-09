//! The microprogram interpreter. One call to `step` is one token through the chip.

use crate::weights::Weights;
use compiler::{Op, Program};
use rayon::prelude::*;
use std::collections::HashMap;
use std::time::Instant;

const EPS: f32 = 1e-6;

/// How intermediate values are represented while the program runs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Numeric {
    Float,
    /// Round activations and state to signed fixed point after every operation.
    Fixed { activation_bits: u32, activation_frac: u32, state_bits: u32, state_frac: u32 },
    /// Each vector keeps its own exponent; only the mantissas are rounded.
    Block { activation_bits: u32, state_bits: u32, table_bits: Option<u32> },
}

/// Elementary functions: exact, or read from tables of `2^bits` entries like the hardware.
#[derive(Clone, Copy, Debug)]
struct Functions {
    table_bits: Option<u32>,
}

const TABLE_RANGE: f32 = 8.0;

impl Functions {
    /// The centre of the table cell that `x` falls in, over `[-RANGE, RANGE)`.
    fn snap(self, x: f32) -> f32 {
        match self.table_bits {
            None => x,
            Some(bits) => {
                let cells = (1u32 << bits) as f32;
                let step = 2.0 * TABLE_RANGE / cells;
                let clamped = x.clamp(-TABLE_RANGE, TABLE_RANGE - step);
                ((clamped + TABLE_RANGE) / step).floor() * step - TABLE_RANGE + step / 2.0
            }
        }
    }

    fn sigmoid(self, x: f32) -> f32 {
        let x = self.snap(x);
        1.0 / (1.0 + (-x).exp())
    }

    fn silu(self, x: f32) -> f32 {
        x * self.sigmoid(x)
    }

    fn exp(self, x: f32) -> f32 {
        self.snap(x).exp()
    }

    fn softplus(self, x: f32) -> f32 {
        (1.0 + self.snap(x).exp()).ln()
    }

    /// Applies `f` to the mantissa in `[low, 1)`, as a table over the mantissa does.
    fn mantissa_table(self, value: f32, low: f32, f: impl Fn(f32) -> f32, undo: impl Fn(i32) -> f32) -> f32 {
        let Some(bits) = self.table_bits else {
            return f(value);
        };
        if value <= 0.0 {
            return 0.0;
        }
        let mut exponent = value.log2().floor() as i32 + 1;
        if low < 0.5 && exponent % 2 != 0 {
            exponent += 1;
        }
        let mantissa = value / 2f32.powi(exponent);
        // Signed tables give half of their cells to positive mantissas.
        let step = 2f32.powi(1 - bits as i32);
        f((mantissa / step).floor() * step + step / 2.0) * undo(exponent)
    }

    fn inv_sqrt(self, value: f32) -> f32 {
        self.mantissa_table(value, 0.25, |v| 1.0 / v.sqrt(), |e| 2f32.powi(-e / 2))
    }

    fn reciprocal(self, value: f32) -> f32 {
        self.mantissa_table(value, 0.5, |v| 1.0 / v, |e| 2f32.powi(-e))
    }
}

#[derive(Clone, Debug, Default)]
pub struct StepStats {
    pub seconds: f64,
    pub saturated: u64,
}

struct Kv {
    keys: Vec<f32>,
    values: Vec<f32>,
    width: usize,
}

pub struct Engine {
    program: Program,
    weights: Weights,
    floats: HashMap<String, Vec<f32>>,
    buffers: Vec<Vec<f32>>,
    kv: Vec<Kv>,
    delta: Vec<Vec<f32>>,
    conv: Vec<Vec<f32>>,
    position: u64,
    numeric: Numeric,
    inverse_freq: Vec<f32>,
    saturated: u64,
    functions: Functions,
    rope_position: u64,
    draft_tokens: usize,
}

fn round_fixed(values: &mut [f32], bits: u32, frac: u32, saturated: &mut u64) {
    let scale = (1u64 << frac) as f32;
    let top = (1u64 << (bits - 1)) as f32;
    for v in values.iter_mut() {
        let scaled = (*v * scale).round();
        if scaled >= top || scaled < -top {
            *saturated += 1;
        }
        *v = scaled.clamp(-top, top - 1.0) / scale;
    }
}

fn round_block(values: &mut [f32], bits: u32) {
    let largest = values.iter().fold(0f32, |m, v| m.max(v.abs()));
    if largest == 0.0 || !largest.is_finite() {
        return;
    }
    let exponent = largest.log2().floor() as i32 + 1;
    let scale = 2f32.powi(bits as i32 - 1 - exponent);
    let top = 2f32.powi(bits as i32 - 1) - 1.0;
    for v in values.iter_mut() {
        *v = (*v * scale).round().clamp(-top, top) / scale;
    }
}

impl Engine {
    pub fn new(program: Program, weights: Weights, numeric: Numeric) -> Result<Self, String> {
        let mut floats = HashMap::new();
        let draft_ops = program.draft.iter().flat_map(|d| d.ops.iter());
        for inst in program.ops.iter().chain(draft_ops) {
            let names: Vec<String> = match &inst.op {
                Op::RmsNorm { weight, .. } | Op::RmsGated { weight, .. } => vec![format!("{weight}.weight")],
                Op::Dconv { weight, .. } => vec![format!("{weight}.weight")],
                Op::DeltaStep { a_log, dt_bias, .. } => vec![a_log.clone(), dt_bias.clone()],
                _ => vec![],
            };
            for name in names {
                let mut values = weights.floats(&name)?;
                // MLX converters store these centred on 1; the program expects 0.
                if matches!(inst.op, Op::RmsNorm { .. }) && values.iter().sum::<f32>() / values.len() as f32 > 0.5 {
                    values.iter_mut().for_each(|v| *v -= 1.0);
                }
                floats.insert(name.clone(), values);
            }
        }
        let buffers = program.buffers.iter().map(|len| vec![0f32; *len]).collect();
        let kv = program.kv_slots.iter().map(|s| Kv { keys: Vec::new(), values: Vec::new(), width: s.heads * s.head_dim }).collect();
        let delta = program.delta_slots.iter().map(|len| vec![0f32; *len]).collect();
        let conv = program.conv_slots.iter().map(|(dim, kernel)| vec![0f32; dim * (kernel - 1)]).collect();
        let rotary = program
            .ops
            .iter()
            .find_map(|i| if let Op::Rope { rotary, .. } = i.op { Some(rotary) } else { None })
            .unwrap_or(0);
        let inverse_freq = (0..rotary / 2).map(|i| 1.0 / program.rope_theta.powf((2 * i) as f32 / rotary as f32)).collect();
        let table_bits = if let Numeric::Block { table_bits, .. } = numeric { table_bits } else { None };
        Ok(Self { program, weights, floats, buffers, kv, delta, conv, position: 0, numeric, inverse_freq, saturated: 0, functions: Functions { table_bits }, rope_position: 0, draft_tokens: 0 })
    }

    pub fn vocabulary(&self) -> usize {
        self.program.vocab
    }

    pub fn reset(&mut self) {
        self.position = 0;
        self.draft_tokens = 0;
        for k in &mut self.kv {
            k.keys.clear();
            k.values.clear();
        }
        for d in &mut self.delta {
            d.fill(0.0);
        }
        for c in &mut self.conv {
            c.fill(0.0);
        }
    }

    pub fn position(&self) -> u64 {
        self.position
    }

    pub fn buffer(&self, index: usize) -> &[f32] {
        &self.buffers[index]
    }

    pub fn program(&self) -> &Program {
        &self.program
    }

    pub fn delta_state(&self, slot: usize) -> &[f32] {
        &self.delta[slot]
    }

    fn quantize(&mut self, buf: usize) {
        match self.numeric {
            Numeric::Float => {}
            Numeric::Fixed { activation_bits, activation_frac, .. } => {
                round_fixed(&mut self.buffers[buf], activation_bits, activation_frac, &mut self.saturated);
            }
            Numeric::Block { activation_bits, .. } => round_block(&mut self.buffers[buf], activation_bits),
        }
    }

    /// Runs one token. Logits are only produced when `want_logits` is set.
    pub fn step(&mut self, token: u32, want_logits: bool) -> Result<(Option<&[f32]>, StepStats), String> {
        let started = Instant::now();
        self.saturated = 0;
        self.rope_position = self.position;
        let limit = if want_logits { self.program.ops.len() } else { self.program.logits_from };
        for index in 0..limit {
            let op = self.program.ops[index].op.clone();
            self.run_op(&op, token)?;
        }
        self.position += 1;
        let stats = StepStats { seconds: started.elapsed().as_secs_f64(), saturated: self.saturated };
        let logits = want_logits.then(|| self.buffers[self.program.logits].as_slice());
        Ok((logits, stats))
    }

    pub fn has_draft_head(&self) -> bool {
        self.program.draft.is_some()
    }

    pub fn draft_tokens(&self) -> usize {
        self.draft_tokens
    }

    /// Forgets speculative draft tokens, back to `tokens` entries in the draft cache.
    pub fn truncate_draft(&mut self, tokens: usize) {
        let Some(draft) = &self.program.draft else { return };
        let k = &mut self.kv[draft.kv_slot];
        k.keys.truncate(tokens * k.width);
        k.values.truncate(tokens * k.width);
        self.draft_tokens = tokens;
    }

    /// Copies the main model's final hidden state into the draft head's input.
    pub fn feed_draft_from_main(&mut self) {
        let Some(draft) = &self.program.draft else { return };
        let (src, dst) = (self.program.final_hidden, draft.hidden_in);
        let (x, y) = pair(&mut self.buffers, src, dst);
        y.copy_from_slice(x);
    }

    /// Feeds the draft head its own previous output, for deeper speculation.
    pub fn feed_draft_from_draft(&mut self) {
        let Some(draft) = &self.program.draft else { return };
        let (src, dst) = (draft.hidden_out, draft.hidden_in);
        let (x, y) = pair(&mut self.buffers, src, dst);
        y.copy_from_slice(x);
    }

    /// One step of the draft head for `next_token`. Logits only when asked.
    pub fn draft(&mut self, next_token: u32, want_logits: bool) -> Result<Option<&[f32]>, String> {
        let draft = self.program.draft.as_ref().ok_or("este modelo no tiene cabeza MTP")?.clone();
        self.rope_position = self.draft_tokens as u64;
        let limit = if want_logits { draft.ops.len() } else { draft.logits_from };
        for inst in &draft.ops[..limit] {
            self.run_op(&inst.op, next_token)?;
        }
        self.draft_tokens += 1;
        Ok(want_logits.then(|| self.buffers[draft.logits].as_slice()))
    }

    #[allow(clippy::too_many_arguments)]
    fn run_moe(&mut self, prefix: &str, src: usize, dst: usize, experts: usize, top_k: usize, intermediate: usize, shared: usize) -> Result<(), String> {
        let x = self.buffers[src].clone();
        let hidden = x.len();
        let f = self.functions;
        let mut gates = vec![0f32; experts];
        self.weights.matrix(&format!("{prefix}.gate"))?.apply(&x, &mut gates);
        let top = gates.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut probabilities: Vec<(usize, f32)> = gates.iter().enumerate().map(|(i, g)| (i, (g - top).exp())).collect();
        probabilities.select_nth_unstable_by(top_k - 1, |a, b| b.1.total_cmp(&a.1));
        probabilities.truncate(top_k);
        let total: f32 = probabilities.iter().map(|(_, p)| p).sum();

        let gate = self.weights.matrix(&format!("{prefix}.switch_mlp.gate_proj"))?;
        let up = self.weights.matrix(&format!("{prefix}.switch_mlp.up_proj"))?;
        let down = self.weights.matrix(&format!("{prefix}.switch_mlp.down_proj"))?;
        let mut result = vec![0f32; hidden];
        let (mut gated, mut lifted) = (vec![0f32; intermediate], vec![0f32; intermediate]);
        let mut out = vec![0f32; hidden];
        for (expert, probability) in probabilities {
            gate.apply_rows(expert * intermediate, &x, &mut gated);
            up.apply_rows(expert * intermediate, &x, &mut lifted);
            let product: Vec<f32> = gated.iter().zip(&lifted).map(|(g, u)| f.silu(*g) * u).collect();
            down.apply_rows(expert * hidden, &product, &mut out);
            let score = probability / total;
            result.iter_mut().zip(&out).for_each(|(r, o)| *r += score * o);
        }

        let (mut g, mut u) = (vec![0f32; shared], vec![0f32; shared]);
        self.weights.matrix(&format!("{prefix}.shared_expert.gate_proj"))?.apply(&x, &mut g);
        self.weights.matrix(&format!("{prefix}.shared_expert.up_proj"))?.apply(&x, &mut u);
        let product: Vec<f32> = g.iter().zip(&u).map(|(a, b)| f.silu(*a) * b).collect();
        self.weights.matrix(&format!("{prefix}.shared_expert.down_proj"))?.apply(&product, &mut out);
        let mut gate_value = [0f32; 1];
        self.weights.matrix(&format!("{prefix}.shared_expert_gate"))?.apply(&x, &mut gate_value);
        let shared_gate = f.sigmoid(gate_value[0]);
        result.iter_mut().zip(&out).for_each(|(r, o)| *r += shared_gate * o);

        self.buffers[dst].copy_from_slice(&result);
        self.quantize(dst);
        Ok(())
    }

    fn run_op(&mut self, op: &Op, token: u32) -> Result<(), String> {
        match op {
            Op::EmbRead { table, dst } => {
                let matrix = self.weights.matrix(table)?;
                let len = self.buffers[*dst].len();
                let row = matrix.row(token as usize, len)?;
                self.buffers[*dst].copy_from_slice(&row);
                self.quantize(*dst);
            }
            Op::Matvec { matrix, src, dst } => {
                let m = self.weights.matrix(matrix)?;
                let (x, y) = pair(&mut self.buffers, *src, *dst);
                m.apply(x, y);
                self.quantize(*dst);
            }
            Op::Copy { src, dst } => {
                let (x, y) = pair(&mut self.buffers, *src, *dst);
                y.copy_from_slice(x);
            }
            Op::Add { a, b, dst } => {
                let sum: Vec<f32> = self.buffers[*a].iter().zip(&self.buffers[*b]).map(|(p, q)| p + q).collect();
                self.buffers[*dst].copy_from_slice(&sum);
                self.quantize(*dst);
            }
            Op::MulElem { a, b, dst } => {
                let product: Vec<f32> = self.buffers[*a].iter().zip(&self.buffers[*b]).map(|(p, q)| p * q).collect();
                self.buffers[*dst].copy_from_slice(&product);
                self.quantize(*dst);
            }
            Op::Silu { src, dst } => {
                let (x, y) = pair(&mut self.buffers, *src, *dst);
                let f = self.functions;
                y.iter_mut().zip(x).for_each(|(o, v)| *o = f.silu(*v));
                self.quantize(*dst);
            }
            Op::Sigmoid { src, dst } => {
                let (x, y) = pair(&mut self.buffers, *src, *dst);
                let f = self.functions;
                y.iter_mut().zip(x).for_each(|(o, v)| *o = f.sigmoid(*v));
                self.quantize(*dst);
            }
            Op::RmsNorm { src, weight, offset, block, dst } => {
                let w = self.floats[&format!("{weight}.weight")].clone();
                let f = self.functions;
                let (x, y) = pair(&mut self.buffers, *src, *dst);
                for (xs, ys) in x.chunks_exact(*block).zip(y.chunks_exact_mut(*block)) {
                    let mean = xs.iter().map(|v| v * v).sum::<f32>() / *block as f32;
                    let inv = f.inv_sqrt(mean + EPS);
                    for i in 0..*block {
                        ys[i] = xs[i] * inv * (offset + w[i]);
                    }
                }
                self.quantize(*dst);
            }
            Op::SplitQueryGate { src, query, gate, block } => {
                let joined = self.buffers[*src].clone();
                for (h, pair) in joined.chunks_exact(block * 2).enumerate() {
                    self.buffers[*query][h * block..(h + 1) * block].copy_from_slice(&pair[..*block]);
                    self.buffers[*gate][h * block..(h + 1) * block].copy_from_slice(&pair[*block..]);
                }
            }
            Op::Rope { buf, block, rotary } => {
                let position = self.rope_position as f32;
                let (cos, sin): (Vec<f32>, Vec<f32>) = self.inverse_freq.iter().map(|f| ((position * f).cos(), (position * f).sin())).unzip();
                let half = rotary / 2;
                for head in self.buffers[*buf].chunks_exact_mut(*block) {
                    let original: Vec<f32> = head[..*rotary].to_vec();
                    for i in 0..half {
                        // rotate_half(x) = (-x2, x1), and cos/sin repeat over both halves.
                        head[i] = original[i] * cos[i] - original[half + i] * sin[i];
                        head[half + i] = original[half + i] * cos[i] + original[i] * sin[i];
                    }
                }
                self.quantize(*buf);
            }
            Op::KvAppend { slot, key, value } => {
                let (k, v) = (self.buffers[*key].clone(), self.buffers[*value].clone());
                self.kv[*slot].keys.extend_from_slice(&k);
                self.kv[*slot].values.extend_from_slice(&v);
            }
            Op::AttnSum { slot, query, head_dim, heads, kv_heads, dst } => {
                let kv = &self.kv[*slot];
                let group = heads / kv_heads;
                let scale = 1.0 / (*head_dim as f32).sqrt();
                let tokens = kv.keys.len() / kv.width;
                let q = &self.buffers[*query];
                let f = self.functions;
                let results: Vec<Vec<f32>> = (0..*heads)
                    .into_par_iter()
                    .map(|h| {
                        let kv_head = h / group;
                        let qh = &q[h * head_dim..(h + 1) * head_dim];
                        let mut scores = vec![0f32; tokens];
                        for (t, s) in scores.iter_mut().enumerate() {
                            let key = &kv.keys[t * kv.width + kv_head * head_dim..t * kv.width + (kv_head + 1) * head_dim];
                            *s = qh.iter().zip(key).map(|(a, b)| a * b).sum::<f32>() * scale;
                        }
                        let top = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                        let mut sum = 0f32;
                        for s in scores.iter_mut() {
                            *s = f.exp(*s - top);
                            sum += *s;
                        }
                        let inverse = f.reciprocal(sum);
                        let mut out = vec![0f32; *head_dim];
                        for (t, s) in scores.iter().enumerate() {
                            let p = s * inverse;
                            let value = &kv.values[t * kv.width + kv_head * head_dim..t * kv.width + (kv_head + 1) * head_dim];
                            out.iter_mut().zip(value).for_each(|(o, v)| *o += p * v);
                        }
                        out
                    })
                    .collect();
                for (h, r) in results.iter().enumerate() {
                    self.buffers[*dst][h * head_dim..(h + 1) * head_dim].copy_from_slice(r);
                }
                self.quantize(*dst);
            }
            Op::Dconv { slot, weight, src, dst } => {
                let w = &self.floats[&format!("{weight}.weight")];
                let (dim, kernel) = self.program.conv_slots[*slot];
                let input = self.buffers[*src].clone();
                let state = &mut self.conv[*slot];
                let output = &mut self.buffers[*dst];
                let history = kernel - 1;
                for c in 0..dim {
                    let taps = &w[c * kernel..(c + 1) * kernel];
                    let mut y = taps[history] * input[c];
                    for t in 0..history {
                        y += taps[t] * state[t * dim + c];
                    }
                    output[c] = y;
                }
                state.copy_within(dim.., 0);
                let keep = dim * (history - 1);
                state[keep..keep + dim].copy_from_slice(&input);
                self.quantize(*dst);
            }
            Op::DeltaStep { slot, mixed, a, b, a_log, dt_bias, key_dim, key_heads, value_dim, value_heads, dst } => {
                self.delta_step(*slot, *mixed, *a, *b, a_log, dt_bias, *key_dim, *key_heads, *value_dim, *value_heads, *dst);
            }
            Op::Concat { a, b, dst } => {
                let joined: Vec<f32> = self.buffers[*a].iter().chain(&self.buffers[*b]).copied().collect();
                self.buffers[*dst].copy_from_slice(&joined);
            }
            Op::Moe { prefix, src, dst, experts, top_k, intermediate, shared_intermediate } => {
                self.run_moe(prefix, *src, *dst, *experts, *top_k, *intermediate, *shared_intermediate)?;
            }
            Op::RmsGated { src, z, weight, block, dst } => {
                let w = self.floats[&format!("{weight}.weight")].clone();
                let gate = self.buffers[*z].clone();
                let f = self.functions;
                let (x, y) = pair(&mut self.buffers, *src, *dst);
                for (h, (xs, ys)) in x.chunks_exact(*block).zip(y.chunks_exact_mut(*block)).enumerate() {
                    let mean = xs.iter().map(|v| v * v).sum::<f32>() / *block as f32;
                    let inv = f.inv_sqrt(mean + EPS);
                    for i in 0..*block {
                        ys[i] = w[i] * xs[i] * inv * f.silu(gate[h * block + i]);
                    }
                }
                self.quantize(*dst);
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn delta_step(&mut self, slot: usize, mixed: usize, a: usize, b: usize, a_log: &str, dt_bias: &str, key_dim: usize, key_heads: usize, value_dim: usize, value_heads: usize, dst: usize) {
        let m = &self.buffers[mixed];
        let key_len = key_dim * key_heads;
        let (query, key, value) = (&m[..key_len], &m[key_len..2 * key_len], &m[2 * key_len..]);
        let ratio = value_heads / key_heads;
        let log_a = &self.floats[a_log];
        let bias = &self.floats[dt_bias];
        let f = self.functions;
        let beta: Vec<f32> = self.buffers[b].iter().map(|v| f.sigmoid(*v)).collect();
        let decay: Vec<f32> = self.buffers[a]
            .iter()
            .enumerate()
            .map(|(h, v)| f.exp(-log_a[h].exp() * f.softplus(v + bias[h])))
            .collect();

        let normalize = |x: &[f32]| -> Vec<f32> {
            let inv = f.inv_sqrt(x.iter().map(|v| v * v).sum::<f32>() + EPS);
            x.iter().map(|v| v * inv).collect()
        };
        let q_heads: Vec<Vec<f32>> = query.chunks_exact(key_dim).map(|c| {
            let scale = (key_dim as f32).sqrt();
            normalize(c).into_iter().map(|v| v / scale).collect()
        }).collect();
        let k_heads: Vec<Vec<f32>> = key.chunks_exact(key_dim).map(normalize).collect();

        let state = &mut self.delta[slot];
        let cell = key_dim * value_dim;
        let outputs: Vec<Vec<f32>> = state
            .par_chunks_exact_mut(cell)
            .enumerate()
            .map(|(h, s)| {
                let q = &q_heads[h / ratio];
                let k = &k_heads[h / ratio];
                let v = &value[h * value_dim..(h + 1) * value_dim];
                s.iter_mut().for_each(|x| *x *= decay[h]);
                let mut memory = vec![0f32; value_dim];
                for (i, row) in s.chunks_exact(value_dim).enumerate() {
                    memory.iter_mut().zip(row).for_each(|(m, x)| *m += x * k[i]);
                }
                let delta: Vec<f32> = v.iter().zip(&memory).map(|(a, b)| (a - b) * beta[h]).collect();
                for (i, row) in s.chunks_exact_mut(value_dim).enumerate() {
                    row.iter_mut().zip(&delta).for_each(|(x, d)| *x += k[i] * d);
                }
                let mut out = vec![0f32; value_dim];
                for (i, row) in s.chunks_exact(value_dim).enumerate() {
                    out.iter_mut().zip(row).for_each(|(o, x)| *o += x * q[i]);
                }
                out
            })
            .collect();
        for (h, r) in outputs.iter().enumerate() {
            self.buffers[dst][h * value_dim..(h + 1) * value_dim].copy_from_slice(r);
        }
        match self.numeric {
            Numeric::Float => {}
            Numeric::Fixed { state_bits, state_frac, .. } => {
                round_fixed(&mut self.delta[slot], state_bits, state_frac, &mut self.saturated);
            }
            Numeric::Block { state_bits, .. } => {
                let cell = key_dim * value_dim;
                self.delta[slot].chunks_exact_mut(cell).for_each(|head| round_block(head, state_bits));
            }
        }
        self.quantize(dst);
    }
}

fn pair(buffers: &mut [Vec<f32>], src: usize, dst: usize) -> (&[f32], &mut [f32]) {
    if src < dst {
        let (left, right) = buffers.split_at_mut(dst);
        (&left[src], &mut right[0])
    } else {
        let (left, right) = buffers.split_at_mut(src);
        (&right[0], &mut left[dst])
    }
}

