//! Chooses the next token in hardware. Logits stream through a sorted list of the best
//! candidates; one pass adds their probabilities and a second finds the chosen one.

use crate::cell::Gen;
use crate::rv32::{constant, Logic};
use crate::{Bit, Builder, Design};
use serde::Serialize;
use std::collections::BTreeMap;

pub const TABLE_BITS: u32 = 8;
/// Fractional bits of the inverse temperature.
pub const INVERSE_FRACTION: u32 = 8;
const SUM_BITS: usize = 24;

#[derive(Clone, Copy, Debug, Serialize)]
pub struct SamplerConfig {
    pub logit_bits: u32,
    pub index_bits: u32,
    pub entries: u32,
}

/// `exp(-i / 16)` as 16-bit values.
pub fn exponential_table() -> Vec<u32> {
    (0..1u32 << TABLE_BITS).map(|i| ((-(i as f64) / 16.0).exp() * 65535.0).round() as u32).collect()
}

fn selector_bits(config: &SamplerConfig) -> usize {
    (config.entries as usize + 1).next_power_of_two().trailing_zeros() as usize
}

fn sampler(gen: &mut Gen, c: SamplerConfig) -> String {
    let (bits, index_bits, entries) = (c.logit_bits as usize, c.index_bits as usize, c.entries as usize);
    let count_bits = selector_bits(&c);
    let mut b = Builder::new("token_sampler", "Unidad de muestreo", "Conserva los candidatos con mayor logit y elige uno según su probabilidad, con temperatura y top-k.");
    let start = b.input("start", 1)[0];
    let greedy = b.input("greedy", 1)[0];
    let seed_write = b.input("seed_write", 1)[0];
    let seed = b.input("seed", 32);
    let inverse = b.input("inverse", 16);
    let limit = b.input("limit", count_bits as u32);
    let logit_valid = b.input("logit_valid", 1)[0];
    let logit_data = b.input("logit_data", c.logit_bits);
    let logit_last = b.input("logit_last", 1)[0];
    let table_write = b.input("table_write", 1)[0];
    let table_address = b.input("table_address", TABLE_BITS);
    let table_data = b.input("table_data", 16);
    let done_out = b.output("done", 1)[0];
    let token_out = b.output("token", c.index_bits);

    let table_read = b.wire("table_read", TABLE_BITS);
    let table_value = b.wire("table_value", 16);
    let (read_address, write_enable);
    {
        let mut l = Logic { b: &mut b, next: 0 };
        l.b.group("Control");
        let streaming = l.fresh();
        let first_pass = l.fresh();
        let scale = l.fresh();
        let second_pass = l.fresh();
        let done = l.fresh();
        let add_phase = l.fresh();
        l.b.or_to(done, Bit::Zero, done_out);

        l.b.group("Candidatos");
        let values: Vec<Vec<Bit>> = (0..entries).map(|_| l.fresh_bus(bits)).collect();
        let indices: Vec<Vec<Bit>> = (0..entries).map(|_| l.fresh_bus(index_bits)).collect();
        let used: Vec<Bit> = (0..entries).map(|_| l.fresh()).collect();
        let position = l.fresh_bus(index_bits);

        let accept = l.and(streaming, logit_valid);
        let stays: Vec<Bit> = (0..entries)
            .map(|j| {
                // value >= new, as signed numbers, keeps the earlier of two equal logits ahead.
                let widen = |v: &[Bit]| -> Vec<Bit> { v.iter().copied().chain([v[bits - 1]]).collect() };
                let (left, right) = (widen(&values[j]), widen(&logit_data));
                let inverted: Vec<Bit> = right.iter().map(|x| l.not(*x)).collect();
                let (difference, _) = l.adder(&left, &inverted, Bit::One);
                let not_less = l.not(difference[bits]);
                l.and(used[j], not_less)
            })
            .collect();
        let not_start = l.not(start);
        for j in 0..entries {
            let previous_stays = if j == 0 { Bit::Zero } else { stays[j - 1] };
            let take_new = {
                let moves = l.not(stays[j]);
                let open = if j == 0 { Bit::One } else { previous_stays };
                l.and(moves, open)
            };
            let incoming_value = if j == 0 { logit_data.clone() } else { l.mux_bus(take_new, &values[j - 1], &logit_data) };
            let incoming_index = if j == 0 { position.clone() } else { l.mux_bus(take_new, &indices[j - 1], &position) };
            let incoming_used = if j == 0 { Bit::One } else { l.or(take_new, used[j - 1]) };
            let write = {
                let moves = l.not(stays[j]);
                l.and(accept, moves)
            };
            let next_value = l.mux_bus(write, &values[j], &incoming_value);
            let next_index = l.mux_bus(write, &indices[j], &incoming_index);
            let next_used = {
                let t = l.mux(write, used[j], incoming_used);
                l.and(t, not_start)
            };
            l.connect_flops(&next_value, &values[j]);
            l.connect_flops(&next_index, &indices[j]);
            l.connect_flops(&[next_used], &[used[j]]);
        }
        let mut carry = accept;
        let next_position: Vec<Bit> = position
            .iter()
            .map(|bit| {
                let out = l.xor(*bit, carry);
                carry = l.and(*bit, carry);
                l.and(out, not_start)
            })
            .collect();
        l.connect_flops(&next_position, &position);

        l.b.group("Paso por los candidatos");
        let cursor = l.fresh_bus(count_bits);
        let leaf_count = 1usize << count_bits;
        let value_leaves: Vec<Vec<Bit>> = (0..leaf_count).map(|i| values[i.min(entries - 1)].clone()).collect();
        let index_leaves: Vec<Vec<Bit>> = (0..leaf_count).map(|i| indices[i.min(entries - 1)].clone()).collect();
        let used_leaves: Vec<Vec<Bit>> = (0..leaf_count).map(|i| if i < entries { vec![used[i]] } else { vec![Bit::Zero] }).collect();
        let current_value = l.select(&cursor, &value_leaves);
        let current_index = l.select(&cursor, &index_leaves);

        let mut carry = Bit::One;
        let cursor_next: Vec<Bit> = cursor
            .iter()
            .map(|bit| {
                let out = l.xor(*bit, carry);
                carry = l.and(*bit, carry);
                out
            })
            .collect();
        let next_used = l.select(&cursor_next, &used_leaves)[0];
        let at_limit = {
            let same: Vec<Bit> = cursor_next.iter().zip(&limit).map(|(a, b)| { let d = l.xor(*a, *b); l.not(d) }).collect();
            l.all(&same)
        };
        let finished_pass = {
            let none = l.not(next_used);
            l.or(at_limit, none)
        };

        l.b.group("Multiplicador");
        let delta = {
            let widen = |v: &[Bit]| -> Vec<Bit> { v.iter().copied().chain([v[bits - 1]]).collect() };
            let (left, right) = (widen(&values[0]), widen(&current_value));
            let inverted: Vec<Bit> = right.iter().map(|x| l.not(*x)).collect();
            l.adder(&left, &inverted, Bit::One).0
        };
        let factor_width = (bits + 1).max(SUM_BITS);
        let sum = l.fresh_bus(SUM_BITS);
        let rng = l.fresh_bus(32);
        let wide_delta: Vec<Bit> = delta.iter().copied().chain(std::iter::repeat_n(Bit::Zero, factor_width - delta.len())).collect();
        let wide_sum: Vec<Bit> = sum.iter().copied().chain(std::iter::repeat_n(Bit::Zero, factor_width - SUM_BITS)).collect();
        let left_factor = l.mux_bus(scale, &wide_delta, &wide_sum);
        let right_factor = l.mux_bus(scale, &inverse, &rng[16..32]);
        let width = factor_width + 16;
        let mut product = constant(0, width);
        for (shift, select) in right_factor.iter().enumerate() {
            let partial: Vec<Bit> = std::iter::repeat_n(Bit::Zero, shift).chain(left_factor.iter().map(|v| l.and(*v, *select))).chain(std::iter::repeat_n(Bit::Zero, 16 - shift)).collect();
            product = l.adder(&product, &partial, Bit::Zero).0;
        }
        let shifted_index = &product[INVERSE_FRACTION as usize..];
        let overflow = l.any(&shifted_index[TABLE_BITS as usize..]);
        let saturated: Vec<Bit> = shifted_index[..TABLE_BITS as usize].iter().map(|v| l.or(*v, overflow)).collect();
        read_address = saturated;

        l.b.group("Suma de probabilidades");
        let in_pass = l.or(first_pass, second_pass);
        let adding = l.and(in_pass, add_phase);
        let extended: Vec<Bit> = table_value.iter().copied().chain(std::iter::repeat_n(Bit::Zero, SUM_BITS - 16)).collect();
        let added = l.adder(&sum, &extended, Bit::Zero).0;
        let next_phase = {
            let t = l.not(add_phase);
            l.and(t, in_pass)
        };
        let stored = l.mux_bus(adding, &sum, &added);
        let zero = constant(0, SUM_BITS);
        let begin_first = {
            let ended = l.and(accept, logit_last);
            let not_greedy = l.not(greedy);
            l.and(ended, not_greedy)
        };
        let clear_sum = l.or(begin_first, scale);
        let next_sum = l.mux_bus(clear_sum, &stored, &zero);
        l.connect_flops(&next_sum, &sum);
        l.connect_flops(&[next_phase], &[add_phase]);

        l.b.group("Número aleatorio");
        let threshold = l.fresh_bus(SUM_BITS);
        let next_threshold = l.mux_bus(scale, &threshold, &product[16..16 + SUM_BITS]);
        l.connect_flops(&next_threshold, &threshold);
        let xor_shift = |l: &mut Logic, value: &[Bit], amount: usize, left: bool| -> Vec<Bit> {
            (0..32usize)
                .map(|i| {
                    let other = if left { i.checked_sub(amount).map(|j| value[j]) } else { value.get(i + amount).copied() };
                    match other {
                        Some(bit) => l.xor(value[i], bit),
                        None => value[i],
                    }
                })
                .collect()
        };
        let a = xor_shift(&mut l, &rng, 13, true);
        let d = xor_shift(&mut l, &a, 17, false);
        let f = xor_shift(&mut l, &d, 5, true);
        let advanced = l.mux_bus(scale, &rng, &f);
        let next_rng = l.mux_bus(seed_write, &advanced, &seed);
        l.connect_flops(&next_rng, &rng);

        l.b.group("Control");
        let over_threshold = {
            // threshold < sum after adding, that is, threshold - sum borrows.
            let inverted: Vec<Bit> = added.iter().map(|v| l.not(*v)).collect();
            let (_, no_borrow) = l.adder(&threshold, &inverted, Bit::One);
            l.not(no_borrow)
        };
        let first_ends = {
            let t = l.and(first_pass, add_phase);
            l.and(t, finished_pass)
        };
        let second_add = l.and(second_pass, add_phase);
        let second_ends = {
            let ending = l.or(over_threshold, finished_pass);
            l.and(second_add, ending)
        };
        let ended_greedy = {
            let t = l.and(accept, logit_last);
            l.and(t, greedy)
        };
        let advance_cursor = {
            let more_first = {
                let t = l.and(first_pass, add_phase);
                let more = l.not(finished_pass);
                l.and(t, more)
            };
            let more_second = {
                let more = l.not(over_threshold);
                let also = l.not(finished_pass);
                let t = l.and(second_add, more);
                l.and(t, also)
            };
            l.or(more_first, more_second)
        };
        let reset_cursor = l.any(&[start, first_ends, ended_greedy]);
        let zero_cursor = constant(0, count_bits);
        let moved = l.mux_bus(advance_cursor, &cursor, &cursor_next);
        let next_cursor = l.mux_bus(reset_cursor, &moved, &zero_cursor);
        l.connect_flops(&next_cursor, &cursor);

        let stream_ending = l.and(accept, logit_last);
        let keep_streaming = {
            let t = l.not(stream_ending);
            let u = l.and(streaming, t);
            l.and(u, not_start)
        };
        let next_streaming = l.or(start, keep_streaming);
        let next_first = {
            let stay = {
                let more = l.not(first_ends);
                l.and(first_pass, more)
            };
            let t = l.or(begin_first, stay);
            l.and(t, not_start)
        };
        let next_scale = l.and(first_ends, not_start);
        let next_second = {
            let stay = {
                let more = l.not(second_ends);
                l.and(second_pass, more)
            };
            let t = l.or(scale, stay);
            l.and(t, not_start)
        };
        let next_done = {
            let t = l.any(&[done, second_ends, ended_greedy]);
            l.and(t, not_start)
        };
        l.connect_flops(&[next_streaming, next_first, next_scale, next_second, next_done], &[streaming, first_pass, scale, second_pass, done]);

        l.b.group("Resultado");
        // A greedy choice is simply the head of the list, once the last logit is in it.
        let token_reg = l.fresh_bus(index_bits);
        let next_token = l.mux_bus(second_ends, &token_reg, &current_index);
        l.connect_flops(&next_token, &token_reg);
        let chosen = l.mux_bus(greedy, &token_reg, &indices[0]);
        for (from, to) in chosen.iter().zip(&token_out) {
            l.b.or_to(*from, Bit::Zero, *to);
        }
        write_enable = table_write;
    }
    let mut address = Vec::with_capacity(TABLE_BITS as usize);
    {
        let mut l = Logic { b: &mut b, next: 100_000 };
        l.b.group("Tabla exponencial");
        let chosen = l.mux_bus(table_write, &read_address, &table_address);
        address.extend(chosen);
        for (from, to) in read_address.iter().zip(&table_read) {
            l.b.or_to(*from, Bit::Zero, *to);
        }
    }
    b.sram("exponential", TABLE_BITS, 16, &[("addr", address), ("we", vec![write_enable]), ("wdata", table_data), ("rdata", table_value)]);
    gen.add(b.finish())
}

pub fn sampler_design(config: SamplerConfig) -> Design {
    let mut gen = Gen { modules: BTreeMap::new() };
    let top = sampler(&mut gen, config);
    Design { top, modules: gen.modules }
}

/// The unit at instruction level: same choice and same number of cycles.
#[derive(Clone, Debug)]
pub struct SamplerModel {
    pub config: SamplerConfig,
    pub rng: u32,
    pub table: Vec<u32>,
}

impl SamplerModel {
    /// The chosen token and the cycles after the last logit until `done` is set.
    pub fn sample(&mut self, logits: &[i64], inverse: u32, limit: usize, greedy: bool) -> (u32, u64) {
        let capacity = self.config.entries as usize;
        let mut best: Vec<(i64, u32)> = Vec::with_capacity(capacity + 1);
        for (index, value) in logits.iter().enumerate() {
            let at = best.iter().position(|(v, _)| v < value).unwrap_or(best.len());
            if at < capacity {
                best.insert(at, (*value, index as u32));
                best.truncate(capacity);
            }
        }
        if greedy {
            return (best[0].1, 1);
        }
        let usable = limit.min(best.len());
        let top = best[0].0;
        let weight = |value: i64| -> u32 {
            let address = ((top - value) as u64 * inverse as u64) >> INVERSE_FRACTION;
            self.table[address.min((1 << TABLE_BITS) - 1) as usize]
        };
        let total: u64 = best[..usable].iter().map(|(v, _)| weight(*v) as u64).sum();
        let threshold = (((total as u32 & 0xFF_FFFF) as u64 * (self.rng >> 16) as u64) >> 16) as u32;
        let mut sum = 0u32;
        let mut chosen = usable - 1;
        let mut steps = usable;
        for (i, (v, _)) in best[..usable].iter().enumerate() {
            sum += weight(*v);
            if threshold < sum {
                chosen = i;
                steps = i + 1;
                break;
            }
        }
        let mut x = self.rng;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.rng = x;
        (best[chosen].1, 1 + 2 * usable as u64 + 1 + 2 * steps as u64)
    }
}
