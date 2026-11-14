//! Alternative signed multipliers. Partial products become columns of bits, reduced
//! by carry-save adders and finished by a parallel-prefix adder.

use crate::cell::Gen;
use crate::{Bit, Builder};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MultiplierKind {
    /// One ripple adder per weight bit: smallest to describe, slowest.
    #[default]
    Array,
    /// Baugh-Wooley partial products with a carry-save tree.
    Wallace,
    /// Radix-4 Booth recoding halves the rows, then the same tree.
    Booth,
    /// No multiplier in the cell: a shared unit precomputes the odd multiples of each
    /// activation and every weight picks one, shifts it and fixes its sign.
    Multiples,
}

/// `value` added to the columns, modulo the product's width.
fn push_constant(columns: &mut [Vec<Bit>], value: u128) {
    for (i, column) in columns.iter_mut().enumerate() {
        if value >> i & 1 == 1 {
            column.push(Bit::One);
        }
    }
}

impl Gen {
    fn half_adder(&mut self) -> String {
        if self.modules.contains_key("half_adder") {
            return "half_adder".into();
        }
        let mut b = Builder::new("half_adder", "Half adder", "Adds two bits and gives the sum bit and the carry bit.");
        let a = b.input("a", 1)[0];
        let x = b.input("b", 1)[0];
        let sum = b.output("sum", 1)[0];
        let cout = b.output("cout", 1)[0];
        b.group("Gates");
        b.xor_to(a, x, sum);
        b.and_to(a, x, cout);
        self.add(b.finish())
    }

    /// Parallel-prefix adder: its delay grows with the logarithm of the width.
    pub(crate) fn prefix_adder(&mut self, width: u32) -> String {
        let name = format!("prefix_adder_{width}");
        if self.modules.contains_key(&name) {
            return name;
        }
        let mut b = Builder::new(&name, &format!("{width}-bit fast adder"), "Computes the carries in parallel with a Kogge-Stone network.");
        let a = b.input("a", width);
        let x = b.input("b", width);
        let sum = b.output("sum", width);
        let w = width as usize;
        b.group("Generate and propagate");
        let mut generate = b.wire("g0", width);
        let mut propagate = b.wire("p0", width);
        for i in 0..w {
            b.and_to(a[i], x[i], generate[i]);
            b.xor_to(a[i], x[i], propagate[i]);
        }
        let half_sum = propagate.clone();
        let levels = 32 - width.saturating_sub(1).leading_zeros();
        for level in 0..levels as usize {
            b.group(&format!("Level {}", level + 1));
            let span = 1usize << level;
            let next_g = b.wire(&format!("g{}", level + 1), width);
            let next_p = b.wire(&format!("p{}", level + 1), width);
            for i in 0..w {
                if i >= span {
                    let carried = b.wire(&format!("t{}_{i}", level + 1), 1)[0];
                    b.and_to(propagate[i], generate[i - span], carried);
                    b.or_to(generate[i], carried, next_g[i]);
                    b.and_to(propagate[i], propagate[i - span], next_p[i]);
                } else {
                    b.or_to(generate[i], Bit::Zero, next_g[i]);
                    b.or_to(propagate[i], Bit::Zero, next_p[i]);
                }
            }
            generate = next_g;
            propagate = next_p;
        }
        b.group("Sum");
        for i in 0..w {
            let carry = if i == 0 { Bit::Zero } else { generate[i - 1] };
            b.xor_to(half_sum[i], carry, sum[i]);
        }
        self.add(b.finish())
    }

    fn baugh_wooley(&mut self, b: &mut Builder, columns: &mut [Vec<Bit>], weight: &[Bit], x: &[Bit]) {
        let (m, n) = (weight.len(), x.len());
        let total = columns.len() as u32;
        for (i, &a) in weight.iter().enumerate() {
            for (j, &bit) in x.iter().enumerate() {
                let product = b.wire(&format!("pp{i}_{j}"), 1)[0];
                b.and_to(a, bit, product);
                let is_negative = (i == m - 1) != (j == n - 1);
                if is_negative {
                    let inverted = b.wire(&format!("pp{i}_{j}_n"), 1)[0];
                    b.not_to(product, inverted);
                    columns[i + j].push(inverted);
                } else {
                    columns[i + j].push(product);
                }
            }
        }
        let constant = (1u128 << (m - 1)) + (1u128 << (n - 1)) + (1u128 << (m + n - 1));
        push_constant(columns, constant & mask(total));
    }

    fn booth_rows(&mut self, b: &mut Builder, columns: &mut [Vec<Bit>], weight: &[Bit], x: &[Bit]) {
        let total = columns.len() as u32;
        let (m, n) = (weight.len(), x.len());
        let sign = weight[m - 1];
        let digit_bit = |at: isize| -> Bit {
            if at < 0 {
                Bit::Zero
            } else {
                weight.get(at as usize).copied().unwrap_or(sign)
            }
        };
        let mut constant = 0u128;
        let mut position = 0usize;
        while position < m {
            let name = position / 2;
            let (high, mid, low) = (digit_bit(position as isize + 1), digit_bit(position as isize), digit_bit(position as isize - 1));
            b.group(&format!("Booth digit {name}"));
            let one = b.wire(&format!("one{name}"), 1)[0];
            b.xor_to(mid, low, one);
            let (not_high, not_mid, not_low) = (b.wire(&format!("nh{name}"), 1)[0], b.wire(&format!("nm{name}"), 1)[0], b.wire(&format!("nl{name}"), 1)[0]);
            b.not_to(high, not_high);
            b.not_to(mid, not_mid);
            b.not_to(low, not_low);
            let (both_low, plus_two, both_high, minus_two) = (b.wire(&format!("t0{name}"), 1)[0], b.wire(&format!("t1{name}"), 1)[0], b.wire(&format!("t2{name}"), 1)[0], b.wire(&format!("t3{name}"), 1)[0]);
            b.and_to(not_mid, not_low, both_low);
            b.and_to(high, both_low, plus_two);
            b.and_to(mid, low, both_high);
            b.and_to(not_high, both_high, minus_two);
            let two = b.wire(&format!("two{name}"), 1)[0];
            b.or_to(plus_two, minus_two, two);
            let top = x[n - 1];
            let mut last = Bit::Zero;
            for j in 0..=n {
                let current = x.get(j).copied().unwrap_or(top);
                let (take_one, take_two, chosen, out) = (b.wire(&format!("b{name}_{j}a"), 1)[0], b.wire(&format!("b{name}_{j}b"), 1)[0], b.wire(&format!("b{name}_{j}c"), 1)[0], b.wire(&format!("b{name}_{j}d"), 1)[0]);
                b.and_to(current, one, take_one);
                b.and_to(last, two, take_two);
                b.or_to(take_one, take_two, chosen);
                b.xor_to(chosen, high, out);
                last = current;
                let place = position + j;
                if j == n {
                    // The row's sign counts negatively: invert it and correct with a constant.
                    let inverted = b.wire(&format!("s{name}"), 1)[0];
                    b.not_to(out, inverted);
                    if place < columns.len() {
                        columns[place].push(inverted);
                        constant = constant.wrapping_sub(1u128 << place);
                    }
                } else if place < columns.len() {
                    columns[place].push(out);
                }
            }
            columns[position].push(high);
            position += 2;
        }
        push_constant(columns, constant & mask(total));
    }

    pub(crate) fn tree_multiplier(&mut self, kind: MultiplierKind, weight_bits: u32, x_bits: u32, stages: u32) -> String {
        let width = weight_bits + x_bits;
        let label = if kind == MultiplierKind::Booth { "booth" } else { "wallace" };
        let name = format!("{label}_multiplier_{weight_bits}x{x_bits}_s{stages}");
        if self.modules.contains_key(&name) {
            return name;
        }
        let full = self.full_adder();
        let half = self.half_adder();
        let finish = self.prefix_adder(width);

        let title = if kind == MultiplierKind::Booth { format!("Booth multiplier {weight_bits} × {x_bits}") } else { format!("Wallace multiplier {weight_bits} × {x_bits}") };
        let what = "Generates the partial products, reduces them with carry-save adders and finishes with a fast adder.";
        let mut b = Builder::new(&name, &title, what);
        let weight = b.input("weight", weight_bits);
        let x = b.input("x", x_bits);
        let product = b.output("product", width);

        let mut columns: Vec<Vec<Bit>> = vec![Vec::new(); width as usize];
        b.group("Partial products");
        if kind == MultiplierKind::Booth {
            self.booth_rows(&mut b, &mut columns, &weight, &x);
        } else {
            self.baugh_wooley(&mut b, &mut columns, &weight, &x);
        }

        let levels = reduction_levels(&columns);
        let cuts: Vec<u32> = (1..stages).map(|k| levels * k / stages).collect();
        let mut level = 0u32;
        let mut unique = 0u32;
        let register = |b: &mut Builder, columns: &mut Vec<Vec<Bit>>, unique: &mut u32| {
            for column in columns.iter_mut() {
                for bit in column.iter_mut().filter(|bit| !matches!(bit, Bit::One | Bit::Zero)) {
                    *unique += 1;
                    let q = b.wire(&format!("cut{unique}"), 1)[0];
                    b.dff_to(*bit, q);
                    *bit = q;
                }
            }
        };
        for _ in cuts.iter().filter(|cut| **cut == 0) {
            b.group("First stage");
            register(&mut b, &mut columns, &mut unique);
        }
        while columns.iter().any(|c| c.len() > 2) {
            level += 1;
            b.group(&format!("Reduction {level}"));
            let mut next: Vec<Vec<Bit>> = vec![Vec::new(); width as usize];
            for (i, column) in columns.iter().enumerate() {
                let (triples, rest) = column.as_chunks::<3>();
                for (index, chunk) in triples.iter().enumerate() {
                    let sum = b.wire(&format!("l{level}c{i}s{index}"), 1)[0];
                    let carry = b.wire(&format!("l{level}c{i}c{index}"), 1)[0];
                    b.inst(&format!("fa_l{level}c{i}_{index}"), &full, &[("a", vec![chunk[0]]), ("b", vec![chunk[1]]), ("cin", vec![chunk[2]]), ("sum", vec![sum]), ("cout", vec![carry])]);
                    next[i].push(sum);
                    if i + 1 < width as usize {
                        next[i + 1].push(carry);
                    }
                }
                if rest.len() == 2 && column.len() > 2 {
                    let sum = b.wire(&format!("l{level}c{i}hs"), 1)[0];
                    let carry = b.wire(&format!("l{level}c{i}hc"), 1)[0];
                    b.inst(&format!("ha_l{level}c{i}"), &half, &[("a", vec![rest[0]]), ("b", vec![rest[1]]), ("sum", vec![sum]), ("cout", vec![carry])]);
                    next[i].push(sum);
                    if i + 1 < width as usize {
                        next[i + 1].push(carry);
                    }
                } else {
                    next[i].extend_from_slice(rest);
                }
            }
            columns = next;
            for _ in cuts.iter().filter(|cut| **cut == level) {
                b.group(&format!("Stage after reduction {level}"));
                register(&mut b, &mut columns, &mut unique);
            }
        }
        for _ in cuts.iter().filter(|cut| **cut > level) {
            b.group("Final stage");
            register(&mut b, &mut columns, &mut unique);
        }

        b.group("Final sum");
        let first: Vec<Bit> = columns.iter().map(|c| c.first().copied().unwrap_or(Bit::Zero)).collect();
        let second: Vec<Bit> = columns.iter().map(|c| c.get(1).copied().unwrap_or(Bit::Zero)).collect();
        let sum = b.wire("final_sum", width);
        b.inst("final_adder", &finish, &[("a", first), ("b", second), ("sum", sum.clone())]);
        for (bit, out) in sum.iter().zip(&product) {
            b.xor_to(*bit, Bit::Zero, *out);
        }
        self.add(b.finish())
    }
}

fn mask(bits: u32) -> u128 {
    if bits >= 128 {
        u128::MAX
    } else {
        (1u128 << bits) - 1
    }
}

/// How many carry-save levels these columns need.
fn reduction_levels(columns: &[Vec<Bit>]) -> u32 {
    let mut heights: Vec<usize> = columns.iter().map(Vec::len).collect();
    let mut levels = 0;
    while heights.iter().any(|h| *h > 2) {
        levels += 1;
        let mut next = vec![0usize; heights.len()];
        for (i, height) in heights.iter().enumerate() {
            let triples = height / 3;
            let rest = height % 3;
            let halves = (rest == 2 && *height > 2) as usize;
            next[i] += triples + halves + if halves == 1 { 0 } else { rest };
            if i + 1 < next.len() {
                next[i + 1] += triples + halves;
            }
        }
        heights = next;
    }
    levels
}
