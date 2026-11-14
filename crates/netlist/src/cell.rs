//! Generates the compute cell: weight memory, multiplier and accumulator.
//! Weights and activations are signed two's complement.

use crate::format::WeightCoding;
use crate::multiplier::MultiplierKind;
use crate::{Bit, Builder, Design, Module};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// log2 of the most codes a cell's word can hold: four, of a quarter of its width.
pub const MAX_PACKING: u32 = 2;

/// Bits of the field that says how many codes each word holds.
pub fn bits_for_packing(most: u32) -> u32 {
    u32::BITS - most.leading_zeros()
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct CellConfig {
    pub weight_bits: u32,
    pub activation_bits: u32,
    pub accumulator_bits: u32,
    pub mac_stages: u32,
    pub adder_stages: u32,
    /// log2 of the number of weights the cell's memory holds.
    pub addr_bits: u32,
    /// How the stored codes are turned into numbers. See `format`.
    pub coding: WeightCoding,
    /// Float: exponent bits. Posit: `es`. Logarithmic: fraction bits. Otherwise unused.
    pub exponent_bits: u32,
    pub multiplier: MultiplierKind,
    /// The weights are fixed at manufacture: the cell reads a ROM and has no way to write it.
    pub fixed_weights: bool,
    /// Weights are stored in words with error correction. Ignored for fixed weights.
    pub ecc: bool,
    /// Holds the multiplier's input still when the weight is zero, to save switching.
    pub skip_zeros: bool,
}

impl CellConfig {
    /// Bits that travel to every cell for each activation.
    pub fn broadcast_bits(&self) -> u32 {
        if self.multiplier == MultiplierKind::Multiples {
            odd_multiples(self.significand_bits()) * (self.significand_bits() + self.activation_bits)
        } else {
            self.activation_bits
        }
    }

    /// Weights held in each protected word: about 64 bits' worth.
    pub fn ecc_lanes(&self) -> u32 {
        1 << (64 / self.weight_bits.max(1)).max(1).ilog2()
    }

    /// log2 of the most integer codes the cell can keep in each word: two of half its width,
    /// or four of a quarter, while a code keeps two bits. Zero when its codes are not
    /// plain integers or its words are protected.
    pub fn most_packing(&self) -> u32 {
        if !matches!(self.coding, WeightCoding::Affine | WeightCoding::Fixed) || self.has_ecc() {
            return 0;
        }
        (0..=MAX_PACKING).rev().find(|p| self.weight_bits.is_multiple_of(1 << p) && self.weight_bits >> p >= 2).unwrap_or(0)
    }

    /// Whether this cell really protects its memory.
    pub fn has_ecc(&self) -> bool {
        self.ecc && !self.fixed_weights && self.addr_bits > self.ecc_lanes().ilog2()
    }

    /// Cycles from presenting an input until the accumulator has absorbed it.
    pub fn latency(&self) -> u32 {
        let table = (self.coding == WeightCoding::Codebook) as u32;
        // The corrected weight is registered before the multiplier takes it, unless the
        // multiplier only picks among shared multiples, which is shallow anyway.
        let correction = (self.has_ecc() && self.multiplier != MultiplierKind::Multiples) as u32;
        1 + table + correction + (self.mac_stages.min(self.significand_bits()).max(1) - 1) + (self.adder_stages.max(1) - 1) + 1
    }

    pub(crate) fn checked(self) -> Self {
        let weight_bits = self.weight_bits.clamp(2, 32);
        let exponent_bits = self.coding.legal_exponent_bits(weight_bits, self.exponent_bits);
        // Beyond a few bits the shared multiples would need more wires than a multiplier has gates.
        let significand = Self { weight_bits, exponent_bits, ..self }.significand_bits();
        let too_wide = self.coding == WeightCoding::Codebook || significand > MAX_MULTIPLES_BITS;
        let multiplier = if self.multiplier == MultiplierKind::Multiples && too_wide { MultiplierKind::Array } else { self.multiplier };
        Self {
            weight_bits,
            activation_bits: self.activation_bits.clamp(2, 64),
            accumulator_bits: self.accumulator_bits.clamp(2, 128),
            mac_stages: if multiplier == MultiplierKind::Multiples { 1 } else { self.mac_stages.clamp(1, 6) },
            adder_stages: self.adder_stages.clamp(1, 6),
            addr_bits: self.addr_bits.clamp(1, 20),
            coding: self.coding,
            exponent_bits: self.coding.legal_exponent_bits(weight_bits, self.exponent_bits),
            multiplier,
            fixed_weights: self.fixed_weights,
            ecc: self.ecc,
            skip_zeros: self.skip_zeros && multiplier != MultiplierKind::Multiples,
        }
    }
}

/// The widest significand that may use precomputed multiples.
pub const MAX_MULTIPLES_BITS: u32 = 8;

/// How many odd multiples a signed significand of this width can need.
pub fn odd_multiples(significand: u32) -> u32 {
    1 << significand.max(2).saturating_sub(2)
}

pub(crate) struct Gen {
    pub(crate) modules: BTreeMap<String, Module>,
}

impl Gen {
    pub(crate) fn add(&mut self, module: Module) -> String {
        let name = module.name.clone();
        self.modules.insert(name.clone(), module);
        name
    }

    pub(crate) fn full_adder(&mut self) -> String {
        if self.modules.contains_key("full_adder") {
            return "full_adder".into();
        }
        let mut b = Builder::new("full_adder", "Full adder", "Adds three bits and gives the sum bit and the carry bit.");
        let a = b.input("a", 1)[0];
        let x = b.input("b", 1)[0];
        let cin = b.input("cin", 1)[0];
        let sum = b.output("sum", 1)[0];
        let cout = b.output("cout", 1)[0];
        let p = b.wire("p", 1)[0];
        let g1 = b.wire("g1", 1)[0];
        let g2 = b.wire("g2", 1)[0];
        b.group("Sum");
        b.xor_to(a, x, p);
        b.xor_to(p, cin, sum);
        b.group("Carry");
        b.and_to(a, x, g1);
        b.and_to(p, cin, g2);
        b.or_to(g1, g2, cout);
        self.add(b.finish())
    }

    pub(crate) fn ripple_adder(&mut self, width: u32) -> String {
        let name = format!("ripple_adder_{width}");
        if self.modules.contains_key(&name) {
            return name;
        }
        let fa = self.full_adder();
        let wide = width as usize >= crate::rv32::PREFIX_FROM;
        let how = if wide { "The carries come from a prefix tree (Brent-Kung): as deep as twice the logarithm of the bits." } else { "A chain of full adders: each carry feeds the next bit." };
        let mut b = Builder::new(&name, &format!("{width}-bit adder"), how);
        let a = b.input("a", width);
        let x = b.input("b", width);
        let cin = b.input("cin", 1)[0];
        let sum = b.output("sum", width);
        let cout = b.output("cout", 1)[0];
        if wide {
            b.group("Carry tree");
            let mut l = crate::rv32::Logic { b: &mut b, next: 0 };
            let (s, c) = l.prefix_adder(&a, &x, cin);
            for (from, to) in s.iter().zip(&sum) {
                l.b.or_to(*from, Bit::Zero, *to);
            }
            l.b.or_to(c, Bit::Zero, cout);
            return self.add(b.finish());
        }
        b.group("Carry chain");
        let carries = b.wire("carry", width.saturating_sub(1));
        let mut carry = cin;
        for i in 0..width as usize {
            let out = if i + 1 == width as usize { cout } else { carries[i] };
            b.inst(&format!("fa{i}"), &fa, &[("a", vec![a[i]]), ("b", vec![x[i]]), ("cin", vec![carry]), ("sum", vec![sum[i]]), ("cout", vec![out])]);
            carry = out;
        }
        self.add(b.finish())
    }

    pub(crate) fn register(&mut self, width: u32) -> String {
        let name = format!("register_{width}");
        if self.modules.contains_key(&name) {
            return name;
        }
        let mut b = Builder::new(&name, &format!("{width}-bit register"), "Stores the input value on each clock edge.");
        let d = b.input("d", width);
        let q = b.output("q", width);
        b.group("Flip-flops");
        for i in 0..width as usize {
            b.dff_to(d[i], q[i]);
        }
        self.add(b.finish())
    }

    pub(crate) fn mux_bus(&mut self, width: u32) -> String {
        let name = format!("mux_bus_{width}");
        if self.modules.contains_key(&name) {
            return name;
        }
        let mut b = Builder::new(&name, &format!("{width}-bit selector"), "Passes “a” if the select is 0 and “b” if it is 1.");
        let s = b.input("s", 1)[0];
        let a = b.input("a", width);
        let x = b.input("b", width);
        let y = b.output("y", width);
        b.group("Selectors");
        for i in 0..width as usize {
            b.mux_to(s, a[i], x[i], y[i]);
        }
        self.add(b.finish())
    }

    /// `value & enable`, or for the sign row `!value & enable` so the row subtracts.
    fn partial_product(&mut self, width: u32, negate: bool) -> String {
        let name = format!("partial_product_{width}{}", if negate { "_neg" } else { "" });
        if self.modules.contains_key(&name) {
            return name;
        }
        let what = if negate {
            "Sign row: if the weight's bit is 1, inverts the value to subtract it."
        } else {
            "Passes the shifted value if the weight's bit is 1."
        };
        let mut b = Builder::new(&name, "Partial product", what);
        let enable = b.input("enable", 1)[0];
        let value = b.input("value", width);
        let out = b.output("out", width);
        b.group("Gates");
        for i in 0..width as usize {
            if negate {
                let inverted = b.wire(&format!("inv{i}"), 1)[0];
                b.not_to(value[i], inverted);
                b.and_to(enable, inverted, out[i]);
            } else {
                b.and_to(enable, value[i], out[i]);
            }
        }
        self.add(b.finish())
    }

    pub(crate) fn multiplier(&mut self, weight_bits: u32, x_bits: u32, stages: u32) -> String {
        let width = weight_bits + x_bits;
        let stages = stages.min(weight_bits).max(1);
        let name = format!("multiplier_{weight_bits}x{x_bits}_s{stages}");
        if self.modules.contains_key(&name) {
            return name;
        }
        let adder = self.ripple_adder(width);
        let plain = self.partial_product(width, false);
        let negate = self.partial_product(width, true);
        let weight_reg = self.register(weight_bits);
        let x_reg = self.register(x_bits);
        let sum_reg = self.register(width);

        let mut b = Builder::new(
            &name,
            &format!("Signed multiplier {weight_bits} × {x_bits}"),
            "Adds one shifted row per bit of the weight. The last row subtracts, because the top bit has a negative sign.",
        );
        let weight_in = b.input("weight", weight_bits);
        let x_in = b.input("x", x_bits);
        let product = b.output("product", width);

        let cuts: Vec<u32> = (1..stages).map(|k| weight_bits * k / stages).collect();
        let mut weight = weight_in;
        let mut x = x_in;
        let mut running = vec![Bit::Zero; width as usize];

        for row in 0..weight_bits {
            b.group(&format!("Row {row}"));
            let last = row + 1 == weight_bits;
            let sign = *x.last().expect("activation without bits");
            let shifted: Vec<Bit> = (0..width as usize)
                .map(|i| {
                    let i = i as i64 - row as i64;
                    if i < 0 {
                        Bit::Zero
                    } else {
                        x.get(i as usize).copied().unwrap_or(sign)
                    }
                })
                .collect();
            let selected = weight[row as usize];
            let partial = b.wire(&format!("pp{row}"), width);
            b.inst(&format!("partial{row}"), if last { &negate } else { &plain }, &[("enable", vec![selected]), ("value", shifted), ("out", partial.clone())]);
            let sum = b.wire(&format!("sum{row}"), width);
            let unused = b.wire(&format!("carry_out{row}"), 1);
            let cin = if last { selected } else { Bit::Zero };
            b.inst(&format!("adder{row}"), &adder, &[("a", running), ("b", partial), ("cin", vec![cin]), ("sum", sum.clone()), ("cout", unused)]);
            running = sum;

            // A pipeline cut stores everything the later rows still need.
            if cuts.contains(&(row + 1)) && !last {
                b.group(&format!("Stage after row {row}"));
                let q_sum = b.wire(&format!("sum{row}_q"), width);
                b.inst(&format!("sum_reg{row}"), &sum_reg, &[("d", running), ("q", q_sum.clone())]);
                running = q_sum;
                let q_weight = b.wire(&format!("weight{row}_q"), weight_bits);
                b.inst(&format!("weight_reg{row}"), &weight_reg, &[("d", weight), ("q", q_weight.clone())]);
                weight = q_weight;
                let q_x = b.wire(&format!("x{row}_q"), x_bits);
                b.inst(&format!("x_reg{row}"), &x_reg, &[("d", x), ("q", q_x.clone())]);
                x = q_x;
            }
        }

        b.group("Result");
        for (i, bit) in running.iter().enumerate() {
            b.xor_to(*bit, Bit::Zero, product[i]);
        }
        self.add(b.finish())
    }

    /// `fast` swaps the ripple adder for a parallel-prefix one.
    /// With `restartable`, «restart» drops the total and starts again from this value, so
    /// one sum ends and the next begins in the same cycle.
    pub(crate) fn restartable_accumulator(&mut self, width: u32, fast: bool, restartable: bool) -> String {
        let name = format!("accumulator_{width}{}{}", if fast { "_fast" } else { "" }, if restartable { "_restart" } else { "" });
        if self.modules.contains_key(&name) {
            return name;
        }
        let adder = if fast { self.prefix_adder(width) } else { self.ripple_adder(width) };
        let mux = self.mux_bus(width);
        let reg = self.register(width);
        let mut b = Builder::new(&name, &format!("{width}-bit accumulator"), "Adds each valid product to the total. “clear” sets it back to zero.");
        let mut sum_ports: Vec<(&str, Vec<Bit>)> = Vec::new();
        let clear = b.input("clear", 1)[0];
        let restart = restartable.then(|| b.input("restart", 1)[0]);
        let valid = b.input("valid", 1)[0];
        let value = b.input("value", width);
        let total = b.output("total", width);
        let kept = match restart {
            Some(restart) => {
                let keep = b.wire("keep", 1)[0];
                b.not_to(restart, keep);
                let kept = b.wire("kept", width);
                for i in 0..width as usize {
                    b.and_to(keep, total[i], kept[i]);
                }
                kept
            }
            None => total.clone(),
        };
        let not_clear = b.wire("not_clear", 1)[0];
        let sum = b.wire("sum", width);
        let unused = b.wire("carry_out", 1);
        let chosen = b.wire("chosen", width);
        let next = b.wire("next", width);

        b.group("Sum");
        sum_ports.extend([("a", kept.clone()), ("b", value), ("sum", sum.clone())]);
        if !fast {
            sum_ports.extend([("cin", vec![Bit::Zero]), ("cout", unused)]);
        }
        b.inst("adder", &adder, &sum_ports);
        b.group("Decision");
        b.inst("select", &mux, &[("s", vec![valid]), ("a", kept), ("b", sum), ("y", chosen.clone())]);
        b.not_to(clear, not_clear);
        for i in 0..width as usize {
            b.and_to(not_clear, chosen[i], next[i]);
        }
        b.group("Register");
        b.inst("total_reg", &reg, &[("d", next), ("q", total)]);
        self.add(b.finish())
    }

    pub(crate) fn address_counter(&mut self, width: u32) -> String {
        let name = format!("address_counter_{width}");
        if self.modules.contains_key(&name) {
            return name;
        }
        let adder = self.ripple_adder(width);
        let reg = self.register(width);
        let mut b = Builder::new(&name, &format!("{width}-bit address counter"), "Moves to the next memory position at each multiplication.");
        let clear = b.input("clear", 1)[0];
        let advance = b.input("advance", 1)[0];
        let address = b.output("address", width);
        let not_clear = b.wire("not_clear", 1)[0];
        let sum = b.wire("sum", width);
        let unused = b.wire("carry_out", 1);
        let next = b.wire("next", width);

        b.group("Increment");
        b.inst("adder", &adder, &[("a", address.clone()), ("b", vec![Bit::Zero; width as usize]), ("cin", vec![advance]), ("sum", sum.clone()), ("cout", unused)]);
        b.group("Clear");
        b.not_to(clear, not_clear);
        for i in 0..width as usize {
            b.and_to(not_clear, sum[i], next[i]);
        }
        b.group("Register");
        b.inst("address_reg", &reg, &[("d", next), ("q", address)]);
        self.add(b.finish())
    }

    /// An address counter that can also jump to any position.
    pub(crate) fn seek_counter(&mut self, width: u32) -> String {
        let name = format!("seek_counter_{width}");
        if self.modules.contains_key(&name) {
            return name;
        }
        let adder = self.ripple_adder(width);
        let reg = self.register(width);
        let mux = self.mux_bus(width);
        let mut b = Builder::new(&name, &format!("{width}-bit counter with jump"), "Moves to the next position, goes back to zero or jumps to the one it is given.");
        let clear = b.input("clear", 1)[0];
        let advance = b.input("advance", 1)[0];
        let seek = b.input("seek", 1)[0];
        let target = b.input("target", width);
        let address = b.output("address", width);
        let sum = b.wire("sum", width);
        let unused = b.wire("carry_out", 1);
        let next = b.wire("next", width);
        let chosen = b.wire("chosen", width);
        let not_clear = b.wire("not_clear", 1)[0];
        b.group("Increment");
        b.inst("adder", &adder, &[("a", address.clone()), ("b", vec![Bit::Zero; width as usize]), ("cin", vec![advance]), ("sum", sum.clone()), ("cout", unused)]);
        b.group("Clear");
        b.not_to(clear, not_clear);
        for i in 0..width as usize {
            b.and_to(not_clear, sum[i], next[i]);
        }
        b.group("Jump");
        b.inst("target_select", &mux, &[("s", vec![seek]), ("a", next), ("b", target), ("y", chosen.clone())]);
        b.group("Register");
        b.inst("address_reg", &reg, &[("d", chosen), ("q", address)]);
        self.add(b.finish())
    }

    /// Weight memory organised in words of several weights plus their check bits.
    /// Whole words arrive already encoded, so the encoder can be shared by many cells.
    #[allow(clippy::too_many_arguments)]
    fn protected_memory(&mut self, b: &mut Builder, c: &CellConfig, address: Vec<Bit>, load_en: Bit, load_addr: Vec<Bit>, load_data: Vec<Bit>, weight: Vec<Bit>) {
        let lanes = c.ecc_lanes();
        let lane_bits = lanes.trailing_zeros() as usize;
        let word_bits = lanes * c.weight_bits;
        let check_width = crate::reliability::lane_check_bits(lanes, c.weight_bits);
        let corrector = self.lane_corrector(lanes, c.weight_bits);
        let corrected_out = b.output("ecc_corrected", 1);
        let failed_out = b.output("ecc_failed", 1);
        let stored_check = b.wire("stored_check", check_width);
        let stored_word = b.wire("stored_word", word_bits);

        b.group("Weight memory");
        let (word_address, lane_now);
        {
            let mut l = crate::rv32::Logic { b, next: 0 };
            word_address = l.mux_bus(load_en, &address[lane_bits..], &load_addr);
            lane_now = l.fresh_bus(lane_bits);
            l.connect_flops(&address[..lane_bits], &lane_now);
        }
        let mut full_read = stored_word.clone();
        full_read.extend(&stored_check);
        b.sram("weight_sram", c.addr_bits - lane_bits as u32, word_bits + check_width, &[("addr", word_address), ("we", vec![load_en]), ("wdata", load_data), ("rdata", full_read)]);
        b.group("Error correction");
        b.inst("ecc_corrector", &corrector, &[("data", stored_word), ("check", stored_check), ("lane", lane_now), ("weight", weight), ("fixed", corrected_out), ("failed", failed_out)]);
    }

    /// Computes 1×, 3×, 5×… of an activation and registers them, for many cells to share.
    fn multiples_unit(&mut self, significand: u32, a: u32) -> String {
        let name = format!("multiples_unit_{significand}_{a}");
        if self.modules.contains_key(&name) {
            return name;
        }
        let (count, width) = (odd_multiples(significand) as usize, (significand + a) as usize);
        let mut b = Builder::new(&name, "Activation multiples", "Computes each activation's odd multiples once, so the cells only have to choose.");
        let x = b.input("x", a);
        let out = b.output("multiples", (count * width) as u32);
        {
            let mut l = crate::rv32::Logic { b: &mut b, next: 0 };
            let sign = *x.last().expect("empty activation");
            let extended: Vec<Bit> = (0..width).map(|i| x.get(i).copied().unwrap_or(sign)).collect();
            let doubled: Vec<Bit> = std::iter::once(Bit::Zero).chain(extended[..width - 1].iter().copied()).collect();
            let mut current = extended.clone();
            for index in 0..count {
                l.connect_flops(&current, &out[index * width..(index + 1) * width]);
                current = l.adder(&current, &doubled, Bit::Zero).0;
            }
        }
        self.add(b.finish())
    }

    /// The product of a signed weight and an activation, from its precomputed multiples.
    fn pick_multiple(b: &mut Builder, weight: &[Bit], multiples: &[Bit], width: usize, product: &[Bit]) {
        let mut l = crate::rv32::Logic { b, next: 2000 };
        let s = weight.len();
        let sign = weight[s - 1];
        let flipped: Vec<Bit> = weight.iter().map(|bit| l.xor(*bit, sign)).collect();
        let magnitude = l.adder(&flipped, &crate::rv32::constant(0, s), sign).0;
        let nonzero = l.any(&magnitude);

        // Strip trailing zeros: what remains is odd and says which multiple to take.
        let levels = usize::BITS as usize - (s - 1).leading_zeros() as usize;
        let mut value = magnitude;
        let mut shifts = vec![Bit::Zero; levels];
        for level in (0..levels).rev() {
            let amount = 1usize << level;
            let low: Vec<Bit> = value.iter().take(amount).copied().collect();
            let any_low = l.any(&low);
            let strip = l.not(any_low);
            shifts[level] = strip;
            let moved: Vec<Bit> = (0..s).map(|i| value.get(i + amount).copied().unwrap_or(Bit::Zero)).collect();
            value = l.mux_bus(strip, &value, &moved);
        }
        let leaves: Vec<Vec<Bit>> = multiples.chunks(width).map(|m| m.to_vec()).collect();
        let index = &value[1..1 + (s.max(2) - 2)];
        let mut chosen = l.select(index, &leaves);
        for (level, select) in shifts.iter().enumerate() {
            let amount = 1usize << level;
            let moved: Vec<Bit> = (0..width).map(|i| if i >= amount { chosen[i - amount] } else { Bit::Zero }).collect();
            chosen = l.mux_bus(*select, &chosen, &moved);
        }
        let negated: Vec<Bit> = chosen.iter().map(|bit| l.xor(*bit, sign)).collect();
        let signed = l.adder(&negated, &crate::rv32::constant(0, width), sign).0;
        for (from, to) in signed.iter().zip(product) {
            let gated = l.and(*from, nonzero);
            l.b.or_to(gated, Bit::Zero, *to);
        }
    }

    /// `seekable` adds the ports that let the sequencer point the cell at a matrix.
    pub(crate) fn cell(&mut self, c: CellConfig, seekable: bool) -> String {
        let (w, a, acc) = (c.weight_bits, c.activation_bits, c.accumulator_bits);
        let significand = c.significand_bits();
        let product_bits = significand + a;
        let mac_stages = c.mac_stages.min(significand).max(1);
        let counter = if seekable { self.seek_counter(c.addr_bits) } else { self.address_counter(c.addr_bits) };
        let x_reg = self.register(a);
        let multiplier = match c.multiplier {
            MultiplierKind::Array => self.multiplier(significand, a, mac_stages),
            MultiplierKind::Multiples => self.multiples_unit(significand, a),
            kind => self.tree_multiplier(kind, significand, a, mac_stages),
        };
        let uses_multiples = c.multiplier == MultiplierKind::Multiples;
        let acc_reg = self.register(acc);
        let accumulator = self.restartable_accumulator(acc, c.multiplier != MultiplierKind::Array, seekable);

        let mut b = Builder::new(
            "compute_cell",
            "Compute cell",
            "Keeps its weights, multiplies each by the incoming activation and accumulates the sum.",
        );
        let clear = b.input("clear", 1)[0];
        let clear_total = b.input("clear_total", 1)[0];
        // A cell that can seek also ends one group and starts the next in the same cycle.
        let restart = seekable.then(|| b.input("restart", 1)[0]);
        let protected = c.has_ecc();
        let (load_addr_bits, load_data_bits) = if protected {
            let word = c.ecc_lanes() * w;
            (c.addr_bits - c.ecc_lanes().trailing_zeros(), word + crate::reliability::lane_check_bits(c.ecc_lanes(), w))
        } else {
            (c.addr_bits, w)
        };
        let load_ports = (!c.fixed_weights).then(|| (b.input("load_en", 1), b.input("load_addr", load_addr_bits), b.input("load_data", load_data_bits)));
        let run = b.input("run", 1)[0];
        let x = b.input("x", a);
        let total = b.output("total", acc);
        let table_ports = (c.coding == WeightCoding::Codebook && !c.fixed_weights).then(|| (b.input("table_en", 1)[0], b.input("table_addr", w), b.input("table_data", significand)));
        let exp_base = c.shifts().then(|| b.input("exp_base", c.exponent_width()));

        b.group("Addresses");
        let address = b.wire("address", c.addr_bits);
        let seek_ports = seekable.then(|| (b.input("seek", 1), b.input("seek_addr", c.addr_bits)));
        // A matrix whose codes are narrower keeps 2^p of them in each word: the cell takes them
        // from the low end up, and moves to the next word after the last.
        let most = if seekable { c.most_packing() } else { 0 };
        let slots = (most > 0).then(|| {
            let packing = b.input("packing", bits_for_packing(most));
            let seek = seek_ports.as_ref().map_or(Bit::Zero, |(seek, _)| seek[0]);
            let mut l = crate::rv32::Logic { b: &mut b, next: 70_000 };
            let slot = l.fresh_bus(most as usize);
            // The last slot of the word for each packing, and whether that one is chosen.
            let last: Vec<Bit> = (0..=most).map(|p| {
                let chosen = l.equals(&packing, p);
                let full = l.all(&slot[..p as usize]);
                l.and(chosen, full)
            }).collect();
            let last = l.any(&last);
            let stepped = l.adder(&slot, &vec![Bit::Zero; most as usize], Bit::One).0;
            let wrapped = l.mux_bus(last, &stepped, &vec![Bit::Zero; most as usize]);
            let moved = l.mux_bus(run, &slot, &wrapped);
            let restart = l.or(clear, seek);
            let kept = l.not(restart);
            let next: Vec<Bit> = moved.iter().map(|bit| l.and(*bit, kept)).collect();
            l.connect_flops(&next, &slot);
            let advance = l.and(run, last);
            // The word read now was addressed a cycle ago, with that cycle's slot.
            let late_slot = l.fresh_bus(most as usize);
            l.connect_flops(&slot, &late_slot);
            let late_packing = l.fresh_bus(packing.len());
            l.connect_flops(&packing, &late_packing);
            (advance, late_slot, late_packing)
        });
        let advance = slots.as_ref().map_or(run, |(advance, _, _)| *advance);
        let mut counter_pins = vec![("clear", vec![clear]), ("advance", vec![advance]), ("address", address.clone())];
        if let Some((seek, target)) = seek_ports {
            counter_pins.extend([("seek", seek), ("target", target)]);
        }
        b.inst("counter", &counter, &counter_pins);
        let weight = b.wire("weight", w);
        match load_ports {
            Some((load_en, load_addr, load_data)) if protected => {
                self.protected_memory(&mut b, &c, address, load_en[0], load_addr, load_data, weight.clone());
            }
            Some((load_en, load_addr, load_data)) => {
                let addr_mux = self.mux_bus(c.addr_bits);
                let mem_addr = b.wire("mem_addr", c.addr_bits);
                b.inst("address_select", &addr_mux, &[("s", load_en.clone()), ("a", address), ("b", load_addr), ("y", mem_addr.clone())]);
                b.group("Weight memory");
                b.sram("weight_sram", c.addr_bits, w, &[("addr", mem_addr), ("we", load_en), ("wdata", load_data), ("rdata", weight.clone())]);
            }
            None => {
                b.group("Weight memory");
                b.rom("weight_rom", c.addr_bits, w, &[("addr", address), ("rdata", weight.clone())]);
            }
        }

        let weight = match slots {
            Some((_, slot, packing)) => {
                let mut l = crate::rv32::Logic { b: &mut b, next: 70_100 };
                let mut chosen = weight.clone();
                for p in 1..=most {
                    let width = (w >> p) as usize;
                    let pieces: Vec<Vec<Bit>> = weight.chunks(width).map(<[Bit]>::to_vec).collect();
                    let piece = l.select(&slot[..p as usize], &pieces);
                    let sign = piece[width - 1];
                    let widened: Vec<Bit> = (0..w as usize).map(|i| piece.get(i).copied().unwrap_or(sign)).collect();
                    let here = l.equals(&packing, p);
                    chosen = l.mux_bus(here, &chosen, &widened);
                }
                chosen
            }
            None => weight,
        };
        // Correcting a word is as deep as a multiplier: the corrected weight waits in a
        // register, and the activation with it, so the correction has a cycle of its own.
        let weight = if protected && !uses_multiples {
            let held = b.wire("weight_corrected", w);
            let weight_reg = self.register(w);
            b.inst("weight_corrected_reg", &weight_reg, &[("d", weight), ("q", held.clone())]);
            held
        } else {
            weight
        };
        b.group("Format decoding");
        let decoded = self.decode(&mut b, &c, weight, table_ports);

        b.group("Alignment");
        let x_aligned = b.wire("x_aligned", a);
        let multiples = uses_multiples.then(|| {
            let bus = b.wire("multiples", odd_multiples(significand) * product_bits);
            b.inst("multiples_unit", &multiplier, &[("x", x.clone()), ("multiples", bus.clone())]);
            bus
        });
        if !uses_multiples {
            b.inst("x_reg", &x_reg, &[("d", x), ("q", x_aligned.clone())]);
        }
        let mut valid = b.wire("valid0", 1)[0];
        b.dff_to(run, valid);
        let x_aligned = if c.coding == WeightCoding::Codebook {
            let second = b.wire("x_table", a);
            b.inst("x_table_reg", &x_reg, &[("d", x_aligned), ("q", second.clone())]);
            let next = b.wire("valid_table", 1)[0];
            b.dff_to(valid, next);
            valid = next;
            second
        } else {
            x_aligned
        };
        let x_aligned = if protected && !uses_multiples {
            let late = b.wire("x_corrected", a);
            b.inst("x_corrected_reg", &x_reg, &[("d", x_aligned), ("q", late.clone())]);
            let next = b.wire("valid_corrected", 1)[0];
            b.dff_to(valid, next);
            valid = next;
            late
        } else {
            x_aligned
        };
        let window = exp_base.map(|base| self.exponent_window(&mut b, &c, &decoded.exponent, &base));

        b.group("Multiplier");
        let product = b.wire("product", product_bits);
        match &multiples {
            Some(bus) => Self::pick_multiple(&mut b, &decoded.significand, bus, product_bits as usize, &product),
            None => {
                let operand = if c.skip_zeros {
                    let mut l = crate::rv32::Logic { b: &mut b, next: 3000 };
                    let live = l.any(&decoded.significand);
                    x_aligned.iter().map(|bit| l.and(*bit, live)).collect()
                } else {
                    x_aligned
                };
                b.inst("multiplier", &multiplier, &[("weight", decoded.significand), ("x", operand), ("product", product.clone())]);
            }
        }
        let mut window = window;
        for stage in 1..mac_stages {
            let next = b.wire(&format!("valid_mul{stage}"), 1)[0];
            b.dff_to(valid, next);
            valid = next;
            window = window.map(|(amount, under)| {
                let q = b.wire(&format!("amount_mul{stage}"), amount.len() as u32);
                for (d, out) in amount.iter().zip(&q) {
                    b.dff_to(*d, *out);
                }
                let q_under = b.wire(&format!("under_mul{stage}"), 1)[0];
                b.dff_to(under, q_under);
                (q, q_under)
            });
        }

        b.group("Adder stages");
        let sign = *product.last().expect("product without bits");
        let mut value: Vec<Bit> = (0..acc as usize).map(|i| product.get(i).copied().unwrap_or(sign)).collect();
        if let Some((amount, under)) = window {
            value = self.place_product(&mut b, acc, value, &amount, under);
        }
        for stage in 1..c.adder_stages {
            let q = b.wire(&format!("value_q{stage}"), acc);
            b.inst(&format!("value_reg{stage}"), &acc_reg, &[("d", value), ("q", q.clone())]);
            value = q;
            let next = b.wire(&format!("valid_add{stage}"), 1)[0];
            b.dff_to(valid, next);
            valid = next;
        }

        b.group("Accumulator");
        let clear_acc = b.wire("clear_acc", 1)[0];
        b.or_to(clear, clear_total, clear_acc);
        let mut pins = vec![("clear", vec![clear_acc]), ("valid", vec![valid]), ("value", value), ("total", total)];
        if let Some(restart) = restart {
            pins.push(("restart", vec![restart]));
        }
        b.inst("accumulator", &accumulator, &pins);
        self.add(b.finish())
    }
}

/// The complete cell netlist for a configuration.
/// The part of a cell that the real chip shares among all of them, if any.
pub fn shared_unit_design(config: CellConfig) -> Option<Design> {
    let c = config.checked();
    (c.multiplier == MultiplierKind::Multiples).then(|| {
        let mut gen = Gen { modules: BTreeMap::new() };
        let top = gen.multiples_unit(c.significand_bits(), c.activation_bits);
        Design { top, modules: gen.modules }
    })
}

pub fn cell_design(config: CellConfig) -> Design {
    let mut gen = Gen { modules: BTreeMap::new() };
    let top = gen.cell(config.checked(), false);
    Design { top, modules: gen.modules }
}
