//! Weight storage formats and the gates that read them. Every format reduces to a
//! signed significand and, for those that float, an exponent that shifts the product.

use crate::cell::{CellConfig, Gen};
use crate::die::{barrel_left, leading_zeros, mux_tree};
use crate::{Bit, Builder};
use serde::{Deserialize, Serialize};

/// Bits of each codebook entry.
pub const TABLE_BITS: u32 = 16;
/// Octaves below the largest exponent the accumulator keeps for floating formats.
pub const FLOAT_WINDOW: u32 = 20;
/// Spare accumulator bits, so that long sums do not overflow.
pub const SUM_GUARD_BITS: u32 = 8;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WeightCoding {
    /// Qn two's complement integers with one scale per tensor.
    Fixed,
    /// Integer codes with a scale, and optionally a bias, per group of weights.
    #[default]
    Affine,
    Float,
    Posit,
    /// Sign and the base-2 logarithm of the magnitude.
    Log,
    /// Codes that index a table of real values.
    Codebook,
}

impl WeightCoding {
    /// Clamps the format's second parameter to what `bits` total can hold.
    pub fn legal_exponent_bits(self, bits: u32, wanted: u32) -> u32 {
        match self {
            WeightCoding::Float => wanted.clamp(1, bits - 1),
            WeightCoding::Posit => wanted.min(bits.saturating_sub(3)),
            WeightCoding::Log => wanted.clamp(1, 8).min(bits.saturating_sub(2).max(1)),
            _ => 0,
        }
    }

    pub fn shifts(self) -> bool {
        matches!(self, WeightCoding::Float | WeightCoding::Posit | WeightCoding::Log)
    }
}

/// A complete weight format, as the simulators need it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WeightFormat {
    pub coding: WeightCoding,
    pub bits: u32,
    pub exponent_bits: u32,
    pub bias: bool,
    pub group_size: u32,
    /// The group scale is a full float (NVFP4) rather than a power of two (MXFP4).
    #[serde(default)]
    pub fine_scale: bool,
    /// Keeps the checkpoint's own quantisation whenever it can.
    pub keep_source: bool,
}

const fn log_mantissa(fraction: u32) -> u32 {
    fraction + 4
}

fn ceil_log2(value: u32) -> u32 {
    32 - value.saturating_sub(1).leading_zeros()
}

impl CellConfig {
    pub fn shifts(&self) -> bool {
        self.coding.shifts()
    }

    /// Width of the signed significand that reaches the multiplier.
    pub fn significand_bits(&self) -> u32 {
        let (w, e) = (self.weight_bits, self.exponent_bits);
        match self.coding {
            WeightCoding::Fixed | WeightCoding::Affine => w,
            WeightCoding::Float => w - 1 - e + 2,
            WeightCoding::Posit => w - 3 - e + 2,
            WeightCoding::Log => log_mantissa(e) + 2,
            WeightCoding::Codebook => TABLE_BITS,
        }
    }

    /// Width of the decoded exponent, and of the `exp_base` input.
    pub fn exponent_width(&self) -> u32 {
        let (w, e) = (self.weight_bits, self.exponent_bits);
        match self.coding {
            WeightCoding::Float => e,
            WeightCoding::Posit => ceil_log2(2 * w) + e,
            WeightCoding::Log => w - 1 - e,
            _ => 0,
        }
    }

    /// Bits needed to accumulate these products without losing the guard or the window.
    pub fn required_accumulator(&self) -> u32 {
        let window = if self.shifts() { FLOAT_WINDOW.min(1u32.checked_shl(self.exponent_width()).unwrap_or(u32::MAX) - 1) } else { 0 };
        (self.significand_bits() + self.activation_bits + SUM_GUARD_BITS + window).min(128)
    }

    /// Reference arithmetic: the significand and biased exponent a stored code means.
    pub fn decode_code(&self, code: u64) -> (i64, u64) {
        let (w, e) = (self.weight_bits, self.exponent_bits);
        let mask = |bits: u32| (1u64 << bits) - 1;
        let code = code & mask(w);
        match self.coding {
            WeightCoding::Fixed | WeightCoding::Affine => (((code << (64 - w)) as i64) >> (64 - w), 0),
            WeightCoding::Float => {
                let fraction_bits = w - 1 - e;
                let exponent = (code >> fraction_bits) & mask(e);
                let fraction = code & mask(fraction_bits);
                let magnitude = (((exponent != 0) as u64) << fraction_bits | fraction) as i64;
                (if code >> (w - 1) == 1 { -magnitude } else { magnitude }, exponent.max(1))
            }
            WeightCoding::Posit => {
                let negative = code >> (w - 1) == 1;
                let body = if negative { code.wrapping_neg() & mask(w) } else { code };
                let rest = body & mask(w - 1);
                if rest == 0 {
                    return (0, 0);
                }
                let first = (rest >> (w - 2)) & 1;
                let inverted = if first == 1 { !rest & mask(w - 1) } else { rest };
                let run = if inverted == 0 { w - 1 } else { (w - 1) - (64 - inverted.leading_zeros()) };
                let regime = if first == 1 { w as u64 - 2 + run as u64 } else { w as u64 - 1 - run as u64 };
                let aligned = (rest << (run + 1).min(63)) & mask(w - 1);
                let exponent = if e == 0 { 0 } else { aligned >> (w - 1 - e) };
                let fraction_bits = w - 3 - e;
                let fraction = (aligned >> 2) & mask(fraction_bits);
                let magnitude = ((1u64 << fraction_bits) | fraction) as i64;
                (if negative { -magnitude } else { magnitude }, regime << e | exponent)
            }
            WeightCoding::Log => {
                let integer = (code >> e) & mask(w - 1 - e);
                let fraction = code & mask(e);
                if integer == 0 && fraction == 0 {
                    return (0, 0);
                }
                let magnitude = log_table(e)[fraction as usize] as i64;
                (if code >> (w - 1) == 1 { -magnitude } else { magnitude }, integer)
            }
            WeightCoding::Codebook => (0, 0),
        }
    }
}

/// `2^(i / 2^fraction)` as an integer with the table's mantissa bits.
pub fn log_table(fraction: u32) -> Vec<u64> {
    let mantissa = log_mantissa(fraction);
    let top = (1u64 << (mantissa + 1)) - 1;
    (0..1u32 << fraction)
        .map(|i| ((2f64.powf(i as f64 / (1u64 << fraction) as f64) * (1u64 << mantissa) as f64).round() as u64).min(top))
        .collect()
}

pub(crate) struct Decoded {
    pub(crate) significand: Vec<Bit>,
    pub(crate) exponent: Vec<Bit>,
}

fn literal(value: u64, width: usize) -> Vec<Bit> {
    (0..width).map(|i| if value >> i & 1 == 1 { Bit::One } else { Bit::Zero }).collect()
}

fn any(b: &mut Builder, name: &str, bits: &[Bit]) -> Bit {
    let mut seen = Bit::Zero;
    for (i, bit) in bits.iter().enumerate() {
        let next = b.wire(&format!("{name}{i}"), 1)[0];
        b.or_to(seen, *bit, next);
        seen = next;
    }
    seen
}

impl Gen {
    /// `-value` when `negative`, widened by nothing: the caller leaves room for the sign.
    fn negate_if(&mut self, b: &mut Builder, name: &str, negative: Bit, value: &[Bit]) -> Vec<Bit> {
        let width = value.len() as u32;
        let adder = self.ripple_adder(width);
        let flipped = b.wire(&format!("{name}_xor"), width);
        for (bit, out) in value.iter().zip(&flipped) {
            b.xor_to(*bit, negative, *out);
        }
        let sum = b.wire(name, width);
        let unused = b.wire(&format!("{name}_cout"), 1);
        b.inst(&format!("{name}_adder"), &adder, &[("a", flipped), ("b", vec![Bit::Zero; width as usize]), ("cin", vec![negative]), ("sum", sum.clone()), ("cout", unused)]);
        sum
    }

    /// Turns the stored code into what the multiplier and the shifter need.
    pub(crate) fn decode(&mut self, b: &mut Builder, c: &CellConfig, code: Vec<Bit>, table: Option<(Bit, Vec<Bit>, Vec<Bit>)>) -> Decoded {
        let (w, e) = (c.weight_bits as usize, c.exponent_bits as usize);
        match c.coding {
            WeightCoding::Fixed | WeightCoding::Affine => Decoded { significand: code, exponent: Vec::new() },
            WeightCoding::Float => {
                let fraction_bits = w - 1 - e;
                let sign = code[w - 1];
                let exponent_field = code[fraction_bits..w - 1].to_vec();
                let hidden = any(b, "hidden", &exponent_field);
                let mut magnitude = code[..fraction_bits].to_vec();
                magnitude.push(hidden);
                magnitude.push(Bit::Zero);
                let significand = self.negate_if(b, "signed_float", sign, &magnitude);
                // A subnormal shares the exponent of the smallest normal number.
                let mut exponent = exponent_field;
                let subnormal = b.wire("subnormal", 1)[0];
                b.not_to(hidden, subnormal);
                let lowest = b.wire("exponent_low", 1)[0];
                b.or_to(exponent[0], subnormal, lowest);
                exponent[0] = lowest;
                Decoded { significand, exponent }
            }
            WeightCoding::Posit => self.decode_posit(b, c, code),
            WeightCoding::Log => {
                let integer_bits = w - 1 - e;
                let sign = code[w - 1];
                let fraction = code[..e].to_vec();
                let integer = code[e..w - 1].to_vec();
                let nonzero = any(b, "log_nonzero", &code[..w - 1]);
                let mantissa = log_mantissa(c.exponent_bits) as usize;
                let leaves: Vec<Vec<Bit>> = log_table(c.exponent_bits).iter().map(|v| literal(*v, mantissa + 1)).collect();
                let mux = self.mux_bus(mantissa as u32 + 1);
                let value = mux_tree(b, &mux, leaves, &fraction);
                let mut magnitude = b.wire("log_magnitude", mantissa as u32 + 1);
                for (bit, out) in value.iter().zip(magnitude.clone()) {
                    b.and_to(nonzero, *bit, out);
                }
                magnitude.push(Bit::Zero);
                let significand = self.negate_if(b, "signed_log", sign, &magnitude);
                debug_assert_eq!(integer.len(), integer_bits);
                Decoded { significand, exponent: integer }
            }
            WeightCoding::Codebook => {
                let significand = b.wire("table_value", TABLE_BITS);
                match table {
                    Some((enable, addr, data)) => {
                        let mux = self.mux_bus(c.weight_bits);
                        let chosen = b.wire("table_select", c.weight_bits);
                        b.inst("table_address", &mux, &[("s", vec![enable]), ("a", code), ("b", addr), ("y", chosen.clone())]);
                        b.sram("codebook", c.weight_bits, TABLE_BITS, &[("addr", chosen), ("we", vec![enable]), ("wdata", data), ("rdata", significand.clone())]);
                    }
                    None => b.rom("codebook", c.weight_bits, TABLE_BITS, &[("addr", code), ("rdata", significand.clone())]),
                }
                Decoded { significand, exponent: Vec::new() }
            }
        }
    }

    fn decode_posit(&mut self, b: &mut Builder, c: &CellConfig, code: Vec<Bit>) -> Decoded {
        let (w, e) = (c.weight_bits as usize, c.exponent_bits as usize);
        let width = c.exponent_width() as usize;
        let sign = code[w - 1];
        let body = self.negate_if(b, "posit_body", sign, &code);
        let rest = body[..w - 1].to_vec();
        let nonzero = any(b, "posit_nonzero", &rest);
        let first = rest[w - 2];

        // A run of ones is inverted, so one counter measures both kinds of regime.
        let inverted = b.wire("posit_inverted", w as u32 - 1);
        for (bit, out) in rest.iter().zip(&inverted) {
            b.xor_to(*bit, first, *out);
        }
        let run = leading_zeros(b, "posit_run", &inverted, ceil_log2(w as u32 + 1));

        // Remove the regime and its terminator.
        let mux = self.mux_bus(w as u32 - 1);
        let once: Vec<Bit> = (0..w - 1).map(|i| if i >= 1 { rest[i - 1] } else { Bit::Zero }).collect();
        let shifted = barrel_left(b, &mux, "posit_shift", &once, &run);

        let fraction_bits = w - 3 - e;
        let fraction = shifted[2..2 + fraction_bits].to_vec();
        let exponent_field = shifted[w - 1 - e..w - 1].to_vec();

        let adder = self.ripple_adder(width as u32);
        let run_wide: Vec<Bit> = (0..width).map(|i| run.get(i).copied().unwrap_or(Bit::Zero)).collect();
        let above = b.wire("posit_above", width as u32);
        let unused = b.wire("posit_above_c", 1);
        b.inst("posit_above_adder", &adder, &[("a", run_wide.clone()), ("b", literal(w as u64 - 2, width)), ("cin", vec![Bit::Zero]), ("sum", above.clone()), ("cout", unused)]);
        let inverted_run = b.wire("posit_run_n", width as u32);
        for (bit, out) in run_wide.iter().zip(&inverted_run) {
            b.not_to(*bit, *out);
        }
        let below = b.wire("posit_below", width as u32);
        let unused = b.wire("posit_below_c", 1);
        b.inst("posit_below_adder", &adder, &[("a", inverted_run), ("b", literal(w as u64 - 1, width)), ("cin", vec![Bit::One]), ("sum", below.clone()), ("cout", unused)]);
        let select = self.mux_bus(width as u32);
        let regime = b.wire("posit_regime", width as u32);
        b.inst("posit_regime_select", &select, &[("s", vec![first]), ("a", below), ("b", above), ("y", regime.clone())]);

        let mut exponent = exponent_field;
        exponent.extend(regime.iter().take(width - e).copied());

        let mut magnitude = fraction;
        magnitude.push(nonzero);
        magnitude.push(Bit::Zero);
        let significand = self.negate_if(b, "signed_posit", sign, &magnitude);
        Decoded { significand, exponent }
    }

    /// `exponent - base`, saturated to the shifter's range, and whether it fell below it.
    pub(crate) fn exponent_window(&mut self, b: &mut Builder, c: &CellConfig, exponent: &[Bit], base: &[Bit]) -> (Vec<Bit>, Bit) {
        let width = exponent.len() + 1;
        let adder = self.ripple_adder(width as u32);
        let a: Vec<Bit> = exponent.iter().copied().chain([Bit::Zero]).collect();
        let inverted = b.wire("base_n", width as u32);
        for (i, out) in inverted.iter().enumerate() {
            b.not_to(base.get(i).copied().unwrap_or(Bit::Zero), *out);
        }
        let difference = b.wire("exponent_difference", width as u32);
        let unused = b.wire("exponent_c", 1);
        b.inst("exponent_subtract", &adder, &[("a", a), ("b", inverted), ("cin", vec![Bit::One]), ("sum", difference.clone()), ("cout", unused)]);
        let under = difference[width - 1];

        let shift_bits = ceil_log2(c.accumulator_bits) as usize;
        let high: Vec<Bit> = difference.iter().take(width - 1).skip(shift_bits).copied().collect();
        let too_big = any(b, "exponent_big", &high);
        let amount = b.wire("shift_amount", shift_bits as u32);
        for (i, out) in amount.iter().enumerate() {
            b.or_to(difference.get(i).copied().filter(|_| i < width - 1).unwrap_or(Bit::Zero), too_big, *out);
        }
        (amount, under)
    }

    /// Puts the product at its binary position in the accumulator.
    pub(crate) fn place_product(&mut self, b: &mut Builder, acc: u32, value: Vec<Bit>, amount: &[Bit], under: Bit) -> Vec<Bit> {
        let mux = self.mux_bus(acc);
        let moved = barrel_left(b, &mux, "product_shift", &value, amount);
        let keep = b.wire("in_window", 1)[0];
        b.not_to(under, keep);
        let out = b.wire("placed", acc);
        for (bit, y) in moved.iter().zip(&out) {
            b.and_to(keep, *bit, *y);
        }
        out
    }
}

/// Average toggles per multiplication, weighted by each gate's relative capacitance,
/// measured by running the real cell gates on pseudo-random weights and activations.
pub fn switching_per_mac(config: CellConfig, zero_fraction: f64) -> f64 {
    // A stored weight toggles the same gates however it got into the memory.
    let config = CellConfig { fixed_weights: false, ecc: false, ..config.checked() };
    let design = crate::cell_design(config);
    let Ok(mut sim) = crate::Simulator::new(&design) else { return 0.0 };
    const WORDS: u64 = 32;
    const CYCLES: u32 = 64;
    let mut state = 0x1F2E_3D4C_5B6A_7988u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let w = config.weight_bits;
    let set = |sim: &mut crate::Simulator, port: &str, value: u64| sim.set(port, value).is_ok();
    set(&mut sim, "clear", 1);
    sim.tick();
    set(&mut sim, "clear", 0);
    if config.shifts() {
        set(&mut sim, "exp_base", 1u64 << (config.exponent_width().max(1) - 1));
    }
    if config.coding == WeightCoding::Codebook {
        set(&mut sim, "table_en", 1);
        for address in 0..1u64 << w.min(8) {
            set(&mut sim, "table_addr", address);
            set(&mut sim, "table_data", next() >> 48);
            sim.tick();
        }
        set(&mut sim, "table_en", 0);
    }
    set(&mut sim, "load_en", 1);
    for address in 0..WORDS.min(1 << config.addr_bits) {
        set(&mut sim, "load_addr", address);
        let zero = (next() % 1000) as f64 / 1000.0 < zero_fraction;
        set(&mut sim, "load_data", if zero { 0 } else { next() >> (64 - w) });
        sim.tick();
    }
    set(&mut sim, "load_en", 0);
    set(&mut sim, "run", 1);
    let shift = 64 - config.activation_bits;
    let counts = sim.switching_activity(CYCLES, |sim, _| {
        sim.set("x", next() >> shift).expect("puerto de activación");
    });
    let weights = [1.0, 1.0, 1.6, 0.6, 1.4, 2.2];
    let score = |counts: [u64; 6]| counts.iter().zip(weights).map(|(count, weight)| *count as f64 * weight).sum::<f64>() / CYCLES as f64;
    let total = score(counts);
    // A unit shared by every cell is not paid for by each multiplication.
    let shared = crate::shared_unit_design(config).and_then(|unit| crate::Simulator::new(&unit).ok()).map_or(0.0, |mut unit| {
        score(unit.switching_activity(CYCLES, |sim, _| {
            sim.set("x", next() >> shift).expect("puerto de activación");
        }))
    });
    (total - shared).max(0.0)
}
