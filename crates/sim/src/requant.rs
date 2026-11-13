//! Rewrites a checkpoint's weights in the format the chip stores, so a simulation
//! sees exactly the precision the hardware would have.

use netlist::format::{WeightCoding, WeightFormat, FLOAT_WINDOW};
use rayon::prelude::*;
use std::collections::BTreeSet;

/// What a format needs to know about a whole tensor.
#[derive(Debug)]
pub struct Reference {
    pub min: f32,
    pub max: f32,
    /// Sorted real values for codebook formats.
    pub table: Vec<f32>,
}

impl Reference {
    pub fn largest(&self) -> f32 {
        self.min.abs().max(self.max.abs())
    }
}

/// Representative values for a table of `entries`: quantiles, refined by k-means.
pub fn train_table(mut sample: Vec<f32>, entries: usize) -> Vec<f32> {
    sample.sort_by(f32::total_cmp);
    if sample.is_empty() {
        return vec![0.0; entries];
    }
    // Spread evenly over the range: starting from quantiles leaves the rare large weights,
    // which matter most, without an entry of their own.
    let (low, high) = (sample[0] as f64, sample[sample.len() - 1] as f64);
    let mut table: Vec<f32> = (0..entries).map(|i| (low + (high - low) * i as f64 / (entries - 1).max(1) as f64) as f32).collect();
    for _ in 0..8 {
        let mut sums = vec![(0f64, 0u32); entries];
        for v in &sample {
            let slot = nearest(&table, *v);
            sums[slot].0 += *v as f64;
            sums[slot].1 += 1;
        }
        for (entry, (sum, count)) in table.iter_mut().zip(&sums) {
            if *count > 0 {
                *entry = (*sum / *count as f64) as f32;
            }
        }
        table.sort_by(f32::total_cmp);
    }
    table
}

fn nearest(table: &[f32], value: f32) -> usize {
    let after = table.partition_point(|t| *t < value);
    if after == 0 {
        0
    } else if after == table.len() || value - table[after - 1] <= table[after] - value {
        after - 1
    } else {
        after
    }
}

/// The binary exponent `e` such that `2^(e-1) <= |value| < 2^e`, or `None` for zero.
#[inline]
fn binary_exponent(value: f64) -> Option<i32> {
    if value == 0.0 {
        return None;
    }
    let raw = ((value.to_bits() >> 52) & 0x7FF) as i32;
    Some(if raw == 0 { value.abs().log2().floor() as i32 + 1 } else { raw - 1022 })
}

#[inline]
fn power(exponent: i32) -> f64 {
    f64::from_bits(((exponent + 1023).clamp(1, 2046) as u64) << 52)
}

/// Rounds in place. Formats with groups see the row one group at a time.
pub fn quantize(format: &WeightFormat, reference: &Reference, row: &mut [f32]) {
    if format.group_size > 0 && row.len() >= format.group_size as usize {
        let size = format.group_size as usize;
        for group in row.chunks_mut(size) {
            let (min, max) = group.iter().fold((f32::MAX, f32::MIN), |(lo, hi), v| (lo.min(*v), hi.max(*v)));
            quantize_values(format, min, max, &reference.table, group);
        }
    } else {
        quantize_values(format, reference.min, reference.max, &reference.table, row);
    }
}

fn quantize_values(format: &WeightFormat, min: f32, max: f32, table: &[f32], values: &mut [f32]) {
    let bits = format.bits;
    let largest = min.abs().max(max.abs()) as f64;
    if largest == 0.0 {
        values.fill(0.0);
        return;
    }
    // Floating formats place their range so that the largest weight is representable.
    let top = binary_exponent(largest).unwrap_or(0);
    match format.coding {
        WeightCoding::Fixed => {
            let high = ((1u64 << (bits - 1)) - 1) as f64;
            let step = largest / high;
            for v in values.iter_mut() {
                *v = (((*v as f64 / step).round()).clamp(-high - 1.0, high) * step) as f32;
            }
        }
        WeightCoding::Affine if format.bias => {
            let levels = ((1u64 << bits) - 1) as f64;
            let scale = (max as f64 - min as f64) / levels;
            if scale > 0.0 {
                for v in values.iter_mut() {
                    *v = (((*v as f64 - min as f64) / scale).round().clamp(0.0, levels) * scale + min as f64) as f32;
                }
            }
        }
        WeightCoding::Affine => {
            let high = ((1u64 << (bits - 1)) - 1) as f64;
            let scale = largest / high;
            for v in values.iter_mut() {
                *v = ((*v as f64 / scale).round().clamp(-high - 1.0, high) * scale) as f32;
            }
        }
        WeightCoding::Float => {
            let fraction = (bits - 1 - format.exponent_bits) as i32;
            let range = (1i64 << format.exponent_bits.min(30)) - 2;
            let lowest = -(range.min(FLOAT_WINDOW as i64) as i32);
            let flushes = range > FLOAT_WINDOW as i64;
            let top_mantissa = power(fraction + 1) - 1.0;
            // A fine scale stretches the group so its largest weight uses the whole range.
            let gain = if format.fine_scale { power(top) * (1.0 - power(-fraction - 1)) / largest } else { 1.0 };
            for v in values.iter_mut() {
                let x = *v as f64 * gain;
                let Some(exponent) = binary_exponent(x) else { continue };
                let relative = exponent - 1 - top + 1;
                if relative < lowest && flushes {
                    *v = 0.0;
                    continue;
                }
                let at = relative.max(lowest) + top - 1 - fraction;
                let mantissa = (x / power(at)).round().clamp(-top_mantissa, top_mantissa);
                *v = (mantissa * power(at) / gain) as f32;
            }
        }
        WeightCoding::Posit => {
            let es = format.exponent_bits as i32;
            let n = bits as i32;
            for v in values.iter_mut() {
                let x = *v as f64;
                let Some(exponent) = binary_exponent(x) else { continue };
                let relative = exponent - top - 1;
                if relative < -(FLOAT_WINDOW as i32) {
                    *v = 0.0;
                    continue;
                }
                let k = relative.div_euclid(1 << es);
                let regime = if k >= 0 { k + 2 } else { -k + 1 };
                let fraction = (n - 1 - regime - es).max(0);
                let at = exponent - 1 - fraction;
                *v = ((x / power(at)).round() * power(at)) as f32;
            }
        }
        WeightCoding::Log => {
            let fraction = format.exponent_bits as i32;
            let steps = (1i64 << fraction) as f64;
            let limit = -(((1i64 << (bits as i32 - 1 - fraction).min(30)) - 1) as f64);
            let reference = power(top) ;
            for v in values.iter_mut() {
                let x = *v as f64;
                if x == 0.0 {
                    continue;
                }
                let level = ((x.abs() / reference).log2() * steps).round() / steps;
                *v = if level < limit { 0.0 } else { (x.signum() * 2f64.powf(level) * reference) as f32 };
            }
        }
        WeightCoding::Codebook => {
            for v in values.iter_mut() {
                *v = table[nearest(table, *v)];
            }
        }
    }
}

/// Fractional bits used to compare normalised weights across groups.
const KEY_FRACTION: i32 = 40;

/// A tensor rewritten once in a narrow format: each weight is `table[code] · scale + bias`.
pub struct Recoded {
    pub codes: Vec<u8>,
    pub table: Box<[f32; 256]>,
    pub group_size: usize,
    pub scales: Vec<f32>,
    pub biases: Vec<f32>,
}

type GroupVisitor<'a> = Box<dyn FnMut(usize, &[f32], f32, f32) + 'a>;

/// Whether a format's values are shared across groups once each group is normalised.
fn is_framed(format: &WeightFormat, cols: usize) -> bool {
    format.group_size > 0 && cols >= format.group_size as usize && matches!(format.coding, WeightCoding::Affine | WeightCoding::Float)
}

/// The scale and bias that `quantize` used for one group, from its original values.
fn group_frame(format: &WeightFormat, group: &[f32]) -> (f32, f32) {
    let (min, max) = group.iter().fold((f32::MAX, f32::MIN), |(lo, hi), v| (lo.min(*v), hi.max(*v)));
    let largest = min.abs().max(max.abs()) as f64;
    if largest == 0.0 {
        return (1.0, 0.0);
    }
    match format.coding {
        WeightCoding::Affine if format.bias => {
            let scale = (max as f64 - min as f64) / ((1u64 << format.bits) - 1) as f64;
            if scale > 0.0 {
                (scale as f32, min)
            } else {
                (1.0, min)
            }
        }
        WeightCoding::Affine => ((largest / ((1u64 << (format.bits - 1)) - 1) as f64) as f32, 0.0),
        _ => {
            let top = binary_exponent(largest).unwrap_or(0);
            let fraction = (format.bits - 1 - format.exponent_bits) as i32;
            let gain = if format.fine_scale { power(top) * (1.0 - power(-fraction - 1)) / largest } else { 1.0 };
            ((power(top) / gain) as f32, 0.0)
        }
    }
}

/// Identifies a value exactly, snapping normalised weights to the grid of their format.
#[inline]
fn key_of(format: &WeightFormat, framed: bool, value: f32, scale: f32, bias: f32) -> i64 {
    if !framed {
        return value.to_bits() as i64;
    }
    let normal = (value - bias) as f64 / scale as f64;
    if format.coding == WeightCoding::Affine {
        return normal.round() as i64 * (1i64 << KEY_FRACTION);
    }
    let Some(exponent) = binary_exponent(normal) else { return 0 };
    let at = exponent - 1 - (format.bits - 1 - format.exponent_bits) as i32;
    ((normal / power(at)).round() * power(at + KEY_FRACTION)) as i64
}

/// Rewrites a whole tensor, or returns `None` when it needs more than 256 distinct codes.
pub fn recode(format: &WeightFormat, reference: &Reference, rows: usize, cols: usize, decode: impl Fn(usize, &mut [f32]) + Sync) -> Option<Recoded> {
    let framed = is_framed(format, cols);
    let group_size = if framed { format.group_size as usize } else { cols };
    let groups = cols.div_ceil(group_size);
    let (scale_len, bias_len) = (rows * groups, rows * groups);

    let visit = |index: usize, original: &mut Vec<f32>, row: &mut Vec<f32>, mut each: GroupVisitor<'_>| {
        decode(index, original);
        row.copy_from_slice(original);
        quantize(format, reference, row);
        for (g, (source, values)) in original.chunks(group_size).zip(row.chunks(group_size)).enumerate() {
            let (scale, bias) = if framed { group_frame(format, source) } else { (1.0, 0.0) };
            each(g, values, scale, bias);
        }
    };

    let keys: BTreeSet<i64> = (0..rows)
        .into_par_iter()
        .with_min_len(16)
        .fold(
            || (BTreeSet::new(), vec![0f32; cols], vec![0f32; cols]),
            |(mut keys, mut original, mut row), index| {
                visit(index, &mut original, &mut row, Box::new(|_, values, scale, bias| keys.extend(values.iter().map(|v| key_of(format, framed, *v, scale, bias)))));
                (keys, original, row)
            },
        )
        .map(|(keys, _, _)| keys)
        .reduce(BTreeSet::new, |mut a, b| {
            a.extend(b);
            a
        });
    if keys.len() > 256 {
        return None;
    }
    let keys: Vec<i64> = keys.into_iter().collect();

    let mut table = Box::new([0f32; 256]);
    for (entry, key) in table.iter_mut().zip(&keys) {
        *entry = if framed { (*key as f64 / power(KEY_FRACTION)) as f32 } else { f32::from_bits(*key as u32) };
    }

    let mut codes = vec![0u8; rows * cols];
    let mut scales = vec![1f32; scale_len];
    let mut biases = vec![0f32; bias_len];
    codes.par_chunks_mut(cols).zip(scales.par_chunks_mut(groups)).zip(biases.par_chunks_mut(groups)).enumerate().with_min_len(16).for_each_init(
        || (vec![0f32; cols], vec![0f32; cols]),
        |(original, row), (index, ((codes, scales), biases))| {
            let mut at = 0;
            visit(
                index,
                original,
                row,
                Box::new(|g, values, scale, bias| {
                    scales[g] = scale;
                    biases[g] = bias;
                    for value in values {
                        codes[at] = keys.partition_point(|k| *k < key_of(format, framed, *value, scale, bias)) as u8;
                        at += 1;
                    }
                }),
            );
        },
    );
    Some(Recoded { codes, table, group_size, scales, biases })
}

impl Recoded {
    pub fn groups(&self, cols: usize) -> usize {
        cols.div_ceil(self.group_size)
    }

    pub fn decode_into(&self, index: usize, out: &mut [f32]) {
        let cols = out.len();
        let groups = self.groups(cols);
        let codes = &self.codes[index * cols..(index + 1) * cols];
        for (g, (values, group)) in out.chunks_mut(self.group_size).zip(codes.chunks(self.group_size)).enumerate() {
            let (scale, bias) = (self.scales[index * groups + g], self.biases[index * groups + g]);
            for (value, code) in values.iter_mut().zip(group) {
                *value = self.table[*code as usize] * scale + bias;
            }
        }
    }

    /// `Σ w · x` over one row. `sums` holds `Σ x` per group.
    #[inline]
    pub fn dot(&self, index: usize, x: &[f32], sums: &[f32]) -> f32 {
        let cols = x.len();
        let groups = sums.len();
        let codes = &self.codes[index * cols..(index + 1) * cols];
        let mut total = 0f32;
        for (g, (group, xs)) in codes.chunks(self.group_size).zip(x.chunks(self.group_size)).enumerate() {
            let dot: f32 = group.iter().zip(xs).map(|(code, v)| self.table[*code as usize] * v).sum();
            total += dot * self.scales[index * groups + g] + self.biases[index * groups + g] * sums[g];
        }
        total
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn format(coding: WeightCoding, bits: u32, exponent_bits: u32) -> WeightFormat {
        WeightFormat { coding, bits, exponent_bits, bias: true, group_size: 0, fine_scale: false, keep_source: false }
    }

    fn reference(values: &[f32]) -> Reference {
        let (min, max) = values.iter().fold((f32::MAX, f32::MIN), |(a, b), v| (a.min(*v), b.max(*v)));
        Reference { min, max, table: train_table(values.to_vec(), 16) }
    }

    fn rms(a: &[f32], b: &[f32]) -> f32 {
        (a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f32>() / a.len() as f32).sqrt()
    }

    fn sample() -> Vec<f32> {
        (0..4096).map(|i| { let t = i as f32 * 0.137; t.sin() * (t * 0.31).cos() * 0.8 * (1.0 + (t * 0.07).sin()) / 2.0 }).collect()
    }

    #[test]
    fn more_bits_mean_less_error_for_every_q() {
        let values = sample();
        let reference = reference(&values);
        let mut last = f32::MAX;
        for bits in 2..=16 {
            let mut copy = values.clone();
            quantize(&format(WeightCoding::Fixed, bits, 0), &reference, &mut copy);
            let error = rms(&values, &copy);
            assert!(error < last, "Q{bits}: {error} >= {last}");
            last = error;
        }
        assert!(last < 1e-4);
    }

    #[test]
    fn q_codes_land_on_the_grid() {
        let values = sample();
        let reference = reference(&values);
        let mut copy = values.clone();
        quantize(&format(WeightCoding::Fixed, 4, 0), &reference, &mut copy);
        let step = reference.largest() / 7.0;
        for v in copy {
            assert!(((v / step).round() - v / step).abs() < 1e-4, "{v}");
        }
    }

    #[test]
    fn floats_keep_relative_precision() {
        let values: Vec<f32> = vec![1.0, 0.5, 0.123456, 0.001234, -0.0000456, 3.0e-5, 0.75];
        let reference = reference(&values);
        for (bits, exponent, tolerance) in [(16, 5, 1e-3), (32, 8, 1e-6)] {
            let mut copy = values.clone();
            quantize(&format(WeightCoding::Float, bits, exponent), &reference, &mut copy);
            for (a, b) in values.iter().zip(&copy) {
                assert!(((a - b) / a).abs() < tolerance, "{a} {b}");
            }
        }
    }

    #[test]
    fn small_floats_flush_only_outside_their_range() {
        let reference = Reference { min: 0.0, max: 1.0, table: Vec::new() };
        let mut values = vec![1.0f32, 0.5, 1e-9];
        quantize(&format(WeightCoding::Float, 32, 8), &reference, &mut values);
        assert_eq!(values[2], 0.0, "outside the accumulator's window");
    }

    #[test]
    fn posits_are_most_accurate_near_their_scale() {
        let reference = Reference { min: -1.0, max: 1.0, table: Vec::new() };
        let mut values = vec![0.7f32, 0.0007];
        quantize(&format(WeightCoding::Posit, 8, 1), &reference, &mut values);
        let big = (values[0] - 0.7).abs() / 0.7;
        let small = (values[1] - 0.0007).abs() / 0.0007;
        assert!(big < small, "{big} {small}");
    }

    #[test]
    fn codebook_reduces_error_compared_with_a_uniform_grid() {
        let values = sample();
        let reference = reference(&values);
        let (mut book, mut grid) = (values.clone(), values.clone());
        quantize(&format(WeightCoding::Codebook, 4, 0), &reference, &mut book);
        quantize(&format(WeightCoding::Fixed, 4, 0), &reference, &mut grid);
        assert!(rms(&values, &book) < rms(&values, &grid));
    }

    #[test]
    fn a_fine_scale_beats_a_power_of_two() {
        let values: Vec<f32> = (0..64).map(|i| 0.9 * (i as f32 * 0.7).sin() * if i % 2 == 0 { 0.8 } else { 0.3 }).collect();
        let reference = reference(&values);
        let mut coarse = format(WeightCoding::Float, 4, 2);
        coarse.group_size = 16;
        let mut fine = coarse;
        fine.fine_scale = true;
        let (mut a, mut b) = (values.clone(), values.clone());
        quantize(&coarse, &reference, &mut a);
        quantize(&fine, &reference, &mut b);
        assert!(rms(&values, &b) < rms(&values, &a), "{} {}", rms(&values, &b), rms(&values, &a));
    }

    #[test]
    fn groups_get_their_own_range() {
        let reference = Reference { min: -1.0, max: 1.0, table: Vec::new() };
        let mut f = format(WeightCoding::Affine, 2, 0);
        f.group_size = 2;
        let mut values = vec![1.0, -1.0, 0.01, -0.01];
        quantize(&f, &reference, &mut values);
        assert_eq!(values, vec![1.0, -1.0, 0.01, -0.01]);
    }

    fn wavy_weights(rows: usize, cols: usize) -> Vec<f32> {
        (0..rows * cols).map(|i| ((i as f32 * 0.37).sin() * (1.0 + (i % 11) as f32)) * 0.01).collect()
    }

    #[test]
    fn recoding_keeps_the_values_of_every_narrow_format() {
        let (rows, cols) = (20, 64);
        let values = wavy_weights(rows, cols);
        let formats = [
            format(WeightCoding::Fixed, 6, 0),
            format(WeightCoding::Float, 8, 4),
            format(WeightCoding::Posit, 8, 1),
            format(WeightCoding::Log, 8, 4),
            WeightFormat { group_size: 16, ..format(WeightCoding::Affine, 4, 0) },
            WeightFormat { group_size: 16, bias: false, ..format(WeightCoding::Affine, 5, 0) },
            WeightFormat { group_size: 16, bias: false, ..format(WeightCoding::Float, 4, 2) },
            WeightFormat { group_size: 16, bias: false, fine_scale: true, ..format(WeightCoding::Float, 4, 2) },
        ];
        for format in formats {
            let (min, max) = values.iter().fold((f32::MAX, f32::MIN), |(lo, hi), v| (lo.min(*v), hi.max(*v)));
            let reference = Reference { min, max, table: Vec::new() };
            let recoded = recode(&format, &reference, rows, cols, |index, out| out.copy_from_slice(&values[index * cols..(index + 1) * cols])).expect("fits in 256 codes");
            let mut decoded = vec![0f32; cols];
            for index in 0..rows {
                let mut expected = values[index * cols..(index + 1) * cols].to_vec();
                quantize(&format, &reference, &mut expected);
                recoded.decode_into(index, &mut decoded);
                for (a, b) in expected.iter().zip(&decoded) {
                    assert!((a - b).abs() <= 1e-5 * a.abs().max(1e-3), "{format:?}: {a} frente a {b}");
                }
            }
        }
    }
}
