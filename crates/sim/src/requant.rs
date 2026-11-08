//! Rewrites a checkpoint's weights in the format the chip stores, so a simulation
//! sees exactly the precision the hardware would have.

use netlist::format::{WeightCoding, WeightFormat, FLOAT_WINDOW};

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
    let mut table: Vec<f32> = (0..entries).map(|i| sample[((i * 2 + 1) * sample.len() / (2 * entries)).min(sample.len() - 1)]).collect();
    for _ in 0..4 {
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
        assert_eq!(values[2], 0.0, "fuera de la ventana del acumulador");
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
}
