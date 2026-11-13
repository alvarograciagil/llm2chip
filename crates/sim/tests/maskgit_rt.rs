//! The integer MaskGIT step the firmware runs, against floating point.

use maskgit_rt::{cut, draw, masked, Rng, FRACTION};

#[test]
fn a_draw_follows_the_softmax_and_its_confidence_is_the_log_probability() {
    // Mantissas of 16 bits with exponent 3: scores of a few units.
    let mantissas: Vec<i32> = vec![12000, -3000, 20000, 19000, 0, -20000, 5000, 16000];
    let (exponent, bits) = (3, 16);
    let value = |m: i32| m as f64 / 32768.0 * 8.0;
    let top = mantissas.iter().map(|m| value(*m)).fold(f64::MIN, f64::max);
    let weights: Vec<f64> = mantissas.iter().map(|m| (value(*m) - top).exp()).collect();
    let total: f64 = weights.iter().sum();
    let probabilities: Vec<f64> = weights.iter().map(|w| w / total).collect();
    let score = |c: u32| mantissas[c as usize];
    let mut rng = Rng(12345);
    let mut counts = [0u32; 8];
    let draws = 40_000;
    for _ in 0..draws {
        let (code, confidence) = draw(8, &score, exponent, bits, 0, &mut rng);
        counts[code as usize] += 1;
        let expected = probabilities[code as usize].ln();
        let got = confidence as f64 / (1 << FRACTION) as f64;
        assert!((got - expected).abs() < 0.02, "código {code}: {got} frente a {expected}");
    }
    for (c, p) in probabilities.iter().enumerate() {
        let seen = counts[c] as f64 / draws as f64;
        assert!((seen - p).abs() < 0.01, "código {c}: {seen} frente a {p}");
    }
}

#[test]
fn the_noise_spreads_the_confidences_and_the_least_sure_stay_masked() {
    let score = |c: u32| [100, 200][c as usize];
    let mut rng = Rng(7);
    let quiet: Vec<i64> = (0..50).map(|_| draw(2, &score, 0, 16, 0, &mut rng).1).collect();
    let noisy: Vec<i64> = (0..50).map(|_| draw(2, &score, 0, 16, 1 << FRACTION, &mut rng).1).collect();
    let spread = |v: &[i64]| v.iter().max().unwrap() - v.iter().min().unwrap();
    assert!(spread(&noisy) > 4 * spread(&quiet).max(1));
    assert_eq!(masked(100, 10), 9);
    assert_eq!(masked(0, 10), 1);
    let confidences = [5, 1, 9, 3, 7];
    assert_eq!(cut(&confidences, 2), 5);
    assert_eq!(confidences.iter().filter(|c| **c < cut(&confidences, 2)).count(), 2);
}
