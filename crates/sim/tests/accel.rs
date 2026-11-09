//! The GPU must agree with the CPU: closely in floating point, and bit for bit in integers.

use half::f16;
use sim::Weights;
use std::path::{Path, PathBuf};

const ROWS: usize = 1100;
const COLS: usize = 1024;
const GROUPS: usize = 32;

struct Noise(u64);

impl Noise {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }

    fn unit(&mut self) -> f32 {
        self.next() as f32 / (1u32 << 31) as f32
    }
}

/// `gap` shifts every tensor in the file: real checkpoints do not align them.
fn write_checkpoint(dir: &Path, gap: usize) {
    let mut noise = Noise(7);
    let mut data = vec![0u8; gap];
    let mut header = serde_json::Map::new();
    let mut add = |name: &str, dtype: &str, shape: Vec<usize>, bytes: Vec<u8>, data: &mut Vec<u8>| {
        let start = data.len();
        data.extend_from_slice(&bytes);
        header.insert(name.to_string(), serde_json::json!({ "dtype": dtype, "shape": shape, "data_offsets": [start, data.len()] }));
    };
    for (name, bits) in [("four", 4), ("eight", 8)] {
        let words_per_row = COLS * bits / 32;
        let words: Vec<u8> = (0..ROWS * words_per_row).flat_map(|_| noise.next().to_le_bytes()).collect();
        let scales: Vec<u8> = (0..ROWS * GROUPS).flat_map(|_| f16::from_f32((1.0 + noise.unit()) * 1e-3).to_le_bytes()).collect();
        let biases: Vec<u8> = (0..ROWS * GROUPS).flat_map(|_| f16::from_f32((noise.unit() - 0.5) * 0.04).to_le_bytes()).collect();
        add(&format!("{name}.weight"), "U32", vec![ROWS, words_per_row], words, &mut data);
        add(&format!("{name}.scales"), "F16", vec![ROWS, GROUPS], scales, &mut data);
        add(&format!("{name}.biases"), "F16", vec![ROWS, GROUPS], biases, &mut data);
    }
    let mut text = serde_json::to_vec(&header).unwrap();
    while !(8 + text.len()).is_multiple_of(8) {
        text.push(b' ');
    }
    let mut file = (text.len() as u64).to_le_bytes().to_vec();
    file.extend_from_slice(&text);
    file.extend_from_slice(&data);
    std::fs::write(dir.join("model.safetensors"), file).unwrap();
}

fn checkpoint(gap: usize) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("llm2chip-accel-{gap}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    write_checkpoint(&dir, gap);
    dir
}

#[test]
fn gpu_matches_cpu_at_any_alignment() {
    for gap in [0, 1, 2, 3] {
        compare(gap);
    }
}

fn compare(gap: usize) {
    let dir = checkpoint(gap);
    let cpu = Weights::open(&dir).unwrap();
    let gpu = Weights::open(&dir).unwrap().with_gpu();
    let Some(name) = gpu.accelerator_name() else {
        eprintln!("sin GPU: prueba omitida");
        return;
    };
    eprintln!("GPU: {name}, desplazamiento {gap}");
    let mut noise = Noise(99);
    for matrix in ["four", "eight"] {
        let (a, b) = (cpu.matrix(matrix).unwrap(), gpu.matrix(matrix).unwrap());

        let x: Vec<f32> = (0..COLS).map(|_| noise.unit() * 2.0 - 1.0).collect();
        for first in [0, 40] {
            let (mut slow, mut fast) = (vec![0f32; ROWS - first], vec![0f32; ROWS - first]);
            a.apply_rows(first, &x, &mut slow);
            b.apply_rows(first, &x, &mut fast);
            for (s, f) in slow.iter().zip(&fast) {
                assert!((s - f).abs() <= 1e-3 * s.abs().max(1.0), "{matrix}: {s} frente a {f}");
            }
        }

        let exponent = a.exponent(COLS);
        for bits in [8, 12, 16, 20, 24] {
            let limit = 1i64 << (bits - 1);
            let x: Vec<i32> = (0..COLS).map(|_| ((noise.unit() * 2.0 - 1.0) * (limit - 1) as f32) as i32).collect();
            for first in [0, 40] {
                let slow = a.exact_sums(first, ROWS - first, &x, exponent, bits);
                let fast = b.exact_sums(first, ROWS - first, &x, exponent, bits);
                assert_eq!(slow, fast, "{matrix} a {bits} bits, desde la fila {first}");
            }
        }
    }
    assert_eq!(gpu.gpu_products(), 2 * (2 + 5 * 2), "la GPU debe ejecutar todos los productos");
    std::fs::remove_dir_all(dir).ok();
}
