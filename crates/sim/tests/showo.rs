//! Show-o's image tokenizer, MAGVIT-v2, on the real weights: a picture encoded and decoded
//! comes back close. The model is not in the repository: it is taken from `LLM2CHIP_SHOWO`
//! or `~/.llm2chip/models/Show-o`.

use sim::showo::ShowO;
use std::path::PathBuf;

#[test]
#[ignore = "slow: opens Show-o (6 GB) and passes an image through MAGVIT-v2"]
fn a_picture_comes_back_through_the_image_tokenizer() {
    let dir = std::env::var_os("LLM2CHIP_SHOWO").map(PathBuf::from).unwrap_or_else(|| PathBuf::from(std::env::var_os("HOME").unwrap_or_default()).join(".llm2chip/models/Show-o"));
    let Some(model) = (if dir.join("config.json").exists() { ShowO::open(&dir).unwrap() } else { None }) else {
        eprintln!("skipped: Show-o is not in {}", dir.display());
        return;
    };
    let side = model.config.side();
    // A red disc on white, and a gradient across it.
    let rgb: Vec<u8> = (0..side * side)
        .flat_map(|i| {
            let (x, y) = ((i % side) as f32, (i / side) as f32);
            let (dx, dy) = (x - side as f32 / 2.0, y - side as f32 / 2.0);
            if dx * dx + dy * dy < (side as f32 * 0.3).powi(2) { [220, 30, (x / side as f32 * 200.0) as u8] } else { [255, 255, 255] }
        })
        .collect();
    let planes: Vec<f32> = (0..3).flat_map(|c| (0..side * side).map(move |i| (c, i))).map(|(c, i)| rgb[i * 3 + c] as f32 / 127.5 - 1.0).collect();
    let codes = model.tokenizer().unwrap().encode(&planes).unwrap();
    assert_eq!(codes.len(), model.config.image_tokens);
    assert!(codes.iter().all(|c| *c < model.config.codebook));
    let back = model.tokenizer().unwrap().decode(&codes).unwrap();
    let error: f64 = rgb.iter().zip(&back).map(|(a, b)| (*a as f64 - *b as f64).powi(2)).sum::<f64>() / rgb.len() as f64;
    let psnr = 10.0 * (255.0f64 * 255.0 / error).log10();
    eprintln!("PSNR {psnr:.1} dB");
    assert!(psnr > 25.0, "PSNR {psnr:.1} dB");
}
