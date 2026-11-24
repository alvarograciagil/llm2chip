//! Executes a compiled microprogram on real weights.

pub mod accel;
pub mod audio;
pub mod chip;
pub mod dac;
pub mod dac_chip;
mod exact;
mod exec;
pub mod image;
pub mod janus;
pub mod lower;
pub mod magvit_chip;
pub mod moonshine;
pub mod moonshine_chip;
pub mod optimize;
pub mod prompt;
pub mod showo;
pub mod showo_chip;
pub mod sizing;
pub mod system;
pub mod tokenizer;
pub mod vision;
pub mod vision_chip;
mod requant;
mod vm;
mod weights;

pub use exact::ExactEngine;
pub use exec::{Engine, Numeric, StepStats};
pub use vm::BlockVm;
pub use weights::{SamplerConfig, Weights};

/// What the server needs from any execution engine.
pub trait Runner: Send {
    fn reset(&mut self);
    /// The length of the logits vector.
    fn vocabulary(&self) -> usize;
    fn step(&mut self, token: u32, want_logits: bool) -> Result<(Option<&[f32]>, StepStats), String>;
    fn has_draft_head(&self) -> bool;
    fn draft_tokens(&self) -> usize;
    fn truncate_draft(&mut self, tokens: usize);
    fn feed_draft_from_main(&mut self);
    fn feed_draft_from_draft(&mut self);
    fn draft(&mut self, token: u32, want_logits: bool) -> Result<Option<&[f32]>, String>;
    /// Runs a vector in place of a token's embedding, such as an image's, at its time,
    /// row and column positions.
    fn step_embedding(&mut self, _embedding: &[f32], _positions: [u64; 3], _want_logits: bool) -> Result<(Option<&[f32]>, StepStats), String> {
        Err("this inference core does not take images yet: choose the floating-point one".into())
    }
    /// Runs a picture through the image encoder the engine carries, if it carries one, and
    /// keeps it there: gives how many tokens it makes, which `step_image` reads one by one.
    fn encode_image(&mut self, _tower: &vision::VisionTower, _image: &vision::Patches) -> Result<Option<usize>, String> {
        Ok(None)
    }
    /// The most patches the engine's own encoder takes.
    fn image_patches(&self) -> Option<usize> {
        None
    }
    /// Token `index` of the image `encode_image` left, at its time, row and column.
    fn step_image(&mut self, _index: u32, _positions: [u64; 3], _want_logits: bool) -> Result<(Option<&[f32]>, StepStats), String> {
        Err("this inference core has no image encoder".into())
    }
    /// Rotary frequencies follow three positions, `section` of them each, interleaved.
    fn set_mrope(&mut self, _section: [usize; 3]) {}
    /// Where the text after an image goes on from.
    fn continue_text_at(&mut self, _position: u64) {}
}

impl Runner for Engine {
    fn reset(&mut self) {
        Engine::reset(self)
    }
    fn step_embedding(&mut self, embedding: &[f32], positions: [u64; 3], want_logits: bool) -> Result<(Option<&[f32]>, StepStats), String> {
        Engine::step_embedding(self, embedding, positions, want_logits)
    }
    fn set_mrope(&mut self, section: [usize; 3]) {
        Engine::set_mrope(self, section)
    }
    fn continue_text_at(&mut self, position: u64) {
        Engine::continue_text_at(self, position)
    }
    fn vocabulary(&self) -> usize {
        Engine::vocabulary(self)
    }
    fn step(&mut self, token: u32, want_logits: bool) -> Result<(Option<&[f32]>, StepStats), String> {
        Engine::step(self, token, want_logits)
    }
    fn has_draft_head(&self) -> bool {
        Engine::has_draft_head(self)
    }
    fn draft_tokens(&self) -> usize {
        Engine::draft_tokens(self)
    }
    fn truncate_draft(&mut self, tokens: usize) {
        Engine::truncate_draft(self, tokens)
    }
    fn feed_draft_from_main(&mut self) {
        Engine::feed_draft_from_main(self)
    }
    fn feed_draft_from_draft(&mut self) {
        Engine::feed_draft_from_draft(self)
    }
    fn draft(&mut self, token: u32, want_logits: bool) -> Result<Option<&[f32]>, String> {
        Engine::draft(self, token, want_logits)
    }
}

impl Runner for ExactEngine {
    fn reset(&mut self) {
        ExactEngine::reset(self)
    }
    fn set_mrope(&mut self, section: [usize; 3]) {
        ExactEngine::set_mrope(self, section)
    }
    fn continue_text_at(&mut self, position: u64) {
        ExactEngine::continue_text_at(self, position)
    }
    fn vocabulary(&self) -> usize {
        ExactEngine::vocabulary(self)
    }
    fn step(&mut self, token: u32, want_logits: bool) -> Result<(Option<&[f32]>, StepStats), String> {
        let started = web_time::Instant::now();
        self.logits = ExactEngine::step(self, token, want_logits)?.unwrap_or_default();
        let stats = StepStats { seconds: started.elapsed().as_secs_f64(), saturated: 0 };
        Ok((want_logits.then_some(self.logits.as_slice()), stats))
    }
    fn has_draft_head(&self) -> bool {
        ExactEngine::has_draft_head(self)
    }
    fn draft_tokens(&self) -> usize {
        ExactEngine::draft_tokens(self)
    }
    fn truncate_draft(&mut self, tokens: usize) {
        ExactEngine::truncate_draft(self, tokens)
    }
    fn feed_draft_from_main(&mut self) {
        ExactEngine::feed_draft_from_main(self)
    }
    fn feed_draft_from_draft(&mut self) {
        ExactEngine::feed_draft_from_draft(self)
    }
    fn draft(&mut self, token: u32, want_logits: bool) -> Result<Option<&[f32]>, String> {
        self.logits = ExactEngine::draft(self, token, want_logits)?.unwrap_or_default();
        Ok(want_logits.then_some(self.logits.as_slice()))
    }
    fn encode_image(&mut self, tower: &vision::VisionTower, image: &vision::Patches) -> Result<Option<usize>, String> {
        if self.image_patches().is_none() {
            return Ok(None);
        }
        ExactEngine::encode_image(self, tower, image).map(Some)
    }
    fn image_patches(&self) -> Option<usize> {
        ExactEngine::image_patches(self)
    }
    fn step_image(&mut self, index: u32, positions: [u64; 3], want_logits: bool) -> Result<(Option<&[f32]>, StepStats), String> {
        let started = web_time::Instant::now();
        self.logits = ExactEngine::step_image(self, index, positions, want_logits)?.unwrap_or_default();
        let stats = StepStats { seconds: started.elapsed().as_secs_f64(), saturated: 0 };
        Ok((want_logits.then_some(self.logits.as_slice()), stats))
    }
    fn step_embedding(&mut self, embedding: &[f32], positions: [u64; 3], want_logits: bool) -> Result<(Option<&[f32]>, StepStats), String> {
        let started = web_time::Instant::now();
        self.logits = ExactEngine::step_embedding(self, embedding, positions, want_logits)?.unwrap_or_default();
        let stats = StepStats { seconds: started.elapsed().as_secs_f64(), saturated: 0 };
        Ok((want_logits.then_some(self.logits.as_slice()), stats))
    }
}

/// Picks the next token: the likeliest at temperature 0, otherwise one of the `top_k`
/// likeliest by their softmax at `temperature`, drawn with `seed` (xorshift).
pub fn sample(logits: &[f32], temperature: f32, top_k: usize, seed: &mut u64) -> u32 {
    if temperature <= 0.0 {
        return logits.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map_or(0, |(i, _)| i as u32);
    }
    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    let k = top_k.clamp(1, indexed.len());
    indexed.select_nth_unstable_by(k - 1, |a, b| b.1.total_cmp(&a.1));
    indexed.truncate(k);
    let top = indexed.iter().map(|(_, v)| *v).fold(f32::MIN, f32::max);
    let weights: Vec<f64> = indexed.iter().map(|(_, v)| (((v - top) / temperature) as f64).exp()).collect();
    let total: f64 = weights.iter().sum();
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    let mut point = (*seed as f64 / u64::MAX as f64) * total;
    for ((i, _), w) in indexed.iter().zip(&weights) {
        point -= w;
        if point <= 0.0 {
            return *i as u32;
        }
    }
    indexed[0].0 as u32
}
