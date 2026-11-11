//! Executes a compiled microprogram on real weights.

pub mod accel;
pub mod chip;
mod exact;
mod exec;
pub mod image;
pub mod janus;
pub mod lower;
pub mod sizing;
pub mod tokenizer;
pub mod vision;
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
        Err("este núcleo de inferencia aún no admite imágenes: elige el de coma flotante".into())
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
    fn vocabulary(&self) -> usize {
        ExactEngine::vocabulary(self)
    }
    fn step(&mut self, token: u32, want_logits: bool) -> Result<(Option<&[f32]>, StepStats), String> {
        let started = std::time::Instant::now();
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
    fn step_embedding(&mut self, embedding: &[f32], positions: [u64; 3], want_logits: bool) -> Result<(Option<&[f32]>, StepStats), String> {
        let started = std::time::Instant::now();
        self.logits = ExactEngine::step_embedding(self, embedding, positions, want_logits)?.unwrap_or_default();
        let stats = StepStats { seconds: started.elapsed().as_secs_f64(), saturated: 0 };
        Ok((want_logits.then_some(self.logits.as_slice()), stats))
    }
}
