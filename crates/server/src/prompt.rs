//! How a checkpoint wants its text framed: the rules live in `sim`, shared with the browser.

use std::path::Path;
use tokenizers::Tokenizer;

pub use sim::prompt::{PromptStyle, ReasoningBudget};

/// The framing of the checkpoint in `dir`, read from its files.
pub fn read(dir: &Path, family: &str, tokenizer: &Tokenizer) -> PromptStyle {
    PromptStyle::read(family, |name| std::fs::read_to_string(dir.join(name)).ok(), |token| tokenizer.token_to_id(token))
}
