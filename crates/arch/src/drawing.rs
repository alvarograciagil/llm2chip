//! The die of drawing: a model that draws and retouches pictures from a text (the Images
//! view), on a die of its own beside the language model's.
//!
//! The reference die is Show-o's transformer (1,300 M parameters, `showlab/show-o`), guided
//! in the die, on the die its firmware addresses: 8-bit affine weights in groups of 64,
//! 16 units of 64 lanes and 4,096 cells, measured in N6 by `sim`'s `draw_dies` example.
//! Its embedding table stays in external memory and the loader brings a token's row at a
//! time. MAGVIT-v2, which turns codes into pixels and back, keeps whole maps of the picture
//! in the vector memory, more than a reticle, so it runs on the host.

use crate::side::{die, CountedDie, Part, SideDie};

/// The stage's name, as the floorplan shows it.
pub const DRAW_STAGES: [&str; 1] = ["image drawer"];

/// Bytes of the reference drawer's embedding table, which stays in external memory: 119.8 M
/// elements at the die's 16 bits.
pub const OUTSIDE_TABLE_BYTES: f64 = 119_803_904.0 * 2.0;

/// The reference die, measured with Show-o.
pub fn reference_dies() -> Vec<SideDie> {
    vec![die(&DRAW_STAGES, 16, 4096, [6.285, 314.376, 109.599, 1.83], 1000.0)]
}

/// The dies a project that draws counts: the reference die until it is planned with its own
/// model.
pub fn counted(draws: bool) -> Vec<CountedDie> {
    if !draws {
        return Vec::new();
    }
    reference_dies().into_iter().map(|die| CountedDie { die, parts: vec![Part::Draw], reference: true }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_chip_that_draws_counts_show_o_s_die() {
        assert!(counted(false).is_empty());
        let dies = counted(true);
        assert_eq!(dies.len(), 1);
        assert_eq!(dies[0].parts, [Part::Draw]);
        assert!(dies[0].reference);
        assert!((dies[0].mm2() - 432.1).abs() < 0.5, "{}", dies[0].mm2());
        let leak = crate::side::leakage_w(&dies[0]);
        assert!(leak > 0.1 && leak < 20.0, "{leak}");
    }
}
