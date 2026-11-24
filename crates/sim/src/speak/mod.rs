//! Speaking: text to speech. OuteTTS's prompt, codes and sampling (`outetts`), and DAC's
//! decoder from codes to sound, in floating point (`dac`) and on the die with its schedule
//! (`dac_chip`). Built with the `speak` feature.

pub mod dac;
pub mod dac_chip;
pub mod outetts;
