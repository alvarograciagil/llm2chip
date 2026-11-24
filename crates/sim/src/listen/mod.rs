//! Listening: speech to text. Moonshine in floating point (`moonshine`), on the die with its
//! schedule (`moonshine_chip`), and its streaming model, which listens while the question is
//! spoken (`moonshine_stream_chip`). Built with the `listen` feature.

pub mod moonshine;
pub mod moonshine_chip;
pub mod moonshine_stream_chip;
