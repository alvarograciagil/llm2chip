//! Runs a compiled model with the die's exact integer arithmetic.

use crate::lower::Layout;
use crate::vm::BlockVm;
use crate::weights::Weights;
use compiler::Program;
use netlist::block::exponent_register;
use netlist::{CellConfig, DieConfig};
use std::sync::Arc;

/// Large register space and 12-bit tables: real models are not limited by a test die.
pub(crate) fn exact_config(bits: u32) -> DieConfig {
    DieConfig {
        cells: 1,
        cell: CellConfig { weight_bits: 8, activation_bits: bits, accumulator_bits: 32, mac_stages: 1, adder_stages: 1, addr_bits: 3, coding: Default::default(), exponent_bits: 0, multiplier: netlist::multiplier::MultiplierKind::Array, fixed_weights: false, ecc: false, skip_zeros: false },
        scale_bits: 16,
        reg_bits: 24,
        index_bits: 24,
        microcode_bits: 18,
        broadcast_stages: 1,
        function_bits: 12,
        function_select_bits: 3, features: Default::default(),
    }
}

pub struct ExactEngine {
    program: Program,
    layout: Layout,
    bits: u32,
    vm: BlockVm,
    position: usize,
    draft_tokens: usize,
    exponents: u32,
    pub(crate) logits: Vec<f32>,
    /// M-RoPE's sections, for models whose images take three positions.
    mrope: Option<[usize; 3]>,
    /// How far the text's rotary positions run ahead of the cache's, after an image.
    rope_offset: i64,
}

impl ExactEngine {
    /// `bits` is the mantissa width of the die's vectors.
    pub fn new(program: Program, weights: Weights, bits: u32) -> Result<Self, String> {
        if !(8..=24).contains(&bits) {
            return Err("vectors take from 8 to 24 bits".into());
        }
        let config = exact_config(bits);
        let weights = Arc::new(weights);
        let layout = Layout::new(&program, &weights, config, None)?;
        let exponents = exponent_register(&config);
        let mut vm = BlockVm::new(&config, layout.registers_used() as usize + 8, Some(weights));
        vm.compute_in_cells();
        let mut engine = Self { program, layout, bits, vm, position: 0, draft_tokens: 0, exponents, logits: Vec::new(), mrope: None, rope_offset: 0 };
        engine.reset();
        Ok(engine)
    }

    /// Adds up matrix products weight by weight, as a die that holds them in registers.
    pub fn with_weights_in_registers(mut self) -> Self {
        self.vm.compute_in_registers();
        self
    }

    pub fn reset(&mut self) {
        self.vm.clear();
        crate::chip::load_initial(&mut self.vm, &self.layout.setup().initial, self.exponents);
        self.position = 0;
        self.draft_tokens = 0;
        self.rope_offset = 0;
        self.layout.clear_rotations();
    }

    /// Rotary frequencies follow three positions, `section` of them each, interleaved.
    pub fn set_mrope(&mut self, section: [usize; 3]) {
        self.mrope = Some(section);
    }

    /// Where the text after an image goes on from: its next token takes `position`.
    pub fn continue_text_at(&mut self, position: u64) {
        self.rope_offset = position as i64 - self.position as i64;
    }

    pub fn vocabulary(&self) -> usize {
        self.program.vocab
    }

    pub fn register_of(&self, buffer: usize) -> u32 {
        self.layout.register_of(buffer)
    }

    pub fn buffer(&self, buffer: usize) -> Vec<f32> {
        let reg = self.layout.register_of(buffer);
        self.vm.floats(reg, self.program.buffers[buffer])
    }

    pub fn mantissa(&self, reg: u32, index: u32) -> i64 {
        self.vm.mantissa(reg, index)
    }

    pub fn position(&self) -> usize {
        self.position
    }

    /// A vector in place of a token's embedding, such as a picture's: the chip loads it,
    /// in its own format, where the token's row would go, in the cache's next position, and
    /// rotates it by its own positions: three under M-RoPE.
    pub fn step_embedding(&mut self, embedding: &[f32], positions: [u64; 3], want_logits: bool) -> Result<Option<Vec<f32>>, String> {
        self.layout.place_rotation(&self.program, self.position as u32, positions, self.mrope)?;
        let values: Vec<f64> = embedding.iter().map(|v| *v as f64).collect();
        let (mantissas, exponent) = netlist::block::encode_vector(&values, self.bits);
        self.vm.feed_row(mantissas, exponent);
        self.layout.run_token(&mut self.vm, &self.program, 0, self.position as u32, want_logits)?;
        self.position += 1;
        Ok(want_logits.then(|| self.vm.floats(self.layout.logits_register(&self.program), self.program.vocab)))
    }

    /// The most patches of a picture the encoder the chip carries takes, if it carries one.
    pub fn image_patches(&self) -> Option<usize> {
        self.program.rows().map(|part| part.patches)
    }

    /// Runs a picture through the model's own image encoder, which the chip carries with it
    /// (`compiler::attach_vision`), and keeps it there: gives how many tokens it makes, which
    /// `step_image` then reads one by one.
    pub fn encode_image(&mut self, tower: &crate::vision::VisionTower, image: &crate::vision::Patches) -> Result<usize, String> {
        let part = self.program.rows().cloned().ok_or("this model has no image encoder")?;
        crate::vision_chip::run_encoder(&self.layout, &mut self.vm, &self.program, &part, self.bits, tower, image)
    }

    /// Token `index` of the encoded image: the encoder's fusion makes it, and the language
    /// model reads it in place of an embedding, in the cache's next position.
    pub fn step_image(&mut self, index: u32, positions: [u64; 3], want_logits: bool) -> Result<Option<Vec<f32>>, String> {
        let part = self.program.rows().cloned().ok_or("this model has no image encoder")?;
        self.layout.run_ops(&mut self.vm, &self.program, &part.ops[part.merge.clone()], index, index, false)?;
        self.layout.place_rotation(&self.program, self.position as u32, positions, self.mrope)?;
        self.layout.run_token_from_image(&mut self.vm, &self.program, self.position as u32, want_logits)?;
        self.position += 1;
        Ok(want_logits.then(|| self.vm.floats(self.layout.logits_register(&self.program), self.program.vocab)))
    }

    /// Runs a token, or a vector in its place, up to the product that makes the logits, and
    /// gives the vector that product would take: the last hidden state, normalised. Heads
    /// other than the text's read it, such as an image generator's.
    pub fn step_hidden(&mut self, token: u32, embedding: Option<&[f32]>) -> Result<Vec<f32>, String> {
        let last = self.program.ops.len() - 1;
        let compiler::Op::Matvec { src, .. } = self.program.ops[last].op else {
            return Err("the model does not end with the logits' product".into());
        };
        let text = (self.position as i64 + self.rope_offset) as u64;
        self.layout.place_rotation(&self.program, self.position as u32, [text; 3], self.mrope)?;
        if let Some(embedding) = embedding {
            let values: Vec<f64> = embedding.iter().map(|v| *v as f64).collect();
            let (mantissas, exponent) = netlist::block::encode_vector(&values, self.bits);
            self.vm.feed_row(mantissas, exponent);
        }
        self.layout.run_until(&mut self.vm, &self.program, token, self.position as u32, last)?;
        self.position += 1;
        Ok(self.buffer(src))
    }

    /// The image head's score of each code of the codebook, from the hidden state the last
    /// `step_hidden` left, in the die's arithmetic.
    pub fn image_logits(&mut self) -> Result<Vec<f32>, String> {
        self.layout.run_image_logits(&mut self.vm, &self.program)?;
        let image = self.program.codes().ok_or("this model does not draw")?;
        Ok(self.buffer(image.logits))
    }

    /// The vector the image head makes for a drawn code, to go in as the next token's.
    pub fn image_input(&mut self, code: u32) -> Result<Vec<f32>, String> {
        self.layout.run_image_input(&mut self.vm, &self.program, code)?;
        let image = self.program.codes().ok_or("this model does not draw")?;
        Ok(self.buffer(image.input))
    }

    pub fn step(&mut self, token: u32, want_logits: bool) -> Result<Option<Vec<f32>>, String> {
        let text = (self.position as i64 + self.rope_offset) as u64;
        self.layout.place_rotation(&self.program, self.position as u32, [text; 3], self.mrope)?;
        self.layout.run_token(&mut self.vm, &self.program, token, self.position as u32, want_logits)?;
        self.position += 1;
        Ok(want_logits.then(|| self.vm.floats(self.layout.logits_register(&self.program), self.program.vocab)))
    }

    pub fn has_draft_head(&self) -> bool {
        self.program.draft.is_some()
    }

    pub fn draft_tokens(&self) -> usize {
        self.draft_tokens
    }

    pub fn truncate_draft(&mut self, tokens: usize) {
        self.draft_tokens = tokens;
    }

    pub fn feed_draft_from_main(&mut self) {
        self.layout.feed_draft(&mut self.vm, &self.program, true);
    }

    pub fn feed_draft_from_draft(&mut self) {
        self.layout.feed_draft(&mut self.vm, &self.program, false);
    }

    pub fn draft(&mut self, token: u32, want_logits: bool) -> Result<Option<Vec<f32>>, String> {
        self.layout.run_draft(&mut self.vm, &self.program, token, self.draft_tokens as u32, want_logits)?;
        self.draft_tokens += 1;
        let draft = self.program.draft.as_ref().ok_or("this model has no MTP head")?;
        Ok(want_logits.then(|| self.vm.floats(self.layout.register_of(draft.logits), self.program.vocab)))
    }
}
