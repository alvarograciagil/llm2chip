//! Runs a compiled model with the die's exact integer arithmetic.

use crate::lower::Layout;
use crate::vm::BlockVm;
use crate::weights::Weights;
use compiler::Program;
use netlist::block::{exponent_register, BlockOps};
use netlist::{CellConfig, DieConfig};
use std::sync::Arc;

/// Large register space and 12-bit tables: real models are not limited by a test die.
fn exact_config(bits: u32) -> DieConfig {
    DieConfig {
        cells: 1,
        cell: CellConfig { weight_bits: 8, activation_bits: bits, accumulator_bits: 32, mac_stages: 1, adder_stages: 1, addr_bits: 3, coding: Default::default(), exponent_bits: 0, multiplier: netlist::multiplier::MultiplierKind::Array },
        scale_bits: 16,
        reg_bits: 24,
        index_bits: 24,
        microcode_bits: 18,
        broadcast_stages: 1,
        function_bits: 12,
        function_select_bits: 3,
    }
}

pub struct ExactEngine {
    program: Program,
    layout: Layout,
    vm: BlockVm,
    position: usize,
    draft_tokens: usize,
    exponents: u32,
    pub(crate) logits: Vec<f32>,
}

impl ExactEngine {
    /// `bits` is the mantissa width of the die's vectors.
    pub fn new(program: Program, weights: Weights, bits: u32) -> Result<Self, String> {
        if !(8..=24).contains(&bits) {
            return Err("los vectores admiten de 8 a 24 bits".into());
        }
        let config = exact_config(bits);
        let weights = Arc::new(weights);
        let layout = Layout::new(&program, &weights, config, false)?;
        let vm = BlockVm::new(&config, layout.registers_used() as usize + 8, Some(weights));
        let mut engine = Self { program, layout, vm, position: 0, draft_tokens: 0, exponents: exponent_register(&config), logits: Vec::new() };
        engine.reset();
        Ok(engine)
    }

    pub fn reset(&mut self) {
        self.vm.clear();
        let mut by_reg: std::collections::BTreeMap<u32, Vec<(u32, i64)>> = std::collections::BTreeMap::new();
        let mut exponents = Vec::new();
        for ((reg, index), value) in &self.layout.setup().initial {
            if *reg == self.exponents {
                exponents.push((*index, *value));
            } else {
                by_reg.entry(*reg).or_default().push((*index, *value));
            }
        }
        for (reg, mut entries) in by_reg {
            entries.sort();
            let mantissas: Vec<i64> = entries.iter().map(|(_, v)| *v).collect();
            self.vm.load_mantissas(reg, &mantissas, 0);
        }
        for (reg, exponent) in exponents {
            self.vm.set_exponent(reg, exponent);
        }
        self.position = 0;
        self.draft_tokens = 0;
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

    pub fn step(&mut self, token: u32, want_logits: bool) -> Result<Option<Vec<f32>>, String> {
        self.layout.run_token(&mut self.vm, &self.program, token, self.position, want_logits)?;
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
        self.layout.run_draft(&mut self.vm, &self.program, token, self.draft_tokens, want_logits)?;
        self.draft_tokens += 1;
        let draft = self.program.draft.as_ref().ok_or("este modelo no tiene cabeza MTP")?;
        Ok(want_logits.then(|| self.vm.floats(self.layout.register_of(draft.logits), self.program.vocab)))
    }
}
