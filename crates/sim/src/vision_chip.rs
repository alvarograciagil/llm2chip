//! The image encoder in the die's arithmetic: its routines run patch by patch and block by
//! block, in the order the firmware gives them, on the exact machine.

use crate::lower::Layout;
use crate::vision::{patch_angles, Patches, VisionTower};
use crate::vm::BlockVm;
use crate::weights::Weights;
use compiler::VisionProgram;
use netlist::block::{encode_vector, exponent_register, BlockOps};
use netlist::DieConfig;
use std::sync::Arc;

pub struct ExactVision {
    pub vision: VisionProgram,
    layout: Layout,
    vm: BlockVm,
    bits: u32,
    exponents: u32,
}

impl ExactVision {
    /// The encoder of `config_json`, for images of at most `patches` patches, on a die like
    /// `config` with vectors of `config.cell.activation_bits`.
    pub fn new(config_json: &str, weights: Weights, patches: usize, config: DieConfig) -> Result<Self, String> {
        let vision = compiler::compile_vision(config_json, patches)?;
        let weights = Arc::new(weights);
        let layout = Layout::new(&vision.program, &weights, config, None)?;
        let mut vm = BlockVm::new(&config, layout.registers_used() as usize + 8, Some(weights));
        vm.compute_in_cells();
        Ok(Self { vision, layout, vm, bits: config.cell.activation_bits, exponents: exponent_register(&config) })
    }

    /// A buffer's values as the die holds them, to look inside the encoder.
    pub fn buffer(&self, buffer: usize) -> Vec<f32> {
        self.vm.floats(self.layout.register_of(buffer), self.vision.program.buffers[buffer])
    }

    fn load(&mut self, buffer: usize, values: &[f32]) {
        let values: Vec<f64> = values.iter().map(|v| *v as f64).collect();
        let (mantissas, exponent) = encode_vector(&values, self.bits);
        self.vm.load_mantissas(self.layout.register_of(buffer), &mantissas, exponent);
    }

    /// The tokens of an image, as `VisionTower::encode` gives them. The tower supplies what
    /// the host computes: each patch's learned position and rotation.
    pub fn encode(&mut self, tower: &VisionTower, image: &Patches) -> Result<Vec<Vec<f32>>, String> {
        let count = image.rows * image.cols;
        if count > self.vision.patches {
            return Err(format!("la imagen tiene {count} parches y el codificador admite {}", self.vision.patches));
        }
        self.vm.clear();
        crate::chip::load_initial(&mut self.vm, &self.layout.setup().initial, self.exponents);
        self.layout.clear_rotations();
        let program = &self.vision.program;
        let size = program.buffers[self.vision.pixels];
        let positions = tower.positions(image.rows, image.cols)?;
        for (patch, angles) in patch_angles(&tower.config, image.rows, image.cols).iter().enumerate() {
            self.layout.set_rotation(patch as u32, &angles.iter().map(|a| *a as f64).collect::<Vec<_>>());
        }
        for (patch, (values, position)) in image.values.chunks(size).zip(&positions).enumerate() {
            self.load(self.vision.pixels, values);
            self.load(self.vision.position, position);
            let program = &self.vision.program;
            self.layout.run_ops(&mut self.vm, program, self.vision.embed.clone(), patch as u32, patch as u32, false)?;
        }
        let program = &self.vision.program;
        for (first, rest) in &self.vision.blocks {
            for patch in 0..count as u32 {
                self.layout.run_ops(&mut self.vm, program, first.clone(), patch, patch, true)?;
            }
            for patch in 0..count as u32 {
                self.layout.run_ops(&mut self.vm, program, rest.clone(), patch, count as u32 - 1, false)?;
            }
        }
        let merge = tower.config.merge * tower.config.merge;
        let output = self.layout.register_of(self.vision.output);
        let width = program.buffers[self.vision.output];
        (0..(count / merge) as u32)
            .map(|token| {
                self.layout.run_ops(&mut self.vm, program, self.vision.merge.clone(), token, token, false)?;
                Ok(self.vm.floats(output, width))
            })
            .collect()
    }
}
