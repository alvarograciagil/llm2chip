//! The image encoder in the die's arithmetic: its routines run patch by patch and block by
//! block, in the order the firmware gives them, on the exact machine.

use crate::lower::Layout;
use crate::vision::{patch_angles, Patches, VisionTower};
use crate::vm::BlockVm;
use crate::weights::Weights;
use compiler::{Inst, Program, VisionPart};
use netlist::block::{encode_vector, exponent_register, BlockOps};
use netlist::DieConfig;
use std::sync::Arc;

/// The order the firmware runs an encoder's routines in for an image of `count` patches,
/// before the fusion: `(operations, patch as token, position, load the patch's rotation)`.
/// Each patch is embedded, and each block runs its first routine on every patch and then
/// the second, which attends over all of them.
pub fn routine_order(part: &VisionPart, count: usize) -> Vec<(&[Inst], u32, u32, bool)> {
    let count = count as u32;
    let embed = (0..count).map(|p| (&part.ops[part.embed.clone()], p, p, false));
    let blocks = part.blocks.iter().flat_map(move |(first, rest)| {
        let first = (0..count).map(move |p| (&part.ops[first.clone()], p, p, true));
        let rest = (0..count).map(move |p| (&part.ops[rest.clone()], p, count - 1, false));
        first.chain(rest)
    });
    embed.chain(blocks).collect()
}

pub struct ExactVision {
    pub program: Program,
    pub part: VisionPart,
    layout: Layout,
    vm: BlockVm,
    bits: u32,
    exponents: u32,
}

impl ExactVision {
    /// The encoder of `config_json` alone, for images of at most `patches` patches, on a
    /// die like `config` with vectors of `config.cell.activation_bits`.
    pub fn new(config_json: &str, weights: Weights, patches: usize, config: DieConfig) -> Result<Self, String> {
        let compiler::VisionProgram { program, vision } = compiler::compile_vision(config_json, patches)?;
        Self::with_program(program, vision, weights, config)
    }

    /// The encoder alone with the exact engine's large die and `bits`-bit vectors.
    pub fn exact(config_json: &str, weights: Weights, patches: usize, bits: u32) -> Result<Self, String> {
        Self::new(config_json, weights, patches, crate::exact::exact_config(bits))
    }

    /// The encoder `part` of `program`, which may be a language model's as well.
    pub fn with_program(program: Program, part: VisionPart, weights: Weights, config: DieConfig) -> Result<Self, String> {
        let weights = Arc::new(weights);
        let layout = Layout::new(&program, &weights, config, None)?;
        let mut vm = BlockVm::new(&config, layout.registers_used() as usize + 8, Some(weights));
        vm.compute_in_cells();
        Ok(Self { program, part, layout, vm, bits: config.cell.activation_bits, exponents: exponent_register(&config) })
    }

    /// Adds up matrix products weight by weight, as a die that holds them in registers.
    pub fn with_weights_in_registers(mut self) -> Self {
        self.vm.compute_in_registers();
        self
    }

    /// A buffer's values as the die holds them, to look inside the encoder.
    pub fn buffer(&self, buffer: usize) -> Vec<f32> {
        self.vm.floats(self.layout.register_of(buffer), self.program.buffers[buffer])
    }


    /// The tokens of an image, as `VisionTower::encode` gives them. The tower supplies what
    /// the host computes: each patch's learned position and rotation.
    pub fn encode(&mut self, tower: &VisionTower, image: &Patches) -> Result<Vec<Vec<f32>>, String> {
        self.vm.clear();
        crate::chip::load_initial(&mut self.vm, &self.layout.setup().initial, self.exponents);
        let count = run_encoder(&self.layout, &mut self.vm, &self.program, &self.part, self.bits, tower, image)?;
        let (output, width) = (self.layout.register_of(self.part.output), self.program.buffers[self.part.output]);
        (0..count as u32)
            .map(|token| {
                self.layout.run_ops(&mut self.vm, &self.program, &self.part.ops[self.part.merge.clone()], token, token, false)?;
                Ok(self.vm.floats(output, width))
            })
            .collect()
    }
}

/// Runs an image through the encoder up to the fusion, with the host's part: each patch's
/// pixels, learned position and rotation. Gives how many tokens the fusion can then make.
pub(crate) fn run_encoder(layout: &Layout, vm: &mut BlockVm, program: &Program, part: &VisionPart, bits: u32, tower: &VisionTower, image: &Patches) -> Result<usize, String> {
    let count = image.rows * image.cols;
    if count > part.patches {
        return Err(format!("la imagen tiene {count} parches y el codificador admite {}", part.patches));
    }
    layout.clear_rotations();
    for (patch, angles) in patch_angles(&tower.config, image.rows, image.cols).iter().enumerate() {
        layout.set_rotation(patch as u32, &angles.iter().map(|a| *a as f64).collect::<Vec<_>>());
    }
    let size = program.buffers[part.pixels];
    let positions = tower.positions(image.rows, image.cols)?;
    let mut inputs = image.values.chunks(size).zip(&positions);
    let load = |vm: &mut BlockVm, buffer: usize, values: &[f32]| {
        let (mantissas, exponent) = encode_vector(&values.iter().map(|v| *v as f64).collect::<Vec<_>>(), bits);
        vm.load_mantissas(layout.register_of(buffer), &mantissas, exponent);
    };
    for (index, (list, token, position, rope)) in routine_order(part, count).iter().enumerate() {
        // The host writes each patch's pixels and position before its embedding.
        if index < count {
            let (values, place) = inputs.next().ok_or("faltan parches")?;
            load(vm, part.pixels, values);
            load(vm, part.position, place);
        }
        layout.run_ops(vm, program, list, *token, *position, *rope)?;
    }
    Ok(count / (tower.config.merge * tower.config.merge))
}

/// What the host sends a chip that carries the encoder before the prompt: how many patches,
/// the grid's rows and columns, and for each patch its pixels and learned position, each
/// as its exponent and its elements in the die's format of `bits` bits, and its row of the
/// encoder's rotation table. Every number is 4 bytes, little-endian. No image is a count
/// of zero.
pub fn picture_bytes(tower: &VisionTower, image: Option<&Patches>, bits: u32) -> Result<Vec<u8>, String> {
    let mut bytes = Vec::new();
    let mut word = |value: i64| bytes.extend((value as i32).to_le_bytes());
    let Some(image) = image else {
        word(0);
        return Ok(bytes);
    };
    let c = &tower.config;
    word((image.rows * image.cols) as i64);
    word(image.rows as i64);
    word(image.cols as i64);
    let positions = tower.positions(image.rows, image.cols)?;
    let angles = patch_angles(c, image.rows, image.cols);
    for ((pixels, place), angles) in image.values.chunks(3 * c.temporal * c.patch * c.patch).zip(&positions).zip(&angles) {
        for values in [pixels, place.as_slice()] {
            let (mantissas, exponent) = encode_vector(&values.iter().map(|v| *v as f64).collect::<Vec<_>>(), bits);
            word(exponent);
            mantissas.into_iter().for_each(&mut word);
        }
        let row = crate::lower::patch_rotation_row(&angles.iter().map(|a| *a as f64).collect::<Vec<_>>());
        crate::lower::encode_at(&row, bits, crate::lower::PATCH_ROTATION_EXPONENT).into_iter().for_each(&mut word);
    }
    Ok(bytes)
}

/// What the firmware needs to run an encoder of `config` on a die run in software, which
/// takes the image from the host and leaves its work to the engine: no blocks of its own
/// and no rotations to write, and its vectors in registers 3 to 5 of `index_bits`.
pub fn software_vision_info(config: &crate::vision::VisionConfig, patches: usize, index_bits: u32, register_bits: u32) -> netlist::chip::VisionInfo {
    let exponents = ((1u32 << register_bits) - 1) << index_bits;
    let vector = |register: u32, size: usize| (register << index_bits, exponents | register, size as u32);
    let head_dim = config.hidden / config.heads;
    netlist::chip::VisionInfo {
        patches: patches as u32,
        image_token: config.image_token,
        merge: config.merge as u32,
        blocks: 0,
        pixels: vector(3, 3 * config.temporal * config.patch * config.patch),
        position: vector(4, config.hidden),
        patch_rope: (5 << index_bits, 3 * (head_dim / 2) as u32 + 1),
        rope: 0,
        rope_half: 0,
        mrope_rows: config.mrope_section[1] as u32,
        mrope_cols: config.mrope_section[2] as u32,
    }
}
