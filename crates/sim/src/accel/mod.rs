//! GPU backends for the matrix products that dominate simulation time.
//! Each system has its own native backend; the CPU path is always available and is the reference.

#[cfg(target_os = "macos")]
mod metal;
// Compiled everywhere so that it keeps building, but only selected outside macOS.
#[cfg_attr(target_os = "macos", allow(dead_code))]
mod vulkan;

/// A tensor in MLX affine form: `w = q · scale + bias`, with one scale and bias per group.
pub struct PackedTensor<'a> {
    pub words: &'a [u8],
    pub scales: &'a [u8],
    pub biases: &'a [u8],
    pub bits: u32,
    pub stride: usize,
    pub groups: usize,
}

/// The integer arithmetic of the exact machine.
#[derive(Clone, Copy)]
pub struct ExactScale {
    /// `2^shift` turns a scale or bias into an integer with four extra fractional bits.
    pub shift: i32,
    /// Zero asks for the sum the cells make, with no limit per weight.
    pub limit: i64,
}

/// Both products return `false` when the backend cannot run this job, and the CPU takes over.
pub trait Accelerator: Send + Sync {
    fn name(&self) -> String;
    fn float_product(&self, tensor: &PackedTensor<'_>, first: usize, x: &[f32], y: &mut [f32]) -> bool;
    fn exact_product(&self, tensor: &PackedTensor<'_>, first: usize, x: &[i32], scale: ExactScale, sums: &mut [i64]) -> bool;
}

/// The native backend of this system, or `None` when no usable GPU exists.
pub fn open(files: &[&[u8]]) -> Option<Box<dyn Accelerator>> {
    #[cfg(target_os = "macos")]
    {
        metal::MetalAccelerator::new(files).map(|a| Box::new(a) as Box<dyn Accelerator>)
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = files;
        vulkan::VulkanAccelerator::new().map(|a| Box::new(a) as Box<dyn Accelerator>)
    }
}
