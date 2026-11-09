//! Metal backend. Checkpoint files are mapped straight into GPU buffers, without copies.

use super::{Accelerator, ExactScale, PackedTensor};
use objc2::rc::{autoreleasepool, Retained};
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::*;
use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::Mutex;

type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;
type Pipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;

const PAGE: usize = 16384;

#[repr(C)]
#[derive(Clone, Copy)]
struct Job {
    cols: u32,
    groups: u32,
    group_size: u32,
    words_per_row: u32,
    factor: f32,
    limit: i64,
}

struct File {
    start: usize,
    end: usize,
    buffer: Buffer,
}

struct Scratch {
    input: Buffer,
    output: Buffer,
}

pub struct MetalAccelerator {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    files: Vec<File>,
    float_4: Pipeline,
    float_8: Pipeline,
    exact_4: Pipeline,
    exact_8: Pipeline,
    scratch: Mutex<Scratch>,
}

// SAFETY: Metal devices, queues, buffers and pipelines may be shared between threads;
// the mutable scratch buffers are only reached through the mutex.
unsafe impl Send for MetalAccelerator {}
unsafe impl Sync for MetalAccelerator {}

fn shared(device: &ProtocolObject<dyn MTLDevice>, length: usize) -> Option<Buffer> {
    device.newBufferWithLength_options(length.max(16), MTLResourceOptions::StorageModeShared)
}

impl MetalAccelerator {
    pub fn new(mapped: &[&[u8]]) -> Option<Self> {
        autoreleasepool(|_| {
            let device = MTLCreateSystemDefaultDevice()?;
            let queue = device.newCommandQueue()?;
            let source = NSString::from_str(include_str!("matvec.metal"));
            let library = device.newLibraryWithSource_options_error(&source, None).map_err(|e| eprintln!("Metal: {e}")).ok()?;
            let pipeline = |name: &str| -> Option<Pipeline> {
                let function = library.newFunctionWithName(&NSString::from_str(name))?;
                device.newComputePipelineStateWithFunction_error(&function).map_err(|e| eprintln!("Metal: {e}")).ok()
            };
            let (float_4, float_8, exact_4, exact_8) = (pipeline("float_4")?, pipeline("float_8")?, pipeline("exact_4")?, pipeline("exact_8")?);
            let mut files = Vec::new();
            for bytes in mapped {
                if bytes.as_ptr() as usize % PAGE != 0 {
                    return None;
                }
                let pointer = NonNull::new(bytes.as_ptr() as *mut c_void)?;
                let length = bytes.len().next_multiple_of(PAGE);
                // SAFETY: the mapping is page aligned, read-only here, and outlives this object,
                // which the owning `Weights` drops first.
                let buffer = unsafe { device.newBufferWithBytesNoCopy_length_options_deallocator(pointer, length, MTLResourceOptions::StorageModeShared, None) }?;
                files.push(File { start: bytes.as_ptr() as usize, end: bytes.as_ptr() as usize + bytes.len(), buffer });
            }
            let scratch = Mutex::new(Scratch { input: shared(&device, 1 << 20)?, output: shared(&device, 1 << 20)? });
            Some(Self { device, queue, files, float_4, float_8, exact_4, exact_8, scratch })
        })
    }

    fn locate(&self, bytes: &[u8]) -> Option<(&Buffer, usize)> {
        let address = bytes.as_ptr() as usize;
        let file = self.files.iter().find(|f| address >= f.start && address + bytes.len() <= f.end)?;
        Some((&file.buffer, address - file.start))
    }

    #[allow(clippy::too_many_arguments)]
    fn run(&self, pipeline: &Pipeline, tensor: &PackedTensor<'_>, first: usize, rows: usize, job: Job, input: &[u8], sums_at: Option<usize>, output: &mut [u8]) -> bool {
        let (Some((words, words_at)), Some((scales, scales_at)), Some((biases, biases_at))) = (self.locate(tensor.words), self.locate(tensor.scales), self.locate(tensor.biases)) else {
            return false;
        };
        if words_at % 4 != 0 || scales_at % 2 != 0 || biases_at % 2 != 0 || rows * tensor.stride >= u32::MAX as usize {
            return false;
        }
        let mut scratch = self.scratch.lock().expect("búferes de Metal");
        autoreleasepool(|_| {
            if scratch.input.length() < input.len() {
                let Some(buffer) = shared(&self.device, input.len() * 2) else { return false };
                scratch.input = buffer;
            }
            if scratch.output.length() < output.len() {
                let Some(buffer) = shared(&self.device, output.len() * 2) else { return false };
                scratch.output = buffer;
            }
            // SAFETY: both buffers were just sized to hold these copies and no GPU work is running.
            unsafe { std::ptr::copy_nonoverlapping(input.as_ptr(), scratch.input.contents().as_ptr() as *mut u8, input.len()) };
            let Some(commands) = self.queue.commandBuffer() else { return false };
            let Some(encoder) = commands.computeCommandEncoder() else { return false };
            encoder.setComputePipelineState(pipeline);
            let groups_bytes = first * tensor.groups * 2;
            // SAFETY: every offset lies inside its buffer, as `locate` and the tensor's bounds guarantee.
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(words), words_at + first * tensor.stride, 0);
                encoder.setBuffer_offset_atIndex(Some(scales), scales_at + groups_bytes, 1);
                encoder.setBuffer_offset_atIndex(Some(biases), biases_at + groups_bytes, 2);
                encoder.setBuffer_offset_atIndex(Some(&scratch.input), 0, 3);
                if let Some(at) = sums_at {
                    encoder.setBuffer_offset_atIndex(Some(&scratch.input), at, 4);
                }
                encoder.setBuffer_offset_atIndex(Some(&scratch.output), 0, 5);
                encoder.setBytes_length_atIndex(NonNull::from(&job).cast(), std::mem::size_of::<Job>(), 6);
            }
            let width = pipeline.threadExecutionWidth();
            encoder.dispatchThreads_threadsPerThreadgroup(MTLSize { width: rows, height: 1, depth: 1 }, MTLSize { width, height: 1, depth: 1 });
            encoder.endEncoding();
            commands.commit();
            commands.waitUntilCompleted();
            if commands.status() != MTLCommandBufferStatus::Completed {
                eprintln!("Metal: la orden no terminó: {:?}", commands.error());
                return false;
            }
            // SAFETY: the GPU has finished and the buffer holds at least `output.len()` bytes.
            unsafe { std::ptr::copy_nonoverlapping(scratch.output.contents().as_ptr() as *const u8, output.as_mut_ptr(), output.len()) };
            true
        })
    }

    fn job(tensor: &PackedTensor<'_>, cols: usize, factor: f32, limit: i64) -> Job {
        Job { cols: cols as u32, groups: tensor.groups as u32, group_size: (cols / tensor.groups) as u32, words_per_row: (tensor.stride / 4) as u32, factor, limit }
    }
}

fn bytes_of<T>(values: &[T]) -> &[u8] {
    // SAFETY: these are plain numbers; any byte view of them is valid.
    unsafe { std::slice::from_raw_parts(values.as_ptr() as *const u8, std::mem::size_of_val(values)) }
}

fn bytes_of_mut<T>(values: &mut [T]) -> &mut [u8] {
    // SAFETY: these are plain numbers, and every bit pattern is a valid value.
    unsafe { std::slice::from_raw_parts_mut(values.as_mut_ptr() as *mut u8, std::mem::size_of_val(values)) }
}

impl Accelerator for MetalAccelerator {
    fn name(&self) -> String {
        format!("Metal ({})", self.device.name())
    }

    fn float_product(&self, tensor: &PackedTensor<'_>, first: usize, x: &[f32], y: &mut [f32]) -> bool {
        let pipeline = match tensor.bits {
            4 => &self.float_4,
            8 => &self.float_8,
            _ => return false,
        };
        let group_size = x.len() / tensor.groups;
        let mut input: Vec<f32> = Vec::with_capacity(x.len() + tensor.groups);
        input.extend_from_slice(x);
        input.extend(x.chunks_exact(group_size).map(|c| c.iter().sum::<f32>()));
        let job = Self::job(tensor, x.len(), 0.0, 0);
        self.run(pipeline, tensor, first, y.len(), job, bytes_of(&input), Some(x.len() * 4), bytes_of_mut(y))
    }

    fn exact_product(&self, tensor: &PackedTensor<'_>, first: usize, x: &[i32], scale: ExactScale, sums: &mut [i64]) -> bool {
        let pipeline = match tensor.bits {
            4 => &self.exact_4,
            8 => &self.exact_8,
            _ => return false,
        };
        let job = Self::job(tensor, x.len(), 2f32.powi(scale.shift), scale.limit);
        self.run(pipeline, tensor, first, sums.len(), job, bytes_of(x), None, bytes_of_mut(sums))
    }
}
