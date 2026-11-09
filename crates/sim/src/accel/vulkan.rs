//! Vulkan backend for Linux and Windows. Tensors are copied to the GPU the first time
//! they are used; when one does not fit, that product runs on the CPU instead.

use super::{Accelerator, ExactScale, PackedTensor};
use ash::vk;
use std::collections::HashMap;
use std::ffi::CStr;
use std::sync::Mutex;

const BINDINGS: u32 = 8;
const GROUP_THREADS: usize = 64;

#[repr(C)]
#[derive(Clone, Copy)]
struct Job {
    cols: u32,
    groups: u32,
    group_size: u32,
    words_per_row: u32,
    first: u32,
    factor: f32,
    limit: i64,
}

/// Compiles the shader for one code width to SPIR-V.
fn spirv(bits: u32, entry: &str) -> Result<Vec<u32>, String> {
    use naga::back::spv;
    use naga::valid::{Capabilities, ValidationFlags, Validator};
    let source = include_str!("matvec.wgsl").replace("const BITS: u32 = 0u;", &format!("const BITS: u32 = {bits}u;"));
    let module = naga::front::wgsl::parse_str(&source).map_err(|e| e.emit_to_string(&source))?;
    let capabilities = Capabilities::SHADER_INT64 | Capabilities::IMMEDIATES | Capabilities::SHADER_FLOAT16_IN_FLOAT32;
    let info = Validator::new(ValidationFlags::all(), capabilities).validate(&module).map_err(|e| format!("{e:?}"))?;
    let options = spv::Options { lang_version: (1, 1), flags: spv::WriterFlags::empty(), ..Default::default() };
    let pipeline = spv::PipelineOptions { shader_stage: naga::ShaderStage::Compute, entry_point: entry.to_string() };
    spv::write_vec(&module, &info, &options, Some(&pipeline)).map_err(|e| e.to_string())
}

#[derive(Clone, Copy)]
struct Allocation {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    size: vk::DeviceSize,
}

struct Mapped {
    allocation: Allocation,
    pointer: *mut u8,
}

struct State {
    pool: vk::CommandPool,
    commands: vk::CommandBuffer,
    fence: vk::Fence,
    descriptors: vk::DescriptorSet,
    tensors: HashMap<usize, Allocation>,
    input: Option<Mapped>,
    output: Option<Mapped>,
}

pub struct VulkanAccelerator {
    _entry: ash::Entry,
    instance: ash::Instance,
    device: ash::Device,
    name: String,
    memory: vk::PhysicalDeviceMemoryProperties,
    queue: vk::Queue,
    layout: vk::PipelineLayout,
    set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    float_4: vk::Pipeline,
    float_8: vk::Pipeline,
    exact_4: vk::Pipeline,
    exact_8: vk::Pipeline,
    state: Mutex<State>,
}

// SAFETY: every use of the queue, command buffer and mapped pointers holds the `state` mutex.
unsafe impl Send for VulkanAccelerator {}
unsafe impl Sync for VulkanAccelerator {}

impl VulkanAccelerator {
    pub fn new() -> Option<Self> {
        // SAFETY: standard Vulkan set-up; every handle is owned by the returned object.
        unsafe {
            let entry = ash::Entry::load().ok()?;
            let app = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_1);
            let instance = entry.create_instance(&vk::InstanceCreateInfo::default().application_info(&app), None).ok()?;
            let devices = instance.enumerate_physical_devices().ok()?;
            let rank = |d: vk::PhysicalDevice| match instance.get_physical_device_properties(d).device_type {
                vk::PhysicalDeviceType::DISCRETE_GPU => 2,
                vk::PhysicalDeviceType::INTEGRATED_GPU => 1,
                _ => 0,
            };
            let physical = devices.into_iter().filter(|d| rank(*d) > 0 && instance.get_physical_device_features(*d).shader_int64 == vk::TRUE).max_by_key(|d| rank(*d))?;
            let family = instance.get_physical_device_queue_family_properties(physical).iter().position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE))? as u32;
            let name = CStr::from_ptr(instance.get_physical_device_properties(physical).device_name.as_ptr()).to_string_lossy().into_owned();
            let memory = instance.get_physical_device_memory_properties(physical);
            let queues = [vk::DeviceQueueCreateInfo::default().queue_family_index(family).queue_priorities(&[1.0])];
            let features = vk::PhysicalDeviceFeatures::default().shader_int64(true);
            let device = instance.create_device(physical, &vk::DeviceCreateInfo::default().queue_create_infos(&queues).enabled_features(&features), None).ok()?;
            let queue = device.get_device_queue(family, 0);

            let bindings: Vec<_> = (0..BINDINGS).map(|i| vk::DescriptorSetLayoutBinding::default().binding(i).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE)).collect();
            let set_layout = device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings), None).ok()?;
            let range = [vk::PushConstantRange::default().stage_flags(vk::ShaderStageFlags::COMPUTE).size(std::mem::size_of::<Job>() as u32)];
            let layouts = [set_layout];
            let layout = device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts).push_constant_ranges(&range), None).ok()?;

            let pipeline = |bits: u32, entry_point: &str| -> Option<vk::Pipeline> {
                let code = spirv(bits, entry_point).map_err(|e| eprintln!("Vulkan: {e}")).ok()?;
                let module = device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None).ok()?;
                let name = std::ffi::CString::new(entry_point).ok()?;
                let stage = vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(module).name(&name);
                let info = [vk::ComputePipelineCreateInfo::default().stage(stage).layout(layout)];
                let created = device.create_compute_pipelines(vk::PipelineCache::null(), &info, None).ok()?;
                device.destroy_shader_module(module, None);
                created.into_iter().next()
            };
            let (float_4, float_8, exact_4, exact_8) = (pipeline(4, "float_product")?, pipeline(8, "float_product")?, pipeline(4, "exact_product")?, pipeline(8, "exact_product")?);

            let sizes = [vk::DescriptorPoolSize::default().ty(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(BINDINGS)];
            let descriptor_pool = device.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(1).pool_sizes(&sizes), None).ok()?;
            let descriptors = device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(descriptor_pool).set_layouts(&layouts)).ok()?[0];
            let pool = device.create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(family).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER), None).ok()?;
            let commands = device.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(pool).level(vk::CommandBufferLevel::PRIMARY).command_buffer_count(1)).ok()?[0];
            let fence = device.create_fence(&vk::FenceCreateInfo::default(), None).ok()?;
            let state = Mutex::new(State { pool, commands, fence, descriptors, tensors: HashMap::new(), input: None, output: None });
            Some(Self { _entry: entry, instance, device, name, memory, queue, layout, set_layout, descriptor_pool, float_4, float_8, exact_4, exact_8, state })
        }
    }

    fn memory_type(&self, bits: u32, wanted: vk::MemoryPropertyFlags) -> Option<u32> {
        (0..self.memory.memory_type_count).find(|i| bits & (1 << i) != 0 && self.memory.memory_types[*i as usize].property_flags.contains(wanted))
    }

    unsafe fn allocate(&self, size: vk::DeviceSize, usage: vk::BufferUsageFlags, wanted: vk::MemoryPropertyFlags) -> Option<Allocation> {
        let buffer = self.device.create_buffer(&vk::BufferCreateInfo::default().size(size).usage(usage).sharing_mode(vk::SharingMode::EXCLUSIVE), None).ok()?;
        let needs = self.device.get_buffer_memory_requirements(buffer);
        let free = |device: &ash::Device| device.destroy_buffer(buffer, None);
        let Some(kind) = self.memory_type(needs.memory_type_bits, wanted) else {
            free(&self.device);
            return None;
        };
        let Ok(memory) = self.device.allocate_memory(&vk::MemoryAllocateInfo::default().allocation_size(needs.size).memory_type_index(kind), None) else {
            free(&self.device);
            return None;
        };
        self.device.bind_buffer_memory(buffer, memory, 0).ok()?;
        Some(Allocation { buffer, memory, size })
    }

    unsafe fn release(&self, allocation: Allocation) {
        self.device.destroy_buffer(allocation.buffer, None);
        self.device.free_memory(allocation.memory, None);
    }

    unsafe fn host_buffer(&self, size: vk::DeviceSize) -> Option<Mapped> {
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC;
        let allocation = self.allocate(size, usage, vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT)?;
        let pointer = self.device.map_memory(allocation.memory, 0, size, vk::MemoryMapFlags::empty()).ok()? as *mut u8;
        Some(Mapped { allocation, pointer })
    }

    /// Records, submits and waits for one batch of commands.
    unsafe fn submit(&self, state: &State, record: impl FnOnce(vk::CommandBuffer)) -> bool {
        let commands = state.commands;
        if self.device.begin_command_buffer(commands, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)).is_err() {
            return false;
        }
        record(commands);
        if self.device.end_command_buffer(commands).is_err() {
            return false;
        }
        let list = [commands];
        let submit = [vk::SubmitInfo::default().command_buffers(&list)];
        let done = self.device.queue_submit(self.queue, &submit, state.fence).is_ok() && self.device.wait_for_fences(&[state.fence], true, u64::MAX).is_ok();
        self.device.reset_fences(&[state.fence]).is_ok() && done
    }

    /// The GPU's copy of a tensor, uploading it on first use.
    unsafe fn tensor(&self, state: &mut State, bytes: &[u8]) -> Option<Allocation> {
        let key = bytes.as_ptr() as usize;
        if let Some(found) = state.tensors.get(&key) {
            return Some(*found);
        }
        let size = bytes.len() as vk::DeviceSize;
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST;
        let allocation = self.allocate(size, usage, vk::MemoryPropertyFlags::DEVICE_LOCAL)?;
        let staging = self.host_buffer(size);
        let Some(staging) = staging else {
            self.release(allocation);
            return None;
        };
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), staging.pointer, bytes.len());
        let region = [vk::BufferCopy::default().size(size)];
        let copied = self.submit(state, |c| self.device.cmd_copy_buffer(c, staging.allocation.buffer, allocation.buffer, &region));
        self.release(staging.allocation);
        if !copied {
            self.release(allocation);
            return None;
        }
        state.tensors.insert(key, allocation);
        Some(allocation)
    }

    #[allow(clippy::too_many_arguments)]
    fn run(&self, pipeline: vk::Pipeline, tensor: &PackedTensor<'_>, rows: usize, job: Job, float: bool, input: &[u8], sums_at: usize, output: &mut [u8]) -> bool {
        let mut guard = self.state.lock().expect("estado de Vulkan");
        let state = &mut *guard;
        // SAFETY: the mutex serialises GPU work, and the fence wait ends each submission
        // before any buffer is read, reused or released.
        unsafe {
            let (Some(words), Some(scales), Some(biases)) = (self.tensor(state, tensor.words), self.tensor(state, tensor.scales), self.tensor(state, tensor.biases)) else {
                return false;
            };
            if state.input.as_ref().is_none_or(|b| (b.allocation.size as usize) < input.len()) {
                state.input = self.host_buffer((input.len() * 2) as vk::DeviceSize);
            }
            if state.output.as_ref().is_none_or(|b| (b.allocation.size as usize) < output.len()) {
                state.output = self.host_buffer((output.len() * 2) as vk::DeviceSize);
            }
            let (Some(input_buffer), Some(output_buffer)) = (state.input.as_ref(), state.output.as_ref()) else {
                return false;
            };
            std::ptr::copy_nonoverlapping(input.as_ptr(), input_buffer.pointer, input.len());
            let whole = |buffer: vk::Buffer, offset: vk::DeviceSize, range: vk::DeviceSize| [vk::DescriptorBufferInfo::default().buffer(buffer).offset(offset).range(range)];
            let input_size = input.len() as vk::DeviceSize;
            let (sums_offset, sums_range) = if float { (sums_at as vk::DeviceSize, input_size - sums_at as vk::DeviceSize) } else { (0, input_size) };
            let infos = [
                whole(words.buffer, 0, vk::WHOLE_SIZE),
                whole(scales.buffer, 0, vk::WHOLE_SIZE),
                whole(biases.buffer, 0, vk::WHOLE_SIZE),
                whole(input_buffer.allocation.buffer, 0, input_size),
                whole(input_buffer.allocation.buffer, sums_offset, sums_range),
                whole(output_buffer.allocation.buffer, 0, vk::WHOLE_SIZE),
                whole(input_buffer.allocation.buffer, 0, input_size),
                whole(output_buffer.allocation.buffer, 0, vk::WHOLE_SIZE),
            ];
            let writes: Vec<_> = infos.iter().enumerate().map(|(i, info)| vk::WriteDescriptorSet::default().dst_set(state.descriptors).dst_binding(i as u32).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(info)).collect();
            self.device.update_descriptor_sets(&writes, &[]);
            let descriptors = [state.descriptors];
            let constants = std::slice::from_raw_parts((&job as *const Job).cast::<u8>(), std::mem::size_of::<Job>());
            let finished = self.submit(state, |c| {
                self.device.cmd_bind_pipeline(c, vk::PipelineBindPoint::COMPUTE, pipeline);
                self.device.cmd_bind_descriptor_sets(c, vk::PipelineBindPoint::COMPUTE, self.layout, 0, &descriptors, &[]);
                self.device.cmd_push_constants(c, self.layout, vk::ShaderStageFlags::COMPUTE, 0, constants);
                self.device.cmd_dispatch(c, rows.div_ceil(GROUP_THREADS) as u32, 1, 1);
            });
            if finished {
                std::ptr::copy_nonoverlapping(output_buffer.pointer, output.as_mut_ptr(), output.len());
            }
            finished
        }
    }

    fn job(tensor: &PackedTensor<'_>, first: usize, cols: usize, factor: f32, limit: i64) -> Option<Job> {
        Some(Job { cols: cols as u32, groups: tensor.groups as u32, group_size: (cols / tensor.groups) as u32, words_per_row: (tensor.stride / 4) as u32, first: u32::try_from(first).ok()?, factor, limit })
    }
}

impl Drop for VulkanAccelerator {
    fn drop(&mut self) {
        let state = self.state.get_mut().expect("estado de Vulkan");
        // SAFETY: nothing else can be using the device while it is being dropped.
        unsafe {
            let _ = self.device.device_wait_idle();
            for allocation in state.tensors.values() {
                self.device.destroy_buffer(allocation.buffer, None);
                self.device.free_memory(allocation.memory, None);
            }
            for mapped in [state.input.as_ref(), state.output.as_ref()].into_iter().flatten() {
                self.device.destroy_buffer(mapped.allocation.buffer, None);
                self.device.free_memory(mapped.allocation.memory, None);
            }
            self.device.destroy_fence(state.fence, None);
            self.device.destroy_command_pool(state.pool, None);
            self.device.destroy_descriptor_pool(self.descriptor_pool, None);
            for pipeline in [self.float_4, self.float_8, self.exact_4, self.exact_8] {
                self.device.destroy_pipeline(pipeline, None);
            }
            self.device.destroy_pipeline_layout(self.layout, None);
            self.device.destroy_descriptor_set_layout(self.set_layout, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
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

impl Accelerator for VulkanAccelerator {
    fn name(&self) -> String {
        format!("Vulkan ({})", self.name)
    }

    fn float_product(&self, tensor: &PackedTensor<'_>, first: usize, x: &[f32], y: &mut [f32]) -> bool {
        let pipeline = match tensor.bits {
            4 => self.float_4,
            8 => self.float_8,
            _ => return false,
        };
        let Some(job) = Self::job(tensor, first, x.len(), 0.0, 0) else { return false };
        let group_size = x.len() / tensor.groups;
        let mut input: Vec<f32> = Vec::with_capacity(x.len() + tensor.groups);
        input.extend_from_slice(x);
        input.extend(x.chunks_exact(group_size).map(|c| c.iter().sum::<f32>()));
        self.run(pipeline, tensor, y.len(), job, true, bytes_of(&input), x.len() * 4, bytes_of_mut(y))
    }

    fn exact_product(&self, tensor: &PackedTensor<'_>, first: usize, x: &[i32], scale: ExactScale, sums: &mut [i64]) -> bool {
        let pipeline = match tensor.bits {
            4 => self.exact_4,
            8 => self.exact_8,
            _ => return false,
        };
        let Some(job) = Self::job(tensor, first, x.len(), 2f32.powi(scale.shift), scale.limit) else { return false };
        self.run(pipeline, tensor, sums.len(), job, false, bytes_of(x), 0, bytes_of_mut(sums))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shaders_compile_to_spirv() {
        for bits in [4, 8] {
            for entry in ["float_product", "exact_product"] {
                let code = spirv(bits, entry).unwrap_or_else(|e| panic!("{bits} bits, {entry}: {e}"));
                assert_eq!(code[0], 0x0723_0203, "número mágico de SPIR-V");
            }
        }
    }
}
