// Vulkan version of `matvec.metal`. `BITS` is substituted before compiling.
const BITS: u32 = 0u;
const PER_WORD: u32 = 32u / BITS;
const MASK: u32 = (1u << BITS) - 1u;

struct Job {
    cols: u32,
    groups: u32,
    group_size: u32,
    words_per_row: u32,
    first: u32,
    factor: f32,
    limit: i64,
}

@group(0) @binding(0) var<storage, read> words: array<u32>;
@group(0) @binding(1) var<storage, read> scales: array<u32>;
@group(0) @binding(2) var<storage, read> biases: array<u32>;
@group(0) @binding(3) var<storage, read> float_x: array<f32>;
@group(0) @binding(4) var<storage, read> sums: array<f32>;
@group(0) @binding(5) var<storage, read_write> float_y: array<f32>;
@group(0) @binding(6) var<storage, read> int_x: array<i32>;
@group(0) @binding(7) var<storage, read_write> int_y: array<i64>;
var<immediate> job: Job;

fn half_in(index: u32, is_scale: bool) -> f32 {
    let word = select(biases[index / 2u], scales[index / 2u], is_scale);
    return unpack2x16float(word)[index % 2u];
}

fn code_at(row: u32, at: u32) -> u32 {
    return (words[row * job.words_per_row + at / PER_WORD] >> (BITS * (at % PER_WORD))) & MASK;
}

// Rounds halves away from zero like the CPU; WGSL's `round` goes to even.
fn to_integer(value: f32) -> i64 {
    return i64(sign(value) * floor(abs(value) + 0.5));
}

@compute @workgroup_size(64)
fn float_product(@builtin(global_invocation_id) id: vec3<u32>) {
    let row = id.x + job.first;
    var total = 0.0;
    var at = 0u;
    for (var g = 0u; g < job.groups; g++) {
        var dot = 0.0;
        for (var j = 0u; j < job.group_size; j++) {
            dot += f32(code_at(row, at)) * float_x[at];
            at++;
        }
        total += dot * half_in(row * job.groups + g, true) + half_in(row * job.groups + g, false) * sums[g];
    }
    float_y[id.x] = total;
}

@compute @workgroup_size(64)
fn exact_product(@builtin(global_invocation_id) id: vec3<u32>) {
    let row = id.x + job.first;
    var total: i64 = 0li;
    var at = 0u;
    for (var g = 0u; g < job.groups; g++) {
        let scale = to_integer(half_in(row * job.groups + g, true) * job.factor);
        let bias = to_integer(half_in(row * job.groups + g, false) * job.factor);
        var dot: i64 = 0li;
        var sum: i64 = 0li;
        for (var j = 0u; j < job.group_size; j++) {
            if job.limit == 0li {
                dot += i64(code_at(row, at)) * i64(int_x[at]);
                sum += i64(int_x[at]);
            } else {
                var weight = (i64(code_at(row, at)) * scale + bias) >> 4u;
                weight = min(max(weight, -job.limit), job.limit);
                total += weight * i64(int_x[at]);
            }
            at++;
        }
        // A limit of zero asks for the sum the cells make: factors apply to the whole group.
        if job.limit == 0li {
            total += dot * scale + sum * bias;
        }
    }
    if job.limit == 0li {
        total = total >> 4u;
    }
    int_y[id.x] = total;
}
