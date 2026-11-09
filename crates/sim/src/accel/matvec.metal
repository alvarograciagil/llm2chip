#include <metal_stdlib>
using namespace metal;

struct Job {
    uint cols;
    uint groups;
    uint group_size;
    uint words_per_row;
    float factor;
    long limit;
};

template <uint BITS>
inline uint code_at(device const uint* row, uint at) {
    const uint per_word = 32 / BITS;
    return (row[at / per_word] >> (BITS * (at % per_word))) & ((1u << BITS) - 1);
}

template <uint BITS>
inline float float_row(device const uint* row, device const half* scales, device const half* biases, device const float* x, device const float* sums, constant Job& job) {
    float total = 0;
    uint at = 0;
    for (uint g = 0; g < job.groups; ++g) {
        float dot = 0;
        for (uint j = 0; j < job.group_size; ++j, ++at) {
            dot += float(code_at<BITS>(row, at)) * x[at];
        }
        total += dot * float(scales[g]) + float(biases[g]) * sums[g];
    }
    return total;
}

template <uint BITS>
inline long exact_row(device const uint* row, device const half* scales, device const half* biases, device const int* x, constant Job& job) {
    long total = 0;
    uint at = 0;
    for (uint g = 0; g < job.groups; ++g) {
        const long scale = long(round(float(scales[g]) * job.factor));
        const long bias = long(round(float(biases[g]) * job.factor));
        for (uint j = 0; j < job.group_size; ++j, ++at) {
            const long weight = clamp((long(code_at<BITS>(row, at)) * scale + bias) >> 4, -job.limit, job.limit);
            total += weight * long(x[at]);
        }
    }
    return total;
}

#define FLOAT_KERNEL(NAME, BITS)                                                                                     \
    kernel void NAME(device const uint* words [[buffer(0)]], device const half* scales [[buffer(1)]],                \
                     device const half* biases [[buffer(2)]], device const float* x [[buffer(3)]],                   \
                     device const float* sums [[buffer(4)]], device float* y [[buffer(5)]],                          \
                     constant Job& job [[buffer(6)]], uint id [[thread_position_in_grid]]) {                         \
        y[id] = float_row<BITS>(words + id * job.words_per_row, scales + id * job.groups, biases + id * job.groups,  \
                                x, sums, job);                                                                       \
    }

#define EXACT_KERNEL(NAME, BITS)                                                                                     \
    kernel void NAME(device const uint* words [[buffer(0)]], device const half* scales [[buffer(1)]],                \
                     device const half* biases [[buffer(2)]], device const int* x [[buffer(3)]],                     \
                     device long* y [[buffer(5)]], constant Job& job [[buffer(6)]],                                  \
                     uint id [[thread_position_in_grid]]) {                                                          \
        y[id] = exact_row<BITS>(words + id * job.words_per_row, scales + id * job.groups, biases + id * job.groups,  \
                                x, job);                                                                             \
    }

FLOAT_KERNEL(float_4, 4)
FLOAT_KERNEL(float_8, 8)
EXACT_KERNEL(exact_4, 4)
EXACT_KERNEL(exact_8, 8)
