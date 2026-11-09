#include <metal_stdlib>
using namespace metal;

struct Job {
    uint cols;
    uint groups;
    uint group_size;
    uint words_per_row;
    uint skip;
    float factor;
    long limit;
};

// Checkpoints do not align their tensors, so halves are assembled from bytes.
inline float half_at(device const uchar* bytes, uint index) {
    return float(as_type<half>(ushort(ushort(bytes[2 * index]) | (ushort(bytes[2 * index + 1]) << 8))));
}

// Loads are the scarce resource on the GPU: words and activations are read in blocks,
// one word for every eight 4-bit codes or four 8-bit codes.
struct Words {
    device const uint* next;
    uint previous;
    uint shift;

    inline uint take() {
        const uint current = *next++;
        const uint word = shift == 0 ? previous : (previous >> shift) | (current << (32 - shift));
        previous = current;
        return word;
    }
};

inline Words words_at(device const uint* row, uint shift) {
    return Words { row + 1, row[0], shift };
}

template <uint BITS>
inline float dot_block(uint word, device const float4* x, uint at) {
    if (BITS == 8) {
        return dot(float4(as_type<uchar4>(word)), x[at / 4]);
    }
    const uint4 low = uint4(word, word >> 4, word >> 8, word >> 12) & 15u;
    const uint4 high = uint4(word >> 16, word >> 20, word >> 24, word >> 28) & 15u;
    return dot(float4(low), x[at / 4]) + dot(float4(high), x[at / 4 + 1]);
}

template <uint BITS>
inline float float_row(device const uint* row, device const uchar* scales, device const uchar* biases, device const float4* x, device const float* sums, constant Job& job) {
    const uint per_word = 32 / BITS;
    const uint blocks = job.group_size / per_word;
    Words words = words_at(row, job.skip);
    float total = 0;
    uint at = 0;
    for (uint g = 0; g < job.groups; ++g) {
        float sum = 0;
        for (uint j = 0; j < blocks; ++j, at += per_word) {
            sum += dot_block<BITS>(words.take(), x, at);
        }
        total += sum * half_at(scales, g) + half_at(biases, g) * sums[g];
    }
    return total;
}

inline long wide_product(int weight, int x) {
    return as_type<long>(int2(weight * x, mulhi(weight, x)));
}

inline int exact_weight(uint code, int scale, int bias, int limit) {
    return clamp((int(code) * scale + bias) >> 4, -limit, limit);
}

template <uint BITS>
inline long exact_row(device const uint* row, device const uchar* scales, device const uchar* biases, device const int4* x, constant Job& job) {
    const uint per_word = 32 / BITS;
    const uint blocks = job.group_size / per_word;
    const uint mask = (1u << BITS) - 1;
    const int limit = int(job.limit);
    Words words = words_at(row, job.skip);
    long total = 0;
    uint at = 0;
    for (uint g = 0; g < job.groups; ++g) {
        const int scale = int(round(half_at(scales, g) * job.factor));
        const int bias = int(round(half_at(biases, g) * job.factor));
        long dot = 0, sum = 0;
        for (uint j = 0; j < blocks; ++j) {
            uint word = words.take();
            int4 activations = x[at / 4];
            for (uint i = 0; i < per_word; ++i, ++at, word >>= BITS) {
                if (i != 0 && i % 4 == 0) {
                    activations = x[at / 4];
                }
                if (limit == 0) {
                    dot += wide_product(int(word & mask), activations[i % 4]);
                    sum += long(activations[i % 4]);
                } else {
                    total += wide_product(exact_weight(word & mask, scale, bias, limit), activations[i % 4]);
                }
            }
        }
        // A limit of zero asks for the sum the cells make: factors apply to the whole group.
        if (limit == 0) {
            total += dot * long(scale) + sum * long(bias);
        }
    }
    return limit == 0 ? total >> 4 : total;
}

#define FLOAT_KERNEL(NAME, BITS)                                                                                     \
    kernel void NAME(device const uint* words [[buffer(0)]], device const uchar* scales [[buffer(1)]],                \
                     device const uchar* biases [[buffer(2)]], device const float4* x [[buffer(3)]],                   \
                     device const float* sums [[buffer(4)]], device float* y [[buffer(5)]],                          \
                     constant Job& job [[buffer(6)]], uint id [[thread_position_in_grid]]) {                         \
        y[id] = float_row<BITS>(words + id * job.words_per_row, scales + 2 * id * job.groups, biases + 2 * id * job.groups,  \
                                x, sums, job);                                                                       \
    }

#define EXACT_KERNEL(NAME, BITS)                                                                                     \
    kernel void NAME(device const uint* words [[buffer(0)]], device const uchar* scales [[buffer(1)]],                \
                     device const uchar* biases [[buffer(2)]], device const int4* x [[buffer(3)]],                     \
                     device long* y [[buffer(5)]], constant Job& job [[buffer(6)]],                                  \
                     uint id [[thread_position_in_grid]]) {                                                          \
        y[id] = exact_row<BITS>(words + id * job.words_per_row, scales + 2 * id * job.groups, biases + 2 * id * job.groups,  \
                                x, job);                                                                             \
    }

FLOAT_KERNEL(float_4, 4)
FLOAT_KERNEL(float_8, 8)
EXACT_KERNEL(exact_4, 4)
EXACT_KERNEL(exact_8, 8)
