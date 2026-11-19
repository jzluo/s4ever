// `msl.source` puts this file at the start of every Metal library of s4ever. Complex numbers are
// float2 (re, im) pairs.
//
// torch.mps.compile_shader passes Python ints to a kernel as 64-bit integers, so the kernels take
// scalar arguments as `constant long&`. Metal emulates 64-bit integer division, which is slow, so
// the kernels cast index arithmetic to uint before they divide or take a remainder.

#include <metal_stdlib>
using namespace metal;

// IDS declares the thread indices. Every kernel takes them as its last parameters, except the FFT
// kernels, which declare their own indices.
#define IDS \
    uint t [[thread_position_in_threadgroup]], uint g [[threadgroup_position_in_grid]], \
    uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]]

inline float2 cmul(float2 a, float2 b) {
    return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}
inline float2 cmulc(float2 a, float2 b) {  // a conj(b)
    return float2(a.x * b.x + a.y * b.y, a.y * b.x - a.x * b.y);
}
inline float2 conjugate(float2 a) {
    return float2(a.x, -a.y);
}
