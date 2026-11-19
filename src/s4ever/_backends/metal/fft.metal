// The FFT stages that fft_rows.metal and fft_columns.metal share, compiled after the constants and
// DFTs of `fft.source`. A transform of NF points runs on FFT_THREADS threads, which hold
// E = NF / FFT_THREADS values each, and a threadgroup runs GROUP_FFTS transforms. The forward
// stages leave frequency rev(p) at position p.
//
// `fft.source` adds three functions after this file. forward_stages and inverse_stages run all
// stages on the E values v of a thread in place, and inverse_dft applies the unscaled N1-point
// inverse DFT to N1 <= SPLIT values in place.
//
// The kernels whose threadgroups read the same data launch 1-D grids and give that data to
// consecutive threadgroups. Metal starts the threadgroups of a 1-D grid in about index order, so
// the data that one threadgroup loads is still in the cache when the next one reads it. (Metal
// starts a 2-D grid one x column at a time.) If the order did not hold, the outputs would not
// change, but more of the loads would go to DRAM.

// Return w^j from coarse[i] = w^(NF_MAX i) and fine[i] = w^i. The row kernels call it only for
// N1 > 1, where NF = NF_MAX.
inline float2 root(uint j, device const float2* coarse, device const float2* fine) {
    return cmul(coarse[j / NF_MAX], fine[j % NF_MAX]);
}

// Return the position of value e of butterfly j of thread t in a stage of radix R whose
// sub-transforms have M points.
template <uint R, uint M> inline uint pos(uint t, uint j, uint e) {
    constexpr uint Q = M / R;
    uint beta = t + j * FFT_THREADS;
    return (beta / Q) * M + beta % Q + e * Q;
}

// The unscaled inverse DFT. Swapping the parts of z gives i conj(z), so swapping them before and
// after the forward DFT inverts it.
template <uint R> inline void idft(thread float2* v) {
    for (uint e = 0; e < R; e++) v[e] = v[e].yx;
    dft<R>(v);
    for (uint e = 0; e < R; e++) v[e] = v[e].yx;
}

// A decimation-in-frequency stage, which computes R-point DFTs at stride M / R and then multiplies
// by the twiddles.
template <uint R, uint M> inline void forward(thread float2* v, uint t, device const float2* tw) {
    constexpr uint Q = M / R;
    for (uint j = 0; j < E / R; j++) {
        dft<R>(v + j * R);
        if (M > R) {
            uint q = (t + j * FFT_THREADS) % Q;
            for (uint e = 1; e < R; e++) v[j * R + e] = cmul(v[j * R + e], tw[q * e * (NF / M)]);
        }
    }
}

// The inverse of `forward`, which multiplies by the conjugate twiddles and then computes R-point
// inverse DFTs.
template <uint R, uint M> inline void inverse(thread float2* v, uint t, device const float2* tw) {
    constexpr uint Q = M / R;
    for (uint j = 0; j < E / R; j++) {
        if (M > R) {
            uint q = (t + j * FFT_THREADS) % Q;
            for (uint e = 1; e < R; e++) v[j * R + e] = cmulc(v[j * R + e], tw[q * e * (NF / M)]);
        }
        idft<R>(v + j * R);
    }
}

// Move the values of a transform through buf, from the positions of stage (R0, M0) to the
// positions of stage (R1, M1).
template <uint R0, uint M0, uint R1, uint M1>
inline void exchange(thread float2* v, threadgroup float2* buf, uint off, uint t) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint j = 0; j < E / R0; j++)
        for (uint e = 0; e < R0; e++) buf[off + pos<R0, M0>(t, j, e)] = v[j * R0 + e];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint j = 0; j < E / R1; j++)
        for (uint e = 0; e < R1; e++) v[j * R1 + e] = buf[off + pos<R1, M1>(t, j, e)];
}

// Return the time position of value i of thread t at the first forward stage.
inline uint time_pos(uint t, uint i) {
    return pos<FIRST_RADIX, NF>(t, i / FIRST_RADIX, i % FIRST_RADIX);
}

// Return the position p of value i of thread t at the last forward stage, where the value is
// frequency rev(p).
inline uint freq_pos(uint t, uint i) {
    return pos<LAST_RADIX, LAST_RADIX>(t, i / LAST_RADIX, i % LAST_RADIX);
}

// Return the addend ad[index] of an output, or 0 without has_ad.
inline float addend(device const float* ad, long has_ad, long index) {
    return has_ad ? ad[index] : 0;
}

// The kernel parameters l3 and g, from which a kernel computes the indices of its thread.
#define FFT_IDS \
    uint3 l3 [[thread_position_in_threadgroup]], uint3 g [[threadgroup_position_in_grid]]
