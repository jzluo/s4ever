// powers.py compiles these kernels after the constants of `powers.library`.

constant constexpr uint S2 = 2 * NP;  // the padded state dimension

// The factors of mode n are m = 2 J Lambda, u = J P, q = conj(P) J and db = J B.
//
//     J = (2/dt - Lambda)^-1
//     s = 1 + sum_n ts    for ts = 2 Re(q P)
//     beta = sum_n tb    for tb = 2 Re(q B)
//     dB = 2 (J B - u beta / s)
//     c = 4 / (dt s)
//     v = 2 conj(q)
//     dA = I + blockdiag(m) - c u v^T
struct Mode {
    float2 m, u, q, db;
    float ts, tb;
};

// Return the factors of mode n of feature h. A lane past the last mode reads Lambda of mode N - 1,
// so that the load stays in bounds, and sets P, B and m to 0, which makes all its factors 0.
inline Mode mode(float d, device const float2* Lam, device const float2* P,
                 device const float2* B, long h, uint n, long N) {
    Mode f;
    bool live = n < N;
    long i = h * N + min(long(n), N - 1);
    float2 lam = Lam[i], p = live ? P[i] : float2(0), b = live ? B[i] : float2(0);
    float2 z = float2(2 / d - lam.x, -lam.y), J = conjugate(z) / dot(z, z);
    f.m = live ? 2 * cmul(J, lam) : float2(0);
    f.u = cmul(J, p);
    f.q = cmul(conjugate(p), J);
    f.db = cmul(J, b);
    f.ts = 2 * cmul(f.q, p).x;
    f.tb = 2 * cmul(f.q, b).x;
    return f;
}

// Store the squares Es[k] = dA^(2^k) - I (K, H, S, S) for S = 2 N. The parameters are dt (H) and
// the (H, N, 2) tensors Lambda, P and B, and a threadgroup handles one feature h.
kernel void squares(device float* Es, device const float* dt, device const float2* Lam,
                    device const float2* P, device const float2* B, constant long& H,
                    constant long& N, constant long& K, IDS) {
    threadgroup float E[S2 * S2];
    threadgroup float2 ms[NP], us[NP], vs[NP];
    threadgroup float c;
    long h = g;
    uint S = 2 * N;
    float d = dt[h];
    if (simd == 0) {
        Mode m = mode(d, Lam, P, B, h, lane, N);
        ms[lane] = m.m;
        us[lane] = m.u;
        vs[lane] = 2 * conjugate(m.q);
        float s = 1 + simd_sum(m.ts);
        if (lane == 0) c = 4 / (d * s);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // E = blockdiag(m) - c u v^T. The 2 x 2 block of mode n is ((m.x, -m.y), (m.y, m.x)) for
    // m = ms[n].
    for (uint i = t; i < S2 * S2; i += THREADS) {
        uint row = i / S2, col = i % S2, n = row / 2;
        float blk = n != col / 2         ? 0
                    : row % 2 == col % 2 ? ms[n].x
                    : row % 2            ? ms[n].y
                                         : -ms[n].y;
        E[i] = blk - c * us[n][row % 2] * vs[col / 2][col % 2];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (long k = 0; k < K; k++) {
        if (k > 0) {
            // E <- E^2 + 2 E, which is (I + E)^2 - I. The kernel adds 2 E after E^2, so that the
            // partial sums of E^2 are rounded relative to E^2 and not to the larger 2 E. The
            // constructor simdgroup_float8x8(2.0f) builds the diagonal matrix 2 I, while
            // make_filled_simdgroup_matrix fills every entry.
            simdgroup_float8x8 acc[8], x, y, two = simdgroup_float8x8(2.0f);
            // A SIMD group computes one row of 8 x 8 tiles. Tile j starts at row row0 and column
            // 8 j.
            ulong row0 = simd * 8;
            for (uint j = 0; j < 8; j++) {
                acc[j] = make_filled_simdgroup_matrix<float, 8, 8>(0);
                for (uint q = 0; q < 8; q++) {
                    simdgroup_load(x, E, S2, ulong2(q * 8, row0));
                    simdgroup_load(y, E, S2, ulong2(j * 8, q * 8));
                    simdgroup_multiply_accumulate(acc[j], x, y, acc[j]);
                }
                simdgroup_load(x, E, S2, ulong2(j * 8, row0));
                simdgroup_multiply_accumulate(acc[j], x, two, acc[j]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint j = 0; j < 8; j++) simdgroup_store(acc[j], E, S2, ulong2(j * 8, row0));
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        for (uint i = t; i < S * S; i += THREADS)
            Es[(k * H + h) * S * S + i] = E[i / S * S2 + i % S];
    }
}

// Store the Krylov block In[h, T - 1 - j] = dA^j dB (H, T, N) for j < T. Lane n holds mode n of x,
// and dA x = x + m x - c u (v . x), where v . x is the dot product of v and x as real vectors.
kernel void krylov(device float2* In, device const float* dt, device const float2* Lam,
                   device const float2* P, device const float2* B, constant long& N,
                   constant long& T, IDS) {
    long h = g;
    float d = dt[h];
    Mode m = mode(d, Lam, P, B, h, lane, N);
    float s = 1 + simd_sum(m.ts), beta = simd_sum(m.tb), c = 4 / (d * s);
    float2 v = 2 * conjugate(m.q), x = 2 * (m.db - m.u * beta / s);
    for (long j = 0; j < T; j++) {
        if (lane < N) In[(h * T + T - 1 - j) * N + lane] = x;
        float vx = simd_sum(dot(v, x));
        x += cmul(m.m, x) - c * vx * m.u;
    }
}
