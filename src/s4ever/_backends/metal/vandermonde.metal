// The kernels of vandermonde.py, compiled after discretize.metal and the constants of
// `vandermonde.library`.

// Write the convolution kernel k (C, H, L) from dt (H), A and B (H, N, 2), and Cm (C, H, N, 2). A
// tile covers the positions l0 to l0 + 8 T - 1 of one channel, and it is the product At Z, summed
// over the modes. Each SIMD group computes the tiles tile0 + simd, tile0 + simd + SIMDS, ... of
// the threadgroup's TILES tiles, and the modes run in steps of STEP_MODES, one per lane.
//     W = 2 C dB z^l0
//     row i of At = [Re, -Im] of W z^(T i)
//     column j of Z = [Re; Im] of z^j    for j < T
kernel void forward(device float* k, device const float* dt, device const float2* A,
                    device const float2* B, device const float2* Cm, constant long& H,
                    constant long& N, constant long& L, IDS) {
    // ws and dBs hold w = log z and dB of THREADS modes. Zi[i][j] = z^(T i) for mode n0 + j.
    threadgroup float2 ws[THREADS], dBs[THREADS], Zi[8][STEP_MODES];
    threadgroup float Z[2 * STEP_MODES][T];
    threadgroup float At[SIMDS][8][2 * STEP_MODES];
    uint tiles = uint(L + 8 * T - 1) / (8 * T), total_tiles = C * tiles;  // per channel and in all
    uint groups = (total_tiles + TILES - 1) / TILES, h = g / groups, tile0 = g % groups * TILES;
    auto tables = [&](uint first) {
        uint n = first + t;
        if (n < N) {
            float2 w, dB;
            discretize(dt[h], A[h * N + n], B[h * N + n], w, dB);
            ws[n % THREADS] = w;
            dBs[n % THREADS] = dB;
        }
    };
    tables(0);
    simdgroup_float8x8 Y[TILES / SIMDS][T / 8];  // Y[u] holds tile tile0 + simd + SIMDS u.
    for (uint u = 0; u < TILES / SIMDS; u++)
        for (uint p = 0; p < T / 8; p++) Y[u][p] = make_filled_simdgroup_matrix<float, 8, 8>(0);
    for (uint n0 = 0; n0 < N; n0 += STEP_MODES) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (n0 > 0 && n0 % THREADS == 0) {
            tables(n0);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        for (uint q = t; q < 8 * STEP_MODES; q += THREADS) {
            uint n = n0 + q % STEP_MODES;
            Zi[q / STEP_MODES][q % STEP_MODES] =
                n < N ? power(q / STEP_MODES * T, ws[n % THREADS]) : float2(0);
        }
        for (uint q = t; q < STEP_MODES * T; q += THREADS) {
            uint n = n0 + q / T;
            float2 z = n < N ? power(q % T, ws[n % THREADS]) : float2(0);
            Z[q / T][q % T] = z.x;
            Z[STEP_MODES + q / T][q % T] = z.y;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint u = 0; u < TILES / SIMDS; u++) {
            uint tile = tile0 + simd + SIMDS * u, n = n0 + lane;
            if (tile >= total_tiles) continue;  // Every lane of a SIMD group has the same tile.
            uint c = tile / tiles, l0 = tile % tiles * 8 * T;
            float2 W = 0;
            if (n < N)
                W = cmul(2 * cmul(Cm[(c * H + h) * N + n], dBs[n % THREADS]),
                         power(l0, ws[n % THREADS]));
            for (uint i = 0; i < 8; i++) {
                float2 v = cmul(W, Zi[i][lane]);
                At[simd][i][lane] = v.x;
                At[simd][i][STEP_MODES + lane] = -v.y;
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (uint ks = 0; ks < 2 * STEP_MODES / 8; ks++) {
                simdgroup_float8x8 lhs, rhs;
                simdgroup_load(lhs, &At[simd][0][8 * ks], 2 * STEP_MODES);
                for (uint p = 0; p < T / 8; p++) {
                    simdgroup_load(rhs, &Z[8 * ks][8 * p], T);
                    simdgroup_multiply_accumulate(Y[u][p], lhs, rhs, Y[u][p]);
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    // The 8 T outputs of a tile fit in the 16 STEP_MODES floats of At[simd].
    threadgroup float* o = At[simd][0];
    for (uint u = 0; u < TILES / SIMDS; u++) {
        uint tile = tile0 + simd + SIMDS * u;
        if (tile >= total_tiles) continue;
        uint c = tile / tiles, l0 = tile % tiles * 8 * T;
        for (uint p = 0; p < T / 8; p++) simdgroup_store(Y[u][p], o + 8 * p, T);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        device float* out = k + (c * H + h) * L;
        for (uint q = lane; q < 8 * T; q += 32) {
            uint l = l0 + q;
            if (l < L) out[l] = o[q];
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// Write S (C, H, chunks, N), the sums over each chunk of positions for the gradient gk (C, H, L)
// of k.
//     S0 = sum_l gk_l conj(z)^l
//     S1 = sum_l l gk_l conj(z)^l
// A threadgroup sums one chunk of one feature for GROUP_MODES modes and ROWS / segments channels,
// where segments = 1 << log_segments. It splits the chunk of each channel into that many segments
// of `span` positions, which form the ROWS rows of the matrix G. For the block of T positions at
// offset f in each segment, the product G V holds P0 and P1 in MODE_COLUMNS columns per mode, and
// G V D holds the same sums with j counted from the start of the segment. D is the same for all
// segments, because they share f.
//     P0 = sum_j g_j conj(z)^j
//     P1 = sum_j j g_j conj(z)^j
//     G V D = conj(z)^f (P0, f P0 + P1)
kernel void sums(device float4* S, device const float* gk, device const float* dt,
                 device const float2* A, device const float2* B, constant long& H,
                 constant long& N, constant long& L, constant long& span,
                 constant long& log_segments, constant long& chunks, IDS) {
    constexpr uint MODE_COLUMNS = 4;
    // Each SIMD group computes SIMD_TILES column tiles of the product G V.
    constexpr uint SIMD_TILES = MODE_COLUMNS * GROUP_MODES / 8 / SIMDS;
    constexpr uint THREAD_ENTRIES = ROWS * T / THREADS;  // entries of G that each thread stages
    threadgroup float2 ws[GROUP_MODES];
    threadgroup float V[T][MODE_COLUMNS * GROUP_MODES], G[ROWS][T];
    threadgroup float D[MODE_COLUMNS * GROUP_MODES / 8][64];  // the 8 x 8 block of each column tile
    uint segments = 1u << uint(log_segments), seg_len = uint(span);
    uint groups = (uint(N) + GROUP_MODES - 1) / GROUP_MODES, n0 = g % groups * GROUP_MODES;
    uint chunk_id = g / groups, chunk = chunk_id % uint(chunks);
    uint h = chunk_id / uint(chunks) % uint(H);
    uint c0 = chunk_id / uint(chunks) / uint(H) * (ROWS >> log_segments);
    uint l0 = chunk * seg_len << log_segments;
    // Return conj(z)^e for e < L, and 0 for the positions e >= L, where gk is padded with zeros.
    // The powers of a growing mode can overflow there, and 0 times an infinite power would be NaN.
    auto cpow = [&](uint e, float2 w) { return e < L ? conjugate(power(e, w)) : float2(0); };
    if (t < GROUP_MODES) {
        float2 w = 0, dB;
        if (n0 + t < N) discretize(dt[h], A[h * N + n0 + t], B[h * N + n0 + t], w, dB);
        ws[t] = w;
    }
    for (uint q = t; q < MODE_COLUMNS * GROUP_MODES / 8 * 64; q += THREADS) D[q / 64][q % 64] = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint q = t; q < T * GROUP_MODES; q += THREADS) {
        uint j = q / GROUP_MODES, m = q % GROUP_MODES;
        float2 y = cpow(j, ws[m]);
        threadgroup float* v = &V[j][MODE_COLUMNS * m];
        v[0] = y.x;
        v[1] = y.y;
        v[2] = j * y.x;
        v[3] = j * y.y;
    }
    // The thread's entry p of G reads the row gk_rows[p] of gk at the position starts[p] in
    // block 0, and next[p] holds its value for the next block. For rows past the last channel,
    // starts[p] is L, so that entry p is 0.
    device const float* gk_rows[THREAD_ENTRIES];
    uint starts[THREAD_ENTRIES];
    float next[THREAD_ENTRIES];
    for (uint p = 0; p < THREAD_ENTRIES; p++) {
        uint q = t + THREADS * p, c = c0 + (q / T >> log_segments);
        gk_rows[p] = gk + (min(c, C - 1) * H + h) * L;
        starts[p] = c < C ? l0 + (q / T & (segments - 1)) * seg_len + q % T : uint(L);
        next[p] = starts[p] < L ? gk_rows[p][starts[p]] : 0;
    }
    simdgroup_float8x8 acc[2][SIMD_TILES];  // [row tile][column tile]
    for (uint b = 0; b < 2; b++)
        for (uint col = 0; col < SIMD_TILES; col++)
            acc[b][col] = make_filled_simdgroup_matrix<float, 8, 8>(0);
    for (uint f = 0; f < seg_len; f += T) {
        for (uint p = 0; p < THREAD_ENTRIES; p++) G[(t + THREADS * p) / T][t % T] = next[p];
        for (uint p = 0; p < THREAD_ENTRIES; p++) {
            uint e = starts[p] + f + T;
            next[p] = e < L ? gk_rows[p][e] : 0;
        }
        if (t < GROUP_MODES) {
            // Thread t writes the 4 x 4 block of D for mode t, which maps (P0, P1) to
            // conj(z)^f (P0, f P0 + P1).
            float2 y = power(f, ws[t]);
            float wr = y.x, wi = -y.y, ff = float(f);
            threadgroup float* d = D[MODE_COLUMNS * t / 8] + 9 * (MODE_COLUMNS * t % 8);
            d[0] = wr, d[1] = wi, d[8] = -wi, d[9] = wr;
            d[2] = ff * wr, d[3] = ff * wi, d[10] = -ff * wi, d[11] = ff * wr;
            d[18] = wr, d[19] = wi, d[26] = -wi, d[27] = wr;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint b = 0; b < 2; b++) {
            simdgroup_float8x8 P[SIMD_TILES], lhs, rhs;
            for (uint col = 0; col < SIMD_TILES; col++)
                P[col] = make_filled_simdgroup_matrix<float, 8, 8>(0);
            for (uint ks = 0; ks < T / 8; ks++) {
                simdgroup_load(lhs, &G[8 * b][8 * ks], T);
                for (uint col = 0; col < SIMD_TILES; col++) {
                    simdgroup_load(rhs, &V[8 * ks][8 * (SIMD_TILES * simd + col)],
                                   MODE_COLUMNS * GROUP_MODES);
                    simdgroup_multiply_accumulate(P[col], lhs, rhs, P[col]);
                }
            }
            for (uint col = 0; col < SIMD_TILES; col++) {
                simdgroup_load(rhs, D[SIMD_TILES * simd + col], 8);
                simdgroup_multiply_accumulate(acc[b][col], P[col], rhs, acc[b][col]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // o holds the sums of G V D over the blocks as [ROWS][MODE_COLUMNS GROUP_MODES].
    threadgroup float* o = V[0];
    for (uint b = 0; b < 2; b++)
        for (uint col = 0; col < SIMD_TILES; col++)
            simdgroup_store(acc[b][col],
                            o + 8 * b * MODE_COLUMNS * GROUP_MODES + 8 * (SIMD_TILES * simd + col),
                            MODE_COLUMNS * GROUP_MODES);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Segment seg starts at position e, so with the positions counted from 0, its sums are
    // conj(z)^e (S0, S1 + e S0).
    for (uint q = t; q < segments * GROUP_MODES; q += THREADS) {
        uint seg = q / GROUP_MODES, m = q % GROUP_MODES, e = l0 + seg * seg_len;
        float2 w = cpow(e, ws[m]);
        for (uint i = seg; i < ROWS; i += segments) {
            threadgroup float* v = o + MODE_COLUMNS * (i * GROUP_MODES + m);
            float2 s0 = float2(v[0], v[1]), w0 = cmul(w, s0);
            float2 w1 = cmul(w, float2(v[2], v[3]) + float(e) * s0);
            v[2] = w1.x, v[3] = w1.y;
            v[0] = w0.x, v[1] = w0.y;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Add the segments of each channel and store S.
    for (uint q = t; q < (ROWS >> log_segments) * GROUP_MODES; q += THREADS) {
        uint m = q % GROUP_MODES, c = c0 + q / GROUP_MODES;
        if (c >= C || n0 + m >= N) continue;
        float4 s = 0;
        for (uint i = q / GROUP_MODES * segments; i < (q / GROUP_MODES + 1) * segments; i++) {
            threadgroup float* v = o + MODE_COLUMNS * (i * GROUP_MODES + m);
            s += float4(v[0], v[1], v[2], v[3]);
        }
        S[((c * H + h) * chunks + chunk) * N + n0 + m] = s;
    }
}

// Write the gradients of dt (H), A, B (H, N, 2) and Cm (C, H, N, 2) from the chunk sums S.
kernel void grads(device float* gdt, device float2* gA, device float2* gB, device float2* gC,
                  device const float4* S, device const float* dt, device const float2* A,
                  device const float2* B, device const float2* Cm, constant long& H,
                  constant long& N, constant long& chunks, IDS) {
    threadgroup float red[SIMDS];
    long h = g;
    float gdt_thread = 0;
    for (uint n = t; n < N; n += THREADS) {
        float2 w, dB, gw = 0, gb = 0, Am = A[h * N + n], Bm = B[h * N + n];
        discretize(dt[h], Am, Bm, w, dB);
        for (uint c = 0; c < C; c++) {
            float4 s = 0;
            for (long q = 0; q < chunks; q++) s += S[((c * H + h) * chunks + q) * N + n];
            // W = C dB, G_W = 2 S0 and G_log z = 2 conj(W) S1
            float2 Cn = Cm[(c * H + h) * N + n], gW = 2 * s.xy;
            gC[(c * H + h) * N + n] = cmulc(gW, dB);
            gb += cmulc(gW, Cn);
            gw += 2 * cmulc(s.zw, cmul(Cn, dB));
        }
        float2 gAm, gBm;
        float gdt_mode;
        discretize_grad(dt[h], Am, Bm, gw, gb, gAm, gBm, gdt_mode);
        gA[h * N + n] = gAm;
        gB[h * N + n] = gBm;
        gdt_thread += gdt_mode;
    }
    gdt_thread = simd_sum(gdt_thread);
    if (lane == 0) red[simd] = gdt_thread;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t == 0) {
        for (uint i = 1; i < SIMDS; i++) gdt_thread += red[i];
        gdt[h] = gdt_thread;
    }
}
