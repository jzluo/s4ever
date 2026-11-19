// chunk.py compiles these kernels after discretize.metal and the constants of `chunk.library`.
//
// A row block is 8 batch rows. Its states X (8, M) hold the (re, im) pairs of NP modes in M = 2 NP
// columns, where NP is N padded to the modes of the state tiles, which hold 4 modes each. The
// kernels split the state tiles into mode groups of GROUP_TILES tiles, which a threadgroup
// processes one after another.

constant constexpr uint M = 8 * STATE_TILES, NP = 4 * STATE_TILES;
constant constexpr uint STEP_TILES = T / 8;  // the tiles of 8 steps in a chunk
// A `backward` SIMD group holds BWD_TILES state tiles. KtT[c] has UPPER_TILES 8 x 8 tiles on and
// above its diagonal.
constant constexpr uint BWD_TILES = GROUP_TILES / BWD_SIMDS;
constant constexpr uint UPPER_TILES = STEP_TILES * (STEP_TILES + 1) / 2;
constant constexpr uint BWD_THREADS = 32 * BWD_SIMDS;
// The MATS_SIZE floats of a feature in mats are KtT (C, T, T), InT (T, M) and OutT (C, M, T),
// followed at MATS_Z by the powers z^T and z^rem. The PART_SIZE partials of a row block are dOutT
// (C, M, T), dInT (T, M) at PART_IN and dKtT (C, T, T) at PART_KT. The diagonal tiles of X^T Lam
// for the rotations by z^T and by z^rem follow at PART_DIAG. `chunk.layout` computes these offsets
// and sizes.

// Return the row and the mode of the entries that a lane holds in an 8 x 8 SIMD-group matrix
// whose columns are the (re, im) pairs of 4 modes. Metal does not document the layout of
// thread_elements(), in which a lane holds the columns 2 lane_mode(lane) and 2 lane_mode(lane) + 1
// of row lane_row(lane). MLX relies on the same layout (`BaseMMAFrag<T, 8, 8>::get_coord` in
// mlx/backend/metal/kernels/steel/gemm/mma.h). Because a lane holds both parts of one mode's state,
// it can rotate the state by z^T without reading the entries of other lanes.
inline uint lane_row(uint lane) {
    return ((lane >> 2) & 4) + ((lane >> 1) & 3);
}
inline uint lane_mode(uint lane) {
    return ((lane >> 3) & 1) * 2 + (lane & 1);
}
// Return the lane's two entries of m as one complex number.
inline thread float2& elems(thread simdgroup_float8x8& m) {
    return reinterpret_cast<thread float2&>(m.thread_elements());
}

inline simdgroup_float8x8 zeros() {
    return make_filled_simdgroup_matrix<float, 8, 8>(0);
}

// Store the matrices of feature g in mats (H, MATS_SIZE). D[c] is added to the tap K_c[0], and the
// last chunk has rem steps.
//
//     K_c[m] = 2 Re sum_n C[c, n] dB_n z_n^m
//     KtT[c][j][i] = K_c[i - j]    for i >= j
//     InT[j][2n] + i InT[j][2n + 1] = z_n^(T-1-j) dB_n
//     OutT[c][2n][j] - i OutT[c][2n + 1][j] = 2 C[c, n] z_n^(j+1)
kernel void matrices(device float* mats, device const float* dt, device const float2* A,
                     device const float2* B, device const float2* Cm, device const float* D,
                     constant long& H, constant long& N, constant long& L, constant long& has_D,
                     IDS) {
    threadgroup float2 ws[NP], dBs[NP];
    threadgroup float Ks[C][T];
    uint h = g, rem = uint(L) - (uint(L) - 1) / T * T;
    device float* KtT = mats + h * MATS_SIZE;
    device float* InT = KtT + C * T * T;
    device float* OutT = InT + T * M;
    device float2* zs = (device float2*)(KtT + MATS_Z);
    for (uint n = t; n < NP; n += THREADS) {
        float2 w = 0, dB = 0;
        if (n < N) discretize(dt[h], A[h * N + n], B[h * N + n], w, dB);
        ws[n] = w;
        dBs[n] = dB;
        zs[n] = power(T, w);
        zs[NP + n] = power(rem, w);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // The loop indexes the rows of InT forward and takes the exponent as T - 1 - j, because the
    // Metal compiler of macOS 26.5.1 to 27.2 drops some stores when the row index runs backward.
    // In a kernel of 128 threads where thread t stores float(m + 1) to
    // out[256 + (15 - m) * 32 + 2 * n] for m = t / 16 and n = t % 16, only the 16 threads with
    // m = 0 store.
    for (uint q = t; q < T * NP; q += THREADS) {
        uint j = q / NP, n = q % NP;
        float2 zdB = cmul(power(T - 1 - j, ws[n]), dBs[n]), zj1 = power(j + 1, ws[n]);
        InT[j * M + 2 * n] = zdB.x;
        InT[j * M + 2 * n + 1] = zdB.y;
        for (uint c = 0; c < C; c++) {
            float2 out = n < N ? 2 * cmul(Cm[(c * H + h) * N + n], zj1) : float2(0);
            OutT[(c * M + 2 * n) * T + j] = out.x;
            OutT[(c * M + 2 * n + 1) * T + j] = -out.y;
        }
    }
    for (uint q = t; q < C * T; q += THREADS) {
        uint c = q / T, m = q % T;
        float k = m == 0 && has_D ? D[c * H + h] : 0;
        for (uint n = 0; n < N; n++) {
            float2 Cn = Cm[(c * H + h) * N + n], zdB = cmul(power(m, ws[n]), dBs[n]);
            k += 2 * (Cn.x * zdB.x - Cn.y * zdB.y);
        }
        Ks[c][m] = k;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint q = t; q < C * T * T; q += THREADS) {
        uint c = q / (T * T), j = q / T % T, i = q % T;
        KtT[q] = i >= j ? Ks[c][i - j] : 0;
    }
}

// Set the offsets ofs of the `staged` = ceil(8 T / n) entries that thread i of n stages. The
// thread stages the entries i, i + n, ... of the 8 x T block of the rows r0 to r0 + 7 and the
// feature h. Because n is a multiple of T, all the entries of a thread are in column i % T.
// ofs[p] is the offset of the row of entry p in a (B, H, L) tensor, or -1 for an entry past the
// block or a row past Bn.
inline void row_offsets(thread long* ofs, uint staged, uint i, uint n, uint r0, uint h, long Bn,
                        long H, long L) {
    for (uint p = 0; p < staged; p++) {
        uint q = i + n * p, rr = r0 + q / T;
        ofs[p] = q < 8 * T && rr < Bn ? (long(rr) * H + h) * L : -1;
    }
}

// Scan u (B, H, L) into y (B, C, H, L).
//     has_x0: whether the scan starts from the states x0 (B, H, N, 2).
//     final: whether the final states are stored in xN (B, H, N, 2).
//     save: whether the states at the start of chunks 0, SEG, 2 SEG, ... are stored in
//         Xs (H, row blocks, segments, 8, M).
//
// A threadgroup scans FS row blocks of one feature. The FWD_SIMDS SIMD groups of a row block split
// the state tiles of a mode group, and each writes its part of the outputs to y_parts, which the
// threads of the row block sum into y.
kernel void forward(device YO* y, device float2* xN, device float* Xs, device const IO* u,
                    device const float2* x0, device const float* mats, constant long& H,
                    constant long& N, constant long& L, constant long& Bn, constant long& reverse,
                    constant long& has_x0, constant long& final, constant long& save, IDS) {
    constexpr uint FWD_TILES = GROUP_TILES / FWD_SIMDS, ROW_THREADS = 32 * FWD_SIMDS;
    threadgroup float Us[FS][8 * T], Ushift[FS][8 * T], y_parts[FS][FWD_SIMDS][C * 8 * T];
    // block is the thread's row block in the threadgroup, and block_simd and block_t are the
    // indices of its SIMD group and of the thread in that row block.
    uint block = simd / FWD_SIMDS, block_simd = simd % FWD_SIMDS, block_t = t % ROW_THREADS;
    // A feature has `groups` threadgroups and row_blocks row blocks.
    uint groups = (uint(Bn) + 8 * FS - 1) / (8 * FS), row_blocks = (uint(Bn) + 7) / 8;
    uint h = g / groups, r0 = g % groups * 8 * FS + 8 * block;
    uint chunks = (uint(L) + T - 1) / T, rem = uint(L) - (chunks - 1) * T;
    uint segments = (chunks + SEG - 1) / SEG;
    device const float* KtT = mats + h * MATS_SIZE;
    device const float* InT = KtT + C * T * T;
    device const float* OutT = InT + T * M;
    device const float2* zT = (device const float2*)(KtT + MATS_Z);
    device const float2* zr = zT + NP;
    uint fmode = lane_mode(lane), row = r0 + lane_row(lane);
    threadgroup float* us = Us[block];
    threadgroup float* ushift = Ushift[block];
    constexpr uint STAGED = (8 * T + ROW_THREADS - 1) / ROW_THREADS;
    uint col = block_t % T, Lu = uint(L);
    long ofs[STAGED], yofs[STAGED];
    row_offsets(ofs, STAGED, block_t, ROW_THREADS, r0, h, Bn, H, L);
    for (uint p = 0; p < STAGED; p++)
        yofs[p] = (long(r0 + (block_t + ROW_THREADS * p) / T) * C * H + h) * L;
    float next_u[STAGED];
    auto load = [&](uint k, uint shift) {
        uint s = k * T + col - shift, pos = reverse ? Lu - 1 - s : s;
        bool ok = col >= shift && s < Lu;
        for (uint p = 0; p < STAGED; p++)
            next_u[p] = ok && ofs[p] >= 0 ? float(u[ofs[p] + pos]) : 0;
    };
    // Only the SIMD groups of a row block share its threadgroup arrays, so one SIMD group per row
    // block needs only a SIMD-group barrier.
    auto sync = [&]() {
        if (FWD_SIMDS > 1) threadgroup_barrier(mem_flags::mem_threadgroup);
        else simdgroup_barrier(mem_flags::mem_threadgroup);
    };
    for (uint gi = 0; gi < STATE_TILES / GROUP_TILES; gi++) {
        // g0 is the first state tile of mode group gi. With one mode group, g0 is the constant 0,
        // so the compiler removes the branches on g0.
        const uint g0 = STATE_TILES == GROUP_TILES ? 0 : gi * GROUP_TILES;
        uint j0 = g0 + block_simd * FWD_TILES;  // the first state tile of the SIMD group
        simdgroup_float8x8 X[FWD_TILES];
        for (uint j = 0; j < FWD_TILES; j++) {
            uint n = 4 * (j0 + j) + fmode;
            elems(X[j]) = has_x0 && row < Bn && n < N ? x0[(row * H + h) * N + n] : float2(0);
        }
        load(0, 0);
        for (uint k = 0; k < chunks; k++) {
            bool last = k + 1 == chunks;
            sync();
            for (uint p = 0; p < STAGED; p++)
                if (block_t + ROW_THREADS * p < 8 * T) us[block_t + ROW_THREADS * p] = next_u[p];
            if (last) {
                // For the final state, the inputs of the last chunk are moved T - rem steps later,
                // so that they end at the last column.
                load(k, T - rem);
                for (uint p = 0; p < STAGED; p++)
                    if (block_t + ROW_THREADS * p < 8 * T)
                        ushift[block_t + ROW_THREADS * p] = next_u[p];
            } else {
                load(k + 1, 0);
            }
            sync();
            if (save && r0 < Bn && k % SEG == 0) {
                device float* xs =
                    Xs + ((long(h) * row_blocks + r0 / 8) * segments + k / SEG) * 8 * M;
                for (uint j = 0; j < FWD_TILES; j++) simdgroup_store(X[j], xs + 8 * (j0 + j), M);
            }
            // The first SIMD group of the first mode group adds U KtT, which includes D u.
            for (uint c = 0; c < C; c++)
                for (uint i = 0; i < STEP_TILES; i++) {
                    simdgroup_float8x8 acc = zeros(), a, b;
                    for (uint kk = 0; block_simd == 0 && g0 == 0 && kk <= i; kk++) {
                        simdgroup_load(a, us + 8 * kk, T);
                        simdgroup_load(b, KtT + (c * T + 8 * kk) * T + 8 * i, T);
                        simdgroup_multiply_accumulate(acc, a, b, acc);
                    }
                    for (uint j = 0; j < FWD_TILES; j++) {
                        simdgroup_load(b, OutT + (c * M + 8 * (j0 + j)) * T + 8 * i, T);
                        simdgroup_multiply_accumulate(acc, X[j], b, acc);
                    }
                    simdgroup_store(acc, y_parts[block][block_simd] + c * 8 * T + 8 * i, T);
                }
            if (!last || final) {
                threadgroup float* ub = last ? ushift : us;
                device const float2* z = last ? zr : zT;
                for (uint j = 0; j < FWD_TILES; j++) {
                    simdgroup_float8x8 acc, a, b;
                    elems(acc) = cmul(z[4 * (j0 + j) + fmode], elems(X[j]));
                    for (uint kk = 0; kk < STEP_TILES; kk++) {
                        simdgroup_load(a, ub + 8 * kk, T);
                        simdgroup_load(b, InT + 8 * kk * M + 8 * (j0 + j), M);
                        simdgroup_multiply_accumulate(acc, a, b, acc);
                    }
                    X[j] = acc;
                }
            }
            sync();
            uint s = k * T + col, pos = reverse ? Lu - 1 - s : s;
            for (uint p = 0; p < STAGED; p++) {
                uint q = block_t + ROW_THREADS * p;
                if (ofs[p] < 0 || s >= Lu) continue;
                for (uint c = 0; c < C; c++) {
                    float v = 0;
                    for (uint src_simd = 0; src_simd < FWD_SIMDS; src_simd++)
                        v += y_parts[block][src_simd][c * 8 * T + q];
                    long idx = yofs[p] + c * H * L + pos;
                    if (g0 > 0) v += float(y[idx]);
                    y[idx] = YO(v);
                }
            }
        }
        if (final && row < Bn)
            for (uint j = 0; j < FWD_TILES; j++) {
                uint n = 4 * (j0 + j) + fmode;
                if (n < N) xN[(row * H + h) * N + n] = elems(X[j]);
            }
    }
}

// Compute du (B, H, L) from gy (B, C, H, L).
//     has_gf: whether gf (B, H, N, 2) holds the gradient of the final states.
//     has_dx0: whether the gradient of x0 is stored in dx0 (B, H, N, 2).
//     part: the partials (H, row blocks, PART_SIZE) of each row block.
//     scratch: the recomputed states (H, row blocks, SEG, 8, M) of a segment.
//
// A threadgroup handles one row block of one feature, and its SIMD groups split the state tiles of
// a mode group. The kernel walks the chunks backward. G is chunk k of gy, and Lam is the adjoint of
// the state after chunk k, which the update makes the adjoint of the state before chunk k.
//
//     du_k = G KtT^T + Lam InT^T
//     Lam <- conj(z^T) Lam + G OutT^T
//
// The forward saves the states only at the start of each segment of SEG chunks, so the kernel
// recomputes the states of a segment into scratch before it walks the segment's chunks.
kernel void backward(device YO* du, device float2* dx0, device float* part, device float* scratch,
                     device const IO* u, device const IO* gy, device const float2* gf,
                     device const float* Xs, device const float* mats, constant long& H,
                     constant long& N, constant long& L, constant long& Bn, constant long& reverse,
                     constant long& has_gf, constant long& has_dx0, IDS) {
    // useg holds the inputs of each chunk of the segment, and gs holds G. du_parts[s] holds the
    // part of du_k from SIMD group s, and gkt holds G KtT^T in the last chunk.
    threadgroup float useg[SEG][8 * T], ushift[8 * T], gs[C * 8 * T];
    threadgroup float du_parts[BWD_SIMDS][8 * T], gkt[8 * T];
    uint row_blocks = (uint(Bn) + 7) / 8, h = g / row_blocks, r0 = g % row_blocks * 8;
    uint chunks = (uint(L) + T - 1) / T, rem = uint(L) - (chunks - 1) * T;
    uint segments = (chunks + SEG - 1) / SEG;
    device const float* KtT = mats + h * MATS_SIZE;
    device const float2* zT = (device const float2*)(KtT + MATS_Z);
    device const float* InT = KtT + C * T * T;
    device const float* OutT = InT + T * M;
    device const float2* zr = zT + NP;
    uint fmode = lane_mode(lane), row = r0 + lane_row(lane);
    constexpr uint STAGED = (8 * T + BWD_THREADS - 1) / BWD_THREADS;
    uint col = t % T, Lu = uint(L);
    long ofs[STAGED], gofs[STAGED];
    row_offsets(ofs, STAGED, t, BWD_THREADS, r0, h, Bn, H, L);
    for (uint p = 0; p < STAGED; p++)
        gofs[p] = (long(r0 + (t + BWD_THREADS * p) / T) * C * H + h) * L;
    auto load = [&](device const IO* src, thread long* offsets, uint k, uint shift, uint p) {
        uint s = k * T + col - shift, pos = reverse ? Lu - 1 - s : s;
        bool ok = col >= shift && s < Lu && ofs[p] >= 0;
        return ok ? float(src[offsets[p] + pos]) : 0.f;
    };
    // dOutT, dInT and dKtT sum the partials over the chunks. XtLam sums the diagonal tiles of
    // X^T Lam over the chunks before the last.
    simdgroup_float8x8 Lam[BWD_TILES], dOutT[C][BWD_TILES][STEP_TILES], dInT[STEP_TILES][BWD_TILES];
    simdgroup_float8x8 dKtT[C][UPPER_TILES], XtLam[BWD_TILES];
    device float* block_part = part + g * PART_SIZE;
    device float* xseg = scratch + g * SEG * 8 * M;
    // next_u and next_g hold the thread's entries of u for the next segment and of gy for the
    // next chunk.
    float next_u[SEG][STAGED], next_g[C][STAGED];
    auto load_seg = [&](uint segment) {
        for (uint kseg = 0; kseg < SEG; kseg++)
            for (uint p = 0; p < STAGED; p++)
                next_u[kseg][p] = load(u, ofs, segment * SEG + kseg, 0, p);
    };
    auto load_g = [&](uint k) {
        for (uint c = 0; c < C; c++)
            for (uint p = 0; p < STAGED; p++) next_g[c][p] = load(gy + c * H * L, gofs, k, 0, p);
    };
    for (uint gi = 0; gi < STATE_TILES / GROUP_TILES; gi++) {
        const uint g0 = STATE_TILES == GROUP_TILES ? 0 : gi * GROUP_TILES;
        uint j0 = g0 + simd * BWD_TILES;
        for (uint j = 0; j < BWD_TILES; j++) {
            uint n = 4 * (j0 + j) + fmode;
            elems(Lam[j]) = has_gf && row < Bn && n < N ? gf[(row * H + h) * N + n] : float2(0);
            XtLam[j] = zeros();
            for (uint i = 0; i < STEP_TILES; i++) {
                dInT[i][j] = zeros();
                for (uint c = 0; c < C; c++) dOutT[c][j][i] = zeros();
            }
        }
        for (uint c = 0; c < C; c++)
            for (uint q = 0; q < UPPER_TILES; q++) dKtT[c][q] = zeros();
        load_seg(segments - 1);
        load_g(chunks - 1);
        for (uint segment = segments; segment-- > 0;) {
            // The segment runs from chunk k0 to the chunk before k1.
            uint k0 = segment * SEG, k1 = min(k0 + SEG, chunks);
            for (uint p = 0; p < STAGED; p++) {
                uint q = t + BWD_THREADS * p;
                if (q >= 8 * T) break;
                for (uint kseg = 0; kseg < SEG; kseg++) useg[kseg][q] = next_u[kseg][p];
                if (k1 == chunks) ushift[q] = load(u, ofs, chunks - 1, T - rem, p);
            }
            if (segment > 0) load_seg(segment - 1);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            device const float* xs =
                Xs + ((long(h) * row_blocks + r0 / 8) * segments + segment) * 8 * M;
            simdgroup_float8x8 X[BWD_TILES];
            for (uint j = 0; j < BWD_TILES; j++) simdgroup_load(X[j], xs + 8 * (j0 + j), M);
            for (uint k = k0; k < k1; k++) {
                for (uint j = 0; j < BWD_TILES; j++)
                    simdgroup_store(X[j], xseg + (k - k0) * 8 * M + 8 * (j0 + j), M);
                if (k + 1 == k1) break;
                for (uint j = 0; j < BWD_TILES; j++) {
                    simdgroup_float8x8 acc, a, b;
                    elems(acc) = cmul(zT[4 * (j0 + j) + fmode], elems(X[j]));
                    for (uint kk = 0; kk < STEP_TILES; kk++) {
                        simdgroup_load(a, useg[k - k0] + 8 * kk, T);
                        simdgroup_load(b, InT + 8 * kk * M + 8 * (j0 + j), M);
                        simdgroup_multiply_accumulate(acc, a, b, acc);
                    }
                    X[j] = acc;
                }
            }
            simdgroup_barrier(mem_flags::mem_device);
            for (uint k = k1; k-- > k0;) {
                bool last = k + 1 == chunks;
                for (uint p = 0; p < STAGED; p++) {
                    uint q = t + BWD_THREADS * p;
                    if (q < 8 * T)
                        for (uint c = 0; c < C; c++) gs[c * 8 * T + q] = next_g[c][p];
                }
                if (k > 0) load_g(k - 1);
                threadgroup_barrier(mem_flags::mem_threadgroup);
                threadgroup float* us = useg[k - k0];
                device const float* xk = xseg + (k - k0) * 8 * M;  // the states at the start of k
                // The last SIMD group adds G KtT^T, and SIMD group 0 adds U^T G to dKtT below.
                // `chunk.tiles` gives the backward at least 2 SIMD groups, so these are different
                // SIMD groups. In the last chunk, G KtT^T goes to gkt, because Lam InT^T is then
                // the gradient of the inputs that were moved to the end of the chunk for the
                // final state.
                for (uint i = 0; i < STEP_TILES; i++) {
                    simdgroup_float8x8 acc = zeros(), a, b;
                    if (simd == BWD_SIMDS - 1 && g0 == 0) {
                        for (uint c = 0; c < C; c++)
                            for (uint i2 = i; i2 < STEP_TILES; i2++) {
                                simdgroup_load(a, gs + c * 8 * T + 8 * i2, T);
                                simdgroup_load(b, KtT + (c * T + 8 * i) * T + 8 * i2, T, ulong2(0),
                                               true);
                                simdgroup_multiply_accumulate(acc, a, b, acc);
                            }
                        if (last) {
                            simdgroup_store(acc, gkt + 8 * i, T);
                            acc = zeros();
                        }
                    }
                    for (uint j = 0; j < BWD_TILES; j++) {
                        simdgroup_load(b, InT + 8 * i * M + 8 * (j0 + j), M, ulong2(0), true);
                        simdgroup_multiply_accumulate(acc, Lam[j], b, acc);
                    }
                    simdgroup_store(acc, du_parts[simd] + 8 * i, T);
                }
                // dInT += U^T Lam
                threadgroup float* ub = last ? ushift : us;
                for (uint i = 0; i < STEP_TILES; i++) {
                    simdgroup_float8x8 a;
                    simdgroup_load(a, ub + 8 * i, T, ulong2(0), true);
                    for (uint j = 0; j < BWD_TILES; j++)
                        simdgroup_multiply_accumulate(dInT[i][j], a, Lam[j], dInT[i][j]);
                }
                // dOutT += X^T G. The diagonal tiles of X^T Lam go to XtLam for the rotations by
                // z^T, and in the last chunk to the partials for the rotation by z^rem.
                for (uint j = 0; j < BWD_TILES; j++) {
                    simdgroup_float8x8 a, b;
                    simdgroup_load(a, xk + 8 * (j0 + j), M, ulong2(0), true);
                    if (last) {
                        b = zeros();
                        simdgroup_multiply_accumulate(b, a, Lam[j], b);
                        simdgroup_store(b, block_part + PART_DIAG + 8 * M + 64 * (j0 + j), 8);
                    } else {
                        simdgroup_multiply_accumulate(XtLam[j], a, Lam[j], XtLam[j]);
                    }
                    for (uint c = 0; c < C; c++)
                        for (uint i = 0; i < STEP_TILES; i++) {
                            simdgroup_load(b, gs + c * 8 * T + 8 * i, T);
                            simdgroup_multiply_accumulate(dOutT[c][j][i], a, b, dOutT[c][j][i]);
                        }
                }
                // dKtT += U^T G, on and above the diagonal
                if (simd == 0 && g0 == 0)
                    for (uint ia = 0, q = 0; ia < STEP_TILES; ia++) {
                        simdgroup_float8x8 a, b;
                        simdgroup_load(a, us + 8 * ia, T, ulong2(0), true);
                        for (uint ic = ia; ic < STEP_TILES; ic++, q++)
                            for (uint c = 0; c < C; c++) {
                                simdgroup_load(b, gs + c * 8 * T + 8 * ic, T);
                                simdgroup_multiply_accumulate(dKtT[c][q], a, b, dKtT[c][q]);
                            }
                    }
                // Lam <- conj(z^T) Lam + G OutT^T
                device const float2* z = last ? zr : zT;
                for (uint j = 0; j < BWD_TILES; j++) {
                    simdgroup_float8x8 acc, a, b;
                    elems(acc) = cmulc(elems(Lam[j]), z[4 * (j0 + j) + fmode]);
                    for (uint c = 0; c < C; c++)
                        for (uint i = 0; i < STEP_TILES; i++) {
                            simdgroup_load(a, gs + c * 8 * T + 8 * i, T);
                            simdgroup_load(b, OutT + (c * M + 8 * (j0 + j)) * T + 8 * i, T,
                                           ulong2(0), true);
                            simdgroup_multiply_accumulate(acc, a, b, acc);
                        }
                    Lam[j] = acc;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                // In the last chunk, step col of the chunk is column col + T - rem of du_parts.
                uint s = k * T + col, pos = reverse ? Lu - 1 - s : s;
                for (uint p = 0; p < STAGED; p++) {
                    uint q = t + BWD_THREADS * p;
                    if (ofs[p] < 0 || s >= Lu) continue;
                    float v = last && g0 == 0 ? gkt[q] : 0;
                    for (uint src_simd = 0; src_simd < BWD_SIMDS; src_simd++)
                        v += !last ? du_parts[src_simd][q] : du_parts[src_simd][q + T - rem];
                    long idx = ofs[p] + pos;
                    if (g0 > 0) v += float(du[idx]);
                    du[idx] = YO(v);
                }
            }
        }
        if (has_dx0 && row < Bn)
            for (uint j = 0; j < BWD_TILES; j++) {
                uint n = 4 * (j0 + j) + fmode;
                if (n < N) dx0[(row * H + h) * N + n] = elems(Lam[j]);
            }
        for (uint j = 0; j < BWD_TILES; j++) {
            for (uint c = 0; c < C; c++)
                for (uint i = 0; i < STEP_TILES; i++)
                    simdgroup_store(dOutT[c][j][i], block_part + (c * M + 8 * (j0 + j)) * T + 8 * i,
                                    T);
            for (uint i = 0; i < STEP_TILES; i++)
                simdgroup_store(dInT[i][j], block_part + PART_IN + 8 * i * M + 8 * (j0 + j), M);
            simdgroup_store(XtLam[j], block_part + PART_DIAG + 64 * (j0 + j), 8);
        }
        if (simd == 0 && g0 == 0)
            for (uint c = 0; c < C; c++)
                for (uint ia = 0, q = 0; ia < STEP_TILES; ia++)
                    for (uint ic = ia; ic < STEP_TILES; ic++, q++)
                        simdgroup_store(dKtT[c][q],
                                        block_part + PART_KT + (c * T + 8 * ia) * T + 8 * ic, T);
    }
}

// Compute the gradients of dt, A, B, C and D of feature g from the partials of its R row blocks,
// which the kernel first sums into the partials of the first row block. gw, gb and gc are the
// gradients of log dA, dB and C of mode n.
kernel void grads(device float* gdt, device float2* gA, device float2* gB, device float2* gC,
                  device float* gD, device float* part, device const float* dt,
                  device const float2* A, device const float2* B, device const float2* Cm,
                  constant long& H, constant long& N, constant long& L, constant long& R,
                  constant long& has_D, IDS) {
    threadgroup float dK[C][T], gdts[THREADS / 32];  // gdts holds the sum of each SIMD group
    uint h = g, rem = uint(L) - (uint(L) - 1) / T * T;
    device float* sums = part + h * R * PART_SIZE;
    for (uint q = t; q < PART_SIZE; q += THREADS) {
        float v = sums[q];
        for (uint r = 1; r < R; r++) v += sums[r * PART_SIZE + q];
        sums[q] = v;
    }
    threadgroup_barrier(mem_flags::mem_device);
    // The gradients of the taps are the diagonal sums of dKtT.
    for (uint q = t; q < C * T; q += THREADS) {
        uint c = q / T, m = q % T;
        float v = 0;
        for (uint j = 0; j + m < T; j++) v += sums[PART_KT + (c * T + j) * T + j + m];
        dK[c][m] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (has_D && t < C) gD[t * H + h] = dK[t][0];
    float gdt_sum = 0;
    for (uint n = t; n < N; n += THREADS) {
        float2 Am = A[h * N + n], Bm = B[h * N + n], w, dB, gb = 0, gw = 0, gc[C];
        discretize(dt[h], Am, Bm, w, dB);
        // lT and lr are the values of conj(x) Lam for the rotations by z^T and by z^rem, which
        // the kernel sums from the 2 x 2 block of mode n in the diagonal tiles of X^T Lam.
        device const float* diag = sums + PART_DIAG + 64 * (n / 4) + 18 * (n % 4);
        float2 lT = float2(diag[0] + diag[9], diag[1] - diag[8]);
        float2 lr = float2(diag[8 * M] + diag[8 * M + 9], diag[8 * M + 1] - diag[8 * M + 8]);
        gw += cmulc(lT, float(T) * power(T, w)) + cmulc(lr, float(rem) * power(rem, w));
        for (uint c = 0; c < C; c++) gc[c] = 0;
        for (uint m = 0; m < T; m++) {
            // InT[T - 1 - m] = z^m dB
            float2 zm = power(m, w), zm1 = power(m + 1, w);
            device const float* din = sums + PART_IN + (T - 1 - m) * M + 2 * n;
            float2 li = float2(din[0], din[1]);
            gb += cmulc(li, zm);
            gw += cmulc(li, float(m) * cmul(zm, dB));
            for (uint c = 0; c < C; c++) {
                // OutT[c][., m] = 2 C z^(m+1), and the tap K_c[m] is 2 Re(C dB z^m).
                device const float* dout = sums + (c * M + 2 * n) * T + m;
                float2 Cn = Cm[(c * H + h) * N + n], lo = float2(dout[0], -dout[T]);
                float2 lk = 2 * dK[c][m] * conjugate(zm);
                gc[c] += cmulc(lo, 2 * zm1) + cmulc(lk, dB);
                gb += cmulc(lk, Cn);
                gw += cmulc(lo, float(2 * m + 2) * cmul(Cn, zm1));
                gw += cmulc(lk, float(m) * cmul(Cn, dB));
            }
        }
        for (uint c = 0; c < C; c++) gC[(c * H + h) * N + n] = gc[c];
        float2 gAm, gBm;
        float gdtm;
        discretize_grad(dt[h], Am, Bm, gw, gb, gAm, gBm, gdtm);
        gA[h * N + n] = gAm;
        gB[h * N + n] = gBm;
        gdt_sum += gdtm;
    }
    gdt_sum = simd_sum(gdt_sum);
    if (lane == 0) gdts[simd] = gdt_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t == 0) {
        for (uint i = 1; i < THREADS / 32; i++) gdt_sum += gdts[i];
        gdt[h] = gdt_sum;
    }
}
