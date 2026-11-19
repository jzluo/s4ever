// recurrent.py compiles these scans after discretize.metal and the constants of
// `recurrent.library`. A SIMD group scans the batch row b of the feature h, and lane j holds the
// states of the modes j + 32 m for m < LANE_MODES. A scan runs in blocks of 32 steps. The steps of
// a block do not access device memory, because lane j loads the input of step j before the block
// and stores the outputs yl of step j after it.

// The kernel parameters that follow the parameters of each scan.
#define ARGS \
    device const float* D, device const float2* x0, constant long& Bn, constant long& H, \
    constant long& N, constant long& L, constant long& has_D, constant long& has_x0, \
    constant long& reverse

// Return the position in a sequence of L steps that a scan reads at step s. With `reverse`, the
// scan starts at the last position.
inline long position(long s, long L, long reverse) {
    return reverse ? L - 1 - s : s;
}

// Load the states x of the lane's modes from row `row` of x0 (B H, N) if has_x0, and set them to
// 0 otherwise.
inline void load_states(thread float2* x, device const float2* x0, long row, long N, long has_x0,
                        uint lane) {
    for (uint m = 0; m < LANE_MODES; m++) {
        uint n = lane + 32 * m;
        x[m] = has_x0 && n < N ? x0[row * N + n] : float2(0);
    }
}

// Store the states x of the lane's modes in row `row` of xN (B H, N).
inline void store_states(device float2* xN, thread float2* x, long row, long N, uint lane) {
    for (uint m = 0; m < LANE_MODES; m++) {
        uint n = lane + 32 * m;
        if (n < N) xN[row * N + n] = x[m];
    }
}

// Set the outputs yl of a step to scale times the sum of acc over the lanes, plus D u, in the lane
// that holds the step. The `s4d` kernel passes scale = 2, because its output is
// y = 2 Re(C x) + D u.
inline void set_outputs(thread float* yl, thread float* acc, float scale, thread float* Ds,
                        float us, bool holds_step) {
    for (uint c = 0; c < C; c++) {
        float o = scale * simd_sum(acc[c]) + Ds[c] * us;
        if (holds_step) yl[c] = o;
    }
}

// Scan S4D into the outputs y (B, L, C, H) and the final states xN (B, H, N, 2). The parameters
// are dt (H), the (H, N, 2) tensors A and B, Cm (C, H, N, 2) and D (C, H) if has_D.
kernel void s4d(device IO* y, device float2* xN, device const IO* u,
                device const float* dt, device const float2* A, device const float2* B,
                device const float2* Cm, ARGS, IDS) {
    // zs, dBs and cs hold dA, dB and C of each mode.
    threadgroup float2 zs[32 * LANE_MODES], dBs[32 * LANE_MODES], cs[C][32 * LANE_MODES];
    uint blocks = uint(Bn + ROWS - 1) / ROWS, h = g / blocks;
    long b = g % blocks * ROWS + simd;
    device const IO* ur = u + b * L * H + h;
    for (uint n = t; n < 32 * LANE_MODES; n += 32 * ROWS) {
        float2 w = 0, dB = 0;
        if (n < N) discretize(dt[h], A[h * N + n], B[h * N + n], w, dB);
        zs[n] = n < N ? precise::exp(w.x) * float2(precise::cos(w.y), precise::sin(w.y)) : 0;
        dBs[n] = dB;
        for (uint c = 0; c < C; c++) cs[c][n] = n < N ? Cm[(c * H + h) * N + n] : 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (b >= Bn) return;
    float2 x[LANE_MODES];
    float Ds[C];
    load_states(x, x0, b * H + h, N, has_x0, lane);
    for (uint c = 0; c < C; c++) Ds[c] = has_D ? D[c * H + h] : 0;
    for (long s0 = 0; s0 < L; s0 += 32) {
        // has_step is whether the lane has a step in the block, and pos is the step's position.
        bool has_step = s0 + lane < L;
        long pos = position(s0 + lane, L, reverse);
        float uj = has_step ? ur[pos * H] : 0;
        float yl[C];
        for (uint j = 0; j < 32 && s0 + j < L; j++) {
            float us = simd_broadcast(uj, ushort(j));
            float acc[C] = {};
            for (uint m = 0; m < LANE_MODES; m++) {
                uint n = lane + 32 * m;
                x[m] = cmul(zs[n], x[m]) + dBs[n] * us;
                for (uint c = 0; c < C; c++) acc[c] += cs[c][n].x * x[m].x - cs[c][n].y * x[m].y;
            }
            set_outputs(yl, acc, 2, Ds, us, lane == j);
        }
        if (has_step)
            for (uint c = 0; c < C; c++) y[((b * L + pos) * C + c) * H + h] = IO(yl[c]);
    }
    store_states(xN, x, b * H + h, N, lane);
}

// Scan S4 into the outputs y (B, L, C, H) and the final states xN (B, H, N, 2). The parameters are
// dt (H) and the (H, N, 2) tensors Lambda, P and B. The vector cm (C, H, N, 2) is 2 conj(c) for the
// readout c. A step is x' = lam x - g (w . x) + dB u with the factors of `ops._s4.factors`.
kernel void s4(device IO* y, device float2* xN, device const IO* u, device const float* dt,
               device const float2* Lam, device const float2* P, device const float2* B,
               device const float2* cm, ARGS, IDS) {
    // zs, ws, es and fs hold lam, w, E P and E B of each mode, and ts holds the mode's terms of
    // w . P and w . B.
    threadgroup float2 zs[32 * LANE_MODES], ws[32 * LANE_MODES], es[32 * LANE_MODES],
        fs[32 * LANE_MODES], ts[32 * LANE_MODES], cs[C][32 * LANE_MODES];
    uint blocks = uint(Bn + ROWS - 1) / ROWS, h = g / blocks;
    long b = g % blocks * ROWS + simd;
    device const IO* ur = u + b * L * H + h;
    float dth = dt[h];
    for (uint n = t; n < 32 * LANE_MODES; n += 32 * ROWS) {
        bool live = n < N;
        float2 lambda = live ? Lam[h * N + n] : 0, p = live ? P[h * N + n] : 0,
               Bm = live ? B[h * N + n] : 0;
        for (uint c = 0; c < C; c++) cs[c][n] = live ? cm[(c * H + h) * N + n] : 0;
        // a = dt Lambda / 2 and E = (1 - a)^-1, so that (2/dt - Lambda)^-1 = E dt / 2
        float2 a = dth / 2 * lambda, e = float2(1 - a.x, a.y) / ((1 - a.x) * (1 - a.x) + a.y * a.y);
        // The factor lam = 1 + 2 a E equals 2 E - 1. The kernel takes the form that adds the
        // smaller term to +-1, because the rounding error of lam compounds over the steps.
        zs[n] = dot(a, a) < 1 ? float2(1, 0) + 2 * cmul(a, e) : 2 * e - float2(1, 0);
        float2 w = dth * cmulc(p, e);
        ws[n] = w;
        es[n] = cmul(e, p);
        fs[n] = cmul(e, Bm);
        ts[n] = float2(dot(w, p), dot(w, Bm));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (b >= Bn) return;
    // s = 1 + w . P, g = 2 E P / s and dB = dt (E B - E P (w . B) / s)
    float2 sums = 0, gm[LANE_MODES], dBm[LANE_MODES];
    for (uint m = 0; m < LANE_MODES; m++) sums += ts[lane + 32 * m];
    float inv_s = 1 / (1 + simd_sum(sums.x)), wb = simd_sum(sums.y);
    for (uint m = 0; m < LANE_MODES; m++) {
        uint n = lane + 32 * m;
        gm[m] = 2 * inv_s * es[n];
        dBm[m] = dth * (fs[n] - es[n] * wb * inv_s);
    }
    float2 x[LANE_MODES];
    float Ds[C], wn = 0;
    load_states(x, x0, b * H + h, N, has_x0, lane);
    for (uint c = 0; c < C; c++) Ds[c] = has_D ? D[c * H + h] : 0;
    for (uint m = 0; m < LANE_MODES; m++) wn += dot(ws[lane + 32 * m], x[m]);
    for (long s0 = 0; s0 < L; s0 += 32) {
        // has_step is whether the lane has a step in the block, and pos is the step's position.
        bool has_step = s0 + lane < L;
        long pos = position(s0 + lane, L, reverse);
        float uj = has_step ? ur[pos * H] : 0;
        float yl[C];
        for (uint j = 0; j < 32 && s0 + j < L; j++) {
            // The sum cm . x' is 2 Re(c x').
            float us = simd_broadcast(uj, ushort(j)), wx = simd_sum(wn);
            float acc[C] = {};
            wn = 0;
            for (uint m = 0; m < LANE_MODES; m++) {
                uint n = lane + 32 * m;
                x[m] = cmul(zs[n], x[m]) - gm[m] * wx + dBm[m] * us;
                wn += dot(ws[n], x[m]);
                for (uint c = 0; c < C; c++) acc[c] += dot(cs[c][n], x[m]);
            }
            set_outputs(yl, acc, 1, Ds, us, lane == j);
        }
        if (has_step)
            for (uint c = 0; c < C; c++) y[((b * L + pos) * C + c) * H + h] = IO(yl[c]);
    }
    store_states(xN, x, b * H + h, N, lane);
}
