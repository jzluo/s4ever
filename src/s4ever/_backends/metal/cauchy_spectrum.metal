// The kernels of cauchy.py, which use the method that `tilelang.kernels.cauchy` describes. They are
// compiled after the constants THREADS, N and C of `spectrum_library`. In `spectrum` and
// `spectrum_grads`, each thread handles one frequency of a tile of THREADS frequencies.

// The resolvent r_XY of the pair (X_x, Y_y) has index x (C + 1) + y, for X = (B, P),
// Y = (C..., Q) and Q = conj(P), so r[BQ] is r_BQ and r[PQ] is r_PQ. Each pair has the two Cauchy
// coefficients coef_0 and coef_1.
constant constexpr uint PAIRS = 2 * (C + 1), COEFS = 2 * PAIRS, BQ = C, PQ = 2 * C + 1;
constant constexpr uint SIMDS = THREADS / 32;  // SIMD groups per threadgroup
// The tables of the Cauchy sums hold one chunk of the modes at a time. The modes form CHUNKS chunks
// of CHUNK_MODES <= THREADS modes, so that each thread loads at most one mode of a chunk.
constant constexpr uint CHUNKS = (N + THREADS - 1) / THREADS;
constant constexpr uint CHUNK_MODES = (N + CHUNKS - 1) / CHUNKS;
// `others` adds the terms of r_XY(0) in blocks of BLOCK_MODES modes, so that it takes about
// N / BLOCK_MODES + BLOCK_MODES steps per mode instead of N.
constant constexpr uint BLOCK_MODES = 16, BLOCKS = (N + BLOCK_MODES - 1) / BLOCK_MODES;

// Return nu = 2 tan(pi k / L) for 0 <= k <= L / 2. For k > L / 4, the function evaluates
// 2 cot(pi (L - 2 k) / (2 L)) instead, so that the angle stays at most pi / 4 and each argument of
// sinpi and cospi is computed with one rounding.
inline float frequency(long k, long L) {
    float x = float(k) / float(L), z = float(L - 2 * k) / float(2 * L);
    return 4 * k <= L ? 2 * precise::sinpi(x) / precise::cospi(x)
                      : 2 * precise::cospi(z) / precise::sinpi(z);
}

// The parameters of one mode, with u[x (C + 1) + y] = X_x Y_y.
struct Mode {
    float2 lam, X[2], Y[C + 1], u[PAIRS];
};

inline Mode load_mode(device const float2* Lam, device const float2* P, device const float2* B,
                      device const float2* Cm, long H, long h, uint n) {
    Mode mode;
    long i = h * N + n;
    mode.lam = Lam[i];
    mode.X[0] = B[i];
    mode.X[1] = P[i];
    for (uint c = 0; c < C; c++) mode.Y[c] = Cm[c * H * N + i];
    mode.Y[C] = conjugate(P[i]);
    for (uint x = 0; x < 2; x++)
        for (uint y = 0; y <= C; y++) mode.u[x * (C + 1) + y] = cmul(mode.X[x], mode.Y[y]);
    return mode;
}

// Load the tables of the chunk that starts at mode m0, whose entry n holds mode i = m0 + n. aw[n]
// holds a = Re(w) and |w|^2 for the pole w, and cs[n] holds the coefficients.
//     w = dt Lambda
//     v = dt u
//     coef_0 = -2 Re(v conj(w))
//     coef_1 = 2 Re(v)
//     cs[n][p PAIRS + j] = coef_p[j, i]
// The entries past the last mode have zero coefficients, so they add nothing to the sums.
inline void tables(threadgroup float (&cs)[CHUNK_MODES][COEFS],
                   threadgroup float2 (&aw)[CHUNK_MODES], device const float* dt,
                   device const float2* Lam, device const float2* P, device const float2* B,
                   device const float2* Cm, long H, long h, uint t, uint m0) {
    for (uint n = t; n < CHUNK_MODES; n += THREADS) {
        uint i = m0 + n;
        Mode mode = load_mode(Lam, P, B, Cm, H, h, min(i, N - 1));
        float dt_h = dt[h];
        float2 w = dt_h * mode.lam;
        aw[n] = float2(w.x, dot(w, w));
        for (uint j = 0; j < PAIRS; j++) {
            float2 v = dt_h * mode.u[j];
            cs[n][j] = i < N ? -2 * dot(v, w) : 0;
            cs[n][PAIRS + j] = i < N ? 2 * v.x : 0;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Return m = 1 / q_n at z = i nu for the table entry aw = (a_n, |w_n|^2).
//     q_n = (|w_n|^2 - nu^2) - 2i nu a_n
// The forward and the backward both call this function, so that they round |w_n|^2 - nu^2 the same
// way. Near resonance, |w_n|^2 - nu^2 is small, and a rounding difference between the passes would
// make the gradients of P, B and C inaccurate.
inline float2 inverse_q(float nu, float2 aw) {
    float2 q_conj = float2(fma(-nu, nu, aw.y), 2 * nu * aw.x);
    return q_conj * (1 / dot(q_conj, q_conj));
}

// Add sum_n cs[n][j] / q_n over the modes of the tables to S[j] at z = i nu.
inline void sums(threadgroup const float (&cs)[CHUNK_MODES][COEFS],
                 threadgroup const float2 (&aw)[CHUNK_MODES], float nu,
                 thread float2 (&S)[COEFS]) {
    for (uint n = 0; n < CHUNK_MODES; n++) {
        float2 m = inverse_q(nu, aw[n]);
        for (uint j = 0; j < COEFS; j++) S[j] += cs[n][j] * m;
    }
}

// Set the resolvents r = S0 + i nu S1 at z = i nu, s = 1 / (1 + r_PQ) and b = r_BQ s.
inline void resolvents(thread const float2 (&S)[COEFS], float nu, thread float2 (&r)[PAIRS],
                       thread float2& s, thread float2& b) {
    for (uint j = 0; j < PAIRS; j++) r[j] = S[j] + nu * float2(-S[PAIRS + j].y, S[PAIRS + j].x);
    float2 den = float2(1 + r[PQ].x, r[PQ].y);
    s = conjugate(den) / dot(den, den);
    b = cmul(r[BQ], s);
}

// Return the term -2 Re(u_j / Lambda) that the mode adds to r_XY(0) for pair j.
inline float term(Mode mode, uint j) {
    return -2 * dot(mode.u[j], mode.lam) / dot(mode.lam, mode.lam);
}

// Set block_terms[block][j] to the sum of terms[i][j] over the modes i of the block. Thread t sums
// block t.
template <uint J>
inline void block_sums(threadgroup const float (&terms)[N][J],
                       threadgroup float (&block_terms)[BLOCKS][J], uint t) {
    if (t < BLOCKS)
        for (uint j = 0; j < J; j++) {
            float sum = 0;
            for (uint i = t * BLOCK_MODES; i < min(N, t * BLOCK_MODES + BLOCK_MODES); i++)
                sum += terms[i][j];
            block_terms[t][j] = sum;
        }
}

// Set o[j] to the sum of terms[i][j] over the modes i other than n, and r0[j] to the sum over all
// modes. For N > 2 BLOCK_MODES, the sums add the block sums and then the terms of n's block. The
// sum o never adds the term of mode n, so a large term of mode n does not leave a rounding error
// in o.
template <uint J>
inline void others(threadgroup const float (&terms)[N][J],
                   threadgroup const float (&block_terms)[BLOCKS][J], uint n, thread float (&o)[J],
                   thread float (&r0)[J]) {
    for (uint j = 0; j < J; j++) o[j] = r0[j] = 0;
    if (N <= 2 * BLOCK_MODES) {
        for (uint i = 0; i < N; i++)
            for (uint j = 0; j < J; j++) {
                o[j] += i == n ? 0 : terms[i][j];
                r0[j] += terms[i][j];
            }
        return;
    }
    for (uint block = 0; block < BLOCKS; block++)
        for (uint j = 0; j < J; j++) {
            o[j] += block == n / BLOCK_MODES ? 0 : block_terms[block][j];
            r0[j] += block_terms[block][j];
        }
    uint i0 = n / BLOCK_MODES * BLOCK_MODES;
    for (uint i = i0; i < min(N, i0 + BLOCK_MODES); i++)
        for (uint j = 0; j < J; j++) o[j] += i == n ? 0 : terms[i][j];
}

// Return entry n of the solution x of (Lambda - W W^*) x = V at z = 0, for (V, W) = (B, P) or
// (C, Q). The argument o_pq sums the terms of r_PQ(0) over the modes other than n, and o_v sums
// those of r_BQ(0) for V = B and of r_PC(0) for V = C. The argument den is 1 + r_PQ(0). The
// function adds the terms of mode n in closed form.
//     x_n = (V_n (1 + o_pq) - W_n o_v - 2i W_n Im(V_n conj(W_n)) / conj(Lambda_n)) / (Lambda_n den)
inline float2 solve(float2 V, float2 W, float2 lam, float o_pq, float o_v, float den) {
    float scale = 1 / (dot(lam, lam) * den), im = V.y * W.x - V.x * W.y;  // Im(V conj(W))
    float2 num = V + (V * o_pq - W * o_v);
    return float2(num.x * lam.x + num.y * lam.y + 2 * im * W.y,
                  num.y * lam.x - num.x * lam.y - 2 * im * W.x) * scale;
}

// Write k_f / L (C, H, L / 2 + 1, 2) from dt (H), Lambda, P and B (H, N, 2), and Cm (C, H, N, 2).
kernel void spectrum(device float2* out, device const float* dt, device const float2* Lam,
                     device const float2* P, device const float2* B, device const float2* Cm,
                     constant long& H, constant long& L, IDS) {
    threadgroup float cs[CHUNK_MODES][COEFS];
    threadgroup float terms[N][2], block_terms[BLOCKS][2];
    threadgroup float zero_parts[C][CHUNK_MODES], nyquist_parts[C][CHUNK_MODES];
    threadgroup float2 aw[CHUNK_MODES];
    long F = L / 2 + 1;
    uint tiles = uint(F + THREADS - 1) / THREADS, h = g / tiles, tile = g % tiles;
    long k = tile * THREADS + t, K = (L + 1) / 2 - 1;  // K is the last bin below L / 2.
    float scale = 1 / float(L);
    tables(cs, aw, dt, Lam, P, B, Cm, H, h, t, 0);
    if (tile == 0) {
        // At k = 0, the terms of r_XY(0) are large for nearly undamped modes, and their rounding
        // errors would dominate r_BC - b r_PC. The kernel computes bin 0 as -2 Re sum_n C_n x_n
        // instead, where x solves (Lambda - P P^*) x = B. Thread t < CHUNK_MODES handles the modes
        // t, t + CHUNK_MODES, ...
        uint first = t < CHUNK_MODES ? t : N;
        for (uint n = first; n < N; n += CHUNK_MODES) {
            Mode mode = load_mode(Lam, P, B, Cm, H, h, n);
            terms[n][0] = term(mode, BQ);
            terms[n][1] = term(mode, PQ);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (N > 2 * BLOCK_MODES) {
            block_sums(terms, block_terms, t);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float zero_sum[C] = {}, nyquist_sum[C] = {};
        for (uint n = first; n < N; n += CHUNK_MODES) {
            Mode mode = load_mode(Lam, P, B, Cm, H, h, n);
            float o[2], r0[2];
            others(terms, block_terms, n, o, r0);
            float2 x = solve(mode.X[0], mode.X[1], mode.lam, o[1], o[0], 1 + r0[1]);
            for (uint c = 0; c < C; c++) {
                zero_sum[c] += mode.Y[c].x * x.x - mode.Y[c].y * x.y;
                nyquist_sum[c] += dt[h] * mode.u[c].x;
            }
        }
        if (t < CHUNK_MODES)
            for (uint c = 0; c < C; c++) {
                zero_parts[c][t] = zero_sum[c];
                nyquist_parts[c][t] = nyquist_sum[c];
            }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (t < C) {
            // For even L, the Nyquist bin is the limit sum_n Re v_BC.
            float zero = 0, nyquist = 0;
            for (uint n = 0; n < CHUNK_MODES; n++) {
                zero += zero_parts[t][n];
                nyquist += nyquist_parts[t][n];
            }
            out[(t * H + h) * F] = float2(-2 * zero * scale, 0);
            if (L % 2 == 0) out[(t * H + h) * F + L / 2] = float2(nyquist * scale, 0);
        }
    }
    // With more than one chunk, the threads with k outside 0 < k <= K stay to load the tables of
    // the later chunks.
    bool active = k >= 1 && k <= K;
    if (CHUNKS == 1 && !active) return;
    float nu = frequency(k, L);
    float2 S[COEFS] = {}, r[PAIRS], s, b;
    for (uint m0 = 0; m0 < N; m0 += CHUNK_MODES) {
        if (m0 > 0) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            tables(cs, aw, dt, Lam, P, B, Cm, H, h, t, m0);
        }
        if (active) sums(cs, aw, nu, S);
    }
    if (!active) return;
    resolvents(S, nu, r, s, b);
    // k_f / L = (1 + z / 2) (r_BC - b r_PC) / L with z = i nu
    for (uint c = 0; c < C; c++) {
        float2 v = r[c] - cmul(b, r[C + 1 + c]);
        out[(c * H + h) * F + k] = (v + nu / 2 * float2(-v.y, v.x)) * scale;
    }
}

// Write the partial sums of the gradients to part (H, splits, COEFS + 2, N). Rows j < COEFS hold
// the gradients of the Cauchy coefficients, and rows COEFS and COEFS + 1 hold the gradients of a
// and |w|^2. The input gf (C, H, L / 2 + 1, 2) is the rfft of the kernel's gradient. Threadgroup
// (h, split) sums over the frequencies 0 < k < L / 2 of its split_tiles tiles. In each tile, the
// threads first store nu and the gradients GS of the sums S at their frequencies. Then thread
// t = k0 CHUNK_MODES + n sums the terms of table entry n over the frequencies k0,
// k0 + ENTRY_THREADS, ... of the tile.
kernel void spectrum_grads(device float* part, device const float2* gf, device const float* dt,
                           device const float2* Lam, device const float2* P,
                           device const float2* B, device const float2* Cm, constant long& H,
                           constant long& L, constant long& splits, constant long& split_tiles,
                           IDS) {
    // ENTRY_THREADS threads share each table entry. With more than one chunk,
    // CHUNK_MODES > THREADS / 2, so ENTRY_THREADS is 1, and the threads with k0 = 0 hold all the
    // sums.
    constexpr uint ENTRY_THREADS = THREADS / CHUNK_MODES;
    threadgroup float cs[CHUNK_MODES][COEFS], nus[THREADS];
    threadgroup float2 aw[CHUNK_MODES], GS[THREADS][COEFS];
    long F = L / 2 + 1, K = (L + 1) / 2 - 1;
    uint h = g / uint(splits), split = g % uint(splits);
    tables(cs, aw, dt, Lam, P, B, Cm, H, h, t, 0);
    uint n = t % CHUNK_MODES, k0 = t / CHUNK_MODES;
    float cn[COEFS], acc[COEFS + 2] = {};
    float2 wn;
    auto load_entry = [&]() {
        for (uint j = 0; j < COEFS; j++) cn[j] = cs[n][j];
        wn = aw[n];
    };
    if (CHUNKS == 1) load_entry();
    for (long i = 0; i < split_tiles; i++) {
        long k = (split * split_tiles + i) * THREADS + t + 1, kc = min(k, K);
        float nu = frequency(kc, L), weight = k <= K ? 2 / float(L) : 0;
        float2 S[COEFS] = {}, r[PAIRS], s, b, gr[PAIRS], gb = 0;
        sums(cs, aw, nu, S);
        for (uint m0 = CHUNK_MODES; m0 < N; m0 += CHUNK_MODES) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            tables(cs, aw, dt, Lam, P, B, Cm, H, h, t, m0);
            sums(cs, aw, nu, S);
        }
        resolvents(S, nu, r, s, b);
        for (uint c = 0; c < C; c++) {
            float2 gk = gf[(c * H + h) * F + kc] * weight;
            float2 p = gk + nu / 2 * float2(gk.y, -gk.x);  // p = conj(1 + i nu / 2) gk
            // k_f = (1 + z/2) (r_BC - b r_PC)
            gr[c] = p;
            gr[C + 1 + c] = -cmulc(p, b);
            gb -= cmulc(p, r[C + 1 + c]);
        }
        // G_BQ = conj(s) G_b and G_PQ = -conj(s^2 r_BQ) G_b.
        gr[BQ] = cmulc(gb, s);
        gr[PQ] = -cmulc(gb, cmul(b, s));
        // r = S0 + i nu S1 gives G_S0 = G_r and G_S1 = -i nu G_r.
        for (uint j = 0; j < PAIRS; j++) {
            GS[t][j] = gr[j];
            GS[t][PAIRS + j] = nu * float2(gr[j].y, -gr[j].x);
        }
        nus[t] = nu;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // The chunks run in reverse order, because the tables already hold the last chunk.
        for (uint ci = CHUNKS; ci-- > 0;) {
            uint m0 = ci * CHUNK_MODES;
            if (CHUNKS > 1) {
                if (ci < CHUNKS - 1) {
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    tables(cs, aw, dt, Lam, P, B, Cm, H, h, t, m0);
                }
                load_entry();
                for (uint j = 0; j < COEFS + 2; j++) acc[j] = 0;
            }
            if (k0 < ENTRY_THREADS)
                for (uint freq = k0; freq < THREADS; freq += ENTRY_THREADS) {
                    float nuk = nus[freq];
                    float2 m = inverse_q(nuk, wn), gm = 0;
                    for (uint j = 0; j < COEFS; j++) {
                        gm += cn[j] * GS[freq][j];
                        acc[j] += dot(m, GS[freq][j]);
                    }
                    // G_q = -conj(m^2) G_m, and q_n = (|w|^2 - nu^2) - 2i nu a gives the gradients
                    // of a and |w|^2.
                    float c2 = m.x * m.x - m.y * m.y, s2 = 2 * m.x * m.y;
                    acc[COEFS] += 2 * nuk * (c2 * gm.y - s2 * gm.x);
                    acc[COEFS + 1] -= c2 * gm.x + s2 * gm.y;
                }
            if (CHUNKS > 1 && k0 == 0 && m0 + n < N)
                for (uint j = 0; j < COEFS + 2; j++) {
                    device float* partial =
                        part + ((h * splits + split) * (COEFS + 2) + j) * N + m0 + n;
                    *partial = i == 0 ? acc[j] : *partial + acc[j];
                }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (CHUNKS > 1) return;
    threadgroup float* red = (threadgroup float*)GS;  // (ENTRY_THREADS, N, COEFS + 2)
    if (k0 < ENTRY_THREADS)
        for (uint j = 0; j < COEFS + 2; j++) red[(k0 * N + n) * (COEFS + 2) + j] = acc[j];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t < N)
        for (uint j = 0; j < COEFS + 2; j++) {
            float tot = 0;
            for (uint e = 0; e < ENTRY_THREADS; e++) tot += red[(e * N + t) * (COEFS + 2) + j];
            part[((h * splits + split) * (COEFS + 2) + j) * N + t] = tot;
        }
}

// Write the gradients of dt (H), Lambda, P and B (H, N, 2), and Cm (C, H, N, 2) from the partial
// sums in `part` and the bins 0 and L / 2 of gf.
kernel void spectrum_finish(device float* gdt, device float2* gLam, device float2* gP,
                            device float2* gB, device float2* gC, device const float* part,
                            device const float2* gf, device const float* dt,
                            device const float2* Lam, device const float2* P,
                            device const float2* B, device const float2* Cm, constant long& H,
                            constant long& L, constant long& splits, IDS) {
    // The columns of `terms` are the terms of r_BQ(0), r_PC(0)... and r_PQ(0).
    threadgroup float terms[N][C + 2], block_terms[BLOCKS][C + 2], red[SIMDS];
    long h = g, F = L / 2 + 1;
    for (uint n = t; n < N; n += THREADS) {
        Mode mode = load_mode(Lam, P, B, Cm, H, h, n);
        for (uint j = 0; j < C + 2; j++) terms[n][j] = term(mode, BQ + j);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (N > 2 * BLOCK_MODES) {
        block_sums(terms, block_terms, t);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float gdt_thread = 0;
    for (uint m0 = 0; m0 < N; m0 += THREADS) {
        uint n = min(m0 + t, N - 1);
        Mode mode = load_mode(Lam, P, B, Cm, H, h, n);
        float o[C + 2], r0[C + 2];
        others(terms, block_terms, n, o, r0);
        // M = Lambda - P P^* is a matrix over the modes and their conjugates. The gradient of a
        // parameter is 2 conj(G), where G is the derivative of k_f(0) / L with respect to the
        // parameter.
        //     k_f(0) / L = -C^T x / L
        //     x = M^-1 B
        //     mu = M^-T C
        float den = 1 + r0[C + 1], gs = 0;  // gs = sum_c g_c P^T mu_c, P^T mu_c = -r_PC / den
        float2 x = solve(mode.X[0], mode.X[1], mode.lam, o[C + 1], o[0], den), gmu = 0, gc[C];
        for (uint c = 0; c < C; c++) {
            float gk = gf[(c * H + h) * F].x / float(L);
            gmu += gk * solve(mode.Y[c], mode.Y[C], mode.lam, o[C + 1], o[1 + c], den);
            gs -= gk * r0[1 + c] / den;
            gc[c] = -2 * gk * conjugate(x);
        }
        float qx = -r0[0] / den;  // Q^T x
        // Bin 0 adds these gradients of Lambda and P.
        float2 zero_gLam = 2 * conjugate(cmul(gmu, x));
        float2 zero_gP = -2 * (qx * conjugate(gmu) + gs * x);
        // Sum the rows of `part` for mode n over the splits.
        float gcoef[COEFS + 2] = {};
        for (long split = 0; split < splits; split++)
            for (uint j = 0; j < COEFS + 2; j++)
                gcoef[j] += part[((h * splits + split) * (COEFS + 2) + j) * N + n];
        float dt_h = dt[h], nyquist = L % 2 == 0 ? 1 / float(L) : 0;
        float2 w = dt_h * mode.lam;
        float2 gw = float2(gcoef[COEFS] + 2 * w.x * gcoef[COEFS + 1], 2 * w.y * gcoef[COEFS + 1]);
        float2 gv[PAIRS];
        for (uint j = 0; j < PAIRS; j++) {
            float g0 = gcoef[j], g1 = gcoef[PAIRS + j];
            gv[j] = float2(-2 * w.x * g0 + 2 * g1, -2 * w.y * g0);
            gw -= 2 * dt_h * mode.u[j] * g0;
            if (j < C) gv[j].x += nyquist * gf[(j * H + h) * F + L / 2].x;
        }
        float gdt_mode = dot(mode.lam, gw);
        // gX and gY start from the gradients that bin 0 gives.
        float2 gX[2] = {-2 * conjugate(gmu), zero_gP}, gY[C + 1];
        for (uint c = 0; c < C; c++) gY[c] = gc[c];
        gY[C] = 0;
        for (uint i = 0; i < 2; i++)
            for (uint c = 0; c <= C; c++) {
                uint j = i * (C + 1) + c;
                gdt_mode += dot(gv[j], mode.u[j]);
                gX[i] += cmulc(dt_h * gv[j], mode.Y[c]);
                gY[c] += cmulc(dt_h * gv[j], mode.X[i]);
            }
        if (m0 + t < N) {
            long i = h * N + n;
            gB[i] = gX[0];
            gP[i] = gX[1] + conjugate(gY[C]);
            gLam[i] = dt_h * gw + zero_gLam;
            for (uint c = 0; c < C; c++) gC[c * H * N + i] = gY[c];
            gdt_thread += gdt_mode;
        }
    }
    gdt_thread = simd_sum(gdt_thread);
    if (lane == 0) red[simd] = gdt_thread;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t == 0) {
        for (uint i = 1; i < SIMDS; i++) gdt_thread += red[i];
        gdt[h] = gdt_thread;
    }
}
