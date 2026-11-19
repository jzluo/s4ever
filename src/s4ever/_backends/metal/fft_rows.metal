// The row kernels of fft.py, compiled after fft.metal. The slice k1 of a transform of n = N1 NF
// points is Z_k1, and positions[f] is the position of the slice of frequency f in Kf, U and the
// column transforms A.
//
// In each kernel, buf is the exchange buffer of the threadgroup's transforms, l is the thread's
// index in the threadgroup, t is its index in its transform, and off is the offset of its
// transform in buf. A transform holds the rows a and b = a + P as one complex sequence. `live` is
// whether the transform has a row a, and has_b is whether it has a row b.

// The kernel parameters for the twiddles of the NF-point transforms and for the coarse and fine
// roots of `fft.twiddles`.
#define TABLES device const float2* tw, device const float2* coarse, device const float2* fine

// Return the slice of threadgroup g and its index among the `groups` threadgroups of the slice.
// For N1 > SPLIT, consecutive threadgroups take one slice, because they read the same spectra.
// Otherwise consecutive threadgroups take the slices of one row, because these slices read the
// same samples.
inline uint2 slice_and_group(uint g, uint N1, uint groups) {
    return N1 > SPLIT ? uint2(g / groups, g % groups) : uint2(g % N1, g / N1);
}

// Load Z_k1 into v for the row pair at offsets ra and rb of src. The real part of z is the row at
// ra, and the imaginary part is the row at rb if has_b, else 0. For N1 > SPLIT, Z_k1 is row arow of
// the column transforms A. Otherwise the function sums the first `segments` segments z[p + NF n1]
// with their twiddles, and it issues the loads of a segment together, so that more loads are in
// flight at once.
template <typename T>
inline void load_slice(thread float2* v, device const T* src, long ra, long rb,
                       device const float2* A, long arow, uint k1, uint N1, uint segments, long L,
                       bool live, bool has_b, device const float2* coarse,
                       device const float2* fine, uint t) {
    if (N1 > SPLIT) {
        for (uint i = 0; i < E; i++)
            v[i] = live ? A[(arow * N1 + k1) * NF + time_pos(t, i)] : float2(0);
        return;
    }
    for (uint i = 0; i < E; i++) v[i] = 0;
    if (live)
        for (uint n1 = 0; n1 < segments; n1++) {
            float2 r = coarse[n1 * k1 % N1];
            for (uint i = 0; i < E; i++) {
                long s = time_pos(t, i) + long(NF) * n1;
                if (s < L) v[i] += cmul(float2(src[ra + s], has_b ? src[rb + s] : 0), r);
            }
        }
    if (N1 > 1)
        for (uint i = 0; i < E; i++) v[i] = cmul(v[i], root(time_pos(t, i) * k1, coarse, fine));
}

// Return the share of the value x of slice k1 in the outputs at time position p. For N1 <= SPLIT,
// the share is x conj(w^(p k1)), and `combine_pairs` sums the shares. For N1 > SPLIT, the share is
// x, because `columns_inverse_pairs` applies the twiddle.
inline float2 share(float2 x, uint p, uint k1, uint N1, device const float2* coarse,
                    device const float2* fine) {
    return N1 > SPLIT ? x : cmulc(x, root(p * k1, coarse, fine));
}

// Store the shares of the values v of slice k1 in row `row` of U (rows, N1, NF) if live.
inline void store_shares(device float2* U, thread float2* v, long row, uint k1, uint N1, bool live,
                         device const float2* coarse, device const float2* fine, uint t) {
    if (!live) return;
    for (uint i = 0; i < E; i++) {
        uint p = time_pos(t, i);
        U[(row * N1 + k1) * NF + p] = share(v[i], p, k1, N1, coarse, fine);
    }
}

// Store the values v of a transform in buf in frequency order, between barriers.
inline void by_freq(thread float2* v, threadgroup float2* buf, uint off, uint t) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = 0; i < E; i++) buf[off + rev(freq_pos(t, i))] = v[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Return the conjugate of the value at frequency -(f + N1 rev(p)), which is in slice -f mod N1.
// `by_freq` has stored that slice in buf.
inline float2 mirrored(threadgroup float2* buf, uint off, uint p, uint f) {
    return conjugate(buf[off + (f ? NF - 1 - rev(p) : (NF - rev(p)) % NF)]);
}

// Store the spectra / n of rows a and b at slice f in Kf, from the values v of slice f and the
// mirrored slice in buf. Z is the transform of z = k_a + i k_b, and `scale` is 1 / (2 n).
//     K_a = (Z_f + conj Z_-f) / 2
//     K_b = (Z_f - conj Z_-f) / 2i
inline void store_spectra(device float2* Kf, thread float2* v, uint f, device const int* positions,
                          long a, long b, bool live, bool has_b, uint N1, float scale,
                          threadgroup float2* buf, uint off, uint t) {
    for (uint i = 0; i < E; i++) {
        uint p = freq_pos(t, i);
        float2 mirror = mirrored(buf, off, p, f);
        float2 ka = (v[i] + mirror) * scale, ikb = (v[i] - mirror) * scale;
        if (live) Kf[(a * N1 + positions[f]) * NF + p] = ka;
        if (has_b) Kf[(b * N1 + positions[f]) * NF + p] = float2(ikb.y, -ikb.x);
    }
}

// Write the spectra / n (rows, N1, NF) of the real kernels k (rows, n). A threadgroup handles the
// slices f1 and f2 = -f1 mod N1 for GROUP_FFTS row pairs, because the negative of each frequency
// of one slice is in the other slice. `two` is whether f2 differs from f1.
kernel void spectrum(device float2* Kf, device const float* k, device const float2* A,
                     device const int* positions, TABLES, constant long& rows,
                     constant long& slice_count, FFT_IDS) {
    threadgroup float2 buf[GROUP_FFTS * NF];
    uint N1 = slice_count, slice_pairs = N1 / 2 + 1;
    uint l = l3.x, t = l % FFT_THREADS, off = l / FFT_THREADS * NF;
    uint f1 = g.x % slice_pairs, f2 = (N1 - f1) % N1;
    long P = (rows + 1) / 2, a = g.x / slice_pairs * GROUP_FFTS + l / FFT_THREADS, b = a + P;
    long n = long(NF) * N1;
    bool live = a < P, has_b = b < rows, two = f2 != f1;
    float scale = 0.5f / float(n);
    float2 v[E], v2[E];
    load_slice(v, k, a * n, b * n, A, a, positions[f1], N1, N1, n, live, has_b, coarse, fine, t);
    forward_stages(v, buf, off, t, tw);
    if (two) {
        load_slice(v2, k, a * n, b * n, A, a, positions[f2], N1, N1, n, live, has_b, coarse, fine,
                   t);
        forward_stages(v2, buf, off, t, tw);
        by_freq(v2, buf, off, t);
        store_spectra(Kf, v, f1, positions, a, b, live, has_b, N1, scale, buf, off, t);
        by_freq(v, buf, off, t);
        store_spectra(Kf, v2, f2, positions, a, b, live, has_b, N1, scale, buf, off, t);
    } else {
        by_freq(v, buf, off, t);
        store_spectra(Kf, v, f1, positions, a, b, live, has_b, N1, scale, buf, off, t);
    }
}

// Write the convolutions y (B, C, H, L) of x (B, H, L) with the kernels whose spectra are
// Kf (C, H, N1, NF). For n > NF, the threadgroups write their shares to U (P, C, H, N1, NF)
// instead, and a combine pass writes y.
kernel void conv(device IO* y, device const float* ad, device float2* U, device const IO* x,
                 device const float2* A, device const float2* Kf, TABLES, constant long& B,
                 constant long& H, constant long& L, constant long& slice_count,
                 constant long& has_ad, FFT_IDS) {
    threadgroup float2 buf[GROUP_FFTS * NF];
    uint N1 = slice_count;
    uint l = l3.x, t = l % FFT_THREADS, off = l / FFT_THREADS * NF;
    long P = (B + 1) / 2;
    uint2 slice = slice_and_group(g.x, N1, (P * H + GROUP_FFTS - 1) / GROUP_FFTS);
    uint k1 = slice.x, pair = slice.y * GROUP_FFTS + l / FFT_THREADS;  // h P + a
    long row = pair, h = pair / uint(P), a = pair % uint(P), b = a + P;
    uint segments = min(long(N1), (L + NF - 1) / NF);
    bool live = row < P * H, has_b = b < B;
    float2 X[E], work[C > 1 ? E : 1];
    load_slice(X, x, (a * H + h) * L, (b * H + h) * L, A, a * H + h, k1, N1, segments, L, live,
               has_b, coarse, fine, t);
    forward_stages(X, buf, off, t, tw);
    for (uint c = 0; c < C; c++) {
        thread float2* v = X;
        if (C > 1) {
            for (uint i = 0; i < E; i++) work[i] = X[i];
            v = work;
        }
        for (uint i = 0; i < E; i++) {
            uint p = freq_pos(t, i);
            v[i] = live ? cmul(v[i], Kf[((c * H + h) * N1 + k1) * NF + p]) : float2(0);
        }
        inverse_stages(v, buf, off, t, tw);
        if (!live) continue;
        if (N1 == 1) {
            for (uint i = 0; i < E; i++) {
                uint p = time_pos(t, i);
                if (p < L) {
                    long qa = ((a * C + c) * H + h) * L + p, qb = qa + P * C * H * L;
                    y[qa] = IO(v[i].x + addend(ad, has_ad, qa));
                    if (has_b) y[qb] = IO(v[i].y + addend(ad, has_ad, qb));
                }
            }
        } else {
            store_shares(U, v, (a * C + c) * H + h, k1, N1, live, coarse, fine, t);
        }
    }
}

// `conv` for n = 2 NF, where one threadgroup transforms both slices, so that the convolution does
// not need a combine pass. Because L <= NF, only the segment n1 = 0 holds samples, so Z_0 = z and
// Z_1 = z w^m. The outputs lie in the first half, where y = v_0 + v_1 conj(w^m).
kernel void conv_halves(device IO* y, device const float* ad, device const IO* x,
                        device const float2* Kf, TABLES, constant long& B, constant long& H,
                        constant long& L, constant long& has_ad, FFT_IDS) {
    threadgroup float2 buf[GROUP_FFTS * NF];
    uint l = l3.x, t = l % FFT_THREADS, off = l / FFT_THREADS * NF;
    uint pair = g.x * GROUP_FFTS + l / FFT_THREADS;  // h P + a
    long P = (B + 1) / 2, row = pair, h = pair / uint(P), a = pair % uint(P), b = a + P;
    bool live = row < P * H, has_b = b < B;
    float2 X[2 * E], work[C > 1 ? 2 * E : 1];
    for (uint i = 0; i < E; i++) {
        uint p = time_pos(t, i);
        long ra = (a * H + h) * L + p, rb = (b * H + h) * L + p;
        X[i] = live && p < L ? float2(x[ra], has_b ? x[rb] : 0) : float2(0);
        X[E + i] = cmul(X[i], fine[p]);
    }
    forward_stages(X, buf, off, t, tw);
    forward_stages(X + E, buf, off, t, tw);
    for (uint c = 0; c < C; c++) {
        thread float2* v = X;
        if (C > 1) {
            for (uint i = 0; i < 2 * E; i++) work[i] = X[i];
            v = work;
        }
        for (uint k1 = 0; k1 < 2; k1++) {
            for (uint i = 0; i < E; i++) {
                uint p = freq_pos(t, i);
                v[k1 * E + i] = live ? cmul(v[k1 * E + i], Kf[((c * H + h) * 2 + k1) * NF + p])
                                     : float2(0);
            }
            inverse_stages(v + k1 * E, buf, off, t, tw);
        }
        if (!live) continue;
        for (uint i = 0; i < E; i++) {
            uint p = time_pos(t, i);
            if (p < L) {
                float2 o = v[i] + cmulc(v[E + i], fine[p]);
                long qa = ((a * C + c) * H + h) * L + p, qb = qa + P * C * H * L;
                y[qa] = IO(o.x + addend(ad, has_ad, qa));
                if (has_b) y[qb] = IO(o.y + addend(ad, has_ad, qb));
            }
        }
    }
}

// Write the gradient gx (B, H, L) and the sums gK (chunks, C, H, N1, NF). The gradient gx is
// gy (B, C, H, L) correlated with the kernels and summed over the channels. For each chunk of
// chunk_pairs row pairs, gK sums the products V conj(X), where V and X are the transforms of the
// row pairs of gy and x. The pairing adds products of row a with row b to gK, which drop out of
// gk, because their Hermitian parts are zero. For n > NF, the threadgroups write the shares of gx
// to U (P, H, N1, NF). Without want_gx, the kernel writes only gK.
kernel void conv_grads(device IO* gx, device float2* U, device float2* gK, device const IO* x,
                       device const IO* gy, device const float2* A, device const float2* Ag,
                       device const float2* Kf, TABLES, constant long& B, constant long& H,
                       constant long& L, constant long& slice_count, constant long& chunk_pairs,
                       constant long& want_gx, FFT_IDS) {
    threadgroup float2 buf[GROUP_FFTS * NF];
    uint N1 = slice_count;
    uint l = l3.x, t = l % FFT_THREADS, off = l / FFT_THREADS * NF;
    long P = (B + 1) / 2;
    uint chunks = uint(P + chunk_pairs - 1) / uint(chunk_pairs);
    uint2 slice = slice_and_group(g.x, N1, (H + GROUP_FFTS - 1) / GROUP_FFTS * chunks);
    uint k1 = slice.x;
    long chunk = slice.y % chunks, h = slice.y / chunks * GROUP_FFTS + l / FFT_THREADS;
    long p0 = chunk * chunk_pairs, p1 = min(P, p0 + chunk_pairs);
    uint segments = min(long(N1), (L + NF - 1) / NF);
    bool live = h < H;
    float2 acc[C][E];
    for (uint c = 0; c < C; c++)
        for (uint i = 0; i < E; i++) acc[c][i] = 0;
    for (long a = p0; a < p1; a++) {
        long b = a + P;
        bool has_b = b < B;
        float2 X[E], S[E], v[E];
        load_slice(X, x, (a * H + h) * L, (b * H + h) * L, A, a * H + h, k1, N1, segments, L, live,
                   has_b, coarse, fine, t);
        forward_stages(X, buf, off, t, tw);
        for (uint i = 0; i < E; i++) S[i] = 0;
        for (uint c = 0; c < C; c++) {
            long row = (a * C + c) * H + h;
            load_slice(v, gy, row * L, (row + P * C * H) * L, Ag, row, k1, N1, segments, L, live,
                       has_b, coarse, fine, t);
            forward_stages(v, buf, off, t, tw);
            if (live)
                for (uint i = 0; i < E; i++) acc[c][i] += cmulc(v[i], X[i]);
            if (live && want_gx)
                for (uint i = 0; i < E; i++) {
                    uint p = freq_pos(t, i);
                    S[i] += cmulc(v[i], Kf[((c * H + h) * N1 + k1) * NF + p]);
                }
        }
        if (!want_gx) continue;
        inverse_stages(S, buf, off, t, tw);
        if (!live) continue;
        if (N1 == 1) {
            for (uint i = 0; i < E; i++) {
                uint p = time_pos(t, i);
                if (p < L) {
                    long qa = (a * H + h) * L + p, qb = qa + P * H * L;
                    gx[qa] = IO(S[i].x);
                    if (has_b) gx[qb] = IO(S[i].y);
                }
            }
        } else {
            store_shares(U, S, a * H + h, k1, N1, live, coarse, fine, t);
        }
    }
    if (live)
        for (uint c = 0; c < C; c++)
            for (uint i = 0; i < E; i++)
                gK[(((chunk * C + c) * H + h) * N1 + k1) * NF + freq_pos(t, i)] = acc[c][i];
}

// Set X = G_a + i G_b and Y = G_a - i G_b at slice f, where G is gK summed over its chunks.
inline void gather(thread float2* X, thread float2* Y, uint f, device const float2* gK,
                   device const int* positions, long chunks, long rows, long a, long b, bool live,
                   bool has_b, uint N1, uint t) {
    for (uint i = 0; i < E; i++) {
        uint p = freq_pos(t, i);
        float2 ga = 0, gb = 0;
        for (long chunk = 0; chunk < chunks; chunk++) {
            if (live) ga += gK[((chunk * rows + a) * N1 + positions[f]) * NF + p];
            if (has_b) gb += gK[((chunk * rows + b) * N1 + positions[f]) * NF + p];
        }
        X[i] = ga + float2(-gb.y, gb.x);
        Y[i] = ga - float2(-gb.y, gb.x);
    }
}

// Write the shares U of gk (rows, n) for n > NF. The gradient gk is the real part of the inverse
// transform of G / n, which equals the inverse transform of the Hermitian part H / n of G. Rows a
// and b share the inverse transform of Z = H_a + i H_b. The threadgroups take the slices and the
// rows that they take in `spectrum`.
//     H_f = (G_f + conj G_-f) / 2
//     Z_f = (X_f + conj Y_-f) / 2
kernel void spectrum_grads(device float2* U, device const float2* gK, device const int* positions,
                           TABLES, constant long& rows, constant long& chunks,
                           constant long& slice_count, FFT_IDS) {
    threadgroup float2 buf[GROUP_FFTS * NF];
    uint N1 = slice_count, slice_pairs = N1 / 2 + 1;
    uint l = l3.x, t = l % FFT_THREADS, off = l / FFT_THREADS * NF;
    uint f1 = g.x % slice_pairs, f2 = (N1 - f1) % N1;
    long P = (rows + 1) / 2, a = g.x / slice_pairs * GROUP_FFTS + l / FFT_THREADS, b = a + P;
    bool live = a < P, has_b = b < rows, two = f2 != f1;
    float scale = 0.5f / float(long(NF) * N1);
    float2 v[E], v2[E], Y[E];
    gather(v2, Y, f2, gK, positions, chunks, rows, a, b, live, has_b, N1, t);
    by_freq(Y, buf, off, t);
    if (two) {
        gather(v, Y, f1, gK, positions, chunks, rows, a, b, live, has_b, N1, t);
    } else {
        for (uint i = 0; i < E; i++) v[i] = v2[i];
    }
    for (uint i = 0; i < E; i++)
        v[i] = (v[i] + mirrored(buf, off, freq_pos(t, i), f1)) * scale;
    inverse_stages(v, buf, off, t, tw);
    store_shares(U, v, a, positions[f1], N1, live, coarse, fine, t);
    if (two) {
        by_freq(Y, buf, off, t);
        for (uint i = 0; i < E; i++)
            v2[i] = (v2[i] + mirrored(buf, off, freq_pos(t, i), f2)) * scale;
        inverse_stages(v2, buf, off, t, tw);
        store_shares(U, v2, a, positions[f2], N1, live, coarse, fine, t);
    }
}

// Write gk (rows, NF) for n = NF, where each transform holds one row and does not need shares.
kernel void spectrum_grads_whole(device float* gk, device const float2* gK, TABLES,
                                 constant long& rows, constant long& chunks, FFT_IDS) {
    threadgroup float2 buf[GROUP_FFTS * NF];
    uint l = l3.x, t = l % FFT_THREADS, off = l / FFT_THREADS * NF;
    long row = long(g.x) * GROUP_FFTS + l / FFT_THREADS;
    bool live = row < rows;
    float2 v[E];
    for (uint i = 0; i < E; i++) {
        uint p = freq_pos(t, i);
        v[i] = 0;
        if (live)
            for (long chunk = 0; chunk < chunks; chunk++) v[i] += gK[(chunk * rows + row) * NF + p];
    }
    inverse_stages(v, buf, off, t, tw);
    if (live)
        for (uint i = 0; i < E; i++) gk[row * NF + time_pos(t, i)] = v[i].x / float(NF);
}

// Write k (rows, n), the unscaled inverse transforms of the half spectra f (rows, n / 2 + 1), and
// drop the imaginary parts of bins 0 and n / 2. Bin j > n / 2 is conj(f_(n - j)). Rows a and
// b = a + P share z = k_a + i k_b, the inverse transform of Z = f_a + i f_b. Consecutive
// threadgroups take the slices of one row, because consecutive slices read adjacent bins of f.
kernel void real_inverse(device float* k, device float2* U, device const float2* f,
                         device const int* positions, TABLES, constant long& rows,
                         constant long& slice_count, FFT_IDS) {
    threadgroup float2 buf[GROUP_FFTS * NF];
    uint N1 = slice_count;
    uint l = l3.x, t = l % FFT_THREADS, off = l / FFT_THREADS * NF, k1 = g.x % N1;
    uint position = positions[k1], n = NF * N1;
    uint nyquist = n / 2;
    long P = (rows + 1) / 2, a = g.x / N1 * GROUP_FFTS + l / FFT_THREADS, b = a + P;
    bool live = a < P, has_b = b < rows;
    float2 v[E];
    for (uint i = 0; i < E; i++) {
        uint j = k1 + N1 * rev(freq_pos(t, i)), bin = j <= nyquist ? j : n - j;
        float2 fa = live ? f[a * (nyquist + 1) + bin] : float2(0);
        float2 fb = has_b ? f[b * (nyquist + 1) + bin] : float2(0);
        float sign = bin == 0 || bin == nyquist ? 0 : j > nyquist ? -1 : 1;
        v[i] = float2(fa.x - sign * fb.y, sign * fa.y + fb.x);
    }
    inverse_stages(v, buf, off, t, tw);
    if (!live) return;
    if (N1 == 1) {
        for (uint i = 0; i < E; i++) {
            uint p = time_pos(t, i);
            k[a * NF + p] = v[i].x;
            if (has_b) k[b * NF + p] = v[i].y;
        }
    } else {
        for (uint i = 0; i < E; i++) {
            uint p = time_pos(t, i);
            U[(a * N1 + position) * NF + p] = share(v[i], p, k1, N1, coarse, fine);
        }
    }
}

// Write rows a and P + a of y (B, R, H, L) at t = m + NF n1 < L, plus ad if has_ad, as the N1-point
// inverse DFTs sum_k1 U[k1, m] conj(w^(NF n1 k1)) of the shares U (P R H, N1, NF). Thread
// (m, r) of the grid handles column m of the row pair r.
kernel void combine_pairs(device IO* y, device const float* ad, device const float2* U,
                          constant long& B, constant long& R, constant long& H, constant long& L,
                          constant long& slice_count, constant long& has_ad,
                          uint2 gid [[thread_position_in_grid]]) {
    uint N1 = slice_count;
    uint rh = R * H;
    long P = (B + 1) / 2, row = gid.y, a = gid.y / rh, rest = gid.y % rh, b = a + P;
    float2 u[SPLIT];
    for (uint k1 = 0; k1 < N1; k1++) u[k1] = U[(row * N1 + k1) * NF + gid.x];
    inverse_dft(u, N1);
    for (uint n1 = 0; n1 < N1; n1++) {
        long s = gid.x + long(NF) * n1;
        if (s >= L) return;
        long qa = (a * R * H + rest) * L + s, qb = (b * R * H + rest) * L + s;
        y[qa] = IO(u[n1].x + addend(ad, has_ad, qa));
        if (b < B) y[qb] = IO(u[n1].y + addend(ad, has_ad, qb));
    }
}
