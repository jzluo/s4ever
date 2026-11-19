// The column kernels of fft.py, compiled after fft.metal. A row of n = N1 NF_MAX points has the
// NF_MAX columns z[m + NF_MAX n1] of N1 points, and the libraries of these kernels have NF = N1. A
// threadgroup transforms GROUP_FFTS adjacent columns of one row, and consecutive threadgroups take
// the next columns of the same row, so that their loads share cache lines.
//
// In both kernels, buf is the exchange buffer of the threadgroup's transforms, l is the thread's
// index in the threadgroup, t is its index in its transform, off is the offset of its transform in
// buf, and m and row are the column and the row of its transform.

// The kernel parameters for the coarse and fine roots of `fft.twiddles`. The coarse roots of n
// points are the twiddles of the N1-point transforms, so the kernels bind them to tw, which the
// stages read.
#define COLUMN_TABLES device const float2* tw, device const float2* fine

// Write A (rows, NF, NF_MAX), the column transforms of the row pairs of src (B, R, H, L),
// multiplied by the twiddles w^(m k1), where k1 = rev(p) is the frequency at position p.
kernel void columns_pairs(device float2* A, device const IO* src, COLUMN_TABLES,
                          constant long& B, constant long& R, constant long& H, constant long& L,
                          FFT_IDS) {
    threadgroup float2 buf[GROUP_FFTS * NF];
    uint l = l3.x, t = l / GROUP_FFTS, off = l % GROUP_FFTS * NF;
    uint m = g.x % (NF_MAX / GROUP_FFTS) * GROUP_FFTS + l % GROUP_FFTS;
    long row = g.x / (NF_MAX / GROUP_FFTS);
    long P = (B + 1) / 2, ra = row * L, rb = ra + P * R * H * L;
    bool has_b = uint(row) / uint(R * H) + P < B;
    float2 v[E];
    for (uint i = 0; i < E; i++) {
        long s = m + long(NF_MAX) * time_pos(t, i);
        v[i] = s < L ? float2(src[ra + s], has_b ? src[rb + s] : 0) : float2(0);
    }
    forward_stages(v, buf, off, t, tw);
    for (uint i = 0; i < E; i++) {
        uint p = freq_pos(t, i);
        A[(row * NF + p) * NF_MAX + m] = cmul(v[i], root(m * rev(p), tw, fine));
    }
}

// Write rows a and P + a of y (B, R, H, L) at t = m + NF_MAX n1 < L, plus ad if has_ad, as the
// NF-point inverse DFTs sum_k1 U[k1, m] conj(w^(t k1)) of the shares U (P R H, NF, NF_MAX).
kernel void columns_inverse_pairs(device IO* y, device const float2* U, COLUMN_TABLES,
                                  device const float* ad, constant long& B, constant long& R,
                                  constant long& H, constant long& L, constant long& has_ad,
                                  FFT_IDS) {
    threadgroup float2 buf[GROUP_FFTS * NF];
    uint l = l3.x, t = l / GROUP_FFTS, off = l % GROUP_FFTS * NF;
    uint m = g.x % (NF_MAX / GROUP_FFTS) * GROUP_FFTS + l % GROUP_FFTS;
    long row = g.x / (NF_MAX / GROUP_FFTS);
    long P = (B + 1) / 2, ra = row * L, rb = ra + P * R * H * L;
    bool has_b = uint(row) / uint(R * H) + P < B;
    float2 v[E];
    for (uint i = 0; i < E; i++) {
        uint p = freq_pos(t, i);
        v[i] = cmulc(U[(row * NF + p) * NF_MAX + m], root(m * rev(p), tw, fine));
    }
    inverse_stages(v, buf, off, t, tw);
    for (uint i = 0; i < E; i++) {
        long s = m + long(NF_MAX) * time_pos(t, i);
        if (s < L) {
            y[ra + s] = IO(v[i].x + addend(ad, has_ad, ra + s));
            if (has_b) y[rb + s] = IO(v[i].y + addend(ad, has_ad, rb + s));
        }
    }
}
