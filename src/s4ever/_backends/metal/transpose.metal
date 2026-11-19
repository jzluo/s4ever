// transpose.py compiles this kernel after the constants of `transposes`.

// Store y (B, N, M) = x (B, M, N)^T. A threadgroup moves a BM x BN tile through threadgroup memory,
// so that it reads contiguous rows of x and writes contiguous rows of y.
kernel void transpose(device IO* y, device const IO* x, constant long& Mn, constant long& Nn,
                      IDS) {
    // The extra column puts the rows of the tile BN + 1 elements apart. The second loop reads a
    // column of the tile in consecutive threads, and the kernel runs slower when those elements
    // are BN apart.
    threadgroup IO tile[BM][BN + 1];
    uint M = uint(Mn), N = uint(Nn), tn = (N + BN - 1) / BN, tm = (M + BM - 1) / BM;
    uint bn = g % tn, bm = g / tn % tm;
    ulong base = ulong(g / tn / tm) * M * N;
    for (uint q = t; q < BM * BN; q += THREADS) {
        uint m = bm * BM + q / BN, n = bn * BN + q % BN;
        if (m < M && n < N) tile[q / BN][q % BN] = x[base + m * N + n];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint q = t; q < BM * BN; q += THREADS) {
        uint m = bm * BM + q % BM, n = bn * BN + q / BM;
        if (m < M && n < N) y[base + n * M + m] = tile[q % BM][q / BM];
    }
}
