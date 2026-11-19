"""Kernel that transposes the last two axes of a (B, M, N) tensor."""

import tilelang.language as T

from ..runtime import compiled
from .dsl import int_var

# A thread block holds its 32 x 64 tile in at most 8.3 KB of shared memory, so that 7 blocks fit in
# the 64 KB of an AMD compute unit. The loads of 7 blocks are enough to reach the memory bandwidth.
TILE = 32, 64
THREADS = 256


@compiled
def kernel(dtype, out):
    """Build the kernel that transposes x (B, M, N) into y (B, N, M) of dtype `out`."""
    BM, BN = TILE
    B, M, N = T.dynamic("B"), T.dynamic("M"), T.dynamic("N")

    @T.prim_func
    def transpose(x: T.Tensor((B, M, N), dtype), y: T.Tensor((B, N, M), out)):
        tn, tm = T.ceildiv(N, BN), T.ceildiv(M, BM)  # the tile counts along N and M
        # The grid is one-dimensional, because the y and z axes of a grid are limited to 65535
        # thread blocks, and a sequence can have more tiles than that.
        with T.Kernel(tn * tm * B, threads=THREADS) as block:
            bn, rest = int_var(block % tn), int_var(block // tn)
            bm, b = int_var(rest % tm), int_var(rest // tm)
            # The extra column spreads the 32 elements of a column, which a warp reads in the
            # second loop, over the banks of shared memory.
            S = T.alloc_shared((BM, BN + 1), dtype)
            for i, j in T.Parallel(BM, BN):
                m, n = bm * BM + i, bn * BN + j
                S[i, j] = T.if_then_else(T.And(m < M, n < N), x[b, m, n], T.cast(0, dtype))
            for j, i in T.Parallel(BN, BM):
                m, n = bm * BM + i, bn * BN + j
                if T.And(m < M, n < N):
                    y[b, n, m] = T.cast(S[i, j], out)

    return transpose
