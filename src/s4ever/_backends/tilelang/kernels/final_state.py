"""Kernel that scans S4's states over groups of chunks on tensor cores (`ops._s4.final_state`)."""

import tilelang.language as T
import torch

from ..layout import FINAL_STATE_ROWS as ROWS
from ..runtime import compiled
from .dsl import f32, store_tf32


def padded_states(S: int) -> int:
    """Return the state dimension S padded to 16, 32 or 64.

    On AMD the padded dimension is at least 32, because the kernel has 32 threads per 16 states and
    T.gemm needs a wavefront of 64 threads.
    """
    return max(32 if torch.version.hip else 16, 1 << (S - 1).bit_length())


def macros(Tc, S, dtype, split):
    Sp = padded_states(S)

    @T.macro
    def setup(h, r0, B, R, Cg, L, In, E, Ih, Il, Eh, El, row_batch, row_start):
        """Store In^T and E as TF32 halves, zero-padded to Sp states.

        For each row, store its batch row in row_batch and the step of u at which its group starts
        in row_start.
        """
        for s, j in T.Parallel(Sp, Tc):
            v = T.if_then_else(s < S, In[h, j, T.min(s, S - 1)], 0.0)
            store_tf32(Ih, Il, v, split, s, j)
        for i, j in T.Parallel(Sp, Sp):
            v = T.if_then_else(i < S and j < S, E[h, T.min(i, S - 1), T.min(j, S - 1)], 0.0)
            store_tf32(Eh, El, v, split, i, j)
        for m in T.Parallel(ROWS):
            r = T.min(r0 + m, R - 1)
            row_batch[m] = r % B
            row_start[m] = L - (R // B - r // B) * Cg * Tc

    @T.macro
    def load(Un, u, h, r0, R, row_batch, row_start, j):
        """Load chunk j of each row's group into Un, with zeros at the steps before u starts."""
        for m, i in T.Parallel(ROWS, Tc):
            t = row_start[m] + j * Tc + i
            ok = r0 + m < R and t >= 0
            Un[m, i] = T.if_then_else(ok, u[row_batch[m], h, T.max(t, 0)], T.cast(0, dtype))

    return setup, load


def threads(Sp):
    return 32 * (Sp // 16)


@compiled
def forward_kernel(Tc, S, dtype, precision):
    """Build the kernel that computes the state at the end of each group.

    Each row of X is the state at the end of a scan that starts from zero and runs over a group of
    Cg consecutive chunks of Tc steps in one batch row. The rows are ordered by group, then by batch
    row. The G groups cover G Cg Tc steps that end at the last step of u, and the steps before the
    start of u count as zero inputs. For each chunk, the kernel updates the rows by

        X <- X + X E^T + U In
        E = dA^Tc - I

    where U holds the chunk's inputs and In (Tc, S) is the Krylov block of the state powers.
    `precision` is the number of TF32 passes of a float32 product (`runtime.precision`).
    """
    Sp = padded_states(S)
    split = precision == 3
    split_u = split and dtype == f32  # bf16 and fp16 values are exact in TF32
    setup, load = macros(Tc, S, dtype, split)
    B, H, L, R = (T.dynamic(n) for n in ("B", "H", "L", "R"))

    @T.prim_func
    def final_state_forward(
        u: T.Tensor((B, H, L), dtype),
        In: T.Tensor((H, Tc, S), f32),
        E: T.Tensor((H, S, S), f32),
        X: T.Tensor((H, R, S), f32),
        Cg: T.int32,
    ):
        with T.Kernel(T.ceildiv(R, ROWS), H, threads=threads(Sp)) as (rt, h):
            # TileLang 0.1.15 assumes only that sizes are non-negative. Without these assumes, it
            # adds a correction for a negative B or a negative min(r, R - 1) to each floor division
            # and modulo by B.
            T.assume(B > 0)
            T.assume(R > 0)
            Ih = T.alloc_fragment((Sp, Tc), f32)
            Il = T.alloc_fragment((Sp, Tc), f32)
            Eh = T.alloc_fragment((Sp, Sp), f32)
            El = T.alloc_fragment((Sp, Sp), f32)
            row_batch = T.alloc_shared((ROWS,), T.int32)
            row_start = T.alloc_shared((ROWS,), T.int32)
            r0 = rt * ROWS  # the block's first row
            setup(h, r0, B, R, Cg, L, In, E, Ih, Il, Eh, El, row_batch, row_start)

            Un = T.alloc_fragment((ROWS, Tc), dtype)  # the next chunk's inputs, prefetched
            Uh = T.alloc_shared((ROWS, Tc), f32)
            Ul = T.alloc_shared((ROWS, Tc), f32)
            Xh = T.alloc_shared((ROWS, Sp), f32)
            Xl = T.alloc_shared((ROWS, Sp), f32)
            states = T.alloc_fragment((ROWS, Sp), f32)
            # Each product term has a separate accumulator, so that the MMA chains of the terms do
            # not depend on each other.
            P1 = T.alloc_fragment((ROWS, Sp), f32)
            P2 = T.alloc_fragment((ROWS, Sp), f32)
            P3 = T.alloc_fragment((ROWS, Sp), f32)
            P4 = T.alloc_fragment((ROWS, Sp), f32)
            P5 = T.alloc_fragment((ROWS, Sp), f32)
            T.clear(states)
            load(Un, u, h, r0, R, row_batch, row_start, 0)
            for j in T.serial(Cg):
                for m, i in T.Parallel(ROWS, Tc):
                    v = T.cast(Un[m, i], f32)
                    store_tf32(Uh, Ul, v, split_u, m, i)
                for m, s in T.Parallel(ROWS, Sp):
                    v = states[m, s]
                    store_tf32(Xh, Xl, v, split, m, s)
                if j + 1 < Cg:
                    load(Un, u, h, r0, R, row_batch, row_start, j + 1)
                T.gemm(Xh, Eh, P1, transpose_B=True, clear_accum=True)
                if split:
                    T.gemm(Xl, Eh, P2, transpose_B=True, clear_accum=True)
                    T.gemm(Xh, El, P3, transpose_B=True, clear_accum=True)
                    T.gemm(Uh, Il, P4, transpose_B=True, clear_accum=True)
                if split_u:
                    T.gemm(Ul, Ih, P5, transpose_B=True, clear_accum=True)
                T.gemm(Uh, Ih, states, transpose_B=True)
                for m, s in T.Parallel(ROWS, Sp):
                    if split_u:
                        states[m, s] += P1[m, s] + (P2[m, s] + P3[m, s]) + (P4[m, s] + P5[m, s])
                    elif split:
                        states[m, s] += P1[m, s] + (P2[m, s] + P3[m, s]) + P4[m, s]
                    else:
                        states[m, s] += P1[m, s]
            for m, s in T.Parallel(ROWS, Sp):
                if r0 + m < R and s < S:
                    X[h, r0 + m, s] = states[m, s]

    return final_state_forward
