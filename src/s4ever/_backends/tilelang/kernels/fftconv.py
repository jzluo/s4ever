"""Fused FFT convolutions of real sequences with per-feature kernels, computed in float32.

Two batch rows u_a and u_b form one complex sequence z = u_a + i u_b. Because the two rows have the
same real kernel, the inverse transform of Z K is y_a + i y_b.

The kernels take the table tw of `fft.Layout.twiddles` and, for split transforms, the table T1 of
`fft.split_twiddles`. Some also take the table fr of `fft.Layout.frequencies`.
"""

import tilelang.language as T
import torch

from s4ever._backends.shared import ceil_div

from ..layout import FFT_POINTS_PER_THREAD as E
from ..runtime import compiled
from .dsl import cmul, f32, int_var, store, var
from .fft import Blocks, Columns, Regs, brev, idft, order_pad, pack, unpack

u64 = T.uint64
UNROLLED = 4
# The kernels check the bounds of their own loads, so TileLang's safe-memory pass is off. That pass
# would put each load in a branch, which keeps the loads from being issued together.
CONFIG = {"tl.disable_safe_memory_legalize": True}


def loaded(v):
    """Return the 64-bit word v in a variable, so that the kernel loads it once.

    TileLang would otherwise emit a load at each use of v.
    """
    return T.alloc_var(u64, init=v)[0]


def lengths(b0, B, L):
    return tuple(int_var(T.if_then_else(b0 + i < B, L, 0)) for i in (0, 1))


def pair_values(src, b0, lens, *lead):
    """Return value(n) for rows b0 and b0 + 1 of src[b, *lead, n].

    Row b0 + i is zero for n >= lens[i].
    """
    La, Lb = lens
    if torch.version.hip:
        # On AMD, LLVM keeps a load under a condition as a branch and waits for the load before it
        # issues the next one. The buffer loads of `BUFFER_LOADS_SOURCE` return zero past the row,
        # so they do not need a condition.
        load = f"s4ever_load_{BUFFER_TYPES[str(src.dtype)]}"

        def value(n):
            n = int_var(n)
            rows = ((b0, La), (b0 + 1, Lb))
            return [
                var(T.call_extern(f32, load, T.address_of(src[b, *lead, 0]), n, length))
                for b, length in rows
            ]

        return value
    zero = T.cast(0, src.dtype)

    def value(n):
        # On NVIDIA, the condition of each load is one comparison of the `int_var` n, so the loads
        # compile to predicated loads at offsets from one address.
        n = int_var(n)
        va = T.cast(T.if_then_else(n < La, src[b0, *lead, n], zero), f32)
        vb = T.cast(T.if_then_else(n < Lb, src[b0 + 1, *lead, n], zero), f32)
        return var(va), var(vb)

    return value


def staged(f, split, parts):
    """Return the function y(m) of `fft.FFT.gather`, staged in f's exchange buffer.

    Segment r < N1 of x holds the points x(m + NF r) for m < NF. For each part (value, r0, count),
    the segments r0 <= r < r0 + count hold value(n), and the other segments are zero.
    """
    NF, N1 = f.lay.NF, split.N1
    # When every part has at most UNROLLED segments at trace time, the segments are unrolled one
    # point at a time, so that no sums are held across segments. Otherwise the segments run as
    # loops, which keeps the code small enough for the instruction cache.
    if all(isinstance(count, int) and count <= UNROLLED for _, _, count in parts):
        terms = [
            (value, r, *(var(v) for v in split.root(2 * E * (r * split.k1 % N1))))
            for value, r0, count in parts
            for r in range(r0, r0 + count)
        ]
        for i in range(E):
            m = i * f.lay.seq_threads + f.t
            ys = [cmul(value(m + NF * r), (gr, gi)) for value, r, gr, gi in terms]
            store(f.buf, pack(sum(y[0] for y in ys), sum(y[1] for y in ys)), f.off + m)
        return lambda m: unpack(f.buf[f.off + m])
    yr, yi = T.alloc_local((E,), f32), T.alloc_local((E,), f32)
    for i in range(E):
        store(yr, 0.0, i)
        store(yi, 0.0, i)
    for value, r0, count in parts:
        with T.serial(count) as step:
            r = r0 + step
            gr, gi = (var(v) for v in split.root(2 * E * (r * split.k1 % N1)))
            for i in range(E):
                vr, vi = cmul(value(i * f.lay.seq_threads + f.t + NF * r), (gr, gi))
                store(yr, yr[i] + vr, i)
                store(yi, yi[i] + vi, i)
    for i in range(E):
        store(f.buf, pack(yr[i], yi[i]), f.off + i * f.lay.seq_threads + f.t)
    return lambda m: unpack(f.buf[f.off + m])


def sums(src, p, *idx):
    """Return the function y(m) of `fft.FFT.gather` from the output src of `columns_kernel`.

    y(m) is src[p, *idx, m], or zero when p is not a row of src.
    """
    P = src.shape[0]
    return lambda m: unpack(
        loaded(T.if_then_else(p < P, src[(T.min(p, P - 1), *idx, m)], T.uint64(0)))
    )


def load_pair(f, regs, split, src, b0, lens, *lead):
    if split is None:
        f.load(regs, pair_values(src, b0, lens, *lead))
    elif split.passes:
        f.gather(regs, split, sums(src, b0 // 2, *lead, split.k1))
    else:
        f.gather(
            regs, split, staged(f, split, [(pair_values(src, b0, lens, *lead), 0, split.N1 // 2)])
        )


def share(f, split, xr, xi, e):
    """Return the share w^(-n k1) x[e] of block k1 in its output n, and n.

    x[e] is point e of the output of `fft.FFT.inverse`, which is output n = point(e) + j NF / 2.
    The power w^(n k1) is w^(t' k1) share_root(e, j), where t' is the thread's index within its
    half.
    """
    wr, wi = cmul((split.A[0], split.A[1]), split.share_root(e, f.j))
    return cmul((xr[e], xi[e]), (wr, -wi)), f.point(e) + f.j * (f.lay.NF // 2)


def plus(v, ad, b, n):
    return v if ad is None else v + ad(b, n)


def addend_shape(B, addend):
    if not addend:
        return (1, 1, 1, 1)
    return T.dynamic("Ca"), B, T.dynamic("Ha"), T.dynamic("La")


def row_addend(ad, r):
    H = ad.shape[2]
    c, h = int_var(r // H), int_var(r % H)
    return lambda b, n: ad[c, b, h, n]


@T.macro
def store_rows(dst, b0, lens, lead, NT, ad, n, re, im):
    if n < lens[0]:
        store(dst, T.cast(plus(re / NT, ad, b0, n), dst.dtype), b0, *lead, n)
    if n < lens[1]:
        store(dst, T.cast(plus(im / NT, ad, b0 + 1, n), dst.dtype), b0 + 1, *lead, n)


@T.macro
def put_taps(gk, gkr, gD, has, NT, r, n, v):
    """Store the gradient v of entry n of circular kernel r in the gradients of its taps."""
    if T.And(has[0] != 0, n < gk.shape[1]):
        gk[r, n] = v
    if T.And(has[2] != 0, n == 0):
        gD[r] = v
    if T.And(has[1] != 0, NT - 1 - n < gkr.shape[1]):
        gkr[r, NT - 1 - n] = v


def intrinsic(name, *args):
    T.evaluate(T.call_extern("void", name, *args))


# The L2 cache hints of `L2_HINTS_SOURCE` are PTX instructions.
L2_HINTS = not torch.version.hip
# Whether the thread blocks of a split transform combine their shares of the outputs inside the
# kernels. On AMD, the blocks write their shares to memory for later kernels to sum. A combine
# inside the kernels would need device-scope fences. MI300X has one L2 per XCD, so each of these
# fences writes back the whole L2 and then invalidates it.
COMBINE = not torch.version.hip
L2_HINTS_SOURCE = """
__device__ __forceinline__ void s4ever_keep(void* p, unsigned long long v) {
  unsigned long long q;
  asm("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(q));
  asm volatile("st.global.L2::cache_hint.u64 [%0], %1, %2;" :: "l"(p), "l"(v), "l"(q) : "memory");
}
__device__ __forceinline__ void s4ever_discard(const void* p) {
  asm volatile("discard.global.L2 [%0], 128;" :: "l"(p) : "memory");
}
"""


# The AMD buffer loads of `pair_values`. A buffer resource holds the size of the buffer, and loads
# past that size return zero. 0x00020000 is the last word of the resource on CDNA GPUs, as in
# composable_kernel.
BUFFER_LOADS_SOURCE = """
__attribute__((device)) inline __amdgpu_buffer_rsrc_t s4ever_buffer(const void* p, int bytes) {
  return __builtin_amdgcn_make_buffer_rsrc((void*)p, (short)0, bytes, 0x00020000);
}
__attribute__((device)) inline float s4ever_load_f32(const void* p, int n, int len) {
  return __builtin_bit_cast(float,
      __builtin_amdgcn_raw_buffer_load_b32(s4ever_buffer(p, 4 * len), 4 * n, 0, 0));
}
__attribute__((device)) inline float s4ever_load_bf16(const void* p, int n, int len) {
  unsigned short v = __builtin_amdgcn_raw_buffer_load_b16(s4ever_buffer(p, 2 * len), 2 * n, 0, 0);
  return __builtin_bit_cast(float, (unsigned)v << 16);
}
__attribute__((device)) inline float s4ever_load_f16(const void* p, int n, int len) {
  unsigned short v = __builtin_amdgcn_raw_buffer_load_b16(s4ever_buffer(p, 2 * len), 2 * n, 0, 0);
  return (float)__builtin_bit_cast(_Float16, v);
}
"""
BUFFER_TYPES = {"float32": "f32", "bfloat16": "bf16", "float16": "f16"}


def store_kept(buf, v, *idx):
    """Store v at buf[idx], with an L2 evict_last policy when L2_HINTS is set.

    The policy keeps the value in L2 for the thread block that reads it next. Because the asm
    statement has a memory clobber, the compiler neither keeps memory values in registers across
    the store nor moves memory accesses across it.
    """
    if L2_HINTS:
        intrinsic("s4ever_keep", T.address_of(buf[idx]), v)
    else:
        store(buf, v, *idx)


def discard(buf, *idx):
    """Drop the 128-byte line at buf[idx] from L2 without writing it back."""
    intrinsic("s4ever_discard", T.address_of(buf[idx]))


def combiner(f, split, V, arrived, outputs, put, combines):
    """Return the macro combine(xr, xi, w, cnt, lead, out) for transforms split over N1 blocks.

    The macro forms the outputs y of the transform from the shares that each thread block computes
    with `share` from the output xr, xi of `fft.FFT.inverse`.

        y[n2 + NF n1] = sum_k1 w_N1^(-n1 k1) share_k1[n2]    for n1 < outputs
        w_N1 = exp(-2 pi i / N1)

    Each block increments the zeroed counter cnt[*lead, 0] when it arrives. The blocks that arrive
    before the last write their shares to w[*lead] (N1, NF) and then increment cnt[*lead, 1]. The
    last block waits until cnt[*lead, 1] is N1 - 1, sums the shares in the registers V (2 N1,) and
    passes each y[n] to put(*out, n, re, im). With three passes or without `combines`, the macro
    only writes the shares of every block to w.

    The macro assumes G = 1, which holds because the launchers split transforms only at
    NF >= 4096.
    """
    N1, NF = split.N1, f.lay.NF
    if split.passes or not combines:

        @T.macro
        def leave(xr, xi, w, cnt, lead, out):
            for e in T.unroll(E):
                (sr, si), n = share(f, split, xr, xi, e)
                store(w, pack(sr, si), *lead, split.k1, n)

        return leave

    def combined(xr, xi, w, lead, out, e):
        (sr, si), n = share(f, split, xr, xi, e)
        for k in range(N1):
            own = split.k1 == k
            vr, vi = unpack(T.if_then_else(own, T.uint64(0), w[(*lead, k, n)]))
            store(V, T.if_then_else(own, sr, vr), 2 * brev(k, N1))
            store(V, T.if_then_else(own, si, vi), 2 * brev(k, N1) + 1)
        idft(V, N1, outputs)
        for n1 in range(outputs):
            put(*out, int_var(n + NF * n1), V[2 * n1], V[2 * n1 + 1])

    @T.macro
    def combine(xr, xi, w, cnt, lead, out):
        if f.t == 0:
            arrived[0] = T.atomic_add(cnt[(*lead, 0)], 1, return_prev=True)
        T.sync_threads()
        if arrived[0] < N1 - 1:
            for e in T.unroll(E):
                (sr, si), n = share(f, split, xr, xi, e)
                store_kept(w, pack(sr, si), *lead, split.k1, n)
            # TileLang inserts barriers only between conflicting accesses to shared memory, so the
            # macro emits the fence and the barrier that order the stores to w before thread 0
            # increments cnt[*lead, 1].
            intrinsic("__threadfence")
            intrinsic("__syncthreads")
            if f.t == 0:
                T.atomic_add(cnt[(*lead, 1)], 1)
        else:
            if f.t == 0:
                while T.atomic_load(cnt[(*lead, 1)], memory_order="acquire") < N1 - 1:
                    pass
            intrinsic("__syncthreads")
            for e in T.unroll(E):
                combined(xr, xi, w, lead, out, e)
            # Because w is scratch, its lines are dropped from L2 after the reads, and L2 does
            # not write them back.
            if L2_HINTS:
                intrinsic("__syncthreads")
                for i in T.unroll(N1 * NF // 16 // f.lay.threads):
                    n = (i * f.lay.threads + f.t) * 16
                    discard(w, *lead, n // NF, n % NF)

    return combine


def mul(Z, Kf, c, h, k1, t, X, adjoint=False):
    for e in range(E):
        kr, ki = unpack(Kf[c, h, k1, e, t])
        xr, xi = cmul((Z[0][e], Z[1][e]), (kr, -ki if adjoint else ki))
        store(X[0], X[0][e] + xr if adjoint else xr, e)
        store(X[1], X[1][e] + xi if adjoint else xi, e)


def conj_mac(Z, X, A):
    for e in range(E):
        xr, xi = cmul((Z[0][e], -Z[1][e]), (X[0][e], X[1][e]))
        store(A[0], A[0][e] + xr, e)
        store(A[1], A[1][e] + xi, e)


def conj_mac_global(gK, group, c, h, k1, t, Z, X, it, NT):
    for e in range(E):
        xr, xi = cmul((Z[0][e], -Z[1][e]), (X[0][e], X[1][e]))
        pr, pi = unpack(T.if_then_else(it == 0, T.uint64(0), gK[group, c, h, k1, e, t]))
        store_kept(gK, pack(pr + xr / NT, pi + xi / NT), group, c, h, k1, e, t)


@compiled(pass_configs=CONFIG)
def forward_kernel(NF, C, dtype, N1, addend, combines):
    """Build the kernel that convolves u (B, H, L) with the kernels into y (B, C, H, L).

    Each thread block processes `rounds` rounds of G pairs of rows.

    Args:
        N1: the number of thread blocks per pair, or None for three passes. For three passes, u
            holds the transforms of the pairs' columns from `columns_kernel`.
        addend: whether ad[c, b, h, n] is added to y[b, c, h, n] in float32.
        combines: whether the N1 > 1 blocks of a pair combine their outputs through w. Without
            `combines` and for three passes, the blocks write their shares to w and do not use y,
            cnt and ad.

    Kernel arguments:
        Kf: the spectra (C, H, N1, E, NF / E) of the kernels, from `spectrum_kernel`.
        w: the shares (P, C, H, N1, NF), where P is B / 2 rounded up to a multiple of `rounds`.
        cnt: the counters (P, C, H, 2) of the blocks that combine, which the caller zeroes.
    """
    blocks = Blocks(NF, N1)
    lay, N1, passes, splits = blocks.lay, blocks.N1, blocks.passes, blocks.splits
    seq_threads, threads, G = lay.seq_threads, lay.threads, lay.G
    # Pw and Hw are P and H for split transforms, and 1 for the stand-ins that unsplit ones take.
    B, H, L, Pw, Hw, P = (T.dynamic(n) for n in ("B", "H", "L", "Pw", "Hw", "P"))
    if passes:
        u_shape, u_dtype, pairs = (P, H, N1, NF), u64, P
    else:
        u_shape, u_dtype, pairs = (B, H, L), dtype, T.ceildiv(B, 2)
    ad_shape = (C, B, H, T.dynamic("La")) if addend else (1, 1, 1, 1)

    @T.prim_func
    def fft_forward(
        u: T.Tensor(u_shape, u_dtype),
        Kf: T.Tensor((C, H, N1, E, seq_threads), u64),
        tw: T.Tensor((lay.n_twiddles, 2), f32),
        T1: T.Tensor((blocks.n_split_twiddles,), u64),
        y: T.Tensor((B, C, H, L), dtype),
        ad: T.Tensor(ad_shape, f32),
        w: T.Tensor((Pw, C, Hw, N1, NF), u64),
        cnt: T.Tensor((Pw, C, Hw, 2), T.int32),
        rounds: T.int32,
    ):
        with T.Kernel(T.ceildiv(pairs, G * rounds) * N1, H, threads=threads) as (bx, h):
            if torch.version.hip:
                T.import_source(BUFFER_LOADS_SOURCE)
            if splits:
                if L2_HINTS:
                    T.import_source(L2_HINTS_SOURCE)
                # T.annotate_restrict_buffers removes __restrict__ from u, which keeps the compiler
                # from moving the loads of a round above the barriers of earlier rounds.
                T.annotate_restrict_buffers(u)
            row_blocks = T.ceildiv(pairs, G * rounds)
            f, split, k1, row_block, seq = blocks.setup(tw, T1, bx, row_blocks)
            Xr = T.alloc_local((E,), f32)
            Xi = T.alloc_local((E,), f32)
            Yr = T.alloc_local((E,), f32)
            Yi = T.alloc_local((E,), f32)
            Zr = T.alloc_local((E,), f32)
            Zi = T.alloc_local((E,), f32)
            V = None if passes else T.alloc_local((2 * N1,), f32)
            arrived = T.alloc_shared((1,), T.int32)
            combine = (
                combiner(f, split, V, arrived, N1 // 2, store_rows, combines) if splits else None
            )
            for it in T.serial(rounds):
                b0 = 2 * ((row_block * rounds + it) * G + seq)
                lens = lengths(b0, B, L)
                regs = Regs((Xr, Xi), (Yr, Yi))
                load_pair(f, regs, split, u, b0, lens, h)
                f.forward(regs)
                U = regs.cur
                if C > 1:  # The inverse transforms overwrite U in regs.cur.
                    for e in T.unroll(E):
                        Zr[e] = U[0][e]
                        Zi[e] = U[1][e]
                    U = (Zr, Zi)
                for c in T.serial(C):
                    prod = Regs(regs.other, regs.cur)
                    mul(U, Kf, c, h, k1, f.t, prod.cur)
                    xr, xi = f.inverse(prod)
                    add = (lambda b, n, c=c: ad[c, b, h, n]) if addend else None
                    if splits:
                        out = (y, b0, lens, (c, h), N1 * NF, add)
                        combine(xr, xi, w, cnt, (b0 // 2, c, h), out)
                    else:
                        # Only the first half holds outputs, because L <= NF / 2.
                        for e in T.unroll(E):
                            n = int_var(f.point(e))
                            if T.And(f.j == 0, n < lens[0]):
                                y[b0, c, h, n] = T.cast(plus(xr[e] / NF, add, b0, n), dtype)
                            if T.And(f.j == 0, n < lens[1]):
                                y[b0 + 1, c, h, n] = T.cast(plus(xi[e] / NF, add, b0 + 1, n), dtype)

    return fft_forward


@compiled(pass_configs=CONFIG)
def backward_kernel(NF, C, dtype, acc_regs, parts, N1, combines):
    """Build the kernel that computes the gradients of `forward_kernel` that `parts` names.

    The gradients are gu (B, H, L) and gK (S, C, H, N1, E, NF / E), which is packed like Kf. A
    thread block transforms G sequences at a time, and the pairs of rows that one of these
    sequences holds over the block's rounds form a group. For each of the S groups, the kernel
    sums the terms conj(U) GY / NT into the group's row of gK, where U and GY are the spectra of
    u and gy. For three passes, u and gy hold column transforms.

    Args:
        acc_regs: whether the sums are formed in registers, which is correct only for C = 1.
        combines: whether, for S > 1, the last thread block to finish for each (h, k1) adds the S
            sums into gK[0] in a fixed order, so that the result does not depend on which block
            finishes last. Without `combines`, gK holds the S sums.

    Kernel arguments:
        cntK: the counts of the finished blocks of each (h, k1), which the caller zeroes.
        w, cnt: the shares and counters of the split transforms of gu, as in `forward_kernel`.
    """
    blocks = Blocks(NF, N1)
    lay, N1, passes, splits = blocks.lay, blocks.N1, blocks.passes, blocks.splits
    seq_threads, threads, G = lay.seq_threads, lay.threads, lay.G
    want_gu, want_gK = "gu" in parts, "gK" in parts
    # Hk is H when the kernel computes gK, and 1 for the stand-ins of gK and cntK otherwise.
    names = ("B", "H", "L", "S", "Hk", "Pw", "Hw", "P")
    B, H, L, groups, Hk, Pw, Hw, P = (T.dynamic(n) for n in names)
    if passes:
        (u_shape, gy_shape), in_dtype, pairs = ((P, H, N1, NF), (P, C, H, N1, NF)), u64, P
    else:
        (u_shape, gy_shape), in_dtype, pairs = ((B, H, L), (B, C, H, L)), dtype, T.ceildiv(B, 2)
    NT = N1 * NF
    points_per_thread = ceil_div(NF, threads)  # the points of each sum of gK that a thread adds

    @T.macro
    def reduce(gK, cntK, arrived, row_blocks, h, k1, tx, Xr, Xi):
        """Add the sums gK[1:] into gK[0] in the last thread block to finish for (h, k1)."""
        # As in `combiner`, the fence and the barrier order the block's stores to gK before
        # thread 0 counts the block as finished.
        intrinsic("__threadfence")
        intrinsic("__syncthreads")
        if tx == 0:
            arrived[0] = T.atomic_add(cntK[h, k1], 1, return_prev=True)
            intrinsic("__threadfence")
        T.sync_threads()
        if arrived[0] == row_blocks - 1:
            for c in T.serial(C):
                for i in T.unroll(points_per_thread):
                    Xr[i] = 0.0
                    Xi[i] = 0.0
                for group in T.serial(groups):
                    for i in T.unroll(points_per_thread):
                        point = i * threads + tx
                        if point < NF:
                            vr, vi = unpack(
                                gK[group, c, h, k1, point // seq_threads, point % seq_threads]
                            )
                            Xr[i] = Xr[i] + vr
                            Xi[i] = Xi[i] + vi
                for i in T.unroll(points_per_thread):
                    point = i * threads + tx
                    if point < NF:
                        gK[0, c, h, k1, point // seq_threads, point % seq_threads] = pack(
                            Xr[i], Xi[i]
                        )
                if L2_HINTS:  # No kernel reads gK[1:] after the last block adds the sums.
                    intrinsic("__syncthreads")
                    for group in T.serial(groups - 1):
                        for i in T.unroll(ceil_div(NF, 16 * threads)):
                            point = (i * threads + tx) * 16
                            if point < NF:
                                discard(
                                    gK,
                                    group + 1,
                                    c,
                                    h,
                                    k1,
                                    point // seq_threads,
                                    point % seq_threads,
                                )

    @T.prim_func
    def fft_backward(
        u: T.Tensor(u_shape, in_dtype),
        gy: T.Tensor(gy_shape, in_dtype),
        Kf: T.Tensor((C, H, N1, E, seq_threads), u64),
        tw: T.Tensor((lay.n_twiddles, 2), f32),
        T1: T.Tensor((blocks.n_split_twiddles,), u64),
        gu: T.Tensor((B, H, L), dtype),
        gK: T.Tensor((groups, C, Hk, N1, E, seq_threads), u64),
        cntK: T.Tensor((Hk, N1), T.int32),
        w: T.Tensor((Pw, 1, Hw, N1, NF), u64),
        cnt: T.Tensor((Pw, 1, Hw, 2), T.int32),
        rounds: T.int32,
    ):
        with T.Kernel(T.ceildiv(pairs, G * rounds) * N1, H, threads=threads) as (bx, h):
            if torch.version.hip:
                T.import_source(BUFFER_LOADS_SOURCE)
            if L2_HINTS:
                T.import_source(L2_HINTS_SOURCE)
            if splits:
                # As in `forward_kernel`, u and gy are not __restrict__.
                T.annotate_restrict_buffers(u, gy)
            row_blocks = T.ceildiv(pairs, G * rounds)
            f, split, k1, row_block, seq = blocks.setup(tw, T1, bx, row_blocks)
            Xr = T.alloc_local((E,), f32)
            Xi = T.alloc_local((E,), f32)
            Yr = T.alloc_local((E,), f32)
            Yi = T.alloc_local((E,), f32)
            Zr = T.alloc_local((E,), f32)
            Zi = T.alloc_local((E,), f32)
            Vr = T.alloc_local((E,), f32)
            Vi = T.alloc_local((E,), f32)
            Ar = T.alloc_local((E,), f32)
            Ai = T.alloc_local((E,), f32)
            V = None if passes else T.alloc_local((2 * N1,), f32)
            arrived = T.alloc_shared((1,), T.int32)
            group = row_block * G + seq
            combine = (
                combiner(f, split, V, arrived, N1 // 2, store_rows, combines) if splits else None
            )
            if acc_regs:
                for e in T.unroll(E):
                    Ar[e] = 0.0
                    Ai[e] = 0.0
            for it in T.serial(rounds):
                b0 = 2 * ((row_block * rounds + it) * G + seq)
                lens = lengths(b0, B, L)
                if want_gK:
                    u_regs = Regs((Xr, Xi), (Yr, Yi))
                    load_pair(f, u_regs, split, u, b0, lens, h)
                    f.forward(u_regs)
                    for e in T.unroll(E):
                        Zr[e] = u_regs.cur[0][e]
                        Zi[e] = u_regs.cur[1][e]
                if want_gu:
                    for e in T.unroll(E):
                        Vr[e] = 0.0
                        Vi[e] = 0.0
                for c in T.serial(C):
                    gy_regs = Regs((Xr, Xi), (Yr, Yi))
                    load_pair(f, gy_regs, split, gy, b0, lens, c, h)
                    f.forward(gy_regs)
                    if want_gK:
                        if acc_regs:
                            conj_mac((Zr, Zi), gy_regs.cur, (Ar, Ai))
                        else:
                            conj_mac_global(gK, group, c, h, k1, f.t, (Zr, Zi), gy_regs.cur, it, NT)
                    if want_gu:
                        mul(gy_regs.cur, Kf, c, h, k1, f.t, (Vr, Vi), adjoint=True)
                if want_gu:
                    xr, xi = f.inverse(Regs((Vr, Vi), (Yr, Yi)))
                    if splits:
                        combine(xr, xi, w, cnt, (b0 // 2, 0, h), (gu, b0, lens, (h,), NT, None))
                    else:
                        for e in T.unroll(E):
                            n = int_var(f.point(e))
                            if T.And(f.j == 0, n < lens[0]):
                                gu[b0, h, n] = T.cast(xr[e] / NF, dtype)
                            if T.And(f.j == 0, n < lens[1]):
                                gu[b0 + 1, h, n] = T.cast(xi[e] / NF, dtype)
            if want_gK:
                if acc_regs:
                    for e in T.unroll(E):
                        store_kept(gK, pack(Ar[e] / NT, Ai[e] / NT), group, 0, h, k1, e, f.t)
                if combines and groups > 1:
                    reduce(gK, cntK, arrived, row_blocks, h, k1, T.get_thread_binding(), Xr, Xi)

    return fft_backward


def circ(k, kr, r, NT, has_k, has_kr):
    """Return the functions lower and upper that give circular kernel r of `spectrum_kernel`.

    lower(n) gives entry n for n < NT / 2, and upper(n) gives it for n >= NT / 2. Neither includes
    D.
    """
    Rk, Lk, R2, Lr = k.shape[0], k.shape[1], kr.shape[0], kr.shape[1]

    def lower(n):
        v = k[T.min(r, Rk - 1), T.min(n, Lk - 1)] * T.cast(T.And(has_k != 0, n < Lk), f32)
        return var(v), 0.0

    def upper(n):
        i = NT - 1 - n
        v = kr[T.min(r, R2 - 1), T.min(i, Lr - 1)] * T.cast(T.And(has_kr != 0, i < Lr), f32)
        return var(v), 0.0

    return lower, upper


@compiled(pass_configs=CONFIG)
def spectrum_kernel(NF, N1):
    """Build the kernel that computes the spectra Kf (R, N1, E, NF / E) of the circular kernels.

    The circular kernels have NT = N1 NF points, and Kf has the layout that `forward_kernel` and
    `backward_kernel` read. The flags has_k, has_kr and has_D select the terms of circular kernel r,
    and the lengths Lk and Lr are at most NT / 2.

        entry n          = k[r, n]     for n < Lk
        entry NT - 1 - i = kr[r, i]    for i < Lr
        entry 0         += D[r]

    For three passes (N1 None), k holds the column transforms from `columns_kernel`, and kr is
    unused.
    """
    blocks = Blocks(NF, N1)
    lay, N1, passes, splits = blocks.lay, blocks.N1, blocks.passes, blocks.splits
    seq_threads, threads, G = lay.seq_threads, lay.threads, lay.G
    # Rk, R2 and RD are 1 for the stand-ins of absent inputs.
    R, Rk, Lk, R2, Lr, RD = (T.dynamic(n) for n in ("R", "Rk", "Lk", "R2", "Lr", "RD"))
    k_shape, k_dtype = ((R, N1, NF), u64) if passes else ((Rk, Lk), f32)

    @T.prim_func
    def fft_spectrum(
        k: T.Tensor(k_shape, k_dtype),
        kr: T.Tensor((R2, Lr), f32),
        D: T.Tensor((RD,), f32),
        tw: T.Tensor((lay.n_twiddles, 2), f32),
        T1: T.Tensor((blocks.n_split_twiddles,), u64),
        Kf: T.Tensor((R, N1, E, seq_threads), u64),
        has_k: T.int32,
        has_kr: T.int32,
        has_D: T.int32,
    ):
        with T.Kernel(T.ceildiv(R, G) * N1, threads=threads) as bx:
            f, split, k1, row_block, seq = blocks.setup(tw, T1, bx, T.ceildiv(R, G))
            Xr = T.alloc_local((E,), f32)
            Xi = T.alloc_local((E,), f32)
            Yr = T.alloc_local((E,), f32)
            Yi = T.alloc_local((E,), f32)
            r = row_block * G + seq
            lower, upper = circ(k, kr, T.min(r, R - 1), N1 * NF, has_k, has_kr)
            regs = Regs((Xr, Xi), (Yr, Yi))
            if passes:
                f.gather(regs, split, sums(k, r, k1))
            elif splits:
                lo = int_var(T.ceildiv(T.if_then_else(has_k != 0, k.shape[1], 0), NF))
                up = int_var(T.ceildiv(T.if_then_else(has_kr != 0, kr.shape[1], 0), NF))
                f.gather(regs, split, staged(f, split, [(lower, 0, lo), (upper, N1 - up, up)]))
            else:
                f.load(regs, lower, lambda n: upper(n + NF // 2))
            f.forward(regs)
            d = var(D[T.min(r, RD - 1)] * T.cast(has_D != 0, f32))
            if r < R:
                for e in T.unroll(E):
                    Kf[r, k1, e, f.t] = pack(regs.cur[0][e] + d, regs.cur[1][e])

    return fft_spectrum


@T.macro
def group_sums(Xr, Xi, gK, groups, row, k1, t):
    for e in T.unroll(E):
        Xr[e] = 0.0
        Xi[e] = 0.0
    for group in T.serial(groups):
        for e in T.unroll(E):
            vr, vi = unpack(gK[group, row, k1, e, t])
            Xr[e] = Xr[e] + vr
            Xi[e] = Xi[e] + vi


@compiled(pass_configs=CONFIG)
def spectrum_grad_kernel(NF, N1, combines):
    """Build the kernel that computes the gradients gk, gkr and gD of `spectrum_kernel`'s inputs.

    The kernel reads the output gK of a `backward_kernel` with the same `combines`. The flags has_k,
    has_kr and has_D select the gradients, and split transforms use w and cnt as in
    `forward_kernel`.
    """
    blocks = Blocks(NF, N1)
    lay, N1, passes, splits = blocks.lay, blocks.N1, blocks.passes, blocks.splits
    seq_threads, threads, G = lay.seq_threads, lay.threads, lay.G
    # Rw is R for split transforms, and the unsplit kernels take stand-ins for w and cnt.
    names = ("S", "R", "Rk", "Lk", "R2", "Lr", "RD", "Rw")
    groups, R, Rk, Lk, R2, Lr, RD, Rw = (T.dynamic(n) for n in names)
    NT = N1 * NF

    # The gradients of the taps are the real parts of the inverse transform, because the taps are
    # real.
    def put(gk, gkr, gD, has, r, n, re, im):
        put_taps(gk, gkr, gD, has, NT, r, n, re)

    @T.prim_func
    def fft_spectrum_grad(
        gK: T.Tensor((groups, R, N1, E, seq_threads), u64),
        tw: T.Tensor((lay.n_twiddles, 2), f32),
        T1: T.Tensor((blocks.n_split_twiddles,), u64),
        gk: T.Tensor((Rk, Lk), f32),
        gkr: T.Tensor((R2, Lr), f32),
        gD: T.Tensor((RD,), f32),
        w: T.Tensor((Rw, N1, NF), u64),
        cnt: T.Tensor((Rw, 2), T.int32),
        has_k: T.int32,
        has_kr: T.int32,
        has_D: T.int32,
    ):
        with T.Kernel(T.ceildiv(R, G) * N1, threads=threads) as bx:
            if splits and not passes and L2_HINTS:
                T.import_source(L2_HINTS_SOURCE)
            f, split, k1, row_block, seq = blocks.setup(tw, T1, bx, T.ceildiv(R, G))
            Xr = T.alloc_local((E,), f32)
            Xi = T.alloc_local((E,), f32)
            Yr = T.alloc_local((E,), f32)
            Yi = T.alloc_local((E,), f32)
            V = None if passes else T.alloc_local((2 * N1,), f32)
            arrived = T.alloc_shared((1,), T.int32)
            r = row_block * G + seq
            if combines:
                for e in T.unroll(E):
                    vr, vi = unpack(gK[0, T.min(r, R - 1), k1, e, f.t])
                    Xr[e] = vr
                    Xi[e] = vi
            else:
                group_sums(Xr, Xi, gK, groups, T.min(r, R - 1), k1, f.t)
            xr, xi = f.inverse(Regs((Xr, Xi), (Yr, Yi)))
            if splits:
                combine = combiner(f, split, V, arrived, N1, put, combines)
                combine(xr, xi, w, cnt, (r,), (gk, gkr, gD, (has_k, has_kr, has_D), r))
            else:
                for e in T.unroll(E):
                    n = int_var(f.point(e) + f.j * (NF // 2))
                    if r < R:
                        put_taps(gk, gkr, gD, (has_k, has_kr, has_D), NF, r, n, xr[e])

    return fft_spectrum_grad


@compiled(pass_configs=CONFIG)
def columns_kernel(N1, NF, kind, dtype):
    """Build the first pass of three, which transforms the columns of each sequence x.

        c[k1, m] = sum_n1 x[m + NF n1] w_N1^(n1 k1)    for m < NF
        w_N1 = exp(-2 pi i / N1)

    Block k1 of the second pass reads c[k1] through `fft.FFT.gather`. For "pairs", the sequences
    are u[2 p] + i u[2 p + 1], which are zero past L, and L is at most NT / 2. For "circular", the
    sequences are the circular kernels of `spectrum_kernel` without D. The tables tw and fr are
    those of N1-point transforms.
    """
    cols = Columns(N1, NF)
    lay, G = cols.lay, cols.G
    NT = N1 * NF
    names = ("B", "R", "L", "P", "Rk", "Lk", "R2", "Lr")
    B, R, L, P, Rk, Lk, R2, Lr = (T.dynamic(n) for n in names)

    if kind == "pairs":

        @T.prim_func
        def pairs(
            u: T.Tensor((B, R, L), dtype),
            tw: T.Tensor((lay.n_twiddles, 2), f32),
            fr: T.Tensor((E, lay.seq_threads), T.int32),
            c: T.Tensor((P, R, N1, NF), u64),
        ):
            with T.Kernel(NF // G, P, R, threads=cols.threads) as (bx, p, r):
                if torch.version.hip:
                    T.import_source(BUFFER_LOADS_SOURCE)
                f, regs, m = cols.setup(tw, fr, bx, T.get_thread_binding())
                value = pair_values(u, 2 * p, lengths(2 * p, B, L), r)
                f.load(regs, lambda n: value(m + NF * n))
                f.forward(regs)
                for e in T.unroll(E):
                    c[p, r, fr[e, f.t], m] = pack(regs.cur[0][e], regs.cur[1][e])

        return pairs

    @T.prim_func
    def circular(
        k: T.Tensor((Rk, Lk), f32),
        kr: T.Tensor((R2, Lr), f32),
        tw: T.Tensor((lay.n_twiddles, 2), f32),
        fr: T.Tensor((E, lay.seq_threads), T.int32),
        c: T.Tensor((R, N1, NF), u64),
        has_k: T.int32,
        has_kr: T.int32,
    ):
        with T.Kernel(NF // G, R, threads=cols.threads) as (bx, r):
            f, regs, m = cols.setup(tw, fr, bx, T.get_thread_binding())
            lower, upper = circ(k, kr, r, NT, has_k, has_kr)
            f.load(regs, lambda n: lower(m + NF * n), lambda n: upper(m + NF * (n + N1 // 2)))
            f.forward(regs)
            for e in T.unroll(E):
                c[r, fr[e, f.t], m] = pack(regs.cur[0][e], regs.cur[1][e])

    return circular


@compiled(pass_configs=CONFIG)
def columns_inverse_kernel(N1, NF, kind, dtype, addend):
    """Build the last pass of three, which sums the shares w of the second pass.

        y[n2 + NF n1] = sum_k1 w_N1^(-n1 k1) w[k1, n2]

    For "pairs", the kernel divides each sum by NT and stores its real and imaginary parts in rows
    2 p and 2 p + 1 of y, adding ad as in `forward_kernel` with `addend`. For "taps", the real parts
    are the gradients gk, gkr and gD of `spectrum_grad_kernel`. The tables tw and fr are those of
    N1-point transforms.
    """
    cols = Columns(N1, NF)
    lay, G = cols.lay, cols.G
    NT = N1 * NF
    names = ("B", "R", "L", "P", "Rk", "Lk", "R2", "Lr", "RD")
    B, R, L, P, Rk, Lk, R2, Lr, RD = (T.dynamic(n) for n in names)

    if kind == "pairs":

        @T.prim_func
        def pairs(
            w: T.Tensor((P, R, N1, NF), u64),
            tw: T.Tensor((lay.n_twiddles, 2), f32),
            fr: T.Tensor((E, lay.seq_threads), T.int32),
            y: T.Tensor((B, R, L), dtype),
            ad: T.Tensor(addend_shape(B, addend), f32),
        ):
            with T.Kernel(NF // G, P, R, threads=cols.threads) as (bx, p, r):
                f, regs, m = cols.setup(tw, fr, bx, T.get_thread_binding())
                load_bins(regs, f, lambda q: unpack(loaded(w[p, r, q, m])))
                xr, xi = f.inverse(regs)
                lens = lengths(2 * p, B, L)
                add = row_addend(ad, r) if addend else None
                for e in T.unroll(E):
                    if f.j == 0:  # Only the points n < NT / 2 are stored, because L <= NT / 2.
                        n = int_var(m + NF * f.point(e))
                        store_rows(y, 2 * p, lens, (r,), NT, add, n, xr[e], xi[e])

        return pairs

    @T.prim_func
    def taps(
        w: T.Tensor((R, N1, NF), u64),
        tw: T.Tensor((lay.n_twiddles, 2), f32),
        fr: T.Tensor((E, lay.seq_threads), T.int32),
        gk: T.Tensor((Rk, Lk), f32),
        gkr: T.Tensor((R2, Lr), f32),
        gD: T.Tensor((RD,), f32),
        has_k: T.int32,
        has_kr: T.int32,
        has_D: T.int32,
    ):
        with T.Kernel(NF // G, R, threads=cols.threads) as (bx, r):
            f, regs, m = cols.setup(tw, fr, bx, T.get_thread_binding())
            load_bins(regs, f, lambda q: unpack(loaded(w[r, q, m])))
            xr, _ = f.inverse(regs)
            for e in T.unroll(E):
                n = int_var(m + NF * (f.point(e) + f.j * (N1 // 2)))
                put_taps(gk, gkr, gD, (has_k, has_kr, has_D), NT, r, n, xr[e])

    return taps


@compiled(pass_configs=CONFIG)
def combine_kernel(N1, NF, kind, dtype, addend):
    """Build the kernel that sums the shares w of split transforms.

    Split kernels that are built without `combines` write their shares to w. The transforms are
    split over N1 <= MAX_SPLIT thread blocks, and the sums, the kinds and `addend` are as in
    `columns_inverse_kernel`.
    """
    NT = N1 * NF
    threads = 256

    def load(V, w, n, *idx):
        for k in range(N1):
            vr, vi = unpack(loaded(w[(*idx, k, n)]))
            store(V, vr, 2 * brev(k, N1))
            store(V, vi, 2 * brev(k, N1) + 1)

    if kind == "pairs":
        B, R, L, P = (T.dynamic(n) for n in ("B", "R", "L", "P"))

        @T.prim_func
        def combine_pairs(
            w: T.Tensor((P, R, N1, NF), u64),
            y: T.Tensor((B, R, L), dtype),
            ad: T.Tensor(addend_shape(B, addend), f32),
        ):
            # The grid index r varies before p, so that consecutive thread blocks read adjacent
            # rows of w.
            with T.Kernel(NF // threads, R, P, threads=threads) as (bx, r, p):
                n = bx * threads + T.get_thread_binding()
                V = T.alloc_local((2 * N1,), f32)
                load(V, w, n, p, r)
                idft(V, N1, N1 // 2)
                lens = lengths(2 * p, B, L)
                add = row_addend(ad, r) if addend else None
                for n1 in range(N1 // 2):
                    m = int_var(n + NF * n1)
                    store_rows(y, 2 * p, lens, (r,), NT, add, m, V[2 * n1], V[2 * n1 + 1])

        return combine_pairs

    names = ("R", "Rk", "Lk", "R2", "Lr", "RD")
    R, Rk, Lk, R2, Lr, RD = (T.dynamic(n) for n in names)

    @T.prim_func
    def combine_taps(
        w: T.Tensor((R, N1, NF), u64),
        gk: T.Tensor((Rk, Lk), f32),
        gkr: T.Tensor((R2, Lr), f32),
        gD: T.Tensor((RD,), f32),
        has_k: T.int32,
        has_kr: T.int32,
        has_D: T.int32,
    ):
        with T.Kernel(NF // threads, R, threads=threads) as (bx, r):
            n = bx * threads + T.get_thread_binding()
            V = T.alloc_local((2 * N1,), f32)
            load(V, w, n, r)
            idft(V, N1, N1)
            for n1 in range(N1):
                put_taps(
                    gk, gkr, gD, (has_k, has_kr, has_D), NT, r, int_var(n + NF * n1), V[2 * n1]
                )

    return combine_taps


def separate(z, z_neg):
    """Return the transforms A[q] and B[q] of the real sequences a and b.

    z and z_neg are the transform Z of a + i b at q and at -q.

        A[q] = (Z[q] + conj Z[-q]) / 2
        B[q] = (Z[q] - conj Z[-q]) / 2i
    """
    (zr, zi), (nr, ni) = z, z_neg
    return ((zr + nr) / 2, (zi - ni) / 2), ((zi + ni) / 2, (nr - zr) / 2)


class Pair:
    """Rows a = 2 p and b = 2 p + 1 of R rows, and the words of f's buffer that hold their points.

    Word at(q) holds point q. Reads use the rows read_a and read_b, which are ra and rb clamped to
    R - 1.
    """

    def __init__(self, f, p, R):
        self.f = f
        self.ra, self.rb = 2 * p, 2 * p + 1
        self.read_a, self.read_b = T.min(self.ra, R - 1), T.min(self.rb, R - 1)
        self.pad = order_pad(f.lay)

    def at(self, q):
        return self.f.off + self.pad(q)

    def scatter(self, x):
        """Store x(e) at the word of the frequency of the thread's point e."""
        for e in range(E):
            store(self.f.buf, x(e), self.at(self.f.fr[e, self.f.t]))

    def scatter_regs(self, regs):
        self.scatter(lambda e: pack(regs.cur[0][e], regs.cur[1][e]))

    def hermitian(self, word, dst):
        """Store Ma + i Mb in dst, the Hermitian parts of the gradients gKa and gKb of rows a and b.

            Ma[q] = (gKa[q] + conj gKa[-q]) / 2

        The inverse transform of Ma + i Mb is Re(ga) + i Re(gb), where ga and gb are the inverse
        transforms of gKa and gKb. word(k, row, e) is the packed gradient of the thread's point e
        in row a (k = 0) or row b (k = 1).
        """
        for k, row in enumerate((self.read_a, self.read_b)):
            self.scatter(lambda e, k=k, row=row: word(k, row, e))
            for e in range(E):
                hr, hi = separate(unpack(word(k, row, e)), self.partner(e))[0]
                if k == 0:
                    store(dst[0], hr, e)
                    store(dst[1], hi, e)
                else:
                    store(dst[0], dst[0][e] - hi, e)
                    store(dst[1], dst[1][e] + hr, e)

    def mirror(self, q):
        """Return the index of the word that holds point -q.

        The index -q mod NF goes through an `int_var`, because TileLang's canonical simplifier
        miscompiles floor divisions of negative multiples, which the padding of -q would contain.
        For example, it rewrites floordiv(-floordiv(x, 2), 2) to floordiv(x, -4), which is -1 at
        x = 1 where the original is 0.
        """
        return self.at(int_var(T.if_then_else(q == 0, 0, self.f.lay.NF - q)))

    def partner(self, e):
        """Return the point at the negated frequency of the thread's point e, after `scatter`."""
        return unpack(self.f.buf[self.mirror(self.f.fr[e, self.f.t])])


def bins(kf, kfr, ra, rb, s, Lk):
    """Return value(q), bin q of the spectrum kf[ra] + i kfr[rb] spread over NF = s Lk bins.

    kf and kfr hold half spectra of Lk points. Bin s j is bin j of the full spectrum
    kf[ra] + i kfr[rb], and the other bins are zero.
    """

    def value(q):
        j = q // s
        up = T.cast(j > Lk // 2, T.int32)
        m = j + up * (Lk - 2 * j)  # Bins j > Lk / 2 are the conjugates of bins Lk - j.
        sign = (1 - 2 * T.cast(up, f32)) * T.cast(q % s == 0, f32)
        live = T.cast(q % s == 0, f32)
        ar, ai = kf[ra, m, 0] * live, kf[ra, m, 1] * sign
        br, bi = kfr[rb, m, 0] * live, kfr[rb, m, 1] * sign
        return ar - bi, ai + br

    return value


def load_bins(regs, f, value):
    """Load value(q) at the frequency q of each of the thread's points into regs.cur."""
    for e in range(E):
        vr, vi = value(f.fr[e, f.t])
        store(regs.cur[0], vr, e)
        store(regs.cur[1], vi, e)


def copy_regs(dst, src):
    for e in range(E):
        store(dst[0], src[0][e], e)
        store(dst[1], src[1][e], e)


@T.macro
def put_word(buf, i, v, cond):
    if cond:
        buf[i] = v


@T.macro
def put_bin(out, row, j, re, im, cond):
    if cond:
        out[row, j, 0] = re
        out[row, j, 1] = im


@compiled(pass_configs=CONFIG)
def spectra_kernel(NF, s, bidirectional):
    """Build the kernel that computes the spectra Kf (R, 1, E, NF / E) from half spectra.

    Kf equals the output of `spectrum_kernel` with N1 = 1. The input kf holds the half spectra of
    the kernels k, and with `bidirectional`, kfr holds those of kr. `kernels.cauchy` scales both by
    1 / Lk, where Lk = NF / s. Because only every s-th bin is nonzero, the inverse transform repeats
    each k and kr s times, and `spectra_kernel` forms the circular kernels from their first L
    taps. With has_D, D is added at n = 0.
    """
    blocks = Blocks(NF, 1)
    lay = blocks.lay
    Lk, half = NF // s, NF // 2
    seq_threads, threads, G = lay.seq_threads, lay.threads, lay.G
    R, R2, RD = (T.dynamic(n) for n in ("R", "R2", "RD"))

    def circular(buf, at, Da, Db, L):
        """Return the functions value and upper of `fft.FFT.load` for the circular kernels a + i b.

        They read the taps from the first halves of the repeated kernels in buf. With
        `bidirectional`, buf holds k + i kr of row a from word at(0) and that of row b from word
        at(NF / 2). Without `bidirectional`, buf holds ka + i kb from word at(0).
        """

        def taps(n):
            a = unpack(buf[at(T.min(n, L - 1))])
            if not bidirectional:
                return a
            return a[0], unpack(buf[at(half + T.min(n, L - 1))])[0]

        def value(n):
            ka, kb = taps(n)
            live, d = T.cast(n < L, f32), T.cast(n == 0, f32)
            return ka * live + Da * d, kb * live + Db * d

        def upper(n):  # the point at NF / 2 + n is kr[NF / 2 - 1 - n]
            i = half - 1 - n
            live = T.cast(i < L, f32)
            a = unpack(buf[at(T.min(i, L - 1))])[1]
            b = unpack(buf[at(half + T.min(i, L - 1))])[1]
            return a * live, b * live

        return value, upper if bidirectional else None

    @T.prim_func
    def fft_spectra(
        kf: T.Tensor((R, Lk // 2 + 1, 2), f32),
        kfr: T.Tensor((R2, Lk // 2 + 1, 2), f32),
        D: T.Tensor((RD,), f32),
        tw: T.Tensor((lay.n_twiddles, 2), f32),
        fr: T.Tensor((E, seq_threads), T.int32),
        Kf: T.Tensor((R, 1, E, seq_threads), u64),
        L: T.int32,
        has_D: T.int32,
    ):
        with T.Kernel(T.ceildiv(T.ceildiv(R, 2), G), threads=threads) as bx:
            f, _, _, row_block, seq = blocks.setup(tw, None, bx, 1, fr)
            Xr = T.alloc_local((E,), f32)
            Xi = T.alloc_local((E,), f32)
            Yr = T.alloc_local((E,), f32)
            Yi = T.alloc_local((E,), f32)
            Ar = T.alloc_local((E,), f32)
            Ai = T.alloc_local((E,), f32)
            pair = Pair(f, row_block * G + seq, R)
            if bidirectional:
                load_bins(
                    Regs((Xr, Xi), (Yr, Yi)), f, bins(kf, kfr, pair.read_a, pair.read_a, s, Lk)
                )
                copy_regs((Ar, Ai), f.inverse(Regs((Xr, Xi), (Yr, Yi))))
                load_bins(
                    Regs((Xr, Xi), (Yr, Yi)), f, bins(kf, kfr, pair.read_b, pair.read_b, s, Lk)
                )
                yr, yi = f.inverse(Regs((Xr, Xi), (Yr, Yi)))
                for e in T.unroll(E):
                    put_word(f.buf, pair.at(f.point(e)), pack(Ar[e], Ai[e]), f.j == 0)
                    put_word(f.buf, pair.at(half + f.point(e)), pack(yr[e], yi[e]), f.j == 0)
            else:
                load_bins(
                    Regs((Xr, Xi), (Yr, Yi)), f, bins(kf, kf, pair.read_a, pair.read_b, s, Lk)
                )
                xr, xi = f.inverse(Regs((Xr, Xi), (Yr, Yi)))
                for e in T.unroll(E):
                    put_word(f.buf, pair.at(f.point(e)), pack(xr[e], xi[e]), f.j == 0)
            Da = D[T.min(pair.ra, RD - 1)] * T.cast(has_D != 0, f32)
            Db = D[T.min(pair.rb, RD - 1)] * T.cast(has_D != 0, f32)
            regs = Regs((Xr, Xi), (Yr, Yi))
            f.load(regs, *circular(f.buf, pair.at, Da, Db, L))
            f.forward(regs)
            pair.scatter_regs(regs)
            for e in T.unroll(E):
                a, b = separate((regs.cur[0][e], regs.cur[1][e]), pair.partner(e))
                if pair.ra < R:
                    Kf[pair.ra, 0, e, f.t] = pack(*a)
                if pair.rb < R:
                    Kf[pair.rb, 0, e, f.t] = pack(*b)

    return fft_spectra


@compiled(pass_configs=CONFIG)
def spectra_grad_kernel(NF, s, bidirectional, combines):
    """Build the kernel that computes gkf, gkfr and gD from the gradient gK of `spectra_kernel`.

    gK comes from a `backward_kernel` with the same `combines`. gkf and gkfr are the Lk-point
    forward transforms of the gradients of the taps, which `kernels.cauchy` converts to the
    gradients of kf and kfr. Without `bidirectional`, gkfr is unused.
    """
    blocks = Blocks(NF, 1)
    lay = blocks.lay
    Lk = NF // s
    seq_threads, threads, G = lay.seq_threads, lay.threads, lay.G
    half_bins = Lk // 2 + 1
    groups, R, R2, RD = (T.dynamic(n) for n in ("S", "R", "R2", "RD"))

    def taps(buf, at, L, part):
        """Return the function value of `fft.FFT.load` that reads the taps' gradients from buf.

        Only the first L taps are nonzero. Without `bidirectional`, value(n) is ga[n] + i gb[n] for
        the gradients ga and gb of the circular kernels of rows a and b. With `bidirectional`,
        value(n) is g[n] + i g[NF - 1 - n] for the gradient g of row a (`part` 0) or row b
        (`part` 1).
        """

        def value(n):
            a = unpack(buf[at(T.min(n, L - 1))])
            if bidirectional:
                b = unpack(buf[at(NF - 1 - T.min(n, L - 1))])
                a = a[part], b[part]
            live = T.cast(n < L, f32)
            return a[0] * live, a[1] * live

        return value

    def halves(pair, regs, outs, rows):
        """Transform a + i b in regs and store bins s j of the transforms of a and b in outs.

        The bins of a go to outs[0][rows[0]], and the bins of b go to outs[1][rows[1]].
        """
        pair.f.forward(regs)
        pair.scatter_regs(regs)
        for i in range(ceil_div(half_bins, seq_threads)):
            j = i * seq_threads + pair.f.t
            q = s * T.min(j, half_bins - 1)
            a, b = separate(unpack(pair.f.buf[pair.at(q)]), unpack(pair.f.buf[pair.mirror(q)]))
            live = [T.And(j < half_bins, r < R) for r in rows]
            put_bin(outs[0], rows[0], j, *a, live[0])
            put_bin(outs[1], rows[1], j, *b, live[1])

    @T.prim_func
    def fft_spectra_grad(
        gK: T.Tensor((groups, R, 1, E, seq_threads), u64),
        tw: T.Tensor((lay.n_twiddles, 2), f32),
        fr: T.Tensor((E, seq_threads), T.int32),
        gkf: T.Tensor((R, half_bins, 2), f32),
        gkfr: T.Tensor((R2, half_bins, 2), f32),
        gD: T.Tensor((RD,), f32),
        L: T.int32,
        has_D: T.int32,
    ):
        with T.Kernel(T.ceildiv(T.ceildiv(R, 2), G), threads=threads) as bx:
            f, _, _, row_block, seq = blocks.setup(tw, None, bx, 1, fr)
            Xr = T.alloc_local((E,), f32)
            Xi = T.alloc_local((E,), f32)
            Yr = T.alloc_local((E,), f32)
            Yi = T.alloc_local((E,), f32)
            Zr = T.alloc_local((E,), f32)
            Zi = T.alloc_local((E,), f32)
            pair = Pair(f, row_block * G + seq, R)
            if combines:
                pair.hermitian(lambda k, row, e: gK[0, row, 0, e, f.t], (Xr, Xi))
            else:
                totals = (Yr, Yi), (Zr, Zi)
                for (Ar, Ai), row in zip(totals, (pair.read_a, pair.read_b), strict=True):
                    group_sums(Ar, Ai, gK, groups, row, 0, f.t)
                pair.hermitian(lambda k, row, e: pack(totals[k][0][e], totals[k][1][e]), (Xr, Xi))
            xr, xi = f.inverse(Regs((Xr, Xi), (Yr, Yi)))
            if T.And(has_D != 0, f.t == 0):
                if pair.ra < R:
                    gD[pair.ra] = xr[0]
                if pair.rb < R:
                    gD[pair.rb] = xi[0]
            for e in T.unroll(E):
                f.buf[pair.at(f.point(e) + f.j * (NF // 2))] = pack(xr[e], xi[e])
            if bidirectional:
                regs_a, regs_b = Regs((Xr, Xi), (Yr, Yi)), Regs((Zr, Zi), (Yr, Yi))
                f.load(regs_a, taps(f.buf, pair.at, L, 0))
                f.load(regs_b, taps(f.buf, pair.at, L, 1))
                halves(pair, regs_a, (gkf, gkfr), (pair.ra, pair.ra))
                halves(pair, regs_b, (gkf, gkfr), (pair.rb, pair.rb))
            else:
                regs = Regs((Xr, Xi), (Yr, Yi))
                f.load(regs, taps(f.buf, pair.at, L, 0))
                halves(pair, regs, (gkf, gkf), (pair.ra, pair.rb))

    return fft_spectra_grad
