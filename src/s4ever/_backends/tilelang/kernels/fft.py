"""Float32 FFTs for the convolution kernels in `fftconv`.

NF / E threads transform a sequence of NF points, a power of two, with in-place
decimation-in-frequency stages. The frequencies stay in the order in which the stages produce them,
and TileLang inserts the shared-memory barriers between the stages.

A transform of NT = N1 NF points is split over N1 thread blocks (the four-step FFT). Block k1
computes the frequencies k1 + N1 k2 as the NF-point transform of the sequence x_k1.

    x_k1[m] = sum_n1 x[m + NF n1] w^((m + NF n1) k1)    for m < NF
    w = exp(-2 pi i / NT)

In the inverse transform, each block computes a share of every output, and the outputs are sums
of the blocks' shares. With more than MAX_SPLIT blocks, separate kernels compute the N1-point
transforms of the columns x[m + NF n1] before and after the NF-point transforms, so a transform
takes three passes through memory.
"""

import cmath
import functools
import math

import tilelang.language as T

from ..layout import FFT_POINTS_PER_THREAD as E
from .dsl import bits, cmul, const, f32, from_bits, int_var, store, var

u32, u64 = T.uint32, T.uint64
MAX_SPLIT = 16
# Each entry maps word p of a shared-memory buffer to p + p // d, which leaves a gap after every d
# words. The gaps spread the strided accesses of a warp over the banks.
PADS = [lambda p, d=d: p + p // d for d in (16, 32)]


@functools.cache
def power_rules():
    """Return the rules (a, b, sign) by which w^k = w^a w^(sign b) for 0 < k < E.

    The twiddle table holds w^k for the powers of two k, which do not need a rule. Each rule gives
    w^k the shortest chain of products from these powers.
    """
    depth = {1 << b: 0 for b in range(E.bit_length() - 1)}
    rules = {}
    while len(depth) < E - 1:
        found = {}
        for k in range(3, E):
            for a in depth:
                for b in depth:
                    for sign in (1, -1):
                        d = max(depth[a], depth[b]) + 1
                        if k not in depth and a + sign * b == k and d < found.get(k, (E,))[0]:
                            found[k] = (d, a, b, sign)
        least = min(v[0] for v in found.values())
        for k, (d, a, b, sign) in found.items():
            if d == least:
                depth[k], rules[k] = d, (a, b, sign)
    return rules


def brev(x, r):
    b = r.bit_length() - 1
    return int(format(x, f"0{b}b")[::-1], 2) if b else 0


def wavefronts(words):
    """Return the number of shared-memory wavefronts for a warp's 64-bit accesses to `words`.

    Each half-warp is counted separately. A half-warp takes as many wavefronts as the most distinct
    words that fall in one pair of 4-byte banks, and word w falls in pair w mod 16.
    """
    total = 0
    for half in (words[:16], words[16:]):
        banks = {}
        for w in set(half):
            banks[w % 16] = banks.get(w % 16, 0) + 1
        total += max(banks.values(), default=0)
    return total


class Layout:
    """The stages of an NF-point transform and the threads that run them.

    Stage s splits each sequence of N[s] points into radix[s] sequences of N[s + 1] points. Stage 0
    has radix 2, and the later stages have radix E, except that the last one may have a smaller
    radix. A sequence takes `seq_threads` = NF / E threads, and a thread block of `threads` threads
    transforms G sequences.
    """

    def __init__(self, NF: int):
        b = E.bit_length() - 1
        rest = NF.bit_length() - 2
        self.NF = NF
        self.radix = [2] + [E] * (rest // b) + ([1 << (rest % b)] if rest % b else [])
        self.N = [math.prod(self.radix[s:]) for s in range(len(self.radix) + 1)]
        self.seq_threads = NF // E
        self.threads = max(self.seq_threads, 128)
        self.G = self.threads // self.seq_threads
        # Each half has this many threads, which is also the stride between a thread's points at
        # stage 1.
        self.half_threads = NF // (2 * E)
        # The index in `twiddles` at which stage s's twiddles start, for the stages between the
        # first and the last.
        self.twiddle_offsets, n = {}, self.half_threads
        for s in range(1, len(self.radix)):
            if self.N[s + 1] > 1:
                self.twiddle_offsets[s] = n
                n += b * self.N[s + 1]
        self.n_twiddles = n
        self.buffer_words = PADS[0](NF)  # the words of a sequence's exchange buffer

    def pos(self, s, t, e):
        """Return the position in input order of thread t's point e at stage s."""
        N, r = self.N, self.radix[s]
        group = t + self.seq_threads * (e // r)  # the group of r points that stage s transforms
        return group // N[s + 1] * N[s] + e % r * N[s + 1] + group % N[s + 1]

    def frequencies(self) -> list[list[int]]:
        """Return the frequencies of the threads' points after the forward transform.

        Entry [e][t] is the frequency of thread t's point e.
        """
        radix, stages = self.radix, len(self.radix)
        out = [[0] * self.seq_threads for _ in range(E)]
        for t in range(self.seq_threads):
            for e in range(E):
                p, digits = self.pos(stages - 1, t, e), []
                for s in reversed(range(stages)):
                    digits.append(p % radix[s])
                    p //= radix[s]
                freq, mul = 0, 1
                for s, d in enumerate(reversed(digits)):
                    freq += brev(d, radix[s]) * mul
                    mul *= radix[s]
                out[e][t] = freq
        return out

    def twiddles(self) -> list[complex]:
        """Return the twiddle table that the transforms read.

        The table starts with the bases exp(-2 pi i m / NF) of the radix-2 stage for
        m < NF / (2 E). For each later stage s with twiddles, it continues with the powers w^(2^b),
        for b < log2(E), of that stage's bases w = exp(-2 pi i m / N[s]) for m < N[s + 1].
        """
        tw = [cmath.exp(-2j * math.pi * m / self.NF) for m in range(self.half_threads)]
        for s in self.twiddle_offsets:
            n = self.N[s + 1]
            tw += [
                cmath.exp(-2j * math.pi * (m << b) / self.N[s])
                for b in range(E.bit_length() - 1)
                for m in range(n)
            ]
        return tw


@functools.cache
def layout(NF: int) -> Layout:
    return Layout(NF)


def split_twiddles(NF: int, N1: int) -> list[complex]:
    """Return the table T1 for transforms of NT = N1 NF points that N1 thread blocks share.

    The table holds w^m for m < NT / (2 E), then the (2 E N1)-th roots of unity.
    """
    n, NT = 2 * E * N1, N1 * NF
    return [cmath.exp(-2j * math.pi * m / NT) for m in range(NT // (2 * E))] + [
        cmath.exp(-2j * math.pi * m / n) for m in range(n)
    ]


def pack(re, im):
    return T.bitwise_or(T.shift_left(T.cast(bits(im), u64), T.uint64(32)), T.cast(bits(re), u64))


def unpack(v):
    return from_bits(T.cast(v, u32)), from_bits(T.cast(T.shift_right(v, T.uint64(32)), u32))


def cmul_const(xr, xi, ang):
    """Return x exp(i ang) for a constant angle, without multiplications at multiples of pi / 2."""
    c, s = math.cos(ang), math.sin(ang)
    if abs(s) < 1e-12:
        return (xr, xi) if c > 0 else (-xr, -xi)
    if abs(c) < 1e-12:
        return (-xi, xr) if s > 0 else (xi, -xr)
    c, s = const(c), const(s)
    return xr * c - xi * s, xr * s + xi * c


class Regs:
    """Two (re, im) pairs of register arrays.

    An out-of-place step reads `cur`, writes `other` and then calls `swap`.
    """

    def __init__(self, X, Y):
        self.cur, self.other = X, Y

    def swap(self):
        self.cur, self.other = self.other, self.cur


class FFT:
    """Code generator for the transforms of one sequence by thread t.

    TW holds the table of `Layout.twiddles` in shared memory, and the registers W (2 E,) hold the
    twiddles of the current stage. The threads of the sequence exchange points through the
    shared-memory words of `buf` from word `off` on.
    """

    def __init__(self, lay, t, TW, buf, off, W, fr=None):
        self.lay, self.t, self.TW, self.buf, self.off, self.W = lay, t, TW, buf, off, W
        self.fr = fr  # the table of `Layout.frequencies`
        # The radix-2 stage's output half (0 or 1) in which thread t holds its stage-1 points.
        self.j = t // lay.half_threads

    def point(self, e):
        """Return the index n < NF / 2 of the thread's point e within its half at stage 1."""
        return e * self.lay.half_threads + self.t % self.lay.half_threads

    def _base(self, inverse):
        """Load the radix-2 stage's base twiddle exp(-2 pi i t' / NF), or its conjugate, into W.

        Here t' = t mod (NF / (2 E)) is the thread's index within its half.
        """
        br, bi = unpack(self.TW[self.t % self.lay.half_threads])
        store(self.W, br, 0)
        store(self.W, -bi if inverse else bi, 1)

    def _fold(self, e, inverse, xr, xi):
        """Return x times the radix-2 stage's twiddle exp(-+2 pi i j n / NF) at n = point(e).

        The twiddle is 1 in the first half. In the second half, it is the base times
        exp(-+2 pi i e / (2 E)).
        """
        ang = math.pi * e / E * (1 if inverse else -1)
        wr, wi = cmul_const(self.W[0], self.W[1], ang)
        # The twiddle is 1 + j (wr - 1) + i j wi rather than a select on j, because TileLang 0.1.15
        # prints T.if_then_else as an if-else statement, with the code that computes x in each
        # branch.
        jm = T.cast(self.j, f32)
        return cmul((xr, xi), (1 + jm * (wr - 1), jm * wi))

    def _powers(self, s, inverse):
        """Return a function of k that returns w^k for the thread's base twiddle w at stage s.

        With `inverse`, the function returns conj(w)^k. The tabled powers go from TW into W first,
        and the function forms each other power by the rules of `power_rules` when it is first
        asked for, and keeps it in W.
        """
        W, o, n = self.W, self.lay.twiddle_offsets[s], self.lay.N[s + 1]
        for b in range(E.bit_length() - 1):
            wr, wi = unpack(self.TW[o + b * n + self.t % n])
            store(W, wr, 2 << b)
            store(W, -wi if inverse else wi, (2 << b) + 1)
        have = {1 << b for b in range(E.bit_length() - 1)}
        rules = power_rules()

        def power(k):
            if k not in have:
                a, b, sign = rules[k]
                (ar, ai), (br, bi) = power(a), power(b)
                wr, wi = cmul((ar, ai), (br, bi if sign > 0 else -bi))
                store(W, wr, 2 * k)
                store(W, wi, 2 * k + 1)
                have.add(k)
            return W[2 * k], W[2 * k + 1]

        return power

    def load(self, regs, value, upper=None):
        """Load the radix-2 stage's outputs for the inputs value(n), n < NF / 2, into regs.cur.

        The input at n + NF / 2 is upper(n), or zero without `upper`.
        """
        self._base(False)
        sign = 1 - 2 * T.cast(self.j, f32)
        for e in range(E):
            n = self.point(e)
            vr, vi = value(n)
            if upper is not None:
                ur, ui = upper(n)
                vr, vi = vr + sign * ur, vi + sign * ui
            xr, xi = self._fold(e, False, vr, vi)
            store(regs.cur[0], xr, e)
            store(regs.cur[1], xi, e)

    def gather(self, regs, split, y):
        """Load the radix-2 stage's outputs for block k1 of a split transform into regs.cur.

        The block's input is x_k1 of the module docstring, which this method forms from the
        function y.

            y(m) = sum_r g^r x(m + NF r)
            x_k1[m] = sum_r x(m + NF r) w^((m + NF r) k1) = w^(m k1) y(m)
            x_k1[n] +- x_k1[n + NF / 2] = w^(n k1) (y(n) +- h y(n + NF / 2))
            g = w^(NF k1)
            h = w^(NF k1 / 2)
        """
        NF, A = self.lay.NF, split.A
        for e in range(E):
            vr, vi = y(self.point(e))
            ur, ui = y(self.point(e) + NF // 2)
            tr, ti = vr + (A[4] * ur - A[5] * ui), vi + (A[4] * ui + A[5] * ur)
            # The radix-2 stage's twiddle times w^(n k1) is w^(n (k1 + j N1)), which is
            # w^(t' (k1 + j N1)) times a (2 E N1)-th root of unity.
            wr, wi = cmul((A[2], A[3]), split.gather_root(e, self.j))
            xr, xi = cmul((tr, ti), (wr, wi))
            store(regs.cur[0], xr, e)
            store(regs.cur[1], xi, e)

    def dfts(self, regs, r, inverse):
        """Compute the r-point DFT of each group of r points in regs.

        The forward DFTs are radix-2 decimations in frequency from natural to bit-reversed order,
        and the inverse DFTs are radix-2 decimations in time from bit-reversed to natural order.
        """
        spans = [r >> (i + 1) for i in range(r.bit_length() - 1)]
        for span in reversed(spans) if inverse else spans:
            (sr, si), (dr, di) = regs.cur, regs.other
            for base in range(0, E, r):
                for blk in range(0, r, 2 * span):
                    for i in range(span):
                        a, b = base + blk + i, base + blk + i + span
                        ang = math.pi * i / span * (1 if inverse else -1)
                        ar, ai = sr[a], si[a]
                        if inverse:
                            br, bi = cmul_const(sr[b], si[b], ang)
                            store(dr, ar + br, a)
                            store(di, ai + bi, a)
                            store(dr, ar - br, b)
                            store(di, ai - bi, b)
                        else:
                            br, bi = sr[b], si[b]
                            xr, xi = cmul_const(ar - br, ai - bi, ang)
                            store(dr, ar + br, a)
                            store(di, ai + bi, a)
                            store(dr, xr, b)
                            store(di, xi, b)
            regs.swap()

    def twiddle(self, regs, s, inverse):
        """Multiply the radix-E DFT outputs in regs by the twiddles of stage s.

        Register e holds output k = brev(e, E), whose twiddle is w^k for the thread's base w, or
        conj(w)^k with `inverse`.
        """
        power = self._powers(s, inverse)
        (sr, si), (dr, di) = regs.cur, regs.other
        for e in range(E):
            k = brev(e, E)
            if k == 0:
                store(dr, sr[e], e)
                store(di, si[e], e)
                continue
            xr, xi = cmul((sr[e], si[e]), power(k))
            store(dr, xr, e)
            store(di, xi, e)
        regs.swap()

    def exchange(self, regs, s_from, s_to):
        xr, xi = regs.cur
        pad = PADS[0]
        for e in range(E):
            store(self.buf, pack(xr[e], xi[e]), self.off + pad(self.lay.pos(s_from, self.t, e)))
        for e in range(E):
            vr, vi = unpack(self.buf[self.off + pad(self.lay.pos(s_to, self.t, e))])
            store(xr, vr, e)
            store(xi, vi, e)

    def forward(self, regs):
        """Run the stages after the radix-2 stage on regs.cur.

        The outputs are in the order of `Layout.frequencies`.
        """
        lay = self.lay
        for s in range(1, len(lay.radix)):
            if s > 1:
                self.exchange(regs, s - 1, s)
            self.dfts(regs, lay.radix[s], False)
            if lay.N[s + 1] > 1:
                self.twiddle(regs, s, False)

    def inverse(self, regs):
        """Return NF times the inverse transform of `forward`'s output in regs.cur.

        Entry e of the returned arrays is output point(e) + j NF / 2.
        """
        lay = self.lay
        for s in reversed(range(1, len(lay.radix))):
            if lay.N[s + 1] > 1:
                self.twiddle(regs, s, True)
            self.dfts(regs, lay.radix[s], True)
            if s > 1:
                self.exchange(regs, s, s - 1)
        # From the outputs Y0 and Y1 of the two halves, the radix-2 stage computes
        # x[n] = Y0[n] + v and x[n + NF / 2] = Y0[n] - v, where v = Y1[n] exp(2 pi i n / NF). Each
        # thread writes Y0[n] or v to its partner's half of the buffer and reads the partner's
        # value from its own half.
        (xr, xi), (vr, vi) = regs.cur, regs.other
        pad = PADS[0]
        half_words = pad(lay.NF // 2)
        self._base(True)
        for e in range(E):
            n = self.point(e)
            ar, ai = self._fold(e, True, xr[e], xi[e])
            store(vr, ar, e)
            store(vi, ai, e)
            store(self.buf, pack(vr[e], vi[e]), self.off + (1 - self.j) * half_words + pad(n))
        sign = 1 - 2 * T.cast(self.j, f32)
        for e in range(E):
            n = self.point(e)
            pr, pi = unpack(self.buf[self.off + self.j * half_words + pad(n)])
            store(xr, pr + sign * vr[e], e)
            store(xi, pi + sign * vi[e], e)
        return xr, xi


def copy_tables(lay, N1, threads=None):
    if N1 is None:
        N1 = 1
    threads = threads or lay.threads
    n_twiddles, n_roots, roots_start = lay.n_twiddles, 2 * E * N1, N1 * lay.half_threads

    @T.macro
    def copy(TW, tw, Q, T1, tx):
        for i in T.serial(T.ceildiv(n_twiddles, threads)):
            if i * threads + tx < n_twiddles:
                TW[i * threads + tx] = pack(tw[i * threads + tx, 0], tw[i * threads + tx, 1])
        if N1 > 1:
            for i in T.serial(T.ceildiv(n_roots, threads)):
                if i * threads + tx < n_roots:
                    Q[i * threads + tx] = T1[roots_start + i * threads + tx]

    return copy


@T.macro
def block_roots(Q, T1, roots_start, t, k1, N1):
    """Copy the 4 E roots of unity that block k1 reads from T1 into Q.

    Thread t < 4 E copies one root. For e < E and j = 0, 1, Q then holds these roots.

        Q[e + E j] = root(e (k1 + j N1))
        Q[2 E + e + E j] = root((e + E j) k1)
        root(m) = exp(-2 pi i m / (2 E N1))
    """
    if t < 4 * E:
        e, j = t % E, t // E % 2
        Q[t] = T1[roots_start + T.if_then_else(t < 2 * E, e * (k1 + j * N1), (e + E * j) * k1)]


class Split:
    """The twiddles that thread t of `FFT` f reads in block k1 of a split transform.

    The registers A (6,) hold w^(t' k1), w^(t' (k1 + j N1)) and +-w^(NF k1 / 2), where t' is the
    thread's index within its half and the sign is - in the second half. Q holds the roots of
    unity that `root` reads, or with three passes the roots of `block_roots`.
    """

    def __init__(self, f, N1, k1, T1, Q, A, passes):
        self.N1, self.k1, self.Q, self.A, self.passes = N1, k1, Q, A, passes
        if passes:
            block_roots(Q, T1, N1 * f.lay.half_threads, f.t, k1, N1)
        ar, ai = unpack(T1[f.t % f.lay.half_threads * k1])
        br, bi = unpack(f.TW[f.t % f.lay.half_threads])  # w^(t' N1)
        j = T.cast(f.j, f32)
        hr, hi = self.share_root(0, 1)
        for i, v in enumerate((ar, ai, *cmul((ar, ai), (1 + j * (br - 1), j * bi)))):
            store(A, v, i)
        store(A, (1 - 2 * j) * hr, 4)
        store(A, (1 - 2 * j) * hi, 5)

    def root(self, m):
        """Return the root of unity exp(-2 pi i m / (2 E N1)) from Q."""
        return unpack(self.Q[m])

    def gather_root(self, e, j):
        k1, N1 = self.k1, self.N1
        return unpack(self.Q[e + E * j]) if self.passes else self.root(e * (k1 + j * N1))

    def share_root(self, e, j):
        k1 = self.k1
        return unpack(self.Q[2 * E + e + E * j]) if self.passes else self.root((e + E * j) * k1)


def block(bx, N1, row_blocks, passes):
    """Return the index k1 of block bx and the index of its block of rows among `row_blocks`.

    The N1 blocks of one block of rows have adjacent indices, so that blocks that combine their
    shares run at the same time. With three passes, the blocks of one k1 have adjacent indices
    instead, because they read the same spectra.
    """
    return (int_var(bx // row_blocks), int_var(bx % row_blocks)) if passes else (bx % N1, bx // N1)


class Blocks:
    """Thread blocks that transform N1 NF points, where N1 is None for three passes.

    T1 (n_split_twiddles,) holds `split_twiddles`, and it is a stand-in when N1 = 1. Q (n_roots,)
    in shared memory holds the (2 E N1)-th roots of unity from T1, or with three passes only the
    4 E roots that the block reads.
    """

    def __init__(self, NF, N1):
        self.lay = lay = layout(NF)
        self.copy = copy_tables(lay, N1)
        self.passes, self.splits = N1 is None, N1 is None or N1 > 1
        if self.passes:
            self.n_split_twiddles, self.N1, self.n_roots = T.dynamic("NT1"), T.dynamic("N1"), 4 * E
        else:
            self.N1, self.n_roots = N1, 2 * E * N1
            self.n_split_twiddles = N1 * lay.half_threads + 2 * E * N1 if N1 > 1 else 1

    def setup(self, tw, T1, bx, row_blocks, fr=None):
        """Return the thread's `FFT`, the block's `Split`, k1, block of rows and sequence index.

        k1 and the block of rows come from `block`, and the sequence index is the index of the
        thread's sequence within the block. The `Split` is None for N1 = 1.
        """
        lay = self.lay
        S = T.alloc_shared((lay.G * lay.buffer_words,), u64)
        TW = T.alloc_shared((lay.n_twiddles,), u64)
        W = T.alloc_local((2 * E,), f32)
        Q = T.alloc_shared((self.n_roots,), u64)
        A = T.alloc_local((6,), f32)
        tx = T.get_thread_binding()
        seq = tx // lay.seq_threads
        k1, row_block = block(bx, self.N1, row_blocks, self.passes)
        f = FFT(lay, tx % lay.seq_threads, TW, S, seq * lay.buffer_words, W, fr)
        self.copy(TW, tw, Q, T1, tx)
        split = Split(f, self.N1, k1, T1, Q, A, self.passes) if self.splits else None
        return f, split, k1, row_block, seq


def shared_bytes(NF: int) -> int:
    """Return the shared memory in bytes that a kernel using `Blocks(NF, N1)` needs at most.

    The bound covers every N1 up to MAX_SPLIT.
    """
    lay = layout(NF)
    # TileLang pads the split kernels' 4-byte count of arrived blocks to 16 bytes.
    return 8 * (lay.G * lay.buffer_words + lay.n_twiddles + 2 * E * MAX_SPLIT) + 16


class Columns:
    """Thread blocks of the column passes, which transform G = NF / N1 columns of N1 points each.

    Thread tx is thread tx // G of column tx mod G, so that consecutive threads load adjacent
    columns, which are adjacent words in memory. The exchange buffers of the columns are an odd
    number of words apart, which spreads the accesses of a warp over the banks.
    """

    def __init__(self, N1, NF):
        self.lay = layout(N1)
        self.G = NF // N1
        self.threads = self.G * self.lay.seq_threads
        self.stride = self.lay.buffer_words | 1
        self.copy = copy_tables(self.lay, 1, self.threads)

    def setup(self, tw, fr, bx, tx):
        lay, G = self.lay, self.G
        S = T.alloc_shared((G * self.stride,), u64)
        TW = T.alloc_shared((lay.n_twiddles,), u64)
        X = T.alloc_local((E,), f32), T.alloc_local((E,), f32)
        Y = T.alloc_local((E,), f32), T.alloc_local((E,), f32)
        W = T.alloc_local((2 * E,), f32)
        self.copy(TW, tw, None, None, tx)
        f = FFT(lay, tx // G, TW, S, tx % G * self.stride, W, fr)
        return f, Regs(X, Y), bx * G + tx % G


@functools.cache
def order_pad(lay):
    """Return the entry of PADS with the fewest bank conflicts for points in frequency order.

    The count covers the scatter of `FFT.forward`'s outputs and the loads of their partners at -q.
    """
    fr, NF = lay.frequencies(), lay.NF

    def cost(pad):
        total = 0
        for w in range(0, lay.seq_threads, 32):
            for e in range(E):
                qs = [fr[e][t] for t in range(w, min(w + 32, lay.seq_threads))]
                total += wavefronts([pad(q) for q in qs]) + wavefronts([pad(-q % NF) for q in qs])
        return total

    return min(PADS, key=cost)


def idft(V, N1, outputs):
    """Replace the N1 points x in V with their inverse DFT sum_k w_N1^(-n k) x[k] for n < outputs.

    V holds x[k] at V[2 brev(k)] and V[2 brev(k) + 1], and output n goes to V[2 n] and
    V[2 n + 1]. The transform is a radix-2 decimation in time with w_N1 = exp(-2 pi i / N1), and
    `outputs` is N1 or N1 / 2.
    """
    span = 1
    while span < N1:
        for blk in range(0, N1, 2 * span):
            for i in range(span):
                a, b = blk + i, blk + i + span
                ar, ai = var(V[2 * a]), var(V[2 * a + 1])
                br, bi = (var(v) for v in cmul_const(V[2 * b], V[2 * b + 1], math.pi * i / span))
                store(V, ar + br, 2 * a)
                store(V, ai + bi, 2 * a + 1)
                if 2 * span < N1 or b < outputs:
                    store(V, ar - br, 2 * b)
                    store(V, ai - bi, 2 * b + 1)
        span *= 2
