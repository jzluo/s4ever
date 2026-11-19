"""Helpers that the kernel definitions share.

The complex helpers take and return complex values as (re, im) pairs of expressions.
"""

import struct

import tilelang.language as T
import torch

f32 = T.float32
DTYPES = {torch.float32: T.float32, torch.bfloat16: T.bfloat16, torch.float16: T.float16}


def exp(x):
    """Return exp(x) for a float32 x.

    TileLang 0.1.15's HIP code generation lowers T.exp, T.sin and T.cos to the approximate __expf,
    __sinf and __cosf (tile-ai/tilelang#3618, fixed after 0.1.15). On AMD, `exp`, `sin` and `cos`
    therefore call expf, sinf and cosf.
    """
    return T.call_extern(f32, "expf", x) if torch.version.hip else T.exp(x)


def sin(x):
    return T.call_extern(f32, "sinf", x) if torch.version.hip else T.sin(x)


def cos(x):
    return T.call_extern(f32, "cosf", x) if torch.version.hip else T.cos(x)


def fma(a, b, c):
    """Return a b + c as one fused multiply-add.

    On AMD this calls __fmaf_rn, because TileLang's HIP code generation has no lowering for T.fma.
    Lowering a kernel that uses T.fma for the target "hip" fails with "Unresolved call".
    """
    return T.call_extern(f32, "__fmaf_rn", a, b, c) if torch.version.hip else T.fma(a, b, c)


def var(x):
    """Return x in a float32 variable, so that x is computed once.

    TileLang 0.1.15 substitutes a float expression that is bound to a Python name into each of its
    uses in later loops, which then evaluate the expression, transcendental calls included, in
    every iteration. TileLang does not substitute a variable from T.alloc_var.
    """
    return T.alloc_var(f32, init=x)[0]


def int_var(x):
    """Return the integer x in an int32 variable, which TileLang's simplifier treats as opaque.

    TileLang substitutes an integer that is bound to a Python name into each of its uses, so a
    division by a runtime size would be repeated at every use. An opaque index also keeps the
    rewrite simplifier from asking Z3, at every comparison such as tx % 8 + 120 < n with a runtime
    n, whether n is a multiple of 8. TileLang 0.1.15 does not cache the answers, so these queries
    can take most of the time of lowering a kernel.
    """
    return T.alloc_var(T.int32, init=x)[0]


def bits(v):
    """Return the bits of the float32 v as a uint32.

    `bits` and `from_bits` call __float_as_uint and __uint_as_float, because TileLang 0.1.15's CUDA
    code generation caches the operand of T.reinterpret by its printed text. A
    `T.reinterpret(x[0], T.uint32)` after a store to x[0] then returns the bits of the value before
    the store.
    """
    return T.call_extern(T.uint32, "__float_as_uint", v)


def from_bits(v):
    return T.call_extern(f32, "__uint_as_float", v)


def const(x: float):
    """Return the float32 constant x, which is passed by its bits on AMD.

    TileLang 0.1.15's HIP code generation prints float32 constants with 7 significant digits, as in
    6.283185e+00f for 2 pi. Float32 needs 9 digits, so some constants round to a different float32.
    """
    if not torch.version.hip:
        return x
    return from_bits(T.uint32(struct.unpack("<I", struct.pack("<f", x))[0]))


def tf32_hi(v):
    """Return v truncated to TF32, so that v - tf32_hi(v) is exact."""
    return from_bits(T.bitwise_and(bits(v), T.uint32(0xFFFFE000)))


def store(buf, v, *idx):
    """Store v at buf[idx].

    Helpers outside the kernels call T.buffer_store, because TileLang does not parse assignments in
    them.
    """
    T.buffer_store(buf, v, list(idx))


def store_tf32(hi, lo, v, split, *idx):
    """Store v at idx in hi, or its TF32 halves in hi and lo when `split` is set.

    The halves are v truncated to TF32 and the exact remainder, so that tensor cores can compute a
    float32 product from products of the halves.
    """
    if split:
        store(hi, tf32_hi(v), *idx)
        store(lo, v - tf32_hi(v), *idx)
    else:
        store(hi, v, *idx)


def cmul(a, b):
    return a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]


def conj(a):
    return a[0], -a[1]


def add(*zs):
    return sum(z[0] for z in zs), sum(z[1] for z in zs)


def scale(k, z):
    return k * z[0], k * z[1]


def keep(z):
    return var(z[0]), var(z[1])
