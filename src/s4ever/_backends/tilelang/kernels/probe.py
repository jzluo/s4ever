"""Small kernels that `runtime.try_toolchain` builds and runs to check TileLang's toolchain."""

import tilelang.language as T

from .dsl import f32


def kernel():
    """Build a kernel that fills x with zeros and y with ones.

    The kernel has two parameters, because TileLang 0.1.15's NVRTC launcher cannot launch a kernel
    with one parameter. With execution_backend="nvrtc", that launch raises "Argument 'kernelParams'
    is not a valid type".
    """

    @T.prim_func
    def probe(x: T.Tensor((32,), f32), y: T.Tensor((32,), f32)):
        with T.Kernel(1, threads=32):
            x[T.get_thread_binding()] = 0.0
            y[T.get_thread_binding()] = 1.0

    return probe


def gemm_kernel():
    """Build a kernel that computes c = a b^T in float32 with T.gemm.

    The thread block is one AMD wavefront of 64 threads.
    """

    @T.prim_func
    def probe_gemm(
        a: T.Tensor((16, 16), f32), b: T.Tensor((16, 16), f32), c: T.Tensor((16, 16), f32)
    ):
        with T.Kernel(1, threads=64):
            As = T.alloc_shared((16, 16), f32)
            Bs = T.alloc_shared((16, 16), f32)
            Cf = T.alloc_fragment((16, 16), f32)
            T.copy(a, As)
            T.copy(b, Bs)
            T.gemm(As, Bs, Cf, transpose_B=True, clear_accum=True)
            T.copy(Cf, c)

    return probe_gemm
