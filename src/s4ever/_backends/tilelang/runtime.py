"""Compilation and caching of the TileLang kernels."""

import concurrent.futures
import functools
import threading
import warnings
from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor

from s4ever._backends import shared
from s4ever._backends.shared import ceil_div


def enabled(x: Tensor, fused_kernels: bool | None) -> bool:
    return (
        x.is_cuda
        and x.dtype in shared.INPUT_DTYPES
        and shared.wanted(fused_kernels)
        and not unavailable(x.device)
    )


# Dynamo cannot trace the probe.
@shared.dynamo_constant
def unavailable(device: torch.device) -> str:
    """Return why no toolchain builds and runs kernels on device, or "" if one does."""
    return probe_toolchains(device)


@functools.cache
def probe_toolchains(device: torch.device) -> str:
    if not torch.version.hip and (cc := torch.cuda.get_device_capability(device)) < (8, 0):
        errors = [f"compute capability {cc[0]}.{cc[1]}, not 8.0 or later"]
    else:
        errors = []
        for toolchain in ["hipcc"] if torch.version.hip else ["nvcc", "NVRTC"]:
            try:
                options = toolchain_options(toolchain)
                try_toolchain(device, options)
            except Exception as e:  # TileLang fails in environment-specific ways
                errors.append(f"{toolchain}: {e}")
            else:
                TOOLCHAINS[device.index] = options
                return ""
    reason = "; ".join(errors)
    warnings.warn(
        "s4ever: the ops use PyTorch unless fused_kernels=True, because no toolchain can build "
        f"and run the TileLang kernels: {reason}",
        stacklevel=3,
    )
    return reason


TOOLCHAINS: dict[int, dict] = {}  # the tilelang.compile options that passed the probe, by device


def toolchain_options(toolchain: str) -> dict:
    """Return the options of tilelang.compile for toolchain, or raise if TileLang cannot use it.

    NVRTC doesn't require nvcc nor the host C++ compiler.
    """
    if toolchain == "NVRTC":
        from cuda import pathfinder

        headers = [pathfinder.find_nvidia_header_directory(h) for h in ("cudart", "cccl")]
        if None in headers:
            raise RuntimeError(
                "cuda.pathfinder did not find the CUDA headers (install cuda-toolkit[cccl])"
            )
        flags = [f"-I{d}" for d in headers]
        # TileLang 0.1.15 detects the CUDA target with `tilelang.contrib.nvcc.find_cuda_path`, which
        # fails without a CUDA toolkit, so the options name the target.
        return {"execution_backend": "nvrtc", "target": "cuda", "compile_flags": flags}
    if toolchain == "nvcc":
        from tilelang import env

        # TileLang finds nvcc through CUDA_HOME, PATH or the nvidia-cuda-nvcc package.
        if not env.CUDA_HOME:
            raise RuntimeError("TileLang did not find a CUDA compiler (set CUDA_HOME)")
    # TileLang's default backend compiles with nvcc on NVIDIA GPUs and with hipcc on AMD GPUs.
    return {}


def device_options(device: int) -> dict:
    unavailable(torch.device("cuda", device))  # the first call probes the toolchains
    return TOOLCHAINS.get(device, {})


def try_toolchain(device: torch.device, options: dict) -> None:
    """Build and run a small kernel, and raise if either step fails.

    A failure here means that every kernel would fail. Such failures come from a compiler that
    rejects TileLang's headers, a driver older than the compiler, or a torch version whose tensors
    TileLang cannot pass to a kernel.
    """
    from .kernels import probe

    with torch.cuda.device(device):
        build_kernel(device, probe.kernel, (), None, options)(*torch.empty(2, 32, device=device))
        # On AMD GPUs without float32 matrix instructions, such as RDNA GPUs, TileLang cannot lower
        # the float32 T.gemm that the chunk and final-state kernels use.
        if torch.version.hip:
            a = torch.arange(256.0, device=device).view(16, 16) % 7 - 3  # products are exact
            c = torch.empty_like(a)
            build_kernel(device, probe.gemm_kernel, (), None, options)(a, a.T.contiguous(), c)
            if not torch.equal(c, a @ a):
                raise RuntimeError("T.gemm's float32 product differs from torch.matmul")


def on_device(fn):
    """Run fn with the device of its first argument as the current device.

    TileLang compiles for the architecture of the current device, and a TileLang launch makes the
    device of its tensors current without restoring the previous device. In TileLang 0.1.15, a
    launch on a tensor on cuda:0 after `torch.cuda.set_device(1)` leaves
    `torch.cuda.current_device()` at 0.
    """

    @functools.wraps(fn)
    def run(x, *args):
        # Inductor sets the device around compiled graphs.
        if torch.compiler.is_compiling() or x.get_device() == torch.cuda.current_device():
            return fn(x, *args)
        with torch.cuda.device_of(x):
            return fn(x, *args)

    return run


POOL = concurrent.futures.ThreadPoolExecutor(thread_name_prefix="s4ever-compile")
TRACING = threading.Lock()
FUTURES: dict = {}
FUTURES_LOCK = threading.Lock()
STARTS: dict = {}


@functools.cache
def architecture(device: int) -> tuple[int, int]:
    return torch.cuda.get_device_capability(device)


def cache_key(trace, args, device: int) -> tuple:
    return trace, args, architecture(device)


def compile_future(trace, args, pass_configs, device: int) -> concurrent.futures.Future:
    """Return the future of the kernel from trace(*args), and submit the compile if it is new."""
    key = cache_key(trace, args, device)
    options = device_options(device)
    with FUTURES_LOCK:
        future = FUTURES.get(key)
        if future is None:
            try:
                future = POOL.submit(build_kernel, device, trace, args, pass_configs, options)
            except RuntimeError:  # at interpreter exit, the pool does not accept more work
                future = concurrent.futures.Future()
                future.set_result(build_kernel(device, trace, args, pass_configs, options))
            FUTURES[key] = future
    return future


def build_kernel(device: torch.device | int, trace, args, pass_configs, options: dict):
    import tilelang

    with TRACING:  # concurrent traces contend for the GIL and run slower than one at a time
        func = trace(*args)
    with torch.cuda.device(device):
        # In a thread without a current context, TileLang 0.1.15's NVRTC loader
        # (`NVRTCLibraryGenerator.load_lib`) calls torch.cuda.synchronize(), which waits for all
        # the work queued on the device.
        context(torch.cuda.current_device())
        # pyrefly: ignore[no-matching-overload]  TileLang imports compile inside an if statement
        kernel = tilelang.compile(func, pass_configs=pass_configs, **options)
    return NVRTCKernel(kernel) if kernel.execution_backend == "nvrtc" else kernel


class NVRTCKernel:
    """A kernel from TileLang's NVRTC backend, launched on torch's raw current stream.

    The raw stream costs less to get than the stream object that TileLang looks up by default.
    """

    def __init__(self, kernel):
        self.kernel = kernel

    def __call__(self, *args):
        device = torch.cuda.current_device()
        context(device)
        return self.kernel(*args, stream=torch._C._cuda_getCurrentRawStream(device))


THREADS = threading.local()


def context(device: int) -> None:
    """Make device's primary CUDA context current in this thread, once per thread and device.

    PyTorch makes the context current only when one of its CUDA calls needs it, so a thread whose
    first CUDA call is a launch through the driver API, such as a backward thread of autograd, has
    no current context. In TileLang 0.1.15, an NVRTC kernel launched in such a thread fails with
    CUDA_ERROR_INVALID_CONTEXT.
    """
    devices = THREADS.__dict__.setdefault("devices", set())
    if device not in devices:
        torch.cuda.set_device(device)
        devices.add(device)


def compiled(trace=None, *, pass_configs=None) -> Callable[..., Any]:
    """Make the factory `trace`, which builds a TileLang function, return the compiled kernel.

    The kernels compile in a thread pool, so that `lazy` can start several compiles at once. A
    failed compile is removed from the cache, so that a later call compiles again.
    """
    if trace is None:
        return functools.partial(compiled, pass_configs=pass_configs)

    def start(args):
        return compile_future(trace, args, pass_configs, torch.cuda.current_device())

    @functools.cache
    def build(device, args):
        future = compile_future(trace, args, pass_configs, device)
        try:
            return future.result()
        except Exception:
            key = cache_key(trace, args, device)
            with FUTURES_LOCK:
                if FUTURES.get(key) is future:
                    del FUTURES[key]
            raise

    @functools.wraps(trace)
    def get(*args):
        return build(torch.cuda.current_device(), args)

    STARTS[get] = start
    return get


def lazy(factory, *args):
    """Start compiling factory(*args), and return a function that launches the kernel.

    The kernel compiles for the device that is current at this call, and the first launch waits for
    the compile.
    """
    device = torch.cuda.current_device()
    STARTS[factory](args)
    kernel = None

    def launch(*inputs):
        nonlocal kernel
        if kernel is None:
            with torch.cuda.device(device):
                kernel = factory(*args)
        return kernel(*inputs)

    return launch


# Caches keyed by input shapes keep the 256 most recently used entries, because shapes can vary
# without bound.
shape_cache = functools.lru_cache(maxsize=256)
LAUNCHERS: list = []


def launcher(fn):
    """Cache fn(device, *args), which returns the kernels of one configuration or their launcher.

    Because fn runs once for each configuration, fn may read only its arguments and the settings
    that stay fixed for the process. Tests that patch such settings, such as
    `kernels.fftconv.COMBINE`, clear these caches.
    """

    @functools.wraps(fn)
    def build(device, *args):
        with torch.cuda.device(device):
            return fn(device, *args)

    cached = shape_cache(build)
    LAUNCHERS.append(cached)
    return cached


MAKING = threading.local()
PERSISTENT = concurrent.futures.ThreadPoolExecutor(
    1, "s4ever-persistent", initializer=lambda: setattr(MAKING, "persistent", True)
)


def persistent(make):
    """Cache make(*args), a device tensor that the launchers keep for the process.

    A separate thread runs make, so that the tensor is not allocated from the memory pool of
    Inductor's CUDA graphs. While Inductor warms up the graphs, it allocates the warming thread's
    tensors from that pool and checks that every live tensor in the pool is a graph output.
    """

    @functools.cache
    @functools.wraps(make)
    def run(*args):
        # A call from the pool's thread runs make directly, because the pool has only that thread.
        # During a CUDA graph capture, make runs in the capturing thread, because the default
        # capture mode of torch.cuda.graph makes a synchronization in another thread an error.
        if getattr(MAKING, "persistent", False) or torch.cuda.is_current_stream_capturing():
            return make(*args)
        try:
            future = PERSISTENT.submit(finished, make, args)
        except RuntimeError:  # at interpreter exit, the pool does not accept more work
            return finished(make, args)
        return future.result()

    LAUNCHERS.append(run)
    return run


def finished(make, args) -> Tensor:
    t = make(*args)
    # Kernels on other streams read the tensor without waiting for this stream.
    torch.cuda.current_stream(t.device).synchronize()
    return t


def or_unused(t: Tensor | None, like: Tensor, *shape: int) -> Tensor:
    return unused(like.device, shape) if t is None else t


@persistent
def unused(device: torch.device, shape: tuple[int, ...], dtype=torch.float32) -> Tensor:
    """Return zeros that stand in for an absent input.

    Some kernels still load the stand-in and multiply it by 0, which would give NaN if the memory
    held NaN.
    """
    return torch.zeros(shape, device=device, dtype=dtype)


def precision() -> int:
    """Return the number of TF32 passes in the tensor-core kernels' float32 products."""
    # AMD's matrix cores compute float32 products at full precision in one pass. Both of PyTorch's
    # precision APIs set fp32_precision, but the older getter torch.get_float32_matmul_precision()
    # raises once the newer API has set the precision.
    return 1 if torch.version.hip or torch.backends.cuda.matmul.fp32_precision == "tf32" else 3


@functools.cache
def sms(device: torch.device) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


def blocks_per_row(H: int, device: torch.device) -> int:
    """Return the thread blocks per row that give H rows at least 8 blocks per SM."""
    return ceil_div(8 * sms(device), H)


@functools.cache
def registers(device: torch.device) -> int:
    """Return the number of 32-bit registers per SM (per CU on AMD)."""
    # pyrefly: ignore[missing-attribute]  torch's stubs omit regs_per_multiprocessor
    return torch.cuda.get_device_properties(device).regs_per_multiprocessor


@functools.cache
def smem(device: torch.device) -> int:
    """Return the bytes of shared memory that one thread block can use.

    ROCm's torch does not report an opt-in limit, so on AMD this is the default limit per block.
    """
    p = torch.cuda.get_device_properties(device)
    # pyrefly: ignore[missing-attribute]  torch's stubs omit shared_memory_per_block_optin
    return p.shared_memory_per_block if torch.version.hip else p.shared_memory_per_block_optin
