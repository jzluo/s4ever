"""Benchmark s4ever against the original S4.

Each call is a forward pass without autograd (`--pass fwd`) or a forward and a backward pass
(`--pass fwdbwd`). With `--what`, a call runs the layer (`layer`), the layer with an initial state
(`state`), the layer on one time step (`step`), the computation of the convolution kernel from the
parameters (`kernel`) or the convolution of an input with that kernel (`conv`). Results are
appended to `--out` as JSON lines, which bench/table.py prints as tables.

Each configuration runs in its own process, so the memory that one configuration leaves reserved
cannot make a later one run out of memory.

The implementations are `fused` (s4ever with its GPU kernels), `pytorch` (s4ever with
fused_kernels=False) and `original`. An implementation with the suffix `-compile` runs under
torch.compile.

The `wall_ms` field of a result is the time per call in the fastest of five runs of back-to-back
calls, so it always includes CPU overhead. The `ms` field is the median time from triton's
do_bench. Because do_bench clears the L2 cache between calls, `ms` includes CPU overhead only when
the host launches a call more slowly than the GPU clears the cache and runs the call. It is NaN on
devices other than CUDA. The `peak_mb` field is the peak memory that a call allocates beyond the
memory allocated before it. A configuration whose calls run out of memory is recorded with NaN
times and an infinite `peak_mb`, and a sweep does not run the same configuration at longer lengths.
Repeated calls can run out of memory where one call fits, because PyTorch's allocator fragments the
memory that it has reserved.

The original needs the baseline environment (bench/baseline/setup.sh):

    baseline/.venv/bin/python bench/bench.py --impl original,pytorch-compile,fused
"""

import argparse
import itertools
import json
import math
import subprocess
import sys
import time
from functools import partial
from pathlib import Path

import torch
from torch import accelerator

from s4ever import S4, S4D
from s4ever.ops import fftconv, s4_conv_kernel, s4d_conv_kernel

ROOT = Path(__file__).resolve().parents[1]
DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}
DEVICE = accelerator.current_accelerator()
OOM_STATUS = 3  # the exit status of a process whose configuration ran out of memory


def s4ever_layer(model, H, N, L, mode, fused_kernels):
    fk = {"fused_kernels": fused_kernels}
    module = S4(H, N, l_max=L, mode=mode, **fk) if model == "s4" else S4D(H, N, mode=mode, **fk)
    module = module.to(DEVICE)

    def kernel():
        if model == "s4":
            return s4_conv_kernel(*module.ssm_params(), module.l_kernel, **fk)[..., :L]
        return s4d_conv_kernel(*module.ssm_params(), L, module.disc, **fk)

    def conv(x, k):
        return fftconv(x, k, module.D, **fk)

    return module, lambda x, state=None: module(x, state)[0], kernel, conv


def original_layer(model, H, N, L, mode):
    sys.path.insert(0, str(ROOT / "bench" / "baseline"))
    from original import original

    orig = original(S4(H, N) if model == "s4" else S4D(H, N), l_max=L if model == "s4" else None)
    orig = orig.to(DEVICE)

    def conv(x, k):  # the convolution in the original's FFTConv.forward
        n = 2 * x.shape[-1]
        y_f = torch.einsum("bhl,chl->bchl", torch.fft.rfft(x, n=n), torch.fft.rfft(k, n=n))
        y = torch.fft.irfft(y_f, n=n)[..., : x.shape[-1]]
        return y + torch.einsum("bhl,ch->bchl", x, orig.D)

    def layer(x, state=None):
        return orig(x, state=None if state is None else torch.view_as_complex(state))[0]

    return orig, layer, lambda: orig.kernel(L=L)[0], conv


LAYERS = {
    "fused": partial(s4ever_layer, fused_kernels=True),
    "pytorch": partial(s4ever_layer, fused_kernels=False),
    "original": original_layer,
}


def wall(fn, params, calls):
    accelerator.synchronize()
    start = time.perf_counter()
    for _ in range(calls):
        for p in params:
            p.grad = None
        fn()
    accelerator.synchronize()
    return (time.perf_counter() - start) / calls * 1e3


def measure(fn, params):
    fn()
    ms = float("nan")
    if DEVICE.type == "cuda":
        from triton.testing import do_bench

        ms = do_bench(fn, warmup=25, rep=100, grad_to_none=params, return_mode="median")
    # The number of calls is set before the timed run, because the host can launch calls faster than
    # the GPU runs them. A loop that checked the elapsed time would fill the launch queue before it
    # stopped.
    calls = max(3, round(100 / wall(fn, params, 3)))
    wall_ms = min(wall(fn, params, calls) for _ in range(5))
    for p in params:
        p.grad = None
    accelerator.reset_peak_memory_stats()
    base = accelerator.memory_allocated()
    fn()
    accelerator.synchronize()
    return ms, wall_ms, (accelerator.max_memory_allocated() - base) / 2**20


def bench(impl, model, what, pass_, B, H, N, L, dtype, mode):
    base = impl.removesuffix("-compile")
    module, layer, kernel, conv = LAYERS[base](model, H, N, L, mode)
    if impl != base:
        layer, kernel, conv = map(torch.compile, (layer, kernel, conv))
    params = list(module.parameters())
    bwd = pass_ == "fwdbwd"
    shape = (B, H, L) if base == "original" else (B, L, H)
    x = torch.randn(shape, device=DEVICE, dtype=DTYPES[dtype], requires_grad=bwd)
    if what == "kernel":
        out = kernel
    elif what == "conv":
        k = kernel().detach().requires_grad_(bwd)
        params = [*params, k]
        out = partial(conv, x, k)
    elif what == "layer":
        out = partial(layer, x)
    elif what == "state":
        state = torch.randn(B, H, N // 2, 2, device=DEVICE)
        out = partial(layer, x, state)
    else:
        raise ValueError(what)
    if not bwd:
        with torch.no_grad():
            return measure(out, params)
    grad = torch.randn_like(out())
    return measure(lambda: out().backward(grad), [*params, x])


def bench_step(impl, model, B, H, N, dtype):
    base = impl.removesuffix("-compile")
    module, _, _, _ = LAYERS[base](model, H, N, 1024, "auto")
    if base == "original":
        module.setup_step()
        state = module.default_state(B)
        x = torch.randn(B, H, device=DEVICE)
        step = module.step if impl == base else torch.compile(module.step)
        fn = partial(step, x, state)
    else:
        fn_layer = module if impl == base else torch.compile(module)
        state = torch.zeros(B, H, N // 2, 2, device=DEVICE)
        x = torch.randn(B, 1, H, device=DEVICE, dtype=DTYPES[dtype])

        # The cache of the layer is reused only when `state` is the final state of the previous
        # call.
        def fn():
            nonlocal state
            state = fn_layer(x, state)[1]

    with torch.no_grad():
        return measure(fn, [])


def strs(s):
    return s.split(",")


def ints(s):
    return [int(v) for v in s.split(",")]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--impl", type=strs, default=["fused"], help="fused,pytorch,original[-compile]")
    p.add_argument("--model", type=strs, default=["s4d", "s4"])
    p.add_argument("--what", type=strs, default=["layer"], help="kernel,conv,layer,state,step")
    p.add_argument("--pass", dest="pass_", type=strs, default=["fwd", "fwdbwd"])
    p.add_argument("--mode", type=strs, default=["auto"], help="auto,fft,chunk,recurrent (s4ever)")
    p.add_argument("--B", type=ints, default=[64])
    p.add_argument("--H", type=ints, default=[512], help="d_model")
    p.add_argument("--N", type=ints, default=[64], help="d_state")
    p.add_argument("--L", type=ints, default=[256, 1024, *(2**k for k in range(12, 17))])
    p.add_argument("--dtype", type=strs, default=["fp32"])
    p.add_argument("--out", type=Path, default=ROOT / "results" / "bench.jsonl")
    args = p.parse_args()
    args.out.parent.mkdir(exist_ok=True)
    keys = ("impl", "model", "what", "pass_", "mode", "B", "H", "N", "L", "dtype")
    configs = []
    for values in itertools.product(*(getattr(args, k) for k in keys)):
        impl, model, what, pass_, mode, B, H, N, L, dtype = values
        if impl.startswith("original") and (mode not in ("auto", "fft") or dtype != "fp32"):
            continue  # the original supports only fft mode and float32 inputs
        if mode == "recurrent" and pass_ == "fwdbwd" or model == "s4" and mode == "chunk":
            continue  # recurrent mode does not support gradients, and S4 has no chunk mode
        # S4 does not support states while gradients are tracked, and S4D's fft mode does not
        # support states.
        if what == "state" and (pass_ == "fwdbwd" if model == "s4" else mode == "fft"):
            continue
        # A step runs in recurrent mode without gradients, whatever L and the mode are, so it is
        # timed once.
        if what == "step" and (pass_ != "fwd" or args.L[0] != L or mode != args.mode[0]):
            continue
        configs.append(dict(zip(keys, values, strict=True)))
    if len(configs) != 1:
        shortest_oom = {}
        for config in configs:
            row = tuple(v for k, v in config.items() if k != "L")
            if config["L"] >= shortest_oom.get(row, math.inf):
                continue
            flags = [f"--{k.rstrip('_')}={v}" for k, v in config.items()]
            run = subprocess.run([sys.executable, __file__, *flags, f"--out={args.out}"])
            if run.returncode == OOM_STATUS:
                shortest_oom[row] = config["L"]
        return
    (config,) = configs
    impl, model, what, pass_, mode, B, H, N, L, dtype = config.values()
    if DEVICE.type == "cuda":
        device = torch.cuda.get_device_name()
    else:
        device = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True)
    meta = {
        "device": device.strip(),
        "torch": torch.__version__,
        "s4ever": subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True
        ).stdout.strip(),
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    torch.manual_seed(0)
    try:
        if what == "step":
            config.update(L=1, mode="recurrent")
            ms, wall_ms, mb = bench_step(impl, model, B, H, N, dtype)
        else:
            ms, wall_ms, mb = bench(impl, model, what, pass_, B, H, N, L, dtype, mode)
    except torch.OutOfMemoryError:
        ms, wall_ms, mb = float("nan"), float("nan"), float("inf")
    times = f"ms={ms:.4g} wall_ms={wall_ms:.4g} MB={mb:.4g}"
    print(" ".join(f"{k}={v}" for k, v in config.items()), times, flush=True)
    with args.out.open("a") as f:
        record = {**config, "ms": ms, "wall_ms": wall_ms, "peak_mb": mb, **meta}
        f.write(json.dumps(record) + "\n")
    if math.isnan(wall_ms):
        sys.exit(OOM_STATUS)


if __name__ == "__main__":
    main()
