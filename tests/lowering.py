"""Lower the kernels that the tests compile, and compare two lowerings of them.

Lowering translates a kernel's TileLang definition to CUDA source. The script lowers each kernel
for the architectures in `ARCHS`, and nvcc reports the kernel's registers, spill bytes and stack
frame.

Run these steps before widening TileLang's pin in pyproject.toml:

1. On a GPU, record the tests' kernel configurations in the directory `configs` with the pinned
   TileLang:

       S4EVER_RECORD=configs python -m pytest -p tests.lowering

2. Lower the configurations with the pinned version, and again in an environment with the candidate
   version. Lowering does not need a GPU.

       python tests/lowering.py lower configs pinned
       python tests/lowering.py lower configs candidate

3. Compare the two lowerings. `diff` lists the kernels whose CUDA source changed, with their
   registers, spill bytes, stack frame and shared memory in both versions.

       python tests/lowering.py diff pinned candidate

For each changed kernel, run the tests and compute-sanitizer's racecheck, synccheck, memcheck and
initcheck on the layers and modes that use the kernel. Then time those layers against the pinned
version in alternating runs.

Do not widen the pin if a kernel fails to lower, has a stack frame larger than 1 KB, or needs more
shared memory than `smem_bound` returns for it. A stack frame larger than 1 KB makes CUDA raise the
stack limit for the rest of the process, and every spilling kernel launched after that runs slower.
The launchers choose kernel configurations by the bounds of `smem_bound`, so a kernel that needs
more shared memory can fail to launch.

The steps also check an edit of the kernel definitions: lower the configurations from the source
tree before the edit and from the tree after it, and compare the two lowerings.
"""

import argparse
import concurrent.futures as cf
import hashlib
import importlib
import inspect
import json
import multiprocessing as mp
import os
import pickle
import re
import subprocess
import tempfile
import threading
from pathlib import Path

ARCHS = ["sm_80", "sm_86", "sm_89", "sm_90a", "sm_100a", "sm_120a"]
STACK_LIMIT = 1024  # bytes, CUDA's default
FIELDS = ("regs", "spill", "stack", "smem")
# These fftconv kernels use `fft.Blocks`, and their first argument is NF.
BLOCK_KERNELS = {
    "forward_kernel",
    "backward_kernel",
    "spectrum_kernel",
    "spectrum_grad_kernel",
    "spectra_kernel",
    "spectra_grad_kernel",
}


def pytest_configure(config):
    """Record the configuration of each kernel that the tests compile, if S4EVER_RECORD is set."""
    out = os.environ.get("S4EVER_RECORD")
    if not out:
        return
    from s4ever._backends.tilelang import runtime

    os.makedirs(out, exist_ok=True)
    compile_ = runtime.compile_future
    lock = threading.Lock()

    def recorded(trace, args, pass_configs, device):
        with lock, open(os.path.join(out, f"{os.getpid()}.pkl"), "ab") as f:
            pickle.dump((trace.__module__, trace.__qualname__, args, pass_configs), f)
        return compile_(trace, args, pass_configs, device)

    runtime.compile_future = recorded


def configs(path):
    """Return the distinct kernel configurations recorded in `path`."""
    recs = set()
    for p in Path(path).glob("*.pkl"):
        with open(p, "rb") as f:
            while True:
                try:
                    module, factory, args, cfg = pickle.load(f)
                except EOFError:
                    break
                recs.add((module, factory, args, tuple(sorted((cfg or {}).items()))))
    return sorted(recs, key=repr)


def smem_bound(module, factory, args):
    """Return the bound in bytes that the launchers use for the kernel's shared memory, or None."""
    from s4ever._backends.tilelang.kernels import fft, vandermonde

    if module.endswith(".vandermonde"):
        M, C, _, NA = args
        return vandermonde.smem_bytes(M, C, NA, factory == "backward_kernel")
    if module.endswith(".fftconv") and factory in BLOCK_KERNELS:
        return fft.shared_bytes(args[0])
    return None


def lower_one(rec, arch, out):
    """Lower one kernel for `arch`, compile it with nvcc and return its row of the report."""
    import tilelang
    import tvm
    from tilelang import env
    from tilelang.contrib.nvcc import get_nvcc_compiler
    from tvm.target import Target

    module, factory, args, cfg = rec
    name = f"{module.split('.')[-1]}.{factory}"
    row = {"kernel": name, "args": repr(args), "bound": smem_bound(module, factory, args)}
    trace = inspect.unwrap(getattr(importlib.import_module(module), factory))
    target = Target({"kind": "cuda", "arch": arch})
    try:
        with tvm.transform.PassContext(config=dict(cfg)), target:
            art = tilelang.lower(trace(*args), target=target)
    except Exception as e:
        return row | {"file": None, "error": str(e).strip().splitlines()[-1][:300]}
    file = f"{name}_{hashlib.sha1(repr(args).encode()).hexdigest()[:12]}.cu"
    (out / arch / file).write_text(art.kernel_source)
    fast = ["--use_fast_math"] if dict(cfg).get("tl.enable_fast_math") else []
    with tempfile.TemporaryDirectory() as d:
        log = subprocess.run(
            [
                get_nvcc_compiler(),
                "--cubin",
                "-std=c++20",
                "-lineinfo",
                f"-arch={arch}",
                *fast,
                f"-I{env.TILELANG_TEMPLATE_PATH}",
                f"-I{env.CUTLASS_INCLUDE_DIR}",
                "-Xptxas=-v",
                "-o",
                os.path.join(d, "k.cubin"),
                str(out / arch / file),
            ],
            capture_output=True,
            text=True,
        )
    text = log.stdout + log.stderr
    if log.returncode:
        return row | {"file": file, "error": text.strip().splitlines()[-1][:300]}

    def count(pattern):
        x = re.search(pattern, text)
        return int(x.group(1)) if x else 0

    return row | {
        "file": file,
        "error": None,
        "regs": count(r"Used (\d+) registers"),
        "spill": count(r"(\d+) bytes spill stores"),
        "stack": count(r"(\d+) bytes stack"),
        "smem": sum(
            int(fn.attrs.get("dyn_shared_memory_buf", 0))
            for fn in art.device_mod.functions.values()
        ),
    }


def lower(config_dir, out, archs):
    """Lower the configurations in `config_dir` for each architecture, and write `out`."""
    import tilelang

    out = Path(out)
    recs = configs(config_dir)
    for arch in archs:
        (out / arch).mkdir(parents=True, exist_ok=True)
    jobs = [(rec, arch) for arch in archs for rec in recs]
    with cf.ProcessPoolExecutor(mp_context=mp.get_context("spawn")) as pool:
        rows = list(pool.map(lower_one, *zip(*jobs, strict=True), [out] * len(jobs)))
    for arch in archs:
        with open(out / f"{arch}.jsonl", "w") as f:
            for (_, job_arch), row in zip(jobs, rows, strict=True):
                if job_arch == arch:
                    f.write(json.dumps(row) + "\n")
    (out / "version").write_text(tilelang.__version__)
    report(out, archs)


def read(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def report(out, archs):
    """Print the kernels that fail to build or exceed the stack limit or their shared memory."""
    for arch in archs:
        rows = read(out / f"{arch}.jsonl")
        failed = [r for r in rows if r["error"]]
        large = [r for r in rows if not r["error"] and r["stack"] > STACK_LIMIT]
        print(
            f"{arch}: {len(rows) - len(failed)}/{len(rows)} build, stack max "
            f"{max((r.get('stack', 0) for r in rows), default=0)} B"
        )
        for r in failed:
            print(f"  fails: {r['kernel']}{r['args']}: {r['error']}")
        for r in large:
            print(f"  stack {r['stack']} B: {r['kernel']}{r['args']}")
        for r in rows:
            if not r["error"] and r["bound"] is not None and r["smem"] > r["bound"]:
                print(f"  shared memory {r['smem']} B > {r['bound']} B: {r['kernel']}{r['args']}")


def diff(old, new):
    """Print the kernels whose CUDA source differs between the lowerings `old` and `new`."""
    old, new = Path(old), Path(new)
    print(f"TileLang {(old / 'version').read_text()} -> {(new / 'version').read_text()}")
    for path in sorted(old.glob("*.jsonl")):
        arch = path.stem
        old_rows = {r["kernel"] + r["args"]: r for r in read(path)}
        new_rows = {r["kernel"] + r["args"]: r for r in read(new / path.name)}
        both = old_rows.keys() & new_rows.keys()
        changed = []
        for key in sorted(both):
            ro, rn = old_rows[key], new_rows[key]
            if ro["error"] or rn["error"]:
                if ro["error"] != rn["error"]:
                    changed.append((key, f"error: {ro['error']} -> {rn['error']}"))
                continue
            if (old / arch / ro["file"]).read_text() != (new / arch / rn["file"]).read_text():
                changed.append((key, ", ".join(f"{k} {ro[k]} -> {rn[k]}" for k in FIELDS)))
        print(
            f"{arch}: {len(both) - len(changed)} unchanged, {len(changed)} changed, "
            f"{len(old_rows.keys() - both)} only in {old}, {len(new_rows.keys() - both)} "
            f"only in {new}"
        )
        for key, what in changed:
            print(f"  {key}: {what}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    lower_parser = sub.add_parser("lower")
    lower_parser.add_argument("configs")
    lower_parser.add_argument("out")
    lower_parser.add_argument("--arch", type=lambda s: s.split(","), default=ARCHS)
    diff_parser = sub.add_parser("diff")
    diff_parser.add_argument("old")
    diff_parser.add_argument("new")
    args = parser.parse_args()
    if args.cmd == "lower":
        lower(args.configs, args.out, args.arch)
    else:
        diff(args.old, args.new)
