"""Print the layer results of one bench/bench.py sweep as Markdown tables.

    python bench/table.py results/bench.jsonl

Each layer and pass has one table, where `fwd` is a forward pass without gradients and `fwd+bwd`
is a forward and a backward pass. A table has one row of times per call in milliseconds for each
implementation, with the fastest time of each sequence length in bold. The last two rows are the
speedup and memory savings of s4ever's kernels over the baseline. The baseline is the original S4
when the results have it. Otherwise it is the one other implementation in the results, or the one
that `--baseline` names.

bench.py does not run an implementation at lengths beyond the first at which it ran out of memory,
and the tables show OOM for those lengths too. A table leaves out a length at which no
implementation has a time.
"""

import argparse
import json
import math
from pathlib import Path

NAMES = {
    "original": "state-spaces/s4",
    "pytorch": "s4ever, torch eager",
    "pytorch-compile": "s4ever, torch.compile",
    "fused": "s4ever, fused kernels",
}
PASSES = {"fwd": "fwd", "fwdbwd": "fwd+bwd"}


def name(impl):
    return NAMES.get(impl) or NAMES[impl.removesuffix("-compile")] + ", torch.compile"


def measured(ms):
    return ms is not None and not math.isnan(ms)


def order(impl):
    base = impl.removesuffix("-compile")
    return list(NAMES).index(base), impl != base


def length(L):
    return f"{L // 1024}K" if L >= 1024 else str(L)


def time(ms):
    return f"{ms:,.0f}" if ms >= 100 else f"{ms:#.3g}"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("results", type=Path)
    p.add_argument("--baseline", help="the implementation to compare the kernels with")
    args = p.parse_args()
    latest = {}
    for line in args.results.read_text().splitlines():
        r = json.loads(line)
        if r["what"] == "layer":
            latest[r["model"], r["pass_"], r["impl"], r["L"]] = r
    rows = list(latest.values())
    impls = sorted({r["impl"] for r in rows}, key=order)
    lengths = sorted({r["L"] for r in rows})
    baseline = args.baseline or ("original" if "original" in impls else None)
    if baseline is None:
        others = [i for i in impls if i != "fused"]
        if len(others) != 1:
            p.error(
                f"the results have {len(others)} implementations other than the kernels and no "
                "original S4; use --baseline"
            )
        baseline = others[0]
    if missing := {"fused", baseline} - set(impls):
        p.error(f"the results have no rows for {', '.join(sorted(missing))}")
    r = rows[0]
    config = f"{r['device']}, batch {r['B']}, d_model {r['H']}, d_state {r['N']}, {r['dtype']}"
    print(f"{config}, ms per call of a single layer.\n")
    for model, pass_ in dict.fromkeys((r["model"], r["pass_"]) for r in rows):
        print(f"**{model.upper()}, {PASSES[pass_]}**\n")
        times = {}
        for impl in impls:
            times[impl], oom = [], False
            for L in lengths:
                r = latest.get((model, pass_, impl, L))
                # A time is NaN for a length that ran out of memory, in a result or at a length
                # that bench.py did not run after one, and None for any other missing result.
                if r is not None:
                    oom = math.isnan(r["wall_ms"])
                times[impl].append(math.nan if oom else r["wall_ms"] if r else None)
        shown = [k for k in range(len(lengths)) if any(measured(t[k]) for t in times.values())]
        print("| L | " + " | ".join(length(lengths[k]) for k in shown) + " |")
        print("|---|" + "---:|" * len(shown))
        for impl in impls:
            out = []
            for k in shown:
                ms = times[impl][k]
                if ms is None:
                    out.append("")
                elif math.isnan(ms):
                    out.append("OOM")
                else:
                    fastest = min(t[k] for t in times.values() if measured(t[k]))
                    out.append(f"**{time(ms)}**" if ms == fastest else time(ms))
            print(f"| {name(impl)} | " + " | ".join(out) + " |")
        for label, field in (("Speedup", "wall_ms"), ("Memory savings", "peak_mb")):
            out = []
            for k in shown:
                fused, base = (
                    latest.get((model, pass_, i, lengths[k])) for i in ("fused", baseline)
                )
                ratio = base[field] / fused[field] if fused and base else math.nan
                out.append(f"{ratio:.1f}×" if math.isfinite(ratio) else "")
            print(f"| {label} | " + " | ".join(out) + " |")
        print()


if __name__ == "__main__":
    main()
