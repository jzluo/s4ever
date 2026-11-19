#!/usr/bin/env bash
# Builds the original S4 (state-spaces/s4) and its CUDA Cauchy and Vandermonde extension in
# ./baseline. The build uses the torch and CUDA libraries that uv.lock pins. The script also
# installs s4ever, so that one process can run both implementations.
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
dir="$root/baseline"
mkdir -p "$dir"
[ -d "$dir/s4" ] || git clone -q https://github.com/state-spaces/s4.git "$dir/s4"
git -C "$dir/s4" checkout -q e757cef
uv venv -q --allow-existing --python "$(cat "$root/.python-version")" "$dir/.venv"
uv export -q --project "$root" --frozen --no-dev --no-emit-project --no-hashes -o "$dir/requirements.txt"
export VIRTUAL_ENV="$dir/.venv"
uv pip install -q -r "$dir/requirements.txt" einops numpy scipy setuptools ninja pytest
uv pip install -q --no-build-isolation "$dir/s4/extensions/kernels"
uv pip install -q --no-deps -e "$root"
