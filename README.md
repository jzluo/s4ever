# s4ever

`s4ever` is an implementation of the S4 and S4D state space models with fused [TileLang](https://github.com/tile-ai/tilelang) kernels for NVIDIA/AMD GPUs and Metal kernels for Apple silicon. The kernels compile on first use, so the package doesn't have a C++ or CUDA extension to build at install time. Without a supported GPU or toolchain (or with `fused_kernels=False`), every layer and op runs on vanilla PyTorch. Outputs, gradients, states, and initializations are verified numerically against the original implementation ([state-spaces/s4](https://github.com/state-spaces/s4)).

On an NVIDIA RTX A6000 (batch 64, d_model 512, d_state 64, fp32), an `S4D` or `S4` layer trains 4–8x faster and uses 2.3–2.8x less peak memory than the original implementation at L=256 to 8K.

**S4D, fwd+bwd** (ms per call of a single layer)

| L | 256 | 1K | 4K | 8K | 16K | 32K |
|---|---:|---:|---:|---:|---:|---:|
| state-spaces/s4 | 4.75 | 18.7 | 73.9 | 151 | OOM | OOM |
| s4ever, torch eager | 4.32 | 15.6 | 60.4 | 125 | 254 | OOM |
| s4ever, torch.compile | 3.66 | 14.4 | 56.4 | 116 | 230 | OOM |
| s4ever, fused kernels | **0.745** | **2.78** | **12.4** | **37.1** | **66.8** | **162** |
| Speedup | 6.4× | 6.7× | 6.0× | 4.1× |  |  |
| Memory savings | 2.8× | 2.8× | 2.8× | 2.3× |  |  |

See [Benchmarks](#benchmarks) for measurements and reproduction commands.

## Installation

```bash
pip install s4ever
```

### Requirements

- **Python**: 3.12 or later. *TileLang 0.1.15 does not work with Python 3.15* [(tilelang#3504)](https://github.com/tile-ai/tilelang/issues/3504), so on Python 3.15 the layers run without the fused kernels on NVIDIA/AMD GPUs.
- **PyTorch**: `>= 2.13.0` on Linux, `>= 2.15.0` on macOS.
- **NVIDIA GPUs (Linux)**: compute capability 8.0 or later (Ampere and newer). TileLang and the CUDA header packages install automatically on Linux; kernels compile with `nvcc` when a CUDA toolkit is present, or with NVRTC otherwise.
- **AMD GPUs (Linux)**: ROCm (`hipcc`) on GPUs with float32 matrix instructions (CDNA). Other AMD GPUs fall back to PyTorch. *TileLang 0.1.15 does not work with PyTorch 2.15 on AMD* [(tilelang#3711)](https://github.com/tile-ai/tilelang/issues/3711)
- **Apple GPUs (macOS)**: Metal kernels compiled on first use through `torch.mps.compile_shader`.
- **CPU and other platforms**: plain PyTorch ops. Windows is untested.

So far, `s4ever` has only been tested on:
- Nvidia RTX A6000, L4, A100, RTX Pro 6000
- AMD MI300X
- Apple M5 Pro

## Quick start

```python
import torch
from s4ever import S4D, S4Block, param_groups

layer = S4D(d_model=128)
y, state = layer(torch.randn(64, 1024, 128))  # (B, L, d_model) in and out

block = S4Block(S4D(128), dropout=0.1)  # layer + GELU + dropout + linear (GLU)
optimizer = torch.optim.AdamW(param_groups(block), lr=0.01, weight_decay=0.01)
```

- **Layout**: `s4ever` uses features-last tensors of shape `(B, L, d_model)`. The original repository uses `(B, d_model, L)`.
- **Return value**: `S4`, `S4D`, and `S4Block` return `(y, state)`. `state` is `None` unless an initial state is passed or `output_final_state=True`.
- **Optimizer groups**: `param_groups(model)` puts the SSM parameters (`dt`, `A`, `B`, and `P` in `S4`) into parameter groups with a learning rate of `0.001` and no weight decay, while leaving all other parameters on the optimizer's defaults.
- **Full example**: [example.py](https://github.com/jzluo/s4ever/blob/main/example.py) trains an `S4D` model on sequential CIFAR-10:

```bash
uv run --with torchvision example.py
```

## Features

### Modes

The `mode` argument of a layer selects how it computes its output.

| Mode | Layers | What it does |
|---|---|---|
| `'auto'` (default) | `S4D`, `S4` | Picks `'recurrent'` for a single step without gradients, `'chunk'` when an `S4D` layer is given or asked for a state, and `'fft'` otherwise. |
| `'fft'` | `S4D`, `S4` | Convolves the input with the SSM's kernel by FFT. `S4D` does not take states in this mode, and `S4` takes them only without gradients. |
| `'chunk'` | `S4D` | Scans the sequence in chunks. It takes and returns states, with or without gradients. |
| `'recurrent'` | `S4D`, `S4` | Steps through the sequence for inference. It does not support gradients. |

Bidirectional layers run only in `'fft'` mode and do not support states.

### State passing and streaming

A layer call returns `(y, state)`. The state is the layer's hidden state after the last time step
of the input, a tensor of shape `(B, d_model, d_state // 2, 2)`. It is `None` unless the call
receives a state or sets `output_final_state=True`.

To process a long sequence in segments, pass to each call the state returned by the previous call.

```python
state = None  # the first call starts from the zero state
with torch.no_grad():
    for segment in stream:  # segment has shape (B, L, d_model)
        y, state = layer(segment, state, output_final_state=True)
```

When the data arrives one time step at a time, use segments of length 1:

```python
y_t, state = layer(x_t[:, None], state, output_final_state=True)  # x_t has shape (B, d_model)
```

One call on a segment of L steps is much faster than a Python loop over L single-step calls.

With the default `mode='auto'`, the layer chooses how to run each call. A single step without
gradients runs in recurrent mode. A longer segment with a state runs in chunk mode in `S4D` and
in fft mode in `S4`.

### dtypes

Inputs can be `float32`, `bfloat16` or `float16`, also under `torch.autocast`, and the output has the dtype of the input. The SSM parameters, the convolution kernel, and the FFTs stay in float32.

### Kernels and PyTorch ops

The `fused_kernels` argument of the layers and of the functions in `s4ever.ops` selects between the GPU kernels and the PyTorch ops.

| Value | What it does |
|---|---|
| `None` (default) | Uses the kernels where they support the device and the input, and the PyTorch ops elsewhere. |
| `True` | Uses the kernels, and raises `ValueError` where they do not support the device or the input. |
| `False` | Uses the PyTorch ops. |

If the kernels cannot be compiled, s4ever warns and falls back to PyTorch.

The layers and ops work under `torch.compile`. Compiling speeds up the PyTorch ops and the model code around the layers, and doesn't change the speed of the kernels.

## Differences from `state-spaces/s4`

- **Tensor layout**: inputs and outputs are `(B, L, d_model)` rather than `(B, d_model, L)`.
- **Optimizer setup**: pass `param_groups(model)` to the optimizer instead of writing custom parameter-grouping code around `_optim` attributes.
- **Streaming interface**: `layer(x, state)` takes a segment of any length, including a single step, and returns the state for the next call. The original repository uses separate `setup_step()`, `default_state()` and `step()` methods for single steps.
- **State format**: the state is a real tensor of shape `(B, d_model, d_state // 2, 2)`. It is the real view (`torch.view_as_real`) of the original repository's complex state.
- **Stateful training**: `S4` does not support state passing while gradients are tracked. Use `torch.no_grad()` for stateful `S4` inference, or `S4D` (`mode='chunk'`) to train through states.

## Benchmarks

The original repository is built with its CUDA extensions. `Speedup` and `Memory savings` compare `s4ever, fused kernels` against `state-spaces/s4`. `OOM` indicates that the configuration ran out of memory with the default PyTorch allocator.

NVIDIA RTX A6000, batch 64, d_model 512, d_state 64, fp32, ms per call of a single layer.

**S4D, fwd**

| L | 256 | 1K | 4K | 8K | 16K | 32K |
|---|---:|---:|---:|---:|---:|---:|
| state-spaces/s4 | 1.53 | 6.20 | 24.6 | 50.8 | 98.2 | OOM |
| s4ever, torch eager | 1.58 | 5.68 | 21.9 | 45.8 | 88.9 | 285 |
| s4ever, torch.compile | 1.31 | 5.15 | 20.1 | 41.7 | 79.9 | 260 |
| s4ever, fused kernels | **0.336** | **1.27** | **5.18** | **16.1** | **26.4** | **59.6** |
| Speedup | 4.6× | 4.9× | 4.7× | 3.2× | 3.7× |  |
| Memory savings | 2.6× | 2.6× | 2.6× | 2.0× | 2.0× |  |

**S4D, fwd+bwd**

| L | 256 | 1K | 4K | 8K | 16K | 32K |
|---|---:|---:|---:|---:|---:|---:|
| state-spaces/s4 | 4.75 | 18.7 | 73.9 | 151 | OOM | OOM |
| s4ever, torch eager | 4.32 | 15.6 | 60.4 | 125 | 254 | OOM |
| s4ever, torch.compile | 3.66 | 14.4 | 56.4 | 116 | 230 | OOM |
| s4ever, fused kernels | **0.745** | **2.78** | **12.4** | **37.1** | **66.8** | **162** |
| Speedup | 6.4× | 6.7× | 6.0× | 4.1× |  |  |
| Memory savings | 2.8× | 2.8× | 2.8× | 2.3× |  |  |

**S4, fwd**

| L | 256 | 1K | 4K | 8K | 16K | 32K |
|---|---:|---:|---:|---:|---:|---:|
| state-spaces/s4 | 2.05 | 6.71 | 25.3 | 51.9 | 100 | OOM |
| s4ever, torch eager | 2.62 | 6.63 | 25.8 | 53.7 | 104 | 317 |
| s4ever, torch.compile | 1.39 | 5.67 | 22.7 | 50.4 | 106 | 327 |
| s4ever, fused kernels | **0.379** | **1.29** | **5.29** | **16.3** | **26.9** | **60.7** |
| Speedup | 5.4× | 5.2× | 4.8× | 3.2× | 3.7× |  |
| Memory savings | 2.6× | 2.6× | 2.6× | 2.0× | 2.0× |  |

**S4, fwd+bwd**

| L | 256 | 1K | 4K | 8K | 16K | 32K |
|---|---:|---:|---:|---:|---:|---:|
| state-spaces/s4 | 5.96 | 20.3 | 77.6 | 158 | OOM | OOM |
| s4ever, torch eager | 6.20 | 19.1 | 74.5 | 153 | 312 | OOM |
| s4ever, torch.compile | 4.15 | 16.5 | 65.7 | 138 | 287 | OOM |
| s4ever, fused kernels | **0.756** | **2.86** | **12.7** | **37.8** | **68.2** | **165** |
| Speedup | 7.9× | 7.1× | 6.1× | 4.2× |  |  |
| Memory savings | 2.8× | 2.8× | 2.8× | 2.3× |  |  |

### Reproducing the benchmarks

```bash
bash bench/baseline/setup.sh
baseline/.venv/bin/python bench/bench.py --impl original,pytorch,pytorch-compile,fused
python bench/table.py results/bench.jsonl
```

## References

- **S4**: Albert Gu, Karan Goel, and Christopher Ré, *Efficiently Modeling Long Sequences with Structured State Spaces* (ICLR 2022), [arXiv:2111.00396](https://arxiv.org/abs/2111.00396).
- **S4D**: Albert Gu, Ankit Gupta, Karan Goel, and Christopher Ré, *On the Parameterization and Initialization of Diagonal State Space Models* (NeurIPS 2022), [arXiv:2206.11893](https://arxiv.org/abs/2206.11893).
- **Original implementation**: [state-spaces/s4](https://github.com/state-spaces/s4)
