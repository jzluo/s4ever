import concurrent.futures
import copy
import importlib
import math
import pickle
import subprocess
import sys
import threading
import warnings

import pytest
import torch
from helpers import ACCELERATOR, DEVICES, assert_close, tf32

from s4ever import S4, S4D, S4Block, param_groups
from s4ever._backends import shared
from s4ever._initialization import s4_init, s4d_init
from s4ever.ops import s4, s4d


def make(cls, d_state=16, d_model=8, **kw):
    return cls(d_model, d_state, **{"init": "hippo" if cls is S4 else "diag", "n_ssm": 2, **kw})


@pytest.mark.parametrize("cls,bidirectional", [(S4D, False), (S4, False), (S4D, True)])
def test_modes_agree(cls, bidirectional):
    layer = make(cls, channels=2, bidirectional=bidirectional)
    x = torch.randn(3, 200, 8)
    ys = []
    for mode in ["fft", "chunk", "recurrent"] if cls is S4D else ["fft", "recurrent"]:
        layer.mode = mode
        if mode != "fft" and bidirectional:
            with pytest.raises(ValueError, match="bidirectional"):
                layer(x)
            continue
        with torch.no_grad():
            y, state = layer(x)
        assert y.shape == (3, 200, 16) and state is None
        ys.append(y)
    for y in ys[1:]:
        assert_close(y, ys[0])
    if bidirectional:
        layer.mode = "auto"
        with torch.no_grad():
            assert layer(x[:, :1])[0].shape == (3, 1, 16)  # 'auto' picks fft mode, not recurrent


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "cls,mode,stateful,L,kw",
    [
        (S4D, "fft", False, 100, {}),
        (S4D, "auto", True, 100, {}),
        (S4D, "chunk", False, 100, {}),
        (S4, "fft", False, 100, {}),
        (S4, "fft", True, 100, {}),
        (S4, "fft", False, 100, {"l_max": None}),
        (S4, "recurrent", True, 20, {}),
        # On MPS with the kernels off, `ops._s4d.s4d_steps` computes the output of each compiled
        # step with an einsum. Without the einsum, Inductor would fuse 16 or more of the 20 steps
        # into one kernel. That kernel would need more than the 31 buffers that Metal allows.
        (S4D, "recurrent", True, 20, {}),
        # At 64 modes, PyTorch computes S4's state powers, because S4's state kernels support at
        # most 32 modes.
        (S4, "fft", True, 100, {"d_state": 128}),
    ],
)
@pytest.mark.usefixtures("fresh_dynamo")
def test_compiled(device, cls, mode, stateful, L, kw):
    # S4 uses legs, because at the nearly undamped modes of fout, the compiled and eager gradients
    # of A_imag and dt differ by up to 2e-3.
    kw = {"l_max": 128, "init": "legs", **kw} if cls is S4 else kw
    layer = make(cls, mode=mode, **kw).to(device)
    ref = copy.deepcopy(layer)
    N = layer.d_state // 2
    x = torch.randn(3, L, 8, device=device, requires_grad=True)
    x0 = torch.randn(3, 8, N, 2, device=device, requires_grad=True) if stateful else None
    grad = mode != "recurrent" and (cls is S4D or not stateful)

    def run(module, fn):
        with torch.set_grad_enabled(grad):
            y, state = fn(x, x0)
        if not grad:
            return y, state
        loss = (y * torch.linspace(-1, 1, y.shape[1], device=device)[:, None]).sum()
        if state is not None:
            loss = loss + state.square().sum()
        inputs = [x, *module.parameters()] + ([x0] if stateful else [])
        return y, *torch.autograd.grad(loss, inputs)

    compiled = torch.compile(layer, fullgraph=True)
    for a, b in zip(run(layer, compiled), run(ref, ref), strict=True):
        assert_close(a, b)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.usefixtures("fresh_dynamo")
def test_compiled_bidirectional_fallback(device):
    """Compare the compiled and eager PyTorch paths of a bidirectional layer at d_model 16.

    At this size on MPS, Metal's compiler drops the stores of Inductor kernels that flip the real
    parts of complex values, which is why `ops._fftconv.unpair` is an op.
    """
    layer = S4D(16, 16, bidirectional=True, fused_kernels=False).to(device)
    ref = copy.deepcopy(layer)
    x = torch.randn(2, 100, 16, device=device, requires_grad=True)

    def run(module, fn):
        y = fn(x)[0]
        loss = (y * torch.linspace(-1, 1, y.shape[1], device=device)[:, None]).sum()
        return y, *torch.autograd.grad(loss, [x, *module.parameters()])

    compiled = torch.compile(layer, fullgraph=True)
    for a, b in zip(run(layer, compiled), run(ref, ref), strict=True):
        assert_close(a, b, tol=1e-3)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("cls,mode", [(S4D, "fft"), (S4D, "chunk"), (S4, "fft")])
@pytest.mark.usefixtures("fresh_dynamo")
def test_compiled_autocast(device, cls, mode):
    """Check the gradients of the compiled PyTorch paths under bf16 autocast.

    AOTAutograd traces the backward under the autocast around the compiled call, and the paths of
    S4D and S4 still compute their gradients in float32.
    """
    kw = {"l_max": 128, "init": "legs"} if cls is S4 else {}
    layer = make(cls, mode=mode, fused_kernels=False, **kw).to(device)
    ref = copy.deepcopy(layer)
    x = torch.randn(3, 100, 8, device=device, requires_grad=True)
    final = mode == "chunk"

    def run(module, fn, autocast):
        with torch.autocast(device, dtype=torch.bfloat16, enabled=autocast):
            y, state = fn(x, output_final_state=final)
        loss = (y * torch.linspace(-1, 1, y.shape[1], device=device)[:, None]).sum()
        if final:
            loss = loss + state.square().sum()
        return torch.autograd.grad(loss, [x, *module.parameters()])

    compiled = torch.compile(layer, fullgraph=True)
    for a, b in zip(run(layer, compiled, True), run(ref, ref, False), strict=True):
        assert_close(a, b)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("cls", [S4D, S4])
@pytest.mark.usefixtures("fresh_dynamo")
def test_compiled_frozen(cls):
    """Check that frozen layers compile to one graph while gradients are enabled."""
    layer = make(cls).cuda().requires_grad_(False)
    x = torch.randn(3, 100, 8, device="cuda")
    y = layer(x)[0]
    assert_close(torch.compile(layer, fullgraph=True)(x)[0], y)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("cls", [S4D, S4])
@pytest.mark.usefixtures("fresh_dynamo", "fresh_launchers")
def test_compiled_cuda_graphs(cls):
    """Check compiled layers in reduce-overhead mode, which replays them as CUDA graphs.

    The launchers make their constant tensors during the warm-up of the first call.
    """
    layer = make(cls).cuda()
    x = torch.randn(3, 100, 8, device="cuda")

    def run(fn):
        y = fn(x)[0]
        return y, *torch.autograd.grad(y.sum(), list(layer.parameters()))

    compiled = torch.compile(layer, mode="reduce-overhead")
    for _ in range(3):  # Inductor warms the graphs up, records them and then replays them
        out = run(compiled)
    for a, b in zip(out, run(layer), strict=True):
        assert_close(a, b)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.usefixtures("fresh_dynamo")
def test_compiled_step(device):
    """Compare the compiled and eager gradients of a single step on the PyTorch path.

    At L = 1, the gradients of the parameters are sums over the batch, and Inductor's MPS kernels
    can add such a sum once per thread of a threadgroup (pytorch/pytorch#201455).
    """
    layer = make(S4D, mode="chunk", fused_kernels=False).to(device)
    compiled = torch.compile(layer, fullgraph=True, dynamic=True)

    def run(fn, x, x0):
        y, state = fn(x, x0)
        loss = y.square().sum() + state.square().sum()
        return torch.autograd.grad(loss, [x, x0, *layer.parameters()])

    for B in (2, 3):
        x = torch.randn(B, 1, 8, device=device, requires_grad=True)
        x0 = torch.randn(B, 8, 8, 2, device=device, requires_grad=True)
        for a, b in zip(run(compiled, x, x0), run(layer, x, x0), strict=True):
            # On MPS, compiled and eager results differ by up to about 1e-5 because they round
            # differently. The bug of pytorch/pytorch#201455 would make the gradients of dt, A, B
            # and C several times too large.
            assert_close(a, b, tol=1e-3)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two CUDA devices")
def test_other_device():
    # Because no other test compiles kernels for 6 modes, the first call compiles them while device
    # 1 is current.
    layer = S4D(8, 12).to("cuda:0")
    x = torch.randn(3, 100, 8, device="cuda:0", requires_grad=True)

    def run():
        y = layer(x)[0]
        return y, *torch.autograd.grad(y.square().sum(), [x, *layer.parameters()])

    with torch.cuda.device(1):
        other = run()
        assert torch.cuda.current_device() == 1
    for a, b in zip(other, run(), strict=True):
        assert_close(a, b)


@pytest.mark.parametrize("device", DEVICES)
def test_dropout(device):
    layer = make(S4D, dropout=0.5).to(device)
    x = torch.randn(3, 100, 8, device=device)
    y = layer(x)[0]
    full = layer.eval()(x)[0]
    kept = y != 0
    assert 0.45 < kept.float().mean() < 0.55
    assert_close(y[kept], 2 * full[kept])


@pytest.mark.parametrize("cls", [S4D, S4])
def test_state_passing(cls):
    layer = make(cls)
    x = torch.randn(2, 96, 8)
    with torch.no_grad():
        y, final = layer(x, output_final_state=True)
        y1, state = layer(x[:, :40], output_final_state=True)
        y2, state = layer(x[:, 40:], state)
    assert_close(torch.cat([y1, y2], 1), y)
    assert_close(state, final)
    with torch.no_grad():
        state = torch.zeros_like(final)
        steps = []
        for t in range(x.shape[1]):
            y_t, state = layer(x[:, t : t + 1], state)
            steps.append(y_t)
    assert_close(torch.cat(steps, 1), y)
    assert_close(state, final)


def test_step_with_gradients():
    layer = make(S4D)
    x, x0 = torch.randn(2, 1, 8, requires_grad=True), torch.randn(2, 8, 8, 2, requires_grad=True)
    y, state = layer(x, x0)
    (y.sum() + state.sum()).backward()
    with torch.no_grad():
        y_ref, state_ref = layer(x, x0)
    assert_close(y, y_ref)
    assert_close(state, state_ref)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("cls", [S4D, S4])
def test_keep_cache(cls, device):
    """Check that stateless calls with `keep_cache` reuse the convolution kernel.

    They reuse it until a version counter of a parameter advances.
    """
    layer = make(cls, keep_cache=True).to(device)
    x = torch.randn(2, 20, 8, device=device)
    with torch.no_grad():
        y = layer(x)[0]
        layer.C.data.mul_(2)  # does not advance the version counter
        assert torch.equal(layer(x)[0], y)
        layer.C.mul_(1)
        assert_close(layer(x)[0], copy.deepcopy(layer)(x)[0])
        assert not torch.equal(layer(x)[0], y)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("L", [20, 1])
@pytest.mark.parametrize("cls", [S4D, S4])
def test_inference_cache(cls, L, device):
    """Check calls without gradients after updates that do not advance the version counters.

    Fused optimizers change the parameters in this way, as do AveragedModel's updates on CUDA.
    Calls without gradients use the current parameters after such updates, except between the
    calls of one sequence of states.
    """
    layer = make(cls).to(device)
    x = torch.randn(2, L, 8, device=device)

    def check(state=None):
        with torch.no_grad():
            y, final = layer(x, state, output_final_state=True)
            y_ref, final_ref = copy.deepcopy(layer)(x, state, output_final_state=True)
        assert_close(y, y_ref)
        assert_close(final, final_ref)
        return final

    def update():  # Writes through .data do not increment the version counters.
        for p in layer.parameters():
            p.data.mul_(0.9)

    check()
    update()
    state = check()
    y, state = layer(x, state if cls is S4D else None, output_final_state=cls is S4D)
    y.sum().backward()
    update()
    state = check(state)
    with torch.no_grad():
        layer.inv_dt.add_(0.5)
    check(state)
    pickle.dumps(layer)


@pytest.mark.parametrize("cls", [S4D, S4])
def test_inference_mode(cls):
    """Check a call after an in-place update of a layer made under torch.inference_mode.

    Parameters made under torch.inference_mode have no version counters.
    """
    x = torch.randn(2, 20, 8)
    with torch.inference_mode():
        layer = make(cls)
        layer(x)
        layer.inv_dt.add_(0.5)
        y = layer(x)[0]
    fresh = make(cls)
    fresh.load_state_dict(layer.state_dict())
    with torch.no_grad():
        assert_close(y, fresh(x)[0])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("cls,mode,L", [(S4D, "fft", 64), (S4, "recurrent", 1)])
def test_streams(cls, mode, L):
    """Check a call on a second CUDA stream that follows a slow call on the first stream.

    The second call does not read the parameters or convolution kernels before the first call has
    computed them, and it does not read S4's C before the first call has set C up for L.
    """
    layer = make(cls, mode=mode).cuda()
    x = torch.randn(2, L, 8, device="cuda")
    with torch.no_grad():
        y_ref = copy.deepcopy(layer)(x)[0]
        first, second = torch.cuda.Stream(), torch.cuda.Stream()
        torch.cuda.synchronize()
        with torch.cuda.stream(first):
            # The next allocations on this stream reuse this freed block of NaN.
            torch.full((2**18,), float("nan"), device="cuda")
            torch.cuda._sleep(10**8)
            layer(x)
        with torch.cuda.stream(second):
            y = layer(x)[0]
    torch.cuda.synchronize()
    assert_close(y, y_ref)
    copy.deepcopy(layer)  # A layer with cached values on CUDA can be copied.


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("cls", [S4D, S4])
def test_graphed_step(cls):
    """Check that the replay of a captured step uses the parameters prepared before the capture.

    A later call that prepares new parameters does not change the replay.
    """
    layer = make(cls).cuda()
    x, state = torch.randn(2, 1, 8, device="cuda"), torch.randn(2, 8, 8, 2, device="cuda")
    graph = torch.cuda.CUDAGraph()
    with torch.no_grad():
        y, final = layer(x, state)
        with torch.cuda.graph(graph):
            y_g, final_g = layer(x, state)
        layer.inv_dt.add_(0.5)
        layer(x)
    graph.replay()
    assert_close(y_g, y)
    assert_close(final_g, final)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_s4_capture_rewrite():
    layer = make(S4, l_max=32).cuda()
    graph = torch.cuda.CUDAGraph()
    x = torch.randn(2, 64, 8, device="cuda")
    with torch.no_grad():
        # A call on a copy compiles the kernels outside the capture. That call does not rewrite the
        # C of `layer`.
        copy.deepcopy(layer)(x)
        with pytest.raises(RuntimeError, match="CUDA graph capture"), torch.cuda.graph(graph):
            layer(x)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.usefixtures("fresh_dynamo")
def test_s4_layer_recurrent_tf32_compiled_cuda():
    """Compare the compiled recurrent mode of an S4 layer under TF32 with float64.

    S4's C-tilde and readout are computed from powers of dA, which are inaccurate in TF32.
    """
    layer = make(S4, d_state=64, mode="recurrent")
    x = torch.randn(2, 256, 8)
    with torch.no_grad():
        exact = copy.deepcopy(layer).double()(x.double())[0]
        with tf32():
            y = torch.compile(layer.cuda(), fullgraph=True)(x.cuda())[0]
    assert_close(y, exact, tol=1e-4)


def test_s4_length_doubling():
    layer = make(S4, l_max=64)
    x = torch.randn(2, 200, 8)
    y64 = layer(x[:, :64])[0]
    y = layer(x)[0]
    assert layer.l_kernel == 256
    assert_close(y[:, :64], y64)
    restored = make(S4)
    restored.load_state_dict(layer.state_dict())
    assert restored.l_kernel == 256
    with torch.no_grad():
        # Recurrent mode computes the untruncated convolution.
        layer.mode = restored.mode = "recurrent"
        assert_close(layer(x)[0], y)
        assert_close(restored(x)[0], y)


def test_s4_concurrent_doubling():
    layer = make(S4, l_max=16)
    x = torch.randn(2, 32, 8)
    with torch.no_grad():
        y = copy.deepcopy(layer)(x)[0]
    barrier = threading.Barrier(2)

    def run():
        barrier.wait()
        with torch.no_grad():
            return layer(x)[0]

    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        ys = [f.result() for f in [pool.submit(run) for _ in range(2)]]
    assert layer.l_kernel == 32
    for y_i in ys:
        assert_close(y_i, y)


def test_s4_empty_first_input():
    """Check that an empty first input raises before the layer sets up C.

    The next input then sets up l_kernel.
    """
    layer = make(S4)
    x = torch.randn(2, 32, 8)
    with torch.no_grad():
        y = copy.deepcopy(layer)(x)[0]
        with pytest.raises(ValueError, match="non-empty"):
            layer(x[:, :0])
        assert_close(layer(x)[0], y)


def test_s4_distributed_growth(monkeypatch):
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: 2)
    x = torch.randn(2, 32, 8)
    with pytest.warns(UserWarning, match="distributed training"):
        make(S4)(x)
    with warnings.catch_warnings():
        warnings.filterwarnings("error", "S4")
        make(S4, l_max=32)(x)


def test_import():
    """Check that importing s4ever does not import Dynamo.

    `shared.dynamo_constant` must set the mark that `torch.compiler.assume_constant_result` sets.
    """
    code = "import sys, s4ever; sys.exit('torch._dynamo' in sys.modules)"
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0
    marked = torch.compiler.assume_constant_result(lambda: 0)
    assert vars(marked) == vars(shared.dynamo_constant(lambda: 0))


def overrides(model):
    names = {id(p): n for n, p in model.named_parameters()}
    return {
        names[id(p)]: {k: v for k, v in g.items() if k != "params"}
        for g in param_groups(model)
        for p in g["params"]
    }


def test_optimizer_settings():
    layer = S4D(8, 16, lr={"dt": 0.0, "A": 1e-3, "C": 1e-2}, wd={"A": 0.0})
    assert "inv_dt" in dict(layer.named_buffers())
    A = {"lr": 1e-3, "weight_decay": 0.0}
    assert overrides(layer) == {"A_real": A, "A_imag": A, "B": {}, "C": {"lr": 1e-2}, "D": {}}
    # The parameters of a copy are new tensors with new ids.
    assert overrides(copy.deepcopy(layer)) == overrides(layer)
    ssm = {n: {"lr": 1e-3, "weight_decay": 0.0} for n in ("inv_dt", "A_real", "A_imag", "P", "B")}
    assert overrides(S4(8, 16, lr=1e-3)) == {**ssm, "C": {}, "D": {}}
    linear = torch.nn.Linear(8, 8)
    linear.weight._optim = {"weight_decay": 0.0}  # as in the models of the original S4
    assert overrides(linear) == {"weight": {"weight_decay": 0.0}, "bias": {}}


def test_validation():
    with pytest.raises(ValueError, match="must be a multiple of 2"):
        S4D(8, init="diag")
    with pytest.raises(ValueError, match="unknown S4 init"):
        S4(8, init="diag-lin")
    with pytest.raises(ValueError, match="unknown discretization"):
        S4D(8, disc="ZOH")
    with pytest.raises(ValueError, match="more than 2 channels when bidirectional"):
        S4(8, channels=3, bidirectional=True)
    with pytest.raises(ValueError, match="unknown fused_kernels"):
        S4D(8, fused_kernels="False")
    with pytest.raises(ValueError, match="fft mode does not support states"):
        make(S4D, mode="fft")(torch.randn(2, 20, 8), output_final_state=True)
    for cls in (S4D, S4):
        layer = make(cls)
        with pytest.raises(ValueError, match="input of shape"):
            layer(torch.randn(2, 8, 20))
        with pytest.raises(ValueError, match="state of shape"):
            layer(torch.randn(2, 20, 8), torch.zeros(3, 8, 8, 2))
        with pytest.raises(ValueError, match="float32 or float64"):
            copy.deepcopy(layer).to(torch.bfloat16)(torch.randn(2, 20, 8, dtype=torch.bfloat16))
        with pytest.raises(ValueError, match="unknown mode"):
            cls(8, mode="parallel")


def test_block():
    block = S4Block(
        make(S4D, dropout=0.1),
        activation="glu",
        final_act="gelu",
        dropout=0.1,
        weight_norm=True,
        initializer="normal",
    )
    x = torch.randn(2, 64, 8)
    y, _ = block(x, lengths=torch.tensor([64, 30]))
    assert y.shape == x.shape
    y.sum().backward()
    block.eval()
    x2 = x.clone()
    x2[1, 30:] = torch.randn(34, 8)
    y1, _ = block(x, lengths=torch.tensor([64, 30]))
    y2, _ = block(x2, lengths=torch.tensor([64, 30]))
    assert_close(y2, y1, tol=1e-6)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("cls", [S4D, S4])
def test_rate(cls, device):
    """Compare calls at rate 0.5 with a layer whose dt is halved.

    The last call continues a sequence of states that started at rate 1, so the layer's cache then
    contains values computed at rate 1.
    """
    layer = make(cls, dt_transform="exp", **({"l_max": 64} if cls is S4 else {})).to(device)
    scaled = copy.deepcopy(layer)
    with torch.no_grad():
        scaled.inv_dt += math.log(0.5)
    if cls is S4:
        scaled.l_kernel = 128
    x = torch.randn(3, 50, 8, device=device)
    assert_close(layer(x, rate=0.5)[0], scaled(x)[0])
    with torch.no_grad():
        state = layer(x, output_final_state=True)[1]
        assert_close(layer(x, state, rate=0.5)[0], scaled(x, state)[0])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "cls,mode,kw",
    [
        (S4D, "fft", {}),
        (S4D, "fft", {"bidirectional": True}),
        (S4D, "chunk", {}),
        (S4, "fft", {}),
        (S4, "fft", {"l_max": 128}),  # On CUDA, kernels of 128 taps are used as spectra.
    ],
)
def test_frozen_gradients(cls, mode, kw, device):
    layer = make(cls, mode=mode, **kw).requires_grad_(False)
    x, gy = torch.randn(3, 100, 8), torch.randn(3, 100, 8)

    def run(device, dtype):
        xd = x.to(device, dtype).requires_grad_()
        y = copy.deepcopy(layer).to(device, dtype)(xd)[0]
        return torch.autograd.grad(y, xd, gy.to(device, dtype))[0]

    assert_close(run(device, torch.float32), run("cpu", torch.float64), tol=1e-4)


def test_tied_dropout():
    layer = make(S4D, dropout=0.5, tie_dropout=True)
    block = S4Block(layer, activation=None, final_act=None, dropout=0.5, tie_dropout=True)
    zero = block(torch.randn(4, 50, 8))[0] == 0
    dropped = zero.all(1)
    assert (dropped == zero.any(1)).all()
    assert 0 < dropped.float().mean() < 1


@pytest.mark.skipif(ACCELERATOR is None, reason="needs an accelerator")
@pytest.mark.parametrize(
    "cls,l_max,d_state",
    [
        (S4D, None, 16),
        (S4, None, 16),
        (S4, 256, 16),  # Kernels of 256 taps are used as spectra.
        (S4, 255, 16),  # Kernels of 2^k - 1 taps are not used as spectra.
        # The Cauchy kernels split the 129 frequencies 0 < k < 130 into a full tile of 128 and a
        # second tile with one frequency.
        (S4, 260, 16),
        (S4, None, 2),
        (S4, None, 300),  # The Cauchy kernels load the tables of the 150 modes in two chunks.
    ],
)
@pytest.mark.parametrize("bidirectional", [False, True])
def test_fft_gradients(cls, l_max, d_state, bidirectional):
    """Compare fft mode and its gradients on the accelerator with float64 on the CPU.

    On CUDA, fft mode runs as one `tilelang.layer_conv` call, in which S4's kernels are as long as
    the input or longer.
    """
    # S4 uses legs, because at the nearly undamped modes of fout, the float32 gradients of A_imag
    # and dt have errors of up to 1e-1. S4D uses diag-lin, because the init 'diag' from `make` needs
    # an even n_ssm. No even n_ssm divides d_model=5.
    kw = {"l_max": l_max, "init": "legs"} if cls is S4 else {"init": "diag-lin"}
    # The CUDA spectra kernel transforms the kernels in pairs of rows. With 3 channels of 5
    # features, the last pair has one row. With 2 channels, every pair of rows is full.
    channels = 2 if bidirectional else 3
    kw |= dict(d_model=5, n_ssm=1, mode="fft", channels=channels, bidirectional=bidirectional)
    layer = make(cls, d_state, **kw)
    # Like the gradients of a mean-reduced loss, the output gradients are about 1e-6. An error that
    # does not scale with the output gradients is then large relative to the gradients.
    x, gy = torch.randn(3, 100, 5), 1e-6 * torch.randn(3, 100, 5 * channels)

    def run(device, dtype, x_grad=True):
        m = copy.deepcopy(layer).to(device, dtype, memory_format=torch.channels_last)
        xd = x.to(device, dtype).requires_grad_(x_grad)
        y = m(xd)[0]
        inputs = [*([xd] if x_grad else []), *m.parameters()]
        return [y, *torch.autograd.grad(y, inputs, gy.to(device, dtype))]

    # One wrong tap of a kernel changes y and the input gradient by more than 1e-4.
    exact = run("cpu", torch.float64)
    for i, (ours, e) in enumerate(zip(run(ACCELERATOR.type, torch.float32), exact, strict=True)):
        assert_close(ours, e, tol=5e-5 if i < 2 else 1e-3)
    # When x does not need a gradient, the backward on CUDA runs kernels that compute only the
    # gradients of the parameters.
    for ours, e in zip(run(ACCELERATOR.type, torch.float32, False)[1:], exact[2:], strict=True):
        assert_close(ours, e, tol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("bidirectional", [False, True])
@pytest.mark.usefixtures("fresh_launchers")
def test_fft_gradients_leaves_cuda(monkeypatch, bidirectional):
    monkeypatch.setattr(
        importlib.import_module("s4ever._backends.tilelang.kernels.fftconv"), "COMBINE", False
    )
    test_fft_gradients(S4, 256, 16, bidirectional)


@pytest.mark.skipif(ACCELERATOR is None, reason="needs an accelerator")
@pytest.mark.parametrize("cls", [S4D, S4])
@pytest.mark.parametrize("grad", [True, False])
def test_custom_ssm_params(cls, grad):

    class DoubleB(cls):
        def ssm_params(self, rate=1.0):
            params = list(super().ssm_params(rate))
            i = 2 if cls is S4D else 3
            params[i] = 2 * params[i]
            return tuple(params)

    layer = DoubleB(8, 16)
    x = torch.randn(3, 100, 8)

    def run(device, dtype):
        with torch.set_grad_enabled(grad):
            return copy.deepcopy(layer).to(device, dtype)(x.to(device, dtype))[0]

    assert_close(run(ACCELERATOR.type, torch.float32), run("cpu", torch.float64), tol=1e-4)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("cls,mode", [(S4D, "fft"), (S4D, "chunk"), (S4, "fft")])
def test_no_double_backward(cls, mode, device):
    if mode == "chunk" and device == "cpu":
        pytest.skip("PyTorch's chunk mode is differentiable twice")
    layer = make(cls, mode=mode, fused_kernels=True if mode == "chunk" else None).to(device)
    y = layer(torch.randn(3, 50, 8, device=device))[0]
    with pytest.raises(RuntimeError, match="create_graph"):
        torch.autograd.grad(y.square().sum(), list(layer.parameters()), create_graph=True)


@pytest.mark.parametrize("cls", [S4D, S4])
@pytest.mark.parametrize("kind", ["none", "exp", "relu", "softplus"])
def test_transforms_invert(cls, kind):
    """Check that the transforms return the initial A from the stored parameters.

    At d_state 64, the largest frequencies of the initializations are 1283 for S4D and 1303 for S4.
    """
    layer = make(cls, d_state=64, real_transform=kind, imag_transform=kind)
    A = (s4_init("hippo", 64, 2) if cls is S4 else s4d_init("diag", 64, 2))[0]
    assert_close(layer._A(), torch.view_as_real(A).repeat(4, 1, 1), tol=1e-6)


def test_sigmoid_range():
    with pytest.raises(ValueError, match="sigmoid"):
        make(S4D, imag_transform="sigmoid")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("cls", [S4D, S4])
@pytest.mark.parametrize(
    "transforms,n_ssm",
    # With n_ssm = d_model, each SSM has one feature. The backward kernel then sums each column with
    # one warp.
    [(("softplus", "exp", "none"), 2), (("sigmoid", "relu", "exp"), 8)],
)
def test_kernel_params(cls, transforms, n_ssm):
    dt, real, imag = transforms
    kw = dict(n_ssm=n_ssm, dt_transform=dt, real_transform=real, imag_transform=imag)
    layer = make(cls, fused_kernels=True, **kw).cuda()
    params = [p for p in layer.parameters() if p is not layer.D]
    ours = layer.ssm_params(0.5)
    grads = [torch.randn_like(t) for t in ours]
    ours = [*ours, *torch.autograd.grad(ours, params, grads)]
    layer.fused_kernels = False
    ref = layer.ssm_params(0.5)
    for x, y in zip(ours, [*ref, *torch.autograd.grad(ref, params, grads)], strict=True):
        assert_close(x, y, tol=1e-6)


@pytest.mark.skipif(ACCELERATOR is None, reason="needs CUDA or MPS")
@pytest.mark.parametrize("cls", [S4D, S4])
def test_pytorch_only(monkeypatch, cls):

    def wanted(fused_kernels):
        assert fused_kernels is False
        return False

    monkeypatch.setattr(shared, "wanted", wanted)
    layer = make(cls, fused_kernels=False).to(ACCELERATOR)
    x, x0 = torch.randn(2, 20, 8, device=ACCELERATOR), torch.randn(2, 8, 8, 2, device=ACCELERATOR)
    layer(x)
    with torch.set_grad_enabled(cls is S4D):
        layer(x, x0)
    with torch.no_grad():
        layer(x[:, :1], x0)
        op = s4d if cls is S4D else s4
        cu_seqlens = torch.tensor([0, 5, 20], device=ACCELERATOR)
        op(x[:1], *layer.ssm_params(), layer.D, cu_seqlens=cu_seqlens, fused_kernels=False)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("cls", [S4D, S4])
@pytest.mark.parametrize("stateful", [False, True])
@pytest.mark.parametrize("L", [100, 1])
def test_autocast(device, cls, stateful, L):
    """Compare a block under bf16 autocast with the block in float32.

    S4 sets up C-tilde for the input length inside autocast, and on the CPU PyTorch computes S4's
    state powers.
    """
    block = S4Block(make(cls)).to(device)
    ref = copy.deepcopy(block)
    x = torch.randn(2, L, 8, device=device)
    x0 = torch.randn(2, 8, 8, 2, device=device) if stateful else None
    with torch.set_grad_enabled(not stateful):
        with torch.autocast(device, dtype=torch.bfloat16):
            y, state = block(x.bfloat16(), x0)
        y_ref, state_ref = ref(x, x0)
    assert y.dtype == torch.bfloat16
    assert_close(y, y_ref, tol=3e-2)
    if stateful:
        assert_close(state, state_ref, tol=3e-2)
