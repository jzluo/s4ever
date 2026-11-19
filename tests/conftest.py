import hashlib
import os
from pathlib import Path

import pytest
import torch

import s4ever
from s4ever._backends.tilelang import runtime

# The keys of torch.compile's caches include the traced graph but not the fake implementations of
# s4ever's ops. Without this tag, a test run after an edit of a fake implementation reuses code that
# was compiled with the old one.
torch.compiler.config.cache_key_tag = hashlib.sha256(
    b"".join(p.read_bytes() for p in sorted(Path(s4ever.__file__).parent.rglob("*.py")))
).hexdigest()


def pytest_configure(config):
    if "PYTEST_XDIST_WORKER" in os.environ:
        torch.set_num_threads(1)


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(0)


def clear_launchers():
    for cache in runtime.LAUNCHERS:
        cache.cache_clear()


@pytest.fixture
def fresh_dynamo():
    """Reset Dynamo before a test that compiles a layer.

    Dynamo counts the compilations of a layer's forward across tests, and with fullgraph=True a
    compilation past the recompile limit raises an error.
    """
    torch.compiler.reset()


@pytest.fixture
def fresh_launchers():
    """Clear the cached launchers and constant tensors before and after a test.

    Tests use this fixture when they patch a setting that the launchers read when they are built,
    such as `kernels.fftconv.COMBINE`, or when they need launchers that are built during the test.
    """
    clear_launchers()
    yield
    clear_launchers()
