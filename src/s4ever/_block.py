from collections.abc import Callable
from typing import Literal, get_args

import torch
from torch import Tensor, nn

from ._layers import S4, S4D, make_dropout

type Activation = Literal[
    "id", "tanh", "relu", "gelu", "elu", "silu", "swish", "sigmoid", "softplus", "glu"
]
type Initializer = Literal["uniform", "normal", "xavier", "zero", "one"]
ACTIVATIONS: dict[Activation, Callable[[], nn.Module]] = {
    "id": nn.Identity,
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "elu": nn.ELU,
    "silu": nn.SiLU,
    "swish": nn.SiLU,
    "sigmoid": nn.Sigmoid,
    "softplus": nn.Softplus,
    "glu": lambda: nn.GLU(dim=-1),
}
INITIALIZERS = get_args(Initializer.__value__)


def make_activation(name: Activation | None) -> nn.Module:
    if name is None:
        return nn.Identity()
    if name not in ACTIVATIONS:
        raise ValueError(f"unknown activation {name!r}; choose from {sorted(ACTIVATIONS)}")
    return ACTIVATIONS[name]()


class S4Block(nn.Module):
    """An S4 or S4D layer followed by an activation, dropout, and a position-wise linear layer.

    Args:
        activation: the activation of the layer's output: 'gelu', 'relu', 'silu' ('swish'),
            'elu', 'tanh', 'sigmoid', 'softplus', 'id', 'glu', or None. The activation 'glu' gates
            one half of the output with the other half.
        final_act: the activation after the linear layer. The choices are the same as for
            `activation`. With None, the block has no linear layer.
        dropout, tie_dropout: dropout after the activation. `tie_dropout` drops entire features.
        initializer: the initialization of the linear layer's weight: 'uniform', 'normal', 'xavier',
            'zero', 'one', or None for PyTorch's default.
        weight_norm: whether the linear layer is weight-normalized.
    """

    def __init__(
        self,
        layer: S4 | S4D,
        *,
        activation: Activation | None = "gelu",
        final_act: Activation | None = "glu",
        dropout: float = 0.0,
        tie_dropout: bool = False,
        initializer: Initializer | None = None,
        weight_norm: bool = False,
    ):
        super().__init__()
        self.layer = layer
        self.activation = make_activation(activation)
        self.dropout = make_dropout(dropout, tie_dropout)
        if final_act is None:
            self.output_linear = nn.Identity()
            return
        d_out = layer.d_model * (2 if final_act == "glu" else 1)
        linear = nn.Linear(layer.d_output // (2 if activation == "glu" else 1), d_out)
        if initializer is not None:
            initialize(linear.weight, initializer, final_act)
        if weight_norm:
            linear = nn.utils.parametrizations.weight_norm(linear)
        self.output_linear = nn.Sequential(linear, make_activation(final_act))

    def forward(
        self,
        x: Tensor,
        state: Tensor | None = None,
        *,
        rate: float = 1.0,
        output_final_state: bool = False,
        lengths: Tensor | int | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """Run the layer's `forward` and then the rest of the block.

        With `lengths` (B,), the inputs past each sequence's length are zeroed before the layer
        runs. The final state is nevertheless the state after all L steps.
        """
        if lengths is not None:
            L = x.shape[1]
            lengths = torch.as_tensor(lengths, device=x.device).reshape(-1, 1, 1)
            x = x * (torch.arange(L, device=x.device)[:, None] < lengths)
        y, state = self.layer(x, state=state, rate=rate, output_final_state=output_final_state)
        y = self.output_linear(self.dropout(self.activation(y)))
        return y, state


# GELU and SiLU use the gain of ReLU, because `nn.init.calculate_gain` does not support them.
GAINS: dict[str, Literal["relu", "tanh", "sigmoid", "linear"]] = {
    "relu": "relu",
    "gelu": "relu",
    "silu": "relu",
    "swish": "relu",
    "tanh": "tanh",
    "sigmoid": "sigmoid",
}


def initialize(weight: Tensor, name: str, activation: str) -> None:
    gain = GAINS.get(activation, "linear")
    if name == "uniform":
        nn.init.kaiming_uniform_(weight, nonlinearity=gain)
    elif name == "normal":
        nn.init.kaiming_normal_(weight, nonlinearity=gain)
    elif name == "xavier":
        nn.init.xavier_normal_(weight)
    elif name in ("zero", "one"):
        nn.init.constant_(weight, float(name == "one"))
    else:
        raise ValueError(f"unknown initializer {name!r}; choose from {INITIALIZERS}")
