from torch import nn


class MLP(nn.Sequential):
    def __init__(
        self,
        dim_in: int,
        num_hidden: int,
        dim_hidden: int,
        dim_out: int | None = None,
        dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        layers = []
        for _ in range(num_hidden - 1):
            layers.append(nn.Linear(dim_in, dim_hidden, bias=bias))
            layers.append(nn.LayerNorm(dim_hidden))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            dim_in = dim_hidden

        if dim_out:
            layers.append(nn.Linear(dim_in, dim_out, bias=bias))
        else:
            layers.append(nn.Linear(dim_in, dim_hidden, bias=bias))

        super().__init__(*layers)
