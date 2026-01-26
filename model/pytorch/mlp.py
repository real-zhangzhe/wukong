import torch
from torch import nn


class GELU(nn.Module):
    """
    GELU激活函数的Module实现

    GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))

    参考: https://arxiv.org/abs/1606.08415
    """

    def __init__(self) -> None:
        super(GELU, self).__init__()

    def forward(self, input):
        """
        前向传播计算

        参数:
            input: 输入张量

        返回:
            经过GELU激活的输出张量
        """
        return 0.5 * input * (1.0 + torch.erf(input / torch.sqrt(torch.tensor(2.0))))


class MLP(nn.Sequential):
    def __init__(
        self,
        dim_in: int,
        num_hidden: int,
        dim_hidden: int,
        dim_out: int,
        dropout: float = 0.0,
        bias: bool = False,
        activation: nn.Module = GELU(),
    ) -> None:
        layers = []
        for _ in range(num_hidden - 1):
            layers.append(nn.Linear(dim_in, dim_hidden, bias=bias))
            layers.append(nn.BatchNorm1d(dim_hidden))
            layers.append(activation)
            layers.append(nn.Dropout(dropout))
            dim_in = dim_hidden

        layers.append(nn.Linear(dim_in, dim_out, bias=bias))

        super().__init__(*layers)
