import torch
from torch.optim.optimizer import Optimizer


class RowWiseAdagrad(Optimizer):
    def __init__(self, params, lr=1e-2, eps=1e-10):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        defaults = dict(lr=lr, eps=eps)
        super(RowWiseAdagrad, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # 初始化状态：只为每一行存储一个标量（N,）而不是（N, D）
                if len(state) == 0:
                    state["step"] = 0
                    # 状态大小等于 Embedding 的行数
                    state["sum"] = torch.zeros(p.data.size(0), device=p.device)

                state["step"] += 1
                clr = group["lr"]
                eps = group["eps"]

                if grad.is_sparse:
                    # 处理稀疏梯度（推荐开启 sparse=True）
                    grad = grad.coalesce()
                    indices = grad.indices()[0]
                    values = grad.values()

                    # 计算每行的梯度平方均值 (Row-wise sum of squares / dim)
                    # 也可以用 sum，但 mean 对不同维度更鲁棒
                    grad_sq_row_mean = values.pow(2).mean(dim=1)

                    # 更新对应行的状态
                    state["sum"].index_add_(0, indices, grad_sq_row_mean)

                    # 获取更新后的标准差
                    std = state["sum"][indices].sqrt().add_(eps)

                    # 原地更新参数
                    p.data.index_add_(0, indices, -clr * values / std.unsqueeze(1))

                else:
                    # 处理稠密梯度（如果不开启 sparse=True）
                    grad_sq_row_mean = grad.pow(2).mean(dim=1)
                    state["sum"].add_(grad_sq_row_mean)
                    std = state["sum"].sqrt().add_(eps)
                    p.data.addcdiv_(grad, std.unsqueeze(1), value=-clr)

        return loss
