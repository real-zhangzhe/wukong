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

                if len(state) == 0:
                    state["step"] = 0
                    state["sum"] = torch.zeros(p.data.size(0), device=p.device)

                state["step"] += 1
                clr = group["lr"]
                eps = group["eps"]

                if grad.is_sparse:
                    grad = grad.coalesce()
                    indices = grad.indices()[0]
                    values = grad.values()
                    grad_sq_row_mean = values.pow(2).mean(dim=1)
                    state["sum"].index_add_(0, indices, grad_sq_row_mean)
                    std = state["sum"][indices].sqrt().add_(eps)
                    p.data.index_add_(0, indices, -clr * values / std.unsqueeze(1))
                else:
                    grad_sq_row_mean = grad.pow(2).mean(dim=1)
                    state["sum"].add_(grad_sq_row_mean)
                    std = state["sum"].sqrt().add_(eps)
                    p.data.addcdiv_(grad, std.unsqueeze(1), value=-clr)

        return loss
