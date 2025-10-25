import math

import torch
from jaxtyping import Float, Int
from torch import Tensor
from torch.nn import Parameter


def cross_entropy(
    x: Float[Tensor, " ... seq_len vocab_size"], targets: Int[Tensor, " ... seq_len"], reduction: str = "mean"
):
    x_max: Float[Tensor, " ... seq_len 1"] = torch.max(x, dim=-1, keepdim=True).values

    x = x - x_max
    x_exp: Float[Tensor, " ... seq_len vocab_size"] = torch.exp(x)
    x_sum: Float[Tensor, " ... seq_len"] = torch.sum(x_exp, dim=-1)

    selected_x: Float[Tensor, "... seq_len"] = torch.gather(x, -1, targets.unsqueeze(-1)).squeeze(-1)

    if reduction == "mean":
        loss = torch.sum(torch.log(x_sum) - selected_x) / targets.numel()
    elif reduction == "sum":
        loss = torch.sum(torch.log(x_sum) - selected_x)
    else:
        raise ValueError("reduction must be 'mean', 'sum'")

    return loss


class AdamW(torch.optim.Optimizer):
    def __init__(self, params={}, lr=1e-3, eps=1e-8, weight_decay=0.01, betas=(0.9, 0.999)):
        defaults = {"lr": lr, "eps": eps, "weight_decay": weight_decay, "beta1": betas[0], "beta2": betas[1]}
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get("t", 1)
                grad = p.grad.data

                moment1 = state.get("moment1", torch.zeros_like(p))
                moment2 = state.get("moment2", torch.zeros_like(p))

                moment1 = beta1 * moment1 + (1 - beta1) * grad
                moment2 = beta2 * moment2 + (1 - beta2) * (grad**2)

                lr_t = lr * math.sqrt(1 - beta2**t) / (1 - beta1**t)

                # Update the parameters
                p.data -= lr_t * (moment1 / (torch.sqrt(moment2) + eps))
                # Apply weight decay
                p.data -= lr * weight_decay * p.data

                state["moment1"] = moment1
                state["moment2"] = moment2
                state["t"] = t + 1


def lr_schedule(t: int, lr_max: float, lr_min: float, T_w: int, T_c: int):
    if t < T_w:
        return t / T_w * lr_max
    elif t > T_c:
        return lr_min

    return lr_min + 1 / 2 * (1 + math.cos((t - T_w) / (T_c - T_w) * math.pi)) * (lr_max - lr_min)


def gradient_clipping(params: list[Parameter], max_norm: float):
    grads = [p.grad for p in params if p.grad is not None]
    eps = 1e-6

    total = 0
    for g in grads:
        total += g.norm(2).pow(2)

    norm = total.sqrt()
    if norm >= max_norm:
        for g in grads:
            g *= max_norm / (norm + eps)
