from typing import Tuple
import math

import torch
import torch.optim as optim


class AdamW(optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float=1e-3,
        weight_decay: float=1e-2,
        betas: Tuple[float, float]=(0.9, 0.999),
        eps: float = 1e-8,
    ):
        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "betas": betas,
            "eps": eps,
        }
        super().__init__(params, defaults)
        for group in self.param_groups:
            for param in group["params"]:
                if param.requires_grad:
                    self.state[param]["m"] = torch.zeros_like(param)
                    self.state[param]["v"] = torch.zeros_like(param)
        

    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]

            for param in group["params"]:
                if param.grad is None:
                    continue

                grad = param.grad.data
                t = self.state[param].get("t", 0) + 1
                m = self.state[param]["m"]
                v = self.state[param]["v"]

                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                lr_t = lr * (math.sqrt(1 - beta2 ** t)) / (1- beta1 ** t)

                param.data = param.data - lr_t * torch.div(m, torch.sqrt(v) + eps)
                param.data = param.data - lr * weight_decay * param.data

                self.state[param]["t"] = t
        return loss
