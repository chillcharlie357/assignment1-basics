from math import sqrt
from typing import Any, cast
from torch import optim, Tensor
import torch
from torch.optim.optimizer import ParamsT


class AdamW(optim.Optimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        betas: tuple[float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ) -> None:
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr: float = group['lr']
            beta1, beta2 = group['betas']
            eps: float = group['eps']
            wd: float = group['weight_decay']

            for p in group['params']:
                p = cast(Tensor,p) # 参数本质上是Tensor

                state = self.state[p]
                # 初始化state
                if len(state) == 0:
                    # 初始化一阶动量和二阶动量，形状和参数p一致
                    state['m'] = torch.zeros_like(p)
                    state['v'] = torch.zeros_like(p)
                    # 优化次数
                    state['t'] = 0
                
                g = p.grad
                m: Tensor = state['m']
                v: Tensor = state['v']
                t: int = state['t']

                # m = beta1 * m + (1 - beta1) * g
                # v = beta2 * v + (1 - beta2) * g.square()
                t += 1
                m.mul_(beta1).add_(g, alpha=1 - beta1)
                v.mul_(beta2).add_(g.square(), alpha=1- beta2)

                lr_t = lr * (sqrt(1 - pow(beta2, t)) / (1 - pow(beta1, t)))

                # update parameters
                denom = v.sqrt().add_(eps)
                # p = p - lr_t * (m / denom)
                p.addcdiv_(m, denom, value=-lr_t)

                # weight decay
                # p = p - lr * wd * p
                p.mul_(1 - lr * wd)

                # update state
                state['t'] = t
                state['m'] = m
                state['v'] = v


