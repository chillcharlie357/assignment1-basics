import torch
import torch.nn as nn
import math

class Linear(nn.Module):
    """
    Linear without bias
    """
    def __init__(self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None) -> None:
        super().__init__()

        self.weight = nn.Parameter(torch.empty(out_features, in_features))

        sigma = math.sqrt(2.0 / (in_features + out_features))
        nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=sigma,
            a=-3*sigma,
            b=3*sigma
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.t()