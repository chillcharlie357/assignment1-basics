import torch
import torch.nn as nn
import math

class Linear(nn.Module):
    """
    Linear without bias
    """
    def __init__(self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None) -> None:
        super().__init__()

        self.dtype = dtype if dtype else torch.float32
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=self.device, dtype=self.dtype))

        sigma = math.sqrt(2.0 / (in_features + out_features))
        nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=sigma,
            a=-3*sigma,
            b=3*sigma
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (batch_size, sequence_length, d_model)
        """
        x = x.to(self.device)
        return x @ self.weight.t()