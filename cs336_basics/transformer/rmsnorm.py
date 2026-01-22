from torch import nn
import torch
from .utils import get_device
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None) -> None:
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        
        self.dtype = dtype if dtype else torch.float32
        self.device = get_device(device)
        
        self.gain = nn.Parameter(torch.ones(d_model, device=self.device, dtype=self.dtype))

    def _rms(self, x: torch.Tensor) -> torch.Tensor:
        # mean后需要保留x的最后一个dimension
        return torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (batch_size, sequence_length, d_model)
        """
        in_dtype = x.dtype
        x = x.to(torch.float32).to(self.device)

        result = (x  / self._rms(x)) * self.gain

        return result.to(in_dtype)

        

