from torch import nn
import torch
from einops import einsum, rearrange
from .utils import get_device
from .linear import Linear
class SwiGLU(nn.Module):
    def __init__(self, d_model: int, device: torch.device | None = None, dtype: torch.dtype | None = None) -> None:
        """
        SwiGLU(x, w1, w2, w3) = w2(SiLU(w1x) * w3x)
        x (d_model)
        w1, w3 (d_ff, d_model)
        w2 (d_model, d_ff)
        """
        super().__init__()

        self.d_model = d_model
        self.dtype = dtype if dtype else torch.float32
        self.device = get_device(device)

        initial_dff = int((8 * d_model) / 3)
        alignment = 64
        # d_ff 对齐到64的倍数
        aligned_dff = (initial_dff // alignment) * alignment
        self.d_ff = aligned_dff if aligned_dff > 0 else alignment
        # w1, w3 (d_ff, d_model)
        self.w1 = nn.Parameter(torch.ones(self.d_ff, self.d_model, device=self.device, dtype=self.dtype))
        self.w3 = nn.Parameter(torch.ones(self.d_ff, self.d_model, device=self.device, dtype=self.dtype))
        # w2 (d_model, d_ff)
        self.w2 = nn.Parameter(torch.ones(self.d_model, self.d_ff, device=self.device, dtype=self.dtype)) 

    def _silu(self, x: torch.Tensor) -> torch.Tensor:
        """
        silu(x) = x * sigmod(x)
        """
        x.to(self.device)
        return x * torch.sigmoid(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x =  x.to(self.device)

        part1 = einsum(
            x, self.w1,
            " ... d_model, d_ff d_model -> ... d_ff"
        )
        part2 = einsum(
            x, self.w3,
            "... d_model, d_ff d_model -> ... d_ff"
        )
        
        inner = self._silu(part1) * part2
        
        output = einsum(
            inner, self.w2,
            "... d_ff, d_model d_ff -> ... d_model"
        )
        
        return output