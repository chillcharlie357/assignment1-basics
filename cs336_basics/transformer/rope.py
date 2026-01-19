from torch import nn
import torch
from .utils import get_device
from einops import einsum, rearrange, repeat

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None) -> None:
        """
        theta: float Θ value for the RoPE
        d_k: int dimension of query and key vectors
        max_seq_len: int Maximum sequence length that will be inputted
        device: torch.device | None = None Device to store the buffer on
        """
        super().__init__()

        self.device = get_device(device)

        self.theta = 1.0 / (theta ** (torch.arange(0, d_k, 2) / d_k)) # 1 / theta ^ (2k / d), k in {0, ..., d_k / 2 - 1}
        self.theta = self.theta.to(self.device)

        self.d_k = d_k
        self.max_seq_len = max_seq_len
        
        

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        x: (..., seq_len, d_k)
        token_positions: (..., seq_len), specifying the token positions of x along the sequence dimension.
        """
        x = x.to(self.device)
        token_positions = token_positions.to(self.device)
        

        Theta = einsum(
            token_positions.float(), self.theta,
            "seq_len, d_k_half -> seq_len d_k_half"
        )

        all_theta = repeat(
            Theta, 
            "seq_len d_k_half -> seq_len (d_k_half two)", 
            two=2
        )

        sin = torch.sin(all_theta).to(self.device)
        cos = torch.cos(all_theta).to(self.device)

        x_even = x[..., 0::2] # 0, 2, 4, ...
        x_odd = x[..., 1::2] # 1, 3, 5, ...
        half_neg_x = torch.stack((-x_odd, x_even), dim=-1)
        half_neg_x = rearrange(half_neg_x, "... d_k_half two -> ... (d_k_half two)")
        
        return x * cos +  half_neg_x * sin
