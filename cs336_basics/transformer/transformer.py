import torch
from torch import Tensor, nn
from jaxtyping import Float
from collections.abc import Mapping
from typing import Any
from cs336_basics.log  import logger


from .rmsnorm import RMSNorm
from .ffn import SwiGLU as FFN
from .attention import MultiheadSelfAttention as MHA
from .utils import get_device

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float = 10000.0) -> None:
        """
        d_model: int Dimensionality of the Transformer block inputs.
        num_heads: int Number of heads to use in multi-head self-attention.
        d_ff: int Dimensionality of the position-wise feed-forward inner layer
        """
        super().__init__()

        assert d_model % num_heads == 0

        device = get_device()
        self.device = device

        self.max_seq_len = max_seq_len

        # first linear norm
        self.ln1 = RMSNorm(d_model, device=device)

        self.attn = MHA(d_model, num_heads, max_seq_len=max_seq_len, theta=theta,device=device)

        #second linear norm
        self.ln2 = RMSNorm(d_model, device=device)

        self.ffn = FFN(d_model, d_ff, device)




    def forward(self, x: Float[Tensor, "batch_size seq_len d_model"]) -> Float[Tensor, "batch_size seq_len d_model"]:
        x = x.to(self.device)

        # layer norm 1
        ln1_res = self.ln1.forward(x)

        # mha with causal mask and rope
        batch_size = x.shape[0]
        seq_len = x.shape[-2]
        mask = torch.tril(torch.ones(seq_len, seq_len, device=self.device)).bool()
        # (batch_size, seq_len)
        token_positions = torch.arange(seq_len, device=self.device).unsqueeze(0).expand(batch_size, -1)
                
        attn_out = self.attn.forward(ln1_res,mask,token_positions)

        residual_1 = x + attn_out
        
        # layer norm 2
        ln2_out = self.ln2.forward(residual_1)

        ffn_out = self.ffn.forward(ln2_out)

        residual_2 = ffn_out + residual_1

        return residual_2