from math import inf
from typing import overload, override, Any
from collections.abc import Mapping
import torch
from jaxtyping import Float, Bool, Int
from einops import einsum, rearrange
from .softmax import softmax
from torch import nn
from .rope import RoPE
from .utils import get_device
from .linear import Linear
from cs336_basics.log import logger


def scaled_dot_product_attention(
    queries: Float[torch.Tensor, "batch_size ... seq_len d_k"],
    keys: Float[torch.Tensor, "batch_size ... seq_len d_k"],
    values: Float[torch.Tensor, "batch_size ... seq_len d_v"],
    mask: Bool[torch.Tensor, "seq_len seq_len"] | None = None,
) -> Float[torch.Tensor, "batch_size ... d_v"]:
    d_k = keys.shape[-1]
    pre_softmax = einsum(queries, keys, "... n d_k, ... m d_k -> ... n m")

    pre_softmax /= torch.sqrt(torch.tensor(d_k, dtype=keys.dtype, device=keys.device))

    if mask is not None:
        pre_softmax = pre_softmax.masked_fill(mask.logical_not(), float("-inf"))

    softmax_value = softmax(pre_softmax, dim=-1)

    return softmax_value @ values


class MultiheadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        # RoPE parameters
        max_seq_len: int | None = None,
        theta: float = 10000.0,
    ):
        """
        d_model: int Dimensionality of the Transformer block inputs.
        num_heads: int Number of heads to use in multi-head self-attention.
        max_seq_len: int Maximum sequence length to pre-cache if using RoPE.
        theta: float RoPE theta parameter.
        """

        super().__init__()

        self.d_model = d_model  # output dim of self-attention
        self.num_heads = num_heads

        # d_k = d_v = d_model // num_heads
        self.head_dim = d_model // num_heads

        logger.debug(f"MHA d_model: {d_model}, num_heads: {num_heads}, head_dim: {self.head_dim}")

        self.device = get_device(device)
        self.dtype = dtype if dtype else torch.float32

        # qkv合成一个矩阵
        # self.w_qkv = nn.Parameter(torch.randn(d_model * 3, d_model, dtype=self.dtype).to(self.device)) #(d_out, d_in)
        # self.w_o = nn.Parameter(torch.randn(d_model, d_model, dtype=self.dtype).to(self.device))
        self.w_qkv = Linear(d_model, d_model * 3, device=self.device, dtype=self.dtype)
        self.w_o = Linear(d_model, d_model, device=self.device, dtype=self.dtype)
        
        # Initialize RoPE if max_seq_len is provided
        self.rope = None
        if max_seq_len is not None:
            logger.debug("Enbale RoPE")
            self.rope = RoPE(theta, self.head_dim, max_seq_len, device)

    def forward(
        self,
        x: Float[torch.Tensor, "... seq_len d_model"],
        mask: Bool[torch.Tensor, "... seq_len seq_len"] | None = None,
        token_positions: Int[torch.Tensor, "... seq_len"] | None = None,
    ) -> Float[torch.Tensor, "... seq_len d_model"]:
        x = x.to(self.device)
        logger.debug(f"MHA input shape: {x.shape}")

        qkv_projection = self.w_qkv.forward(x)
        logger.debug(f"MHA qkv projection shape: {qkv_projection.shape}")

        # 拆开q, k, v得到[... num_heads seq_len d_model]
        q, k, v = qkv_projection.chunk(3, dim=-1)
        logger.debug(f"MHA qkv chunk shape: {q.shape}, {k.shape}, {v.shape}")

        # 拆分head，调整顺序，每个head并行处理
        # head_dim = d_model / num_heads
        q = rearrange(q, "... seq_len (num_heads head_dim) -> ... num_heads seq_len head_dim", num_heads=self.num_heads)
        k = rearrange(k, "... seq_len (num_heads head_dim) -> ... num_heads seq_len head_dim", num_heads=self.num_heads)
        v = rearrange(v, "... seq_len (num_heads head_dim) -> ... num_heads seq_len head_dim", num_heads=self.num_heads)
        logger.debug(f"MHA  q: {q.shape} k: {k.shape} v: {v.shape}")


        if self.rope is not None and token_positions is not None:
             # q, k: (batch, num_heads, seq_len, head_dim)
             # token_positions: (batch, seq_len)
             # Unsqueeze token_positions to (batch, 1, seq_len) to broadcast over num_heads
             q = self.rope.forward(q, token_positions.unsqueeze(1))
             k = self.rope.forward(k, token_positions.unsqueeze(1))

        values_after_attention = scaled_dot_product_attention(q, k, v, mask)
        logger.debug(f"MHA scaled_dot_product_attention shape: {values_after_attention.shape}")

        # 合并head
        values_after_attention = rearrange(
            values_after_attention,
            "... num_heads seq_len head_dim -> ... seq_len (num_heads head_dim)",
        )
        logger.debug(f"MHA values rearrange shape: {values_after_attention.shape}")

        output = self.w_o.forward(values_after_attention)
        logger.debug(f"MHA output shape: {output.shape}")

        return output
