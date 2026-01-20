import torch
from jaxtyping import Float, Bool
from einops import einsum, rearrange
from .softmax import softmax

def scaled_dot_product_attention(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    d_k = keys.shape[-1]
    pre_softmax = einsum(
        queries, keys,
        "... n d_k, ... m d_k -> ... n m"
    )
    
    pre_softmax /= torch.sqrt(torch.tensor(d_k, dtype=keys.dtype, device=keys.device))

    if mask is not None:
        pre_softmax = pre_softmax.masked_fill(~mask, -torch.inf)

    softmax_value = softmax(pre_softmax, dim=-1)


    return softmax_value @ values


