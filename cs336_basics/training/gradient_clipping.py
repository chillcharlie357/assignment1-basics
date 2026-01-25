from collections.abc import Iterable
import torch
from torch import Tensor

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-6) -> None:
    """
    Clips the gradients of the parameters to a maximum L2 norm.
    The gradients are modified in-place.

    Args:
        parameters (Iterable[torch.nn.Parameter]): An iterable of parameters to clip gradients for.
        max_l2_norm (float): The maximum L2 norm of the gradients.
        eps (float): A small epsilon for numerical stability.
    """
    # Filter parameters that have gradients
    grads: list[Tensor] = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return

    # Compute the total L2 norm of all gradients
    device: torch.device = grads[0].device
    # MPS device may not support torch.linalg.norm with ord=2.0 for all operations
    # Use torch.norm as a fallback or simpler implementation for better compatibility
    total_norm: Tensor = torch.norm(torch.stack([torch.norm(g.detach(), p=2.0).to(device) for g in grads]), p=2.0)
    
    # Clip gradients if the total norm exceeds the maximum allowed norm
    if total_norm > max_l2_norm:
        clip_coef: Tensor = max_l2_norm / (total_norm + eps)
        for g in grads:
            g.detach().mul_(clip_coef)