from torch import Tensor
from jaxtyping import Float, Int
import torch
from cs336_basics.transformer.softmax import softmax

def cross_entropy(logits: Float[Tensor, "batch_size vocab_size"], targets: Int[Tensor, "batch_size"] ) -> Float[Tensor, ""]:  # noqa: F821

    logits_max,_ = torch.max(logits, dim=-1, keepdim=True)

    logits = logits - logits_max

    targets_exp = torch.exp(logits[targets])

    sum = torch.sum(torch.exp(logits), dim=-1, keepdim=True)    

    batch_p = -torch.log(targets_exp / sum)

    avg_p = torch.mean(batch_p, dim=-1)
    return avg_p


def perplexity():
    pass