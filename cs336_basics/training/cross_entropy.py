from torch import Tensor
from jaxtyping import Float, Int
import torch
from einops import rearrange
from cs336_basics.transformer.softmax import softmax

def cross_entropy(logits: Float[Tensor, "batch_size seq_len vocab_size"], targets: Int[Tensor, "batch_size seq_len"] ) -> Float[Tensor, ""]:  # noqa: F821

    # logits: (batch_size, seq_len, vocab_size)
    # targets: (batch_size, seq_len)
    
    # 展平 batch 和 sequence 维度以便于处理
    # logits -> (batch_size * seq_len, vocab_size)
    # targets -> (batch_size * seq_len)
    if logits.ndim == 3:
        logits = rearrange(logits, 'b s v -> (b s) v')
        targets = rearrange(targets, 'b s -> (b s)')

    logits_max, _ = torch.max(logits, dim=-1, keepdim=True)

    logits = logits - logits_max

    # 高效地获取目标类别对应的 logits
    # logits[range, targets] 选择第 i 行中 target[i] 索引处的 logit 值
    # 确保 targets 被视为索引 (long)，而不是掩码 (bool/uint8)
    target_logits = logits[torch.arange(logits.shape[0], device=logits.device), targets.long()]
    
    # 数值稳定性改进：
    # 代替 -log(exp(x)/sum(exp(x)))，原公式可能导致下溢 exp(x) -> 0 -> log(0) -> inf
    # 使用 log(sum(exp(x))) - x
    sum_exp = torch.sum(torch.exp(logits), dim=-1)
    log_sum_exp = torch.log(sum_exp)
    
    batch_p = log_sum_exp - target_logits

    avg_p = torch.mean(batch_p)
    return avg_p


def perplexity():
    pass