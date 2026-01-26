from jaxtyping import Float
from torch import nn, Tensor

import torch
from typing import Any, override
from collections.abc import Mapping
from .transformer import TransformerBlock
from .embedding import Embedding
from .rmsnorm import RMSNorm
from .linear import Linear
from .softmax import softmax
from .utils import get_device


class Transformer_LM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        # TransformerBlock params
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float = 10000.0,
    ) -> None:
        """
        vocab_size: int The size of the vocabulary, necessary for determining the dimensionality of the token embedding matrix.
        context_length/max_seq_len: int The maximum context length, necessary for determining the dimensionality of the position embedding matrix.
        num_layers: int The number of Transformer blocks to use.

        TransformerBlock params...
        """
        super().__init__()

        self.device = get_device()
        self.vocab_size = vocab_size
        self.token_embeddings = Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.layers = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, d_ff, max_seq_len, theta) for _ in range(num_layers)]
        )
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)
    

    def forward(self, x: Float[Tensor, "batch_size seq_len"]) -> Float[Tensor, "batch_size seq_len vocab_size"]:
        """
        x: batched sequence of integer token IDs
        """
        x = x.to(self.device)
        embeddings = self.token_embeddings.forward(x)

        for layer in self.layers:
            embeddings = layer.forward(embeddings)
        
        normalized = self.ln_final.forward(embeddings)
        
        logits = self.lm_head.forward(normalized)

        return logits

    def load_weight(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False):
        self.token_embeddings.load_state_dict({"weight": state_dict["token_embeddings.weight"]})

        for i, layer in enumerate(self.layers):
             prefix = f"layers.{i}."
             
             # Attention weights
             q_proj = state_dict[prefix + "attn.q_proj.weight"]
             k_proj = state_dict[prefix + "attn.k_proj.weight"]
             v_proj = state_dict[prefix + "attn.v_proj.weight"]
             # (3 * d_model, d_model)
             w_qkv = torch.cat([q_proj, k_proj, v_proj], dim=0)
             
             layer.attn.w_qkv.load_state_dict({"weight": w_qkv})
             layer.attn.w_o.load_state_dict({"weight": state_dict[prefix + "attn.output_proj.weight"]})

             # Layer Norms
             layer.ln1.load_state_dict({"weight": state_dict[prefix + "ln1.weight"]})
             layer.ln2.load_state_dict({"weight": state_dict[prefix + "ln2.weight"]})

             # FFN
             layer.ffn.w1.load_state_dict({"weight": state_dict[prefix + "ffn.w1.weight"]})
             layer.ffn.w2.load_state_dict({"weight": state_dict[prefix + "ffn.w2.weight"]})
             layer.ffn.w3.load_state_dict({"weight": state_dict[prefix + "ffn.w3.weight"]})
        
        self.ln_final.load_state_dict({"weight": state_dict["ln_final.weight"]})
        self.lm_head.load_state_dict({"weight": state_dict["lm_head.weight"]})

    def top_p_sampling(self, logits: torch.Tensor, top_p: float = 0.9, temperature: float = 1.0) -> torch.Tensor:
        # logits: (batch_size, vocab_size)
        
        # 在softmax之前应用 temperature
        if temperature != 1.0 and temperature > 0:
            logits = logits / temperature
        
        # 对 logits 进行排序, 并计算概率分布
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = softmax(sorted_logits, dim=-1)
        
        # 计算累积概率, 从最高概率开始累加
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # 移除累积概率超过阈值的 tokens，得到掩码
        # [False, False, ..., True, True]
        sorted_indices_to_remove = cumulative_probs > top_p
        
        # 将索引向右移动，保留第一个超过阈值的 token
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0 # 保留最高概率的 token
        
        # 将排序后的索引映射回原始索引
        indices_to_remove = sorted_indices_to_remove.scatter(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
        
        logits[indices_to_remove] = float('-inf')
        
        # 从过滤后的分布中采样
        probs = softmax(logits, dim=-1)
        # 从过滤后的分布中随机采样
        next_token = torch.multinomial(probs, num_samples=1)
        
        return next_token

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        max_seq_len: int,
        temperature: float = 1.0,
        top_p: float = 0.9,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        self.eval()
        curr_input = input_ids
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                if curr_input.shape[1] >= max_seq_len:
                    break

                logits = self.forward(curr_input)
                next_token_logits = logits[:, -1, :]
                
                if temperature > 0:
                    next_token = self.top_p_sampling(next_token_logits, top_p, temperature)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                curr_input = torch.cat([curr_input, next_token], dim=1)
                
                if eos_token_id is not None and next_token.item() == eos_token_id:
                    break
                    
        return curr_input
