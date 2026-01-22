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
        theta: int,
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

    @override
    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False):
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
