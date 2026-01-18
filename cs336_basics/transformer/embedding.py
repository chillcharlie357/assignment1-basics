from sympy import tensor
import torch.nn as nn
import torch

class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device | None = None, dtype : torch.dtype | None = None) -> None:
        """
        num_embeddings: int Size of the vocabulary
        embedding_dim: int Dimension of the embedding vectors, i.e., d_model
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()

        self.dtype = dtype if dtype else torch.float32
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=self.device, dtype=self.dtype))

        nn.init.trunc_normal_(
            self.weight,
            std=1,
            mean=0,
            a=-3,
            b=3
        )
    

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        token_ids : (batch_size, sequence_length)
        """
        idx = token_ids.long().to(self.device) # 使用long tensor index
        return self.weight[idx]