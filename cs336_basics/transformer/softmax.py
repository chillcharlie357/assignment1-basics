import torch

def softmax(in_features: torch.Tensor, dim: int, eps: float = 1e-6) -> torch.Tensor:
    global_max,_ = torch.max(in_features, dim, keepdim=True)

    in_features = in_features - global_max


    exp_in_features = torch.exp(in_features)

    global_sum = torch.sum(exp_in_features, dim, keepdim=True)

    return exp_in_features / (global_sum + eps)



