import torch

def get_device(device: torch.device | None = None) -> torch.device:
        if device is not None:
            return device
        elif torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")