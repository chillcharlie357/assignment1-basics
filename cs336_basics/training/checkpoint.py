import typing
from torch import nn, Tensor, optim
import os

import torch
def save_checkpoint(model: nn.Module, optimizer: optim.Optimizer, iteration: int, out : str | os.PathLike | typing.BinaryIO | typing.IO[bytes]):
    states = {
        "iter": iteration,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict()
    }

    torch.save(states, out)


def load_checkpoint(src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes], model: nn.Module, optimizer: optim.Optimizer | None = None) -> int:
    states = torch.load(src)

    iter = int(states['iter'])
    model.load_state_dict(states['model'])
    if optimizer is not None:
        optimizer.load_state_dict(states['optimizer'])

    return iter