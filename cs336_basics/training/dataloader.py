import string
from numpy import  ndarray
from torch import Tensor
import torch
from jaxtyping import Float

def get_batch(dataset: ndarray, batch_size: int, seq_len: int, device: string) -> tuple[Float[Tensor, "batch_size seq_len"], Float[Tensor, "batch_size seq_len"]]:
    """
    Get a batch of data from the inputs array.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): The batch size.
        seq_len (int): The sequence length.
        device (string): The device to move the tensors to.

    Returns:
        (Float[Tensor, "batch_size seq_len"], Float[Tensor, "batch_size seq_len"]): The input and target tensors.
    """
    
    dataset_len = dataset.shape[0]

    # (batch_size,)
    indices = torch.randint(0, dataset_len - seq_len, (batch_size,))
    
    # (seq_len,)
    offsets = torch.arange(seq_len)
    
    # (batch_size, seq_len)
    batch_indices = indices.unsqueeze(1) + offsets.unsqueeze(0)
    
    # Convert to numpy for slicing to support np.memmap without warnings/issues
    # PyTorch does not support non-writable tensors (like read-only memmap) well
    batch_indices_np = batch_indices.numpy()
    
    batch_input: Tensor = torch.from_numpy(dataset[batch_indices_np]).to(device)
    batch_target: Tensor = torch.from_numpy(dataset[batch_indices_np + 1]).to(device)

    return (batch_input, batch_target)