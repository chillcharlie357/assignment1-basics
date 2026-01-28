import torch
import torch.nn.functional as F
import hydra
from omegaconf import DictConfig
import os
from cs336_basics.log import setup_logging
from cs336_basics.training import load_checkpoint
from cs336_basics.transformer import Transformer_LM
from cs336_basics.tokenizer import Tokenizer
import numpy
from cs336_basics.transformer.utils import get_device

def decode(model: Transformer_LM, tokenizer: Tokenizer, max_seq_len: int, device: torch.device):
    logger = setup_logging() # Ensure logger is setup if called independently, or reuse global
    prompt = "Once upon a time, "
    input_ids = tokenizer.encode(prompt)
    # 增加batch维度
    input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)
    
    eos_token_id = None
    if tokenizer.special_tokens:
         # 简单的尝试获取 EOS token id
         try:
             # 假设 <|endoftext|> 是 EOS
             encoded = tokenizer.encode("<|endoftext|>")
             if len(encoded) == 1:
                 eos_token_id = encoded[0]
         except:
             pass

    logger.info(f"Prompt: {prompt}")
    
    output_ids = model.generate(
        input_ids=input_tensor,
        max_new_tokens=50,
        max_seq_len=max_seq_len,
        temperature=0.8,
        top_p=0.9,
        eos_token_id=eos_token_id
    )
    
    # Decode the full sequence
    output_text = tokenizer.decode(output_ids[0].tolist())
    logger.info(f"Generated text: {output_text}")



@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig):
    logger = setup_logging(cfg)
    device = get_device()
    
    # Model params
    max_seq_len = cfg.model.max_seq_len
    num_layers = cfg.model.num_layers
    num_heads = cfg.model.num_heads
    d_model = cfg.model.d_model
    d_ff = cfg.model.d_ff
    
    # Tokenizer
    vocab_path = hydra.utils.to_absolute_path(cfg.tokenizer.vocab_path)
    merges_path = hydra.utils.to_absolute_path(cfg.tokenizer.merges_path)
    special_tokens = list(cfg.tokenizer.special_tokens)
    tokenizer = Tokenizer.from_files(vocab_path, merges_path, special_tokens)
    vocab_size = tokenizer.vocab_size

    model = Transformer_LM(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        num_layers=num_layers,
        num_heads=num_heads,
        d_model=d_model,
        d_ff=d_ff,
    )
    
    checkpoint_dir = hydra.utils.to_absolute_path("data/checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{device}.pt")

    try:
        load_checkpoint(checkpoint_path, model)
    except FileNotFoundError:
        logger.warning("No checkpoint found, using random weights")
    except Exception as e:
        logger.error(f"Error loading checkpoint: {e}")
        raise e

    decode(model, tokenizer, max_seq_len, device)

if __name__ == "__main__":
    main()
