import torch
import torch.nn.functional as F
import hydra
from omegaconf import DictConfig
import os
import glob
from cs336_basics.log import setup_logging
from cs336_basics.training import load_checkpoint
from cs336_basics.transformer import Transformer_LM
from cs336_basics.tokenizer import Tokenizer
import numpy
from cs336_basics.transformer.utils import get_device

def get_checkpoint_path(checkpoint_dir: str, device: str) -> str | None:
    # 1. Try best checkpoint
    best_path = os.path.join(checkpoint_dir, f"best_checkpoint_{device}.pt")
    if os.path.exists(best_path):
        return best_path
        
    # 2. Try final model timestamped
    pattern = os.path.join(checkpoint_dir, f"model_{device}_*.pt")
    files = glob.glob(pattern)
    if files:
        files.sort(key=os.path.getmtime)
        return files[-1]
        
    # 3. Try latest checkpoint timestamped
    pattern = os.path.join(checkpoint_dir, f"checkpoint_{device}_*.pt")
    files = glob.glob(pattern)
    if files:
        files.sort(key=os.path.getmtime)
        return files[-1]
        
    # 4. Fallback
    old_path = os.path.join(checkpoint_dir, f"checkpoint_{device}.pt")
    if os.path.exists(old_path):
        return old_path
        
    return None

def decode(model: Transformer_LM, tokenizer: Tokenizer, cfg: DictConfig, device: torch.device):
    logger = setup_logging(cfg)
    prompt = cfg.decoding.prompt
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
        max_new_tokens=cfg.decoding.max_new_tokens,
        max_seq_len=cfg.model.max_seq_len,
        temperature=cfg.decoding.temperature,
        top_p=cfg.decoding.top_p,
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
    ).to(device)
    
    checkpoint_dir = hydra.utils.to_absolute_path(cfg.decoding.checkpoint_dir)
    
    if cfg.decoding.checkpoint_path:
        checkpoint_path = hydra.utils.to_absolute_path(cfg.decoding.checkpoint_path)
    else:
        checkpoint_path = get_checkpoint_path(checkpoint_dir, str(device))
        
    logger.info(f"Loading checkpoint from {checkpoint_path}")

    if checkpoint_path:
        try:
            load_checkpoint(checkpoint_path, model)
        except FileNotFoundError:
            logger.warning("Checkpoint file not found, using random weights")
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            raise e
    else:
        logger.warning("No checkpoint found, using random weights")

    decode(model, tokenizer, cfg, device)

if __name__ == "__main__":
    main()
