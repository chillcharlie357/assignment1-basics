import numpy
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from cs336_basics.transformer import Transformer_LM, get_device
from cs336_basics.training import gradient_clipping,get_batch,lr_scheduler,load_checkpoint,save_checkpoint,AdamW, cross_entropy
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.log import setup_logging

import random
import os
import wandb
from tqdm import tqdm
from datetime import datetime

import glob

def get_latest_checkpoint(checkpoint_dir: str, device: str) -> str | None:
    """Find the latest checkpoint file in the directory."""
    # Check for timestamped checkpoints
    pattern = os.path.join(checkpoint_dir, f"checkpoint_{device}_*.pt")
    files = glob.glob(pattern)
    if not files:
        # Fallback to old naming convention if exists
        old_path = os.path.join(checkpoint_dir, f"checkpoint_{device}.pt")
        if os.path.exists(old_path):
            return old_path
        return None
    # Sort by modification time
    files.sort(key=os.path.getmtime)
    return files[-1]

def cleanup_checkpoints(checkpoint_dir: str, device: str, keep_last_n: int = 5):
    """Keep only the last N checkpoints."""
    pattern = os.path.join(checkpoint_dir, f"checkpoint_{device}_*.pt")
    files = glob.glob(pattern)
    files.sort(key=os.path.getmtime)
    
    if len(files) > keep_last_n:
        files_to_delete = files[:-keep_last_n]
        for f in files_to_delete:
            try:
                os.remove(f)
            except OSError as e:
                print(f"Error deleting old checkpoint {f}: {e}")

@torch.no_grad()
def evaluate(model, numpy_dataset, batch_size, max_seq_len, device, num_batches=100):
    model.eval()
    losses = []
    # If dataset is small, use all of it, otherwise sample
    # For simplicity, we just take random batches or sequential from start
    # Let's take sequential for reproducibility if we seed properly, or just random
    # Random sampling for validation set is fine
    
    dataset_len = numpy_dataset.shape[0]
    # Ensure we don't go out of bounds
    max_start_idx = dataset_len - (batch_size * max_seq_len + 1)
    
    if max_start_idx <= 0:
        return float('nan')
        
    for _ in range(num_batches):
        # get_batch internally handles random sampling
        input, target = get_batch(numpy_dataset, batch_size, max_seq_len, device)
        output = model.forward(input)
        loss = cross_entropy(output, target)
        losses.append(loss.item())
        
    model.train()
    return sum(losses) / len(losses)

@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig):
    logger = setup_logging(cfg)
    
    # Hyperparameters from config
    epochs = cfg.training.epochs
    learning_rate = cfg.training.learning_rate
    batch_size = cfg.training.batch_size
    max_norm = cfg.training.max_norm
    
    max_seq_len = cfg.model.max_seq_len
    num_layers = cfg.model.num_layers
    num_heads = cfg.model.num_heads
    d_model = cfg.model.d_model
    d_ff = cfg.model.d_ff
    
    # Paths (convert to absolute because Hydra changes cwd)
    dataset_path = hydra.utils.to_absolute_path(cfg.dataset.dataset_path)
    vocab_path = hydra.utils.to_absolute_path(cfg.tokenizer.vocab_path)
    merges_path = hydra.utils.to_absolute_path(cfg.tokenizer.merges_path)
    
    # Valid dataset path
    valid_dataset_path = None
    if "valid_dataset_path" in cfg.dataset and cfg.dataset.valid_dataset_path:
        valid_dataset_path = hydra.utils.to_absolute_path(cfg.dataset.valid_dataset_path)
    
    # Start a new wandb run to track this script.
    run = wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="hly-personal",
        # Set the wandb project where this run will be logged.
        project="cs336",
        # Track hyperparameters and run metadata.
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    numpy_dataset = numpy.memmap(dataset_path, mode="r")
    
    numpy_valid_dataset = None
    if valid_dataset_path and os.path.exists(valid_dataset_path):
        numpy_valid_dataset = numpy.memmap(valid_dataset_path, mode="r")
        logger.info(f"Loaded validation dataset from {valid_dataset_path}")
    else:
        logger.info("No validation dataset found or provided, skipping validation")

    device = get_device()
    
    logger.info(f"Device: {device}")
    
    # Tokenizer
    special_tokens = list(cfg.tokenizer.special_tokens)
    tokenizer = Tokenizer.from_files(vocab_path, merges_path, special_tokens)
    vocab_size = tokenizer.vocab_size
    
    # Datasets
    # Limit dataset size for faster iteration/debugging
    # Use a smaller subset if needed, e.g., first 1000 batches
    dataset_len = numpy_dataset.shape[0]
    steps_per_epoch = None
    if device == torch.device("cuda"):
        steps_per_epoch = dataset_len // (batch_size * max_seq_len)
    else:
        # cpu和mps, 训练速度较慢
        steps_per_epoch = 1000
    
    
    model = Transformer_LM(
        vocab_size,
        num_layers,
        d_model,
        num_heads,
        d_ff,
        max_seq_len,
    ).to(device)
    
    # Calculate and print model size
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {num_params:,}")
    
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    # Try to load checkpoint if exists
    # Use absolute path for checkpoints to persist across runs
    checkpoint_dir = hydra.utils.to_absolute_path(cfg.training.checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Logic to find the latest checkpoint
    checkpoint_path = get_latest_checkpoint(checkpoint_dir, str(device))
    
    start_epoch = 0
    start_iter = 0
    if checkpoint_path:
        try:
            iteration = load_checkpoint(checkpoint_path, model, optimizer)
            start_epoch = iteration // steps_per_epoch
            start_iter = iteration % steps_per_epoch
            logger.info(f"Resuming from iteration {iteration} (Epoch {start_epoch}, Iter {start_iter}) from {checkpoint_path}")
        except FileNotFoundError:
            logger.info("Checkpoint file not found (race condition?), starting from scratch")
        except RuntimeError as e:
            if "size mismatch" in str(e) or "Unexpected key(s)" in str(e):
                 logger.warning(f"Checkpoint configuration mismatch (starting from scratch): {e}")
            else:
                 logger.error(f"Error loading checkpoint: {e}")
                 raise e
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
    else:
        logger.info("No checkpoint found, starting from scratch")
    
    total_steps = epochs * steps_per_epoch
    initial_steps = start_epoch * steps_per_epoch + start_iter
    
    best_valid_loss = float('inf')

    with tqdm(total=total_steps, initial=initial_steps, desc="Training") as pbar:
        for epoch in range(start_epoch, epochs):
            # If resuming, skip iterations in the current epoch
            current_start_iter = start_iter if epoch == start_epoch else 0
            
            for iter in range(current_start_iter, steps_per_epoch):
                input, target = get_batch(numpy_dataset, batch_size, max_seq_len, device)
                
                model.train()
    
                output = model.forward(input)
                
                loss = cross_entropy(output, target)
                loss.backward()
    
                gradient_clipping(model.parameters(), max_l2_norm=max_norm)
    
                optimizer.step()
                optimizer.zero_grad()
                
                current_global_step = iter + epoch * steps_per_epoch
                global_warm_up = 0.1 * current_global_step
                
                # Update learning rate
                lr = lr_scheduler(current_global_step, learning_rate, learning_rate * 0.1, global_warm_up, total_steps)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr
                
                run.log(
                    {
                        "iter": current_global_step,
                        "loss": loss.item(),
                        "lr": lr
                    }
                )
                pbar.update(1)
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    
                # Save checkpoint every 1000 steps or at end of epoch
                if (current_global_step + 1) % 1000 == 0 or (iter == steps_per_epoch - 1):
                     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                     current_checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{device}_{timestamp}.pt")
                     save_checkpoint(model, optimizer, current_global_step + 1, current_checkpoint_path)
                     cleanup_checkpoints(checkpoint_dir, str(device))
            
            # Validation at end of epoch
            if numpy_valid_dataset is not None:
                logger.info(f"Running validation for epoch {epoch}...")
                valid_loss = evaluate(model, numpy_valid_dataset, batch_size, max_seq_len, device)
                logger.info(f"Epoch {epoch} validation loss: {valid_loss:.4f}")
                run.log({"valid_loss": valid_loss, "epoch": epoch})
                
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    best_checkpoint_path = os.path.join(checkpoint_dir, f"best_checkpoint_{device}.pt")
                    save_checkpoint(model, optimizer, current_global_step + 1, best_checkpoint_path)
                    logger.info(f"New best checkpoint saved with loss {valid_loss:.4f}")
        
        # Reset start_iter after the first resumed epoch is done
        start_iter = 0
        # save final checkpoint
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_checkpoint_path = os.path.join(checkpoint_dir, f"model_{device}_{timestamp}.pt")
        save_checkpoint(model, optimizer, current_global_step + 1, final_checkpoint_path)
    
        
    
    
    # Finish the run and upload any remaining data.
    run.finish()

if __name__ == "__main__":
    main()

