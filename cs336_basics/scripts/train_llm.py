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
    dataset_path = hydra.utils.to_absolute_path(cfg.data.dataset_path)
    vocab_path = hydra.utils.to_absolute_path(cfg.data.vocab_path)
    merges_path = hydra.utils.to_absolute_path(cfg.data.merges_path)
    
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
    device = get_device()
    
    logger.info(f"Device: {device}")
    
    # Tokenizer
    special_tokens = list(cfg.dataset.special_tokens)
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
    checkpoint_dir = hydra.utils.to_absolute_path("data/checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{device}.pt")
    
    start_epoch = 0
    start_iter = 0
    try:
        iteration = load_checkpoint(checkpoint_path, model, optimizer)
        start_epoch = iteration // steps_per_epoch
        start_iter = iteration % steps_per_epoch
        print(f"Resuming from iteration {iteration} (Epoch {start_epoch}, Iter {start_iter})")
    except FileNotFoundError:
        print("No checkpoint found, starting from scratch")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
    
    total_steps = epochs * steps_per_epoch
    initial_steps = start_epoch * steps_per_epoch + start_iter
    
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
                
                # Update learning rate
                lr = lr_scheduler(current_global_step, learning_rate, learning_rate * 0.1, 100, total_steps)
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
                     save_checkpoint(model, optimizer, current_global_step + 1, checkpoint_path)
        
        # Reset start_iter after the first resumed epoch is done
        start_iter = 0
        
    
    
    # Finish the run and upload any remaining data.
    run.finish()

if __name__ == "__main__":
    main()

