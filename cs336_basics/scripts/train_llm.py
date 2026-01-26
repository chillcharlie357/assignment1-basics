import numpy
import torch
from cs336_basics.transformer import Transformer_LM, get_device
from cs336_basics.training import gradient_clipping,get_batch,lr_scheduler,load_checkpoint,save_checkpoint,AdamW, cross_entropy
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.log import logger

import random
import os
import wandb
from tqdm import tqdm

epochs = 1
learning_rate = 5e-4
dataset_path = "data/tokenids/tokenids.npy"
batch_size = 256
max_seq_len = 128
num_layers = 2
num_heads = 4
d_model = 128
d_ff = 512 

max_norm = 1e-2
# Start a new wandb run to track this script.
run = wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
    entity="hly-personal",
    # Set the wandb project where this run will be logged.
    project="cs336",
    # Track hyperparameters and run metadata.
    config={
        "epochs": epochs,
        "learning_rate": learning_rate,
        "dataset": dataset_path,
        "batch_size": batch_size,
        "max_seq_len": max_seq_len,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "d_model": d_model,
        "d_ff": d_ff,
    },
)



numpy_dataset = numpy.memmap(dataset_path, mode="r")
device = get_device()

logger.info(f"Device: {device}")

# Tokenizer
vocab_path = "data/vocab/tinystories_sample_5M_vocab.pkl"
merges_path = "data/vocab/tinystories_sample_5M_merges.pkl"
special_tokens = ["<|endoftext|>"]
tokenizer = Tokenizer.from_files(vocab_path, merges_path, special_tokens)
vocab_size = tokenizer.vocab_size

# Datasets
# Limit dataset size for faster iteration/debugging
# Use a smaller subset if needed, e.g., first 1000 batches
dataset_len = numpy_dataset.shape[0]
steps_per_epoch = dataset_len // (batch_size * max_seq_len)

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
checkpoint_path = f"data/checkpoints/checkpoint_{device}.pt"
os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

start_epoch = 0
start_iter = 0
try:
    iteration = load_checkpoint(checkpoint_path, model, optimizer)
    start_epoch = iteration // steps_per_epoch
    start_iter = iteration % steps_per_epoch
    print(f"Resuming from iteration {iteration} (Epoch {start_epoch}, Iter {start_iter})")
except FileNotFoundError:
    print("No checkpoint found, starting from scratch")

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

