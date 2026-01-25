import numpy
import torch
from cs336_basics.transformer import Transformer_LM, get_device
from cs336_basics.training import gradient_clipping,get_batch,lr_scheduler,load_checkpoint,save_checkpoint,AdamW, cross_entropy
from cs336_basics.tokenizer import Tokenizer

import random

import wandb
from tqdm import tqdm

epochs = 10
learning_rate = 1e-3
dataset_path = "data/tokenids/tokenids.npy"
batch_size = 10
max_seq_len = 256
num_layers = 2
num_heads = 4
# vocab_size = 10000 # Will be set by tokenizer
d_model = 128
d_ff = 128

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
        "learning_rate": 1e-3,
        "dataset": dataset_path,
        "batch_size": batch_size,
        "max_seq_len": max_seq_len,
    },
)



numpy_dataset = numpy.memmap(dataset_path, mode="r")
device = get_device()

# Tokenizer
vocab_path = "data/vocab/tinystories_sample_5M_vocab.pkl"
merges_path = "data/vocab/tinystories_sample_5M_merges.pkl"
special_tokens = ["<|endoftext|>"]
tokenizer = Tokenizer.from_files(vocab_path, merges_path, special_tokens)
vocab_size = tokenizer.vocab_size

# Datasets
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
optimizer = AdamW(model.parameters(), lr=learning_rate)
with tqdm(total=epochs * steps_per_epoch, desc="Training") as pbar:
    for epoch in range(epochs):
        for iter in range(steps_per_epoch):
            input, target = get_batch(numpy_dataset, batch_size, max_seq_len, device)
            
            model.train()

            output = model.forward(input)
            
            loss = cross_entropy(output, target)
            loss.backward()

            gradient_clipping(model.parameters(), max_l2_norm=max_norm)

            optimizer.step()
            optimizer.zero_grad()

            run.log(
                {
                    "iter": iter + epoch * steps_per_epoch,
                    "loss": loss.item()
                }
            )
            pbar.update(1)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    


# Finish the run and upload any remaining data.
run.finish()

