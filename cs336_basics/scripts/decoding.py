import torch
import torch.nn.functional as F
from cs336_basics.log import logger
from cs336_basics.training import load_checkpoint
from cs336_basics.transformer import Transformer_LM
from cs336_basics.tokenizer import Tokenizer
import numpy
from cs336_basics.transformer.utils import get_device

def decode(model: Transformer_LM, tokenizer: Tokenizer, max_seq_len: int, device: torch.device):
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



if __name__ == "__main__":
    dataset_path = "data/tokenids/tokenids.npy"
    batch_size = 256
    max_seq_len = 128
    num_layers = 2
    num_heads = 4
    d_model = 128
    d_ff = 512 
    numpy_dataset = numpy.memmap(dataset_path, mode="r")
    device = get_device()

    # Tokenizer
    vocab_path = "data/vocab/tinystories_sample_5M_vocab.pkl"
    merges_path = "data/vocab/tinystories_sample_5M_merges.pkl"
    special_tokens = ["<|endoftext|>"]
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

    load_checkpoint(f"data/checkpoints/checkpoint_{device}.pt", model)

    decode(model, tokenizer, max_seq_len, device)
