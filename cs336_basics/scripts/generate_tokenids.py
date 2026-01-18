from cs336_basics.tokenizer.tokenizer import Tokenizer
from cs336_basics.config import config
from cs336_basics.pretokenization_example import find_chunk_boundaries
from tqdm import tqdm
import numpy as np
import os

class FileChunkIterable:
    def __init__(self, file_path, num_chunks=100):
        self.file_path = file_path
        self.num_chunks = num_chunks

    def __iter__(self):
        with open(self.file_path, "rb") as f:
            boundaries = find_chunk_boundaries(f, self.num_chunks, b"<|endoftext|>")
            for start, end in tqdm(zip(boundaries[:-1], boundaries[1:]), total=len(boundaries)-1):
                f.seek(start)
                chunk = f.read(end - start).decode('utf-8', errors='ignore')
                yield chunk


if __name__ == "__main__":
    
    vocab_path = config.tokenizer.data.vocab_path
    merges_path = config.tokenizer.data.merges_path
    special_tokens = config.tokenizer.training.special_tokens
    input_file = config.tokenizer.data.input_path
    tokenIDs_path = "data/tokenids/"

    tokenizer = Tokenizer.from_files(vocab_filepath=vocab_path, 
                            merges_filepath=merges_path, 
                            special_tokens=special_tokens)

    chunks = FileChunkIterable(input_file)
    ids = tokenizer.encode_iterable(chunks)
    
    os.makedirs(os.path.dirname(tokenIDs_path), exist_ok=True)
    
    # Serialize token IDs as a NumPy array of datatype uint16
    ids_array = np.fromiter(ids, dtype=np.uint16)
    np.save(tokenIDs_path, ids_array)
    print(f"Token IDs saved to {tokenIDs_path}.npy")
    


    