from cs336_basics.tokenizer.tokenizer import Tokenizer
from cs336_basics.pretokenization_example import find_chunk_boundaries
from tqdm import tqdm
import numpy as np
import os

import multiprocessing
from functools import partial

# Global variable for worker process
global_tokenizer = None

def init_worker(vocab_path, merges_path, special_tokens):
    """Initialize tokenizer in worker process"""
    global global_tokenizer
    global_tokenizer = Tokenizer.from_files(
        vocab_filepath=vocab_path,
        merges_filepath=merges_path,
        special_tokens=special_tokens
    )

def process_chunk(args):
    """Process a single file chunk"""
    file_path, start, end = args
    
    with open(file_path, "rb") as f:
        f.seek(start)
        # Read and decode
        chunk_bytes = f.read(end - start)
        chunk_text = chunk_bytes.decode('utf-8', errors='ignore')
        
    # Tokenize
    if global_tokenizer is None:
        raise RuntimeError("Tokenizer not initialized in worker")
        
    return global_tokenizer.encode(chunk_text)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate token IDs")
    parser.add_argument("--input_path", type=str, required=True, help="Path to input text file")
    parser.add_argument("--vocab_path", type=str, required=True, help="Path to vocab pkl")
    parser.add_argument("--merges_path", type=str, required=True, help="Path to merges pkl")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save token IDs npy")
    parser.add_argument("--special_tokens", nargs="+", default=["<|endoftext|>"], help="List of special tokens")
    parser.add_argument("--num_workers", type=int, default=os.cpu_count(), help="Number of worker processes")
    
    args = parser.parse_args()

    # Calculate boundaries using FileChunkIterable logic (but simpler here)
    file_path = args.input_path
    split_token = args.special_tokens[0].encode('utf-8') if args.special_tokens else b"<|endoftext|>"
    
    # Calculate number of chunks
    chunk_size_mb = 64
    file_size = os.path.getsize(file_path)
    num_chunks = max(1, file_size // (chunk_size_mb * 1024 * 1024))
    num_chunks = min(num_chunks, 10000) # Cap at 10000 chunks
    
    print(f"Finding chunk boundaries for {file_path}...")
    with open(file_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_chunks, split_token)
    
    print(f"Found {len(boundaries)-1} chunks. Starting parallel tokenization with {args.num_workers} workers...")
    
    # Prepare arguments for workers
    chunk_args = [
        (file_path, boundaries[i], boundaries[i+1]) 
        for i in range(len(boundaries)-1)
    ]
    
    # Initialize pool
    with multiprocessing.Pool(
        processes=args.num_workers,
        initializer=init_worker,
        initargs=(args.vocab_path, args.merges_path, args.special_tokens)
    ) as pool:
        # Process chunks in parallel
        results = []
        with tqdm(total=len(chunk_args), desc="Tokenizing chunks", unit="chunk") as pbar:
            for chunk_ids in pool.imap(process_chunk, chunk_args):
                results.append(chunk_ids)
                pbar.update(1)
                
    # Flatten results
    print("Flattening results and saving...")
    
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # Flatten and save
    # np.concatenate is faster than np.fromiter if we already have lists
    # Convert each chunk to array first
    arrays = [np.array(chunk, dtype=np.uint16) for chunk in results]
    final_array = np.concatenate(arrays)
    
    np.save(args.output_path, final_array)
    print(f"Token IDs saved to {args.output_path}. Total tokens: {len(final_array)}")
    


    