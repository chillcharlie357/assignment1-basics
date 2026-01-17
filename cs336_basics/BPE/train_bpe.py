import enum
from functools import partial
import glob
import os
from typing import BinaryIO
import regex as re
import multiprocessing
from collections import defaultdict, Counter


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))



def pre_tokenization(input_seq: bytes, special_tokens: list[str]):
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    # print(input_seq)
    # remove special tokens
    pattern = re.escape('|'.join(t for t in special_tokens)) # 防止 | 破坏正则表达式
    splits = re.split(pattern, input_seq.decode("utf-8")) # 去掉special token
    
    counts = Counter[bytes]()
    for part in splits:
        if not part:
            continue
        iters = re.finditer(PAT, part)
        counts.update(item.group().encode("utf-8") for item in iters)

    
    # print(counts)
    return counts


def get_stats(ids: dict[tuple[bytes, ...], int]) -> dict[tuple[bytes, bytes], int]:
    """
    Given a dictionary of (tuple of bytes) -> frequency,
    return a dictionary of (byte pair) -> frequency.
    """
    stats = defaultdict(int)
    for chunk_ids, count in ids.items():
        for i in range(len(chunk_ids) - 1):
            # 尝试在chunk中合并相邻token
            stats[(chunk_ids[i], chunk_ids[i + 1])] += count
    return stats


def merge_vocab(pair: tuple[bytes, bytes], v_in: dict[tuple[bytes, ...], int]) -> dict[tuple[bytes, ...], int]:
    """
    Given a pair of bytes to merge, update the vocabulary.
    """
    v_out: dict[tuple[bytes, ...], int] = {}
    bigram = pair
    replacement = bigram[0] + bigram[1]
    
    for word, count in v_in.items():
        new_word: list[bytes] = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == bigram[0] and word[i + 1] == bigram[1]:
                # replace with merged token
                new_word.append(replacement)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        v_out[tuple[bytes, ...](new_word)] = count
    return v_out


def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]):
    """
    given a path to an input text file, trains a (byte-level) BPE tokenizer.
    """
    

    # init
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    # add special token
    for i,sp in enumerate(special_tokens):
        vocab[256 + i] = sp.encode('utf-8')
    init_vocab_size = 256 + len(special_tokens)

    merges: list[tuple[bytes, bytes]] = []

    # pre-tokenization 
    global_count = Counter[bytes]()
    with open(input_path, "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.

        def chunk_generator():
            for start,end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                yield f.read(end - start)
        # 多线程计算pre token
        with multiprocessing.Pool(num_processes) as pool:
            partial_pre_tokenization = partial(pre_tokenization, special_tokens=special_tokens)
            results = pool.imap_unordered(partial_pre_tokenization, chunk_generator())
            for result in results:
                global_count.update(result)
        
        # print(global_count)
    
    # Convert to tuple of bytes for BPE
    # e.g. b"hello" -> (b"h", b"e", b"l", b"l", b"o")
    word_counts = {tuple(bytes([b]) for b in token): count for token, count in global_count.items()}

    # BPE merge
    for i in range(vocab_size - init_vocab_size):
        stats = get_stats(word_counts)
        if not stats:
            break
        
        # Find the most frequent pair of tokens
        # 排序规则：1. count 2. pair字典序
        most_frequent_pair: tuple[bytes, bytes] = max(stats, key=lambda x: (stats[x], x))
        
        # Update word counts with the new merge
        word_counts = merge_vocab(most_frequent_pair, word_counts)
        
        # Record the merge
        merges.append(most_frequent_pair)
        new_token = most_frequent_pair[0] + most_frequent_pair[1]
        vocab[init_vocab_size + i] = new_token

        
        # if (i + 1) % 100 == 0:
        #     print(f"Merge {i+1}/{vocab_size - init_vocab_size}: {most_frequent_pair} -> {new_token}")
    
    return (vocab, merges)


if __name__ == "__main__":
    special_token = ["<|endoftext|>"]
    # input_file = "data/TinyStoriesV2-GPT4-valid.txt"
    # Use a file that actually exists for testing
    input_file = "../../tests/fixtures/tinystories_sample.txt"
    if not os.path.exists(input_file):
        # Fallback if running from a different CWD
        input_file = "tests/fixtures/tinystories_sample.txt"
        
    if os.path.exists(input_file):
        train_bpe(input_file, 12800, special_token)
    else:
        print(f"Error: Could not find input file at {input_file} or ../../tests/fixtures/tinystories_sample.txt")