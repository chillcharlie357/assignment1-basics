from functools import partial
import os
import regex as re
import multiprocessing
from collections import defaultdict, Counter
from cs336_basics.log import logger
from tqdm import tqdm, trange
from cs336_basics.pretokenization_example import find_chunk_boundaries



def pre_tokenization(input_seq: bytes, special_tokens: list[str]):
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    # print(input_seq)
    # remove special tokens
    pattern = '|'.join(re.escape(t) for t in special_tokens)
    splits = re.split(pattern, input_seq.decode("utf-8")) # 去掉special token
    
    counts = Counter[bytes]()
    for part in tqdm(splits, desc="pre-tokenization splits"):
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
    logger.info("start pre tokenization")
    global_count = Counter[bytes]()
    with open(input_path, "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        def chunk_generator():
            for start,end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                yield f.read(end - start)
        # 多线程计算pre token
        with tqdm(total=len(boundaries)-1, desc="pre tokenization") as pbar:
            pool = multiprocessing.Pool(num_processes)
            try:
                partial_pre_tokenization = partial(pre_tokenization, special_tokens=special_tokens)
                results = pool.imap_unordered(partial_pre_tokenization, chunk_generator())
                for result in results:
                    global_count.update(result)
                    pbar.update(1)
            finally:
                pool.close() # 停止接受新任务
                pool.join() # 等待所有子进程结束
        
        # print(global_count)
    
    # Convert to tuple of bytes for BPE
    # e.g. b"hello" -> (b"h", b"e", b"l", b"l", b"o")
    word_counts = {tuple(bytes([b]) for b in token): count for token, count in global_count.items()}

    # BPE merge
    logger.info("start merge")
    for i in trange(vocab_size - init_vocab_size):
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
        #     logger.info(f"Merge {i+1}: {most_frequent_pair} -> {new_token}")
    logger.info("finish bpe merge")
    return (vocab, merges)


if __name__ == "__main__":
    from cs336_basics.config import config
    
    special_tokens = config.tokenizer.training.special_tokens
    input_file = config.tokenizer.data.input_path
    vocab_file = config.tokenizer.data.vocab_path
    merges_file = config.tokenizer.data.merges_path
    vocab_size = config.tokenizer.training.vocab_size
        
    if os.path.exists(input_file):
        vocab, merges = train_bpe(input_file, vocab_size, special_tokens)
        logger.info("finish train bpe")
        import pickle
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(vocab_file), exist_ok=True)
        os.makedirs(os.path.dirname(merges_file), exist_ok=True)
        
        with open(vocab_file, "wb") as f:
            pickle.dump(vocab, f)
            logger.info(f"save vocab to {vocab_file}")
        with open(merges_file, "wb") as f:
            pickle.dump(merges, f)
            logger.info(f"save merges to {merges_file}")
            
    else:
        logger.error(f"Error: Could not find input file at {input_file}")