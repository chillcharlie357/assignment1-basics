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
    # 移除 special tokens
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


class BPE_Trainer:
    """
    BPE训练器，负责维护词表统计信息和执行合并操作。
    使用了倒排索引 (Inverted Index) 来加速合并过程。
    """
    def __init__(self, token_counts: dict[bytes, int]):
        self.word_list: list[list[bytes]] = []
        self.word_freqs: list[int] = []
        # 统计所有相邻 token pair 的出现频次
        self.stats: dict[tuple[bytes, bytes], int] = defaultdict(int)
        # 倒排索引：记录每个 token pair 出现在哪些 word 中 (word index)
        self.indices: dict[tuple[bytes, bytes], set[int]] = defaultdict(set)
        
        for token_bytes, count in token_counts.items():
            # 初始时，每个字符作为一个 token
            tokens = [bytes([b]) for b in token_bytes]
            self.word_list.append(tokens)
            self.word_freqs.append(count)
            
            idx = len(self.word_list) - 1
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i+1])
                self.stats[pair] += count
                self.indices[pair].add(idx)

    def merge(self, pair: tuple[bytes, bytes], new_token: bytes):
        """
        将最频繁的 token pair 合并为一个新 token。
        只更新包含该 pair 的 word，并增量更新统计信息。
        """
        p0, p1 = pair
        # 如果 pair 不在索引中，说明无需合并
        if pair not in self.indices:
            return
            
        # 获取所有包含该 pair 的 word 索引
        current_indices = list(self.indices[pair])
        
        for idx in current_indices:
            word = self.word_list[idx]
            freq = self.word_freqs[idx]
            
            # 检查 pair 是否仍然存在（可能在之前的迭代中被破坏），并构建新 word
            new_word = []
            i = 0
            has_match = False
            
            # 扫描当前 word，进行合并操作
            while i < len(word):
                if i < len(word) - 1 and word[i] == p0 and word[i+1] == p1:
                    new_word.append(new_token)
                    has_match = True
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            
            if has_match:
                # 更新统计信息 (stats 和 indices)
                # 1. 减少旧 pairs 的计数
                for j in range(len(word) - 1):
                    p = (word[j], word[j+1])
                    self.stats[p] -= freq
                    if self.stats[p] <= 0:
                        if p in self.stats: 
                            del self.stats[p]
                    
                    if idx in self.indices[p]:
                        self.indices[p].remove(idx)
                        if not self.indices[p]:
                            del self.indices[p]

                # 更新 word 列表
                self.word_list[idx] = new_word
                
                # 2. 增加新 pairs 的计数
                for j in range(len(new_word) - 1):
                    p = (new_word[j], new_word[j+1])
                    self.stats[p] += freq
                    self.indices[p].add(idx)
        
        # 清理已合并 pair 的记录
        if pair in self.stats:
             del self.stats[pair]
        if pair in self.indices:
             del self.indices[pair]


def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]):
    """
    给定输入文本文件路径，训练一个 (byte-level) BPE tokenizer。
    """
    

    # 初始化
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    # 添加 special token
    for i,sp in enumerate(special_tokens):
        vocab[256 + i] = sp.encode('utf-8')
    init_vocab_size = 256 + len(special_tokens)

    merges: list[tuple[bytes, bytes]] = []

    # 预分词 (pre-tokenization)
    logger.info("start pre tokenization")
    global_count = Counter[bytes]()
    with open(input_path, "rb") as f:
        num_processes = 8
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        def chunk_generator():
            for start,end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                yield f.read(end - start)
        # 多进程并行计算 pre-token 统计
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
    
    # 初始化 BPE 训练器
    logger.info("initializing BPE trainer stats...")
    trainer = BPE_Trainer(global_count)

    # 执行 BPE 合并循环
    logger.info("start merge")
    for i in trange(vocab_size - init_vocab_size):
        if not trainer.stats:
            break
        
        # 找到频次最高的 token pair
        # 排序规则：1. count 2. pair字典序
        most_frequent_pair: tuple[bytes, bytes] = max(trainer.stats, key=lambda x: (trainer.stats[x], x))
        
        # 创建新 token
        new_token = most_frequent_pair[0] + most_frequent_pair[1]
        
        # 更新 word 列表并进行合并
        trainer.merge(most_frequent_pair, new_token)
        
        # 记录合并操作
        merges.append(most_frequent_pair)
        vocab[init_vocab_size + i] = new_token
        
    logger.info("finish bpe merge")
    return (vocab, merges)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train BPE tokenizer")
    parser.add_argument("--input_path", type=str, required=True, help="Path to input text file")
    parser.add_argument("--vocab_path", type=str, required=True, help="Path to save vocab pkl")
    parser.add_argument("--merges_path", type=str, required=True, help="Path to save merges pkl")
    parser.add_argument("--vocab_size", type=int, default=10000, help="Target vocabulary size")
    parser.add_argument("--special_tokens", nargs="+", default=["<|endoftext|>"], help="List of special tokens")
    
    args = parser.parse_args()
    
    if os.path.exists(args.input_path):
        vocab, merges = train_bpe(args.input_path, args.vocab_size, args.special_tokens)
        logger.info("finish train bpe")
        import pickle
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(args.vocab_path), exist_ok=True)
        os.makedirs(os.path.dirname(args.merges_path), exist_ok=True)
        
        with open(args.vocab_path, "wb") as f:
            pickle.dump(vocab, f)
            logger.info(f"save vocab to {args.vocab_path}")
        with open(args.merges_path, "wb") as f:
            pickle.dump(merges, f)
            logger.info(f"save merges to {args.merges_path}")
            
    else:
        logger.error(f"Error: Could not find input file at {args.input_path}")