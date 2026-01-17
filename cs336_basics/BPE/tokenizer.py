import pickle
import token
from collections.abc import Iterable
import regex as re
from collections import Counter

from sympy import false
from cs336_basics.log import logger

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Tokenizer:
    def __init__(
        self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None
    ) -> None:
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens

        self.vocab_size: int = len(vocab)
        self.vocab_inv: dict[bytes, int] = {v: k for k, v in self.vocab.items()}
        self.vocab_values = set(self.vocab.values())
        self.ranks = dict(zip(merges, range(len(merges))))

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath, special_tokens=None):
        vocab = {}
        merges = []
        try:
            with open(vocab_filepath, "rb") as f:
                vocab = pickle.load(f)
            with open(merges_filepath, "rb") as f:
                merges = pickle.load(f)
        except FileNotFoundError as e:
            logger.error(f"file {e.filename} not found")
            raise
        except Exception as e:
            logger.error(f"load error: {e}")

        return cls(vocab, merges, special_tokens)

    def _pre_tokenize(self, text: str) -> list[tuple[bytes, bool]]:
        # pre tokenize
        # 先处理 special tokens
        if self.special_tokens:
            # 使用捕获组 (...) 来保留分隔符（即 special tokens）
            # 对 special_tokens 按长度降序排序，确保长 token 优先匹配
            sorted_tokens = sorted(self.special_tokens, key=len, reverse=True)
            pattern = "(" + "|".join(re.escape(t) for t in sorted_tokens) + ")"
            
            splits = re.split(pattern, text)
            chunks: list[tuple[bytes, bool]] = []
            for part in splits:
                if not part:
                    continue

                if part in self.special_tokens:
                     chunks.append((part.encode("utf-8"), True))
                else:
                    # 否则应用 regex 进行进一步切分
                    _iters = re.finditer(PAT, part)
                    chunks.extend([(item.group().encode("utf-8"), False) for item in _iters])
            return chunks
        else:
            _iters = re.finditer(PAT,text)
            return [(item.group().encode("utf-8"), False) for item in _iters]

    def _bpe_merge(self, text_chunks: list[tuple[bytes, bool]]):
        # 对每个 pre-tokenized chunk 分别进行 BPE merge
        merged_chunks = []
        for chunk, is_special in text_chunks:
            # special token 不做处理
            if is_special:
                merged_chunks.append(chunk)
                continue

            # 普通token将 bytes 转为 list[bytes] 用于处理，例如 b"hello" -> [b"h", b"e", b"l", b"l", b"o"]
            word = [bytes([b]) for b in chunk]
            
            while len(word) > 1:
                # Find the pair with the lowest rank
                min_rank = float("inf")
                min_pair = None
                
                for i in range(len(word) - 1):
                    pair = (word[i], word[i+1])
                    if pair in self.ranks:
                        rank = self.ranks[pair]
                        if rank < min_rank:
                            min_rank = rank
                            min_pair = pair
                
                if min_pair is None:
                    break
                
                # Merge the pair
                new_word = []
                i = 0
                while i < len(word):
                    if i < len(word) - 1 and (word[i], word[i+1]) == min_pair:
                        new_word.append(word[i] + word[i+1])
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                word = new_word
            
            merged_chunks.extend(word)
        return merged_chunks

    def encode(self, text: str) -> list[int]:
        if not text:
            return []

        text_chunks = self._pre_tokenize(text)

        logger.info("pre tokenized")

        merged_bytes = self._bpe_merge(text_chunks)

        logger.info("merged")

        # tokenize
        try:
            tokens = [self.vocab_inv[byte] for byte in merged_bytes]
            logger.info("toenized")
            return tokens
        except Exception as e:
            logger.error(e)
            raise

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        if not ids:
            return ""
        bytes_list: list[bytes] = []
        for id in ids:
            try:
                bytes_list.append(self.vocab[id])
            except Exception as e:
                logger.error(f"id {id} not exist: {e}")
                raise
        
        output = b"".join(bytes_list)
        return output.decode('utf-8', errors="replace")

if __name__ == "__main__":
    vocab_path = "data/save/tinystories_sample_5M_vocab.pkl"
    merges_path = "data/save/tinystories_sample_5M_merges.pkl"
    special_tokens = ["<|endoftext|>"]

    tokenizer = Tokenizer.from_files(vocab_filepath=vocab_path, 
                        merges_filepath=merges_path, 
                        special_tokens=special_tokens)

    logger.info("load tokenizer")

    tokens = tokenizer.encode("Héllò hôw <|endoftext|><|endoftext|> are ü? 🙃<|endoftext|>")
    logger.info(f"encode :{tokens}")

    decoded_str = tokenizer.decode(tokens)
    logger.info(f"decode: {decoded_str}")