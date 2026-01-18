# Byte-Pair Encoding Tokenizer


## 2.5 Experimenting with BPE Tokenizer Training

### Problem (train_bpe): BPE Tokenizer Training

## 2.6 BPE Tokenizer: Encoding and Decoding


### Problem (tokenizer): Implementing the tokenizer

`cs336_basics/tokenizer/tokenizer.py`


## 2.7 Experiments

### Problem (tokenizer_experiments): Experiments with tokenizers

TODO
#### (a)  

#### (b)

#### (c)

#### (d) 词表为什么使用Numpy uint16类型？


- 高效的序列化 (Serialization) : np.save 将数组以紧凑的二进制格式 ( .npy ) 保存，比文本格式（如 JSON 或 CSV）或 Python 的 pickle 更小、更快。
- 内存映射 (Memory Mapping) : 在训练大型模型时，数据集可能大到无法一次性读入内存。NumPy 允许使用 mmap_mode='r' 打开 .npy 文件，将磁盘上的文件映射为内存中的数组， 按需读取 ，而不需要将整个 5M 或更大的数据集全部加载到 RAM 中。
- Numpy可以直接转成Tensor类型

使用uint16是为了适配词表大小，同时节约内存：
- uint8 (0 ~ 255): 太小，无法容纳所有 Token ID。
- uint16 (0 ~ 65,535) : 正好可以容纳 12,800 个 ID，且还有很大余量。
- uint32 / int32 : 每个 ID 占用 4 字节，相比 uint16 浪费了一倍空间。


## 3.4 Basic Building Blocks: Linear and Embedding Modules

### Problem (linear): Implementing the linear module

`cs336_basics/transformer/linear.py`


### Problem (embedding): Implement the embedding module

`cs336_basics/transformer/embedding.py`