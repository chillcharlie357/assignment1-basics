# 项目实现 (Project Implementation)

本项目从零开始实现了一个基于 Transformer 的语言模型（LLM），架构类似于 Llama。实现代码位于 `cs336_basics` 目录下。

### 核心组件 (`cs336_basics/transformer`)

模型架构使用模块化组件构建：

*   **Transformer Language Model (`lm.py`)**: 主类 `Transformer_LM` 结合了 Embeddings、堆叠的 Transformer Blocks 以及最终的 RMSNorm 和 Linear head。
*   **Transformer Block (`transformer.py`)**: 每个 Block 包含：
    *   **RMSNorm (`rmsnorm.py`)**: Pre-normalization。
    *   **Multi-Head Self-Attention (`attention.py`)**: 实现了带有 Causal Masking 和 **RoPE** (Rotary Positional Embeddings) 的 Scaled Dot-Product Attention。
    *   **Feed-Forward Network (`ffn.py`)**: 使用 **SwiGLU** (Sigmoid Linear Unit with Gated Linear Unit) 激活函数。
*   **Embedding (`embedding.py`)**: 标准的 Token Embedding。
*   **Softmax (`softmax.py`)**: 自定义的 Softmax 实现。

### 训练 (`cs336_basics/training`)

训练流程包括：

*   **Optimizer**: **AdamW** optimizer (`adamW.py`) 配合 **Gradient Clipping** (`gradient_clipping.py`)。
*   **Loss Function**: Cross-Entropy Loss 并监控 **Perplexity** (`cross_entropy.py`)。
*   **Learning Rate Schedule**: Cosine Annealing Scheduler (`schedule.py`)。
*   **Checkpoint Management (`checkpoint.py`)**:
    *   保存带有时间戳的 Checkpoint。
    *   自动保留最近的 5 个 Checkpoint。
    *   支持从最新的 Checkpoint 断点续训。

### 脚本 (`cs336_basics/scripts`)

*   **训练脚本 (`train_llm.py`)**:
    *   使用 **Hydra** 进行配置管理 (`conf/`)。
    *   支持在 TinyStories 数据集上进行训练。
    *   执行验证并记录 Loss 和 Perplexity。
    *   基于验证集 Loss 保存最佳模型。
*   **解码脚本 (`decoding.py`)**:
    *   支持使用训练好的模型进行文本生成。
    *   可配置 Sampling 参数（temperature, top_p）。

### 配置 (`conf/`)

本项目使用 [Hydra](https://hydra.cc/) 进行配置管理。关键配置文件包括：

*   `conf/config.yaml`: 主入口配置文件。
*   `conf/model/small.yaml`: Model 架构超参数。
*   `conf/training/default.yaml`: 训练参数（batch size, learning rate 等）。
*   `conf/dataset/tinystories.yaml`: Dataset 路径。

## 环境搭建 (Setup)

### 环境配置
我们使用 `uv` 来管理环境，以确保可复现性、可移植性和易用性。
请在 [此处](https://github.com/astral-sh/uv) 安装 `uv`（推荐），或者运行 `pip install uv`/`brew install uv`。
我们建议阅读有关在 `uv` 中管理项目的文档 [这里](https://docs.astral.sh/uv/guides/projects/#managing-dependencies)（绝对值得一读！）。

现在您可以使用以下命令运行仓库中的任何代码：
```sh
uv run <python_file_path>
```
环境将会在需要时自动解析并激活。

### 下载数据
下载 TinyStories 数据和 OpenWebText 的子集

``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

### 运行单元测试

```sh
uv run pytest
```

最初，所有测试都应该失败并提示 `NotImplementedError`。
为了将您的实现与测试连接起来，请完成 [./tests/adapters.py](./tests/adapters.py) 中的函数。
