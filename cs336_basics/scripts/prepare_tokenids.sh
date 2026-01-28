#!/bin/bash

# Configuration
DATASET="owt_train"
INPUT_FILE="data/${DATASET}.txt"
VOCAB_DIR="data/vocab"
VOCAB_FILE="${VOCAB_DIR}/${DATASET}_vocab.pkl"
MERGES_FILE="${VOCAB_DIR}/${DATASET}_merges.pkl"
VOCAB_SIZE=10000
SPECIAL_TOKENS="<|endoftext|>"
TOKEN_IDS_FILE="data/tokenids/${DATASET}_tokenids.npy"
NUM_WORKERS=$(($(nproc) / 2))

echo "prepare tokenids for ${DATASET}"
echo "Input: $INPUT_FILE"
echo "Vocab: $VOCAB_FILE"
echo "Merges: $MERGES_FILE"
echo "TokenIDs: $TOKEN_IDS_FILE"

# Train BPE
echo "Running train_bpe.py..."
# skip if vocab file exists
if [ -f "$VOCAB_FILE" ]; then
    echo "Vocab file $VOCAB_FILE already exists. Skipping BPE training."
else
    uv run cs336_basics/scripts/train_bpe.py \
        --input_path "$INPUT_FILE" \
        --vocab_path "$VOCAB_FILE" \
        --merges_path "$MERGES_FILE" \
        --vocab_size "$VOCAB_SIZE" \
        --special_tokens "$SPECIAL_TOKENS"

    if [ $? -ne 0 ]; then
        echo "Error: train_bpe.py failed"
        exit 1
    fi
fi

# Generate Token IDs
echo "generate tokenids for ${DATASET}"
# skip if tokenids file exists
if [ -f "$TOKEN_IDS_FILE" ]; then
    echo "TokenIDs file $TOKEN_IDS_FILE already exists. Skipping tokenids generation."
else
    uv run cs336_basics/scripts/generate_tokenids.py \
        --input_path "$INPUT_FILE" \
        --vocab_path "$VOCAB_FILE" \
        --merges_path "$MERGES_FILE" \
        --output_path "$TOKEN_IDS_FILE" \
        --special_tokens "$SPECIAL_TOKENS" \
        --num_workers "$NUM_WORKERS"

    if [ $? -ne 0 ]; then
        echo "Error: generate_tokenids.py failed"
        exit 1
    fi
fi

# Optional: Run Training (Example)
# echo "Running training..."
# uv run cs336_basics/scripts/train_llm.py \
#     dataset.dataset_path="$TOKEN_IDS_FILE" \
#     tokenizer.vocab_path="$VOCAB_FILE" \
#     tokenizer.merges_path="$MERGES_FILE"

echo "finish prepare tokenids for ${DATASET}"
