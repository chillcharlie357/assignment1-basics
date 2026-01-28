#!/bin/bash

echo "prepare tokenids for owt_train"

uv run cs336_basics/scripts/train_bpe.py
if [ $? -ne 0 ]; then
    echo "Error: train_bpe.py failed"
    exit 1
fi

echo "generate tokenids for owt_train"

uv run cs336_basics/scripts/generate_tokenids.py

if [ $? -ne 0 ]; then
    echo "Error: generate_tokenids.py failed"
    exit 1
fi

echo "finish prepare tokenids for owt_train"