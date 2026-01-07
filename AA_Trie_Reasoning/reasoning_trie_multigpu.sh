#!/bin/bash
# Multi-GPU version of graph_constrained_decoding.sh
# Usage: bash scripts/graph_constrained_decoding_multigpu.sh
#
# To use multiple GPUs, set GPU_ID to comma-separated GPU IDs:
#   GPU_ID="0,1,2" for GPUs 0, 1, and 2
#   GPU_ID="0" for single GPU (default)

DATA_PATH=rmanluo
# DATA_LIST="RoG-webqsp RoG-cwq"
# DATA_LIST="RoG-webqsp"
DATA_LIST="RoG-cwq"

SPLIT="test[:100]"
# SPLIT="test"
INDEX_LEN=2
# ATTN_IMP=flash_attention_2
ATTN_IMP=sdpa

DTYPE=bf16
# DTYPE=fp16

# MODEL_PATH=save_models/FT-Qwen3-8B
MODEL_PATH=rmanluo/GCR-Meta-Llama-3.1-8B-Instruct
MODEL_NAME=$(basename "$MODEL_PATH")

# GPU configuration
# Single GPU: GPU_ID="0"
# Multi-GPU: GPU_ID="0,1,2"
# GPU_ID="${GPU_ID:-0}"
GPU_ID="${GPU_ID:-0,1,2}"  # Default to GPUs 0,1,2 if not set

K="10" # 3 5 10 20
for DATA in ${DATA_LIST}; do
  for k in $K; do
    python AA_Trie_Reasoning/reasoning_trie_multigpu.py \
      --data_path ${DATA_PATH} \
      --d ${DATA} \
      --split ${SPLIT} \
      --index_path_length ${INDEX_LEN} \
      --model_name ${MODEL_NAME} \
      --model_path ${MODEL_PATH} \
      --k ${k} \
      --prompt_mode zero-shot \
      --generation_mode beam \
      --attn_implementation ${ATTN_IMP} \
      --dtype ${DTYPE} \
      --gpu_id ${GPU_ID}
  done
done
