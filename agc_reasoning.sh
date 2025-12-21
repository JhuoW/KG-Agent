#!/bin/bash
# AGC-Agent Reasoning Script
# Aligned with GCR (AA_Trie_Reasoning/reasoning_trie_multigpu.sh) settings
#
# Usage: bash agc_reasoning.sh
#
# To use multiple GPUs, set GPU_ID environment variable:
#   GPU_ID="0,1,2" bash agc_reasoning.sh

DATA_PATH=rmanluo
DATA_LIST="RoG-webqsp"
# DATA_LIST="RoG-webqsp RoG-cwq"

SPLIT="test[:100]"
INDEX_LEN=2  # Same as GCR index_path_length

# Attention implementation
# ATTN_IMP=flash_attention_2
ATTN_IMP=sdpa

DTYPE=bf16

# Model path (same as GCR)
MODEL_PATH=rmanluo/GCR-Meta-Llama-3.1-8B-Instruct
# MODEL_PATH=save_models/FT-Qwen3-8B
MODEL_NAME=$(basename "$MODEL_PATH")

# GPU configuration
# Single GPU: GPU_ID="0"
# Multi-GPU: GPU_ID="0,1,2"
GPU_ID="${GPU_ID:-0,1,2}"

# K: Number of paths to generate (same as GCR)
K="10"

# AGC-Agent specific settings
BEAM_WIDTH=10
RELATION_TOP_K=3
ENTITY_TOP_K=3

for DATA in ${DATA_LIST}; do
  for k in $K; do
    python agc_reasoning.py \
      --data_path ${DATA_PATH} \
      --d ${DATA} \
      --split ${SPLIT} \
      --index_path_length ${INDEX_LEN} \
      --model_name ${MODEL_NAME} \
      --model_path ${MODEL_PATH} \
      --k ${k} \
      --beam_width ${BEAM_WIDTH} \
      --relation_top_k ${RELATION_TOP_K} \
      --entity_top_k ${ENTITY_TOP_K} \
      --generation_mode beam \
      --attn_implementation ${ATTN_IMP} \
      --dtype ${DTYPE} \
      --gpu_id ${GPU_ID}
  done
done
