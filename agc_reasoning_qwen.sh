#!/bin/bash
# AGC-Agent Reasoning Script for Qwen3-8B
#
# This script runs the AGC-Agent with Qwen3-8B as the backbone LLM
# instead of the GCR-finetuned Llama model.
#
# Usage: bash agc_reasoning_qwen.sh
#
# To use multiple GPUs, set GPU_ID environment variable:
#   GPU_ID="0,1,2" bash agc_reasoning_qwen.sh

DATA_PATH=rmanluo
DATA_LIST="RoG-webqsp"
# DATA_LIST="RoG-webqsp RoG-cwq"

SPLIT="test[:100]"
INDEX_LEN=2  # Same as GCR index_path_length

# Attention implementation
# ATTN_IMP=flash_attention_2
ATTN_IMP=sdpa

DTYPE=bf16

# Qwen3-8B Model path
MODEL_PATH=Qwen/Qwen3-8B
MODEL_NAME=$(basename "$MODEL_PATH")


GPU_ID="${GPU_ID:-0,1,2}"
# GPU_ID="${GPU_ID:-1,2}"

# K: Number of paths to generate
K="10"

# AGC-Agent specific settings
BEAM_WIDTH=10
RELATION_TOP_K=3
ENTITY_TOP_K=3

# Quantization (optional, for memory efficiency)
# QUANT="none"    # Full precision
# QUANT="4bit"    # 4-bit quantization
# QUANT="8bit"    # 8-bit quantization
QUANT="${QUANT:-none}"

for DATA in ${DATA_LIST}; do
  for k in $K; do
    echo "Running Qwen3-8B AGC-Agent on ${DATA} with k=${k}..."

    python agc_reasoning_qwen.py \
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
      --quant ${QUANT} \
      --gpu_id ${GPU_ID}
  done
done

echo "Qwen3-8B AGC-Agent reasoning completed!"
