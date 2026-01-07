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
# DATA_LIST="RoG-cwq"
# DATA_LIST="RoG-webqsp RoG-cwq"

SPLIT="test[:100]"
# SPLIT="test"
# INDEX_LEN=2 
INDEX_LEN=2

# Attention implementation
# ATTN_IMP=flash_attention_2
ATTN_IMP=sdpa

DTYPE=bf16

# Model path (same as GCR)
# MODEL_PATH=rmanluo/GCR-Meta-Llama-3.1-8B-Instruct   # must be run under `llama3-1-8B` conda environment
MODEL_PATH=save_models/FT-Qwen2.5-7B-Instruct
MODEL_NAME=$(basename "$MODEL_PATH")

# GPU configuration
# Single GPU: GPU_ID="0"
# Multi-GPU: GPU_ID="0,1,2"
GPU_ID="${GPU_ID:-0,1,2}"
# GPU_ID="${GPU_ID:-1,2}"


# K: Number of paths to generate (same as GCR)
# K="10"
K="10"

# AGC-Agent Hyperparameter Settings
# Speed test results on test[:50] samples:
#   | Config           | BEAM | REL | ENT | Time  | Accuracy | Speedup |
#   |------------------|------|-----|-----|-------|----------|---------|
#   | baseline         | 10   | 3   | 3   | 1105s | 65.8%    | 1.0x    |
#   | reduced_beam     | 8    | 3   | 3   | 1053s | 64.1%    | 1.05x   |
#   | reduced_rel      | 10   | 2   | 3   | 842s  | 63.3%    | 1.31x   |
#   | reduced_ent      | 10   | 3   | 2   | 922s  | 61.7%    | 1.20x   |
#   | reduced_beam_rel | 8    | 2   | 3   | 726s  | 63.3%    | 1.52x   |
#   | half_beam        | 5    | 3   | 3   | 571s  | 58.3%    | 1.94x   |

# DEFAULT: Maximum accuracy settings
BEAM_WIDTH=10
RELATION_TOP_K=5
ENTITY_TOP_K=5

# SPEED-OPTIMIZED: Best speed/accuracy trade-off (1.52x speedup, ~2.5% accuracy drop)
# BEAM_WIDTH=8
# RELATION_TOP_K=2
# ENTITY_TOP_K=3

# FAST: Aggressive speed (1.94x speedup, ~7.5% accuracy drop)
# BEAM_WIDTH=5
# RELATION_TOP_K=3
# ENTITY_TOP_K=3

# Filter Freebase MID answers: set to "true" to filter out invalid MID answers (m.xxx, g.xxx)
# FILTER_MID="${FILTER_MID:-false}"
FILTER_MID="${FILTER_MID:-true}"

for DATA in ${DATA_LIST}; do
  for k in $K; do
    # Build command
    CMD="python agc_reasoning2.py \
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
      --gpu_id ${GPU_ID}"

    # Add --filter_mid flag if enabled
    if [ "${FILTER_MID}" = "true" ]; then
      CMD="${CMD} --filter_mid"
    fi

    # Execute
    eval ${CMD}
  done
done



# Accuracy: 71.38065704763903 
# Hit: 86.0 
# F1: 37.12387608570413 
# Precision: 37.46984126984127 
# Recall: 50.41822323887484 
# Path F1: 40.31173107786974 
# Path Precision: 38.7281746031746 
# Path Recall: 63.51793798624214 
# Path Answer F1: 46.235640608590344 
# Path Answer Precision: 45.6281746031746 
# Path Answer Recall: 71.40843482541682


# Accuracy: 70.54862547055663 
# Hit: 84.0 
# F1: 37.86576877104352 
# Precision: 38.4234126984127 
# Recall: 50.38267959354276 
# Path F1: 40.756643339714884 
# Path Precision: 39.38174603174603 
# Path Recall: 62.74546614135651 
# Path Answer F1: 47.20880828159156 
# Path Answer Precision: 46.58174603174602 
# Path Answer Recall: 70.57640324833442


# Filter-mid=False
# Accuracy: 70.79284342756736 
# Hit: 85.0 
# F1: 37.54908597486585 
# Precision: 37.96984126984127 
# Recall: 50.60818739658093 
# Path F1: 40.63741512281303 
# Path Precision: 39.12817460317461 
# Path Recall: 63.99066525896942 
# Path Answer F1: 46.69437833351049 
# Path Answer Precision: 46.1281746031746 
# Path Answer Recall: 70.82062120534513


# FILTER_MID=True
# Accuracy: 69.4811785406839 
# Hit: 85.0 
# F1: 43.42058277654262 
# Precision: 43.2281746031746 
# Recall: 67.87846706032408 
# Path F1: 40.47588387485401 
# Path Precision: 39.8281746031746 
# Path Recall: 61.89906606868602 
# Path Answer F1: 46.444646222313125 
# Path Answer Precision: 46.6281746031746 
# Path Answer Recall: 69.50895631846167


# Accuracy: 71.16383969980858 
# Hit: 86.0 
# F1: 43.43734347194172 
# Precision: 41.3281746031746 
# Recall: 69.534376406445 
# Path F1: 39.83591415120251 
# Path Precision: 38.428174603174604 
# Path Recall: 62.369309841516 
# Path Answer F1: 46.01096202888759 
# Path Answer Precision: 44.7281746031746 
# Path Answer Recall: 71.19161747758636
