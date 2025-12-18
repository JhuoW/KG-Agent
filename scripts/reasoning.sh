#!/bin/bash

# Reasoning script for QC-Agent with Multi-GPU Support
# This script runs inference using the trained QC-Agent
# Automatically uses all available GPUs for parallel processing

set -e

# =============================================================================
# Configuration
# =============================================================================

# Model configuration
MODEL_PATH="save_models/FT-Qwen3-8B"
CRITIC_CHECKPOINT="save_models/$(basename "$MODEL_PATH")_qc_agent_critic"

# Dataset configuration
DATASET="RoG-cwq"      # Options: RoG-webqsp, RoG-cwq
SPLIT="test[:100]"              # Full test set (use "test[:100]" for quick testing)

# QC-Agent parameters (optimized for accuracy)
BEAM_WIDTH=10             # Number of paths to maintain in beam
MAX_CANDIDATES=5          # Top-K actions from LLM per step (reduced for precision)
MAX_DEPTH=2               # Maximum reasoning hops
HIDDEN_DIM=256            # Hidden dimension for critic

# Self-correction parameters (for avoiding bad paths)
STOP_THRESHOLD=0.5        # Threshold for STOP decision
SCORE_DROP_THRESHOLD=0.3  # Score drop that triggers self-correction

# Multi-GPU configuration
NUM_GPUS=0                # 0 = use all available GPUs

# Output directory
OUTPUT_DIR="results/GenPaths"

# =============================================================================
# Run QC-Agent Reasoning
# =============================================================================

echo "=============================================="
echo "QC-Agent Reasoning Pipeline"
echo "=============================================="
echo "Model: $MODEL_PATH"
echo "Dataset: $DATASET"
echo "Split: $SPLIT"
echo "Available GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"
echo "=============================================="

python reasoning.py \
    --mode qc_agent \
    --dataset $DATASET \
    --split "$SPLIT" \
    --model_path $MODEL_PATH \
    --critic_checkpoint $CRITIC_CHECKPOINT \
    --predict_path $OUTPUT_DIR \
    --beam_width $BEAM_WIDTH \
    --max_candidates $MAX_CANDIDATES \
    --max_depth $MAX_DEPTH \
    --hidden_dim $HIDDEN_DIM \
    --stop_threshold $STOP_THRESHOLD \
    --score_drop_threshold $SCORE_DROP_THRESHOLD \
    --num_gpus $NUM_GPUS \
    --aggregate_answers \
    --enable_self_correction \
    --validate_answers

echo ""
echo "=============================================="
echo "Reasoning complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "=============================================="



  python reasoning_qc_agent.py \
      --model_path save_models/FT-Qwen3-8B \
      --critic_checkpoint save_models/FT-Qwen3-8B_qc_agent_critic/critic.pt \
      --d RoG-webqsp \
      --split test \
      --beam_width 10 \
      --max_depth 2 \
      --aggregate