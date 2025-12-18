#!/bin/bash
# Train QC-Agent Critic Model with Multi-GPU Support
# This script trains the critic to evaluate reasoning paths
#
# Prerequisites:
# 1. Build training data first: bash Critic/scripts/build_critic_data.sh
# 2. Ensure you have enough GPU memory (8B model requires ~16GB+ per GPU)
#
# Usage:
#   Single GPU:  bash Critic/scripts/train_critic.sh
#   Multi-GPU:   NUM_GPUS=3 bash Critic/scripts/train_critic.sh

# Model configuration
MODEL_PATH=rmanluo/GCR-Meta-Llama-3.1-8B-Instruct
HIDDEN_DIM=512
ATTN_IMP=sdpa
DTYPE=bf16

# Data paths
TRAIN_DATA=data/critic_training/RoG-webqsp/train
# Disable validation to speed up training (set to empty to skip)
VAL_DATA=""

# Training hyperparameters
BATCH_SIZE=8
NUM_EPOCHS=3
LEARNING_RATE=1e-4
GRADIENT_ACCUMULATION=4
WARMUP_RATIO=0.1

# Loss configuration
USE_RANKING_LOSS="--use_ranking_loss"
RANKING_LOSS_WEIGHT=0.5
RANKING_MARGIN=0.5

# Output - save under Critic directory
OUTPUT_DIR=Critic/trained_models/critic-llama3.1-8b
LOG_INTERVAL=50
# Set very high to effectively disable intermediate eval/save (only final model saved)
EVAL_INTERVAL=999999
SAVE_INTERVAL=999999

# Optional: Enable wandb logging
# WANDB_ARGS="--use_wandb --wandb_project qc-agent-critic"
WANDB_ARGS=""

# GPU configuration
NUM_GPUS="${NUM_GPUS:-3}"  # Default to 3 GPUs

echo "=========================================="
echo "Training QC-Agent Critic"
echo "=========================================="
echo "Model: ${MODEL_PATH}"
echo "Train data: ${TRAIN_DATA}"
echo "Output: ${OUTPUT_DIR}"
echo "Number of GPUs: ${NUM_GPUS}"
echo "=========================================="

# Check if training data exists
if [ ! -d "${TRAIN_DATA}" ]; then
    echo "Error: Training data not found at ${TRAIN_DATA}"
    echo "Please run: bash Critic/scripts/build_critic_data.sh first"
    exit 1
fi

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Training command
TRAIN_CMD="Critic/train_critic.py \
    --model_path ${MODEL_PATH} \
    --hidden_dim ${HIDDEN_DIM} \
    --dtype ${DTYPE} \
    --attn_implementation ${ATTN_IMP} \
    --train_data ${TRAIN_DATA} \
    --batch_size ${BATCH_SIZE} \
    --num_epochs ${NUM_EPOCHS} \
    --learning_rate ${LEARNING_RATE} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION} \
    --warmup_ratio ${WARMUP_RATIO} \
    ${USE_RANKING_LOSS} \
    --ranking_loss_weight ${RANKING_LOSS_WEIGHT} \
    --ranking_margin ${RANKING_MARGIN} \
    --output_dir ${OUTPUT_DIR} \
    --log_interval ${LOG_INTERVAL} \
    --eval_interval ${EVAL_INTERVAL} \
    --save_interval ${SAVE_INTERVAL} \
    ${WANDB_ARGS}"

# Add validation data if specified and directory exists
if [ -n "${VAL_DATA}" ] && [ -d "${VAL_DATA}" ]; then
    TRAIN_CMD="${TRAIN_CMD} --val_data ${VAL_DATA}"
fi

if [ "${NUM_GPUS}" -gt 1 ]; then
    echo "Running with ${NUM_GPUS} GPUs using accelerate..."
    accelerate launch \
        --num_processes ${NUM_GPUS} \
        --mixed_precision bf16 \
        ${TRAIN_CMD}
else
    echo "Running with single GPU..."
    CUDA_VISIBLE_DEVICES=0 python ${TRAIN_CMD}
fi

echo "=========================================="
echo "Training complete!"
echo "Model saved to: ${OUTPUT_DIR}"
echo "=========================================="
