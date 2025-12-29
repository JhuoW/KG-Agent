#!/bin/bash
# Fine-tuning script for Qwen3-8B on KG reasoning tasks
#
# This script fine-tunes Qwen3-8B to generate reasoning paths in knowledge graphs,
# creating a KG-specialized version similar to GCR-Meta-Llama-3.1-8B-Instruct.
#
# Usage:
#   bash GCR_FT/finetune_qwen.sh
#
# Requirements:
#   - GPU memory: ~80GB for full fine-tuning with DeepSpeed ZeRO-3
#   - For smaller GPUs, use LoRA (set USE_PEFT=True)
#
# Output:
#   - Model saved to save_models/FT-Qwen3-8B

# ============================================================================
# Dataset Configuration
# ============================================================================
# Training data: shortest path reasoning data from WebQSP and CWQ
DATASET_LIST="data/shortest_path_index/RoG-webqsp/train data/shortest_path_index/RoG-cwq/train"

# For quick testing, use only WebQSP:
# DATASET_LIST="data/shortest_path_index/RoG-webqsp/train"

# ============================================================================
# Training Configuration
# ============================================================================

# --- Option 1: Full Fine-tuning (Recommended for best performance) ---
# Requires ~80GB GPU memory with DeepSpeed ZeRO-3
# Using Qwen-specific accelerate configs (separate from Llama configs)
# BATCH_SIZE=4
# USE_PEFT=False
# EPOCH=3
# GRADIENT_CHECKPOINTING=True
# GRADIENT_ACCUMULATION_STEPS=16
# auto_find_batch_size=False
# CONFIG="accelerate_configs_qwen/deepspeed_zero3.yaml"

# --- Option 2: LoRA Fine-tuning (PEFT) ---
# More memory efficient, faster training
BATCH_SIZE=8
USE_PEFT=True
EPOCH=5
GRADIENT_CHECKPOINTING=False
GRADIENT_ACCUMULATION_STEPS=4
auto_find_batch_size=True
CONFIG="accelerate_configs_qwen/multi_gpu.yaml"

# ============================================================================
# Model Configuration
# ============================================================================
MODEL_PATH=Qwen/Qwen3-8B
# Use sdpa (Scaled Dot Product Attention) for Blackwell GPU compatibility
# flash_attention_2 may not be supported on newer GPUs
ATTN_IMP=sdpa

# Note: TRL 0.25.1+ uses completion_only_loss instead of response_template
# The loss is automatically computed only on assistant responses

# ============================================================================
# Output Configuration
# ============================================================================
SAVE_PATH=save_models/FT-Qwen3-8B
SAVE_NAME=$(basename "$SAVE_PATH")

# ============================================================================
# Launch Training
# ============================================================================
echo "=============================================="
echo "Fine-tuning Qwen3-8B for KG Reasoning"
echo "=============================================="
echo "Model: ${MODEL_PATH}"
echo "Output: ${SAVE_PATH}"
echo "Datasets: ${DATASET_LIST}"
echo "Config: ${CONFIG}"
echo "PEFT (LoRA): ${USE_PEFT}"
echo "Epochs: ${EPOCH}"
echo "Batch Size: ${BATCH_SIZE}"
echo "Gradient Accumulation: ${GRADIENT_ACCUMULATION_STEPS}"
echo "=============================================="

accelerate launch --config_file ${CONFIG} GCR_FT/finetune_qwen.py \
    --data_path_list ${DATASET_LIST} \
    --model_name_or_path ${MODEL_PATH} \
    --output_dir ${SAVE_PATH} \
    --use_peft ${USE_PEFT} \
    --bf16 True \
    --num_train_epochs ${EPOCH} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --eval_strategy "no" \
    --save_strategy "epoch" \
    --save_steps 500 \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --tf32 True \
    --report_to "wandb" \
    --gradient_checkpointing ${GRADIENT_CHECKPOINTING} \
    --auto_find_batch_size ${auto_find_batch_size} \
    --neftune_noise_alpha 5 \
    --attn_implementation ${ATTN_IMP} \
    --run_name ${SAVE_NAME}

echo "=============================================="
echo "Fine-tuning completed!"
echo "Model saved to: ${SAVE_PATH}"
echo ""
echo "To use the fine-tuned model, update agc_reasoning_qwen.sh:"
echo "  MODEL_PATH=${SAVE_PATH}"
echo "=============================================="
