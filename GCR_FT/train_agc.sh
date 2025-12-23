#!/bin/bash
# Training script for AGC-Agent fine-tuning
#
# This script fine-tunes meta-llama/Meta-Llama-3.1-8B-Instruct for AGC-Agent
# with special tokens: <PATH>, </PATH>, <REL>, </REL>, <ENT>, </ENT>
#
# Usage:
#   bash GCR_FT/train_agc.sh
#
# Training modes:
#   - all: Train on relation selection, entity selection, and path generation
#   - relation: Train only on relation selection
#   - entity: Train only on entity selection
#   - path: Train only on path generation (backward compatible with GCR)

DATASET_LIST="data/shortest_path_index/RoG-webqsp/train data/shortest_path_index/RoG-cwq/train"

# Training Configuration
BATCH_SIZE=4
USE_PEFT=False
EPOCH=3
GRADIENT_CHECKPOINTING=True
GRADIENT_ACCUMULATION_STEPS=16
auto_find_batch_size=False
CONFIG="accelerate_configs/deepspeed_zero3.yaml"

# Model Configuration
MODEL_PATH=meta-llama/Meta-Llama-3.1-8B-Instruct
ATTN_IMP=flash_attention_2
RESPONSE_TEMPLATE="<|start_header_id|>assistant<|end_header_id|>"

# Output path
SAVE_PATH=save_models/FT-$(basename "$MODEL_PATH")
SAVE_NAME=$(basename "$SAVE_PATH")

# Training mode: all, relation, entity, path
TRAINING_MODE=all

# Negative sampling parameters
MAX_RELATIONS=10
MAX_ENTITIES=20

echo "=============================================="
echo "AGC-Agent Fine-tuning"
echo "=============================================="
echo "Model: ${MODEL_PATH}"
echo "Output: ${SAVE_PATH}"
echo "Training Mode: ${TRAINING_MODE}"
echo "Datasets: ${DATASET_LIST}"
echo "=============================================="

accelerate launch --config_file ${CONFIG} GCR_FT/finetune_agc.py \
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
    --save_strategy "no" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --report_to "wandb" \
    --gradient_checkpointing ${GRADIENT_CHECKPOINTING} \
    --auto_find_batch_size ${auto_find_batch_size} \
    --neftune_noise_alpha 5 \
    --attn_implementation ${ATTN_IMP} \
    --response_template "${RESPONSE_TEMPLATE}" \
    --training_mode ${TRAINING_MODE} \
    --max_relations_per_step ${MAX_RELATIONS} \
    --max_entities_per_step ${MAX_ENTITIES} \
    --run_name ${SAVE_NAME}

echo "=============================================="
echo "Training complete!"
echo "Model saved to: ${SAVE_PATH}"
echo "=============================================="
