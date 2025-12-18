#!/bin/bash
# Build training data for the QC-Agent Critic
# This script generates positive/negative path examples from KGQA datasets

DATA_PATH=rmanluo
OUTPUT_PATH=data/critic_training

# Build training data for both datasets
for DATASET in RoG-webqsp RoG-cwq; do
    echo "Building training data for ${DATASET}..."

    # Training set
    python Critic/data/critic_data_builder.py \
        --data_path ${DATA_PATH} \
        --dataset ${DATASET} \
        --split train \
        --output_path ${OUTPUT_PATH} \
        --neg_ratio 3.0 \
        --max_path_length 3 \
        --seed 42

    # Validation set (small subset of test for validation)
    python Critic/data/critic_data_builder.py \
        --data_path ${DATA_PATH} \
        --dataset ${DATASET} \
        --split "test[:500]" \
        --output_path ${OUTPUT_PATH} \
        --neg_ratio 3.0 \
        --max_path_length 3 \
        --seed 42
done

echo "Done building critic training data!"
