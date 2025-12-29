#!/bin/bash
# Speed test script for AGC-Agent hyperparameter tuning
# Tests different configurations to find optimal speed/accuracy trade-off

DATA_PATH=rmanluo
DATA="RoG-webqsp"
SPLIT="test[:50]"  # Use smaller subset for quick testing
INDEX_LEN=2
MODEL_PATH=rmanluo/GCR-Meta-Llama-3.1-8B-Instruct
MODEL_NAME=$(basename "$MODEL_PATH")
GPU_ID="${GPU_ID:-0,1,2}"
ATTN_IMP=sdpa
DTYPE=bf16
K=10

# Test configurations: [BEAM_WIDTH, RELATION_TOP_K, ENTITY_TOP_K, description]
declare -a CONFIGS=(
    "10 3 3 baseline"
    "8 3 3 reduced_beam"
    "10 2 3 reduced_rel"
    "10 3 2 reduced_ent"
    "8 2 3 reduced_beam_rel"
    "5 3 3 half_beam"
)

echo "=== AGC-Agent Speed Test ==="
echo "Testing ${#CONFIGS[@]} configurations on $SPLIT"
echo ""

for config in "${CONFIGS[@]}"; do
    read -r BEAM REL ENT DESC <<< "$config"

    echo "=== Testing: $DESC (BEAM=$BEAM, REL=$REL, ENT=$ENT) ==="
    START_TIME=$(date +%s)

    python agc_reasoning2.py \
        --data_path ${DATA_PATH} \
        --d ${DATA} \
        --split ${SPLIT} \
        --index_path_length ${INDEX_LEN} \
        --model_name ${MODEL_NAME} \
        --model_path ${MODEL_PATH} \
        --k ${K} \
        --beam_width ${BEAM} \
        --relation_top_k ${REL} \
        --entity_top_k ${ENT} \
        --generation_mode beam \
        --attn_implementation ${ATTN_IMP} \
        --dtype ${DTYPE} \
        --gpu_id ${GPU_ID} \
        --filter_mid \
        --force

    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))

    echo ""
    echo ">>> $DESC: ${ELAPSED}s"
    echo ""
done

echo "=== Speed Test Complete ==="
