DATA_PATH=rmanluo
# DATA_LIST="RoG-webqsp RoG-cwq"
DATA_LIST="RoG-webqsp"

SPLIT="test[:100]"
INDEX_LEN=2
# ATTN_IMP=flash_attention_2
ATTN_IMP=sdpa


DTYPE=bf16
# DTYPE=fp16

# MODEL_PATH=save_models/FT-Qwen3-8B
MODEL_PATH=rmanluo/GCR-Meta-Llama-3.1-8B-Instruct
MODEL_NAME=$(basename "$MODEL_PATH")

K="10" # 3 5 10 20
for DATA in ${DATA_LIST}; do
  for k in $K; do
    python AA_Trie_Reasoning/reasoning_trie.py --data_path ${DATA_PATH} --d ${DATA} --split ${SPLIT} --index_path_length ${INDEX_LEN} --model_name ${MODEL_NAME} --model_path ${MODEL_PATH} --k ${k} --prompt_mode zero-shot --generation_mode beam --attn_implementation ${ATTN_IMP} --dtype ${DTYPE}
  done
done
