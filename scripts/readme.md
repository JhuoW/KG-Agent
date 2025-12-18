# Scripts Usage

This folder contains two runnable scripts:

- `preprocessing.sh`: construct train/test data by adding **ground-truth shortest reasoning paths** to each question sample.
- `finetuning.sh`: fine-tune LLMs using the **ground-truth training reasoning paths** produced by preprocessing.

> Important: run these scripts from the **repo root** (so relative paths like `build_gt_path.py` work).

---

## 0) Environment

### Requirements (minimal)

- Python 3.10+
- PyTorch + CUDA (for training)
- Hugging Face: `datasets`, `transformers`
- Training: `accelerate`, `trl`, `peft`
- Optional: `deepspeed` (if you use the deepspeed config), `flash-attn` (if you use `flash_attention_2`)

Example installation (adjust to your CUDA/PyTorch setup):

```bash
pip install -U datasets transformers accelerate trl peft tqdm python-dotenv wandb
# Optional (depends on your system):
# pip install deepspeed
# pip install flash-attn --no-build-isolation
```

### (Optional) Tokens / logging

`finetune.py` loads `HF_TOKEN` from environment (via `.env`):

```bash
export HF_TOKEN=xxxxxxxxxxxxxxxx
```

If you do not want to use Weights & Biases, edit `scripts/finetuning.sh` and change `--report_to "wandb"` to `--report_to "none"`.

---

## 1) Preprocessing: build ground-truth reasoning paths

Goal: download the RoG datasets from Hugging Face and, for each sample, compute **all shortest paths** from topic entities (`q_entity`) to answer entities (`a_entity`) in the provided KG subgraph, then save to disk.

This script calls `build_gt_path.py`.

### 1.1 Run (train split)

From repo root:

```bash
bash scripts/preprocessing.sh
```

By default, it writes to:

- `data/shortest_gt_paths/RoG-webqsp/train`
- `data/shortest_gt_paths/RoG-cwq/train`

Each example will contain a new field: `ground_truth_paths`.

### 1.2 Build test (or other) split

`build_gt_path.py` supports any Hugging Face `split` string (e.g. `test`, `validation`, `test[:100]`).

Example to build test split (also from repo root):

```bash
python build_gt_path.py \
	--data_path rmanluo \
	--split test \
	--output_path data/shortest_gt_paths \
	--num_processes 8
```

It will save to:

- `data/shortest_gt_paths/RoG-webqsp/test`
- `data/shortest_gt_paths/RoG-cwq/test`

### 1.3 Undirected graph option

If you want to treat the KG as undirected when searching shortest paths, use `--undirected`.

Example:

```bash
python build_gt_path.py \
	--data_path rmanluo \
	--split train \
	--output_path data/shortest_gt_paths \
	--undirected \
	--num_processes 8
```

Notes:

- `scripts/preprocessing.sh` currently hard-codes some arguments (like `--dataset`), but `build_gt_path.py` will still process **both** `RoG-webqsp` and `RoG-cwq`.
- Path construction can be CPU-heavy; increase/decrease `--num_processes` based on your machine.

---

## 2) Fine-tuning: train LLM with ground-truth paths

Goal: fine-tune an instruction-tuned causal LLM to generate:

- a reasoning path (wrapped by `<PATH> ... </PATH>`), and
- the final answer

using the `ground_truth_paths` produced by preprocessing.

### 2.1 Expected input data

`finetune.py` currently loads training data from these fixed locations:

- `data/shortest_gt_paths/RoG-webqsp/train`
- `data/shortest_gt_paths/RoG-cwq/train`

So make sure preprocessing has produced those folders.

### 2.2 Run fine-tuning

From repo root:

```bash
bash scripts/finetuning.sh
```

This uses `accelerate launch` to run `finetune.py`, and saves a model to:

- `save_models/FT-$(basename MODEL_PATH)`

Example default:

- `MODEL_PATH=Qwen/Qwen2-1.5B-Instruct`
- output: `save_models/FT-Qwen2-1.5B-Instruct`

### 2.3 Common edits

Open `scripts/finetuning.sh` and adjust:

- `MODEL_PATH`: base model (e.g. Qwen or Llama)
- `CONFIG`: accelerate config file
  - `accelerate_configs/multi_gpu.yaml` (multi-GPU)
  - `accelerate_configs/deepspeed_zero3.yaml` (ZeRO-3)
- `ATTN_IMP`: attention implementation (e.g. `flash_attention_2`)
- `RESPONSE_TEMPLATE`: depends on the chat template of the base model
  - Qwen2: `<|im_start|>assistant`
  - Llama 3.1 Instruct: `<|start_header_id|>assistant<|end_header_id|>`
- `BATCH_SIZE`, `GRADIENT_ACCUMULATION_STEPS`, `EPOCH`: training throughput & schedule
- `USE_PEFT`: set `True` to enable LoRA (PEFT)

---

## Troubleshooting

- **Out of memory (OOM)**: reduce `BATCH_SIZE`, increase `GRADIENT_ACCUMULATION_STEPS`, or switch to the deepspeed config.
- **FlashAttention not available**: set `ATTN_IMP` to a supported value for your environment or install `flash-attn`.
- **HF gated model / auth error**: set `HF_TOKEN` (and ensure you have access to the model).
