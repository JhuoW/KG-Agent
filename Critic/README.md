# Critic Model for QC-Agent

This module implements the **Critic** component for the Question-Conditioned Agentic Graph Reasoner (QC-Agent). The Critic evaluates reasoning paths by computing a value function V(P_t, q) that predicts whether a partial path can lead to a correct answer.

## Directory Structure

```
Critic/
├── __init__.py
├── README.md                    # This file
├── train_critic.py              # Main training script (supports multi-GPU)
├── models/
│   ├── __init__.py
│   └── critic.py                # CriticModel and ActionScoringHead classes
├── data/
│   ├── __init__.py
│   └── critic_data_builder.py   # Training data generation
├── configs/
│   └── accelerate_multi_gpu.yaml  # Multi-GPU config for accelerate
├── scripts/
│   ├── build_critic_data.sh     # Script to build training data
│   └── train_critic.sh          # Script to train the critic (multi-GPU)
└── trained_models/              # Saved model checkpoints (created during training)
```

## Key Components

### 1. CriticModel (`models/critic.py`)

The core critic model that evaluates reasoning paths.

**Architecture:**
- **Frozen LLM Backbone**: Uses `rmanluo/GCR-Meta-Llama-3.1-8B-Instruct` as the base model
- **Special Tokens**: `[Q_START]`, `[Q_END]`, `[PATH_START]`, `[PATH_END]` for marking boundaries
- **Trainable Projection**: `W_z: d_llm → d_z` with LayerNorm
- **Value Head**: `w_v: d_z → 1` for computing the value function

**Input Format:**
```
[Q_START]
{question_text}
[Q_END]

[PATH_START]
{linearized path: entity1 -> relation1 -> entity2 -> ...}
[PATH_END]
```

**Value Function:**
```
z_t = LayerNorm(W_z · h_{L_t})
V(P_t, q) = σ(w_v^T · z_t) ∈ (0, 1)
```

Where `h_{L_t}` is the last hidden state from the frozen LLM.

**Key Methods:**
- `encode(questions, paths)`: Get path embeddings z_t
- `compute_value(questions, paths)`: Compute V(P_t, q) values
- `compute_ranking_loss(pos_q, pos_p, neg_q, neg_p)`: Margin ranking loss
- `save_pretrained(path)` / `from_pretrained(path)`: Checkpoint management

### 2. ActionScoringHead (`models/critic.py`)

Optional component for pre-filtering candidate neighbors when the neighbor set is large.

```
Q(P_t, a, q) = w_Q^T · ψ([z_t || e' || r])
```

### 3. CriticDataBuilder (`data/critic_data_builder.py`)

Generates training data with positive and negative path examples.

**Positive Samples (label=1):**
- All prefixes of ground truth paths from `q_entity` to `a_entity`

**Negative Samples (label=0):**
- Random walks from topic entities (40%)
- Relation-corrupted paths (30%)
- Entity-corrupted paths (10%)
- Alternative branches not leading to answers (30%)

### 4. Training Script (`train_critic.py`)

Supports **multi-GPU training** via `accelerate` library.

**Loss Functions:**
- **BCE Loss**: `L_value = -Σ[y·log(V) + (1-y)·log(1-V)]`
- **Optional Ranking Loss**: `L_rank = max(0, γ - V(P+) + V(P-))`

**Features:**
- Multi-GPU distributed training with accelerate
- Mixed precision training (bf16)
- Linear warmup + linear decay scheduler
- Automatic gradient accumulation
- Checkpoint saving to `Critic/trained_models/`

## Usage

### Step 1: Build Training Data

First, generate positive/negative path examples from the KGQA datasets:

```bash
bash Critic/scripts/build_critic_data.sh
```

This will create training data at `data/critic_training/` for both RoG-webqsp and RoG-cwq datasets.

**Custom data generation:**
```bash
python Critic/data/critic_data_builder.py \
    --data_path rmanluo \
    --dataset RoG-webqsp \
    --split train \
    --output_path data/critic_training \
    --neg_ratio 3.0 \
    --max_path_length 3
```

### Step 2: Train the Critic

**Multi-GPU training (recommended - uses 3 GPUs by default):**
```bash
# Default: 3 GPUs
bash Critic/scripts/train_critic.sh

# Specify number of GPUs
NUM_GPUS=4 bash Critic/scripts/train_critic.sh
```

**Single GPU training:**
```bash
NUM_GPUS=1 bash Critic/scripts/train_critic.sh
```

**Manual training with accelerate:**
```bash
accelerate launch --num_processes 3 --mixed_precision bf16 \
    Critic/train_critic.py \
    --model_path rmanluo/GCR-Meta-Llama-3.1-8B-Instruct \
    --train_data data/critic_training/RoG-webqsp/train \
    --output_dir Critic/trained_models/critic-llama3.1-8b \
    --batch_size 8 \
    --num_epochs 3 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps 4 \
    --use_ranking_loss \
    --ranking_loss_weight 0.5
```

### Step 3: Load Trained Critic

```python
from Critic.models.critic import CriticModel

# Load pretrained critic
critic = CriticModel.from_pretrained("Critic/trained_models/critic-llama3.1-8b/best")
critic.to("cuda")

# Evaluate paths
questions = ["what is the name of justin bieber brother?"]
paths = ["Justin Bieber -> people.person.sibling -> Jaxon Bieber"]

values = critic.compute_value(questions, paths)
print(f"Path value: {values[0].item():.4f}")  # Should be high for correct path
```

## Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_path` | `rmanluo/GCR-Meta-Llama-3.1-8B-Instruct` | Pretrained LLM path |
| `--hidden_dim` | 512 | Projection dimension d_z |
| `--use_question_pooling` | False | Concatenate explicit question embedding |
| `--batch_size` | 8 | Training batch size per GPU |
| `--num_epochs` | 3 | Number of training epochs |
| `--learning_rate` | 1e-4 | Learning rate |
| `--warmup_ratio` | 0.1 | Warmup ratio for scheduler |
| `--gradient_accumulation_steps` | 4 | Gradient accumulation |
| `--use_ranking_loss` | False | Enable margin ranking loss |
| `--ranking_loss_weight` | 0.5 | Weight for ranking loss |
| `--ranking_margin` | 0.5 | Margin γ for ranking loss |
| `--output_dir` | `Critic/trained_models` | Output directory |
| `--use_wandb` | False | Enable wandb logging |

## Requirements

- PyTorch >= 2.0
- Transformers >= 4.35
- Accelerate >= 0.24
- Datasets
- tqdm
- (Optional) wandb for logging

## GPU Memory & Training Time

- **Single GPU**: ~16GB VRAM required, ~12+ hours for full training
- **Multi-GPU (3 GPUs)**: ~16GB per GPU, ~4 hours for full training
- **Multi-GPU (4 GPUs)**: ~16GB per GPU, ~3 hours for full training

Training with 3 GPUs provides approximately 3x speedup compared to single GPU training.

## Model Checkpoints

Trained models are saved to `Critic/trained_models/` with the following structure:
```
Critic/trained_models/critic-llama3.1-8b/
├── best/                    # Best model checkpoint (lowest val_loss)
│   ├── critic_weights.pt    # Projection + value head weights
│   ├── tokenizer.json       # Tokenizer with special tokens
│   └── training_state.pt    # Optimizer/scheduler state
├── final/                   # Final model after all epochs
├── epoch_1/                 # Checkpoint after epoch 1
├── epoch_2/                 # Checkpoint after epoch 2
├── epoch_3/                 # Checkpoint after epoch 3
└── args.json                # Training arguments
```
