"""
Fine-tuning script for Qwen3-8B on KG reasoning tasks.

This script adapts the GCR fine-tuning approach for Qwen3-8B, training the model
to generate reasoning paths in knowledge graphs.

Key Qwen3-specific adaptations:
- Uses Qwen3 chat template with enable_thinking=False for structured output
- Adds AGC-Agent special tokens (<REL>, </REL>, <ENT>, </ENT>, <PATH>, </PATH>)
- Proper handling of Qwen3's tokenizer and vocabulary

Usage:
    accelerate launch --config_file accelerate_configs/deepspeed_zero3.yaml \
        GCR_FT/finetune_qwen.py --data_path_list data/shortest_path_index/RoG-webqsp/train \
        --model_name_or_path Qwen/Qwen3-8B --output_dir save_models/FT-Qwen3-8B
"""

import sys
import os
from dataclasses import dataclass, field
from typing import Optional, List

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
)
import logging
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig
import datasets

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils.gcr_utils as gcr_utils
import utils.utils as utils

datasets.disable_progress_bar()
import dotenv
from accelerate import Accelerator

dotenv.load_dotenv()

# Special tokens for AGC-Agent
PATH_START_TOKEN = "<PATH>"
PATH_END_TOKEN = "</PATH>"
REL_START_TOKEN = "<REL>"
REL_END_TOKEN = "</REL>"
ENT_START_TOKEN = "<ENT>"
ENT_END_TOKEN = "</ENT>"

ALL_SPECIAL_TOKENS = [
    PATH_START_TOKEN, PATH_END_TOKEN,
    REL_START_TOKEN, REL_END_TOKEN,
    ENT_START_TOKEN, ENT_END_TOKEN
]

HF_TOKEN = os.getenv("HF_TOKEN")
N_CPUS = (
    int(os.environ["SLURM_CPUS_PER_TASK"]) if "SLURM_CPUS_PER_TASK" in os.environ else 4
)

# Prompt template for KG reasoning (same as GCR)
ZERO_SHOT_PROMPT = """Reasoning path is a sequence of triples in the KG that connects the topic entities in the question to answer entities. Given a question, please generate some reasoning paths in the KG starting from the topic entities to answer the question.

# Question:
{question}
# Topic entities:
{entities}
"""

ANS_TEMPLATE = """# Reasoning Path:
{reasoning_path}
# Answer:
{answer}"""


@dataclass
class ScriptArguments:
    data_path_list: List[str] = field(metadata={"help": "Path to the training data."})
    model_name_or_path: Optional[str] = field(
        default="Qwen/Qwen3-8B", metadata={"help": "the model name"}
    )
    use_peft: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use PEFT or not to train adapters"},
    )
    save_merged: Optional[bool] = field(
        default=False, metadata={"help": "Whether to save merged model"}
    )
    lora_alpha: Optional[float] = field(
        default=16, metadata={"help": "the lora alpha parameter"}
    )
    lora_dropout: Optional[float] = field(
        default=0.05, metadata={"help": "the lora dropout parameter"}
    )
    lora_r: Optional[int] = field(default=8, metadata={"help": "the lora r parameter"})
    n_path_per_sample: int = field(
        default=10, metadata={"help": "Number of paths to sample"}
    )
    load_in_4bit: bool = field(default=False, metadata={"help": "Load model in 4bit"})
    load_in_8bit: bool = field(default=False, metadata={"help": "Load model in 8bit"})
    attn_implementation: Optional[str] = field(
        default="sdpa", metadata={"help": "attn implementation (sdpa for Blackwell GPUs)"})
    disable_thinking: bool = field(
        default=True,
        metadata={"help": "Disable Qwen3 thinking mode for structured output"}
    )


@dataclass
class ScriptTrainingArguments(TrainingArguments):
    output_dir: str = field(
        default="save_models/FT-Qwen3-8B",
        metadata={"help": "The output directory"},
    )
    optim: str = field(default="adamw_torch")
    # Note: TRL 0.25.1 uses max_length instead of max_seq_length
    max_length: int = field(
        default=2048,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    ddp_find_unused_parameters: bool = field(default=False)
    dataloader_num_workers: int = field(default=N_CPUS)


def train():
    parser = HfArgumentParser((ScriptArguments, ScriptTrainingArguments))
    script_args, training_args = parser.parse_args_into_dataclasses()

    # IMPORTANT: Load tokenizer FIRST and add special tokens BEFORE loading model
    # This ensures all DDP ranks have the same vocabulary size
    print(f"Loading tokenizer from {script_args.model_name_or_path}...")

    # Load Qwen3 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        script_args.model_name_or_path,
        trust_remote_code=True,
        use_fast=True,  # Qwen3 has a fast tokenizer
        token=HF_TOKEN,
    )

    # Qwen3 uses eos_token as pad_token by default
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training

    # Add special tokens for AGC-Agent BEFORE loading model
    existing_special = tokenizer.additional_special_tokens or []
    tokens_to_add = [t for t in ALL_SPECIAL_TOKENS if t not in existing_special]
    num_added = 0

    if tokens_to_add:
        special_tokens_dict = {'additional_special_tokens': ALL_SPECIAL_TOKENS}
        num_added = tokenizer.add_special_tokens(special_tokens_dict)
        if num_added > 0:
            print(f"Added {num_added} special tokens: {tokens_to_add}")

    # Get final vocab size - this will be consistent across all ranks
    final_vocab_size = len(tokenizer)
    print(f"Tokenizer vocab size: {final_vocab_size}")

    # Now load the model
    print(f"Loading Qwen3 model from {script_args.model_name_or_path}...")

    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        trust_remote_code=True,
        token=HF_TOKEN,
        torch_dtype=torch.bfloat16,
        attn_implementation=script_args.attn_implementation,
        load_in_4bit=script_args.load_in_4bit,
        load_in_8bit=script_args.load_in_8bit,
    )

    model.config.use_cache = False

    # Resize embeddings to match tokenizer (same size on all ranks)
    if num_added > 0:
        model.resize_token_embeddings(final_vocab_size)
        print(f"Resized model embeddings to {final_vocab_size}")

    # PEFT configuration for LoRA
    if script_args.use_peft:
        # Qwen3 uses different attention module names
        peft_config = LoraConfig(
            r=script_args.lora_r,
            lora_alpha=script_args.lora_alpha,
            lora_dropout=script_args.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Qwen3 attention modules
            bias="none",
            task_type="CAUSAL_LM",
        )
    else:
        peft_config = None

    # Load datasets
    data_list = [
        datasets.load_from_disk(data_path) for data_path in script_args.data_path_list
    ]
    dataset = datasets.concatenate_datasets(data_list)
    print(f"Loaded {len(dataset)} training samples")

    def input_formatter(example):
        """Format input for Qwen3 fine-tuning."""
        chunks = []
        for i in range(len(example["q_entity"])):
            question = example["question"][i]
            start_node = example["q_entity"][i]
            ground_paths = example["ground_truth_paths"][i]

            if not question.endswith("?"):
                question += "?"

            raw_input = ZERO_SHOT_PROMPT.format(
                question=question, entities=",".join(start_node)
            )

            # Split ground paths into multiple samples
            if len(ground_paths) > 0:
                for path in ground_paths:
                    if len(path) == 0:
                        continue

                    ground_path_string = f"{PATH_START_TOKEN}{utils.path_to_string(path)}{PATH_END_TOKEN}"
                    # The last entity in the path is always the answer
                    path_answer = path[-1][-1].strip()

                    response = ANS_TEMPLATE.format(
                        reasoning_path=ground_path_string, answer=path_answer
                    )

                    chat = [
                        {"role": "user", "content": raw_input},
                        {"role": "assistant", "content": response},
                    ]

                    # Apply Qwen3 chat template with thinking mode disabled
                    try:
                        final_input = tokenizer.apply_chat_template(
                            chat,
                            tokenize=False,
                            add_generation_prompt=False,
                            enable_thinking=False  # Disable thinking mode for structured output
                        )
                    except TypeError:
                        # Fallback if enable_thinking is not supported
                        final_input = tokenizer.apply_chat_template(
                            chat,
                            tokenize=False,
                            add_generation_prompt=False
                        )
                    except Exception as e:
                        print(f"Error applying chat template: {e}")
                        continue

                    chunks.append(final_input)

        return {"text": chunks}

    train_dataset = dataset.map(
        input_formatter,
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=N_CPUS,
    )

    print(f"Training dataset size after formatting: {len(train_dataset)}")
    if len(train_dataset) > 0:
        print(f"Sample input:\n{train_dataset[0]['text'][:500]}...")

    # Prepare instruct tuning with completion-only loss
    # TRL 0.25.1 uses completion_only_loss in SFTConfig instead of DataCollatorForCompletionOnlyLM
    print(f"Using completion-only loss (TRL 0.25.1+ API)")

    sft_cfg = SFTConfig(
        **training_args.to_dict(),
        dataset_text_field="text",
        packing=False,
        dataset_kwargs={"add_special_tokens": False},
        # Use completion_only_loss to only train on assistant responses
        completion_only_loss=True,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        peft_config=peft_config,
        processing_class=tokenizer,  # TRL 0.25.1 uses processing_class instead of tokenizer
        args=sft_cfg,
    )

    # Detecting last checkpoint
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif (
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            logging.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    print("Starting training...")
    trainer.train(resume_from_checkpoint=checkpoint)

    print(f"Saving model to {training_args.output_dir}...")
    trainer.save_model(training_args.output_dir)

    # Also save tokenizer with special tokens
    tokenizer.save_pretrained(training_args.output_dir)
    print("Training completed!")


if __name__ == "__main__":
    train()
