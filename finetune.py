import sys
import os
from pathlib import Path
import torch
import os.path as osp
import typing
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, TrainingArguments, AutoConfig, BitsAndBytesConfig
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from peft import LoraConfig
import datasets
datasets.disable_progress_bar()
from utils.utils import path_to_string
import dotenv
from trl import SFTTrainer, SFTConfig
from transformers.trainer_utils import get_last_checkpoint

dotenv.load_dotenv()

PATH_START_TOKEN = "<PATH>"
PATH_END_TOKEN = "</PATH>"

HF_TOKEN = os.getenv("HF_TOKEN")

N_CPUS = int(
    os.getenv(
        "SLURM_CPUS_PER_TASK",
        os.getenv("N_CPUS", str(os.cpu_count() or 1))
    )
)

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
    data_path_list: list[str] = field(metadata={"help": "Path to the training data."},
                                      default="data/shortest_path_index/RoG-webqsp/train data/shortest_path_index/RoG-cwq/train")
    # meta-llama/Meta-Llama-3.1-8B-Instruct
    # Qwen/Qwen2-1.5B-Instruct
    model_name_or_path: Optional[str] = field(default="meta-llama/Meta-Llama-3.1-8B-Instruct", metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"})
    use_peft: Optional[bool] = field(default=False, metadata={"help": "Whether to use PEFT for fine-tuning."})
    save_merged: Optional[bool] = field(default=False, metadata={"help": "Wether to save merged model."})
    lora_alpha: Optional[float] = field(default=16.0, metadata={"help": "LoRA alpha parameter."})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "LoRA dropout parameter."})
    lora_r: Optional[int] = field(default=8, metadata={"help": "LoRA r parameter."})
    n_path_per_sample: Optional[int] = field(default=5, metadata={"help": "Number of paths to sample per training example."})
    load_in_4bit: Optional[bool] = field(default=False, metadata={"help": "Whether to load the model in 4-bit precision."})
    load_in_8bit: Optional[bool] = field(default=False, metadata={"help": "Whether to load the model in 8-bit precision."}) 
    attn_impl: Optional[str] = field(default="flash_attention_2", metadata={"help": "Attention implementation to use."})
    # <|start_header_id|>assistant<|end_header_id|>  Meta-Llama-3.1-8B-Instruct
    # <|im_start|>assistant Qwen/Qwen2-1.5B-Instruct
    response_template: Optional[str] = field(default="<|im_start|>assistant", metadata={"help": "Template for the model response. Default is for Qwen"})

@dataclass
class ScriptTrainingArguments(TrainingArguments):
    output_dir: str = field(default="saved_models/finetuned_llm_model", metadata={"help": "The output directory where the model predictions and checkpoints will be written."})
    optim: str = field(default="adamw_torch", metadata={"help": "Optimizer to use."})
    max_length: int = field(default=2048, metadata={"help": "Maximum sequence length."})
    ddp_find_unused_parameters: bool = field(default=False, metadata={"help": "Whether to set ddp_find_unused_parameters to True."})
    dataloader_num_workers: int = field(default=N_CPUS, metadata={"help": "Number of dataloader workers."})

def finetuning():
    parser = HfArgumentParser((ScriptArguments, ScriptTrainingArguments))
    script_args, training_args = parser.parse_args_into_dataclasses()

    # print(training_args.to_dict())

    # Load models and tokenizers
    # model = AutoModelForCausalLM.from_pretrained(script_args.model_name_or_path)
    quantization_config = None
    if script_args.load_in_4bit or script_args.load_in_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=script_args.load_in_4bit,
            load_in_8bit=script_args.load_in_8bit,
        )
    # device_map="auto" 不能和accelerate的分布式训练一起用
    model = AutoModelForCausalLM.from_pretrained(script_args.model_name_or_path, 
                                                 token = HF_TOKEN,
                                                 trust_remote_code = True,
                                                 attn_implementation = script_args.attn_impl,
                                                 dtype = torch.bfloat16,
                                                 quantization_config=quantization_config)
    model.config.use_cache = False
    if script_args.use_peft:
        peft_config = LoraConfig(
            r = script_args.lora_r,
            lora_alpha = script_args.lora_alpha,
            lora_dropout = script_args.lora_dropout,
            target_modules = ["q_proj", "v_proj"],
            bias="none",
            task_type="CAUSAL_LM"
        )
    else:
        peft_config = None
    
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=script_args.model_name_or_path,
                                              trust_remote_code = True,
                                              use_fast=True,
                                              token = HF_TOKEN)
    tokenizer.padding_side = "right"
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = "<PAD>"  # 输入的batch句子要长度一样, 一次处理多各问题
    
    special_tokens_dict['additional_special_tokens'] = [PATH_START_TOKEN, PATH_END_TOKEN]
    tokenizer.add_special_tokens(special_tokens_dict)

    model.resize_token_embeddings(len(tokenizer))
    # Load datasets
    # data/shortest_path_index/RoG-webqsp/train data/shortest_path_index/RoG-cwq/train
    data_path_list = ["data/shortest_gt_paths/RoG-webqsp/train",
                      "data/shortest_gt_paths/RoG-cwq/train"]
    data_list = [datasets.load_from_disk(data_path) for data_path in data_path_list]
    dataset = datasets.concatenate_datasets(data_list)  # 两个数据集中所有的训练条目合并
    print(f"Training question number: {len(dataset)}") 

    def input_formatter(example): # 将每个样本的的内容整理成sentence
        prompts = []
        completions = []
        for i in range(len(example['q_entity'])): #遍历每个样本
            question = example['question'][i]
            start_node = example['q_entity'][i]
            answer_node = example['a_entity'][i]
            gt_paths = example['ground_truth_paths'][i]
            if not question.endswith("?"):
                question += "?"
            raw_input = ZERO_SHOT_PROMPT.format(question=question, entities=",".join(start_node))
            if len(gt_paths) > 0:   # has path from q_entity to a_entity
                for path in gt_paths:
                    if len(path) == 0:
                        continue
                    ground_truth_path_str = f"{PATH_START_TOKEN}{path_to_string(path)}{PATH_END_TOKEN}"
                    path_answer = path[-1][-1].strip()
                    response = ANS_TEMPLATE.format(reasoning_path = ground_truth_path_str,
                                                   answer = path_answer)
                    
                    # chat = [
                    #     {"role": "user", "content": raw_input},  
                    #     {"role": "assistant", "content": response},                        
                    # ]
                    prompts.append([{"role": "user", "content": raw_input}])
                    completions.append([{"role": "assistant", "content": response}])
                    # final_input = tokenizer.apply_chat_template(
                    #     chat, tokenize=False, add_generation_prompt=False
                    # )
                    # chunks.append(final_input)
        return {"prompt": prompts, "completion": completions}

    train_dataset = dataset.map(
        input_formatter,
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=N_CPUS,
    )
    print(train_dataset[0])

    sft_config = SFTConfig(
        **training_args.to_dict(),  # num_train_epochs 3 per_device_train_batch_size 4
        completion_only_loss=True,
        packing=False,
    )

    trainer = SFTTrainer(  # num_train_epochs=3, batch_size=4
        model=model,
        train_dataset=train_dataset,
        peft_config=peft_config,
        args=sft_config,
        processing_class=tokenizer
    )
    last_checkpoint = None
    if (
        osp.isdir(training_args.output_dir) and not training_args.overwrite_output_dir
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
    trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model(training_args.output_dir)

if __name__ == "__main__":
    finetuning()