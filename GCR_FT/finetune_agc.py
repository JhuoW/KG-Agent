"""
Fine-tuning script for AGC-Agent (Adaptive Graph-Constrained Agentic Reasoning).

This script fine-tunes meta-llama/Meta-Llama-3.1-8B-Instruct for step-wise
constrained reasoning with special tokens:
- <PATH>, </PATH>: for path formatting
- <REL>, </REL>: for relation selection
- <ENT>, </ENT>: for entity selection

The training data is generated from ground-truth paths, creating step-by-step
examples for relation selection, entity selection, and answer generation.
"""

import sys
import os

import torch

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

import torch.utils
import torch.utils.data
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
)
import logging
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, SFTConfig
from peft import LoraConfig
import datasets

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

# =============================================================================
# Prompt Templates for AGC-Agent (from agentic_controller.py)
# =============================================================================

RELATION_SELECTOR_SYSTEM_PROMPT = """You are a Knowledge Graph Reasoning Agent specialized in navigating structured knowledge graphs to answer questions. Your task is to select the most promising relation to follow at each step of the reasoning process.

A knowledge graph consists of entities connected by relations, forming triples: (head_entity, relation, tail_entity). For example:
- (Barack Obama, spouse_of, Michelle Obama) means Barack Obama's spouse is Michelle Obama
- (USA, president, Joe Biden) means the president of USA is Joe Biden

When selecting a relation, you should consider:
1. SEMANTIC RELEVANCE: How well does the relation's meaning align with what the question is asking?
2. PATH PROGRESS: Does this relation move closer to the type of entity the question seeks?
3. REASONING CHAIN: How does this relation connect to the previous reasoning steps?

You must ONLY select from the available relations listed. Any relation not in the list does not exist for the current entity in the knowledge graph.

Output your selected relation between <REL> and </REL> tags, exactly as it appears in the available list."""


RELATION_SELECTOR_USER_TEMPLATE = """# Question:
{question}

# Topic Entities:
{topic_entities}

# Current Reasoning State:
- Path So Far: {path_so_far}
- Current Entity: {current_entity}
- Reasoning Depth: {depth} step(s) taken

# Available Relations from "{current_entity}":
{available_relations}

# Task:
Analyze the question and current reasoning state. Select the single best relation to follow that will make progress toward answering the question.
Consider:
1. Which relation semantically connects to the question's intent?
2. Which relation logically continues the reasoning path?
3. Which relation is likely to reach answer-type entities?

# Selected Relation:
<REL>"""


RELATION_SELECTOR_RESPONSE_TEMPLATE = """{relation}</REL>"""


ENTITY_SELECTOR_SYSTEM_PROMPT = """You are a Knowledge Graph Reasoning Agent. Given a reasoning path and a selected relation, your task is to choose the target entity that is most likely to lead toward answering the question.

You must ONLY select from the available target entities listed. Do NOT invent or suggest entities not in the list.

Output your selected entity between <ENT> and </ENT> tags, exactly as it appears in the available list."""


ENTITY_SELECTOR_USER_TEMPLATE = """# Question:
{question}

# Topic Entities:
{topic_entities}

# Reasoning Path So Far:
{path_so_far}

# Current Step:
From entity "{current_entity}", following relation [{selected_relation}]

# Available Target Entities:
{available_entities}

# Task:
Select the entity that best continues the reasoning toward the answer.
Consider:
1. Which entity's type matches what the question asks for?
2. Which entity is most semantically relevant to the question?
3. If multiple entities seem valid, which is most specific?

# Selected Entity:
<ENT>"""


ENTITY_SELECTOR_RESPONSE_TEMPLATE = """{entity}</ENT>"""


# Combined prompt for path generation (backward compatible with GCR)
PATH_GENERATION_SYSTEM_PROMPT = """You are a Knowledge Graph Reasoning Agent. Given a question and topic entities, generate reasoning paths that connect topic entities to answer entities.

A reasoning path is a sequence of (entity, relation, entity) triples that form a chain from topic entities to answer entities.

Output paths between <PATH> and </PATH> tags."""


PATH_GENERATION_USER_TEMPLATE = """# Question:
{question}

# Topic Entities:
{topic_entities}

# Reasoning Path:
<PATH>"""


PATH_GENERATION_RESPONSE_TEMPLATE = """{path}</PATH>
# Answer:
{answer}"""


@dataclass
class ScriptArguments:
    data_path_list: list[str] = field(metadata={"help": "Path to the training data."})
    model_name_or_path: Optional[str] = field(
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        metadata={"help": "the model name"}
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
        default=10, metadata={"help": "Number of paths to sample per question"}
    )
    load_in_4bit: bool = field(default=False, metadata={"help": "Load model in 4bit"})
    load_in_8bit: bool = field(default=False, metadata={"help": "Load model in 8bit"})
    attn_implementation: Optional[str] = field(
        default="flash_attention_2", metadata={"help": "attn implementation"})
    response_template: Optional[str] = field(
        default="<|start_header_id|>assistant<|end_header_id|>",
        metadata={"help": "Response template for chat format"}
    )
    training_mode: Optional[str] = field(
        default="all",
        metadata={
            "help": "Training mode: 'relation' for relation selection only, "
                    "'entity' for entity selection only, 'path' for path generation only, "
                    "'all' for all tasks combined"
        }
    )
    max_relations_per_step: int = field(
        default=10,
        metadata={"help": "Maximum number of relations to show per step (for negative sampling)"}
    )
    max_entities_per_step: int = field(
        default=20,
        metadata={"help": "Maximum number of entities to show per step (for negative sampling)"}
    )


@dataclass
class ScriptTrainingArguments(TrainingArguments):
    output_dir: str = field(
        default="save_models/FT-Meta-Llama-3.1-8B-Instruct",
        metadata={"help": "The output directory"},
    )
    optim: str = field(default="adamw_torch")
    max_seq_length: int = field(
        default=4096,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    ddp_find_unused_parameters: bool = field(default=False)
    dataloader_num_workers: int = field(default=N_CPUS)


def path_to_string(path: list) -> str:
    """Convert a path (list of triples) to string format."""
    result = ""
    for i, p in enumerate(path):
        if i == 0:
            h, r, t = p
            result += f"{h} -> {r} -> {t}"
        else:
            _, r, t = p
            result += f" -> {r} -> {t}"
    return result.strip()


def format_path_with_tags(path: list) -> str:
    """Format path with PATH tags."""
    return f"{PATH_START_TOKEN}{path_to_string(path)}{PATH_END_TOKEN}"


def build_graph_from_triples(triples: List) -> Dict[str, Dict[str, List[str]]]:
    """Build adjacency structure from triples for negative sampling."""
    graph = {}  # entity -> {relation -> [tail_entities]}
    for h, r, t in triples:
        h, r, t = str(h).strip(), str(r).strip(), str(t).strip()
        if h not in graph:
            graph[h] = {}
        if r not in graph[h]:
            graph[h][r] = []
        graph[h][r].append(t)
    return graph


def get_available_relations(graph: Dict, entity: str, max_relations: int = 10) -> List[str]:
    """Get available relations from an entity (for negative sampling)."""
    if entity not in graph:
        return []
    relations = list(graph[entity].keys())
    # Shuffle and limit
    import random
    random.shuffle(relations)
    return relations[:max_relations]


def get_available_entities(graph: Dict, entity: str, relation: str, max_entities: int = 20) -> List[str]:
    """Get available entities via a relation from entity (for negative sampling)."""
    if entity not in graph or relation not in graph[entity]:
        return []
    entities = graph[entity][relation]
    import random
    shuffled = entities.copy()
    random.shuffle(shuffled)
    return shuffled[:max_entities]


def create_relation_selection_sample(
    tokenizer,
    question: str,
    topic_entities: List[str],
    current_entity: str,
    path_so_far: List,
    available_relations: List[str],
    correct_relation: str,
    depth: int
) -> Optional[str]:
    """Create a training sample for relation selection."""
    if correct_relation not in available_relations:
        # Ensure correct relation is in the list
        available_relations = [correct_relation] + available_relations[:len(available_relations)-1]

    # Format path so far
    if path_so_far:
        path_str = f"{PATH_START_TOKEN} {path_to_string(path_so_far)} {PATH_END_TOKEN}"
    else:
        path_str = "(Starting position - no previous steps)"

    # Format available relations
    relations_str = "\n".join(f"- {r}" for r in available_relations)

    user_prompt = RELATION_SELECTOR_USER_TEMPLATE.format(
        question=question,
        topic_entities=", ".join(topic_entities),
        path_so_far=path_str,
        current_entity=current_entity,
        depth=depth,
        available_relations=relations_str
    )

    response = RELATION_SELECTOR_RESPONSE_TEMPLATE.format(relation=correct_relation)

    chat = [
        {"role": "system", "content": RELATION_SELECTOR_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": response},
    ]

    return tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)


def create_entity_selection_sample(
    tokenizer,
    question: str,
    topic_entities: List[str],
    current_entity: str,
    path_so_far: List,
    selected_relation: str,
    available_entities: List[str],
    correct_entity: str
) -> Optional[str]:
    """Create a training sample for entity selection."""
    if correct_entity not in available_entities:
        # Ensure correct entity is in the list
        available_entities = [correct_entity] + available_entities[:len(available_entities)-1]

    # Format path so far
    if path_so_far:
        path_str = f"{PATH_START_TOKEN} {path_to_string(path_so_far)} {PATH_END_TOKEN}"
    else:
        path_str = "(Starting position)"

    # Format available entities
    entities_str = "\n".join(f"- {e}" for e in available_entities)

    user_prompt = ENTITY_SELECTOR_USER_TEMPLATE.format(
        question=question,
        topic_entities=", ".join(topic_entities),
        path_so_far=path_str,
        current_entity=current_entity,
        selected_relation=selected_relation,
        available_entities=entities_str
    )

    response = ENTITY_SELECTOR_RESPONSE_TEMPLATE.format(entity=correct_entity)

    chat = [
        {"role": "system", "content": ENTITY_SELECTOR_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": response},
    ]

    return tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)


def create_path_generation_sample(
    tokenizer,
    question: str,
    topic_entities: List[str],
    path: List,
    answer: str
) -> Optional[str]:
    """Create a training sample for full path generation (backward compatible with GCR)."""
    user_prompt = PATH_GENERATION_USER_TEMPLATE.format(
        question=question,
        topic_entities=", ".join(topic_entities)
    )

    response = PATH_GENERATION_RESPONSE_TEMPLATE.format(
        path=path_to_string(path),
        answer=answer
    )

    chat = [
        {"role": "system", "content": PATH_GENERATION_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": response},
    ]

    return tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)


def train():
    parser = HfArgumentParser((ScriptArguments, ScriptTrainingArguments))
    script_args, training_args = parser.parse_args_into_dataclasses()

    # Load model
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
    if script_args.use_peft:
        peft_config = LoraConfig(
            r=script_args.lora_r,
            lora_alpha=script_args.lora_alpha,
            lora_dropout=script_args.lora_dropout,
            target_modules=["q_proj", "v_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )
    else:
        peft_config = None

    tokenizer = AutoTokenizer.from_pretrained(
        script_args.model_name_or_path,
        trust_remote_code=True,
        use_fast=False,
        token=HF_TOKEN,
    )

    tokenizer.padding_side = "right"

    # Add special tokens for AGC-Agent
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict['pad_token'] = '<PAD>'
    special_tokens_dict['additional_special_tokens'] = ALL_SPECIAL_TOKENS

    num_added = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
    print(f"Added {num_added} special tokens: {ALL_SPECIAL_TOKENS}")
    print(f"Vocabulary size: {len(tokenizer)}")

    # Load datasets
    data_list = [
        datasets.load_from_disk(data_path) for data_path in script_args.data_path_list
    ]
    dataset = datasets.concatenate_datasets(data_list)
    print(f"Loaded {len(dataset)} samples from {len(script_args.data_path_list)} datasets")

    def input_formatter(example):
        """
        Format training examples from ground-truth paths.

        For each path, we create:
        1. Step-wise relation selection samples
        2. Step-wise entity selection samples
        3. Full path generation sample (backward compatible)
        """
        chunks = []
        training_mode = script_args.training_mode
        max_relations = script_args.max_relations_per_step
        max_entities = script_args.max_entities_per_step

        for i in range(len(example["q_entity"])):
            question = example["question"][i]
            if not question.endswith("?"):
                question += "?"

            topic_entities = example["q_entity"][i]
            answer_entities = example["a_entity"][i]
            ground_paths = example["ground_truth_paths"][i]
            graph_triples = example.get("graph", [[]])[i] if "graph" in example else []

            # Build graph for negative sampling
            if graph_triples:
                graph = build_graph_from_triples(graph_triples)
            else:
                graph = {}

            if len(ground_paths) == 0:
                continue

            for path in ground_paths:
                if len(path) == 0:
                    continue

                # Path generation sample (backward compatible with GCR)
                if training_mode in ["path", "all"]:
                    path_answer = path[-1][-1].strip()
                    sample = create_path_generation_sample(
                        tokenizer, question, topic_entities, path, path_answer
                    )
                    if sample:
                        chunks.append(sample)

                # Step-wise samples for relation and entity selection
                path_so_far = []
                for step_idx, (h, r, t) in enumerate(path):
                    h, r, t = str(h).strip(), str(r).strip(), str(t).strip()

                    # Relation selection sample
                    if training_mode in ["relation", "all"]:
                        # Get available relations (including negative samples)
                        available_rels = get_available_relations(graph, h, max_relations)
                        if r not in available_rels:
                            available_rels = [r] + available_rels[:max_relations-1]

                        if len(available_rels) > 1:  # Only if there are choices
                            sample = create_relation_selection_sample(
                                tokenizer=tokenizer,
                                question=question,
                                topic_entities=topic_entities,
                                current_entity=h,
                                path_so_far=path_so_far.copy(),
                                available_relations=available_rels,
                                correct_relation=r,
                                depth=step_idx
                            )
                            if sample:
                                chunks.append(sample)

                    # Entity selection sample
                    if training_mode in ["entity", "all"]:
                        # Get available entities (including negative samples)
                        available_ents = get_available_entities(graph, h, r, max_entities)
                        if t not in available_ents:
                            available_ents = [t] + available_ents[:max_entities-1]

                        if len(available_ents) > 1:  # Only if there are choices
                            sample = create_entity_selection_sample(
                                tokenizer=tokenizer,
                                question=question,
                                topic_entities=topic_entities,
                                current_entity=h,
                                path_so_far=path_so_far.copy(),
                                selected_relation=r,
                                available_entities=available_ents,
                                correct_entity=t
                            )
                            if sample:
                                chunks.append(sample)

                    # Update path so far for next step
                    path_so_far.append((h, r, t))

        return {"text": chunks}

    train_dataset = dataset.map(
        input_formatter,
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=N_CPUS,
    )

    print(f"Created {len(train_dataset)} training samples")
    if len(train_dataset) > 0:
        print("Sample training example:")
        print(train_dataset[0]["text"][:2000] + "..." if len(train_dataset[0]["text"]) > 2000 else train_dataset[0]["text"])

    # Prepare instruct tuning
    response_template = script_args.response_template
    data_collator = DataCollatorForCompletionOnlyLM(
        response_template, tokenizer=tokenizer, mlm=False
    )

    sft_cfg = SFTConfig(
        **training_args.to_dict(),
        dataset_text_field="text",
        packing=False,
        dataset_kwargs={"add_special_tokens": False},
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        peft_config=peft_config,
        tokenizer=tokenizer,
        args=sft_cfg,
        data_collator=data_collator,
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

    trainer.train(resume_from_checkpoint=checkpoint)

    trainer.save_model(training_args.output_dir)

    # Save tokenizer with special tokens
    tokenizer.save_pretrained(training_args.output_dir)

    print(f"Model saved to {training_args.output_dir}")


if __name__ == "__main__":
    train()
