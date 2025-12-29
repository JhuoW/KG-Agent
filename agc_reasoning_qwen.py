"""
AGC-Agent Reasoning Script for Qwen3-8B with Multi-GPU Support.

This script implements the main entry point for running AGC-Agent on KGQA datasets
using Qwen3-8B as the backbone LLM instead of the GCR-finetuned Llama model.

Key differences from agc_reasoning.py:
- Uses Qwen3-8B model with enable_thinking=False for structured KG reasoning
- Uses Qwen3-specific generation parameters (temp=0.7, top_p=0.8 for sampling)
- Handles Qwen3's larger vocabulary size (~151,936 tokens)

Usage:
    # Single GPU
    python agc_reasoning_qwen.py --gpu_id 0 --d RoG-webqsp --split test[:100]

    # Multi-GPU
    python agc_reasoning_qwen.py --gpu_id 0,1,2,3 --d RoG-webqsp --split test
"""

import os
import sys
import argparse
import json
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional, Any

import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.gcr_utils import eval_path_result_w_ans, get_truth_paths, replace_mid_answers_with_path_entity
from utils.utils import build_graph, path_to_string, load_jsonl
from agc_agent_qwen import QwenAGCAgent, QwenAGCAgentConfig, QwenSimplifiedAGCAgent


class QwenReasoningModel:
    """Wrapper for the Qwen3-8B LLM used in AGC-Agent."""

    DTYPE = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}

    # Special tokens used by AGC-Agent
    SPECIAL_TOKENS = ["<REL>", "</REL>", "<ENT>", "</ENT>", "<PATH>", "</PATH>"]

    @staticmethod
    def add_args(parser):
        """Add model-related arguments."""
        parser.add_argument("--model_path", type=str,
                          default="Qwen/Qwen3-8B",
                          help="HuggingFace model path for Qwen3-8B")
        parser.add_argument("--maximum_token", type=int, default=4096,
                          help="Max input length")
        parser.add_argument("--max_new_tokens", type=int, default=1024,
                          help="Max tokens to generate per step")
        parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"],
                          default="bf16")
        parser.add_argument("--quant", choices=["none", "4bit", "8bit"],
                          default="none")
        parser.add_argument("--attn_implementation",
                          default="sdpa",
                          choices=["eager", "sdpa", "flash_attention_2"])

    def __init__(self, args):
        """Initialize the model wrapper."""
        self.args = args
        self.model = None
        self.tokenizer = None

    def _is_peft_adapter(self, model_path: str) -> bool:
        """Check if model_path contains a PEFT adapter."""
        adapter_config = Path(model_path) / "adapter_config.json"
        return adapter_config.exists()

    def _get_base_model_from_adapter(self, model_path: str) -> str:
        """Extract base model path from adapter config."""
        adapter_config = Path(model_path) / "adapter_config.json"
        with open(adapter_config) as f:
            config = json.load(f)
        return config.get("base_model_name_or_path", "Qwen/Qwen3-8B")

    def prepare_for_inference(self):
        """Load Qwen3-8B model and tokenizer (with PEFT adapter support)."""
        model_path = self.args.model_path
        is_peft = self._is_peft_adapter(model_path)

        if is_peft:
            base_model_path = self._get_base_model_from_adapter(model_path)
            print(f"Detected PEFT adapter at {model_path}")
            print(f"Loading base model from {base_model_path}...")
        else:
            base_model_path = model_path
            print(f"Loading Qwen3-8B model from {model_path}...")

        # Load tokenizer from adapter path (has special tokens) or base model
        tokenizer_path = model_path if is_peft else base_model_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True
        )

        # Ensure pad_token is set for Qwen3
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        quantization_config = None
        if self.args.quant in ["4bit", "8bit"]:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=self.args.quant == "4bit",
                load_in_8bit=self.args.quant == "8bit",
            )

        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            trust_remote_code=True,
            torch_dtype=self.DTYPE.get(self.args.dtype),
            attn_implementation=self.args.attn_implementation,
            quantization_config=quantization_config
        ).cuda()

        # Resize embeddings to match tokenizer (MUST happen BEFORE loading PEFT adapter)
        if len(self.tokenizer) != self.model.config.vocab_size:
            print(f"Resizing model embeddings from {self.model.config.vocab_size} to {len(self.tokenizer)}")
            self.model.resize_token_embeddings(len(self.tokenizer))

        # Load PEFT adapter if present
        if is_peft:
            from peft import PeftModel
            print(f"Loading PEFT adapter from {model_path}...")
            self.model = PeftModel.from_pretrained(self.model, model_path)
            print("PEFT adapter loaded successfully")

        self.model.eval()
        print(f"Model loaded successfully. Vocab size: {len(self.tokenizer)}")


def process_sample(
    data: dict,
    agent: QwenAGCAgent,
    undirected: bool = False,
    filter_mid: bool = False
) -> Optional[dict]:
    """
    Process a single sample with Qwen3-8B AGC-Agent.

    Args:
        data: Sample from dataset with 'question', 'answer', 'q_entity', 'graph'
        agent: The Qwen3-8B AGC-Agent instance
        undirected: Whether to treat graph as undirected
        filter_mid: Whether to filter out Freebase MID answers

    Returns:
        Result dict or None if processing fails
    """
    question = data["question"]
    answer = data["answer"]
    q_entity = data["q_entity"]
    a_entity = data.get("a_entity", answer)
    graph_triples = data["graph"]
    sample_id = data["id"]

    # Ensure question ends with "?" for LLM input
    question_for_llm = question
    if not question_for_llm.endswith("?"):
        question_for_llm += "?"

    # Convert graph triples to (head, relation, tail) format
    triples = [(t[0], t[1], t[2]) for t in graph_triples]

    # If undirected, add reverse edges
    if undirected:
        reverse_triples = [(t[2], t[1] + "_inv", t[0]) for t in triples]
        triples = triples + reverse_triples

    try:
        # Run Qwen3-8B AGC-Agent reasoning
        result = agent.reason(
            question=question_for_llm,
            graph_triples=triples,
            topic_entities=q_entity
        )

        # Handle invalid Freebase MID answers (e.g., m.012zbkk5, g.125czvn3w)
        # Instead of filtering (which removes useful paths), replace MID answers
        # with the last valid entity from the reasoning path
        if filter_mid:
            processed_predictions = replace_mid_answers_with_path_entity(
                result.predictions, topic_entities=q_entity
            )
        else:
            processed_predictions = result.predictions

        # Get ground truth paths
        g = build_graph(graph_triples, undirected)
        truth_paths = get_truth_paths(q_entity, a_entity, g)
        ground_paths = [path_to_string(p) for p in truth_paths]

        return {
            "id": sample_id,
            "question": question,
            "prediction": processed_predictions,
            "ground_truth": answer,
            "ground_truth_paths": ground_paths,
            "reasoning_trace": result.reasoning_trace
        }

    except Exception as e:
        print(f"Error processing {sample_id}: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_worker(args, model_class):
    """
    Worker function for multi-GPU execution.
    Called as a subprocess with CUDA_VISIBLE_DEVICES set.
    """
    gpu_id = args.worker_gpu
    start_idx = args.worker_start_idx
    end_idx = args.worker_end_idx
    output_file = args.worker_output_file

    print(f"[GPU {gpu_id}] Starting Qwen3-8B worker for indices {start_idx} to {end_idx}")

    # Load dataset
    input_file = os.path.join(args.data_path, args.d)
    full_dataset = load_dataset(input_file, split=args.split)

    # Get subset for this worker
    dataset_indices = list(range(start_idx, end_idx))
    dataset = full_dataset.select(dataset_indices)

    print(f"[GPU {gpu_id}] Processing {len(dataset)} samples with Qwen3-8B")

    # Initialize Qwen3-8B model
    model_wrapper = model_class(args)
    model_wrapper.prepare_for_inference()

    # Create Qwen3-8B AGC-Agent config
    max_depth = args.index_path_length if args.index_path_length else args.max_depth
    config = QwenAGCAgentConfig(
        beam_width=args.beam_width,
        max_depth=max_depth,
        max_backtracks=args.max_backtracks,
        relation_top_k=args.relation_top_k,
        entity_top_k=args.entity_top_k,
        answer_threshold=args.answer_threshold,
        use_constrained_generation=args.use_constrained_generation,
        generation_mode=args.generation_mode,
        output_top_k=args.k
    )

    # Create Qwen3-8B agent
    if args.simplified:
        agent = QwenSimplifiedAGCAgent(
            model=model_wrapper.model,
            tokenizer=model_wrapper.tokenizer,
            config=config
        )
    else:
        agent = QwenAGCAgent(
            model=model_wrapper.model,
            tokenizer=model_wrapper.tokenizer,
            config=config
        )

    # Process samples
    results = []
    for data in tqdm(dataset, desc=f"GPU {gpu_id} (Qwen3)"):
        result = process_sample(data, agent, args.undirected, args.filter_mid)
        if result is not None:
            results.append(result)
            if args.debug:
                print(f"[GPU {gpu_id}] {json.dumps(result, indent=2)}")
        else:
            print(f"[GPU {gpu_id}] Failed: {data['id']}")

    # Write results - ensure directory exists first
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_file, "w") as f:
        for res in results:
            f.write(json.dumps(res) + "\n")

    print(f"[GPU {gpu_id}] Finished. Saved {len(results)} results to {output_file}")


def main_multigpu(args, model_class):
    """Main function for multi-GPU execution."""
    gpu_ids = [int(x.strip()) for x in args.gpu_id.split(",")]
    num_gpus = len(gpu_ids)

    print(f"Using GPUs: {gpu_ids} for Qwen3-8B AGC-Agent")

    # Setup output directory
    max_depth = args.index_path_length if args.index_path_length else args.max_depth
    post_fix = f"qwen3-agc-agent-{args.generation_mode}-depth{max_depth}-k{args.k}"
    if args.simplified:
        post_fix = "simplified-" + post_fix

    data_name = args.d + ("_undirected" if args.undirected else "")
    output_dir = os.path.join(
        args.predict_path, data_name,
        args.model_name or Path(args.model_path).name,
        args.split.replace("[", "_").replace("]", "_").replace(":", "-"),
        post_fix
    )

    print(f"Output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # Save args
    with open(os.path.join(output_dir, 'args.txt'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Load dataset to get total size
    input_file = os.path.join(args.data_path, args.d)
    full_dataset = load_dataset(input_file, split=args.split)
    total_samples = len(full_dataset)
    print(f"Total samples: {total_samples}")

    # Split across GPUs
    samples_per_gpu = total_samples // num_gpus
    remainder = total_samples % num_gpus

    gpu_ranges = []
    start_idx = 0
    for i in range(num_gpus):
        end_idx = start_idx + samples_per_gpu + (1 if i < remainder else 0)
        gpu_ranges.append((start_idx, end_idx))
        start_idx = end_idx

    for i, gpu_id in enumerate(gpu_ids):
        start, end = gpu_ranges[i]
        print(f"GPU {gpu_id}: {end - start} samples (indices {start} to {end - 1})")

    # Temp output files
    temp_files = [
        os.path.join(output_dir, f'predictions_gpu{gpu_id}.jsonl.tmp')
        for gpu_id in gpu_ids
    ]

    # Build subprocess command
    base_cmd = [
        sys.executable, __file__,
        "--worker_mode",
        "--data_path", args.data_path,
        "--d", args.d,
        "--split", args.split,
        "--model_path", args.model_path,
        "--predict_path", args.predict_path,
        "--k", str(args.k),
        "--beam_width", str(args.beam_width),
        "--max_depth", str(args.max_depth),
        "--index_path_length", str(args.index_path_length),
        "--max_backtracks", str(args.max_backtracks),
        "--relation_top_k", str(args.relation_top_k),
        "--entity_top_k", str(args.entity_top_k),
        "--answer_threshold", str(args.answer_threshold),
        "--generation_mode", args.generation_mode,
        "--attn_implementation", args.attn_implementation,
        "--dtype", args.dtype,
        "--max_new_tokens", str(args.max_new_tokens),
        "--maximum_token", str(args.maximum_token),
    ]

    if args.undirected:
        base_cmd.append("--undirected")
    if args.debug:
        base_cmd.append("--debug")
    if args.simplified:
        base_cmd.append("--simplified")
    if not args.use_constrained_generation:
        base_cmd.append("--no_constrained_generation")
    if args.filter_mid:
        base_cmd.append("--filter_mid")
    if args.model_name:
        base_cmd.extend(["--model_name", args.model_name])
    if args.quant != "none":
        base_cmd.extend(["--quant", args.quant])

    # Launch workers
    processes = []
    for i, gpu_id in enumerate(gpu_ids):
        start_idx, end_idx = gpu_ranges[i]
        worker_cmd = base_cmd + [
            "--worker_gpu", str(gpu_id),
            "--worker_start_idx", str(start_idx),
            "--worker_end_idx", str(end_idx),
            "--worker_output_file", temp_files[i],
        ]

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        print(f"Launching Qwen3-8B worker for GPU {gpu_id}...")
        p = subprocess.Popen(worker_cmd, env=env)
        processes.append(p)

    # Wait for all
    print("Waiting for Qwen3-8B workers...")
    for p in processes:
        p.wait()

    print("Merging results...")

    # Merge results
    final_output = os.path.join(output_dir, 'predictions.jsonl')
    all_results = []

    for temp_file in temp_files:
        if os.path.exists(temp_file):
            with open(temp_file, 'r') as f:
                for line in f:
                    all_results.append(json.loads(line))
            os.remove(temp_file)

    # Sort by ID
    all_results.sort(key=lambda x: x['id'])

    # Write merged results - ensure directory exists
    final_output_dir = os.path.dirname(final_output)
    if final_output_dir:
        os.makedirs(final_output_dir, exist_ok=True)

    with open(final_output, 'w') as f:
        for res in all_results:
            f.write(json.dumps(res) + "\n")

    print(f"Merged {len(all_results)} results to {final_output}")

    # Evaluate
    eval_path_result_w_ans(final_output)


def main_single_gpu(args, model_class):
    """Main function for single GPU execution."""
    # Setup output directory
    max_depth = args.index_path_length if args.index_path_length else args.max_depth
    post_fix = f"qwen3-agc-agent-{args.generation_mode}-depth{max_depth}-k{args.k}"
    if args.simplified:
        post_fix = "simplified-" + post_fix

    data_name = args.d + ("_undirected" if args.undirected else "")
    output_dir = os.path.join(
        args.predict_path, data_name,
        args.model_name or Path(args.model_path).name,
        args.split.replace("[", "_").replace("]", "_").replace(":", "-"),
        post_fix
    )

    print(f"Output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # Save args
    with open(os.path.join(output_dir, 'args.txt'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Load dataset
    input_file = os.path.join(args.data_path, args.d)
    dataset = load_dataset(input_file, split=args.split)
    print(f"Loaded {len(dataset)} samples for Qwen3-8B AGC-Agent")

    # Initialize Qwen3-8B model
    model_wrapper = model_class(args)
    model_wrapper.prepare_for_inference()

    # Create Qwen3-8B AGC-Agent config
    max_depth = args.index_path_length if args.index_path_length else args.max_depth
    config = QwenAGCAgentConfig(
        beam_width=args.beam_width,
        max_depth=max_depth,
        max_backtracks=args.max_backtracks,
        relation_top_k=args.relation_top_k,
        entity_top_k=args.entity_top_k,
        answer_threshold=args.answer_threshold,
        use_constrained_generation=args.use_constrained_generation,
        generation_mode=args.generation_mode,
        output_top_k=args.k
    )

    # Create Qwen3-8B agent
    if args.simplified:
        agent = QwenSimplifiedAGCAgent(
            model=model_wrapper.model,
            tokenizer=model_wrapper.tokenizer,
            config=config
        )
    else:
        agent = QwenAGCAgent(
            model=model_wrapper.model,
            tokenizer=model_wrapper.tokenizer,
            config=config
        )

    # Output file
    output_file = os.path.join(output_dir, 'predictions.jsonl')

    # Check for existing results to resume
    processed_ids = set()
    if os.path.exists(output_file) and not args.force:
        with open(output_file, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    processed_ids.add(data['id'])
                except:
                    pass
        print(f"Resuming from {len(processed_ids)} processed samples")
        fout = open(output_file, 'a')
    else:
        fout = open(output_file, 'w')

    # Process samples
    for data in tqdm(dataset, desc="Qwen3-8B AGC-Agent"):
        if data['id'] in processed_ids:
            continue

        result = process_sample(data, agent, args.undirected, args.filter_mid)
        if result is not None:
            if args.debug:
                print(json.dumps(result, indent=2))
            fout.write(json.dumps(result) + "\n")
            fout.flush()
        else:
            print(f"Failed: {data['id']}")

    fout.close()

    # Evaluate
    eval_path_result_w_ans(output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen3-8B AGC-Agent Reasoning")

    # Data arguments
    parser.add_argument('--data_path', type=str, default='rmanluo',
                       help="HuggingFace dataset path")
    parser.add_argument('--d', '-d', type=str, default='RoG-webqsp',
                       help="Dataset name")
    parser.add_argument('--split', type=str, default='test[:100]',
                       help="Dataset split")
    parser.add_argument('--predict_path', type=str, default='results/AGC-Agent-Qwen',
                       help="Output directory")

    # Model arguments
    parser.add_argument('--model_name', type=str, default=None,
                       help="Model name for output directory")

    # AGC-Agent arguments
    parser.add_argument('--beam_width', type=int, default=10,
                       help="Beam width for search")
    parser.add_argument('--max_depth', type=int, default=2,
                       help="Maximum reasoning depth")
    parser.add_argument('--index_path_length', type=int, default=2,
                       help="Maximum path length (alias for max_depth)")
    parser.add_argument('--max_backtracks', type=int, default=3,
                       help="Maximum backtracks per beam")
    parser.add_argument('--relation_top_k', type=int, default=3,
                       help="Top-k relations per step")
    parser.add_argument('--entity_top_k', type=int, default=3,
                       help="Top-k entities per step")
    parser.add_argument('--answer_threshold', type=float, default=0.5,
                       help="Confidence threshold for ANSWER action")
    parser.add_argument('--k', type=int, default=10,
                       help="Number of paths to output")
    parser.add_argument('--generation_mode', type=str, default='beam',
                       choices=['greedy', 'beam', 'sampling'],
                       help="Generation mode")

    # Execution arguments
    parser.add_argument('--gpu_id', type=str, default="0,1,2",
                       help="GPU ID(s), e.g., '0' or '0,1,2'")
    parser.add_argument('--undirected', action='store_true',
                       help="Treat graph as undirected")
    parser.add_argument('--force', action='store_true',
                       help="Overwrite existing results")
    parser.add_argument('--debug', action='store_true',
                       help="Print debug information")
    parser.add_argument('--simplified', action='store_true',
                       help="Use simplified agent (single LLM call per step)")
    parser.add_argument('--no_constrained_generation', action='store_true',
                       help="Disable trie-constrained generation")
    parser.add_argument('--filter_mid', action='store_true',
                       help="Filter invalid Freebase MID answers")

    # Worker mode arguments
    parser.add_argument("--worker_mode", action="store_true")
    parser.add_argument("--worker_gpu", type=int, default=0)
    parser.add_argument("--worker_start_idx", type=int, default=0)
    parser.add_argument("--worker_end_idx", type=int, default=0)
    parser.add_argument("--worker_output_file", type=str, default="")

    # Parse known args first to add model args
    args, _ = parser.parse_known_args()

    # Add Qwen3-8B model arguments
    QwenReasoningModel.add_args(parser)

    # Parse all args
    args = parser.parse_args()

    # Handle constrained generation flag
    args.use_constrained_generation = not args.no_constrained_generation

    if args.worker_mode:
        print(f"Qwen3-8B Worker mode: GPU {args.worker_gpu}, samples {args.worker_start_idx}-{args.worker_end_idx}")
        run_worker(args, QwenReasoningModel)
    else:
        gpu_ids = [x.strip() for x in args.gpu_id.split(",")]

        if len(gpu_ids) == 1:
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids[0]
            print(f"Qwen3-8B Single GPU mode on GPU {gpu_ids[0]}")
            main_single_gpu(args, QwenReasoningModel)
        else:
            print(f"Qwen3-8B Multi-GPU mode on GPUs {gpu_ids}")
            main_multigpu(args, QwenReasoningModel)
