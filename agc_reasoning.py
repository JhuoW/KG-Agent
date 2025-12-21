"""
AGC-Agent Reasoning Script with Multi-GPU Support.

This script implements the main entry point for running AGC-Agent on KGQA datasets.
It follows the same structure as the reference implementation (reasoning_trie_multigpu.py)
but replaces the static KG-Trie with dynamic step-wise constraints.

Usage:
    # Single GPU
    python agc_reasoning.py --gpu_id 0 --d RoG-webqsp --split test[:100]

    # Multi-GPU
    python agc_reasoning.py --gpu_id 0,1,2,3 --d RoG-webqsp --split test
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

from utils.gcr_utils import eval_path_result_w_ans, get_truth_paths
from utils.utils import build_graph, path_to_string, load_jsonl
from agc_agent import AGCAgent, AGCAgentConfig, SimplifiedAGCAgent


class AGCReasoningModel:
    """Wrapper for the KG-specialized LLM used in AGC-Agent."""

    DTYPE = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}

    @staticmethod
    def add_args(parser):
        """Add model-related arguments."""
        parser.add_argument("--model_path", type=str,
                          default="rmanluo/GCR-Meta-Llama-3.1-8B-Instruct",
                          help="HuggingFace model path")
        parser.add_argument("--maximum_token", type=int, default=4096,
                          help="Max input length")
        parser.add_argument("--max_new_tokens", type=int, default=1024,
                          help="Max tokens to generate per step")
        parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"],
                          default="bf16")
        parser.add_argument("--quant", choices=["none", "4bit", "8bit"],
                          default="none")
        parser.add_argument("--attn_implementation",
                          default="flash_attention_2",
                          choices=["eager", "sdpa", "flash_attention_2"])

    def __init__(self, args):
        """Initialize the model wrapper."""
        self.args = args
        self.model = None
        self.tokenizer = None

    def prepare_for_inference(self):
        """Load model and tokenizer."""
        print(f"Loading model from {self.args.model_path}...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.args.model_path,
            trust_remote_code=True
        )

        quantization_config = None
        if self.args.quant in ["4bit", "8bit"]:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=self.args.quant == "4bit",
                load_in_8bit=self.args.quant == "8bit",
            )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.args.model_path,
            trust_remote_code=True,
            torch_dtype=self.DTYPE.get(self.args.dtype),
            attn_implementation=self.args.attn_implementation,
            quantization_config=quantization_config
        ).cuda()

        self.model.eval()
        print("Model loaded successfully.")


def process_sample(
    data: dict,
    agent: AGCAgent,
    undirected: bool = False
) -> Optional[dict]:
    """
    Process a single sample with AGC-Agent.

    Args:
        data: Sample from dataset with 'question', 'answer', 'q_entity', 'graph'
        agent: The AGC-Agent instance
        undirected: Whether to treat graph as undirected

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
        # Run AGC-Agent reasoning

        result = agent.reason(
            question=question_for_llm,
            graph_triples=triples,
            topic_entities=q_entity
        )

        # Get ground truth paths
        g = build_graph(graph_triples, undirected)
        truth_paths = get_truth_paths(q_entity, a_entity, g)
        ground_paths = [path_to_string(p) for p in truth_paths]

        return {
            "id": sample_id,
            "question": question,  # Use original question in output
            "prediction": result.predictions,
            "ground_truth": answer,
            "ground_truth_paths": ground_paths,
            "reasoning_trace": result.reasoning_trace
        }

    except Exception as e:
        print(f"Error processing {sample_id}: {e}")
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

    print(f"[GPU {gpu_id}] Starting worker for indices {start_idx} to {end_idx}")

    # Load dataset
    input_file = os.path.join(args.data_path, args.d)
    full_dataset = load_dataset(input_file, split=args.split)

    # Get subset for this worker
    dataset_indices = list(range(start_idx, end_idx))
    dataset = full_dataset.select(dataset_indices)

    print(f"[GPU {gpu_id}] Processing {len(dataset)} samples")

    # Initialize model
    model_wrapper = model_class(args)
    model_wrapper.prepare_for_inference()

    # Create AGC-Agent config (aligned with GCR)
    # Use index_path_length for max_depth if specified (GCR compatibility)
    max_depth = args.index_path_length if args.index_path_length else args.max_depth
    config = AGCAgentConfig(
        beam_width=args.beam_width,
        max_depth=max_depth,
        max_backtracks=args.max_backtracks,
        relation_top_k=args.relation_top_k,
        entity_top_k=args.entity_top_k,
        answer_threshold=args.answer_threshold,
        use_constrained_generation=args.use_constrained_generation,
        generation_mode=args.generation_mode,  # beam, greedy, or sampling
        output_top_k=args.k
    )

    # Create agent
    if args.simplified:
        agent = SimplifiedAGCAgent(
            model=model_wrapper.model,
            tokenizer=model_wrapper.tokenizer,
            config=config
        )
    else:
        agent = AGCAgent(
            model=model_wrapper.model,
            tokenizer=model_wrapper.tokenizer,
            config=config
        )

    # Process samples
    results = []
    for data in tqdm(dataset, desc=f"GPU {gpu_id}"):
        result = process_sample(data, agent, args.undirected)
        if result is not None:
            results.append(result)
            if args.debug:
                print(f"[GPU {gpu_id}] {json.dumps(result, indent=2)}")
        else:
            print(f"[GPU {gpu_id}] Failed: {data['id']}")

    # Write results
    with open(output_file, "w") as f:
        for res in results:
            f.write(json.dumps(res) + "\n")

    print(f"[GPU {gpu_id}] Finished. Saved {len(results)} results to {output_file}")


def main_multigpu(args, model_class):
    """Main function for multi-GPU execution."""
    gpu_ids = [int(x.strip()) for x in args.gpu_id.split(",")]
    num_gpus = len(gpu_ids)

    print(f"Using GPUs: {gpu_ids}")

    # Setup output directory (aligned with GCR naming)
    max_depth = args.index_path_length if args.index_path_length else args.max_depth
    post_fix = f"agc-agent-{args.generation_mode}-depth{max_depth}-k{args.k}"
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
        "--generation_mode", args.generation_mode,  # Add generation mode
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
    if args.model_name:
        base_cmd.extend(["--model_name", args.model_name])

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

        print(f"Launching worker for GPU {gpu_id}...")
        p = subprocess.Popen(worker_cmd, env=env)
        processes.append(p)

    # Wait for all
    print("Waiting for workers...")
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

    # Write merged results
    with open(final_output, 'w') as f:
        for res in all_results:
            f.write(json.dumps(res) + "\n")

    print(f"Merged {len(all_results)} results to {final_output}")

    # Evaluate
    eval_path_result_w_ans(final_output)


def main_single_gpu(args, model_class):
    """Main function for single GPU execution."""
    # Setup output directory (aligned with GCR naming)
    max_depth = args.index_path_length if args.index_path_length else args.max_depth
    post_fix = f"agc-agent-{args.generation_mode}-depth{max_depth}-k{args.k}"
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
    print(f"Loaded {len(dataset)} samples")

    # Initialize model
    model_wrapper = model_class(args)
    model_wrapper.prepare_for_inference()

    # Create AGC-Agent config (aligned with GCR)
    # Use index_path_length for max_depth if specified (GCR compatibility)
    max_depth = args.index_path_length if args.index_path_length else args.max_depth
    config = AGCAgentConfig(
        beam_width=args.beam_width,
        max_depth=max_depth,
        max_backtracks=args.max_backtracks,
        relation_top_k=args.relation_top_k,
        entity_top_k=args.entity_top_k,
        answer_threshold=args.answer_threshold,
        use_constrained_generation=args.use_constrained_generation,
        generation_mode=args.generation_mode,  # beam, greedy, or sampling
        output_top_k=args.k
    )

    # Create agent
    if args.simplified:
        agent = SimplifiedAGCAgent(
            model=model_wrapper.model,
            tokenizer=model_wrapper.tokenizer,
            config=config
        )
    else:
        agent = AGCAgent(
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
    for data in tqdm(dataset):
        if data['id'] in processed_ids:
            continue

        result = process_sample(data, agent, args.undirected)
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
    parser = argparse.ArgumentParser(description="AGC-Agent Reasoning")

    # Data arguments
    parser.add_argument('--data_path', type=str, default='rmanluo',
                       help="HuggingFace dataset path")
    parser.add_argument('--d', '-d', type=str, default='RoG-webqsp',
                       help="Dataset name")
    parser.add_argument('--split', type=str, default='test[:100]',
                       help="Dataset split")
    parser.add_argument('--predict_path', type=str, default='results/AGC-Agent',
                       help="Output directory")

    # Model arguments
    parser.add_argument('--model_name', type=str, default=None,
                       help="Model name for output directory")

    # AGC-Agent arguments
    parser.add_argument('--beam_width', type=int, default=10,
                       help="Beam width for search")
    parser.add_argument('--max_depth', type=int, default=2,
                       help="Maximum reasoning depth (aligned with GCR index_path_length)")
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
                       help="Number of paths to output (aligned with GCR)")
    parser.add_argument('--generation_mode', type=str, default='beam',
                       choices=['greedy', 'beam', 'sampling'],
                       help="Generation mode (aligned with GCR)")

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

    # Worker mode arguments
    parser.add_argument("--worker_mode", action="store_true")
    parser.add_argument("--worker_gpu", type=int, default=0)
    parser.add_argument("--worker_start_idx", type=int, default=0)
    parser.add_argument("--worker_end_idx", type=int, default=0)
    parser.add_argument("--worker_output_file", type=str, default="")

    # Parse known args first to add model args
    args, _ = parser.parse_known_args()

    # Add model arguments
    AGCReasoningModel.add_args(parser)

    # Parse all args
    args = parser.parse_args()

    # Handle constrained generation flag
    args.use_constrained_generation = not args.no_constrained_generation

    if args.worker_mode:
        print(f"Worker mode: GPU {args.worker_gpu}, samples {args.worker_start_idx}-{args.worker_end_idx}")
        run_worker(args, AGCReasoningModel)
    else:
        gpu_ids = [x.strip() for x in args.gpu_id.split(",")]

        if len(gpu_ids) == 1:
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids[0]
            print(f"Single GPU mode on GPU {gpu_ids[0]}")
            main_single_gpu(args, AGCReasoningModel)
        else:
            print(f"Multi-GPU mode on GPUs {gpu_ids}")
            main_multigpu(args, AGCReasoningModel)
