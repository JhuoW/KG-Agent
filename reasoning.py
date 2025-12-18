"""
Reasoning Pipeline for Knowledge Graph Question Answering.

Supports two modes:
1. Direct LLM generation (original mode)
2. QC-Agent reasoning (critic-guided beam search)

Multi-GPU Support:
- Automatically distributes dataset across available GPUs
- Each GPU runs its own QC-Agent instance for parallel processing
"""

import os
import argparse
from tqdm import tqdm
import os.path as osp
import torch
import torch.multiprocessing as mp
import json
from datasets import load_dataset
from multiprocessing import Pool
from functools import partial
import queue
import threading
import time

from utils.utils import load_jsonl, build_graph
from utils.kgqa_eval import eval_path_result_w_ans

NUM_GPUS = torch.cuda.device_count()


# =============================================================================
# Original Direct LLM Generation Mode
# =============================================================================

def merge_rule_result(qa_dataset, rule_dataset, n_proc=1, filter_empty=False):
    question_to_rule = dict()
    for data in rule_dataset:
        qid = data["id"]
        predicted_paths = data["prediction"]
        ground_paths = data["ground_paths"]
        question_to_rule[qid] = {
            "predicted_paths": predicted_paths,
            "ground_paths": ground_paths,
        }

    def find_rule(sample):
        qid = sample["id"]
        sample["predicted_paths"] = []
        sample["ground_paths"] = []
        sample["predicted_paths"] = question_to_rule[qid]["predicted_paths"]
        sample["ground_paths"] = question_to_rule[qid]["ground_paths"]
        return sample

    qa_dataset = qa_dataset.map(find_rule, num_proc=n_proc)
    if filter_empty:
        qa_dataset = qa_dataset.filter(
            lambda x: len(x["ground_paths"]) > 0, num_proc=n_proc
        )
    return qa_dataset


def get_output_file(path, force=False):
    if not os.path.exists(path) or force:
        fout = open(path, "w")
        return fout, []
    else:
        with open(path, "r") as f:
            processed_results = []
            for line in f:
                try:
                    results = json.loads(line)
                except:
                    raise ValueError("Error in line: ", line)
                processed_results.append(results["id"])
        fout = open(path, "a")
        return fout, processed_results


def prediction_llm(data, processed_list, input_builder, model):
    """Direct LLM prediction (original mode)."""
    question = data["question"]
    answer = data["answer"]
    id = data["id"]
    if id in processed_list:
        return None
    input_query, gt_paths = input_builder.process_input(data)
    start_token_ids = model.tokenizer.convert_tokens_to_ids(input_builder.PATH_START_TOKEN)
    end_token_ids = model.tokenizer.convert_tokens_to_ids(input_builder.PATH_END_TOKEN)

    input = model.prepare_model_prompt(input_query)
    pred = model.generate_sentence(input, start_token_ids=start_token_ids, end_token_ids=end_token_ids)
    if pred is None:
        return None
    result = {
        "id": id,
        "question": question,
        "prediction": pred,
        "gt_answer": answer,
        "gt_paths": gt_paths,
        "input": input
    }
    return result


# =============================================================================
# QC-Agent Reasoning Mode
# =============================================================================

def prediction_qc_agent(data, processed_list, agent, aggregate_answers=True, validate_answers=True):
    """QC-Agent reasoning with critic-guided beam search."""
    question = data["question"]
    answer = data["answer"]
    id = data["id"]

    if id in processed_list:
        return None

    q_entity = data["q_entity"]
    graph_data = data["graph"]

    # Build graph
    graph = build_graph(graph_data, undirected=False)

    # Run QC-Agent reasoning with validation
    result = agent(
        question=question,
        start_entities=q_entity,
        graph=graph,
        aggregate=aggregate_answers,
        validate_answers=validate_answers,
    )

    # Format predictions as strings matching the expected eval format
    # The eval function expects: "# Reasoning Path:\n{path}\n# Answer:\n{answer}"
    predictions = []
    for path, score in result.finished_paths[:10]:  # Top 10 paths
        path_str = path.to_string()
        end_entity = path.current_entity
        # Format as expected by extract_topk_prediction in kgqa_eval.py
        formatted = f"# Reasoning Path:\n{path_str}\n# Answer:\n{end_entity}"
        predictions.append(formatted)

    # Format aggregated answers
    aggregated = []
    for ans, path_str in result.aggregated_answers:
        aggregated.append({
            "answer": ans,
            "reasoning_path": path_str,
        })

    # Get ground truth paths if available in data
    gt_paths = data.get("gt_paths", [])

    output = {
        "id": id,
        "question": question,
        "prediction": predictions,
        "aggregated_answers": aggregated,
        "gt_answer": answer,
        "gt_paths": gt_paths,
    }

    return output


# =============================================================================
# Multi-GPU Support for QC-Agent
# =============================================================================

def qc_agent_worker(
    gpu_id: int,
    data_shard: list,
    processed_list: list,
    args,
    result_queue: mp.Queue,
    progress_queue: mp.Queue,
):
    """
    Worker function for multi-GPU QC-Agent reasoning.

    Each worker loads its own QC-Agent instance on a specific GPU
    and processes a shard of the dataset.
    """
    import traceback

    try:
        # Set CUDA device for this worker
        torch.cuda.set_device(gpu_id)
        device = f"cuda:{gpu_id}"
        print(f"[GPU {gpu_id}] Initializing on device {device}...")

        from agent import QCAgent, QCAgentConfig

        # QC-Agent configuration for this GPU
        config = QCAgentConfig(
            beam_width=args.beam_width,
            max_candidates_per_step=args.max_candidates,
            max_depth=args.max_depth,
            stop_threshold=args.stop_threshold,
            enable_self_correction=args.enable_self_correction,
            score_drop_threshold=args.score_drop_threshold,
            use_action_scorer=args.use_action_scorer,
            pre_filter_top_k=args.pre_filter_top_k,
            hidden_dim=args.hidden_dim,
            device=device,
        )

        # Initialize QC-Agent on this GPU
        print(f"[GPU {gpu_id}] Loading model from {args.model_path}...")
        agent = QCAgent(
            model_path=args.model_path,
            config=config,
        )

        # Load critic checkpoint if provided
        if args.critic_checkpoint and osp.exists(osp.join(args.critic_checkpoint, "critic.pt")):
            print(f"[GPU {gpu_id}] Loading critic from {args.critic_checkpoint}...")
            critic_state = torch.load(
                osp.join(args.critic_checkpoint, "critic.pt"),
                map_location=device
            )
            agent.critic.load_state_dict(critic_state)

        # Ensure all layers are on correct device
        agent.path_encoder.W_z = agent.path_encoder.W_z.to(device)
        agent.path_encoder.layer_norm = agent.path_encoder.layer_norm.to(device)
        agent.critic.critic.value_head = agent.critic.critic.value_head.to(device)
        if agent.critic.use_action_scorer and agent.critic.action_scorer is not None:
            agent.critic.action_scorer = agent.critic.action_scorer.to(device)

        agent.eval()
        print(f"[GPU {gpu_id}] Agent initialized. Processing {len(data_shard)} samples...")

        # Process data shard
        for idx, data in enumerate(data_shard):
            try:
                res = prediction_qc_agent(
                    data,
                    processed_list=processed_list,
                    agent=agent,
                    aggregate_answers=args.aggregate_answers,
                    validate_answers=args.validate_answers,
                )
                if res is not None:
                    result_queue.put(res)
                progress_queue.put(1)  # Signal progress
            except Exception as e:
                progress_queue.put(1)  # Still count as processed
                print(f"[GPU {gpu_id}] Error processing sample {idx} ({data.get('id', 'unknown')}): {e}")
                if args.debug:
                    traceback.print_exc()

        # Signal completion
        print(f"[GPU {gpu_id}] Worker completed successfully")
        result_queue.put(None)

    except Exception as e:
        print(f"[GPU {gpu_id}] Worker failed with error: {e}")
        traceback.print_exc()
        result_queue.put(None)


def run_qc_agent_mode_multigpu(args):
    """Run QC-Agent reasoning mode with multi-GPU parallelization."""
    from agent import QCAgentConfig

    input_file = osp.join(args.data_path, args.dataset)
    dataset = load_dataset(input_file, split=args.split)

    # Output directory
    data_name = args.dataset + "_undirected" if args.undirected else args.dataset
    post_fix = f"qc_agent-beam{args.beam_width}-depth{args.max_depth}"
    output_dir = os.path.join(args.predict_path, data_name, args.model_name, args.split, post_fix)
    print("Save results to: ", output_dir)

    if not osp.exists(output_dir):
        os.makedirs(output_dir)

    # Save config
    with open(osp.join(output_dir, 'args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=4, default=str)

    fout, processed_list = get_output_file(osp.join(output_dir, 'predictions.jsonl'), force=args.force)

    # Convert HuggingFace dataset items to plain dicts for multiprocessing serialization
    dataset_list = [dict(d) for d in dataset if d["id"] not in processed_list]

    if not dataset_list:
        print("All samples already processed.")
        fout.close()
        eval_path_result_w_ans(osp.join(output_dir, 'predictions.jsonl'))
        return

    num_gpus = min(NUM_GPUS, args.num_gpus) if args.num_gpus > 0 else NUM_GPUS
    print(f"Using {num_gpus} GPUs for parallel reasoning on {len(dataset_list)} samples")

    # Split dataset across GPUs
    shard_size = max(1, len(dataset_list) // num_gpus)
    shards = []
    for i in range(num_gpus):
        start_idx = i * shard_size
        end_idx = start_idx + shard_size if i < num_gpus - 1 else len(dataset_list)
        if start_idx < len(dataset_list):
            shards.append(dataset_list[start_idx:end_idx])

    # Adjust num_gpus if we have fewer shards than GPUs
    num_gpus = len(shards)
    print(f"Distributing {len(dataset_list)} samples across {num_gpus} GPUs")

    # Set up multiprocessing with spawn method
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set

    result_queue = mp.Queue()
    progress_queue = mp.Queue()

    # Start worker processes
    workers = []
    for gpu_id in range(num_gpus):
        print(f"Starting worker on GPU {gpu_id} with {len(shards[gpu_id])} samples")
        p = mp.Process(
            target=qc_agent_worker,
            args=(gpu_id, shards[gpu_id], processed_list, args, result_queue, progress_queue)
        )
        p.start()
        workers.append(p)

    # Collect results with progress bar
    completed_workers = 0
    total_samples = len(dataset_list)
    results_collected = 0

    with tqdm(total=total_samples, desc="QC-Agent Reasoning (Multi-GPU)") as pbar:
        while completed_workers < num_gpus:
            # Check for progress updates (non-blocking)
            try:
                while True:
                    progress_queue.get_nowait()
                    pbar.update(1)
            except:
                pass

            # Check for results
            try:
                result = result_queue.get(timeout=0.5)
                if result is None:
                    completed_workers += 1
                    print(f"\nWorker completed ({completed_workers}/{num_gpus})")
                else:
                    results_collected += 1
                    if args.debug:
                        print(json.dumps(result, indent=2))
                    fout.write(json.dumps(result) + "\n")
                    fout.flush()
            except:
                # Check if any worker crashed
                for i, p in enumerate(workers):
                    if not p.is_alive() and p.exitcode != 0:
                        print(f"\nWarning: Worker {i} exited with code {p.exitcode}")

    print(f"\nCollected {results_collected} results")

    # Wait for all workers to finish
    for p in workers:
        p.join(timeout=10)
        if p.is_alive():
            print(f"Force terminating worker {p.pid}")
            p.terminate()

    fout.close()

    # Evaluate
    if results_collected > 0:
        eval_path_result_w_ans(osp.join(output_dir, 'predictions.jsonl'))
    else:
        print("No results collected. Check for errors above.")


def run_llm_mode(args):
    """Run direct LLM generation mode."""
    from llms.decoding_model import DecodingModel
    from prompt.prompt_builder import PromptBuilder

    input_file = osp.join(args.data_path, args.dataset)
    dataset = load_dataset(input_file, split=args.split)
    post_fix = f"{args.prefix}{args.prompt_mode}-{args.generation_mode}-k{args.k}-index_len{args.index_path_length}"
    data_name = args.dataset + "_undirected" if args.undirected else args.dataset

    if args.add_rule:
        rule_postfix = args.rule_path.replace("/", "_").replace(".", "_")
        rule_dataset = load_jsonl(args.rule_path)
        dataset = merge_rule_result(dataset, rule_dataset, NUM_GPUS, args.filter_empty)
        post_fix += "_" + rule_postfix

    output_dir = os.path.join(args.predict_path, data_name, args.model_name, args.split, post_fix)
    print("Save results to: ", output_dir)

    if not osp.exists(output_dir):
        os.makedirs(output_dir)

    model = DecodingModel(args)

    print("Prepare pipeline for inference...")
    model.prepare_for_inference()
    input_builder = PromptBuilder(
        model.tokenizer,
        prompt=args.prompt_mode,
        undirected=args.undirected,
        index_path_length=args.index_path_length,
        add_rule=args.add_rule
    )

    with open(osp.join(output_dir, 'args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=4)

    fout, processed_list = get_output_file(osp.join(output_dir, 'predictions.jsonl'), force=args.force)

    if NUM_GPUS > 1:
        with Pool(NUM_GPUS) as p:
            for res in tqdm(
                p.imap(
                    partial(
                        prediction_llm,
                        processed_list=processed_list,
                        input_builder=input_builder,
                        model=model,
                    ),
                    dataset,
                ),
                total=len(dataset),
            ):
                if res is not None:
                    if args.debug:
                        print(json.dumps(res))
                    fout.write(json.dumps(res) + "\n")
                    fout.flush()
    else:
        for data in tqdm(dataset):
            res = prediction_llm(
                data,
                processed_list=processed_list,
                input_builder=input_builder,
                model=model,
            )
            if res is not None:
                if args.debug:
                    print(json.dumps(res))
                fout.write(json.dumps(res) + "\n")
                fout.flush()

    fout.close()
    eval_path_result_w_ans(osp.join(output_dir, 'predictions.jsonl'))


def run_qc_agent_mode(args):
    """Run QC-Agent reasoning mode."""
    from agent import QCAgent, QCAgentConfig

    input_file = osp.join(args.data_path, args.dataset)
    dataset = load_dataset(input_file, split=args.split)

    # QC-Agent configuration
    config = QCAgentConfig(
        beam_width=args.beam_width,
        max_candidates_per_step=args.max_candidates,
        max_depth=args.max_depth,
        stop_threshold=args.stop_threshold,
        enable_self_correction=args.enable_self_correction,
        score_drop_threshold=args.score_drop_threshold,
        use_action_scorer=args.use_action_scorer,
        pre_filter_top_k=args.pre_filter_top_k,
        hidden_dim=args.hidden_dim,
    )

    # Output directory
    data_name = args.dataset + "_undirected" if args.undirected else args.dataset
    post_fix = f"qc_agent-beam{args.beam_width}-depth{args.max_depth}"
    output_dir = os.path.join(args.predict_path, data_name, args.model_name, args.split, post_fix)
    print("Save results to: ", output_dir)

    if not osp.exists(output_dir):
        os.makedirs(output_dir)

    # Initialize QC-Agent
    print("Initializing QC-Agent...")
    agent = QCAgent(
        model_path=args.model_path,
        config=config,
    )

    # Load critic checkpoint if provided
    if args.critic_checkpoint:
        print(f"Loading critic from {args.critic_checkpoint}")
        device = config.device
        critic_state = torch.load(
            osp.join(args.critic_checkpoint, "critic.pt"),
            map_location=device
        )
        agent.critic.load_state_dict(critic_state)

    # Ensure all critic layers are on the correct device
    device = config.device
    agent.path_encoder.W_z = agent.path_encoder.W_z.to(device)
    agent.path_encoder.layer_norm = agent.path_encoder.layer_norm.to(device)
    agent.critic.critic.value_head = agent.critic.critic.value_head.to(device)
    if agent.critic.use_action_scorer and agent.critic.action_scorer is not None:
        agent.critic.action_scorer = agent.critic.action_scorer.to(device)

    agent.eval()

    # Save config
    with open(osp.join(output_dir, 'args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=4, default=str)

    fout, processed_list = get_output_file(osp.join(output_dir, 'predictions.jsonl'), force=args.force)

    # Run inference
    for data in tqdm(dataset, desc="QC-Agent Reasoning"):
        res = prediction_qc_agent(
            data,
            processed_list=processed_list,
            agent=agent,
            aggregate_answers=args.aggregate_answers,
            validate_answers=args.validate_answers,
        )
        if res is not None:
            if args.debug:
                print(json.dumps(res, indent=2))
            fout.write(json.dumps(res) + "\n")
            fout.flush()

    fout.close()

    # Evaluate
    eval_path_result_w_ans(osp.join(output_dir, 'predictions.jsonl'))


def main(args):
    """Main entry point."""
    if args.mode == "llm":
        run_llm_mode(args)
    elif args.mode == "qc_agent":
        # Use multi-GPU if more than 1 GPU is available and requested
        num_gpus = min(NUM_GPUS, args.num_gpus) if args.num_gpus > 0 else NUM_GPUS
        if num_gpus > 1:
            run_qc_agent_mode_multigpu(args)
        else:
            run_qc_agent_mode(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Knowledge Graph Reasoning")

    # Common arguments
    parser.add_argument('--data_path', type=str, default='rmanluo')
    parser.add_argument('--dataset', '-d', type=str, default='RoG-webqsp',
                        choices=['RoG-webqsp', 'RoG-cwq'])
    parser.add_argument('--split', type=str, default='test[:100]')
    parser.add_argument('--predict_path', type=str, default='results/GenPaths')
    parser.add_argument('--model_name', type=str, default='FT-Qwen3-8B',
                        help="Model name for output directory")
    parser.add_argument('--force', action='store_true',
                        help="Force overwrite results")
    parser.add_argument("--undirected", type=lambda x: (str(x).lower() == 'true'),
                        default=False)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--prefix", type=str, default="")

    # Mode selection
    parser.add_argument("--mode", type=str, default="qc_agent",
                        choices=["llm", "qc_agent"],
                        help="Reasoning mode: 'llm' for direct generation, 'qc_agent' for agentic reasoning")

    # LLM mode arguments
    parser.add_argument("--prompt_mode", type=str, default="zero-shot",
                        choices=["zero-shot", "mcq-zero-shot", "few-shot"])
    parser.add_argument("--add_rule", action="store_true")
    parser.add_argument("--filter_empty", action="store_true")
    parser.add_argument("--rule_path", type=str, default="")
    parser.add_argument("--index_path_length", type=int, default=2)

    # QC-Agent mode arguments
    parser.add_argument("--model_path", type=str, default="save_models/FT-Qwen3-8B",
                        help="Path to fine-tuned LLM for QC-Agent")
    parser.add_argument("--critic_checkpoint", type=str, default=None,
                        help="Path to trained critic checkpoint")
    parser.add_argument("--beam_width", type=int, default=10,
                        help="Beam width for search")
    parser.add_argument("--max_candidates", type=int, default=3,
                        help="Max candidates per expansion step")
    parser.add_argument("--max_depth", type=int, default=4,
                        help="Maximum reasoning depth")
    parser.add_argument("--stop_threshold", type=float, default=0.5)
    parser.add_argument("--enable_self_correction", action="store_true", default=True)
    parser.add_argument("--score_drop_threshold", type=float, default=0.3)
    parser.add_argument("--use_action_scorer", action="store_true", default=False)
    parser.add_argument("--pre_filter_top_k", type=int, default=10)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--aggregate_answers", action="store_true", default=True,
                        help="Use LLM to aggregate final answers")

    # Multi-GPU arguments
    parser.add_argument("--num_gpus", type=int, default=0,
                        help="Number of GPUs to use (0=all available)")

    # Accuracy improvement arguments
    parser.add_argument("--strict_grounding", action="store_true", default=False,
                        help="Enable strict KG grounding to prevent hallucinations")
    parser.add_argument("--validate_answers", action="store_true", default=False,
                        help="Validate answers exist in the KG before outputting")

    args, remaining = parser.parse_known_args()

    # Add LLM-specific arguments if in LLM mode
    if args.mode == "llm":
        from llms.decoding_model import DecodingModel
        LLM = DecodingModel(args.model_name)
        LLM.add_args(parser)
        args = parser.parse_args()

    main(args)