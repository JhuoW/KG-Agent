"""
前100个问题
Accuracy: 68.86434383638422 
Hit: 80.0 
F1: 39.076460717441805 
Precision: 37.99444444444445 
Recall: 68.68847894675014 
Path F1: 46.95318447897334 
Path Precision: 47.027777777777786 
Path Recall: 68.0561566113839 
Path Answer F1: 48.14609313920996 
Path Answer Precision: 49.25 
Path Answer Recall: 68.85842667662091
"""
import os
import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from tqdm import tqdm
from datasets import load_dataset
from utils.gcr_utils import eval_path_result_w_ans
import json
import subprocess
from llms.base_hf_casual_lm import HFCasualModel
from llms.decoding_model import GraphConstrainedDecoding
from utils.utils import load_jsonl, build_graph, path_to_string
from utils.gcr_utils import MarisaTrie, dfs, bfs_with_rule, get_truth_paths


class PathGenerationWithAnswerPromptBuilder(object):
    PATH_START_TOKEN = "<PATH>"
    PATH_END_TOKEN = "</PATH>"
    ZERO_SHOT_PROMPT = """Reasoning path is a sequence of triples in the KG that connects the topic entities in the question to answer entities. Given a question, please generate some reasoning paths in the KG starting from the topic entities to answer the question.

# Question:
{question}
# Topic entities:
{entities}
"""
    MCQ_ZERO_SHOT_PROMPT = """Reasoning path is a sequence of triples in the KG that connects the topic entities in the question to answer entities. Given a question, please generate some reasoning paths in the KG starting from the topic entities to answer the question.

# Question:
{question}
# Topic entities:
{entities}
# Answer Choices:
{choices}
"""
    ZERO_SHOT_NO_MORE_THAN_10_PROMPT = """Reasoning path is a sequence of triples in the KG that connects the topic entities in the question to answer entities. It should start with <PATH> and end with </PATH>. When given a question, please generate some reasoning paths in the KG starting from the topic entities that you believe can aid in answering it. Then, use these reasoning paths to derive the answer to the question. Do not generate more than 10 reasoning paths.

# Question:
{question}
# Topic entities:
{entities}
"""
    MULTIPATH_GEN_PROMPT = """Reasoning path is a sequence of triples in the KG that connects the topic entities in the question to answer entities. Given the question, please generate some reasoning paths in the KG starting from the topic entities that you believe can aid in answering it.

# Question:
{question}
# Topic entities:
{entities}
# Reasoning paths:
"""
    def __init__(self, tokenizer, prompt="zero-shot", undirected=False, index_path_length=2, add_rule=False):
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.undirected = undirected
        self.index_path_length = index_path_length
        self.add_rule = add_rule
        self.prompt_template = self.get_prompt_template(self.prompt)

    def get_prompt_template(self, template_name):
        try:
            template_name = template_name.upper().replace("-", "_") + "_PROMPT"
            return self.__getattribute__(template_name)
        except:
            raise ValueError(f"The template name: {template_name} is not valid.")

    def format_input_with_template(self, question, start_entities, choices=[]):
        if len(choices) > 0:
            return self.prompt_template.format(
                question=question, entities=",".join(start_entities), choices="\n".join(choices)
            )
        else:
            return self.prompt_template.format(
                question=question, entities=",".join(start_entities)
            )

    def apply_rules(self, graph, rules, srouce_entities):
        results = []
        for entity in srouce_entities:
            for rule in rules:
                res = bfs_with_rule(graph, entity, rule)
                results.extend(res)
        return results

    def get_graph_index(self, question_dict):
        if "paths" in question_dict:
            paths_list = question_dict["paths"]
        else:
            g = build_graph(question_dict["graph"], self.undirected)
            paths_list = dfs(g, question_dict["q_entity"], self.index_path_length)

        paths_list_str = [f"{self.PATH_START_TOKEN}{path_to_string(path)}{self.PATH_END_TOKEN}" for path in paths_list]
        if len(paths_list_str) == 0:
            return None
        tokenized_paths = self.tokenizer(
            paths_list_str, padding=False, add_special_tokens=False
        ).input_ids
        return MarisaTrie(tokenized_paths, max_token_id=len(self.tokenizer) + 1)

    def process_input(self, question_dict, return_tire=True):
        question = question_dict["question"]
        start_node = question_dict["q_entity"]
        anser_node = question_dict["a_entity"]
        choices = question_dict.get("choices", [])
        trie = None
        if return_tire:
            trie = self.get_graph_index(question_dict)

        g = build_graph(question_dict["graph"], self.undirected)
        truth_paths = get_truth_paths(start_node, anser_node, g)
        ground_paths = [path_to_string(path) for path in truth_paths]

        if not question.endswith("?"):
            question += "?"

        input = self.format_input_with_template(question, start_node, choices=choices)
        return input, ground_paths, trie


class GraphConstrainedDecodingModel(HFCasualModel):
    def __init__(self, args):
        super().__init__(args)

    def generate_sentence(self, llm_input, trie, start_token_ids=None, end_token_ids=None, enable_constrained_by_default=True):
        inputs = self.tokenizer(llm_input, return_tensors="pt", add_special_tokens=False)

        input_ids = inputs.input_ids.to(self.model.device)
        attention_mask = inputs.attention_mask.to(self.model.device)

        input_length = input_ids.shape[1]
        gcr = GraphConstrainedDecoding(self.tokenizer, trie, start_token_ids, end_token_ids, enable_constrained_by_default)

        try:
            res = self.model.generate(input_ids=input_ids,
                                      attention_mask=attention_mask,
                                      generation_config=self.generation_cfg,
                                      prefix_allowed_tokens_fn=gcr.allowed_tokens_fn,
                                      return_dict_in_generate=True,
                                      pad_token_id=self.tokenizer.eos_token_id)
        except Exception as e:
            print(e)
            return None

        response = []
        if len(res.sequences) == 1:
            return self.tokenizer.decode(res.sequences[0][input_length:], skip_special_tokens=True)
        for r in res.sequences:
            response.append(self.tokenizer.decode(r[input_length:], skip_special_tokens=True))
        return response


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


def prediction(data, processed_list, input_builder, model):
    question = data["question"]
    answer = data["answer"]
    id = data["id"]
    if id in processed_list:
        return None

    input_query, ground_paths, trie = input_builder.process_input(data)
    if trie is None:
        return None
    start_token_ids = model.tokenizer.convert_tokens_to_ids(
        input_builder.PATH_START_TOKEN
    )
    end_token_ids = model.tokenizer.convert_tokens_to_ids(input_builder.PATH_END_TOKEN)
    input = model.prepare_model_prompt(input_query)
    prediction = model.generate_sentence(
        input,
        trie,
        start_token_ids=start_token_ids,
        end_token_ids=end_token_ids,
        enable_constrained_by_default=False,
    )
    if prediction is None:
        return None
    result = {
        "id": id,
        "question": question,
        "prediction": prediction,
        "ground_truth": answer,
        "ground_truth_paths": ground_paths,
        "input": input,
    }
    return result


def run_worker(args, LLM):
    """
    Worker function that runs on a specific GPU (called via subprocess).
    CUDA_VISIBLE_DEVICES is set in the environment before this process starts.
    """
    gpu_id = args.worker_gpu
    start_idx = args.worker_start_idx
    end_idx = args.worker_end_idx
    output_file = args.worker_output_file

    print(f"[GPU {gpu_id}] Starting worker process for indices {start_idx} to {end_idx}")

    # Load dataset
    input_file = os.path.join(args.data_path, args.d)
    full_dataset = load_dataset(input_file, split=args.split)

    # Handle rule merging if needed
    if args.add_rule:
        rule_dataset = load_jsonl(args.rule_path)
        full_dataset = merge_rule_result(full_dataset, rule_dataset, 1, args.filter_empty)

    # Get subset for this worker
    dataset_indices = list(range(start_idx, end_idx))
    dataset_subset = full_dataset.select(dataset_indices)

    print(f"[GPU {gpu_id}] Processing {len(dataset_subset)} samples")

    # Initialize model on this GPU
    model = LLM(args)
    print(f"[GPU {gpu_id}] Loading model...")
    model.prepare_for_inference()

    # Build input builder
    input_builder = PathGenerationWithAnswerPromptBuilder(
        model.tokenizer,
        args.prompt_mode,
        index_path_length=args.index_path_length,
        undirected=args.undirected,
        add_rule=args.add_rule
    )

    # Process data
    results = []
    for data in tqdm(dataset_subset, desc=f"GPU {gpu_id}"):
        res = prediction(data, [], input_builder, model)
        if res is not None:
            results.append(res)
            if args.debug:
                print(f"[GPU {gpu_id}] {json.dumps(res)}")
        else:
            print(f"[GPU {gpu_id}] None result for: {data['id']}")

    # Write results to temporary file
    with open(output_file, "w") as f:
        for res in results:
            f.write(json.dumps(res) + "\n")

    print(f"[GPU {gpu_id}] Finished processing {len(results)} samples, saved to {output_file}")


def main_multigpu(args, LLM):
    """
    Main function for multi-GPU execution.
    Splits dataset across GPUs and launches separate subprocesses.
    """
    # Parse GPU IDs
    gpu_ids = [int(x.strip()) for x in args.gpu_id.split(",")]
    num_gpus = len(gpu_ids)

    print(f"Using GPUs: {gpu_ids}")

    # Setup output directory
    input_file = os.path.join(args.data_path, args.d)
    post_fix = f"{args.prefix}{args.prompt_mode}-{args.generation_mode}-k{args.k}-index_len{args.index_path_length}"

    if args.add_rule:
        rule_postfix = args.rule_path.replace("/", "_").replace(".", "_")
        post_fix += "_" + rule_postfix

    data_name = args.d + "_undirected" if args.undirected else args.d
    output_dir = os.path.join(args.predict_path, data_name, args.model_name, args.split, post_fix)
    print("Save results to:", output_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save args
    with open(os.path.join(output_dir, 'args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    # Load dataset to get total size
    full_dataset = load_dataset(input_file, split=args.split)
    total_samples = len(full_dataset)
    print(f"Total samples: {total_samples}")

    # Handle rule merging if needed
    if args.add_rule:
        rule_dataset = load_jsonl(args.rule_path)
        full_dataset = merge_rule_result(full_dataset, rule_dataset, 1, args.filter_empty)
        total_samples = len(full_dataset)

    # Split dataset indices across GPUs
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

    # Create temporary output files for each GPU
    temp_output_files = [
        os.path.join(output_dir, f'predictions_gpu{gpu_id}.jsonl.tmp')
        for gpu_id in gpu_ids
    ]

    # Build base command arguments
    base_cmd = [
        sys.executable, __file__,
        "--worker_mode",
        "--data_path", args.data_path,
        "--d", args.d,
        "--split", args.split,
        "--index_path_length", str(args.index_path_length),
        "--predict_path", args.predict_path,
        "--model_name", args.model_name,
        "--model_path", args.model_path,
        "--k", str(args.k),
        "--prompt_mode", args.prompt_mode,
        "--generation_mode", args.generation_mode,
        "--attn_implementation", args.attn_implementation,
        "--dtype", args.dtype,
        "--max_new_tokens", str(args.max_new_tokens),
        "--maximum_token", str(args.maximum_token),
    ]
    if args.undirected:
        base_cmd.append("--undirected")
        base_cmd.append("true")
    if args.debug:
        base_cmd.append("--debug")
    if args.add_rule:
        base_cmd.append("--add_rule")
        base_cmd.extend(["--rule_path", args.rule_path])
    if args.filter_empty:
        base_cmd.append("--filter_empty")

    # Start worker subprocesses
    processes = []
    for i, gpu_id in enumerate(gpu_ids):
        start_idx, end_idx = gpu_ranges[i]
        worker_cmd = base_cmd + [
            "--worker_gpu", str(gpu_id),
            "--worker_start_idx", str(start_idx),
            "--worker_end_idx", str(end_idx),
            "--worker_output_file", temp_output_files[i],
        ]

        # Set CUDA_VISIBLE_DEVICES in the subprocess environment
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        print(f"Launching worker for GPU {gpu_id}...")
        p = subprocess.Popen(worker_cmd, env=env)
        processes.append(p)

    # Wait for all processes to complete
    print("Waiting for all workers to complete...")
    for p in processes:
        p.wait()

    print("All GPU workers finished. Merging results...")

    # Merge results from all temporary files
    final_output_file = os.path.join(output_dir, 'predictions.jsonl')

    # Check if we should force overwrite or append
    if args.force and os.path.exists(final_output_file):
        os.remove(final_output_file)

    # Merge all temporary files into final output
    all_results = []
    for temp_file in temp_output_files:
        if os.path.exists(temp_file):
            with open(temp_file, 'r') as f:
                for line in f:
                    all_results.append(json.loads(line))
            # Clean up temporary file
            os.remove(temp_file)

    # Sort by ID to maintain order
    all_results.sort(key=lambda x: x['id'])

    # Write merged results
    with open(final_output_file, 'w') as f:
        for res in all_results:
            f.write(json.dumps(res) + "\n")

    print(f"Merged {len(all_results)} results to {final_output_file}")

    # Run evaluation
    eval_path_result_w_ans(final_output_file)


def main_single_gpu(args, LLM):
    """
    Single GPU execution (same as original reasoning_trie.py behavior).
    """
    input_file = os.path.join(args.data_path, args.d)
    dataset = load_dataset(input_file, split=args.split)
    post_fix = f"{args.prefix}{args.prompt_mode}-{args.generation_mode}-k{args.k}-index_len{args.index_path_length}"

    if args.add_rule:
        rule_postfix = args.rule_path.replace("/", "_").replace(".", "_")
        rule_dataset = load_jsonl(args.rule_path)
        dataset = merge_rule_result(dataset, rule_dataset, 1, args.filter_empty)
        post_fix += "_" + rule_postfix

    data_name = args.d + "_undirected" if args.undirected else args.d
    output_dir = os.path.join(args.predict_path, data_name, args.model_name, args.split, post_fix)
    print("Save results to:", output_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model = LLM(args)

    print("Prepare pipeline for inference...")
    model.prepare_for_inference()

    input_builder = PathGenerationWithAnswerPromptBuilder(
        model.tokenizer,
        args.prompt_mode,
        index_path_length=args.index_path_length,
        undirected=args.undirected,
        add_rule=args.add_rule
    )

    with open(os.path.join(output_dir, 'args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    fout, processed_list = get_output_file(os.path.join(output_dir, 'predictions.jsonl'), force=args.force)

    for data in tqdm(dataset):
        res = prediction(data, processed_list, input_builder, model)
        if res is not None:
            if args.debug:
                print(json.dumps(res))
            fout.write(json.dumps(res) + "\n")
            fout.flush()
        else:
            print("None result for:", data["id"])
    fout.close()

    eval_path_result_w_ans(os.path.join(output_dir, 'predictions.jsonl'))


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--data_path', type=str, default='rmanluo')
    argparser.add_argument('--d', '-d', type=str, default='RoG-webqsp')
    argparser.add_argument('--split', type=str, default='test[:100]')
    argparser.add_argument('--index_path_length', type=int, default=2)
    argparser.add_argument('--predict_path', type=str, default='results/GenPaths')
    argparser.add_argument('--model_name', type=str, help="model_name for save results", default='save_models/FT-Qwen3-8B')
    argparser.add_argument('--force', action='store_true', help="force to overwrite the results")
    argparser.add_argument("--gpu_id", type=str, default="0", help="GPU ID(s) to use, e.g., '0' for single GPU or '0,1,2' for multi-GPU")
    argparser.add_argument("--undirected", type=lambda x: (str(x).lower() == 'true'), default=False)
    argparser.add_argument("--debug", action="store_true", help="print debug information")
    argparser.add_argument("--prompt_mode", type=str, default="zero-shot", choices=["zero-shot", "mcq-zero-shot", "few-shot"])
    argparser.add_argument("--filter_empty", action="store_true")
    argparser.add_argument("--add_rule", action="store_true")
    argparser.add_argument(
        "--rule_path",
        type=str,
        default="results/gen_rule_path/webqsp_undirected/Llama-2-7b-chat-hf_align-spectoken-joint/test/predictions_3_False.jsonl",
    )
    argparser.add_argument("--prefix", type=str, default="")

    # Worker mode arguments (used internally for subprocess workers)
    argparser.add_argument("--worker_mode", action="store_true", help="Internal: run as worker process")
    argparser.add_argument("--worker_gpu", type=int, default=0, help="Internal: GPU ID for this worker")
    argparser.add_argument("--worker_start_idx", type=int, default=0, help="Internal: start index")
    argparser.add_argument("--worker_end_idx", type=int, default=0, help="Internal: end index")
    argparser.add_argument("--worker_output_file", type=str, default="", help="Internal: output file")

    args, _ = argparser.parse_known_args()

    LLM = GraphConstrainedDecodingModel
    LLM.add_args(argparser)

    args = argparser.parse_args()

    if args.worker_mode:
        # Running as a worker subprocess - data range is automatically assigned
        print(f"Worker mode: GPU {args.worker_gpu}, samples {args.worker_start_idx} to {args.worker_end_idx}")
        run_worker(args, LLM)
    else:
        # Determine single vs multi-GPU mode based on gpu_id argument
        gpu_ids = [x.strip() for x in args.gpu_id.split(",")]

        if len(gpu_ids) == 1:
            # Single GPU mode
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids[0]
            print(f"Running in single GPU mode on GPU {gpu_ids[0]}")
            main_single_gpu(args, LLM)
        else:
            # Multi-GPU mode - data is automatically split across GPUs
            print(f"Running in multi-GPU mode on GPUs {gpu_ids}")
            main_multigpu(args, LLM)
