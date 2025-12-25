from collections import OrderedDict
import json
import re
import string
from sklearn.metrics import precision_score
from statistics import mean
import marisa_trie
import networkx as nx
from typing import Dict, List
from collections import deque

def get_truth_paths(q_entity: list, a_entity: list, graph: nx.Graph) -> list:
    """
    Get shortest paths connecting question and answer entities.
    """
    # Select paths
    paths = []
    for h in q_entity:
        if h not in graph:
            continue
        for t in a_entity:
            if t not in graph:
                continue
            try:
                for p in nx.all_shortest_paths(graph, h, t):
                    paths.append(p)
            except:
                pass
    # Add relation to paths
    result_paths = []
    for p in paths:
        tmp = []
        for i in range(len(p) - 1):
            u = p[i]
            v = p[i + 1]
            tmp.append((u, graph[u][v]["relation"], v))
        result_paths.append(tmp)
    return result_paths

def normalize(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    s = s.lower()
    exclude = set(string.punctuation)
    s = "".join(char for char in s if char not in exclude)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    # remove <pad> token:
    s = re.sub(r"\b(<pad>)\b", " ", s)
    s = " ".join(s.split())
    return s


def is_freebase_mid(s: str) -> bool:
    """
    Check if a string is a Freebase Machine ID (MID) or similar identifier.

    Freebase and related KG identifiers typically follow patterns like:
    - m.xxx (e.g., m.012zbkk5, m.04nb7z0) - Freebase Machine IDs
    - g.xxx (e.g., g.125czvn3w) - Google Knowledge Graph IDs
    - n.xxx - Named entity IDs
    - /m/xxx - URI-style MIDs

    Args:
        s: The string to check

    Returns:
        True if the string appears to be a Freebase MID or similar ID, False otherwise
    """
    if not s:
        return False
    s = s.strip()

    # Pattern 1: Single letter prefix followed by dot and alphanumeric (m.xxx, g.xxx, n.xxx, etc.)
    if re.match(r'^[a-z]\.[0-9a-zA-Z_]+$', s):
        return True

    # Pattern 2: URI-style MIDs (/m/xxx, /g/xxx)
    if re.match(r'^/[a-z]/[0-9a-zA-Z_]+$', s):
        return True

    return False


def filter_invalid_answers(predictions: List[str]) -> List[str]:
    """
    Filter out reasoning paths that have invalid answers (Freebase MIDs).

    NOTE: This function is DEPRECATED. Use filter_mid_from_answers instead.
    
    The problem with filtering entire predictions is that:
    1. The reasoning path itself may contain valid answer entities
    2. eval_acc/eval_hit use " ".join(prediction) which matches against paths too
    3. Removing predictions reduces coverage even if paths were useful
    
    Args:
        predictions: List of prediction strings in format:
            "# Reasoning Path:\n...\n# Answer:\n<answer>"

    Returns:
        List of predictions with valid (non-MID) answers
    """
    valid_predictions = []
    for p in predictions:
        # Extract answer from prediction
        if "# Answer:\n" in p:
            ans = p.split("# Answer:\n")[-1].strip()
        elif "# Answer:" in p:
            ans = p.split("# Answer:")[-1].strip()
        else:
            # Can't extract answer, keep the prediction
            valid_predictions.append(p)
            continue

        # Check if answer is a Freebase MID - if NOT a MID, keep it
        if not is_freebase_mid(ans):
            valid_predictions.append(p)

    return valid_predictions


def replace_mid_answers_with_path_entity(predictions: List[str]) -> List[str]:
    """
    Replace invalid MID answers with the last valid entity from the path.
    
    When the answer is a Freebase MID (e.g., m.04nb7z0), we try to extract
    a meaningful entity from the reasoning path instead.
    
    Args:
        predictions: List of prediction strings in format:
            "# Reasoning Path:\n...\n# Answer:\n<answer>"
    
    Returns:
        List of predictions with MID answers replaced by path entities
    """
    fixed_predictions = []
    for p in predictions:
        if "# Answer:\n" in p:
            parts = p.split("# Answer:\n")
            ans = parts[-1].strip()
            path_part = parts[0]
        elif "# Answer:" in p:
            parts = p.split("# Answer:")
            ans = parts[-1].strip()
            path_part = parts[0]
        else:
            fixed_predictions.append(p)
            continue
        
        # If answer is a valid entity, keep as is
        if not is_freebase_mid(ans):
            fixed_predictions.append(p)
            continue
        
        # Try to extract last entity from path
        # Path format: "# Reasoning Path:\nEntity1 -> rel1 -> Entity2 -> rel2 -> Entity3"
        if "# Reasoning Path:\n" in path_part:
            path_str = path_part.split("# Reasoning Path:\n")[-1].strip()
        else:
            path_str = path_part.strip()
        
        # Split by " -> " and get entities (odd indices are entities in: E -> R -> E -> R -> E)
        path_elements = [x.strip() for x in path_str.split(" -> ")]
        
        # Find the last non-MID entity in the path
        new_ans = ans  # default to original
        for elem in reversed(path_elements):
            if elem and not is_freebase_mid(elem):
                new_ans = elem
                break
        
        # Reconstruct prediction with new answer
        fixed_predictions.append(f"{path_part}# Answer:\n{new_ans}")
    
    return fixed_predictions


def match(s1: str, s2: str) -> bool:
    s1 = normalize(s1)
    s2 = normalize(s2)
    return s2 in s1


def eval_acc(prediction, answer):
    matched = 0.0
    for a in answer:
        if match(prediction, a):
            matched += 1
    return matched / len(answer)


def eval_hit(prediction, answer):
    for a in answer:
        if match(prediction, a):
            return 1
    return 0


def eval_f1(prediction, answer):
    if len(prediction) == 0 or len(answer) == 0:
        return 0, 0, 0
    ans_recalled = 0
    prediction_correct = 0
    prediction_str = " ".join(prediction)
    for a in answer:
        if match(prediction_str, a):
            ans_recalled += 1
    recall = ans_recalled / len(answer)
    for p in prediction:
        for a in answer:
            if match(p, a):
                prediction_correct += 1
                break
    precision = prediction_correct / len(prediction)
    if precision + recall == 0:
        return 0, precision, recall
    else:
        return (2 * precision * recall) / (precision + recall), precision, recall


def extract_topk_prediction(prediction, k=-1):
    if isinstance(prediction, str):
        prediction = prediction.split("\n")
    results = {}
    for p in prediction:  # 遍历每条预测出的推理路径
        if p.strip() == "":
            continue
        if p in results:
            results[p] += 1
        else:
            results[p] = 1
    if k > len(results) or k < 0:
        k = len(results)
    results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    return [r[0] for r in results[:k]]

def eval_rank_results(predict_file, topk=[1, 3, 5, 10]):
    # predict_file = os.path.join(result_path, 'predictions.jsonl')
    eval_name = (
        f"detailed_eval_result_top_{topk}.jsonl"
        if topk
        else "detailed_eval_result.jsonl"
    )
    detailed_eval_file = predict_file.replace("predictions.jsonl", eval_name)
    all_acc_list = OrderedDict({k: [] for k in topk})
    all_hit_list = OrderedDict({k: [] for k in topk})
    all_f1_list = OrderedDict({k: [] for k in topk})
    all_precission_list = OrderedDict({k: [] for k in topk})
    all_recall_list = OrderedDict({k: [] for k in topk})
    
    with open(predict_file, "r") as f, open(detailed_eval_file, "w") as f2:
        for line in f:
            try:
                data = json.loads(line)
            except:
                print(line)
                continue
            id = data["id"]
            answer = list(set(data["answer"]))
            acc_list = OrderedDict()
            hit_list = OrderedDict()
            f1_list = OrderedDict()
            precission_list = OrderedDict()
            recall_list = OrderedDict()
            for k in topk:
                top_k_pred = min(k, len(data['ranks']))
                topk_rank = data['ranks'][:top_k_pred]
                prediction = [r['response'] for r in topk_rank]
                f1_score, precision_score, recall_score = eval_f1(prediction, answer)
                prediction_str = " ".join(prediction)
                acc = eval_acc(prediction_str, answer)
                hit = eval_hit(prediction_str, answer)
                acc_list[k] = acc
                hit_list[k] = hit
                f1_list[k]= f1_score
                precission_list[k] = precision_score
                recall_list[k] = recall_score
            f2.write(
                    json.dumps(
                        {
                            "id": id,
                            "prediction": prediction,
                            "ground_truth": answer,
                            "acc@k": acc_list,
                            "hit@k": hit_list,
                            "f1@k": f1_list,
                            "precission@k": precission_list,
                            "recall@k": recall_list,
                        }
                    )
                    + "\n"
                )
            for k in topk:
                all_acc_list[k].append(acc_list[k])
                all_hit_list[k].append(hit_list[k])
                all_f1_list[k].append(f1_list[k])
                all_precission_list[k].append(precission_list[k])
                all_recall_list[k].append(recall_list[k])
    result_str = ""
    for k in topk:
        result_str += f"Top-{k}:\n"
        result_str += (
            "Accuracy: "
            + str(sum(all_acc_list[k]) * 100 / len(all_acc_list[k]))
            + " Hit: "
            + str(sum(all_hit_list[k]) * 100 / len(all_hit_list[k]))
            + " F1: "
            + str(sum(all_f1_list[k]) * 100 / len(all_f1_list[k]))
            + " Precision: "
            + str(sum(all_precission_list[k]) * 100 / len(all_precission_list[k]))
            + " Recall: "
            + str(sum(all_recall_list[k]) * 100 / len(all_recall_list[k]))
            + "\n"
        )
    print(result_str)
    result_name = f"eval_result_top_{topk}.txt" if topk else "eval_result.txt"
    eval_result_path = predict_file.replace("predictions.jsonl", result_name)
    with open(eval_result_path, "w") as f:
        f.write(result_str)
                
    
def eval_result(predict_file, cal_f1=True, topk=-1):
    # predict_file = os.path.join(result_path, 'predictions.jsonl')
    eval_name = (
        "detailed_eval_result_top_{topk}.jsonl"
        if topk > 0
        else "detailed_eval_result.jsonl"
    )
    detailed_eval_file = predict_file.replace("predictions.jsonl", eval_name)
    # Load results
    acc_list = []
    hit_list = []
    f1_list = []
    precission_list = []
    recall_list = []
    with open(predict_file, "r") as f, open(detailed_eval_file, "w") as f2:
        for line in f:
            try:
                data = json.loads(line)
            except:
                print(line)
                continue
            id = data["id"]
            prediction = data["prediction"]
            answer = list(set(data["ground_truth"]))
            if cal_f1:
                prediction = extract_topk_prediction(prediction, topk)
                f1_score, precision_score, recall_score = eval_f1(prediction, answer)
                f1_list.append(f1_score)
                precission_list.append(precision_score)
                recall_list.append(recall_score)
                prediction_str = " ".join(prediction)
                acc = eval_acc(prediction_str, answer)
                hit = eval_hit(prediction_str, answer)
                acc_list.append(acc)
                hit_list.append(hit)
                f2.write(
                    json.dumps(
                        {
                            "id": id,
                            "prediction": prediction,
                            "ground_truth": answer,
                            "acc": acc,
                            "hit": hit,
                            "f1": f1_score,
                            "precission": precision_score,
                            "recall": recall_score,
                        }
                    )
                    + "\n"
                )
            else:
                acc = eval_acc(prediction, answer)
                hit = eval_hit(prediction, answer)
                acc_list.append(acc)
                hit_list.append(hit)
                f2.write(
                    json.dumps(
                        {
                            "id": id,
                            "prediction": prediction,
                            "ground_truth": answer,
                            "acc": acc,
                            "hit": hit,
                        }
                    )
                    + "\n"
                )

    if len(f1_list) > 0:
        result_str = (
            "Accuracy: "
            + str(sum(acc_list) * 100 / len(acc_list))
            + " Hit: "
            + str(sum(hit_list) * 100 / len(hit_list))
            + " F1: "
            + str(sum(f1_list) * 100 / len(f1_list))
            + " Precision: "
            + str(sum(precission_list) * 100 / len(precission_list))
            + " Recall: "
            + str(sum(recall_list) * 100 / len(recall_list))
        )
    else:
        result_str = (
            "Accuracy: "
            + str(sum(acc_list) * 100 / len(acc_list))
            + " Hit: "
            + str(sum(hit_list) * 100 / len(hit_list))
        )
    print(result_str)
    result_name = "eval_result_top_{topk}.txt" if topk > 0 else "eval_result.txt"
    eval_result_path = predict_file.replace("predictions.jsonl", result_name)
    with open(eval_result_path, "w") as f:
        f.write(result_str)



def eval_joint_result(predict_file):
    # predict_file = os.path.join(result_path, 'predictions.jsonl')
    eval_name = "detailed_eval_result.jsonl"
    detailed_eval_file = predict_file.replace("predictions.jsonl", eval_name)
    # Load results
    acc_list = []
    hit_list = []
    f1_list = []
    precission_list = []
    recall_list = []
    path_f1_list = []
    path_precission_list = []
    path_recall_list = []
    path_ans_f1_list = []
    path_ans_recall_list = []
    path_ans_precision_list = []
    with open(predict_file, "r") as f, open(detailed_eval_file, "w") as f2:
        for line in f:
            try:
                data = json.loads(line)
            except:
                print(line)
                continue
            id = data["id"]
            prediction = data["prediction"]
            answer = list(set(data["ground_truth"]))
            # Extract reasoning paths and answers
            predicted_reasoning_paths = set()
            predicted_answers = set()

            for pre in prediction:
                try:
                    ans_in_pred = False
                    if "the answer is: " in pre:
                        ans_in_pred = True
                        ans_pred = pre.split("the answer is: ")[1]
                        for ans in ans_pred.split("\n"):
                            predicted_answers.add(ans.strip())
                    if "Reasoning path:\n" in pre:
                        if ans_in_pred:
                            path_pred = pre.split("Reasoning path:\n")[1].split("\nthe answer is: ")[0]
                        else:
                            path_pred = pre.split("Reasoning path:\n")[1]
                        for path in path_pred.split("\n")[:-1]:
                            predicted_reasoning_paths.add(path.strip())
                except Exception as e:
                    print("Error in line: ", pre)
                    print(e)
                    continue
            predicted_reasoning_paths = list(predicted_reasoning_paths)
            predicted_answers = list(predicted_answers)
            
            f1_score, precision_score, recall_score = eval_f1(predicted_answers, answer)
            f1_list.append(f1_score)
            precission_list.append(precision_score)
            recall_list.append(recall_score)
            prediction_str = " ".join(predicted_answers)
            acc = eval_acc(prediction_str, answer)
            hit = eval_hit(prediction_str, answer)
            acc_list.append(acc)
            hit_list.append(hit)
            path_f1_score, path_precision_score, path_recall_score = eval_f1(
                predicted_reasoning_paths, data["ground_truth_paths"]
            )
            path_f1_list.append(path_f1_score)
            path_precission_list.append(path_precision_score)
            path_recall_list.append(path_recall_score)
            path_ans_f1_score, path_ans_precision_score, path_ans_recall_score = eval_f1(predicted_reasoning_paths, answer)
            path_ans_f1_list.append(path_ans_f1_score)
            path_ans_precision_list.append(path_ans_precision_score)
            path_ans_recall_list.append(path_ans_recall_score)
            f2.write(
                json.dumps(
                    {
                        "id": id,
                        "prediction": prediction,
                        "ground_truth": answer,
                        "ans_acc": acc,
                        "ans_hit": hit,
                        "ans_f1": f1_score,
                        "ans_precission": precision_score,
                        "ans_recall": recall_score,
                        "path_f1": path_f1_score,
                        "path_precision": path_precision_score,
                        "path_recall": path_recall_score,
                        "path_ans_f1": path_ans_f1_score,
                        "path_ans_precision": path_ans_precision_score,
                        "path_ans_recall": path_ans_recall_score
                    }
                )
                + "\n"
            )


    if len(f1_list) > 0:
        result_str = f"Accuracy: {sum(acc_list) * 100 / len(acc_list)} Hit: {sum(hit_list) * 100 / len(hit_list)} F1: {sum(f1_list) * 100 / len(f1_list)} Precision: {sum(precission_list) * 100 / len(precission_list)} Recall: {sum(recall_list) * 100 / len(recall_list)} Path F1: {sum(path_f1_list) * 100 / len(path_f1_list)} Path Precision: {sum(path_precission_list) * 100 / len(path_precission_list)} Path Recall: {sum(path_recall_list) * 100 / len(path_recall_list)} Path Ans F1: {mean(path_ans_f1_list)} Path Ans Precision: {mean(path_ans_precision_list)} Path Ans recall: {mean(path_ans_recall_list)}"
    else:
        result_str = (
            "Accuracy: "
            + str(sum(acc_list) * 100 / len(acc_list))
            + " Hit: "
            + str(sum(hit_list) * 100 / len(hit_list))
        )
    print(result_str)
    result_name = "eval_result.txt"
    eval_result_path = predict_file.replace("predictions.jsonl", result_name)
    with open(eval_result_path, "w") as f:
        f.write(result_str)

def eval_path_result(predict_file, cal_f1=True, topk=-1):
    # predict_file = os.path.join(result_path, 'predictions.jsonl')
    eval_name = (
        "detailed_eval_result_top_{topk}.jsonl"
        if topk > 0
        else "detailed_eval_result.jsonl"
    )
    detailed_eval_file = predict_file.replace("predictions.jsonl", eval_name)
    # Load results
    acc_list = []
    hit_list = []
    f1_list = []
    precission_list = []
    recall_list = []
    path_f1_list = []
    path_precission_list = []
    path_recall_list = []
    with open(predict_file, "r") as f, open(detailed_eval_file, "w") as f2:
        for line in f:
            try:
                data = json.loads(line)
            except:
                print(line)
                continue
            id = data["id"]
            prediction = data["prediction"]
            answer = list(set(data["ground_truth"]))
            
            if len(data["ground_truth_paths"]) == 0 or len(answer) == 0:
                continue
            
            if cal_f1:
                prediction = extract_topk_prediction(prediction, topk)
                f1_score, precision_score, recall_score = eval_f1(prediction, answer)
                f1_list.append(f1_score)
                precission_list.append(precision_score)
                recall_list.append(recall_score)
                prediction_str = " ".join(prediction)
                acc = eval_acc(prediction_str, answer)
                hit = eval_hit(prediction_str, answer)
                acc_list.append(acc)
                hit_list.append(hit)
                path_f1_score, path_precision_score, path_recall_score = eval_f1(
                    prediction, data["ground_truth_paths"]
                )
                path_f1_list.append(path_f1_score)
                path_precission_list.append(path_precision_score)
                path_recall_list.append(path_recall_score)
                f2.write(
                    json.dumps(
                        {
                            "id": id,
                            "prediction": prediction,
                            "ground_truth": answer,
                            "ans_acc": acc,
                            "ans_hit": hit,
                            "ans_f1": f1_score,
                            "ans_precission": precision_score,
                            "ans_recall": recall_score,
                            "path_f1": path_f1_score,
                            "path_precision": path_precision_score,
                            "path_recall": path_recall_score,
                        }
                    )
                    + "\n"
                )
            else:
                acc = eval_acc(prediction, answer)
                hit = eval_hit(prediction, answer)
                acc_list.append(acc)
                hit_list.append(hit)
                f2.write(
                    json.dumps(
                        {
                            "id": id,
                            "prediction": prediction,
                            "ground_truth": answer,
                            "acc": acc,
                            "hit": hit,
                        }
                    )
                    + "\n"
                )

    if len(f1_list) > 0:
        result_str = f"Accuracy: {sum(acc_list) * 100 / len(acc_list)} Hit: {sum(hit_list) * 100 / len(hit_list)} F1: {sum(f1_list) * 100 / len(f1_list)} Precision: {sum(precission_list) * 100 / len(precission_list)} Recall: {sum(recall_list) * 100 / len(recall_list)} Path F1: {sum(path_f1_list) * 100 / len(path_f1_list)} Path Precision: {sum(path_precission_list) * 100 / len(path_precission_list)} Path Recall: {sum(path_recall_list) * 100 / len(path_recall_list)}"
    else:
        result_str = (
            "Accuracy: "
            + str(sum(acc_list) * 100 / len(acc_list))
            + " Hit: "
            + str(sum(hit_list) * 100 / len(hit_list))
        )
    print(result_str)
    result_name = "eval_result_top_{topk}.txt" if topk > 0 else "eval_result.txt"
    eval_result_path = predict_file.replace("predictions.jsonl", result_name)
    with open(eval_result_path, "w") as f:
        f.write(result_str)


def eval_path_answer(predict_file, cal_f1=True, min_topk=5):
    """
    Evaluate reasoning paths with adaptive topk based on number of ground truth answers.

    For questions with 1-2 ground truth answers: use top min_topk (default 3) reasoning paths
    For questions with K answers (K > 2): use top K reasoning paths

    Args:
        predict_file: Path to predictions.jsonl file
        cal_f1: Whether to calculate F1 score
        min_topk: Minimum number of paths to consider (default 3)
    """
    eval_name = "detailed_eval_result_adaptive.jsonl"
    detailed_eval_file = predict_file.replace("predictions.jsonl", eval_name)

    # Load results
    acc_list = []
    hit_list = []
    f1_list = []
    precission_list = []
    recall_list = []
    path_ans_f1_list = []
    path_ans_precission_list = []
    path_ans_recall_list = []
    path_f1_list = []
    path_precission_list = []
    path_recall_list = []

    with open(predict_file, "r") as f, open(detailed_eval_file, "w") as f2:
        for line in f:
            try:
                data = json.loads(line)
            except:
                print(line)
                continue
            id = data["id"]
            prediction = data["prediction"]  # 表示一个问题对应的10条推理路径
            answer = list(set(data["ground_truth"]))  # 表示一个问题对应的ground-truth answers

            if cal_f1:
                # Adaptive topk: use top min_topk paths for 1-2 answers, top K paths for K answers (K > 2)
                num_answers = len(answer)
                effective_topk = min_topk if num_answers <= 2 else num_answers

                prediction = extract_topk_prediction(prediction, effective_topk)

                predicted_path = []
                predicted_ans = []
                for p in prediction:
                    ans = p.split("# Answer:\n")[-1]
                    path = p.split("# Answer:\n")[0].split("# Reasoning Path:\n")[-1]
                    predicted_path.append(path.strip())
                    predicted_ans.append(ans.strip())

                f1_score, precision_score, recall_score = eval_f1(predicted_ans, answer)
                path_ans_f1_score, path_ans_precision_score, path_ans_recall_score = eval_f1(predicted_path, answer)
                path_ans_f1_list.append(path_ans_f1_score)
                path_ans_precission_list.append(path_ans_precision_score)
                path_ans_recall_list.append(path_ans_recall_score)
                f1_list.append(f1_score)
                precission_list.append(precision_score)
                recall_list.append(recall_score)
                prediction_str = " ".join(prediction)
                acc = eval_acc(prediction_str, answer)
                hit = eval_hit(prediction_str, answer)
                acc_list.append(acc)
                hit_list.append(hit)
                path_f1_score, path_precision_score, path_recall_score = eval_f1(
                    predicted_path, data["ground_truth_paths"]
                )
                path_f1_list.append(path_f1_score)
                path_precission_list.append(path_precision_score)
                path_recall_list.append(path_recall_score)
                f2.write(
                    json.dumps(
                        {
                            "id": id,
                            "prediction": prediction,
                            "ground_truth": answer,
                            "effective_topk": effective_topk,
                            "ans_acc": acc,
                            "ans_hit": hit,
                            "ans_f1": f1_score,
                            "ans_precission": precision_score,
                            "ans_recall": recall_score,
                            "path_f1": path_f1_score,
                            "path_precision": path_precision_score,
                            "path_recall": path_recall_score,
                            "path_ans_f1": path_ans_f1_score,
                            "path_ans_precision": path_ans_precision_score,
                            "path_ans_recall": path_ans_recall_score,
                        }
                    )
                    + "\n"
                )
            else:
                acc = eval_acc(prediction, answer)
                hit = eval_hit(prediction, answer)
                acc_list.append(acc)
                hit_list.append(hit)
                f2.write(
                    json.dumps(
                        {
                            "id": id,
                            "prediction": prediction,
                            "ground_truth": answer,
                            "acc": acc,
                            "hit": hit,
                        }
                    )
                    + "\n"
                )

    if len(f1_list) > 0:
        result_str = f"Accuracy: {sum(acc_list) * 100 / len(acc_list)} Hit: {sum(hit_list) * 100 / len(hit_list)} F1: {sum(f1_list) * 100 / len(f1_list)} Precision: {sum(precission_list) * 100 / len(precission_list)} Recall: {sum(recall_list) * 100 / len(recall_list)} Path F1: {sum(path_f1_list) * 100 / len(path_f1_list)} Path Precision: {sum(path_precission_list) * 100 / len(path_precission_list)} Path Recall: {sum(path_recall_list) * 100 / len(path_recall_list)} Path Answer F1: {sum(path_ans_f1_list) * 100 / len(path_ans_f1_list)} Path Answer Precision: {sum(path_ans_precission_list) * 100 / len(path_ans_precission_list)} Path Answer Recall: {sum(path_ans_recall_list) * 100 / len(path_ans_recall_list)}"
    else:
        result_str = (
            "Accuracy: "
            + str(sum(acc_list) * 100 / len(acc_list))
            + " Hit: "
            + str(sum(hit_list) * 100 / len(hit_list))
        )
    print(result_str)
    result_name = "eval_result_adaptive.txt"
    eval_result_path = predict_file.replace("predictions.jsonl", result_name)
    with open(eval_result_path, "w") as f:
        f.write(result_str)


def eval_path_result_w_ans(predict_file, cal_f1=True, topk=-1):
    # predict_file = os.path.join(result_path, 'predictions.jsonl')
    eval_name = (
        "detailed_eval_result_top_{topk}.jsonl"
        if topk > 0
        else "detailed_eval_result.jsonl"
    )
    detailed_eval_file = predict_file.replace("predictions.jsonl", eval_name)   # 保存了每个问题的详细评估结果 包括10条推理路径，ground truth paths, 和ground-truth answers
    # Load results
    acc_list = []
    hit_list = []
    f1_list = []
    precission_list = []
    recall_list = []
    path_ans_f1_list = []
    path_ans_precission_list = []
    path_ans_recall_list = []
    path_f1_list = []
    path_precission_list = []
    path_recall_list = []
    with open(predict_file, "r") as f, open(detailed_eval_file, "w") as f2:
        for line in f:
            try:
                data = json.loads(line)
            except:
                print(line)
                continue
            id = data["id"] 
            prediction = data["prediction"]  # 表示一个问题对应的10条推理路径
            answer = list(set(data["ground_truth"]))  # 表示一个问题对应的ground-truth answers
            if cal_f1:
                prediction = extract_topk_prediction(prediction, topk)
                
                predicted_path = []
                predicted_ans = []
                for p in prediction:
                    ans = p.split("# Answer:\n")[-1]
                    path = p.split("# Answer:\n")[0].split("# Reasoning Path:\n")[-1]
                    predicted_path.append(path.strip())
                    predicted_ans.append(ans.strip())
                
                f1_score, precision_score, recall_score = eval_f1(predicted_ans, answer)
                path_ans_f1_score, path_ans_precision_score, path_ans_recall_score = eval_f1(predicted_path, answer)
                path_ans_f1_list.append(path_ans_f1_score)
                path_ans_precission_list.append(path_ans_precision_score)
                path_ans_recall_list.append(path_ans_recall_score)
                f1_list.append(f1_score)
                precission_list.append(precision_score)
                recall_list.append(recall_score)
                prediction_str = " ".join(prediction)
                acc = eval_acc(prediction_str, answer)
                hit = eval_hit(prediction_str, answer)
                acc_list.append(acc)
                hit_list.append(hit)
                path_f1_score, path_precision_score, path_recall_score = eval_f1(
                    predicted_path, data["ground_truth_paths"]
                )
                path_f1_list.append(path_f1_score)
                path_precission_list.append(path_precision_score)
                path_recall_list.append(path_recall_score)
                f2.write(
                    json.dumps(
                        {
                            "id": id,
                            "prediction": prediction,
                            "ground_truth": answer,
                            "ans_acc": acc,
                            "ans_hit": hit,
                            "ans_f1": f1_score,
                            "ans_precission": precision_score,
                            "ans_recall": recall_score,
                            "path_f1": path_f1_score,
                            "path_precision": path_precision_score,
                            "path_recall": path_recall_score,
                            "path_ans_f1": path_ans_f1_score,
                            "path_ans_precision": path_ans_precision_score,
                            "path_ans_recall": path_ans_recall_score,
                        }
                    )
                    + "\n"
                )
            else:
                acc = eval_acc(prediction, answer)
                hit = eval_hit(prediction, answer)
                acc_list.append(acc)
                hit_list.append(hit)
                f2.write(
                    json.dumps(
                        {
                            "id": id,
                            "prediction": prediction,
                            "ground_truth": answer,
                            "acc": acc,
                            "hit": hit,
                        }
                    )
                    + "\n"
                )

    if len(f1_list) > 0:
        result_str = f"Accuracy: {sum(acc_list) * 100 / len(acc_list)} Hit: {sum(hit_list) * 100 / len(hit_list)} F1: {sum(f1_list) * 100 / len(f1_list)} Precision: {sum(precission_list) * 100 / len(precission_list)} Recall: {sum(recall_list) * 100 / len(recall_list)} Path F1: {sum(path_f1_list) * 100 / len(path_f1_list)} Path Precision: {sum(path_precission_list) * 100 / len(path_precission_list)} Path Recall: {sum(path_recall_list) * 100 / len(path_recall_list)} Path Answer F1: {sum(path_ans_f1_list) * 100 / len(path_ans_f1_list)} Path Answer Precision: {sum(path_ans_precission_list) * 100 / len(path_ans_precission_list)} Path Answer Recall: {sum(path_ans_recall_list) * 100 / len(path_ans_recall_list)}"
    else:
        result_str = (
            "Accuracy: "
            + str(sum(acc_list) * 100 / len(acc_list))
            + " Hit: "
            + str(sum(hit_list) * 100 / len(hit_list))
        )
    print(result_str)
    result_name = "eval_result_top_{topk}.txt" if topk > 0 else "eval_result.txt"
    eval_result_path = predict_file.replace("predictions.jsonl", result_name)
    with open(eval_result_path, "w") as f:
        f.write(result_str)



class Trie(object):
    def __init__(self, sequences: List[List[int]] = []):
        self.trie_dict = {}
        self.len = 0
        if sequences:
            for sequence in sequences:
                Trie._add_to_trie(sequence, self.trie_dict)
                self.len += 1

        self.append_trie = None
        self.bos_token_id = None

    def append(self, trie, bos_token_id):
        self.append_trie = trie
        self.bos_token_id = bos_token_id

    def add(self, sequence: List[int]):
        Trie._add_to_trie(sequence, self.trie_dict)
        self.len += 1

    def get(self, prefix_sequence: List[int]):
        return Trie._get_from_trie(
            prefix_sequence, self.trie_dict, self.append_trie, self.bos_token_id
        )

    @staticmethod
    def load_from_dict(trie_dict):
        trie = Trie()
        trie.trie_dict = trie_dict
        trie.len = sum(1 for _ in trie)
        return trie

    @staticmethod
    def _add_to_trie(sequence: List[int], trie_dict: Dict):
        if sequence:
            if sequence[0] not in trie_dict:
                trie_dict[sequence[0]] = {}
            Trie._add_to_trie(sequence[1:], trie_dict[sequence[0]])

    @staticmethod
    def _get_from_trie(
        prefix_sequence: List[int],
        trie_dict: Dict,
        append_trie=None,
        bos_token_id: int = None,
    ):

        if len(prefix_sequence) == 0:
            output = list(trie_dict.keys())
            if append_trie and bos_token_id in output:
                output.remove(bos_token_id)
                output += list(append_trie.trie_dict.keys())
            return output
        elif prefix_sequence[0] in trie_dict:
            return Trie._get_from_trie(
                prefix_sequence[1:],
                trie_dict[prefix_sequence[0]],
                append_trie,
                bos_token_id,
            )
        else:
            if append_trie:
                return append_trie.get(prefix_sequence)
            else:
                return []

    def __iter__(self):
        def _traverse(prefix_sequence, trie_dict):
            if trie_dict:
                for next_token in trie_dict:
                    yield from _traverse(
                        prefix_sequence + [next_token], trie_dict[next_token]
                    )
            else:
                yield prefix_sequence

        return _traverse([], self.trie_dict)

    def __len__(self):
        return self.len

    def __getitem__(self, value):
        return self.get(value)


class MarisaTrie(object):
    def __init__(
        self,
        sequences: List[List[int]] = [],  # 当前的一个测试问题的所有候选路径的token ids list形式，为问题entities的所有2条以内路径序列
        cache_fist_branch=True,
        max_token_id=256001,  # 128257 + 1
    ):

        # 前55000个id对应的unicode字符是连续的，中间跳过了一段区间，所以分两段生成字符列表
        # 用于将token id映射为Unicode 字符 比如 chr(10)='\n'   chr(97)='a'
        # 中间跳过 10000 个，是为了避开一整块可能引起编码/显示/兼容性问题的 Unicode 区段，同时预留出额外字符空间，让 marisa_trie 使用的“虚拟字母表”既连续又安全。
        self.int2char = [chr(i) for i in range(min(max_token_id, 55000))] + (
            [chr(i) for i in range(65000, max_token_id + 10000)]
            if max_token_id >= 55000
            else []
        ) 
        # 一个dict，用于将Unicode字符映射回token id，比如 {'\n':10, 'a':97}
        self.char2int = {self.int2char[i]: i for i in range(max_token_id)}

        self.cache_fist_branch = cache_fist_branch
        if self.cache_fist_branch:
            self.zero_iter = list({sequence[0] for sequence in sequences}) # 每个路径的第一个token id集合 表示<PATH>token

        # 构造Trie, 用所有路径的token ids对应的unicode字符序列来构造Trie，用来约束模型生成路径的合法性
        self.trie = marisa_trie.Trie(
            "".join([self.int2char[i] for i in sequence]) for sequence in sequences
        )

    def get(self, prefix_sequence: List[int]):
        if self.cache_fist_branch and len(prefix_sequence) == 0:
            return self.zero_iter
        else:
            key = "".join([self.int2char[i] for i in prefix_sequence])
            return list(
                {
                    self.char2int[e[len(key)]]
                    for e in self.trie.keys(key)
                    if len(e) > len(key)
                }
            )

    def __iter__(self):
        for sequence in self.trie.iterkeys():
            yield [self.char2int[e] for e in sequence]

    def __len__(self):
        return len(self.trie)

    def __getitem__(self, value):
        return self.get(value)


class DummyTrieMention(object):
    def __init__(self, return_values):
        self._return_values = return_values

    def get(self, indices=None):
        return self._return_values


class DummyTrieEntity(object):
    def __init__(self, return_values, codes):
        self._return_values = list(
            set(return_values).difference(
                set(
                    codes[e]
                    for e in (
                        "start_mention_token",
                        "end_mention_token",
                        "start_entity_token",
                    )
                )
            )
        )
        self._codes = codes

    def get(self, indices, depth=0):
        if len(indices) == 0 and depth == 0:
            return self._codes["end_mention_token"]
        elif len(indices) == 0 and depth == 1:
            return self._codes["start_entity_token"]
        elif len(indices) == 0:
            return self._return_values
        elif len(indices) == 1 and indices[0] == self._codes["end_entity_token"]:
            return self._codes["EOS"]
        else:
            return self.get(indices[1:], depth=depth + 1)


def dfs(graph, start_node_list, max_length):
    """
    Find all paths within max_length starting from start_node_list in graph using DFS.

    Args:
        graph (nx.DiGraph): Directed graph
        start_node (List[str]): A list of start nodes
        max_length (int): Maximum length of path = 2

    Returns:
        List[List[tuple]]: Find paths
    """
    def dfs_visit(node, path):
        if len(path) > max_length:
            return
        try:
            for neighbor in graph.neighbors(node): # 遍历节点的所有一阶邻居
                rel = graph[node][neighbor]["relation"] 
                new_path = path + [(node, rel, neighbor)]  # 保存路径添加到现有路径中
                if len(new_path) <= max_length: # 如果长度大于max_length, 那么新的关系不会倍加入path中
                    path_lists.add(tuple(new_path))
                dfs_visit(neighbor, new_path)
        except Exception as e:
            print(e)
            pass

    path_lists = set()
    for start_node in start_node_list: # 一个问题的所有entities
        dfs_visit(start_node, [])

    return list(path_lists) # 保留从question entities出发的所有2-hop以内的路径，每个元素是一条路径，包含了所有1条路径和2跳路径，当max_length=2时。

def bfs_with_rule(graph, start_node, target_rule, max_p=10):
    result_paths = []
    queue = deque([(start_node, [])])  # 使用队列存储待探索节点和对应路径
    while queue:
        current_node, current_path = queue.popleft()

        # 如果当前路径符合规则，将其添加到结果列表中
        if len(current_path) == len(target_rule):
            result_paths.append(current_path)
            # if len(result_paths) >= max_p:
            #     break

        # 如果当前路径长度小于规则长度，继续探索
        if len(current_path) < len(target_rule):
            if current_node not in graph:
                continue
            for neighbor in graph.neighbors(current_node):
                # 剪枝：如果当前边类型与规则中的对应位置不匹配，不继续探索该路径
                rel = graph[current_node][neighbor]["relation"]
                if rel != target_rule[len(current_path)] or len(current_path) > len(
                    target_rule
                ):
                    continue
                queue.append((neighbor, current_path + [(current_node, rel, neighbor)]))

    return result_paths
