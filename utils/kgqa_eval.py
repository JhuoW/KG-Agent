import json
import string
import re
from statistics import mean

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

def match(s1: str, s2: str) -> bool:
    s1 = normalize(s1)
    s2 = normalize(s2)
    return s2 in s1

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
            answer = list(set(data["gt_answer"]))  # 表示一个问题对应的ground-truth answers
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
                    predicted_path, data["gt_paths"]
                )
                path_f1_list.append(path_f1_score)
                path_precission_list.append(path_precision_score)
                path_recall_list.append(path_recall_score)
                f2.write(
                    json.dumps(
                        {
                            "id": id,
                            "prediction": prediction,
                            "gt_answer": answer,
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
                            "gt_answer": answer,
                            "acc": acc,
                            "hit": hit,
                        }
                    )
                    + "\n"
                )

    if len(f1_list) > 0:
        result_str = f"Accuracy: {sum(acc_list) * 100 / len(acc_list)} Hit: {sum(hit_list) * 100 / len(hit_list)} F1: {sum(f1_list) * 100 / len(f1_list)} Precision: {sum(precission_list) * 100 / len(precission_list)} Recall: {sum(recall_list) * 100 / len(recall_list)} Path F1: {sum(path_f1_list) * 100 / len(path_f1_list)} Path Precision: {sum(path_precission_list) * 100 / len(path_precission_list)} Path Recall: {sum(path_recall_list) * 100 / len(path_recall_list)} Path Answer F1: {sum(path_ans_f1_list) * 100 / len(path_ans_f1_list)} Path Answer Precision: {sum(path_ans_precission_list) * 100 / len(path_ans_precission_list)} Path Answer Recall: {sum(path_ans_recall_list) * 100 / len(path_ans_recall_list)}"
    elif len(acc_list) > 0:
        result_str = (
            "Accuracy: "
            + str(sum(acc_list) * 100 / len(acc_list))
            + " Hit: "
            + str(sum(hit_list) * 100 / len(hit_list))
        )
    else:
        result_str = "No valid predictions found. Accuracy: 0.0 Hit: 0.0"
    print(result_str)
    result_name = "eval_result_top_{topk}.txt" if topk > 0 else "eval_result.txt"
    eval_result_path = predict_file.replace("predictions.jsonl", result_name)
    with open(eval_result_path, "w") as f:
        f.write(result_str)