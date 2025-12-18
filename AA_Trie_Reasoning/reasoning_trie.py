"""
Accuracy: 68.86434383638422 
Hit: 80.0 
F1: 40.02177357451937 
Precision: 38.83888888888889 
Recall: 68.68847894675014 
Path F1: 46.822380092441406 
Path Precision: 46.83888888888889 
Path Recall: 68.0561566113839 
Path Answer F1: 48.12701036439963 
Path Answer Precision: 49.16111111111111 
Path Answer Recall: 68.85842667662091
"""
import os

import argparse
from tqdm import tqdm
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import os
from datasets import load_dataset
from utils.gcr_utils import eval_path_result_w_ans
import json
from multiprocessing import Pool
from functools import partial
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
    def __init__(self, tokenizer, prompt="zero-shot", undirected = False, index_path_length=2, add_rule=False):
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.undirected = undirected
        self.index_path_length = index_path_length
        self.add_rule = add_rule
        self.prompt_template = self.get_prompt_template(self.prompt)

    def get_prompt_template(self, template_name): 
        try:
            # self.ZERO_SHOT_PROMPT = Reasoning path is a sequence of triples in the KG that connects the topic entities in the question to answer entities. Given a question, please generate some reasoning paths in the KG starting from the topic entities to answer the question.
            template_name = template_name.upper().replace("-", "_") + "_PROMPT"  # ZERO_SHOT_PROMPT
            return self.__getattribute__(template_name) # 
        except:
            raise ValueError(f"The template name: {template_name} is not valid.")

    def format_input_with_template(self, question, start_entities, choices = []):
        '''
        question: str  问题文本
        start_entities: List[str]  问题的topic entities list形式
        '''
        if len(choices) > 0:
            return self.prompt_template.format(
                question=question, entities=",".join(start_entities), choices="\n".join(choices)
            )
        else:
            # 返回一个prompt 文本，给出问题entities和问题文本，生成到答案的路径
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
            g = build_graph(question_dict["graph"], self.undirected)  # 构造有向图

            # 保存q_entities出发所有index_path_length=2 （1或2）跳以内的路径，作为推理的候选路径，用来约束生成路径使其不产生幻觉
            # 例如 从'Jamaica'出发的2-hop路径有：
            # (('Jamaica', 'location.country.administrative_divisions', 'Westmoreland Parish'), ('Westmoreland Parish', 'location.location.containedby', 'Cornwall County, Jamaica'))
            # 1-hop路径有：(('Jamaica', 'sports.sport_country.athletic_performances', 'm.0b64vhp'),)
            paths_list = dfs(g, question_dict["q_entity"], self.index_path_length)

        # 生成推理路径 (('Jamaica', 'sports.sport_country.athletic_performances', 'm.0b64vhp'),) 转为字符串形式 'Jamaica -> sports.sport_country.athletic_performances -> m.0b64vhp'
        # 每条q_entities 2-hop子图内的候选路径保存为如下形式<PATH>Jamaica -> sports.sport_country.athletic_performances -> m.0b64vhp</PATH>
        # 问题节点衍生的所有推理路径有3593 条
        paths_list_str = [f"{self.PATH_START_TOKEN}{path_to_string(path)}{self.PATH_END_TOKEN}" for path in paths_list]
        if len(paths_list_str) == 0:
            return None
        # paths_list_str = ['<PATH>Jamaica -> sports.sport_country.athletic_performances -> m.0b64vhp</PATH>', '<PATH>Jamaica -> location.country.administrative_divisions -> Westmoreland Parish -> location.location.containedby -> Cornwall County, Jamaica</PATH>', ...]
        # 将所有3593条路径转为token ids形式，如下所示，一行代表一个句子（路径）
        # [
        #   [128257, 41, 3105, 3074, 1492, 3813, 31187, 40596, 21276, 522, 2554, 1265, 65249, 9430, 1492, 342, 13, 16, 20990, 66, 2096, 89, 36825, 128258]
        #   [128257, 41, 3105, 3074, 1492, 3813, 31187, 40596, 21276, 1770, 47262, 488, 70970, 562, 5796, 16793, 6388, 1492, 296, 13, 15, 32837, 19, 51622, 87, 128258]
        # ]
        # 该测试问题有3593条候选路径
        # print(paths_list_str[0])
        tokenized_paths = self.tokenizer(
            paths_list_str, padding=False, add_special_tokens=False
        ).input_ids # 返回所有句子的token ids形式，这些id在llm中有对应的embedding向量
        # print(tokenized_paths[0])
        # len(self.tokenizer) + 1 是因为tokenizer的vocab size + 1
        return MarisaTrie(tokenized_paths, max_token_id=len(self.tokenizer) + 1)

    def process_input(self, question_dict, return_tire = True):
        # 传入的是一个测试实例
        question = question_dict["question"]    # 测试集一个问题文本'what does jamaican people speak'
        start_node = question_dict["q_entity"]  # 开始节点 start_node=['Jamaica']
        anser_node = question_dict["a_entity"] # 回答节点 anser_node = ['Jamaican English', 'Jamaican Creole English Language']
        choices = question_dict.get("choices", [])  # []
        trie = None
        if return_tire: # 返回trie
            # trie为当前问题entities出发的所有2-hop以内路径的token ids 的 unicode字符构造该问题的Trie
            # 用来约束模型生成路径的合法性
            trie = self.get_graph_index(question_dict) # 调用JointReasoningPromptBuilder中的get_graph_index方法构造trie

        g = build_graph(question_dict["graph"], self.undirected)
        truth_paths = get_truth_paths(start_node, anser_node, g) # 问题的真实推理路径（问题到答案的最短路径）
        # 真实路径转成string模式 'Jamaica -> location.country.official_language -> Jamaican English'
        ground_paths = [path_to_string(path) for path in truth_paths]

        if not question.endswith("?"):
            question += "?"

        # prompt文本构造，让模型输出问题entities出发的路径，prompt接受问题文本和问题entities
        input = self.format_input_with_template(question, start_node, choices=choices)
        return input, ground_paths, trie



class GraphConstrainedDecodingModel(HFCasualModel):
    def __init__(self, args):
        super().__init__(args)
    
    def generate_sentence(self, llm_input, trie, start_token_ids = None, end_token_ids = None, enable_constrained_by_default = True):
        '''
        llm_input: 模型的输入为下面文本的“模型可读形式” tokenizer.apply_chat_template(chat_query, tokenize=False, add_generation_prompt=True)
        Reasoning path is a sequence of triples in the KG that connects the topic entities in the question to answer entities. Given a question, please generate some reasoning paths in the KG starting from the topic entities to answer the question.
        Question: 
        what is the name of justin bieber brother 
        Topic entities: 
        [justin bieber]
        Reasoning path:

        trie: 为该问题的KG-Trie，用于约束生成路径的规范 （必须在Trie中存在）
        start_token_ids: 路径起始标记的token id 列表，比如 ["<PATH>"] 的token id 列表
        end_token_ids: 路径结束标记的token id 列表，比如 ["</PATH>"] 的token id 列表
        enable_constrained_by_default： False
        '''

        inputs = self.tokenizer(llm_input, return_tensors="pt", add_special_tokens=False)

        input_ids = inputs.input_ids.to(self.model.device)  # llm_prompt的token ids，包含了prompt句子，问题和问题的topic entities
        attention_mask = inputs.attention_mask.to(self.model.device)

        input_length = input_ids.shape[1]  # llm_prompt的长度
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
        # try:
        #     # 生的是「数字」——token id 序列
        #     res = self.model.generate(
        #         input_ids = input_ids,  # llm_prompt的token ids，包含了prompt句子，问题和问题的topic entities
        #         attention_mask = attention_mask, # llm_prompt的attention mask
        #         generation_config=self.generation_cfg,  # 生成配置，包含生成参数如max_new_tokens等，并且以group-beam的方式生成k=10条多样化路径
        #         # prefix_allowed_tokens_fn(batch_id, generated_ids), generated_id这条样本目前已经生成的 token 序列（包括 prompt + 已生成部分）。
        #         # 返回值：一个 token id 的列表，表示下一步只能从这些 token 里选。
        #         prefix_allowed_tokens_fn=gcr.allowed_tokens_fn, # 用来限制beam search生成允许的tokens，每一步解码都会调用这个函数，基于已经生成的token，允许下一步生成那些tokens
        #         return_dict_in_generate=True, # 它会把生成序列 + 可选的中间信息（scores、attentions、hidden states、beam 树等）一并包装在一个“字典风格”的结构里返回。
        #         pad_token_id=self.tokenizer.eos_token_id  # 如果需要对生成结果做 padding，就用 eos_token 对应的 id 来补齐。因为后续要对res做解码，所以需要让所有sequence长度一致
        #     )
        # except Exception as e:
        #     print(e)
        #     return None
        response = []
        if len(res.sequences) == 1:
            # 将model generate生成的token ids转换成文本返回，跳过special tokens，并且只返回生成部分（不包括prompt部分）
            return self.tokenizer.decode(res.sequences[0][input_length:], skip_special_tokens=True)
        for r in res.sequences:
            response.append(self.tokenizer.decode(r[input_length:],   # 
          skip_special_tokens=True))
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
        return sample  # TODO: ignore the sample with zero paths.

    qa_dataset = qa_dataset.map(find_rule, num_proc=n_proc)
    if filter_empty:
        qa_dataset = qa_dataset.filter(
            lambda x: len(x["ground_paths"]) > 0, num_proc=n_proc
        )
    return qa_dataset

def get_output_file(path, force=False):
    # force = False表示不强制覆盖已有文件
    if not os.path.exists(path) or force:
        fout = open(path, "w")  # 创建文件
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
    '''
    data, 
    processed_list, 开始时是空list [], 然后逐个data写入
    input_builder, 
    model: 用训练集问答预训练好的LLM
    '''
    question = data["question"] # 问题文本 what is the name of justin bieber brother
    answer = data["answer"]     # 答案文本 ["Jaxon Bieber"]
    id = data["id"]             # 问题id，如WebQTrn-0， WebQTrn-1，...
    if id in processed_list:
        return None

    # 构造trie，作为模型生成路径的约束，即生成的路径必须满足trie中存在的路径

    # input_query: prompt文本，给出问题entities和问题文本，生成到答案的路径 
    # ground_paths:真实路径转成string模式，例如 'Jamaica -> location.country.official_language -> Jamaican English'
    # trie: 当前问题entities出发的所有2-hop以内路径的token ids 的 unicode字符构造该问题的Trie，用来约束生成的路径
    input_query, ground_paths, trie = input_builder.process_input(data) # In GraphConstrainedPromptBuilder.process_input
    if trie is None:
        return None
    start_token_ids = model.tokenizer.convert_tokens_to_ids(
        input_builder.PATH_START_TOKEN  # "<PATH>"
    )
    end_token_ids = model.tokenizer.convert_tokens_to_ids(input_builder.PATH_END_TOKEN)  # "</PATH>"
    input = model.prepare_model_prompt(input_query)  # 将prompt+question+q_entities（start_node）作为LLM的输入tokenize化
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
        "prediction": prediction,  # 针对该问题生成的句子
        "ground_truth": answer,
        "ground_truth_paths": ground_paths,
        "input": input,
    }
    return result


def main(args, LLM):
    input_file = os.path.join(args.data_path, args.d)  # rmanluo/RoG-webqsp  or rmanluo/RoG-cwq
    # Load dataset
    dataset = load_dataset(input_file, split=args.split) # rmanluo/RoG-webqsp  test 加载测试样本
    # post_fix = {zero-shot}-{group-beam}-k10-index_len2
    post_fix = f"{args.prefix}{args.prompt_mode}-{args.generation_mode}-k{args.k}-index_len{args.index_path_length}"
    if args.add_rule: # False
        rule_postfix = args.rule_path.replace("/", "_").replace(".", "_")
        rule_dataset = load_jsonl(args.rule_path)
        dataset = merge_rule_result(dataset, rule_dataset, args.n, args.filter_empty)
        post_fix += "_" + rule_postfix
    data_name = args.d + "_undirected" if args.undirected else args.d  # 有向图，args.d = RoG-webqsp or RoG-cwq
    output_dir = os.path.join(args.predict_path, data_name, args.model_name, args.split, post_fix)  # 推理路径的保存地址
    print("Save results to: ", output_dir)

    # Predict
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    model = LLM(args)
    
    print("Prepare pipline for inference...")
    # 加载fine-tuning过的模型和tokenizer等
    # 配置： max_new_tokens=1024, generation_mode=group_beam, k=10等，生成10个句子，用group beam来增强多样性
    model.prepare_for_inference() 
    # prompt_mode = zero-shot
    # index_path_length = 2
    # undirected = False
    # add_rule = False
    input_builder = PathGenerationWithAnswerPromptBuilder(model.tokenizer, args.prompt_mode, index_path_length=args.index_path_length, undirected=args.undirected, add_rule=args.add_rule)
    
    # Save args file
    with open(os.path.join(output_dir, 'args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    
    # fout是文件对象
    # processed_list第一次创建时是[]
    fout, processed_list =  get_output_file(os.path.join(output_dir, 'predictions.jsonl'), force=args.force) 
    
    if args.n > 1:  # n = number of processes 是否多GPU
        with Pool(args.n) as p:  
            for res in tqdm(
                p.imap(
                    partial(
                        prediction,
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
    else:# single GPU
        for data in tqdm(dataset): # dataset是测试集  
            res = prediction(data, processed_list, input_builder, model)  # 为每个问题生成推理路径
            # 下面是一个测试问题的输出，生成的路径，都在候选Trie中，这个候选Trie是从问题entities出发的所有2-hop以内路径构建的
            # {"id": "WebQTest-0", 
            # "question": "what does jamaican people speak", 
            # "prediction": ["# Reasoning Path:\nJamaica -> location.country.languages_spoken -> Jamaican Creole English Language\n# Answer:\nJamaican Creole English Language", 
            #                "# Reasoning Path:\nJamaica -> location.country.languages_spoken -> Jamaican English\n# Answer:\nJamaican English", 
            #                "# Reasoning Path:\nJamaica -> location.country.languages_spoken -> Jamaican Creole English Language\n# Answer:\nJamaican Creole English Language", 
            #                "# Reasoning Path:\nJamaica -> location.country.languages_spoken -> Jamaican English\n# Answer:\nJamaican English", 
            #                "# Reasoning Path:\nJamaica -> location.country.languages_spoken -> Jamaican Creole English Language\n# Answer:\nJamaican Creole English Language", 
            #                "# Reasoning Path:\nJamaica -> location.country.currency_used -> Jamaican dollar -> finance.currency.countries_used -> Jamaica\n# Answer:\nJamaica", 
            #                "# Reasoning Path:\nJamaica -> location.country.languages_spoken -> Jamaican English\n# Answer:\nJamaican English", 
            #                "# Reasoning Path:\nJamaica -> location.country.languages_spoken -> Jamaican Creole English Language\n# Answer:\nJamaican Creole English Language", 
            #                "# Reasoning Path:\nJamaica -> location.country.languages_spoken -> Jamaican English\n# Answer:\nJamaican English", 
            #                "# Reasoning Path:\nJamaica -> location.country.languages_spoken -> Jamaican Creole English Language\n# Answer:\nJamaican Creole English Language"], 
            # "ground_truth": ["Jamaican English", "Jamaican Creole English Language"], 
            # "ground_truth_paths": ["Jamaica -> location.country.languages_spoken -> Jamaican English", "Jamaica -> location.country.languages_spoken -> Jamaican Creole English Language"], 
            # "input": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nReasoning path is a sequence of triples in the KG that connects the topic entities in the question to answer entities. Given a question, please generate some reasoning paths in the KG starting from the topic entities to answer the question.\n\n# Question: \nwhat does jamaican people speak?\n# Topic entities: \nJamaica<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"}

            if res is not None:
                if args.debug:
                    print(json.dumps(res))
                fout.write(json.dumps(res) + "\n")
                fout.flush()
            else:
                print("None result for: ", data["id"])
    fout.close()
            
    eval_path_result_w_ans(os.path.join(output_dir, 'predictions.jsonl'))
    
if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--data_path', type=str, default='rmanluo')  
    argparser.add_argument('--d', '-d', type=str, default='RoG-webqsp')
    argparser.add_argument('--split', type=str, default='test[:100]')
    argparser.add_argument('--index_path_length', type=int, default=2)
    argparser.add_argument('--predict_path', type=str, default='results/GenPaths')
    argparser.add_argument('--model_name', type=str, help="model_name for save results", default='save_models/FT-Qwen3-8B')  # rmanluo/GCR-Meta-Llama-3.1-8B-Instruct
    argparser.add_argument('--force', action='store_true', help="force to overwrite the results")
    argparser.add_argument("--n", type=int, default=1, help="number of processes")
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

    args, _  = argparser.parse_known_args()
    
    LLM = GraphConstrainedDecodingModel
    LLM.add_args(argparser)  # 把配置加入模型中
    
    args = argparser.parse_args()
    
    main(args, LLM)


