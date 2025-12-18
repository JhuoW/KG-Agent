from utils.utils import *


class PromptBuilder(object):
    PATH_START_TOKEN = "<PATH>"
    PATH_END_TOKEN = "</PATH>"
    # TODO: add more prompt templates
    ZERO_SHOT_PROMPT = ""
    ZERO_SHOT_NO_MORE_THAN_10_PROMPT = ""
    MULTIPATH_GEN_PROMPT = ""

    def __init__(self, tokenizer, prompt = 'zero-shot', undirected = False, index_path_length = 2, add_rule = False):
        self.tokenzier = tokenizer
        self.prompt = prompt
        self.undirected = undirected
        self.index_path_length = index_path_length
        self.add_rule = add_rule
        self.prompt_template = self.get_prompt_template(self.prompt)

    def get_prompt_template(self, template_name):
        try:
            template_name = template_name.upper().replace("-", "_") + "_PROMPT"  # ZERO_SHOT_PROMPT
            return self.__getattribute__(template_name)   # self.ZERO_SHOT_PROMPT
        except:
            raise ValueError(f"Unknown prompt template: {template_name}")


    def format_input_with_template(self, question, start_entities, choice = []):
        if len(choice) > 0:
            # TODO: format LLM prompt
            return self.prompt_template.format()
        else:
            # TODO: format LLM prompt
            return self.prompt_template.format()
        return
    # TODO: to be implemented
    def apply_rules(self):
        return

    def process_input(self, question_dict):
        question = question_dict["question"]
        start_node = question_dict["q_entity"]
        answer_node = question_dict['a_entity']
        choice = question_dict.get('choices', [])

        g = build_graph(question_dict['graph'], self.undirected)
        gt_paths = get_truth_paths(start_node, answer_node, g)  # ground truth paths

        # transform paths to strings
        gt_paths_str = [path_to_string(path) for path in gt_paths]

        if not question.endswith("?"):
            question += "?"
        input = self.format_input_with_template(question, start_node, choice = choice)
        return input, gt_paths_str

