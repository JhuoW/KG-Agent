from llms.base_hf_casual_lm import HFCasualModel
import torch

class DecodingModel(HFCasualModel):
    def __init__(self, args):
        super().__init__(args)
    


class GraphConstrainedDecoding:
    def __init__(self, tokenizer, trie, start_token_ids = None, end_token_ids = None, enable_constrained_by_default = False):
        
        self.tokenizer = tokenizer
        self.trie = trie
        self.start_token = start_token_ids
        self.end_token = end_token_ids
        self.all_tokens = list(range(len(tokenizer)))
        self.constrained_flag = enable_constrained_by_default  # 默认是False
        self.L_input = None

    def check_constrained_flag(self, sent: torch.Tensor):  # sent: 当前已经生成的token id 序列
        # Check start
        # 检查是否已经开始生程路径（是否包含<PATH>）
        matched_start_token = torch.where(sent == self.start_token)[0] # 如果当前已生成序列中包含开始序列 <PATH>的token id
        # print(sent)
        # print(self.start_token)
        # print("matched_start_token:", matched_start_token)
        if len(matched_start_token) == 0:  # 尚未开始生成路径那么不对下一步要生成的token做约束
            return False, len(sent)
        last_start_tokens = torch.where(sent == self.start_token)[0][-1]  # 当前已生成序列中最后一个<PATH>的token位置
        end_token_number = len(torch.where(sent[last_start_tokens:] == self.end_token)[0]) # 计算从最后一个<PATH>开始到当前已经生成序列结尾之间，有多少个</PATH>
        # GCR not closed
        if end_token_number == 0:  # 如果从最后一个<PATH>到当前生成序列结尾之间没有</PATH>，说明当前正在生成路径
            self.last_start_token = last_start_tokens
            return True, last_start_tokens  #则后面生成的路径需要被约束
        else:  # 如果最后一个<PATH>之后已经有了</PATH>，说明当前没有在生成路径
            self.last_start_token = None  
            return False, len(sent)

    def allowed_tokens_fn(self, batch_id: int, sent: torch.Tensor):
        constrained_flag = self.constrained_flag
        # print(sent)
        # Check if enter the constrained decoding
        if self.start_token is not None and self.end_token is not None:
            constrained_flag, L_input = self.check_constrained_flag(sent)
        # Assign self.L_input with the input length
        else:
            if self.L_input is None:
                self.L_input = len(sent)
            L_input = self.L_input
            
        allow_tokens = self.all_tokens  # 表示当前问题的所有候选路径的总词汇表
        if constrained_flag: # 没有走这里
            allow_tokens = self.trie.get(sent.tolist()[L_input:]) 
            if len(allow_tokens) == 0:
                return self.all_tokens
        return allow_tokens

    # def check_constrained_flag(self, sent: torch.Tensor):  
    #     matched_start_token = torch.where(sent == self.start_token)[0]
    #     if len(matched_start_token) == 0:
    #         # No <PATH> token found yet, no constraint
    #         return False, len(sent)       
    #     last_start_pos = matched_start_token[-1].item()  #最后一个 <PATH> token的位置
    #     tokens_after_start = sent[last_start_pos:] # 最后一个 <PATH>之后的token序列
    #     end_token_positions = torch.where(tokens_after_start == self.end_token)[0]
    #     if len(end_token_positions) == 0: # 如果从最后一个<PATH>到当前生成序列结尾之间没有</PATH>，说明当前正在生成路径
    #         return True, last_start_pos + 1
    #     else:
    #         return False, len(sent)

    

    # def allowed_tokens_fn(self, batch_id: int, sent: torch.Tensor):
    #     # 用于限制beam search每一步允许生成的tokens
    #     constrained_flag = self.constrained_flag
    #     # Check if enter the constrained decoding
    #     if self.start_token is not None and self.end_token is not None:
    #         constrained_flag, path_start_pos = self.check_constrained_flag(sent)
    #     # Assign self.L_input with the input length
    #     else:
    #         if self.input_length is not None:
    #             path_start_pos = self.input_length
    #         elif self.L_input is None:
    #             self.L_input = len(sent)
    #             path_start_pos = self.L_input
    #         else:
    #             path_start_pos = self. L_input
    #     if not constrained_flag:
    #         return self.all_tokens

    #     generated_tokens = sent[path_start_pos:].tolist()   

    #     allowed_tokens = self.trie.get(generated_tokens)
    #     if len(allowed_tokens) == 0:
    #         return self.all_tokens
        

