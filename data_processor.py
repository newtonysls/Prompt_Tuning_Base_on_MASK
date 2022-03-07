from __future__ import absolute_import, division, print_function
# 放在第一句，不然会报错

from operator import index
import os
import random
from tqdm import tqdm
from random import random
import multi_tasks_pretrain_args as args
import json
def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens):
    """Truncates a pair of sequences to a maximum sequence length. Lifted from Google's BERT repo."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_num_tokens:
            break

        trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
        assert len(trunc_tokens) >= 1

        # We want to sometimes truncate from the front and sometimes from the
        # back to add more randomness and avoid biases.
        if random() < 0.5:
            del trunc_tokens[0]
        else:
            trunc_tokens.pop()

def Json_File_Reader(data_name,data_paths,dynamic_mask_times=None):
    if type(data_paths)==list:
        pass
    else:
        data_paths = [data_paths]
    all_data = []
    for index,data_path in enumerate(data_paths):
            with open(data_path,'r',encoding='utf-8') as f:
                for item in f.readlines():
                    item = json.loads(item)
                    # item['data_type'] = data_name
                    item['is_val'] = index
                    item['data_name'] = data_name
                    all_data.append(item)
    
    if dynamic_mask_times:
        temp_data = all_data.copy()
        for i in range(dynamic_mask_times-1):
            all_data = all_data + temp_data
    return all_data

class RTE_Processing:
    def __init__(self,data,tokenizer,max_seq_length,vocab_list,is_pretraining = True):
        self.data = data
        self.data_type = args.superglue['RTE']
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.vocab_list = vocab_list
        self.is_pretraining = is_pretraining
        self.special_token_length = 2
        # [CLS] and [SEP]
        self.hard_prompt_token_length = 8

        self.max_token_length = self.max_seq_length - self.special_token_length-self.hard_prompt_token_length-args.soft_prompt_length

    def Creat_Input_For_PLMs(self):
        
        data_input = []
        for index,item in tqdm(enumerate(self.data),desc="RTE Data Processing"):
            sentence1 = item['premise']
            sentence2 = item['hypothesis'].strip('.')
            label = item['label']
            label_ids = self.data_type['label_list'].index(label)

            answer = self.data_type['prompt_answer_list'][label_ids]

            tokens_a = self.tokenizer.tokenize(sentence1)
            tokens_b = self.tokenizer.tokenize(sentence2)
            answer_tokens = self.tokenizer.tokenize(answer)
            if self.is_pretraining:
                if item['is_val']:
                    answer_tokens = []
                    max_token_length = self.max_token_length + 1
                else:
                    max_token_length = self.max_token_length 
            else:
                answer_tokens = ["[MASK]"]
                max_token_length = self.max_token_length
            truncate_seq_pair(tokens_a, tokens_b, max_token_length)
            tokens = ["[CLS]"] +["[unused{}]".format(i+1) for i in range(args.soft_prompt_length)]+ tokens_a +["Question",":"] + tokens_b +["?","The","Answer",":"]+ answer_tokens + ["."] + ["[SEP]"]
            start = len(tokens_a)+1+args.soft_prompt_length+2+len(tokens_b)+4
            prompt_positions = [i for i in range(start,start+1)]
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            assert len(tokens)==len(input_ids)

            attention_ids = [1] * len(input_ids)
            segment_ids = [0] * len(input_ids)
            data_item = {'tokens':tokens,
                'input_ids':input_ids,
                'segment_ids':segment_ids,
                'attention_ids':attention_ids,
                'prompt_positions':prompt_positions,
                'label_ids':label_ids,
                'data_name':'RTE'}
            # no padding
            data_input.append(data_item)

        return data_input

class CB_Processing:
    def __init__(self,data,tokenizer,max_seq_length,vocab_list,is_pretraining = True):
        self.data = data
        self.data_type = args.superglue['CB']
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.vocab_list = vocab_list
        self.is_pretraining = is_pretraining
        self.special_token_length = 3
        # [CLS] and [SEP] and [SEP]
        self.hard_prompt_token_length = 6

        self.max_token_length = self.max_seq_length - self.special_token_length-self.hard_prompt_token_length-args.soft_prompt_length

    def Creat_Input_For_PLMs(self):
        # [text_a,  "[SEP]", example.text_b, "?", 'the',  " answer: ", self.mask]
        data_input = []
        for index,item in tqdm(enumerate(self.data),desc="CB Data Processing"):
            sentence1 = item['premise']
            sentence2 = item['hypothesis']
            label = item['label']
            label_ids = self.data_type['label_list'].index(label)

            answer = self.data_type['prompt_answer_list'][label_ids]

            tokens_a = self.tokenizer.tokenize(sentence1)
            tokens_b = self.tokenizer.tokenize(sentence2)
            answer_tokens = self.tokenizer.tokenize(answer)
            if self.is_pretraining:
                if item['is_val']:
                    answer_tokens = []
                    max_token_length = self.max_token_length + 1
                else:
                    max_token_length = self.max_token_length 
            else:
                answer_tokens = ["[MASK]"]
                max_token_length = self.max_token_length
            truncate_seq_pair(tokens_a, tokens_b, max_token_length)
            tokens = ["[CLS]"] +["[unused{}]".format(i+1) for i in range(args.soft_prompt_length)]+ tokens_a +["[SEP]"] + tokens_b +["?","The","Answer",":"]+ answer_tokens + ["."] + ["[SEP]"]

            start = len(tokens_a)+2+args.soft_prompt_length+len(tokens_b)+4
            prompt_positions = [i for i in range(start,start+1)]
            
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            attention_ids = [1] * len(input_ids)
            segment_ids = [0] * (len(tokens_a)+2+args.soft_prompt_length) + [1] * (len(input_ids)-(len(tokens_a)+2+args.soft_prompt_length))
            assert len(input_ids)==len(segment_ids)
            data_item = {'tokens':tokens,
                'input_ids':input_ids,
                'segment_ids':segment_ids,
                'attention_ids':attention_ids,
                'prompt_positions':prompt_positions,
                'label_ids':label_ids,
                'data_name':'CB'}
            # no padding
            data_input.append(data_item)

        return data_input

class BoolQ_Processing:
    def __init__(self,data,tokenizer,max_seq_length,vocab_list,is_pretraining = True):
        self.data = data
        self.data_type = args.superglue['BoolQ']
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.vocab_list = vocab_list
        self.is_pretraining = is_pretraining
        self.special_token_length = 2
        # [CLS] and [SEP]
        self.hard_prompt_token_length = 8

        self.max_token_length = self.max_seq_length - self.special_token_length-self.hard_prompt_token_length-args.soft_prompt_length

    def Creat_Input_For_PLMs(self):
        # [passage, '.', 'the', ' Question: ', question, '? Answer: ', self.mask, '.']
        
        data_input = []
        for index,item in tqdm(enumerate(self.data),desc="BoolQ Data Processing"):
            sentence1 = item['passage']
            sentence2 = item['question']
            # sentence2 = item['question'].capitalize()
            label = item['label']
            label_ids = self.data_type['label_list'].index(label)

            answer = self.data_type['prompt_answer_list'][label_ids]

            tokens_a = self.tokenizer.tokenize(sentence1)
            tokens_b = self.tokenizer.tokenize(sentence2)
            answer_tokens = self.tokenizer.tokenize(answer)
            if self.is_pretraining:
                if item['is_val']:
                    answer_tokens = []
                    max_token_length = self.max_token_length + 1
                else:
                    max_token_length = self.max_token_length 
            else:
                answer_tokens = ["[MASK]"]
                max_token_length = self.max_token_length

            truncate_seq_pair(tokens_a, tokens_b, max_token_length)
            tokens = ["[CLS]"] +["[unused{}]".format(i+1) for i in range(args.soft_prompt_length)]+ tokens_a +['The',"Question",":"] + tokens_b +["?","Answer",":"]+ answer_tokens + ["."] + ["[SEP]"]

            start = len(tokens_a)+1+args.soft_prompt_length+3+len(tokens_b)+3
            prompt_positions = [i for i in range(start,start+1)]
            

            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            attention_ids = [1] * len(input_ids)
            segment_ids = [0] * len(input_ids)

            data_item = {'tokens':tokens,
                'input_ids':input_ids,
                'segment_ids':segment_ids,
                'attention_ids':attention_ids,
                'prompt_positions':prompt_positions,
                'label_ids':label_ids,
                'data_name':'BoolQ'}
            # no padding
            data_input.append(data_item)

        return data_input