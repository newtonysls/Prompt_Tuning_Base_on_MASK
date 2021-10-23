from __future__ import absolute_import, division, print_function
# 放在第一句，不然会报错

import collections
import logging
from operator import index
import os
import random
from xml.dom.minidom import Document
from matplotlib.pyplot import title
import numpy as np
import pandas as pd
import csv

import torch
import torch.nn as nn
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, Dataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from random import random, randrange, randint, shuffle, choice, sample

# from modeling_nezha import  WEIGHTS_NAME
# from modeling_nezha import BertForMaskedLM, BertConfig
# from optimization import BertAdam
# from official_tokenization import BertTokenizer

from transformers import BertForMaskedLM,BertConfig,BertTokenizer
from transformers import AdamW
from transformers.optimization import (
    get_constant_schedule,
    get_constant_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)

import pretrain_args as args
import jieba
import re
import json
from sklearn.model_selection import KFold

# Random Mask
def create_masked_lm_predictions(tokens, masked_lm_prob, max_predictions_per_seq, vocab_list):
    """Creates the predictions for the masked LM objective. This is mostly copied from the Google BERT repo, but
    with several refactors to clean it up and remove a lot of unnecessary variables."""
    # 对一个句子进行MLM
    """
    tokens:一句话的tokens
    masked_lm_prob:覆盖的概率
    max_predictions_per_seq:每个seq，最大预测
    vocab_list:词汇表
    return:经过MLM的tokens，mask的index，mask的label
    """
    cand_indices = []
    for (i, token) in enumerate(tokens):
        if token == "[CLS]" or token == "[SEP]":
            continue
        cand_indices.append(i)
    # 获取除了，特殊的CLS和SEP的词语的token下标
    num_to_mask = min(max_predictions_per_seq,
                      max(1, int(round(len(tokens) * masked_lm_prob))))
    # mask数量，极端情况下，至少为1，最多max_predictions_per_seq
    # print(num_to_mask)
    # print("tokens", len(tokens))
    # print("cand", len(cand_indices))
    shuffle(cand_indices)
    # 打乱cand_indices
    mask_indices = sorted(sample(cand_indices, num_to_mask))
    # 从cand_indices随机抽取num_to_mask个元素，并且以list返回，然后进行排序
    masked_token_labels = []
    for index in mask_indices:
        # 80% of the time, replace with [MASK]
        if random() < 0.8:
            masked_token = "[MASK]"
        else:
            # 10% of the time, keep original
            if random() < 0.5:
                masked_token = tokens[index]
            # 10% of the time, replace with random word
            else:
                masked_token = choice(vocab_list)
                # choice() 方法返回一个列表，元组或字符串的随机项。
        masked_token_labels.append(tokens[index])
        # Once we've saved the true label for that token, we can overwrite it with the masked version
        tokens[index] = masked_token

    return tokens, mask_indices, masked_token_labels

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

class PretrainDataset(Dataset):
    def __init__(self,DATAFRAME,tokenizer,prompt_pattern_list,label_list,max_seq_length,masked_lm_prob,max_predictions_per_seq,vocab_list):
        self.data = DATAFRAME
        """
        DATAFRAME:为pandas的DataFrame，需要第一列文本
        """
        self.tokenizer = tokenizer
        self.prompt_pattern_list = prompt_pattern_list
        self.label_list = label_list
        
        self.max_seq_length = max_seq_length
        self.nums_specical_tokens = 2
        self.max_tokens_nums = max_seq_length - self.nums_specical_tokens - len(prompt_pattern_list[0])
        self.masked_lm_prob = masked_lm_prob
        self.max_predictions_per_seq = max_predictions_per_seq
        self.vocab_list = vocab_list
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self,index):
        text_a = self.data[0].iloc[index].strip('\n').strip('。')
        text_b = self.data[1].iloc[index].strip('\n').strip('。')
        label = self.label_list.index(self.data[2].iloc[index])
        template = self.prompt_pattern_list[label]
        # text = self.data[0].iloc[index].strip('\n')
        tokens_template = self.tokenizer.tokenize(template)
        tokens_a = self.tokenizer.tokenize(text_a)
        tokens_b = self.tokenizer.tokenize(text_b)
        truncate_seq_pair(tokens_a, tokens_b, self.max_tokens_nums)
#         tokens = tokens[:self.max_tokens_nums]
        tokens = ["[CLS]"] + tokens_a + tokens_template + tokens_b + ["[SEP]"]
        # masked_labels = tokens.copy()

        tokens, masked_lm_positions, masked_lm_labels = create_masked_lm_predictions(tokens, self.masked_lm_prob, self.max_predictions_per_seq, self.vocab_list)

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        # label_ids = self.tokenizer.conver_tokens_to_ids(masked_labels)

        attention_ids = [1] * len(input_ids)
        padding_length = self.max_seq_length - len(input_ids)

        input_ids += [0] * padding_length
        # label_ids += [0] * padding_length

        attention_ids += [0] * padding_length

        segment_ids = [0] * len(input_ids)
        label_ids = np.full(self.max_seq_length, dtype=np.int, fill_value=-100)
        label_ids[masked_lm_positions] = self.tokenizer.convert_tokens_to_ids(masked_lm_labels)

        input_ids = torch.tensor(input_ids,dtype=torch.long)
        segment_ids = torch.tensor(segment_ids,dtype=torch.long)
        attention_ids = torch.tensor(attention_ids,dtype=torch.long)
        label_ids = torch.tensor(label_ids,dtype=torch.long)

        return input_ids, segment_ids, attention_ids, label_ids

# def read_data(data_paths,dynamic_mask_times=None,eval_rate=0.1):
#         """
#         data_path:文件格式为txt，tsv，csv的文件路径
#         第一列或者第一二列为文本，最后一列为标签
#         dynamic_mask_times:动态掩词
#         eval_rate:验证集比例
#         """
        
#         for index,data_path in enumerate(data_paths):
#             df = pd.read_csv(data_path, sep='\t', header=None, quoting=csv.QUOTE_NONE,encoding='utf-8')
#             columns = df.shape[1]
#             if columns == 3:
#                 df[3] = df[1] + df[2]
#                 temp_data = df[3]
#             else:
#                 temp_data = df[0] + df[1]
#             if index != 0 :
#                 all_data = pd.concat([all_data,temp_data])
#             else:
#                 all_data = temp_data
            

#         if dynamic_mask_times:
#             temp_data = all_data.copy()
#             for _ in range(dynamic_mask_times):
#                 all_data = pd.concat([all_data,temp_data])
#         eval_data_size = int(all_data.shape[0]*eval_rate)
#         eval_data = all_data[:eval_data_size]
#         train_data = all_data[eval_data_size:]

#         return train_data,eval_data
    
def read_data(data_paths,dynamic_mask_times=None,eval_rate=0.1):
        """
        data_path:文件格式为txt，tsv，csv的文件路径
        第一列或者第一二列为文本，最后一列为标签
        dynamic_mask_times:动态掩词
        eval_rate:验证集比例
        """
        
        for index,data_path in enumerate(data_paths):
            temp_data = pd.read_csv(data_path, sep='\t', header=None, quoting=csv.QUOTE_NONE,encoding='utf-8')

            if index != 0 :
                all_data = pd.concat([all_data,temp_data])
            else:
                all_data = temp_data
            

        if dynamic_mask_times:
            temp_data = all_data.copy()
            for _ in range(dynamic_mask_times):
                all_data = pd.concat([all_data,temp_data])
        eval_data_size = int(all_data.shape[0]*eval_rate)
        eval_data = all_data[:eval_data_size]
        train_data = all_data[eval_data_size:]

        return train_data,eval_data

logger = logging.getLogger(__name__)

def main():
    if args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.distributed.init_process_group(backend='nccl')
        local_rank = torch.distributed.get_rank()
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        n_gpu = torch.cuda.device_count()

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if local_rank in [-1, 0] else logging.WARN)

    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(local_rank != -1), args.fp16))

    if not os.path.exists(args.model_save_path):
        os.makedirs(args.model_save_path)

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
    # seed everything,python,numpy and torch
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    
    # tokenizer = BertTokenizer(vocab_file = os.path.join(args.bert_model,args.vocab_file))
    # vocab_list = list(tokenizer.vocab.keys())
    # config=BertConfig.from_json_file(os.path.join(args.bert_model,args.bert_config_json))
    tokenizer = BertTokenizer.from_pretrained(args.model_name_from_hugging_face)
    # append special tokens and need resize model embedding size
#     tokenizer.add_special_tokens({'additional_special_tokens':['[机构]','[numbers]']})

    vocab_list = list(tokenizer.vocab.keys())
    config = BertConfig.from_pretrained(args.model_name_from_hugging_face)

    

    train_data,eval_data = read_data(args.data_path,args.dynamic_mask_times)

    num_train_optimization_steps = None

    num_train_optimization_steps = int(train_data.shape[0] / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
    if local_rank != -1:
        num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    
    # 定义持续预训练模型
    model = BertForMaskedLM.from_pretrained(args.model_name_from_hugging_face)
    model.resize_token_embeddings(len(tokenizer))

    logger.info("loading model is over!")
    # model = BertForMaskedLM(config) 从头预训练
    model.to(device)

    if args.distributed_training:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank,find_unused_parameters=True)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
        'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    if args.fp16:
        pass
    else:
        # optimizer = BertAdam(optimizer_grouped_parameters, lr=args.learning_rate, warmup=args.warmup_proportion,t_total=num_train_optimization_steps)
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # scheduler = get_linear_schedule_with_warmup(optimizer)
    scheduler = get_linear_schedule_with_warmup(optimizer,num_training_steps=num_train_optimization_steps,num_warmup_steps = num_train_optimization_steps*args.warmup_proportion)
    TrainDataset = PretrainDataset(train_data,tokenizer,args.prompt_pattern_list,args.label_list,args.max_seq_length,args.masked_lm_prob,args.max_predictions_per_seq,vocab_list)
    EvalDataset = PretrainDataset(eval_data,tokenizer,args.prompt_pattern_list,args.label_list,args.max_seq_length,args.masked_lm_prob,args.max_predictions_per_seq,vocab_list)


    # if local_rank == -1:
    #     train_sampler = RandomSampler(TrainDataset)
    # else:
    #     train_sampler = DistributedSampler(TrainDataset)
    # eval_sampler = SequentialSampler(EvalDataset)
    # TrainDataLoader = DataLoader(TrainDataset,batch_size=args.train_batch_size,shuffle=True,sampler=train_sampler)
    # EvalDataLoader = DataLoader(EvalDataset,batch_size=args.eval_batch_size,shuffle=False,sampler=SequentialSampler)

    TrainDataLoader = DataLoader(TrainDataset,batch_size=args.train_batch_size,shuffle=True)
    EvalDataLoader = DataLoader(EvalDataset,batch_size=args.eval_batch_size,shuffle=False)

    global_step = 0
    best_loss = 100000
    patience = 0
    logging.info("***** Running training *****")
    logging.info("  Num examples = %d", len(TrainDataset))
    logging.info("  Batch size = %d", args.train_batch_size)
    logging.info("  Num steps = %d", num_train_optimization_steps)

    logger.info("the pre-training data is alreadly")

    model.train()
    
    for e in trange(int(args.num_train_epochs), desc="Epoch"):
        if patience > args.early_stopping:
            break
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        optimizer.zero_grad()
        for step, batch in enumerate(tqdm(TrainDataLoader, desc="Iteration")):
            batch = tuple(t.to(device) for t in batch)
            input_ids, segment_ids, attention_ids, label_ids = batch
            # masked_lm_loss
            outputs = model(input_ids=input_ids, attention_mask=attention_ids, token_type_ids=segment_ids, labels=label_ids)
            loss = outputs.loss
            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                optimizer.backward(loss)
            else:
                loss.backward()

            tr_loss += loss.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

                global_step += 1
            if nb_tr_steps > 0 and nb_tr_steps % 100 == 0:
                logger.info("===================== -epoch %d -train_step %d -train_loss %.4f\n" % (e, nb_tr_steps, tr_loss / nb_tr_steps))
        if nb_tr_steps > 0:
            #################################EVAL#####################################################
            model.eval()
            eval_loss = 0.
            nb_eval_steps = 0
            for step, batch in enumerate(tqdm(EvalDataLoader, desc="Evaluating")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, segment_ids, attention_ids, label_ids = batch
                with torch.no_grad():
                    outputs = model(input_ids=input_ids, attention_mask=attention_ids, token_type_ids=segment_ids, labels=label_ids)
                    loss = outputs.loss
                eval_loss += loss.item()
                nb_eval_steps += 1

            eval_loss = eval_loss / nb_eval_steps
            if eval_loss < best_loss:
                patience = 0
                # Save a trained model, configuration and tokenizer
                print("The eval loss is decreasing!,so we save model!")
                model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

                # If we save using the predefined names, we can load using `from_pretrained`
#                         output_model_file = os.path.join(model_fold_path, WEIGHTS_NAME)
                output_model_file = os.path.join(args.model_save_path, 'pytorch_model.bin')
                torch.save(model_to_save.state_dict(), output_model_file)
                #torch.save(model.module, output_model_file)

                best_loss = eval_loss
            else:
                patience += 1
            print("============================ -epoch %d -train_loss %.4f -eval_loss %.4f\n"% (e, tr_loss / nb_tr_steps, eval_loss))
    exit(0)
if __name__=='__main__':
    main()







        