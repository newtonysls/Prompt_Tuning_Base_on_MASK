from __future__ import absolute_import, division, print_function
# 放在第一句，不然会报错

import logging
from operator import index
import os
import random
from matplotlib.pyplot import cla, title
import numpy as np
import pandas as pd
import csv

import torch
import torch.nn as nn 
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, Dataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from random import random, randrange, randint, shuffle, choice, sample

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
from data_processor import RTE_Processing,CB_Processing,BoolQ_Processing,Json_File_Reader
import multi_tasks_pretrain_args as args
import json
from sklearn.model_selection import KFold

# Random Mask
def create_masked_lm_predictions(tokens, masked_lm_prob, max_predictions_per_seq, vocab_list):
    """Creates the predictions for the masked LM objective. This is mostly copied from the Google BERT repo, but
    with several refactors to clean it up and remove a lot of unnecessary variables."""
    cand_indices = []
    for (i, token) in enumerate(tokens):
        if token == "[CLS]" or token == "[SEP]":
            continue
        cand_indices.append(i)

    num_to_mask = min(max_predictions_per_seq,
                      max(1, int(round(len(tokens) * masked_lm_prob))))
    # print(num_to_mask)
    # print("tokens", len(tokens))
    # print("cand", len(cand_indices))
    shuffle(cand_indices)
    mask_indices = sorted(sample(cand_indices, num_to_mask))
    masked_token_labels = []
    for index in mask_indices:
        # 80% of the time, replace with [MASK]
        if random() <= 1:
            masked_token = "[MASK]"
        else:
            # 10% of the time, keep original
            if random() < 0.5:
                masked_token = tokens[index]
            # 10% of the time, replace with random word
            else:
                masked_token = choice(vocab_list)
        masked_token_labels.append(tokens[index])
        # Once we've saved the true label for that token, we can overwrite it with the masked version
        tokens[index] = masked_token

    return tokens, mask_indices, masked_token_labels

# def Divide_Data(data_list,eval_rate):
#     """
#     data_list: a list contains a dict of item of input for PLMs
#     It is not approporite for those task whose contains very few data. In other words, It is unfair.
#     """
#     kf = KFold(n_splits=int(1/eval_rate),shuffle=True,random_state=args.seed)
#     train_data = []
#     eval_data = []

#     for step,(train_index,eval_index) in enumerate(kf.split(data_list)):
#         for i in train_index:
#             train_data.append(data_list[i])
#         for i in eval_index:
#             eval_data.append(data_list[i])
#         if step >= 0:
#             break
#     return train_data,eval_data
def Divide_Data(data_list,eval_rate):
    assert eval_rate > 0 and eval_rate < 1
    step = int(1/eval_rate)
    train_data = [data_list[i] for i in range(len(data_list)) if i % step != 0]
    eval_data = [data_list[i] for i in range(len(data_list)) if i % step == 0]
    return train_data,eval_data

class PretrainDataset(Dataset):
    def __init__(self,data,tokenizer,superglue,max_seq_length):
        self.max_seq_length = max_seq_length
        self.data = data
        self.tokenizer = tokenizer
        self.superglue = superglue
        self.vocab_list = list(tokenizer.vocab.keys())
    def __len__(self):
        return len(self.data)

    def __getitem__(self,index):
        data_item = self.data[index]
        tokens = data_item['tokens'].copy()
        # input_ids = data_item['input_ids']
        segment_ids = data_item['segment_ids'].copy()
        attention_ids = data_item['attention_ids'].copy()
        # prompt_positions = data_item['prompt_positions']
        data_name = data_item['data_name']


        assert len(tokens)==len(segment_ids)==len(attention_ids)

        tokens, masked_lm_positions, masked_lm_labels = create_masked_lm_predictions(tokens, self.superglue[data_name]['hyperparameter_of_mask']['masked_lm_prob'], self.superglue[data_name]['hyperparameter_of_mask']['max_predictions_per_seq'], self.vocab_list)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        padding_length = self.max_seq_length - len(input_ids)

        input_ids += [0] * padding_length
        attention_ids += [0] * padding_length
        segment_ids += [0] * padding_length
        label_ids = np.full(self.max_seq_length, dtype=np.int32, fill_value=-100)
        label_ids[masked_lm_positions] = self.tokenizer.convert_tokens_to_ids(masked_lm_labels)

        input_ids = torch.tensor(input_ids,dtype=torch.long)
        segment_ids = torch.tensor(segment_ids,dtype=torch.long)
        attention_ids = torch.tensor(attention_ids,dtype=torch.long)
        label_ids = torch.tensor(label_ids,dtype=torch.long)
        return input_ids, segment_ids, attention_ids, label_ids

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

    tokenizer = BertTokenizer.from_pretrained(args.model_name_from_hugging_face)
    vocab_list = list(tokenizer.vocab.keys())
    config = BertConfig.from_pretrained(args.model_name_from_hugging_face)

    # read three type of tasks of superglue data from json file.
    RTE_data = Json_File_Reader('RTE',args.superglue['RTE']['data_path'],args.dynamic_mask_times)
    CB_data = Json_File_Reader('CB',args.superglue['CB']['data_path'],args.dynamic_mask_times)
    BoolQ_data = Json_File_Reader('BoolQ',args.superglue['BoolQ']['data_path'],args.dynamic_mask_times)
    RTE = RTE_Processing(RTE_data,tokenizer,
                     args.max_seq_length,
                     vocab_list)
    CB = CB_Processing(CB_data,tokenizer,
                        args.max_seq_length,
                        vocab_list)
    BoolQ = BoolQ_Processing(BoolQ_data,tokenizer,
                        args.max_seq_length,
                        vocab_list)

    # for these kinds of down-stream tasks,creating the correct form of input data to PLMs.
    RTE_inputs = RTE.Creat_Input_For_PLMs()
    shuffle(RTE_inputs)
    CB_inputs = CB.Creat_Input_For_PLMs()
    shuffle(CB_inputs)
    BoolQ_inputs = BoolQ.Creat_Input_For_PLMs()
    shuffle(BoolQ_inputs)

    # using the KFOLD to divide all data to train and validation dataset.
    train_data,eval_data = Divide_Data(RTE_inputs+CB_inputs+BoolQ_inputs,0.1)
    # train_data,eval_data = Divide_Data(RTE_inputs,0.1)


    num_train_optimization_steps = None

    num_train_optimization_steps = int(len(train_data) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
    if local_rank != -1:
        num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    
    # creating pretraining MLM model
    model = BertForMaskedLM.from_pretrained(args.model_name_from_hugging_face)
    # There is no need for model to resizing token embeddings if we don't add any additional tokens to vocabulary

    logger.info("loading model is over!")
    # model = BertForMaskedLM(config) pretraining a BERT from scratch
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
    
    TrainDataset = PretrainDataset(train_data,tokenizer,args.superglue,args.max_seq_length)
    EvalDataset = PretrainDataset(eval_data,tokenizer,args.superglue,args.max_seq_length)
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

    with open(os.path.join(args.model_save_path,'readme.txt'),'w+') as f:
        f.write('The training data includes RTE, CB and BoolQ. \
        Considering the varity of the length of sentences for different downstream tasks, we deside to use the appropriate hyperparameters for masked strategy!\n')
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
                # output_model_file = os.path.join(model_fold_path, WEIGHTS_NAME)
                output_model_file = os.path.join(args.model_save_path, 'pytorch_model{:.3f}.bin'.format(eval_loss))
                torch.save(model_to_save.state_dict(), output_model_file)
                tokenizer.save_vocabulary(os.path.join(args.model_save_path,'vocab.txt'))
                config.to_json_file(os.path.join(args.model_save_path,'config.json'))
                #torch.save(model.module, output_model_file)

                best_loss = eval_loss
                
            else:
                patience += 1
            print("============================ -epoch %d -train_loss %.4f -eval_loss %.4f\n"% (e, tr_loss / nb_tr_steps, eval_loss))

        if best_loss<=0.1:
            exit(0)

if __name__=='__main__':
    main()

   