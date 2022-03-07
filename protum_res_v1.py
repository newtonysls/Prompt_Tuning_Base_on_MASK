from __future__ import absolute_import, division, print_function
# 放在第一句，不然会报错

import collections
from inspect import Parameter
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
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from tqdm import tqdm, trange
from random import random, randrange, randint, shuffle, choice, sample
# from transformers import BertModel,BertConfig,BertTokenizer,BertForMaskedLM
from transformers import BertTokenizer
from transformers.models.bert.modeling_bert import (BertEmbeddings,
                                                    BertConfig,
                                                    BertLayer,
                                                    BertPooler,
                                                    BaseModelOutputWithPastAndCrossAttentions,
                                                    BaseModelOutputWithPoolingAndCrossAttentions,
                                                    BertPreTrainedModel,
                                                    BERT_INPUTS_DOCSTRING,
                                                    add_code_sample_docstrings,
                                                    add_start_docstrings_to_model_forward)

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
import protum_args_v1 as args
import re
import json
from sklearn.model_selection import KFold
from sklearn.metrics  import f1_score

# 
_CHECKPOINT_FOR_DOC = "bert-base-uncased"
_CONFIG_FOR_DOC = "BertConfig"
_TOKENIZER_FOR_DOC = "BertTokenizer"

BERT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "bert-base-uncased",
    "bert-large-uncased",
    "bert-base-cased",
    "bert-large-cased",
    "bert-base-multilingual-uncased",
    "bert-base-multilingual-cased",
    "bert-base-chinese",
    "bert-base-german-cased",
    "bert-large-uncased-whole-word-masking",
    "bert-large-cased-whole-word-masking",
    "bert-large-uncased-whole-word-masking-finetuned-squad",
    "bert-large-cased-whole-word-masking-finetuned-squad",
    "bert-base-cased-finetuned-mrpc",
    "bert-base-german-dbmdz-cased",
    "bert-base-german-dbmdz-uncased",
    "cl-tohoku/bert-base-japanese",
    "cl-tohoku/bert-base-japanese-whole-word-masking",
    "cl-tohoku/bert-base-japanese-char",
    "cl-tohoku/bert-base-japanese-char-whole-word-masking",
    "TurkuNLP/bert-base-finnish-cased-v1",
    "TurkuNLP/bert-base-finnish-uncased-v1",
    "wietsedv/bert-base-dutch-cased",
    # See all BERT models at https://huggingface.co/models?filter=bert
]

processor_type = {'RTE':RTE_Processing,
                'CB':CB_Processing,
                'BoolQ':BoolQ_Processing}

def GetMask(hidden_states,masked_positions):
    """
    hidden_states:bs * seq * hidden_size
    """
    mask_hidden_states = torch.stack([hidden_states[i,masked_positions[i]] for i in range(masked_positions.size(0))])
    mask_hidden_states = mask_hidden_states[:,0,:]
    return mask_hidden_states

def Addition(hidden_states,masked_positions,res_states):
    for i in range(res_states.size(0)):
        hidden_states[i,masked_positions[i]] = hidden_states[i,masked_positions[i]].add(res_states[i])
    return hidden_states

class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
        # self.classifier = nn.Sequential(nn.Dropout(args.drop_rate),nn.Linear(config.hidden_size, nums_label),nn.Tanh())
        self.res_step = args.res_step
        # 在这里建议res_step和res_start都使用外部参数
        # self.reslayer = nn.ModuleList([nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),nn.BatchNorm1d(config.hidden_size),nn.ReLU()) for _ in range(int(config.num_hidden_layers / self.res_step))])
        self.reslayer = nn.ModuleList([nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),nn.ReLU()) for _ in range(int(config.num_hidden_layers / self.res_step))])
        # self.res_start = int(config.num_hidden_layers / self.res_step/2)
        self.res_start = args.res_start
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        masked_positions,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        res_output = None
        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None
            
            if self.gradient_checkpointing and self.training:

                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            hidden_states = layer_outputs[0]
            if i==self.res_step * self.res_start:
                res_output = all_hidden_states[-1]
            if (i+1)%self.res_step == 0 :
                if res_output is not None:
                    last_hidden_states = hidden_states
                    res_input = GetMask(res_output,masked_positions) #bs * hidden_size
                    res_output = self.reslayer[int(i/self.res_step)](res_input)
                    res_output = Addition(last_hidden_states,masked_positions,res_output)
            
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,res_output,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )

class BertModel(BertPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in `Attention is
    all you need <https://arxiv.org/abs/1706.03762>`__ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the :obj:`is_decoder` argument of the configuration
    set to :obj:`True`. To be used in a Seq2Seq model, the model needs to initialized with both :obj:`is_decoder`
    argument and :obj:`add_cross_attention` set to :obj:`True`; an :obj:`encoder_hidden_states` is then expected as an
    input to the forward pass.
    """

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)

        self.pooler = BertPooler(config) if add_pooling_layer else None

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPoolingAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        masked_positions=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If :obj:`past_key_values` are used, the user can optionally input only the last :obj:`decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
            instead of all :obj:`decoder_input_ids` of shape :obj:`(batch_size, sequence_length)`.
        use_cache (:obj:`bool`, `optional`):
            If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
            decoding (see :obj:`past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            masked_positions,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )

class ProtumDataset(Dataset):
    def __init__(self,data,max_seq_length):
        self.max_seq_length = max_seq_length
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self,index):
        data_item = self.data[index]
        input_ids = data_item['input_ids']
        segment_ids = data_item['segment_ids']
        attention_ids = data_item['attention_ids']
        prompt_positions = data_item['prompt_positions']
        label_ids = data_item['label_ids']

        assert len(input_ids)==len(segment_ids)==len(attention_ids)
        padding_length = self.max_seq_length - len(input_ids)
        input_ids += [0] * padding_length
        attention_ids += [0] * padding_length
        segment_ids += [0] * padding_length
        
        input_ids = torch.tensor(input_ids,dtype=torch.long)
        segment_ids = torch.tensor(segment_ids,dtype=torch.long)
        attention_ids = torch.tensor(attention_ids,dtype=torch.long)
        prompt_positions = torch.tensor(prompt_positions,dtype=torch.long)
        label_ids = torch.tensor(label_ids,dtype=torch.long)
        return input_ids, segment_ids, attention_ids, prompt_positions,label_ids

def freeze(layer):
    for child in layer.children():
        for param in child.parameters():
            param.requires_grad = False
            
def unfreeze(layer):
    for child in layer.children():
        for param in child.parameters():
            param.requires_grad = True   
                   
class PromptForClassification(nn.Module):

    def __init__(self,config,model_name,nums_label):
        super(PromptForClassification,self).__init__()
        self.config = config
        self.model_name = model_name
        self.nums_label = nums_label
        self.bert = BertModel(config)
        # self.bert = BertModel.from_pretrained(model_name)
        # self.mlm = BertForMaskedLM.from_pretrained(model_name)
        
        if False:
            self.prompt_embeddings = nn.Embedding(args.mask_length,config.hidden_size).from_pretrained(self.mlm.bert.embeddings.word_embeddings.weight[103].view(1,-1))
            self.lstm_head = torch.nn.LSTM(input_size=config.hidden_size,hidden_size=config.hidden_size,
                               num_layers=2,
                               bidirectional=True,
                               batch_first=True)
            self.mlp_head = nn.Sequential(nn.Linear(2 * config.hidden_size, config.hidden_size),
                                          nn.ReLU(),
                                          nn.Linear(config.hidden_size, config.hidden_size))

        # bert_embeddings = self.bert.embeddings.word_embeddings.weight
        # for i in range(args.soft_prompt_length):
        #   self.prompt_embeddings.weight[i] = bert_embeddings[i+1]
        #   if True:
        #       self.bert.load_state_dict(torch.load(args.pretrain_path))
        # freeze(self.mlm)
        freeze(self.bert)
        unfreeze(self.bert.encoder.reslayer)
        # freeze(self.cls)
    
        # unfreeze(self.bert.encoder.layer[-3])
        # self.dropout = nn.Dropout(0.5)
        # self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # self.activation = nn.Tanh()
        if False:
            self.lstm_mask = torch.nn.LSTM(input_size=config.hidden_size,hidden_size=config.hidden_size,
                               num_layers=2,
                               bidirectional=True,
                               batch_first=True)
            self.dense = nn.Sequential(nn.Linear(2 * config.hidden_size, config.hidden_size),
                                          nn.Dropout(args.drop_rate),
                                          nn.ReLU(),
                                          nn.Linear(config.hidden_size, nums_label))
        self.classifier = nn.Sequential(nn.Dropout(args.drop_rate),nn.Linear(config.hidden_size, nums_label),nn.Tanh())
        # self.res_step = 3
         
        # self.reslayer = nn.ModuleList([nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),nn.BatchNorm1d(config.hidden_size),nn.ReLU()) for _ in range(int(config.num_hidden_layers / self.res_step))])
        # self.reslayer = nn.ModuleList([nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),nn.ReLU()) for _ in range(int(config.num_hidden_layers / self.res_step))])
        # self.project = 8
        # self.reslayer = nn.ModuleList([nn.Sequential(nn.Linear(config.hidden_size, self.project),nn.ReLU(),
                                #    nn.Linear(self.project, self.project),nn.ReLU(),
                                #    nn.Linear(self.project, config.hidden_size),nn.ReLU()) for _ in range(int(config.num_hidden_layers / self.res_step))])
        # self.res_start = int(config.num_hidden_layers / self.res_step/2)
        # self.res_start = 4

        # self.classifier = nn.Linear(config.hidden_size * 2, nums_label)
        # self.classifier = nn.Linear(config.hidden_size * 4, nums_label)
        # self.classifier = nn.Linear(config.hidden_size, nums_label)
    def forward(self,input_ids,segment_ids,attention_masks,masked_positions):
        if False:
            if False:
                raw_embeds = self.mlm.bert.embeddings.word_embeddings(input_ids)
                replace_embeds = self.prompt_embeddings(torch.LongTensor(list(range(args.mask_length))).cuda())
                replace_embeds = replace_embeds.unsqueeze(0) # [batch_size, prompt_length, embed_size]

                replace_embeds = self.lstm_head(replace_embeds)[0]  # [batch_size, seq_len, 2 * hidden_dim]
                if args.mask_length == 1:
                    replace_embeds = self.mlp_head(replace_embeds)
                else:
                    replace_embeds = self.mlp_head(replace_embeds).squeeze()# [batch_size, seq_len, hidden_dim]

                for bidx in range(input_ids.size(0)):
                    for i in range(args.mask_length):
                        raw_embeds[bidx, masked_positions[bidx][i], :] = replace_embeds[i, :]
            else:
                raw_embeds = self.mlm.bert.embeddings.word_embeddings(input_ids)
                mask_embedding = torch.stack([raw_embeds[i,masked_positions[i]] for i in range(masked_positions.size(0))])
                mask_embedding = mask_embedding[:,0,:]
                mask_embedding = self.reslayer(mask_embedding)
                for i in range(masked_positions.size(0)):
                    raw_embeds[i,masked_positions[i][0],:].add_(mask_embedding[i])
            ##############################################################        
            outputs = self.mlm.bert(inputs_embeds=raw_embeds,token_type_ids = segment_ids,attention_mask = attention_masks,output_hidden_states=True)
        ##############################################################        
        else:
            outputs = self.bert(input_ids=input_ids,token_type_ids = segment_ids,attention_mask = attention_masks,masked_positions=masked_positions,output_hidden_states=True)
            # outputs = self.mlm.bert(input_ids=input_ids,token_type_ids = segment_ids,attention_mask = attention_masks,output_hidden_states=True)
            
        ##############################################################
        
        sequence_output = outputs.last_hidden_state
        # prediction_scores = self.mlm.cls(sequence_output)
        hidden_states = outputs.hidden_states
        # pooler_output = outputs.pooler_output
        sequence_output = hidden_states[-2]
        # mask_output = sequence_output[:,[4,5],:]
        res_output = hidden_states[-1]
        h12 = hidden_states[-2]
        h11 = hidden_states[-3]
        h10 = hidden_states[-4]
        h09 = hidden_states[-5]

        # for i in range(self.config.num_hidden_layers):
        if False:
            mask_reslayer_output = None
            for i in range(int(self.config.num_hidden_layers / self.res_step)):
                if i < self.res_start:
                    continue
                index_hidden_layer = i * self.res_step
                if i==self.res_start:
                    mask_hidden_states = torch.stack([hidden_states[index_hidden_layer][i,masked_positions[i]] for i in range(masked_positions.size(0))])
                    mask_hidden_states = mask_hidden_states[:,0,:]
                else:
                    mask_hidden_states = mask_reslayer_output
                mask_reslayer_output = self.reslayer[i](mask_hidden_states)
                mask_bert_output = torch.stack([hidden_states[index_hidden_layer+self.res_step][i,masked_positions[i]] for i in range(masked_positions.size(0))])
                mask_reslayer_output = mask_reslayer_output.add(mask_bert_output[:,0,:])
        # mask_output_all = torch.stack([torch.stack([hidden_states[j][i,masked_positions[i]] for i in range(masked_positions.size(0))])[:,0,:] for j in range(self.config.num_hidden_layers-4,self.config.num_hidden_layers)],dim=1)
        # [batch_size, num_hidden_layers, hidden_size]
        
        # lstm_output = self.lstm_mask(mask_output_all)[0][:,-1,:] # [batch_size, 2 * hidden_size]
        # logits = self.dense(lstm_output) # [batch_size, nums_label]
        # for i in range(self.config.num_hidden_layers):
        #     for j in range(masked_positions.size(0)):
        #         mask_output_all.append(hidden_states[i][j,masked_positions[i]])
        # mask_output_all = torch.stack([hidden_states[i,masked_positions[i]] for i in range(masked_positions.size(0))])

        mask_output12 = torch.stack([h12[i,masked_positions[i]] for i in range(masked_positions.size(0))])
        mask_output11 = torch.stack([h11[i,masked_positions[i]] for i in range(masked_positions.size(0))])
        mask_output10 = torch.stack([h10[i,masked_positions[i]] for i in range(masked_positions.size(0))])
        mask_output09 = torch.stack([h09[i,masked_positions[i]] for i in range(masked_positions.size(0))])
        # mlm_output = torch.stack([prediction_scores[i,masked_positions[i]] for i in range(masked_positions.size(0))])
        res_mask_output = torch.stack([res_output[i,masked_positions[i]] for i in range(masked_positions.size(0))])

        # concat_hidden = torch.cat((mask_output11,mask_output10),dim=2)
        # concat_hidden = torch.stack([mask_output12[:,0,:],mask_output11[:,0,:],mask_output10[:,0,:],mask_output09[:,0,:]])
                
        # mask_output = torch.stack([sequence_output[i,masked_positions[i]] for i in range(masked_positions.size(0))])
        # mask_output = torch.stack([sequence_output[i,[4,5],:] for i in range(masked_positions.size(0))])
        
        # pooling
        # mask_output = mask_output10
        # mask_output = self.dense(mask_output)
        # mask_output = self.activation(mask_output)
        # mask_output =torch.cat((mask_output12, mask_output11,mask_output10,mask_output09),dim=1)
        # maxpool = torch.max(mask_output, dim=1)[0]
        # minpool = torch.min(mask_output, dim=1)[0]

        # avgpool = torch.mean(mask_output, dim=1)
        # concat_out = maxpool

        # concat_out = mask_output[:,0,:]
        concat_out = res_mask_output[:,0,:]

        # res_output = self.reslayer(mask_output[:,0,:])
        # concat_out = mask_output12[:,0,:].add(res_output)
        # concat_out = mask_output[:,0,:].add(mask_output12[:,0,:])
        # print(pooler_output.size())
        # concat_out = torch.cat((mask_output[:,0,:], sequence_output[:,0,:]),dim=1)
        # concat_out = torch.cat((maxpool,minpool),dim=1)
        
        # concat_out = self.dropout(concat_out)
        logits = self.classifier(concat_out)
        probability = nn.functional.softmax(logits)
        return logits, probability,res_output

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

    # seed everything,python,numpy and torch
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
    tokenizer = BertTokenizer.from_pretrained(args.model_name_from_hugging_face)

    vocab_list = list(tokenizer.vocab.keys())
    config = BertConfig.from_pretrained(args.model_name_from_hugging_face)
    
    model_save_path = args.model_save_path 
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    train_data_path = args.superglue[args.task_type]['data_path'][0]
    dev_data_path = args.superglue[args.task_type]['data_path'][1]

    train_data = Json_File_Reader(args.task_type,train_data_path)
    eval_data = Json_File_Reader(args.task_type,dev_data_path)
    num_train_optimization_steps = None

    num_train_optimization_steps = int(len(train_data) / args.train_batch_size) * args.num_train_epochs
    if local_rank != -1:
        num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    # 定义持续预训练模型
    # model = BertModel.from_pretrained(args.model_name_from_hugging_face)
    model = PromptForClassification(config,args.model_path,len(args.superglue[args.task_type]['label_list']))

    # update parameters from pretrained models
    model_parameters = model.state_dict()
    pretrained_parameters = torch.load(os.path.join(args.model_path,'pytorch_model.bin'))
    parameters_updating = {k:v for k,v in pretrained_parameters.items() if k in model_parameters.keys()}
    model_parameters.update(parameters_updating)
    model.load_state_dict(model_parameters)
    # model.bert.resize_token_embeddings(len(tokenizer))

    logger.info("loading model weights is finished!")
    # model = BertForMaskedLM(config) 从头预训练

    model.to(device)

    if args.distributed_training:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank,find_unused_parameters=True)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    #         param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],
        'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad], 'weight_decay': 0.0}
    ]
    #     embedding_parameters = [
    #             {'params': [p for p in model.module.lstm_head.parameters()]},
    #             {'params': [p for p in model.module.mlp_head.parameters()]},
    #             {'params': [p for p in model.module.prompt_embeddings.parameters()]}

    #     ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer,num_training_steps=num_train_optimization_steps,num_warmup_steps = num_train_optimization_steps*args.warmup_proportion)
    
    # embedding_optimizer = AdamW(embedding_parameters, lr=args.embeddings_learning_rate)
    # embedding_scheduler = get_linear_schedule_with_warmup(embedding_optimizer, num_warmup_steps=num_train_optimization_steps*args.warmup_proportion, num_training_steps=num_train_optimization_steps)

    loss_fct = CrossEntropyLoss()

    processor = processor_type[args.task_type]
    train_processor = processor(train_data,
                     tokenizer,
                     args.max_seq_length,
                     vocab_list,
                     is_pretraining=False)
    train_inputs = train_processor.Creat_Input_For_PLMs()
    TrainDataset = ProtumDataset(train_inputs,args.max_seq_length)
    eval_processor = processor(eval_data,
                     tokenizer,
                     args.max_seq_length,
                     vocab_list,
                     is_pretraining=False)
    eval_inputs = eval_processor.Creat_Input_For_PLMs()
    EvalDataset = ProtumDataset(eval_inputs,args.max_seq_length)

    TrainDataLoader = DataLoader(TrainDataset,batch_size=args.train_batch_size,shuffle=True)
    EvalDataLoader = DataLoader(EvalDataset,batch_size=args.eval_batch_size,shuffle=False)

    global_step = 0
    best_acc = 0.
    best_f1 = 0.
    best_loss = 1000000.
    patience = 0
    logging.info("***** Running training *****")
    logging.info("  Num examples = %d", len(TrainDataset))
    logging.info("  Batch size = %d", args.train_batch_size)
    logging.info("  Num steps = %d", num_train_optimization_steps)

    logger.info("the pre-training data is alreadly")
    optimizer.zero_grad()
    #     embedding_optimizer.zero_grad()

    model.train()

    for e in trange(int(args.num_train_epochs), desc="Epoch"):
        if patience > args.early_stopping:
            break
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        model.train()
        for step, batch in enumerate(tqdm(TrainDataLoader, desc="Iteration")):
            batch = tuple(t.to(device) for t in batch)
            input_ids, segment_ids, attention_ids, masked_positions, label_ids = batch
            # masked_lm_loss
            logits,probability, mlm_output = model(input_ids=input_ids,segment_ids=segment_ids,attention_masks=attention_ids,masked_positions=masked_positions)
            # loss = outputs.loss
            loss = loss_fct(logits, label_ids)
            # loss = (1-args.p)*loss_fct(logits, label_ids)-args.p*loss_fct(mlm_output.view(-1, config.vocab_size), template_ids.view(-1))
            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.

            tr_loss += loss.item()
            loss.backward()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1

            optimizer.step()
            scheduler.step()
            
            # embedding_optimizer.step()
            # embedding_scheduler.step()
            
            optimizer.zero_grad()
            # embedding_optimizer.zero_grad()

            global_step += 1
            if nb_tr_steps > 0 and nb_tr_steps % 50 == 0:
                logger.info("===================== -epoch %d -train_step %d -train_loss %.4f\n" % (e, nb_tr_steps, tr_loss / nb_tr_steps))

        if nb_tr_steps > 0:
            #################################EVAL#####################################################
            model.eval()
            accuarcy = 0.
            eval_predict = np.array([],dtype=int)
            eval_labels = np.array([],dtype=int)

            for step, batch in enumerate(tqdm(EvalDataLoader, desc="Evaluating")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, segment_ids, attention_ids, masked_positions, label_ids = batch
                with torch.no_grad():
                    logits,probability,_ = model(input_ids=input_ids,segment_ids=segment_ids,attention_masks=attention_ids,masked_positions=masked_positions)
                prediction_each_epoch = np.argmax(probability.detach().to("cpu").numpy(), axis=1)
                eval_predict = np.concatenate([eval_predict,prediction_each_epoch])
                eval_labels = np.concatenate([eval_labels,label_ids.detach().to("cpu").numpy()])
            accuarcy = sum(eval_predict == eval_labels) / np.size(eval_labels,0)
            f1 = f1_score(eval_labels,eval_predict,average = 'macro')
            
            if accuarcy > best_acc:
                patience = 0
                # Save a trained model, configuration and tokenizer
                # print("The eval loss is decreasing!,so we save model!")
                print("The eval accuracy is increasing!,so we save model!")

                model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

                # If we save using the predefined names, we can load using `from_pretrained`
                # output_model_file = os.path.join(model_fold_path, WEIGHTS_NAME)
                output_model_file = os.path.join(model_save_path, 'pytorch_model.bin')
                torch.save(model_to_save.state_dict(), output_model_file)
                tokenizer.save_vocabulary(os.path.join(model_save_path,'vocab.txt'))
                config.to_json_file(os.path.join(model_save_path,'config.json'))
            #torch.save(model.module, output_model_file)

                best_acc = accuarcy
                print
            else:
                patience += 1
            if f1 > best_f1:
                best_f1 = f1
            print("============================ -epoch %d -train_loss %.4f -eval_acc %.4f -best_acc %.4f\n -f1 %.4f -best-f1 %.4f "% (e, tr_loss / nb_tr_steps, accuarcy,best_acc,f1,best_f1))
    exit(0)
if __name__=='__main__':
    main()

        