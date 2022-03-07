# -----------ARGS---------------------#

# model_name_from_hugging_face = 'hfl/chinese-roberta-wwm-ext-large'
model_name_from_hugging_face = 'bert-large-cased'

# label_list = ['entailment','not_entailment']
# label_list = ['entailment','neutral','contradiction']
# label_list = ['Support','Neutral','Against']

# prompt_pattern_list = [['是的','对的'],['可能','也许'],['不是','不对']]
# prompt_pattern_list = [['因为','由于'],['另外','同时'],['但是','可是','然而']]
# prompt_pattern_list = ['Yes','No']
# Given {prem} Should we assume that "{hypo}" is true? [mask]


superglue = {'RTE':{'data_path':['RTE/train.jsonl','RTE/val.jsonl'],
                'label_list':['entailment','not_entailment'],
                'prompt_answer_list':['Yes','No'],
                'hyperparameter_of_mask':{'max_predictions_per_seq':20,'masked_lm_prob':0.25}},
            'CB':{'data_path':['CB/train.jsonl','CB/val.jsonl'],
                'label_list':['entailment','neutral','contradiction'],
                'prompt_answer_list':['Yes','Maybe','No'],
                'hyperparameter_of_mask':{'max_predictions_per_seq':20,'masked_lm_prob':0.25}},
            'BoolQ':{'data_path':['BoolQ/train.jsonl','BoolQ/val.jsonl'],
                'label_list':[True,False],
                'prompt_answer_list':['Yes','No'],
                'hyperparameter_of_mask':{'max_predictions_per_seq':40,'masked_lm_prob':0.25}}}


model_save_path = 'continues_bert_large_cased_superglue'
do_train = True
do_lower_case = True
no_cuda = False
fp16=False
is_first_pretrain = False
do_eval = True
do_lower_case = True
do_whole_word_mask = False
distributed_training = False


max_seq_length = 256
seed = 42
gradient_accumulation_steps = 2
loss_scale = 0.
early_stopping = 5
train_batch_size = 64
eval_batch_size = 64
learning_rate = 2e-5
num_train_epochs = 30
warmup_proportion = 0.1
dynamic_mask_times= 10
soft_prompt_length = 0
mask_length = 1