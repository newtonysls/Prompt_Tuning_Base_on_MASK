# model_name_from_hugging_face = 'hfl/chinese-roberta-wwm-ext-large'
# model_name_from_hugging_face = 'bert-base-cased'
model_name_from_hugging_face = 'bert-large-cased'

# model_path = 'continues_bert_base_cased_superglue'

model_path = 'continues_bert_large_cased_superglue'
# model_path = 'bert-large-cased'





# train_data_path = ['RTE/train.jsonl']
# dev_data_path = ['RTE/val.jsonl']

model_save_path = 'protum_base'
task_type = 'RTE'
superglue = {'RTE':{'data_path':['RTE/train.jsonl','RTE/val.jsonl'],
                'label_list':['entailment','not_entailment'],
                'prompt_answer_list':['Yes','No'],
                'hyperparameter_of_mask':{'max_predictions_per_seq':20,'masked_lm_prob':0.25}},
            'CB':{'data_path':['CB/train.jsonl','CB/val.jsonl'],
                'label_list':['entailment','neutral','contradiction'],
                'prompt_answer_list':['Yes','Maybe','No'],
                'hyperparameter_of_mask':{'max_predictions_per_seq':20,'masked_lm_prob':0.2}},
            'BoolQ':{'data_path':['BoolQ/train.jsonl','BoolQ/val.jsonl'],
                'label_list':[True,False],
                'prompt_answer_list':['Yes','No'],
                'hyperparameter_of_mask':{'max_predictions_per_seq':30,'masked_lm_prob':0.2}}}

do_train = True
do_lower_case = True
no_cuda = False
fp16=False
is_first_pretrain = False
do_eval = True
do_lower_case = True
do_whole_word_mask = False
distributed_training = True

max_seq_length =128
seed =11
loss_scale = 0.
early_stopping = 10

train_batch_size = 32
eval_batch_size = 32
learning_rate = 1e-4
embeddings_learning_rate = 1e-3
num_train_epochs = 30
warmup_proportion = 0.1
soft_prompt_length =0
mask_length = 1
p = 0.5
drop_rate = 0.5
res_step = 7
res_start = 3