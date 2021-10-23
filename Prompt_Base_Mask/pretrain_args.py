# -----------ARGS---------------------#

# model_name_from_hugging_face = 'hfl/chinese-roberta-wwm-ext-large'
model_name_from_hugging_face = 'bert-base-chinese'

label_list = ['Support','Neutral','Against']
prompt_pattern_list = ['因为','并且','但是']
data_path = ['train.txt','test.txt']
model_save_path = 'continues_pretrain_model'
do_train = True
do_lower_case = True
no_cuda = False
fp16=False
is_first_pretrain = False
do_eval = True
do_lower_case = True
do_whole_word_mask = False
distributed_training = True


max_seq_length = 64
seed = 42
gradient_accumulation_steps = 2
loss_scale = 0.
max_predictions_per_seq = 20
early_stopping = 3
short_seq_prob = 0.15
masked_lm_prob = 0.2
train_batch_size = 64
eval_batch_size = 8
learning_rate = 2e-5
num_train_epochs = 20
warmup_proportion = 0.1
dynamic_mask_times=5