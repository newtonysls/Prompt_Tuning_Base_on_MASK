# Prompt_Tuning_Base_on_MASK
In our opinion, during the processing of pre-training, the special token "[MASK]" has learned much more informations than others for predicting the correct word undering the MASK. In this case, in our work, we make giant efforts on this issue and research the impact of "[MASK]" to language models. As result, we find prompt tuning based on MASK is faster and better than finetune and the current prompt tuning methods. Besides, we also use the method we proposed in many down-stream tasks to explore the performance in diffierent scene. Finally, our method archive SOTA results to prove the effectives.

## And we already prepare submit our reaseach to ICML2022


## experiments
P-tuning 新范式

| lr   | bs   | method                                  | epochs | ACC        | 备注                               |
| ---- | ---- | --------------------------------------- | ------ | ---------- | ---------------------------------- |
|      |      | baseline                                | 2      | 0.763      | bert-base-chinese                  |
|      |      | baseline                                | 2      | 0.788      | hfl/chinese-roberta-wwm-ext-large  |
|      |      | BertForClassification                   | 2      | 0.709/0.34 |                                    |
| 2e-5 | 8    | hidden_state[-1]，MAX                   | 2      | 0.779      | hfl/chinese-roberta-wwm-ext-large  |
| 2e-5 | 16   | hidden_state[-1]，AVG                   | 2      | 0.785      | hfl/chinese-roberta-wwm-ext-large  |
| 2e-5 | 8    | hidden_state[-1]，MAX                   | 2      | 0.747      | bert-base-chinese                  |
| 2e-5 | 16   | hidden_state[-1]，AVG                   | 2      | 0.747      | 对于MAX，batch更适合8              |
| 2e-5 | 16   | hidden_state[-2]，AVG                   | 2      | 0.743      | 有降低                             |
| 2e-5 | 8    | hidden_state[-2]，MAX                   | 2      | 0.739      | 有降低                             |
| 2e-5 | 8    | hidden_state[-3]，MAX                   | 2      | 0.747      |                                    |
| 2e-5 | 8    | hidden_state[-3]，MAX                   | 2      | 0.735      |                                    |
| 2e-5 | 16   | hidden_state[-3]，AVG                   | 2      | 0.734      | 逐渐减低，学到的信息更少           |
| 2e-5 | 16   | hidden_state[-4]，AVG                   | 2      | 0.732      |                                    |
| 2e-5 | 8    | hidden_state[-4]，MAX                   | 2      | 0.717      |                                    |
| 2e-5 | 16   | hidden_state[-1]，AVG + MAX             | 2      | 0.747      |                                    |
| 2e-5 | 8    | hidden_state[-1]，AVG + MAX             | 2      | 0.742      |                                    |
| 2e-5 | 16   | hidden_state[-1]+[-2]，AVG              | 2      | 0.744      | 效果不是很好，但是epoch效果为0.756 |
| 2e-5 | 8    | hidden_state[-1]+[-2]，MAX              | 2      | 0.744      | 效果不是很好，但是epoch效果为0.753 |
| 2e-5 | 8    | hidden_state[-1]AVG+hidden_state[-2]MAX | 2      | 0.742      |                                    |
| 2e-5 | 16   | hidden_state[-1]AVG+hidden_state[-2]MAX | 2      | 0.742      | 第一个epoch0.768                   |







随便采取某个位置的output进行分类实验：

| lr   | bs   | method                | epochs | ACC   | 备注                                          |
| ---- | ---- | --------------------- | ------ | ----- | --------------------------------------------- |
| 2e-5 | 16   | hidden_state[-1]，AVG | 2      | 0.714 | 带prompt MASK，取任意两个位置output进行分类   |
| 2e-5 | 16   | hidden_state[-1]，AVG | 2      | 0.698 | 不带prompt MASK，取任意两个位置output进行分类 |
| 2e-5 | 16   | pooler out [CLS]      | 2      | 0.713 |                                               |

> 该实验证明了基于MASK的p-tuning的有效性，学到了更多的知识



进行冻结P-tuning实验

| lr   | bs   | method                                    | epochs | ACC        | 备注                                 |
| ---- | ---- | ----------------------------------------- | ------ | ---------- | ------------------------------------ |
| 2e-5 | 16   | hidden_state[-1]，AVG                     | 10     | 0.479      | bert-base-chinese                    |
| 3e-5 | 16   | hidden_state[-1]，AVG                     | 15     | 0.503      |                                      |
| 4e-5 | 16   | hidden_state[-1]，AVG                     | 16     | 0.512      |                                      |
| 1e-4 | 16   | hidden_state[-1]，AVG                     | 16     | 0.543      |                                      |
| 2e-4 | 16   | hidden_state[-1]，AVG                     | 16     | 0.556      |                                      |
| 4e-4 | 16   | hidden_state[-1]，AVG                     | 16     | 0.576      |                                      |
| 1e-3 | 16   | hidden_state[-1]，AVG                     | 16     | 0.583      |                                      |
| 2e-3 | 16   | hidden_state[-1]，AVG                     | 16     | 0.595      |                                      |
| 2e-3 | 32   | hidden_state[-1]，AVG                     | 16     | 0.587      |                                      |
| 4e-3 | 16   | hidden_state[-1]，AVG                     | 16     | 0.580      |                                      |
| 3e-3 | 16   | hidden_state[-1]，AVG                     | 16     | 0.581      |                                      |
|      |      |                                           |        |            |                                      |
| 2e-3 | 16   | hidden_state[-1]，AVG + MAX               | 16     | 0.577      |                                      |
| 2e-3 | 32   | hidden_state[-1]，AVG + MAX               | 16     | 0.590      | AVG+MAX还是适合bs=32                 |
| 2e-3 | 64   | hidden_state[-1]，AVG + MAX               | 16     | 0.576      |                                      |
| 2e-3 | 16   | hidden_state[-2]，AVG                     | 16     | 0.605      | 有提升，为什么有的时候AVG要比MAX要好 |
| 2e-3 | 16   | hidden_state[-2]，MAX                     | 16     | 0.611      |                                      |
| 2e-3 | 8    | hidden_state[-2]，MAX                     | 16     | **0.630**  | hidden_state[-2]表现比AVG更好        |
| 2e-3 | 8    | hidden_state[-2]，MAX+AVG                 | 16     | 0.611      |                                      |
| 2e-3 | 16   | hidden_state[-2]，MAX+AVG                 | 16     | 0.612      |                                      |
| 2e-3 | 8    | hidden_state[-2]，MAX+AVG                 | 16     | 0.608      |                                      |
| 2e-3 | 8    | hidden_state[-2]，MAX+MAX                 | 16     | 0.608      |                                      |
| 2e-3 | 8    | hidden_state[-1]MAX                       | 16     | 0.573      |                                      |
| 2e-3 | 16   | hidden_state[-1]MAX                       | 16     | 没必要测了 |                                      |
| 2e-3 | 8    | hidden_state[-1]AVG+hidden_state[-2]MAX   | 16     | 0.600      |                                      |
| 2e-3 | 16   | hidden_state[-1]AVG+hidden_state[-2]MAX   | 16     | 0.589      |                                      |
| 2e-3 | 8    | hidden_state[-3]MAX                       | 16     | **0.653**  |                                      |
| 2e-3 | 8    | hidden_state[-3]MAX                       | 16     |            | 持续预训练                           |
| 2e-3 | 8    | hidden_state[-3]MAX                       | 16     | 0.613      | 四个"[MASK]"                         |
| 2e-3 | 8    | hidden_state[-3]MAX                       | 16     | 0.636      | 3个"[MASK]"                          |
| 2e-3 | 8    | hidden_state[-3]MAX                       | 16     | 0.593      | 1个"[MASK]"                          |
| 2e-3 | 8    | hidden_state[-3]MAX                       | 16     | 0.642      | no dropout                           |
| 2e-3 | 16   | hidden_state[-3]AVG                       | 16     | 0.605      |                                      |
| 2e-3 | 8    | hidden_state[-4]MAX                       | 16     | 0.615      |                                      |
| 2e-3 | 8    | hidden_state[-5]MAX                       | 16     | 0.561      |                                      |
| 2e-3 | 8    | hidden_state[-6]MAX                       | 16     | 0.582      |                                      |
| 2e-3 | 8    | hidden_state[-7]MAX                       | 16     |            |                                      |
| 2e-3 | 8    | hidden_state[-2]MAX + hidden_state[-3]MAX | 16     | **0.632**  |                                      |
| 2e-3 | 8    | hidden_state[-3]MAX                       | 16     | 0.667      | hfl/chinese-roberta-wwm-ext-large    |
| 2e-3 | 8    | hidden_state[-2]MAX                       | 16     | 0.679      | hfl/chinese-roberta-wwm-ext-large    |
| 2e-3 | 8    | hidden_state[-2]MAX + hidden_state[-3]MAX | 16     | 0.679      | hfl/chinese-roberta-wwm-ext-large    |

综合后四层的信息，然后取MAX或AVG，然后进行分类

| lr   | bs   | method | epochs | ACC    | 备注              |
| ---- | ---- | ------ | ------ | ------ | ----------------- |
| 2e-3 | 8    | 取四层 | 16     | 0.5545 | bert-base-chinese |

效果不行



把模板带入预训练过程当中 再看一看Prompt tuning

| lr   | bs   | method              | epochs | ACC       | 备注                                         |
| ---- | ---- | ------------------- | ------ | --------- | -------------------------------------------- |
| 2e-3 | 8    | hidden_state[-3]MAX | 16     | **0.835** | continues pretrain model "bert-base-chinese" |

