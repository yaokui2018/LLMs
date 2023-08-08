# -*- coding: utf-8 -*-
# Author: 薄荷你玩
# Date: 2023/05/28

"""
垃圾评论分类
需先执行chatglm_ptuning对模型进行微调，生成checkpoint文件：
>> python chatglm-6b/fine_tune/chatglm_ptuning.py  --do_train --train_file SpamClassify/train.json  --validation_file SpamClassify/dev.json --prompt_column  content --response_column label --overwrite_cache --model_name_or_path ..\\..\\chatglm-6b --output_dir output/spamclassify-chatglm-6b-pt-4-2e-2 --overwrite_output_dir --max_source_length 64 --max_target_length 64 --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --gradient_accumulation_steps 16 --predict_with_generate --max_steps 3000 --logging_steps 10 --save_steps 1000 --learning_rate 2e-2 --pre_seq_len 4 --quantization_bit 8
"""

import os
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig

# config
model_name_or_path = "chatglm-6b"  # 模型存放路径 或 huggingface模型名
# CHECKPOINT_PATH = 'chatglm-6b/fine_tune/output/adgen-chatglm-6b-pt-128-2e-2/checkpoint-3000'  # P-tuning checkout文件路径
CHECKPOINT_PATH = 'chatglm-6b/fine_tune/output/spamclassify-chatglm-6b-pt-4-2e-2/checkpoint-100'  # P-tuning checkout文件路径
PRE_SEQ_LEN = 128  # 这里序列长度需要和微调时候定义的序列长度一致

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True, pre_seq_len=PRE_SEQ_LEN)
model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True, config=config).quantize(8).half().cuda()

prefix_state_dict = torch.load(os.path.join(CHECKPOINT_PATH, "pytorch_model.bin"))
new_prefix_state_dict = {}
for k, v in prefix_state_dict.items():
    if k.startswith("transformer.prefix_encoder."):
        new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)

response, history = model.chat(tokenizer, "你好", history=[])
print(response)

# 最多携带历史会话轮数
max_history = 1

while True:
    pre_response = ''
    for response, history in model.stream_chat(tokenizer, input(">> "), history=[]):
        if response.startswith(pre_response):
            print(response.replace(pre_response, ''), end='')
        else:
            print('\n' + response, end='')
        pre_response = response

        if len(history) > max_history:
            history = history[1:]
    print('\n')
