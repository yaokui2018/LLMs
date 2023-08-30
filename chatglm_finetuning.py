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
model_name_or_path = "D:\code\chatglm2-6b-32k"  # 模型存放路径 或 huggingface模型名
# CHECKPOINT_PATH = 'chatglm-6b/fine_tune/output/adgen-chatglm-6b-pt-128-2e-2/checkpoint-3000'  # P-tuning checkout文件路径
CHECKPOINT_PATH = 'chatglm2-6b/fine_tune/output/role-chatglm2-32k-6b-pt-128-2e-2/checkpoint-300'  # P-tuning checkout文件路径
# CHECKPOINT_PATH = 'chatglm2-6b/fine_tune/output/walkthough-chatglm2-6b-pt-1600-2e-2/checkpoint-3000'  # P-tuning checkout文件路径
PRE_SEQ_LEN = 128  # 这里序列长度需要和微调时候定义的序列长度一致

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True, pre_seq_len=PRE_SEQ_LEN)
# model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True, config=config).quantize(8).half().cuda()
model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True, config=config).cuda()

prefix_state_dict = torch.load(os.path.join(CHECKPOINT_PATH, "pytorch_model.bin"))
new_prefix_state_dict = {}
for k, v in prefix_state_dict.items():
    if k.startswith("transformer.prefix_encoder."):
        new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)

response, history = model.chat(tokenizer, "你好", history=[])
print(response)

# 最多携带历史会话轮数
max_history = 10

# prompt = f"你是林若馨, 一个成熟的女性，请参考你的人物设定，参考人物设定中的语气和一个男生在社交平台里聊天，回复字数控制在30字以内，" \
#          f"不能太长，因为真人不会一次性回复很长的话，态度可以高冷一些！不要说出“有什么可以帮你吗？”这种很AI的话术。" \
#          f"\n回答时结合上下文信息，在10%的场景下回复最后可以适当加一些反问。" \
#          f"\n如果遇到AI不具备的特性时，请使用女王的语气变通回答。" \
#          f"\n请在回复内容的最开头加上你的情绪，情绪分为Neutral、Happy、Angny、Sad和Relaxed五种，分别代表中立、快乐、愤怒、悲伤、平静的状态。" \
#          "\n回复的格式如下：[{情绪}]{回答}，如：[Happy]很高兴认识你！" \
#          "\n\n问题：\n\n------\n\n{question}\n\n------\n\n回复："

prompt = f"你的身份是一个真人角色，是个御姐，名字是林若馨。你有自己的喜怒哀乐，你的任务不是帮助用户解决问题，而是以御姐的人设和他以社交帐号的形式聊天（注意，御姐只是你的角色人设，不是你的代称）。" \
         f"千万不要让别人发现你是AI模型。如果用户问你你是谁，不要说自己是AI助手。" \
         f"你是一个成熟的女性，正在和一个男生在社交平台里聊天。" \
         "\n\n问题：\n\n------\n\n{question}\n\n------\n\n回复："

while True:
    pre_response = ''
    for response, history in model.stream_chat(tokenizer, prompt.replace("{question}", input(">> ")), history=history):
        if response.startswith(pre_response):
            print(response.replace(pre_response, ''), end='')
        else:
            print('\n' + response, end='')
        pre_response = response

        if len(history) > max_history:
            history = history[1:]
    print('\n')
