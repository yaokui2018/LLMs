# -*- coding: utf-8 -*-
# Author: 薄荷你玩
# Date: 2023/05/09

from transformers import AutoTokenizer, AutoModel

# 模型名：chatglm-6b / chatglm2-6b
MODEL = "D:\code\chatglm2-6b-32k"

tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModel.from_pretrained(MODEL, trust_remote_code=True).quantize(8).half().cuda()

response, history = model.chat(tokenizer, "你好", history=[])
print(response)

# 最多携带历史会话轮数
max_history = 5

while True:
    pre_response = ''
    for response, history in model.stream_chat(tokenizer, input(">> "), history=history):
        if response.startswith(pre_response):
            print(response.replace(pre_response, ''), end='')
        else:
            print('\n' + response, end='')
        pre_response = response

        if len(history) > max_history:
            history = history[1:]
    print('\n')
