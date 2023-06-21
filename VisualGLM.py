# -*- coding: utf-8 -*-
# Author: 薄荷你玩
# Date: 2023/06/15

from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("visualglm-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("visualglm-6b", trust_remote_code=True).half().cuda()
image_path = "visualglm-6b/test.jpeg"
response, history = model.chat(tokenizer, image_path, "描述这张图片。", history=[])
print(response)
response, history = model.chat(tokenizer, image_path, "这张图片可能是在什么场所拍摄的？", history=history)
print(response)


# 最多携带历史会话轮数
max_history = 1

while True:
    pre_response = ''
    for response, history in model.stream_chat(tokenizer, input(">> 输入图片路径："),  input(">> 输入指令（默认：描述这张图片）：") or "描述这张图片。", history=[]):
        if response.startswith(pre_response):
            print(response.replace(pre_response, ''), end='')
        else:
            print('\n' + response, end='')
        pre_response = response

        if len(history) > max_history:
            history = history[1:]
    print('\n')
