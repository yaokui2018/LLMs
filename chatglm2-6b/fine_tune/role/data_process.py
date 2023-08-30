import json

from sklearn.model_selection import train_test_split
import re

from tqdm import tqdm


def remove_non_ascii(text):
    return re.sub(r'[^\u4e00-\u9fff\u3040-\u30ff\u1100-\u11ff\u3130-\u318f\uac00-\ud7af\w]+', '', text)

def save_file(filename, titles, contents):
    with open(filename, 'a+', encoding='utf8') as f:
        for X, label in tqdm(zip(titles, contents)):
            data = {
                "question": X,
                "answer": label
            }
            line = json.JSONEncoder(ensure_ascii=False).encode(data)
            f.write(line + '\n')

# 划分数据集
# 参数->  ratio:测试集比例
def process(ratio=0.1):
    questions = []
    answers = []
    max_content_len = 0
    with open('御姐对话数据.txt', encoding='utf8') as f:
        for index, line in enumerate(tqdm(f.readlines())):
            line = line.strip()
            data = line.split("  >>  ")
            assert len(data) == 2
            questions.append(data[0].strip())
            answers.append(data[1].strip())
            if len(data[1]) > max_content_len:
                max_content_len = len(data[1])
                print("max_content_len", max_content_len)
    print(len(questions), len(answers))

    x_train, x_test, y_train, y_test = train_test_split(questions, answers, test_size=ratio, shuffle=True)

    save_file('train.json', x_train, y_train)
    save_file('dev.json', x_test, y_test)

    return x_train, x_test, y_train, y_test


process()
