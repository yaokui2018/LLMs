from sklearn.model_selection import train_test_split
import re

def remove_non_ascii(text):
    return re.sub(r'[^\u4e00-\u9fff\u3040-\u30ff\u1100-\u11ff\u3130-\u318f\uac00-\ud7af\w]+', '', text)

# 划分数据集，返回->  x_train  x_test  y_train  y_test  normal评论数  spam评论数
# 参数->  normaldoc：读取的normal文件数  spamdoc：读取的spam文件数  ratio:测试集比例
def process(ratio=0.1):
    with open('normal.txt', encoding='utf8') as f:
        data = f.readlines()
        normal_size = len(data)
    with open('spam.txt', encoding='utf8') as f:
        data += f.readlines()
        spam_size = len(data) - normal_size

    y = []

    for i in range(normal_size):
        y.append(0)

    for i in range(spam_size):
        y.append(1)
    x_train, x_test, y_train, y_test = train_test_split(data, y, test_size=ratio, shuffle=True)
    with open('train.json', 'a+', encoding='utf8') as f:
        for X, label in zip(x_train, y_train):
            X = remove_non_ascii(X.replace('"', '\\"'))
            print(X.strip(), label)
            f.write('{"content": "' + X.strip() + '", "label": "' + str(label) + '"}\n')
    with open('dev.json', 'a+', encoding='utf8') as f:
        for X, label in zip(x_test, y_test):
            X = remove_non_ascii(X.replace('"', '\\"'))
            print(X.strip(), label)
            f.write('{"content": "' + X.strip() + '", "label": "' + str(label) + '"}\n')

    return x_train, x_test, y_train, y_test


process()
