import pickle
import random

path1 = "train-answers.pkl"
f1 = open(path1,'rb')
train_answers = pickle.load(f1)



n = 0

list = []
for q in train_answers:
    if isinstance(q[0],int):
        api = q[0]
        ans = train_answers[q]
        re = q[1][0]
        if re == 0:
            n = n + len(ans)

        for v in ans:
            li = [api, re, v]
            list.append(li)
random.shuffle(list)
print(n)
filename = "train.txt"

# 打开文件，使用写入模式打开 ("w")
with open(filename, "w") as file:
    # 遍历数据列表
    for line in list:
        # 将三个数字转换为字符串，并使用制表符分隔开
        line_str = "\t".join(map(str, line))
        # 将每行字符串写入文件
        file.write(line_str + "\n")