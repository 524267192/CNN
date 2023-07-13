import re
import numpy as np
np.set_printoptions(threshold=np.inf)

"""
将6*2*64*64*2的数据转为12*64*64*2
"""
# 从文件中读取数据并转换为三维列表格式
input_file = 'data/label_a_2x64x64x2.txt'
def get_label():
    all_label = []#存放所有数据
    single_file_label = []#存放一个文件的数据
    i = 0#计数，计每个文件的行个数，每4096行保存一次
    with open(input_file, 'r') as f:
        lines = f.readlines()

        # 匹配浮点数和科学计数法表示的浮点数的正则表达式
        pattern = r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?"

        matches = re.findall(pattern, str(lines))

        result = [float(match) for match in matches]

        for a in range(len(result)):
            if a==0 or a%(4096*2)!=0:
                single_file_label.append(result[a])

            else:
                # print(len(single_file_label))#8192
                single_file_label = np.array(single_file_label).reshape(64,64,2)
                all_label.append(single_file_label)
                single_file_label=[]
                single_file_label.append(result[a])
        single_file_label = np.array(single_file_label).reshape(64, 64, 2)
        all_label.append(single_file_label)
        all_label = np.array(all_label)
        # print(all_label.shape)

        all_label = all_label.reshape(-1,64, 64, 4)#reshape有问题，是前四个合并成一个，实际上不是这样
        # print(all_label.shape)
        return all_label
all_label = get_label()
    # import torch
    # tensor = torch.tensor(all_label)
    # print(tensor)

with open("./data/formatted_label.txt", "w", encoding="utf-8") as f:
    f.write(str(all_label))
    f.close()


