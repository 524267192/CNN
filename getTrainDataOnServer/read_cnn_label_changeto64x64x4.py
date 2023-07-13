import re
import numpy as np
import torch

np.set_printoptions(threshold=np.inf)

"""
将6*2*64*64*2的数据转为12*64*64*2,再将其转为6*64*64*4
其中4是前面两个是竖着的a值，后面两个是横着的a值，用于作为标签
"""
# 从文件中读取数据并转换为三维列表格式
def get_label():
    input_file = 'data/label_a_2x64x64x2.txt'
    all_label = []  # 存放所有数据
    single_file_label = []  # 存放一个文件的数据
    i = 0  # 计数，计每个文件的行个数，每4096行保存一次
    with open(input_file, 'r') as f:
        lines = f.readlines()

        # 匹配浮点数和科学计数法表示的浮点数的正则表达式
        pattern = r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?"

        matches = re.findall(pattern, str(lines))

        result = [float(match) for match in matches]

        for a in range(len(result)):
            if a == 0 or a % (4096 * 2) != 0:
                single_file_label.append(result[a])

            else:
                # print(len(single_file_label))#8192
                single_file_label = np.array(single_file_label).reshape(64, 64, 2)
                all_label.append(single_file_label)
                single_file_label = []
                single_file_label.append(result[a])
        single_file_label = np.array(single_file_label).reshape(64, 64, 2)
        all_label.append(single_file_label)
        all_label = np.array(all_label)

        two_label = []
        all_4d_label = []
        num = 0
        for i in all_label:
            num += 1
            two_label.append(i)
            if num % 2 == 0:
                two_label = np.array(two_label)
                two_label = torch.tensor(two_label)
                two_label = two_label.permute(1, 2, 3, 0)
                two_label = torch.reshape(two_label, (two_label.size(0), two_label.size(1), -1))

                # 调整第三维元素的列的顺序
                column_indices = [0, 2, 1, 3]
                two_label = two_label[:, :, column_indices]

                all_4d_label.append(two_label)
                two_label = []
        all_4d_label = np.array([i.numpy() for i in all_4d_label])
        # print(all_4d_label.shape)

        return all_4d_label


all_4d_label = get_label()

with open("./data/formatted_label.txt", "w", encoding="utf-8") as f:
    f.write(str(all_4d_label))
    f.close()
