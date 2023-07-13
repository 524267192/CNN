import os
import numpy as np

np.set_printoptions(threshold=np.inf)

"""数据：64*64（*2二元组），标签：64*65*2（*2二元组）
1.任务：将多文件夹下的512*512的矩阵按照8*8网格的压缩，方法是：8*8网格的边框之和求平均值，最后得到64*64的网格
"""
folder_path = "./all_data"  # 文件夹路径
num = 0  # 数据文件U的个数
all_data = []#保存所有的64*64矩阵

import os

folder_path = 'all_data'

# 获取all_data目录下所有子目录。共三级目录
subdirs = [os.path.join(folder_path, name) for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))]

print(subdirs)#['all_data\\kolmogorov_512_105', 'all_data\\kolmogorov_512_245', 'all_data\\kolmogorov_512_50']

# 遍历子目录
for subdir in subdirs:
    # 列出subdir目录下的所有文件和目录
    subdir_contents = os.listdir(subdir)

    # 遍历subdir目录下的所有文件和目录。即0，0.2，0.4文件夹
    for content in subdir_contents:
        try:
            if content=='0':
                continue
            float(content)
        except ValueError:
            continue

        file_path = os.path.join(subdir, content,"U")
        print(file_path)

        if os.path.isfile(file_path):  # 确保文件存在
            num += 1
            with open(file_path, "r") as file:
                data = file.readlines()[23:-29]

                # 将数据列表中的科学计数法表示转换为浮点数
                # ['(0.00421566 1.69946e-08 0)\n', '(0.00421566 4.74559e-08 0)\n']
                # 跳过左括号"("，并使用索引-4跳过" 0)\n"。
                arr = np.array([np.fromstring(t[1:-4], sep=' ') for t in data])

                # 将1维数组转换为512x512的矩阵
                matrix_512 = arr.reshape(512, 512, 2)

                # 按8x8小块进行合并成64x64的三元组矩阵
                matrix_64 = np.zeros((64, 64, 2))
                for i in range(0, 512, 8):
                    for j in range(0, 512, 8):
                        block_8 = matrix_512[i:i + 8, j:j + 8]  # 长度为8的方块
                        # 提取最外层边框的元素
                        border_element = np.concatenate([block_8[0], block_8[-1], block_8[1:-1, 0], block_8[1:-1, -1]],
                                                        axis=0)
                        # print(border_element)
                        a = np.sum(border_element, axis=0)
                        # 计算边框元素的个数
                        border_elements_num = 2 * (block_8.shape[0] + block_8.shape[1]) - 4

                        # 计算最外层边框的值之和并除以边框元素的个数
                        block_sum_8 = np.sum(border_element, axis=0) / border_elements_num

                        matrix_64[i // 8, j // 8] = block_sum_8

                all_data.append(matrix_64)
                print("已处理完第", str(num), "个文件")


print("数据处理完毕，开始写入……")
with open("data/data_64x64x2.txt", "w", encoding="utf-8") as file:
    for line in all_data:
        file.write(str(line) + "\n\n\n")

print("数据写入完成!")
