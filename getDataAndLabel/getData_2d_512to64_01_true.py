import os
import numpy as np

np.set_printoptions(threshold=np.inf)

"""数据：64*64（*2二元组），标签：64*65*2（*2二元组）
1.任务：将多文件夹下的512*512的矩阵按照8*8网格的压缩，方法是：8*8网格的边框之和求平均值，最后得到64*64的网格
"""
folder_path = "cavity_512"  # 文件夹路径
num = 0  # 数据文件U的个数
all_data = []
# 遍历文件夹
for root, dirs, files in os.walk(folder_path):
    for dir_name in dirs:
        file_path = os.path.join(root, dir_name, "U")  # U 文件的路径
        if os.path.isfile(file_path) and file_path != r"cavity_512\0\U" and num <= 0:  # 确保文件存在
            num += 1
            with open(file_path, "r") as file:
                data = file.readlines()[23:-29]

                # 将数据列表中的科学计数法表示转换为浮点数
                # ['(0.00421566 1.69946e-08 0)\n', '(0.00421566 4.74559e-08 0)\n']
                # 跳过左括号"("，并使用索引-4跳过" 0)\n"。
                arr = np.array([np.fromstring(t[1:-4], sep=' ') for t in data])

                # 将1维数组转换为512x512的矩阵
                matrix_512 = arr.reshape(512, 512, 2)

                # matrix_512 = np.ones((512, 512, 2))
                # matrix_512 = np.zeros((512, 512, 2))
                # for i in range(512):
                #     matrix_512[i, :, 0] = i + 1
                #     matrix_512[i, :, 1] = i + 1

                # 按8x8小块进行合并成64x64的三元组矩阵
                matrix_64 = np.zeros((64, 64, 2))
                for i in range(0, 512, 8):
                    for j in range(0, 512, 8):
                        block_8 = matrix_512[i:i + 8, j:j + 8]  # 长度为8的方块
                        # 提取最外层边框的元素
                        border_element = np.concatenate([block_8[0], block_8[-1], block_8[1:-1, 0], block_8[1:-1, -1]],
                                                        axis=0)
                        print(border_element)
                        a = np.sum(border_element, axis=0)
                        # 计算边框元素的个数
                        border_elements_num = 2 * (block_8.shape[0] + block_8.shape[1]) - 4

                        # 计算最外层边框的值之和并除以边框元素的个数
                        block_sum_8 = np.sum(border_element, axis=0) / border_elements_num

                        matrix_64[i // 8, j // 8] = block_sum_8

                all_data.append(matrix_64)
                print("已处理完第", str(num), "个文件")

    # 不进入子文件夹
    break

print("数据处理完毕，开始写入……")
with open("data/data_2d.txt", "w", encoding="utf-8") as file:
    for line in all_data:
        file.write(str(line) + "\n\n\n")

print("数据写入完成!")
