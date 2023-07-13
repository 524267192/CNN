import os
import numpy as np

np.set_printoptions(threshold=np.inf)

"""
1.任务：
  1.由512求64矩阵作为数据
  2.由64矩阵和求得的64*65*2面心值矩阵，求64*65的a矩阵

边心值求解方法：shape为(512, 512, 2)的矩阵matrix_512，求其每个8*8小矩阵的四个面的面心值，并且矩阵边缘为循环条件，
意思是矩阵的最左边与矩阵的最右边相连求面心值，大矩阵求面心值是按照小矩阵从左到右从下到上，
并保存到64*65*2的矩阵中，共64*65个面心值，面心值是一个长度为2的元组，
求的方法是每个8*8矩阵的两个边界的16个元素的平均值


2.代码说明：
两个双层for循环，一个求横着的，一个求竖着的边值。函数返回两个求得的矩阵all_vertical_edge_centers，
一个是64*65的竖着的边值矩阵，一个是64*65的横着的边值矩阵all_horizontal_edge_centers。
两个矩阵都是（64*65）*2大小，合并后变成（64*65*2）*2大小。最后一个*2是每个元素是二元组


3.label是两个64*65矩阵，且横着的那个矩阵与matrix_64矩阵行列相反。data是64*65矩阵。
"""


# 将单个512*512矩阵按照8x8小块合并成一个小块进行合并，最终变成64x64的二元组矩阵
def calculate_a(matrix_512):
    matrix_64 = np.random.random((64, 64, 2))  # 用随机数，表明matrix_64得到0不是因为初始化造成的
    for i in range(0, 512, 8):
        for j in range(0, 512, 8):
            block_8 = matrix_512[i:i + 8, j:j + 8]  # 长度为8的方块

            # 提取最外层边框的元素
            border_element = np.concatenate([block_8[0], block_8[-1], block_8[1:-1, 0], block_8[1:-1, -1]], axis=0)

            # 计算边框元素的个数
            border_elements_num = 2 * (block_8.shape[0] + block_8.shape[1]) - 4

            # 计算最外层边框的值之和并除以边框元素的个数
            # block_sum_8 = np.sum(border_element) / (border_elements_num,border_elements_num)
            sum_axis0 = np.sum(border_element, axis=0)
            block_sum_8 = sum_axis0 / border_elements_num
            # print(block_sum_8)

            matrix_64[i // 8, j // 8] = block_sum_8

    return matrix_64


""":cvar
一、计算边心值
二、根据边心值和matrix_64矩阵计算a
"""


def calculate_face_centers(matrix_512):
    # <一、下面开始求边心值>    #####################################
    all_vertical_edge_centers = np.random.random((64, 65, 2))
    all_horizontal_edge_centers = np.random.random((64, 65, 2))

    #   1.求横着的边，分两种情况，边缘（对称的）和非边缘的边
    for i in range(64):
        for j in range(0, 65):
            # 计算边界索引,下面这里i,j互换与计算竖着的边相比的话
            left = i * 8
            right = (left + 8)
            bottom = j * 8  # 0-7,8-15
            top = (bottom + 8)

            # 两者的面心值是对称的，值一样
            if j == 0 or j == 64:
                matrix_edge = matrix_512[(0, 511), left:right, :]  # 最上边和最、下边的边
                center = np.mean(np.concatenate(matrix_edge), axis=0)
                # print(center)
                all_horizontal_edge_centers[i, j] = center

            else:  # 3错误：[top-1：top+1改成[bottom - 1:bottom + 1,
                matrix_edge = matrix_512[bottom - 1:bottom + 1, left:right, :]  # 中间的边[7:9,0:8,:]
                # print(matrix_edge)
                # print(j)
                center = np.mean(np.concatenate(matrix_edge), axis=0)
                all_horizontal_edge_centers[i, j] = center
                # print(center)

    # print(all_horizontal_edge_centers)
    # print(all_horizontal_edge_centers.shape)#(64, 65, 2)

    # 2.求竖着的边，分两种情况，边缘（对称的）和非边缘的边
    for i in range(64):
        for j in range(65):

            # 计算边界索引
            left = j * 8  # 小网格左边的边
            right = (left + 8)  # 小网格右边的边
            bottom = i * 8  # 小网格下边的边   0-7,8-15
            top = (bottom + 8)  # 小网格上边的边

            # 两者的面心值是对称的，值一样
            if j == 0 or j == 64:  # 1.bottom:top,(0, 511)两个弄反了
                matrix_edge = matrix_512[bottom:top, (0, 511), :]  # 最左边和最右边的边
                center = np.mean(np.concatenate(matrix_edge), axis=0)
                all_vertical_edge_centers[i, j] = center
                # print(center)
            else:
                matrix_edge = matrix_512[bottom:top, left - 1:left + 1, :]  # 中间的边[7:9,0:8,:]
                # matrix_edge = matrix_512[bottom:top, right - 1:right + 1, :]  #4.之前错误：right改为left
                # print(matrix_edge)
                # print(j)
                center = np.mean(np.concatenate(matrix_edge), axis=0)
                all_vertical_edge_centers[i, j] = center
                # print(center)

    # print(all_vertical_edge_centers)
    # print(all_vertical_edge_centers.shape)#(64, 65, 2)

    """易错点：
    1.为了保证横着和竖着时候的两个面心值矩阵都是64*65的格式，而不是一个64*65，另一个65*64格式，所以i和j在遍历的时候是相反的。（为了保证输出标签64*65*2的格式）

    2.下面开始求a,这里用到了materi_64,但是all_horizontal_edge_centers与all_vertical_edge_centers的i与j是相反的
    由上分析知，matrix_64行列的排列与a_vertical（竖着的时候），与all_horizontal_edge_centers行列的排列相反（横着的时候）
    故需要将下面21中的代码matrix行列互换
    """

    # <二、下面开始求a>    #####################################
    a_vertical = np.zeros((64, 65, 2))
    # a_vertical = np.random((64, 65, 2))
    a_horizontal = np.zeros((64, 65, 2))
    # a_horizontal = np.random.random((64, 65, 2))

    # 由512*512得到64*64矩阵
    matrix_64 = calculate_a(matrix_512)

    # 21. 求a:   横着的边，分两种情况，边缘（对称的）和非边缘的边.
    # 因为涉及到matrix_64和all_horizontal_edge_centers，两者存储行列相反。
    # 故这里matrix_64要行列互换，因为横着时面心值是一列一列求得，竖着时是一行一行求的。
    for i in range(64):
        for j in range(65):
            if j == 0 or j == 64:
                diff = matrix_64[63, i] - matrix_64[0, i]
                beichushu = all_horizontal_edge_centers[i, 0] - matrix_64[0, i]
                if np.any((0 <= diff) & (diff < 0.00001)):
                    # 找到符合条件的元素索引
                    indices = np.where((0 <= diff) & (diff < 0.00001))[0]
                    # 将符合条件的元素除以
                    for index in indices:
                        beichushu[index] /= 0.00001
                    a_horizontal[i, j] = beichushu
                elif np.any((-0.00001 < diff) & (diff < 0)):
                    indices = np.where((-0.00001 < diff) & (diff < 0))[0]
                    # 将符合条件的元素除以
                    for index in indices:
                        beichushu[index] /= (-0.00001)
                    a_horizontal[i, j] = beichushu

                # a_horizontal[i, j] = (all_horizontal_edge_centers[i, 0] - matrix_64[i, 0]) / (matrix_64[i, 63] - matrix_64[i, 0])#错误的：换之前
                else:
                    a_horizontal[i, j] = (all_horizontal_edge_centers[i, 0] - matrix_64[0, i]) / (
                        matrix_64[63, i] - matrix_64[0, i])
            else:
                diff = matrix_64[j - 1, i] - matrix_64[j, i]
                beichushu = all_horizontal_edge_centers[i, j] - matrix_64[j, i]
                if np.any((0 <= diff) & (diff < 0.00001)):
                    # 找到符合条件的元素索引
                    indices = np.where((0 <= diff) & (diff < 0.00001))[0]
                    # 将符合条件的元素除以
                    for index in indices:
                        beichushu[index] /= 0.00001
                    a_horizontal[i, j] = beichushu
                elif np.any((-0.00001 < diff) & (diff < 0)):
                    indices = np.where((-0.00001 < diff) & (diff < 0))[0]
                    # 将符合条件的元素除以
                    for index in indices:
                        beichushu[index] /= (-0.00001)
                    a_horizontal[i, j] = beichushu
                else:
                    # aA=(1-a)B=E   a = (E-B)/(A-B) 其中：A为matrix_64[i,j]，B为matrix_64[i,j+1]
                    # a_horizontal[i, j] = (all_horizontal_edge_centers[i, j] - matrix_64[i, j]) / (matrix_64[i, j-1] - matrix_64[i, j])
                    a_horizontal[i, j] = (all_horizontal_edge_centers[i, j] - matrix_64[j, i]) / (
                        matrix_64[j - 1, i] - matrix_64[j, i])

    # 22. 求a:   竖着的边，分两种情况，边缘（对称的）和非边缘的边
    for i in range(64):
        for j in range(65):
            if j == 0 or j == 64:
                diff = matrix_64[i, 63] - matrix_64[i, 0]
                beichushu = all_vertical_edge_centers[i, 0] - matrix_64[i, 0]
                if np.any((0 <= diff) & (diff < 0.00001)):
                    # 找到符合条件的元素索引
                    indices = np.where((0 <= diff) & (diff < 0.00001))[0]
                    # 将符合条件的元素除以
                    for index in indices:
                        beichushu[index] /= 0.00001
                    a_vertical[i, j] = beichushu
                elif np.any((-0.00001 < diff) & (diff < 0)):
                    indices = np.where((-0.00001 < diff) & (diff < 0))[0]
                    # 将符合条件的元素除以
                    for index in indices:
                        beichushu[index] /= (-0.00001)
                    a_vertical[i, j] = beichushu
                else:
                    a_vertical[i, j] = (all_vertical_edge_centers[i, 0] - matrix_64[i, 0]) / (
                        matrix_64[i, 63] - matrix_64[i, 0])



            else:
                diff = matrix_64[i, j - 1] - matrix_64[i, j]
                beichushu = all_vertical_edge_centers[i, j] - matrix_64[i, j]
                if np.any((0 <= diff) & (diff < 0.00001)):
                    # 找到符合条件的元素索引
                    indices = np.where((0 <= diff) & (diff < 0.00001))[0]
                    # 将符合条件的元素除以
                    for index in indices:
                        beichushu[index] /= 0.00001
                    a_vertical[i, j] = beichushu
                elif np.any((-0.00001 < diff) & (diff < 0)):
                    indices = np.where((-0.00001 < diff) & (diff < 0))[0]
                    # 将符合条件的元素除以
                    for index in indices:
                        beichushu[index] /= (-0.00001)
                    a_vertical[i, j] = beichushu
                else:

                    # aA=(1-a)B=E   a = (E-B)/(A-B) 其中：A为matrix_64[i,j]，B为matrix_64[i,j+1]
                    a_vertical[i, j] = (all_vertical_edge_centers[i, j] - matrix_64[i, j]) / (
                        matrix_64[i, j - 1] - matrix_64[i, j])
    return matrix_64, a_vertical, a_horizontal


folder_path = "all_data"  # 文件夹路径
num = 0  # 数据文件U的个数
all_data = []

import time
start_time = time.time()

# 写入标签
with open("./data/data_64x64x2.txt", "w", encoding="utf-8") as f_write_data:
    with open("./data/label_a_2x64x65x2.txt", "w", encoding="utf-8") as f_write_label:

        # 获取all_data目录下所有子目录。共三级目录
        subdirs = [os.path.join(folder_path, name) for name in os.listdir(folder_path) if
                   os.path.isdir(os.path.join(folder_path, name))]
        # 遍历子目录
        for subdir in subdirs:
            # 列出subdir目录下的所有文件和目录
            subdir_contents = os.listdir(subdir)

            # 遍历subdir目录下的所有文件和目录。即0，0.2，0.4文件夹
            #只处理数字文件夹且文件夹0<x<=7的不处理。处理7.01-55
            num += 1#文件夹的个数，即k_50,k_60这样的个数
            for content in subdir_contents:
                try:
                    float(content)
                    if float(content) >= 0 and float(content) <= 7:
                        continue
                except ValueError:
                    continue

                file_path = os.path.join(subdir, content, "U")
                # print(file_path)
                with open(file_path, "r") as file:
                    # 因为两个64*65的矩阵，故写8000多行
                    data = file.readlines()[23:-29]

                    # 将数据列表中的科学计数法表示转换为浮点数
                    # ['(0.00421566 1.69946e-08 0)\n', '(0.00421566 4.74559e-08 0)\n']
                    # 过左括号"("，并使用索引-2跳过右括号")"和换行符"\n"。
                    arr = np.array([np.fromstring(t[1:-4], sep=' ') for t in data])

                    # 将1维数组转换为512x512的矩阵
                    matrix_512 = arr.reshape(512, 512, 2)

                    # 求解新形成的矩阵，返回一个竖着的，一个横着的矩阵
                    matrix_64, a_vertical, a_horizontal = calculate_face_centers(matrix_512)
                    f_write_data.write(str(matrix_64)+"\n\n\n")
                    f_write_label.write(str(a_vertical)+"\n\n\n")
                    f_write_label.write(str(a_horizontal)+"\n\n\n")

            print("第", str(num), "个文件夹内所有文件处理和写入完毕……")

            end_time = time.time()
            execution_time = end_time - start_time
            print("程序一共执行的时间为: {:.2f} 秒".format(execution_time))


f_write_data.close()
f_write_label.close()

print("写入数据和标签完成，程序结束!")
