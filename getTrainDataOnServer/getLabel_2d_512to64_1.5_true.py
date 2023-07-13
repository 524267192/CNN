import os
import numpy as np

np.set_printoptions(threshold=np.inf)

"""
1.任务：
shape为(512, 512, 2)的矩阵matrix_512，求其每个8*8小矩阵的四个面的面心值，并且矩阵边缘为循环条件，意思是矩阵的最左边与矩阵的最右边相连求面心值，大矩阵求面心值是按照小矩阵从左到右从下到上，并保存到64*65*2的矩阵中，共64*65个面心值，面心值是一个长度为2的元组，求的方法是每个8*8矩阵的两个边界的16个元素的平均值


2.代码说明：
两个双层for循环，一个求横着的，一个求竖着的边值。函数返回两个求得的矩阵all_vertical_edge_centers，
一个是64*65的竖着的边值矩阵，一个是64*65的横着的边值矩阵all_horizontal_edge_centers。
两个矩阵都是（64*65）*2大小，合并后变成（64*65*2）*2大小。最后一个*2是每个元素是二元组
"""


def calculate_face_centers(matrix_512):
    all_vertical_edge_centers = np.random.random((64, 65, 2))
    # all_horizontal_edge_centers = np.random.random((64, 65, 2))
    all_horizontal_edge_centers = np.zeros((64, 65, 2))

    # 1.求横着的边，分两种情况，边缘（对称的）和非边缘的边
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

            else:
                matrix_edge = matrix_512[bottom - 1:bottom + 1, left:right, :]  # 中间的边[7:9,0:8,:]
                # print(matrix_edge)
                # print(j)
                center = np.mean(np.concatenate(matrix_edge), axis=0)
                all_horizontal_edge_centers[i, j] = center
                # print(center)

    # print(all_horizontal_edge_centers)
    print(all_horizontal_edge_centers.shape)  # (64, 65, 2)

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
                matrix_edge = matrix_512[bottom:top, right - 1:right + 1, :]  # 中间的边[7:9,0:8,:]
                # print(matrix_edge)
                # print(j)
                center = np.mean(np.concatenate(matrix_edge), axis=0)
                all_vertical_edge_centers[i, j] = center
                # print(center)

    # print(all_vertical_edge_centers)
    # print(all_vertical_edge_centers.shape)#(64, 65, 2)
    return all_vertical_edge_centers, all_horizontal_edge_centers


folder_path = "all_data"  # 文件夹路径
num = 0  # 数据文件U的个数
all_data = []
# 遍历文件夹
for root, dirs, files in os.walk(folder_path):
    for dir_name in dirs:
        file_path = os.path.join(root, dir_name, "U")  # U 文件的路径
        # num<=2.用于测试
        if os.path.isfile(file_path) and file_path != r"cavity_512\0\U" and num <= 2:  # 确保文件存在
            num += 1
            with open(file_path, "r") as file:
                data = file.readlines()[23:-29]

                # 将数据列表中的科学计数法表示转换为浮点数
                # ['(0.00421566 1.69946e-08 0)\n', '(0.00421566 4.74559e-08 0)\n']
                # 过左括号"("，并使用索引-2跳过右括号")"和换行符"\n"。
                arr = np.array([np.fromstring(t[1:-4], sep=' ') for t in data])

                """
                1.reshape:
                可以重新构造一个具有指定形状的新数组，而不改变原始数据的顺序。
                [[ 0  1  2  3  4  5  6  7  8  9 10 11]]
                reshape后重新排列后的数组：
                [[ 0  1  2  3  4  5  6  7  8  9 10 11]]
                
                2.这样原本一维数据512*512，本来放在二维中是从左到右，从上到下
                reshape变成二维后，在openfoam中是从左下角排，如下图所示：
                [[13 14 15 16]
                 [ 8  9 10 11]
                 [ 4  5  6  7]
                 [ 0  1  2  3]]

                """
                # 将1维数组转换为512x512的矩阵
                matrix_512 = arr.reshape(512, 512, 2)

                matrix_512 = np.ones((512, 512, 2))
                for i in range(512):
                    matrix_512[i, :, 0] = i + 1
                    matrix_512[i, :, 1] = i + 1

                # 求解新形成的矩阵，返回一个竖着的，一个横着的矩阵
                all_vertical_edge_centers, all_horizontal_edge_centers = calculate_face_centers(matrix_512)

    # 不进入子文件夹
    break

####这里有小问题：all_vertical_edge_centers, all_horizontal_edge_centers放在for循环里，
# 没有保存所有文件的，其实只保存了最后一个文件的数据，所以下面写入数据也是写入了一个文件的数据，不是所有文件


print("数据处理完毕，开始写入……")
with open("data/label_2d.txt", "w", encoding="utf-8") as file:
    for line in all_vertical_edge_centers:
        file.write(str(line) + "\n\n\n")
    for line in all_horizontal_edge_centers:
        file.write(str(line))
print("数据写入完成!")
