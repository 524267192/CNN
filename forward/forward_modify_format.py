#!/usr/bin/env python
# coding: utf-8

"""唯一输入的数据internal_data格式：internal_data.shape: torch.Size([2, 64, 64])
tensor([[[-3.4494, -3.4069, -3.4643,  ..., -3.4919, -3.5027, -3.4902],
         [-3.2589, -3.3068, -3.4836,  ..., -3.3944, -3.3611, -3.3109],
         [-3.0727, -3.2115, -3.4910,  ..., -3.2027, -3.1334, -3.0582],
         ...,
         [-3.5453, -3.5756, -3.5814,  ..., -3.0769, -3.2865, -3.4580],
         [-3.5999, -3.6052, -3.5593,  ..., -3.3164, -3.4856, -3.5721],
         [-3.5747, -3.5341, -3.4848,  ..., -3.4699, -3.5530, -3.5744]],

        [[ 2.9233,  2.5661,  2.2575,  ...,  3.5713,  3.4351,  3.2216],
         [ 2.8771,  2.6213,  2.4405,  ...,  3.5747,  3.4131,  3.1778],
         [ 2.9102,  2.7809,  2.7337,  ...,  3.5280,  3.3551,  3.1139],
         ...,
         [ 2.9292,  2.6461,  2.3099,  ...,  3.1242,  3.1627,  3.1149],
         [ 2.9585,  2.6522,  2.2620,  ...,  3.3254,  3.3342,  3.2017],
         [ 2.9625,  2.6100,  2.2051,  ...,  3.4915,  3.4193,  3.2263]]])


输出的数据result格式：shape:[length=8064,length=256]
[[1,1,1,1,1,1,1,1, ... , 1][1,1,1,1,1, ..., 1]]

"""

#一、导入包
import numpy as np
import torch

np.set_printoptions(threshold=np.inf)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
修改：2023-16：40
1.添加函数的注释
2.求a值时若分母相同，则分母为0.00001
"""


#输入一个2x64x64的tensor格式数据，输出指定openfoam里面要求的格式数据result,按照先排列内部，再排列外部weights的两个列表
def printStr(internal_data):

    # 二、加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load('model_origin_100.pth', map_location=device)  # from torchsummary import summary
    #将模型移动到适当的设备
    model = model.to(device)

    #三、预测

    # 设置模型为评估模式
    model.eval()

    # 输入数据进行预测
    input_data = internal_data  # 你的输入数据

    # 调整输入input的维度顺序,作为E，用于下面change_Label_to_a中(E-A)/(B-A)得到a值
    matrix_64 = internal_data.cpu()
    matrix_64 = matrix_64.permute(1, 2, 0)
    # print(matrix_64.shape)

    # 转换为四维
    input_data = input_data.unsqueeze(0)  # 用实际数据，cnn要求输入数据格式为(1,2, 64, 64)，不能为2x64x64
    # input_data = torch.randn(1,2, 64, 64)
    input_tensor = input_data.to(device)

    #将前向传播代码放在torch.no_grad()上下文管理器内部，可以确保在该范围内不会计算梯度，目的是提高代码的运行效率。
    with torch.no_grad():
        output = model(input_tensor)  ##如果报错的话需要把网络的设计加上，里面涉及model(input)

    # 打印预测结果,其中4是前面两个是竖着的面心值，后面两个是横着的面心值，用于作为标签
    # print(output)

    # 四、转换格式(label转为最终的weights)

    #根据面心值得到对应的a值
    def change_Label_to_a(all_vertical_edge_centers, all_horizontal_edge_centers):

        # 下面开始求a
        a_vertical = np.zeros((64, 65, 2))
        # a_vertical = np.random((64, 65, 2))
        a_horizontal = np.zeros((64, 65, 2))
        # a_horizontal = np.random.random((64, 65, 2))

        # 21. 求a:   横着的边，分两种情况，边缘（对称的）和非边缘的边.
        # 因为涉及到matrix_64和all_horizontal_edge_centers，两者存储行列相反。
        # 故这里matrix_64要行列互换，因为横着时面心值是一列一列求得，竖着时是一行一行求的。
        for i in range(64):
            for j in range(65):
                if j == 0 or j == 64:
                    diff = matrix_64[63, i] - matrix_64[0, i]
                    # print(diff)
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
                        # aA+(1-a)B=E   a = (E-B)/(A-B) 其中：A为matrix_64[i,j]，B为matrix_64[i,j+1]
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

        a_vertical = torch.tensor(a_vertical[:, :64, :])
        a_horizontal = torch.tensor(a_horizontal[:, :64, :])

        # print(a_vertical.shape)
        # print(a_horizontal.shape)
        return a_vertical, a_horizontal


    # 返回一个列表，里面嵌套两个子列表，第一个子列表存放的是内部的a值，第二个子列表存放的是边框的a值，
    # 且顺序为上（左到右），下（左到右），左（下到上），右（下到上）
    # 调整output的维度顺序
    output = output.permute(0, 2, 3, 1)
    # print(output.shape)
    output = output[0]
    # print(output.shape)
    # print(output)

    # 将输出output拆成两个面心值
    # 其中4是前面两个是竖着的面心值，后面两个是横着的面心值，用于作为标签
    all_vertical_edge_centers = output[:, :, 0:2].cpu()
    all_horizontal_edge_centers = output[:, :, 2:4].cpu()
    # print(all_vertical_edge_centers.shape)
    # print(all_horizontal_edge_centers.shape)

    #防止改为求a时除以0.00001报错
    matrix_64 = matrix_64.numpy()

    #根据面心值得到对应的a值
    a_vertical, a_horizontal = change_Label_to_a(all_vertical_edge_centers, all_horizontal_edge_centers)

    #根据a值，返回最终的result，包含两个子列表，第一个列表是内部的weights,第二个列表是四个边框的weights
    def convert_a_to_weights(a_vertical, a_horizontal):
        # 将竖着的矩阵的a值的两个元素相加除以二得到一个元素(x+y/)2
        avg_vertical = (a_vertical[:, :, 0] + a_vertical[:, :, 1]) / 2
        # 将横着的矩阵的a值的两个元素相加除以二得到另一个元素
        avg_horizontal = (a_horizontal[:, :, 0] + a_horizontal[:, :, 1]) / 2

        # 重新组合成新的形状为(64, 64, 2)的张量
        new_avg_a_output = torch.stack([avg_vertical, avg_horizontal], dim=2)

        # 打印转换后的数据形状
        # print(new_avg_a_output.shape)

        # 返回两个求完平均的面心值,包括两个64*64矩阵，矩阵是求完平均后的a值，一个竖着的，一个横着的
        vertical_1d = new_avg_a_output[:, :, 0]
        horizontal_1d = new_avg_a_output[:, :, 1]
        # print(vertical_1d.shape)
        # print(horizontal_1d.shape)

        left_border = []  # 存左边框
        bottom_border = []  # 存下边框
        inner = []  # 存内部的面心值

        # 下面将2个64x64面心值变换格式，返回指定的格式result,result包括两个列表，第一个列表存的是内部weights,第二个存的是边框weights
        for i in range(len(vertical_1d)):  # 两个for循环等价于for i in range(64):
            for j in range(len(vertical_1d[i])):
                if j == 0:  # j=0添加边框
                    # 添加左边框
                    left_border.append(vertical_1d[i][0])
                    # 添加下边框
                    bottom_border.append(horizontal_1d[i][0])
                else:
                    if i != 63:  # 当竖着的最后一行时，上面没有对应的横着的
                        inner.append(vertical_1d[i][j])  # 竖着的
                        #                 print("{j-1},{i+1}",j-1,i+1)
                        inner.append(horizontal_1d[j - 1][i + 1])  # 再横着的
                        if j == 63:  # 如果j=63的话，还需要再加入最后一列的横着的边
                            inner.append(horizontal_1d[63][i + 1])  # 当i=63,横着的加最后一列
                    else:  # if i ==63 :
                        inner.append(vertical_1d[63][j])  # 当i=63,inner最后添加竖着的一行竖线

        inner = [tensor.cpu().numpy().tolist() for tensor in inner]  # 将一维列表里面的tensor元素转为numpy格式，并返回cpu版本


        four_border = [bottom_border, bottom_border, left_border, left_border]  # 由于边框值上与下相同，左与右相同。顺序是上（左到右），下（左到右），左（下到上），右（下到上）
        # four_border = [sublist.numpy().tolist() for sublist in four_border]  # 存所有边框，四个边框

        #新修改后的格式
        list_four_border = []
        for sublist in four_border:
            tmplist = []
            for item in sublist:
                tmplist.append(item.numpy().tolist())
            list_four_border.append(tmplist)
        list_four_border.append([])
        all_weight = [inner,list_four_border]

        # print("最终处理后的数据：\n",result)
        return all_weight

    result = convert_a_to_weights(a_vertical, a_horizontal)
    return result

