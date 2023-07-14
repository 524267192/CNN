#!/usr/bin/env python
# coding: utf-8

"""internal_data格式：internal_data.shape: torch.Size([2, 64, 64])

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
result格式：shape:[length=8064,length=256]
[[1,1,1,1,1,1,1,1, ... , 1][1,1,1,1,1, ..., 1]]

"""

# ## 1.导入包
import torch
import torch.nn as nn
import numpy as np

np.set_printoptions(threshold=np.inf)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 神经网络模型
class Net(nn.Module):
    def __init__(self, num_output_channels):
        super(Net, self).__init__()

        # 第一层卷积，使用64个大小为3x3的卷积核，输入数据的shape为2x64x64，使用ReLU激活函数。
        # 输入通道数为2，输出通道数恒为64
        self.conv1 = nn.Conv2d(2, 64, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()

        # 第二层卷积，使用64个大小为3x3的卷积核，使用ReLU激活函数
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()

        # 第三层卷积，使用64个大小为3x3的卷积核，使用ReLU激活函数
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()

        # 第四层卷积，使用64个大小为3x3的卷积核，使用ReLU激活函数
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU()

        # 第五层卷积，使用64个大小为3x3的卷积核，使用ReLU激活函数
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.relu5 = nn.ReLU()

        # 第六层卷积，使用64个大小为3x3的卷积核，使用ReLU激活函数
        self.conv6 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.relu6 = nn.ReLU()

        # 第七层卷积，使用64个大小为3x3的卷积核，使用ReLU激活函数,输出数据为4*64*64
        # 输入通道数为64，输出通道数恒为4
        self.conv7 = nn.Conv2d(64, num_output_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(x)  # x:torch.Size([10, 2, 64, 64])
        x = self.relu1(x)

        x = self.conv2(x)  # x:torch.Size([10, 64, 64, 64])
        x = self.relu2(x)

        x = self.conv3(x)  # x:  torch.Size([10, 64, 64, 64])
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.relu4(x)

        x = self.conv5(x)
        x = self.relu5(x)

        x = self.conv6(x)
        x = self.relu6(x)

        x = self.conv7(x)

        # 输出数据shape: torch.Size([10, 4, 64, 64]),10是batch_size大小
        return x


def printStr(internal_data):
    """
    网格搜索(Grid Search)和Optuna，找到模型的最佳超参数组合
    网格搜索适用于超参数空间较小、离散且较少的情况，而Optuna适用于超参数空间较大、连续或离散且较多的情况
    下面要做的事情：
    1.换新的面心值标签，现在数据过拟合，训练集下降但是测试集上升或者波动
    2.考虑正则化或者假如droput层来防止过拟合
    3.考虑数据预处理中采用数据标准化，让数据均匀分布
    4.用不用考虑损失函数，学习率,epoch,adam优化器及其四个参数，的修改，模型用不用再添加几层让模型变复杂些（batch-size越大，训练越快，不影响准确率）
    5.早停法（Early Stopping）：在训练过程中监控验证集上的性能，一旦性能停止改善，在一定epoch后停止训练，并保存模型，以防止过拟合。可以参照外国那案例
    6.数据集的比例，不一定4：1，也可以95：5，当数据集足够大时，这样可以增加训练集数量
    """

    # ## 3.加载模型

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载模型
    model = torch.load('model_origin_100.pth', map_location=device)  # from torchsummary import summary
    # # 将模型移动到适当的设备
    model = model.to(device)

    # ## 4.预测

    # 设置模型为评估模式
    model.eval()

    # 输入数据进行预测
    input_data = internal_data  # 你的输入数据

    # 调整输入input的维度顺序,作为E，用于下面change_Label_to_a中(E-A)/(B-A)得到a值
    matrix_64 = internal_data.cpu()
    matrix_64 = matrix_64.permute(1, 2, 0)
    print(matrix_64.shape)

    # 转换为四维
    input_data = input_data.unsqueeze(0)  # 用实际数据，数据格式为(1,2, 64, 64)，不能为2x64x64
    # input_data = torch.randn(1,2, 64, 64)
    print(input_data.shape)
    input_tensor = input_data.to(device)

    with torch.no_grad():
        output = model(input_tensor)  ##如果报错的话需要把网络的设计加上，里面涉及model(input)

    # 打印预测结果
    print(output)

    # ## 5.转换格式(label转为最终的weights)

    # 将label面心值转为a值
    def change_Label_to_a(all_vertical_edge_centers, all_horizontal_edge_centers):
        a_vertical = np.zeros((64, 65, 2))
        # a_vertical = np.random((64, 65, 2))
        a_horizontal = np.zeros((64, 65, 2))
        # a_horizontal = np.random.random((64, 65, 2))

        # 21. 求a:   横着的边，分两种情况，边缘（对称的）和非边缘的边.这里matrix_64要行列互换，因为横着时面心值是一列一列求得，竖着时是一行一行求的。
        for i in range(64):
            for j in range(65):
                if j == 0 or j == 64:
                    # a_horizontal[i, j] = (all_horizontal_edge_centers[i, 0] - matrix_64[i, 0]) / (matrix_64[i, 63] - matrix_64[i, 0])#换之前
                    a_horizontal[i, j] = (all_horizontal_edge_centers[i, 0] - matrix_64[0, i]) / (
                        matrix_64[63, i] - matrix_64[0, i])
                else:
                    # aA=(1-a)B=E   a = (E-B)/(A-B) 其中：A为matrix_64[i,j]，B为matrix_64[i,j+1]
                    # a_horizontal[i, j] = (all_horizontal_edge_centers[i, j] - matrix_64[i, j]) / (matrix_64[i, j-1] - matrix_64[i, j])
                    a_horizontal[i, j] = (all_horizontal_edge_centers[i, j] - matrix_64[j, i]) / (
                        matrix_64[j - 1, i] - matrix_64[j, i])

        # 22. 求a:   竖着的边，分两种情况，边缘（对称的）和非边缘的边
        for i in range(64):
            for j in range(65):
                if j == 0 or j == 64:
                    a_vertical[i, j] = (all_vertical_edge_centers[i, 0] - matrix_64[i, 0]) / (
                        matrix_64[i, 63] - matrix_64[i, 0])
                else:
                    # aA=(1-a)B=E   a = (E-B)/(A-B) 其中：A为matrix_64[i,j]，B为matrix_64[i,j+1]
                    # 2.错误matrix_64[i, j-1]) / (改成matrix_64[i, j]) / (
                    a_vertical[i, j] = (all_vertical_edge_centers[i, j] - matrix_64[i, j]) / (
                        matrix_64[i, j - 1] - matrix_64[i, j])

        # # 若最终a对应的矩阵里面出现无穷，则将其替换为0.5.解决了分母为0的问题
        # a_vertical[np.isinf(a_vertical)] = 0.5
        # a_horizontal[np.isinf(a_horizontal)] = 0.5
        # 这里64x65x2截成64x64x2,因为边框对称时值相同

        a_vertical = torch.tensor(a_vertical[:, :64, :])
        a_horizontal = torch.tensor(a_horizontal[:, :64, :])

        # print(a_vertical.shape)
        # print(a_horizontal.shape)
        return a_vertical, a_horizontal

    # 返回一个列表，里面嵌套两个子列表，第一个子列表存放的是内部的a值，第二个子列表存放的是边框的a值，
    # 且顺序为上（左到右），下（左到右），左（下到上），右（下到上）
    # 调整output的维度顺序
    output = output.permute(0, 2, 3, 1)
    #     print(output.shape)
    output = output[0]
    #     print(output.shape)
    # print(output)

    # 将输出output拆成两个面心值
    all_vertical_edge_centers = output[:, :, 0:2].cpu()
    all_horizontal_edge_centers = output[:, :, 2:4].cpu()
    #     print(all_vertical_edge_centers.shape)
    #     print(all_horizontal_edge_centers.shape)

    a_vertical, a_horizontal = change_Label_to_a(all_vertical_edge_centers, all_horizontal_edge_centers)

    # 将前两个元素相加除以二得到一个元素(x+y/)2
    avg_vertical = (a_vertical[:, :, 0] + a_vertical[:, :, 1]) / 2
    # 将后两个元素相加除以二得到另一个元素
    avg_horizontal = (a_horizontal[:, :, 0] + a_horizontal[:, :, 1]) / 2

    # 重新组合成新的形状为(64, 64, 2)的张量
    new_avg_a_output = torch.stack([avg_vertical, avg_horizontal], dim=2)
    # 打印转换后的数据形状
    #     print(new_avg_a_output.shape)

    # 返回两个求完平均的面心值,包括两个64*64矩阵，矩阵是求完平均后的a值，一个竖着的，一个横着的
    vertical_1d = new_avg_a_output[:, :, 0]
    horizontal_1d = new_avg_a_output[:, :, 1]
    #     print(vertical_1d.shape)
    #     print(horizontal_1d.shape)

    border = []  # 存所有边框，四个边框
    left_border = []  # 存左边框
    bottom_border = []  # 存下边框
    inner = []  # 存内部的面心值

    # 下面将2个64x64面心值变换格式，返回指定的格式result
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
    # print(inner)
    four_border = [bottom_border, bottom_border, left_border, left_border]  # 顺序是上（左到右），下（左到右），左（下到上），右（下到上）
    border = [item.numpy().tolist() for sublist in four_border for item in sublist]
    result = [inner, border]
    # print(len(result))
    print(result)
    return result


import torch

data = torch.rand(2, 64, 64)
result = printStr(data)
