#!/usr/bin/env python
# coding: utf-8

"""
唯一输入的数据internal_data。格式：list[4096,2]

输出的数据result格式：shape:[[8064],[[64],[64],[64],[64],[]]]
"""

#一、导入包
import numpy as np
import torch

np.set_printoptions(threshold=np.inf)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#输入一个2x64x64的tensor格式数据，输出指定openfoam里面要求的格式数据result,按照先排列内部，再排列外部weights的两个列表
#模型输入：2x64x64,输出：4x64x64
def printStr(internal_data):


    # 二、加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load('model_300_layer8_optuna_final_03.pth')
    model = model.to(device)


    #三、预测

    # 设置模型为评估模式
    model.eval()

    #下面将将一个4096x2的list数据转为torch格式(2, 64, 64)
    tensor_data = torch.tensor(internal_data)#    # 将列表数据转换为 PyTorch 张量
    internal_data = tensor_data.view(2, 64, 64)#    # 将（2096，2）调整张量维度为 (2, 64, 64)

    # 调整输入input的维度顺序,作为E，用于下面change_Label_to_a中(E-A)/(B-A)得到a值
    matrix_64 = internal_data.cpu()
    matrix_64 = matrix_64.permute(1, 2, 0)

    # 转换为四维
    input_data = internal_data.unsqueeze(0)  # 用实际数据，cnn要求输入数据格式为(1,2, 64, 64)，不能为2x64x64
    # input_data = torch.randn(1,2, 64, 64)
    input_tensor = input_data.to(device)

    #将前向传播代码放在torch.no_grad()上下文管理器内部，可以确保在该范围内不会计算梯度，目的是提高代码的运行效率。
    with torch.no_grad():
        output = model(input_tensor)  ##如果报错的话需要把网络的设计加上，里面涉及model(input).output格式：4x64x64

    # 打印预测结果,其中4是前面两个是竖着的a值，后面两个是横着的a值
    # print(output)


    # 四、转换格式(label转为最终的weights)

    # 返回一个列表，里面嵌套两个子列表，第一个子列表存放的是内部的a值，第二个子列表存放的是边框的a值，
    # 且顺序为上（左到右），下（左到右），左（下到上），右（下到上）
    # 返回一个列表，里面嵌套两个子列表，第一个子列表存放的是内部的a值，第二个子列表存放的是边框的a值，
    # 且顺序为上（左到右），下（左到右），左（下到上），右（下到上）
    def conversion_format(output):
        # 调整output的维度顺序
        output = output.reshape(64, 64, 4)

        # 将输出output拆成两个面心值
        a_vertical = output[:, :, 0:2].cpu()
        a_horizontal = output[:, :, 2:4].cpu()

        # 将前两个元素相加除以二得到一个元素(x+y/)2
        vertical_1d = (a_vertical[:, :, 0] + a_vertical[:, :, 1]) / 2
        # 将后两个元素相加除以二得到另一个元素
        horizontal_1d = (a_horizontal[:, :, 0] + a_horizontal[:, :, 1]) / 2

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
                        inner.append(horizontal_1d[j - 1][i + 1])  # 再横着的
                        if j == 63:  # 如果j=63的话，还需要再加入最后一列的横着的边
                            inner.append(horizontal_1d[63][i + 1])  # 当i=63,横着的加最后一列
                    else:  # if i ==63 :
                        inner.append(vertical_1d[63][j])  # 当i=63,inner最后添加竖着的一行竖线


        inner = [tensor.cpu().numpy().tolist() for tensor in inner]  # 将一维列表里面的tensor元素转为numpy格式，并返回cpu版本
        four_border = [bottom_border, bottom_border, left_border, left_border]  # 顺序是上（左到右），下（左到右），左（下到上），右（下到上）
        border = [item.numpy().tolist() for sublist in four_border for item in sublist]
        result = [inner, border]
        return result


    result = conversion_format(output)
    return result

