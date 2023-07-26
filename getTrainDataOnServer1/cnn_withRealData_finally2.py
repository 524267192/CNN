import torch
import torch.nn as nn
import torch.optim as optim
import re
import numpy as np
from sklearn.model_selection import train_test_split

np.set_printoptions(threshold=np.inf)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
1.功能：
通过加载data_64x64x2.txt文件和label_a_2x64x64x2.txt文件，
将数据预处理将文本文件转换为模型要求的格式，然后继续训练和预测。
定义了一个6层卷积神经网络模型。每个卷积层后面跟着一个 ReLU 激活函数。第七层只有卷积，没有relu。
输入数据n*64*64*2,这里的一个样本64*64可以看成一个图片格式（在此次任务中是速度，两者类似）
输出是n*64*64*4
"""

## 从文件中读取数据并转换为三维格式
def get_data():
    input_file = 'data_copy/data_64x64x2.txt'
    all_data = []  # 存放所有数据
    single_file_data = []  # 存放一个文件的数据
    with open(input_file, 'r') as f:
        lines = f.readlines()

        # 匹配浮点数和科学计数法表示的浮点数的正则表达式
        pattern = r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?"

        matches = re.findall(pattern, str(lines))

        result = [float(match) for match in matches]

        for a in range(len(result)):
            if a == 0 or a % (4096 * 2) != 0:
                single_file_data.append(result[a])

            else:
                single_file_data = np.array(single_file_data).reshape(64, 64, 2)
                all_data.append(single_file_data)
                single_file_data = []
                single_file_data.append(result[a])
        single_file_data = np.array(single_file_data).reshape(64, 64, 2)
        all_data.append(single_file_data)
        all_data = np.array(all_data)
    return all_data


"""
从文件中读取数据并转换为三维列表格式
将6*2*64*64*2的数据转为12*64*64*2,再将其转为6*64*64*4
其中4是前面两个是竖着的a值，后面两个是横着的a值，用于作为标签
"""
def get_label():
    input_file = 'data_copy/label_a_2x64x65x2.txt'
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

        return all_4d_label


all_label = get_label()  # 获取所有特征
all_data = get_data()  # 获取所有标签

all_data = torch.tensor(all_data).float()
all_label = torch.tensor(all_label).float()

all_data = all_data.permute(0, 3, 1, 2)  # 调整维度顺序，将第4维插入到第2维
all_label = all_label.permute(0, 3, 1, 2)  # 调整维度顺序，将第4维插入到第2维

"""
维度转换：
(6, 64, 64, 2)
(6, 64, 64, 4)
转为
torch.Size([6, 2, 64, 64])
torch.Size([6, 4, 64, 64])
"""


#神经网络模型
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


def train(num_output_channels):

    # 定义一个batch包含的样本数目
    batch_size = 1

    # 生成数据集
    x_train, x_test, y_train, y_test = train_test_split(all_data, all_label, test_size=0.2)


    # 划分数据集
    trainset = torch.utils.data.TensorDataset(x_train, y_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True)

    # 划分数据集
    testset = torch.utils.data.TensorDataset(x_test, y_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=True)


    # 创建模型实例，并将模型移动到GPU设备上进行计算
    net = Net(num_output_channels).to(device)

    # 定义损失函数为均方误差
    criterion = nn.MSELoss()  # 将预测值与真实标签之间的差值求平方和，再除以样本数n来得到平均损失值

    # 定义优化器为Adam优化器
    optimizer = optim.Adam(net.parameters())

    print('begin to train!!!')

    # 训练模型
    num_epochs = 2  # 训练轮数
    for epoch in range(num_epochs):
        running_train_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)

            # 将参数的梯度设为0
            optimizer.zero_grad()

            # 前向传播+后向传播+优化
            outputs = net(inputs)
            # inputs:torch.Size([10, 2, 64, 64])
            # outputs:torch.Size([10, 4, 64, 64])
            """
            criterion：表示所选的损失函数，这里选择的是MSE均方误差
            outputs：模型在输入后得到的输出，它的形状是(batch_size, num_output_channels, height, width)
            """
            loss = criterion(outputs, labels)  # labels   :torch.Size([10, 120, 64, 64])
            loss.backward()  # 有了损失值后，就可以根据反向传播算法来更新模型参数，使得预测值更接近真实标签。
            optimizer.step()

            # 统计损失值
            running_train_loss += loss.item()  # 在训练过程中，目标就是通过反向传播和优化算法尽可能地减小该损失值，提高模型的性能。

            # 每10个batch打印一次平均损失值，这里batch-size=10,相当于每100个样本打印一次loss
            # if i % 100 == 99:
            if i % 1 == 0:
                print('[epoch:  %d, batch:%5d] train loss: %.3f' %
                      (epoch + 1, i + 1, running_train_loss))
                running_train_loss = 0.0


        running_test_loss = 0.0
        for i, (inputs, labels) in enumerate(testloader):
            inputs, labels = inputs.to(device), labels.to(device)
            # 前向传播(预测)+后向传播+优化
            outputs = net(inputs)
            # inputs:torch.Size([10, 2, 64, 64])
            # outputs:torch.Size([10, 4, 64, 64])
            """
            criterion：表示所选的损失函数，这里选择的是MSE均方误差
            outputs：模型在输入后得到的输出，它的形状是(batch_size, num_output_channels, height, width)
            """
            loss = criterion(outputs, labels)  # labels   :torch.Size([10, 120, 64, 64])

            # 统计损失值
            running_test_loss += loss.item()  # 在训练过程中，目标就是通过反向传播和优化算法尽可能地减小该损失值，提高模型的性能。

            # 每10个batch打印一次平均损失值，这里batch-size=10,相当于每100个样本打印一次loss
            # if i % 100 == 99:
            if i % 1 == 0:
                print('[epoch:  %d, batch:%5d] test loss: %.3f' %
                      (epoch + 1, i + 1, running_test_loss))
                running_test_loss = 0.0


    print('all of tasks Finished')


train(num_output_channels=4)

