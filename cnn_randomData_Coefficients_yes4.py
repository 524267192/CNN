import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

# 使用GPU加速，如果没有GPU，则使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
定义一个6层卷积神经网络模型。
每个卷积层后面跟着一个 ReLU 激活函数，
输入数据有num_data个，每个样本大小为64*64,这里的一个样本64*64可以看成一个图片格式（在此次任务中是速度，两者类似），num_data个速度代表图片的单个像素点的灰度值有num_data个
输出有num_data个，每个的大小为（num_neib-1）*64*64
"""


# CNN模型，6层卷积
class Net(nn.Module):
    def __init__(self, num_output_channels):
        super(Net, self).__init__()

        # 第一层卷积，使用64个大小为3x3的卷积核，输入数据的shape为1x64x64，使用ReLU激活函数。
        # 输出通道数恒为64, padding为默认值
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()

        # 第二层卷积，使用64个大小为3x3的卷积核，使用ReLU激活函数
        # 输出通道数恒为64, padding为默认值
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()

        # 第三层卷积，使用64个大小为3x3的卷积核，使用ReLU激活函数
        # 输出通道数恒为64, padding为默认值
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()

        # 第四层卷积，使用64个大小为3x3的卷积核，使用ReLU激活函数
        # 输出通道数恒为64, padding为默认值
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU()

        # 第五层卷积，使用64个大小为3x3的卷积核，使用ReLU激活函数
        # 输出通道数恒为64, padding为默认值, 步长为2
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.relu5 = nn.ReLU()

        # 第六层卷积，使用64个大小为3x3的卷积核，使用ReLU激活函数
        # 输出通道数为num_output_channels（变量）, padding为默认值, 步长为2
        self.conv6 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.relu6 = nn.ReLU()

        self.conv7 = nn.Conv2d(64, num_output_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # 输入数据shape: 1x64x64
        x = self.conv1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.relu4(x)

        x = self.conv5(x)
        x = self.relu5(x)

        x = self.conv6(x)
        x = self.relu6(x)

        x = self.conv7(x)

        # 输出数据shape: num_output_channelsx64x64
        return x


# 生成数据集和标签（生成num_data个64*64的随机数矩阵，以及对应的标签（64*64*15））
def generate_data(num_data):
    data = torch.randn(num_data, 1, 64, 64)  # 输入矩阵shape为n*1*64*64
    labels = torch.randn(num_data, num_neib - 1, 64, 64)  # 输出矩阵shape为num_data*（num_neib - 1）*64*64
    return data, labels


# 实现将outpuats:64*64*15（可变参数num_neib - 1））与系数矩阵coeff（15(n-1)*16（n)）相乘，最终得到一个64*64*16的矩阵。返回tensors格式
# 调用这个函数，下一步将进行插值interpolation
def multi_coefficients(outputs):
    constraint_matrix = [1] * num_neib  # num_neib = 16
    A = np.array([constraint_matrix])
    _, _, V = np.linalg.svd(A)
    null_space = V[1:, :]
    null_space = torch.Tensor(null_space)
    output_cuda = outputs.cuda()  # outputs:64*64*15(num_neib-1)
    null_space = null_space.cuda()  # 给定的15x16,即(num_neib-1）*n_s_column矩阵部署到CUDA上
    output_reshape = torch.reshape(output_cuda, (64 * 64, num_neib - 1))
    # 用矩阵乘法计算结果
    coefficients = torch.matmul(output_reshape, null_space)  # res =  (64*64, 15) * （15*16） = ((64*64) *16)
    # 将最终结果重塑为64x64xnum_neib的形状
    coefficients = torch.reshape(coefficients, (64, 64, num_neib))  # torch.Size([64, 64, 16])
    return coefficients  # 返回tensors格式,64*64*(8*num_neib),三维


# 构建CNN
def cnn_network(num_output_channels):
    # 实验数据的数量，相当于num_data个速度（64*64）
    num_data = 200

    # 定义一个batch包含的样本数目
    batch_size = 1

    # 生成数据集
    train_data, train_labels = generate_data(num_data)

    # 划分数据集
    trainset = torch.utils.data.TensorDataset(train_data, train_labels)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True)

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
        running_loss = 0.0
        # trainloader目前有200个样本，batch-size设置为n，即一个input里面有n个样本，这里设置为1，方便用于multi_coefficients函数
        for i, (input, label) in enumerate(trainloader):  # torch.Size([200, 1, 64, 64])
            input, label = input.to(device), label.to(device)  # input  torch.Size([10, 1, 64, 64]),10代表一个batch有10个样本

            # 将参数的梯度设为0
            optimizer.zero_grad()

            # 前向传播+后向传播+优化
            # output：模型在输入后得到的输出，它的形状是(batch_size, num_output_channels, height, width)
            output = net(input)  # inputs  :torch.Size([batch_size, 1, 64, 64])


            # !!!!接口，暂时未用到。将output乘以一个系数矩阵，用于保证物理性质。返回一个64*64*128的tensor格式的矩阵
            output_after_coeff = multi_coefficients(output)  # output_after_coeff: torch.Size([64, 64, 16])


            # criterion：表示所选的损失函数，这里选择的是MSE均方误差
            loss = criterion(output, label)  # labels   :torch.Size([batch_size, 120, 64, 64])
            loss.backward()  # 有了损失值后，就可以根据反向传播算法来更新模型参数，使得预测值更接近真实标签。
            optimizer.step()

            # 统计损失值
            running_loss += loss.item()  # 在训练过程中，目标就是通过反向传播和优化算法尽可能地减小该损失值，提高模型的性能。

            # 每50个batch打印一次平均损失值
            if i % 50 == 49:
                print('[epoch:  %d, batch:%5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0

    print('Finished training,begin testing!')

    # 测试数据集
    test_data, test_labels = generate_data(num_data)
    test_data = test_data.to(device)
    test_labels = test_labels.to(device)
    with torch.no_grad():
        outputs = net(test_data)
        diff = outputs - test_labels
        loss = torch.mean(torch.pow(diff, 2))
        """
        diff = outputs - test_labels：首先计算模型在测试集上的输出（outputs）与真实标签（test_labels）之间的误差，得到一个形状与两者相同的张量diff。
        torch.pow(diff, 2)：对diff中的每个元素执行平方运算，得到一个和diff形状相同但每个元素都是其平方值的张量。        
        torch.mean()：函数会对张量的所有元素求和并除以元素数量以计算出均值。输入的张量是上一步得到的每个元素的平方值，所以loss变量将得到所有误差的均方值。
        """
        print('Test loss is', loss)

    print('Finished testing')


global num_neib
num_neib = 16  # 可变参数
cnn_network(num_output_channels=(num_neib - 1))
