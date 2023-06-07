import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

#########################2023.6.7：15：50修改后的

# 使用GPU加速，如果没有GPU，则使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
定义一个6层卷积神经网络模型。
每个卷积层后面跟着一个 ReLU 激活函数，
输入数据n*64*64,这里的一个样本64*64可以看成一个图片格式（在此次任务中是速度，两者类似），n个速度代表图片的单个像素点的灰度值有n个
输出是120*64*64
"""
class Net(nn.Module):
    def __init__(self,num_output_channels ):
        super(Net, self).__init__()

        # 第一层卷积，使用64个大小为3x3的卷积核，输入数据的shape为1x64x64，使用ReLU激活函数。
        # 输出通道数恒为64, padding为默认值
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1,padding=1)
        self.relu1 = nn.ReLU()

        # 第二层卷积，使用64个大小为3x3的卷积核，使用ReLU激活函数
        # 输出通道数恒为64, padding为默认值
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1,padding=1)
        self.relu2 = nn.ReLU()

        # 第三层卷积，使用64个大小为3x3的卷积核，使用ReLU激活函数
        # 输出通道数恒为64, padding为默认值
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1,padding=1)
        self.relu3 = nn.ReLU()

        # 第四层卷积，使用64个大小为3x3的卷积核，使用ReLU激活函数
        # 输出通道数恒为64, padding为默认值
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1,padding=1)
        self.relu4 = nn.ReLU()

        # 第五层卷积，使用64个大小为3x3的卷积核，使用ReLU激活函数
        # 输出通道数恒为64, padding为默认值, 步长为2
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1,padding=1)
        self.relu5 = nn.ReLU()

        # 第六层卷积，使用64个大小为3x3的卷积核，使用ReLU激活函数
        # 输出通道数恒为120, padding为默认值, 步长为2
        self.conv6 = nn.Conv2d(64, num_output_channels, kernel_size=3, stride=1,padding=1)
        self.relu6 = nn.ReLU()

    def forward(self, x):
        # 输入数据shape: 1x64x64
        x = self.conv1(x)#x:  torch.Size([10, 1, 64, 64])
        x = self.relu1(x)

        # shape: 64x62x62
        x = self.conv2(x)#x:torch.Size([10, 64, 62, 62])
        x = self.relu2(x)

        # shape: 64x60x60
        x = self.conv3(x)#x:  torch.Size([10, 64, 60, 60])
        x = self.relu3(x)

        # shape: 64x58x58
        x = self.conv4(x)
        x = self.relu4(x)

        # shape: 64x28x28
        x = self.conv5(x)
        x = self.relu5(x)

        # shape: 120x13x13
        x = self.conv6(x)
        x = self.relu6(x)

        # 输出数据shape: 120x6x6
        return x


def generate_data(num_data):
    """
    生成n个64*64的随机数矩阵，以及对应的标签（64*64*120）
    """
    data = torch.randn(num_data, 1, 64, 64)  # 输入矩阵shape为n*1*64*64
    labels = torch.randn(num_data, 120, 64, 64)  # 输出矩阵shape为n*120*64*64
    return data, labels


def cnn_network(num_output_channels):
    # 实验数据的数量
    num_data = 200#相当于num_data个速度（64*64）

    # 定义一个batch包含的样本数目
    batch_size = 10

    # 生成数据集
    train_data, train_labels = generate_data(num_data)

    # 划分数据集
    trainset = torch.utils.data.TensorDataset(train_data, train_labels)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True)

    # 创建模型实例，并将模型移动到GPU设备上进行计算
    net = Net(num_output_channels).to(device)

    # 定义损失函数为均方误差
    criterion = nn.MSELoss()#将预测值与真实标签之间的差值求平方和，再除以样本数n来得到平均损失值

    # 定义优化器为Adam优化器
    optimizer = optim.Adam(net.parameters())

    print('begin to train!!!')

    # 训练模型
    num_epochs = 2  # 训练轮数
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)

            # 将参数的梯度设为0
            optimizer.zero_grad()

            # 前向传播+后向传播+优化
            outputs = net(inputs)#inputs  :torch.Size([10, 1, 64, 64])
            """
            # outputs.shape: torch.Size([10, 120, 52, 52])
            # labels.shape: torch.Size([10, 120, 64, 64])
            由于padding没有弄，导致变成52*52，这里添加完padding后，输出正确
            outputs.shape: torch.Size([10, 120, 64, 64])
            labels.shape: torch.Size([10, 120, 64, 64])
            """
            """:cvar
            criterion：表示所选的损失函数，这里选择的是MSE均方误差
            outputs：模型在输入后得到的输出，它的形状是(batch_size, num_output_channels, height, width)
            """
            loss = criterion(outputs, labels)#labels   :torch.Size([10, 120, 64, 64])
            loss.backward()#有了损失值后，就可以根据反向传播算法来更新模型参数，使得预测值更接近真实标签。
            optimizer.step()

            # 统计损失值
            running_loss += loss.item()#在训练过程中，目标就是通过反向传播和优化算法尽可能地减小该损失值，提高模型的性能。

            # 每10个batch打印一次平均损失值
            if i % 10 == 9:
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


cnn_network(num_output_channels=120)
