import torch
import torch.nn as nn
import torch.optim as optim
import re
import numpy as np
from sklearn.model_selection import train_test_split

np.set_printoptions(threshold=np.inf)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"""
parameters: {'batch_size': 2048, 'dropout': 0.18644844084215567, 'layers': 16, 'lr': 1.3619761379362602e-05, 'optimizer': 'Adam'}. Best is trial 2 with value: 257.23429346084595.
"""

# 直接加载npy文件为numpy格式
all_data = np.load('./data/all_data.npy')
# #直接加载npy文件为numpy格式,注意标签是面心值，不是a值
all_label = np.load('./data/all_centerFace_label.npy')

all_data = torch.tensor(all_data).float()
all_label = torch.tensor(all_label).float()



#神经网络模型

class Net(nn.Module):
    def __init__(self, num_output_channels, dropout, layers, hidden_units):
        super(Net, self).__init__()

        self.layers = nn.ModuleList()  # 用于存储每个层的列表

        # 添加卷积层、激活函数、dropout层
        for i in range(layers):
            if i == 0:
                self.layers.append(nn.Conv2d(2, hidden_units, kernel_size=3, stride=1, padding=1))
            else:
                self.layers.append(nn.Conv2d(hidden_units, hidden_units, kernel_size=3, stride=1, padding=1))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(p=dropout))

        # 添加最后一层卷积层
        self.conv_final = nn.Conv2d(hidden_units, num_output_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # 运行每个层的forward方法
        for layer in self.layers:
            x = layer(x)

        x = self.conv_final(x)

        return x


import time
from sklearn.model_selection import train_test_split



def train(num_output_channels):
    start_time = time.time()

    # 定义一个batch包含的样本数目
    batch_size = 2048

    # 生成数据集
    x_train, x_test, y_train, y_test = train_test_split(all_data, all_label, test_size=0.2)

    # 设置种子数
    seed = 42
    torch.manual_seed(seed)

    # 划分数据集
    trainset = torch.utils.data.TensorDataset(x_train, y_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    # 划分数据集
    testset = torch.utils.data.TensorDataset(x_test, y_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)

    # 创建模型实例，并将模型移动到GPU设备上进行计算num_output_channels, dropout, layers, hidden_units
    net = Net(num_output_channels,0.18644844084215567,16,64).to(device)
    
    # 加速训练：如果有多个GPU，则使用DataParallel模块
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
        print("采用DataParallel加速，device_count个数为：", str(torch.cuda.device_count()))
        
    # 定义损失函数为均方误差
    criterion = nn.MSELoss()
    
    # 定义优化器为Adam优化器，设置学习率为0.001
    optimizer = optim.Adam(net.parameters(), lr=1.3619761379362602e-05)

    print('begin to train!!!')

    # 训练模型
    num_epochs = 300  # 训练轮数
    
    train_loss = []
    test_loss = []
    best_loss = float('inf')  # 初始化最佳验证集损失值为正无穷大
    patience = 18  # 设置连续多少次验证集损失值不下降时停止训练
    count = 0  # 记录连续不下降次数
    
    for epoch in range(num_epochs):
        
        #训练
        #没有加train和eval!!!!!!!!!!!!!!
        running_train_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            
        train_loss.append(running_train_loss)
        
        # 测试
        running_test_loss = 0.0
        for i, (inputs, labels) in enumerate(testloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            running_test_loss += loss.item()
            
        test_loss.append(running_test_loss)
        print("已完成第：", str(epoch+1), "个epoch! Train Loss:", running_train_loss, "Test Loss:", running_test_loss)

        # 早停法
        """
        如果<连续多个 epoch> 的<验证集损失值>都没有<下降>，即验证集损失值不再降低，那么就会认为模型已经过拟合或者无法继续改善。
        这时，训练会提前停止，并保存当前的模型参数
        """
        if running_test_loss < best_loss:

            best_loss = running_test_loss
            count = 0 #连续十次测试集的epoch loss不下降，故只要又一次下降，就清零重新计算
        else:
            count += 1
            if count >= patience:
                print(f"验证集损失值连续{patience}次不下降，停止训练！")
                break
                
        if epoch==50:
            torch.save(net, './model/model_layer16_tmp_50_batch2048_01.pth')
            np.save('./data/lossa/loss_model_layer16_tmp_50_batch2048_01.npy', np.array([train_loss, test_loss]))


    end_time = time.time()
    process_time = end_time - start_time
    print(f"模型训练和测试共用了: {process_time} 秒！")
    print('all of tasks Finished')
    
    # 保存整个模型
    torch.save(net, './model/model_300_optuna_trainloss_326_layer16_batch2048_03.pth')
    
    return train_loss, test_loss

train_loss, test_loss = train(num_output_channels=4)
print(train_loss)
print(test_loss)
np.save('./data/lossa/loss_model_300_optuna_trainloss_326_layer16_batch2048_03.npy', np.array([train_loss, test_loss]))

print(all_data.shape)
print(all_label.shape)

