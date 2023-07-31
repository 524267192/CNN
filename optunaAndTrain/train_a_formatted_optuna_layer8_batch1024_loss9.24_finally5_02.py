import torch
import torch.nn as nn
import torch.optim as optim
import re
import numpy as np
from sklearn.model_selection import train_test_split
import time
from sklearn.model_selection import train_test_split

np.set_printoptions(threshold=np.inf)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"""
已完成第： 1 个epoch! Train Loss: 7.887098919600248 Test Loss: 1.6204932369291782
已完成第： 2 个epoch! Train Loss: 6.430324245244265 Test Loss: 1.6020091660320759
已完成第： 3 个epoch! Train Loss: 6.314974147826433 Test Loss: 1.5438192784786224
已完成第： 4 个epoch! Train Loss: 6.026678558439016 Test Loss: 1.407630294561386
已完成第： 5 个epoch! Train Loss: 5.543837275356054 Test Loss: 1.264107447117567
已完成第： 6 个epoch! Train Loss: 5.146136976778507 Test Loss: 1.1853917501866817
模型训练和测试共用了: 613.6281049251556 秒！
all of tasks Finished
[32m[I 2023-07-28 23:55:49,607][0m Trial 31 finished with value: 5.146136976778507 and parameters: {'batch_size': 1024, 'dropout': 0.1309991291382774, 'layers': 8, 'lr': 0.002506915874027759, 'optimizer': 'Adam'}. Best is trial 31 with value: 5.146136976778507.
"""
# 设置路径和参数
data_path = '../data/'
model_path = './model/'
loss_path = './loss/'

num_output_channels = 4
dropout = 0.1309991291382774
layers = 8
lr = 0.002506915874027759
batch_size = 1024

model_name =  'model_300_layer8_optuna_final_03.pth'
loss_name =  'loss_model_300_layer8_optuna_final_03.npy'
tmp_model_name = 'tmp_model_final_200_01.pth'
tmp_loss_name = 'tmp_loss_model_final_200_01.npy'

# 直接加载npy文件为numpy格式
all_data = np.load(data_path + 'all_data.npy')
all_label = np.load(data_path + 'all_label_repair01.npy')#修改后的all_label_repair01.npy'

#数据集切片，减小训练时间
# slice = 10000
all_data = torch.tensor(all_data).float()
all_label = torch.tensor(all_label).float()

# print(all_data.shape)
# print(all_label.shape)


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
    


def train(num_output_channels, dropout, layers, lr, batch_size, model_name, loss_name):
    start_time = time.time()

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

    # 创建模型实例，并将模型移动到GPU设备上进行计算
    net = Net(num_output_channels, dropout, layers, 64).to(device)
    
    # 加速训练：如果有多个GPU，则使用DataParallel模块
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
        print("采用DataParallel加速，device_count个数为：", str(torch.cuda.device_count()))
        
    # 定义损失函数为均方误差
    criterion = nn.MSELoss()
    
    # 定义优化器为Adam优化器，设置学习率
    optimizer = optim.Adam(net.parameters(), lr=lr)

    print('begin to train!!!')

    # 训练模型
    num_epochs = 300  # 训练轮数
    
    train_loss = []
    test_loss = []
    best_loss = float('inf')  # 初始化最佳验证集损失值为正无穷大
    patience = 300  # 设置连续多少次验证集损失值不下降时停止训练
    count = 0  # 记录连续不下降次数
    
    for epoch in range(num_epochs):
        
        #训练
        net.train()
        running_train_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader):# trainloader是所有数据包括i个batch组
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            
        train_loss.append(running_train_loss)
        
        # 测试
        net.eval()
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

        #当epoch=30时保存一次model与loss
        if epoch == 99:
            torch.save(net, model_path + tmp_model_name)
            np.save(loss_path + tmp_loss_name, np.array([train_loss, test_loss]))

    end_time = time.time()
    process_time = end_time - start_time
    print(f"模型训练和测试共用了: {process_time} 秒！")
    print('all of tasks Finished')
    
    # 保存整个模型
    torch.save(net, model_path +model_name)
    #保存损失函数
    np.save(loss_path +loss_name, np.array([train_loss, test_loss]))
    print(train_loss)
    print(test_loss)
    return train_loss, test_loss

#开始训练
train(num_output_channels, dropout, layers, lr, batch_size, model_name, loss_name)


