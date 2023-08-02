#!/usr/bin/env python
# coding: utf-8

# ## 1.导入包
import torch
import torch.nn as nn
import torch.optim as optim
import re
import numpy as np
from sklearn.model_selection import train_test_split

np.set_printoptions(threshold=np.inf)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## 在每次迭代完成后调用该函数来释放未使用的显存
torch.cuda.empty_cache()

"""
TRUE
function:用来对a值进行超参数调优。
plan:考虑对a值标签进行数据预处理，删除那些>1 & <0的异常值，可以考虑数据标准化或者加入sigmoid
"""


"""
已完成第： 1 个epoch! Train Loss: 8.532438956201077 Test Loss: 1.6267098747193813
已完成第： 2 个epoch! Train Loss: 6.524033911526203 Test Loss: 1.6227780878543854
已完成第： 3 个epoch! Train Loss: 6.491955656558275 Test Loss: 1.6219284012913704
模型训练和测试共用了: 235.96732807159424 秒！
all of tasks Finished
[32m[I 2023-07-28 10:46:20,552][0m Trial 6 finished with value: 6.491955656558275 and parameters: {'batch_size': 1024, 'dropout': 0.20425596466776189, 'layers': 6, 'lr': 0.001497758981551898, 'optimizer': 'Adam'}. Best is trial 6 with value: 6.491955656558275.
"""
# ## 2.加载数据
# 直接加载npy文件为numpy格式
"""
注意这里使用了切片，并且数据所在位置是..，不是.
"""
all_data = np.load('../data/all_data.npy')
# #直接加载npy文件为numpy格式,注意标签是面心值，不是a值
all_label = np.load('../data/all_label_repair01.npy')##############换成新的处理后0-1的label

# all_data = np.load('../data/all_data.npy')[:20000]
# all_label = np.load('../data/all_label.npy')[:20000]

all_data = torch.tensor(all_data).float()
all_label = torch.tensor(all_label).float()


# ## 3.构建模型
#神经网络模型
import torch
import torch.nn as nn

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


import optuna
import time
def objective1(trial):
    batch_size = trial.suggest_categorical('batch_size', [512,1024])
    dropout = trial.suggest_uniform('dropout', 0.1, 0.5)
    layers = trial.suggest_categorical('layers', [6,8,12])
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam"])


    num_output_channels = 4
    # 生成数据集
    x_train, x_test, y_train, y_test = train_test_split(all_data, all_label, test_size=0.2)

    # 设置种子数
    seed = 42
    torch.manual_seed(seed)

    # 划分数据集
    trainset = torch.utils.data.TensorDataset(x_train, y_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True)

    # 划分数据集
    testset = torch.utils.data.TensorDataset(x_test, y_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=True)

    # 创建模型实例，并将模型移动到GPU设备上进行计算
    net = Net(num_output_channels,dropout,layers,64).to(device)

    # 加速训练：如果有多个GPU，则使用DataParallel模块
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
        print("采用DataParallel加速，device_count个数为：",str(torch.cuda.device_count()))

    # 定义损失函数为均方误差
    criterion = nn.MSELoss()

    # 获取要调优的超参数值
    optimizer = getattr(optim, optimizer_name)(net.parameters(), lr=lr)
        

    # 训练模型
    num_epochs = 6
    train_loss = []
    test_loss = []
    
    best_loss = float('inf')  # 初始化最佳验证集损失值为正无穷大
    patience = 15  # 设置连续多少次验证集损失值不下降时停止训练
    count = 0  # 记录连续不下降次数
    
    start_time =  time.time()
    for epoch in range(num_epochs):
        running_train_loss = 0.0
        net.train()#加了这个,使用dropout和batch-normalization
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()

        train_loss.append(running_train_loss)
        
        running_test_loss = 0.0
        net.eval()#加了这个,不使用dropout和batch-normalization、使用所有网络连接，不舍弃神经元，不进行反向传播
        with torch.no_grad():#节省GPU和显存
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

    
    end_time = time.time()
    process_time = end_time - start_time
    print(f"模型训练和测试共用了: {process_time} 秒！")
    print('all of tasks Finished')
    
    # 保存整个模型
#     torch.save(net, save_model_path)
    
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()

    # 返回训练集上的最终损失作为目标值
    return running_train_loss


"""
使用study.optimize方法来运行Optuna的超参数搜索过程，设置n_trials参数为搜索的迭代次数。
搜索结束后，可以通过study.best_trial获取最佳的超参数组合，打印出最佳超参数以及最小化的损失函数值。
"""
study1 = optuna.create_study(direction='minimize')
study1.optimize(objective1, n_trials=50)

print("Best trial:")
trial = study1.best_trial
print("  Value: ", trial.value)
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))
    





