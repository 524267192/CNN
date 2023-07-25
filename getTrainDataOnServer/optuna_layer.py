#!/usr/bin/env python
# coding: utf-8

# ## 1.导入包

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
import re
import numpy as np
from sklearn.model_selection import train_test_split

np.set_printoptions(threshold=np.inf)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[2]:


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


# ## 2.加载数据

# """
# 1.功能：
# 通过加载data和label文件，然后继续训练和预测。
# 定义了一个6层卷积神经网络模型。每个卷积层后面跟着一个 ReLU 激活函数。第七层只有卷积，没有relu。
# 输入数据n*64*64*2,这里的一个样本64*64可以看成一个图片格式（在此次任务中是速度，两者类似）
# 输出是n*64*64*4
# """
# """txt保存为numpy格式发现可以减少存储大小，约缩小成1/4
# 5.9G	./all_data.npy
# 12G	./all_label.npy
# 27G	./data_64x64x2.txt
# 53G	./label_a_2x64x65x2.txt
# """

# In[3]:


# 直接加载npy文件为numpy格式
all_data = np.load('./data/all_data.npy')
# #直接加载npy文件为numpy格式,注意标签是面心值，不是a值
all_label = np.load('./data/all_centerFace_label.npy')

all_data = torch.tensor(all_data).float()
all_label = torch.tensor(all_label).float()


# ## 3.构建模型

# In[4]:


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



# In[5]:


"""
早停法是一种被广泛使用的方法，在很多案例上都比正则化的方法要好。
其基本含义是在训练中计算模型在验证集上的表现，
当模型在验证集上的表现开始下降的时候，停止训练，这样就能避免继续训练导致过拟合的问题。
"""


# In[6]:


save_model_path = './model/model_optuna_tmp1.pth'
save_loss_path = './data/lossa/loss_model_optuna_tmp1.npy'


# ## 4.模型训练与测试




import optuna
import time
def objective1(trial):
    batch_size = trial.suggest_categorical('batch_size', [128, 256 ,512])
    dropout = trial.suggest_uniform('dropout', 0.1, 0.5)
    layers = trial.suggest_categorical('layers', [4,6,8,16,24])


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
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam"])
    optimizer = getattr(optim, optimizer_name)(net.parameters(), lr=lr)
        

    # 训练模型
    num_epochs = 20
    train_loss = []
    test_loss = []
    
    best_loss = float('inf')  # 初始化最佳验证集损失值为正无穷大
    patience = 15  # 设置连续多少次验证集损失值不下降时停止训练
    count = 0  # 记录连续不下降次数
    
    start_time =  time.time()
    for epoch in range(num_epochs):
        running_train_loss = 0.0
        net.train()#加了这个
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
        net.eval()
        with torch.no_grad():
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

#             if epoch==500:
#                 torch.save(net, './model/model_origin_500_02.pth')
    #     注意
    #缺少保存loss代码
    #急停法和optuna能不能同时用，看那个外国视频代码
    
    end_time = time.time()
    process_time = end_time - start_time
    print(f"模型训练和测试共用了: {process_time} 秒！")
    print('all of tasks Finished')
    
    # 保存整个模型
    torch.save(net, save_model_path)
    
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()

    # 返回验证集上的最终损失作为目标值
    return running_train_loss



# In[ ]:


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
    





