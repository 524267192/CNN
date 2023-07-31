from torch import nn

# 神经网络模型

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