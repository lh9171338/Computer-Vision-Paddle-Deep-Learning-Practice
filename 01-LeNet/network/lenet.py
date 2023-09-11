# -*- encoding: utf-8 -*-
"""
@File    :   net.py
@Time    :   2023/09/09 11:04:07
@Author  :   lihao57
@Version :   1.0
@Contact :   lihao57@baidu.com
"""


import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class LeNet(nn.Layer):
    """
    LeNet

    """
    def __init__(self, act='ReLU', **kwargs):
        super().__init__()

        # 激活函数
        self.act = nn.__getattribute__(act)()

        # 创建卷积和池化层块，每个卷积层后面接着2x2的池化层
        # 卷积层L1: (28, 28, 1) -> (24, 24, 6)
        self.conv1 = nn.Conv2D(in_channels=1,
                                      out_channels=6,
                                      kernel_size=5,
                                      stride=1)
        # 池化层L2: (24, 24, 6) -> (12, 12, 6)
        self.pool1 = nn.MaxPool2D(kernel_size=2,
                                         stride=2)
        # 卷积层L3: (12, 12, 6) -> (8, 8, 16)
        self.conv2 = nn.Conv2D(in_channels=6,
                                      out_channels=16,
                                      kernel_size=5,
                                      stride=1)
        # 池化层L4: (8, 8, 16) -> (4, 4, 16)
        self.pool2 = nn.MaxPool2D(kernel_size=2,
                                         stride=2)
        # 线性层L5: (4, 4, 16) -> 120
        self.fc1 = nn.Linear(256, 120)
        # 线性层L6: 120 -> 84
        self.fc2 = nn.Linear(120, 84)
        # 线性层L7: 84 -> 10
        self.fc3 = nn.Linear(84, 10)
        self.softmax = nn.Softmax(axis=-1)

    def forward(self, x):
        """
        实现一个卷积神经网络模型的前向传播

        Args:
            x (paddle.Tensor): 输入的原始数据, 形状为 [batch_size, channels, height, width]

        Returns:
            paddle.Tensor: 模型的输出结果, 形状为 [batch_size, num_classes]
        """
        x = self.act(self.conv1(x))
        x = self.pool1(x)
        x = self.act(self.conv2(x))
        x = self.pool2(x)
        x = paddle.flatten(x, start_axis=1, stop_axis=-1)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        out = self.softmax(self.fc3(x))
        return out


if __name__ == '__main__':
    model = LeNet()
    print(model)
