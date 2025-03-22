import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN2D(nn.Module):
    def __init__(self, img_x=90, img_y=120, drop_p=0.2, fc_hidden1=256, fc_hidden2=128, num_classes=50):
        super(CNN2D, self).__init__()

        # 设置图像维度
        self.img_x = img_x
        self.img_y = img_y
        # 全连接层隐藏节点
        self.fc_hidden1, self.fc_hidden2 = fc_hidden1, fc_hidden2
        self.drop_p = drop_p
        self.num_classes = num_classes
        self.ch1, self.ch2 = 32, 64
        self.k1, self.k2 = (5, 5), (3, 3)  # 2D kernel size
        self.s1, self.s2 = (2, 2), (2, 2)  # 2D strides
        self.pd1, self.pd2 = (0, 0), (0, 0)  # 2D padding

        # 计算 conv1 & conv2 输出形状
        self.conv1_outshape = self.conv2d_output_size((self.img_x, self.img_y), self.pd1, self.k1, self.s1)
        self.conv2_outshape = self.conv2d_output_size(self.conv1_outshape, self.pd2, self.k2, self.s2)

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.ch1, kernel_size=self.k1, stride=self.s1,
                               padding=self.pd1)
        self.bn1 = nn.BatchNorm2d(self.ch1)
        self.conv2 = nn.Conv2d(in_channels=self.ch1, out_channels=self.ch2, kernel_size=self.k2, stride=self.s2,
                               padding=self.pd2)
        self.bn2 = nn.BatchNorm2d(self.ch2)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(self.drop_p)

        # 全连接层
        self.fc1 = nn.Linear(self.ch2 * self.conv2_outshape[0] * self.conv2_outshape[1], self.fc_hidden1)
        self.fc2 = nn.Linear(self.fc_hidden1, self.fc_hidden2)
        self.fc3 = nn.Linear(self.fc_hidden2, self.num_classes)

    def conv2d_output_size(self, input_size, padding, kernel_size, stride):
        # 计算卷积层的输出尺寸
        output_height = (input_size[0] + 2 * padding[0] - kernel_size[0]) // stride[0] + 1
        output_width = (input_size[1] + 2 * padding[1] - kernel_size[1]) // stride[1] + 1
        return (output_height, output_width)

    def forward(self, x):
        # Conv 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.drop(x)

        # Conv 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.drop(x)

        # Flatten before fully connected layers
        x = x.view(x.size(0), -1)

        # FC 1 and 2
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.fc3(x)

        return x
