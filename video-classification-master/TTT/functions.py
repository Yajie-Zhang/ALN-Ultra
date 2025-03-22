import os
import numpy as np
from PIL import Image
from torch.utils import data
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm

import torch
from skimage import measure
from tttBase import TTTForCausalLM, TTTConfig, TTT_STANDARD_CONFIGS

## ------------------------ 3D CNN module ---------------------- ##
def conv3D_output_size(img_size, padding, kernel_size, stride):
    # compute output shape of conv3D
    outshape = (np.floor((img_size[0] + 2 * padding[0] - (kernel_size[0] - 1) - 1) / stride[0] + 1).astype(int),
                np.floor((img_size[1] + 2 * padding[1] - (kernel_size[1] - 1) - 1) / stride[1] + 1).astype(int),
                np.floor((img_size[2] + 2 * padding[2] - (kernel_size[2] - 1) - 1) / stride[2] + 1).astype(int))
    return outshape

class SimpleConvNet(nn.Module):
    def __init__(self, in_channel=48, drop_p=0.2):
        super(SimpleConvNet, self).__init__()

        # 第一层卷积，输入通道数为 channel，输出通道数为 16
        # 第一层卷积，输入通道数为 3，输出通道数为 16，步长为 2，填充为 0
        self.drop_p=drop_p
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=16, kernel_size=3, stride=2, padding=0)
        self.bn1 = nn.BatchNorm2d(16)  # 批归一化
        self.relu = nn.ReLU()  # 激活函数

        # 第二层卷积，输入通道数为 16，输出通道数为 32，步长为 2，填充为 0
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=2, padding=0)
        self.bn2 = nn.BatchNorm2d(16)

    def forward(self, x):
        # 第一层卷积
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # 第二层卷积
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        # 将特征图展平，为全连接层做准备
        out = out.view(out.size(0), -1)  # 展平
        return out

class CNN3D(nn.Module):
    def __init__(self, t_dim=120, img_x=90, img_y=120, drop_p=0.2, fc_hidden1=256, fc_hidden2=128, num_classes=50):
        super(CNN3D, self).__init__()
        # set video dimension
        self.t_dim = t_dim
        self.img_x = img_x
        self.img_y = img_y
        # fully connected layer hidden nodes
        self.fc_hidden1, self.fc_hidden2 = fc_hidden1, fc_hidden2
        self.drop_p = drop_p
        self.num_classes = num_classes
        self.ch1, self.ch2, self.ch3 = 32, 48, 96
        self.k1, self.k2, self.k3 = (5, 5, 5), (3, 3, 3), (3,3,3)  # 3d kernel size
        self.s1, self.s2, self.s3 = (2, 2, 2), (2, 2, 2), (2,2,2)  # 3d strides
        self.pd1, self.pd2, self.pd3 = (0, 0, 0), (0, 0, 0), (0,0,0)  # 3d padding

        # video embedding
        # compute conv1 & conv2 output shape
        self.conv1_outshape = conv3D_output_size((self.t_dim, self.img_x, self.img_y), self.pd1, self.k1, self.s1)
        self.conv2_outshape = conv3D_output_size(self.conv1_outshape, self.pd2, self.k2, self.s2)
        self.conv3_outshape = conv3D_output_size(self.conv2_outshape, self.pd3, self.k3, self.s3)
        self.conv1 = nn.Conv3d(in_channels=3, out_channels=self.ch1, kernel_size=self.k1, stride=self.s1,padding=self.pd1)
        self.bn1 = nn.BatchNorm3d(self.ch1)
        # self.nonbl1=NLBlockND(in_channels=self.ch1)
        self.conv2 = nn.Conv3d(in_channels=self.ch1, out_channels=self.ch2, kernel_size=self.k2, stride=self.s2,padding=self.pd2)
        self.bn2 = nn.BatchNorm3d(self.ch2)
        # self.nonbl2=NLBlockND(in_channels=self.ch2)

        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout3d(self.drop_p)
        self.pool = nn.MaxPool3d(2)

        self.ft_fc=nn.Sequential(
            nn.Linear(self.conv2_outshape[1] * self.conv2_outshape[2], 2048),
            nn.ReLU(inplace=True)
        )

        configuration = TTTConfig(
            vocab_size=2048,
            num_hidden_layers=2,
            num_attention_heads=4)
        self.ttt_model = TTTForCausalLM(configuration)
        self.classifier=nn.Linear(2048, num_classes)


    def key_frame_forward(self,x_3d):
        # embedding
        # Conv 1
        x = self.conv1(x_3d)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.drop(x)  # 8, 32, 23, 62, 98

        # Conv 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.drop(x)  # 8, 48, 11, 30, 48

        # FC 1 and 2
        x = x.view(x.size(0), -1)
        x=self.frame_attention(x)
        return x

    def video_forward(self,x_3d,att):
        # embedding
        # Conv 1
        x = self.conv1(x_3d)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.drop(x)   # 8, 32, 23, 62, 98

        # Conv 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.drop(x)  # 8, 48, 11, 30, 48

        batch_size, channels, frames, height, width=x.shape

        # FC 1 and 2
        x = x.view(batch_size, channels, frames, -1)
        x=x.view(batch_size,channels*frames, -1)

        x=self.ft_fc(x)
        x=(self.ttt_model(inputs_embeds=x))['logits']
        x=x.mean(dim=1)

        x = self.classifier(x)
        return x


    def forward(self, x_3d, att):
        ####image classification
        # fc2_2d, output_2d=self.image_forward(x_2d)
        # fc_ft=self.image_decoder(fc2_2d)
        # return fc_ft, output_2d

        ####video classification
        # batch_size, channels, n_frame, height, width = x_3d.shape
        # key_frame_attention=self.key_frame_forward(x_3d)
        # key_frame_attention=key_frame_attention.view(batch_size, 1, n_frame, 1,1)
        output_3d=self.video_forward(x_3d, att)
        # output_key_frame=self.video_forward(x_3d*key_frame_attention, att)
        return output_3d, output_3d
## --------------------- end of 3D CNN module ---------------- ##


# # 使用示例
# # 假设我们有以下参数
# batch_size = 8
# n_image = 1
# n_video_frame = 50
# channels = 3
# height, width = 128,200
#
# # 创建一个随机的输入张量
# input_tensor = torch.randn(batch_size, channels, n_video_frame, height, width)
# input_image=torch.rand(batch_size,channels,height,width)
#
# # 创建CrossAttention实例并执行前向传递
# model = CNN3D(t_dim=n_video_frame, img_x=height, img_y=width,
#                   drop_p=0.2, fc_hidden1=256, fc_hidden2=256, num_classes=2)
# # 检查各层的 requires_grad 属性
# # for layer in [model.enc1, model.enc2, model.bottleneck, model.dec2, model.dec1, model.final_conv]:
# #     for param in layer.parameters():
# #         param.requires_grad = True
# # for layer in [model.conv1, model.bn1, model.conv2, model.bn2, model.fc1, model.fc2, model.fc3]:
# #     for param in layer.parameters():
# #         param.requires_grad = False
# # for name, param in model.named_parameters():
# #     print(name, param.requires_grad)
# output_tensor = model(input_tensor,input_image)
# #
# # print(output_tensor.shape)  # 应输出 (batch, n_video_frame, channels, height, width)