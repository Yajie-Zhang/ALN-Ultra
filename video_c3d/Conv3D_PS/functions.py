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

## ------------------- label conversion tools ------------------ ##
def labels2cat(label_encoder, list):
    return label_encoder.transform(list)

def labels2onehot(OneHotEncoder, label_encoder, list):
    return OneHotEncoder.transform(label_encoder.transform(list).reshape(-1, 1)).toarray()

def onehot2labels(label_encoder, y_onehot):
    return label_encoder.inverse_transform(np.where(y_onehot == 1)[1]).tolist()

def cat2labels(label_encoder, y_cat):
    return label_encoder.inverse_transform(y_cat).tolist()


## ---------------------- Dataloaders ---------------------- ##
# for 3DCNN
class Dataset_3DCNN(data.Dataset):
    "Characterizes a dataset for PyTorch"
    def __init__(self, data_path, folders, labels, frames, transform=None):
        "Initialization"
        self.data_path = data_path
        self.labels = labels
        self.folders = folders
        self.transform = transform
        self.frames = frames

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.folders)

    def read_images(self, path, selected_folder, use_transform):
        X = []
        for i in self.frames:
            image = Image.open(os.path.join(path, selected_folder, 'frame{:06d}.jpg'.format(i))).convert('L')

            if use_transform is not None:
                image = use_transform(image)

            X.append(image.squeeze_(0))
        X = torch.stack(X, dim=0)

        return X

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        folder = self.folders[index]

        # Load data
        X = self.read_images(self.data_path, folder, self.transform).unsqueeze_(0)  # (input) spatial images
        y = torch.LongTensor([self.labels[index]])                             # (labels) LongTensor are for int64 instead of FloatTensor

        # print(X.shape)
        return X, y


# for CRNN
class Dataset_CRNN(data.Dataset):
    "Characterizes a dataset for PyTorch"
    def __init__(self, data_path, folders, labels, frames, transform=None):
        "Initialization"
        self.data_path = data_path
        self.labels = labels
        self.folders = folders
        self.transform = transform
        self.frames = frames

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.folders)

    def read_images(self, path, selected_folder, use_transform):
        X = []
        for i in self.frames:
            image = Image.open(os.path.join(path, selected_folder, 'frame{:06d}.jpg'.format(i)))

            if use_transform is not None:
                image = use_transform(image)

            X.append(image)
        X = torch.stack(X, dim=0)

        return X

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        folder = self.folders[index]

        # Load data
        X = self.read_images(self.data_path, folder, self.transform)     # (input) spatial images
        y = torch.LongTensor([self.labels[index]])                  # (labels) LongTensor are for int64 instead of FloatTensor

        # print(X.shape)
        return X, y

## ---------------------- end of Dataloaders ---------------------- ##



## -------------------- (reload) model prediction ---------------------- ##
def Conv3d_final_prediction(model, device, loader):
    model.eval()

    all_y_pred = []
    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(tqdm(loader)):
            # distribute data to device
            X = X.to(device)
            output = model(X)
            y_pred = output.max(1, keepdim=True)[1]  # location of max log-probability as prediction
            all_y_pred.extend(y_pred.cpu().data.squeeze().numpy().tolist())

    return all_y_pred


def CRNN_final_prediction(model, device, loader):
    cnn_encoder, rnn_decoder = model
    cnn_encoder.eval()
    rnn_decoder.eval()

    all_y_pred = []
    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(tqdm(loader)):
            # distribute data to device
            X = X.to(device)
            output = rnn_decoder(cnn_encoder(X))
            y_pred = output.max(1, keepdim=True)[1]  # location of max log-probability as prediction
            all_y_pred.extend(y_pred.cpu().data.squeeze().numpy().tolist())

    return all_y_pred

## -------------------- end of model prediction ---------------------- ##


class CrossAttention(nn.Module):
    def __init__(self, in_channels, n_image, n_video_frame, height, width):
        super(CrossAttention, self).__init__()
        self.n_image = n_image
        self.n_video_frame = n_video_frame
        self.in_channels = in_channels
        self.height=height
        self.width=width

        # 定义线性层用于生成查询、键和值
        self.query_layer = nn.Linear(in_channels*height*width, in_channels)
        self.key_layer = nn.Linear(in_channels*height*width, in_channels)
        self.value_layer = nn.Linear(in_channels*height*width, in_channels)

        # 输出线性层
        self.output_layer = nn.Linear(in_channels, in_channels*height*width)

    def forward(self, key_frames, video_frames):
        # key_frames的形状为 (batch, channel, height, width)
        # video_frames的形状为 (batch, channel, n_frame, height, width)

        batch_size, channels, height, width = key_frames.size()
        key_frames=key_frames.view(batch_size, self.n_image, channels, height, width)

        _,_,n_video_frame,_,_=video_frames.size()
        video_frames=video_frames.permute(0, 2, 1, 3, 4)

        # 将特征维度展平并重排列以便于使用
        key_frames = key_frames.view(batch_size, self.n_image, -1)  # (batch, n_image, channels*height*width)
        video_frames = video_frames.reshape(batch_size, n_video_frame,
                                         -1)  # (batch, n_video_frame, channels*height*width)

        # 计算查询、键和值
        queries = self.query_layer(video_frames)  # (batch, n_video_frame, channels)
        keys = self.key_layer(key_frames)  # (batch, n_image, channels)
        values = self.value_layer(key_frames)  # (batch, n_image, channels)

        # 计算注意力分数
        attention_scores = torch.bmm(queries, keys.transpose(1, 2))  # (batch, n_video_frame, n_image)
        attention_scores = F.softmax(attention_scores, dim=-1)  # 进行softmax以获得注意力权重

        # 使用注意力分数对值进行加权
        context = torch.bmm(attention_scores, values)  # (batch, n_video_frame, channels)

        # 应用输出层
        output = self.output_layer(context)  # (batch, n_video_frame, channels)

        # 可以选择将其重塑回原始维度
        output = output.view(batch_size, self.n_video_frame, channels, height,
                             width)  # (batch, n_video_frame, channels, height, width)
        output = output.permute(0, 2, 1, 3, 4)

        # output=attention_scores.mean(2)
        # output=output.view(batch_size,1,n_video_frame,1,1)

        return output

def conv3D_output_size(img_size, padding, kernel_size, stride):
    # compute output shape of conv3D
    outshape = (np.floor((img_size[0] + 2 * padding[0] - (kernel_size[0] - 1) - 1) / stride[0] + 1).astype(int),
                np.floor((img_size[1] + 2 * padding[1] - (kernel_size[1] - 1) - 1) / stride[1] + 1).astype(int),
                np.floor((img_size[2] + 2 * padding[2] - (kernel_size[2] - 1) - 1) / stride[2] + 1).astype(int))
    return outshape


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
        self.pool = nn.MaxPool3d((1,30,48))
        self.fc1 = nn.Linear(self.ch2 * self.conv2_outshape[0], self.fc_hidden1)  # fully connected hidden layer
        # self.fc1=nn.Linear(self.ch2,self.fc_hidden1)
        self.fc2 = nn.Linear(self.fc_hidden1, self.fc_hidden2)
        self.fc3 = nn.Linear(self.fc_hidden2, self.num_classes)  # fully connected layer, output = multi-classes

        self.frame_attention=nn.Sequential(
            nn.Linear(self.ch2 * self.conv2_outshape[0] * self.conv2_outshape[1] * self.conv2_outshape[2],
                      t_dim),
            nn.Sigmoid()
        )

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

        x=self.pool(x)

        # FC 1 and 2
        x = x.view(x.size(0), -1)
        x=self.frame_attention(x)
        return x

    def video_forward(self,x_3d):
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
        x=self.pool(x)

        # FC 1 and 2
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.fc3(x)
        return x


    def forward(self, x_3d, x_2d):
        ####image classification
        # fc2_2d, output_2d=self.image_forward(x_2d)
        # fc_ft=self.image_decoder(fc2_2d)
        # return fc_ft, output_2d

        ####video classification
        batch_size, channels, n_frame, height, width = x_3d.shape
        output_3d=self.video_forward(x_3d)
        return output_3d,output_3d

## --------------------- end of 3D CNN module ---------------- ##



# 使用示例
# 假设我们有以下参数
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
# cross_attention = CNN3D(t_dim=n_video_frame,img_x=height,img_y=width,n_image=n_image,n_video_frame=n_video_frame)
# output_tensor = cross_attention(input_tensor,input_image)
#
# print(output_tensor.shape)  # 应输出 (batch, n_video_frame, channels, height, width)