import torch
import json
import random
import numpy as np
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import imageio
import cv2

def select_metastasis(data_list,label_list):
    data_selected=[]
    label_selected=[]
    for i in range(len(data_list)):
        cur_data=data_list[i]
        cur_label=label_list[i]
        if cur_label!=2:
            data_selected.append(cur_data)
            label_selected.append(cur_label)
    return data_selected,label_selected

def data_split(json_file,select=False,random_seed=0):
    np.random.seed(random_seed)
    random.seed(random_seed)
    with open(json_file, 'r', encoding='utf-8') as file:  # 使用 'r' 模式读取文件
        data_json = json.load(file)
    data_list=[]
    label_list=[]
    for key in data_json:
        data_list.append(data_json[key])
        cur_label=(data_json[key][0]).split(' ')[1]
        label_list.append(int(cur_label))
    if select==True:
        data_list,label_list=select_metastasis(data_list,label_list)
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)
    label_numpy=np.array(label_list)

    train_index_collection=[]
    test_index_collection=[]
    for train_index, test_index in kf.split(label_numpy,label_numpy):
        train_index_collection.append(train_index)
        test_index_collection.append(test_index)
    return data_list,train_index_collection,test_index_collection


def augment_frame(frame):
    # 随机旋转
    angle = np.random.uniform(-30, 30)
    h, w = frame.shape[1:3]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    frame = cv2.warpAffine(frame.transpose(1, 2, 0), M, (w, h))  # 转置为 (H, W, C)

    # 随机裁剪
    x = np.random.randint(0, w - 100)
    y = np.random.randint(0, h - 100)
    frame = frame[y:y + 100, x:x + 100, :]

    # 随机翻转
    if np.random.rand() > 0.5:
        frame = np.flip(frame, axis=1)  # 水平翻转

    return frame.transpose(2, 0, 1)  # 转回 (C, H, W)

def augment_video(video_tensor):
    video_tensor=torch.tensor(video_tensor)
    # 定义数据增强的转换
    transform = transforms.Compose([
        transforms.ToPILImage(),  # 将 tensor 转为 PIL 图像
        transforms.RandomRotation(30),  # 随机旋转
        # transforms.RandomCrop((120,180)),  # 随机裁剪到 100x100
        # transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.Resize((128, 200)),  # 确保尺寸为 (128, 200)
        transforms.ToTensor(),  # 转回 tensor
    ])

    # 对每一帧进行增强
    augmented_frames = []
    for i in range(video_tensor.size(0)):  # 遍历每一帧
        frame = video_tensor[i, :, :, :]  # 取出当前帧
        augmented_frame = transform(frame)  # 应用数据增强
        augmented_frames.append(augmented_frame)

    # 将增强后的帧合并回 tensor
    augmented_video_tensor = torch.stack(augmented_frames, dim=0)  # 形状为 (N, C, H, W)
    return augmented_video_tensor

class Dataset_Video_Image(Dataset):
    def __init__(self, root, data_list,video_list, img_size=(470, 800),n_frame=100,is_train=True):
        self.root = root
        self.data = data_list
        self.img_size = img_size
        self.n_frame=n_frame
        self.is_train=is_train
        # self.video_list=self.video_extract()
        self.video_list=video_list


    def video_extract(self):
        video_list=[]
        for data in self.data:
            video_path = data[1][0].split('_')[0]
            video_path = self.root + '/video_concate/' + video_path + '.avi'
            video_value = self.video_transforms(video_path, n_frame=self.n_frame)
            video_list.append(video_value)
        return video_list

    def train_transforms(self, img):
        img = transforms.Resize((int(1.05 * self.img_size[0]), int(1.05 * self.img_size[1])), Image.BILINEAR)(img)
        img = transforms.RandomHorizontalFlip()(img)
        img = transforms.CenterCrop(self.img_size)(img)
        # img = transforms.ColorJitter(brightness=0.2, contrast=0.2)(img)
        img = transforms.ToTensor()(img)
        img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
        return img

    def test_transforms(self, img, input_size):
        img = transforms.Resize((int(1.05 * self.img_size[0]), int(1.05 * self.img_size[1])), Image.BILINEAR)(img)
        img = transforms.CenterCrop(self.img_size)(img)
        img = transforms.ToTensor()(img)
        img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
        return img

    # def video_transforms(self,video_path,n_frame=50):
    #     # 定义转换
    #     normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    #     resize = transforms.Resize(self.img_size)  # Resize to (470, 800)
    #     # 使用 OpenCV 打开视频
    #     cap = cv2.VideoCapture(video_path)
    #     # 检查视频是否成功打开
    #     if not cap.isOpened():
    #         print("Error opening video file")
    #     # 获取视频的总帧数
    #     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #     # 列表来存储每一帧的处理结果
    #     frames = []
    #     if total_frames < n_frame:
    #         while True:
    #             ret, frame = cap.read()  # 读取视频的每一帧
    #             if not ret:
    #                 break  # 如果没有帧可读，则结束循环
    #             # OpenCV 读取的图像是 BGR 格式，转换为 RGB 格式
    #             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #             # 将图像转换为 PIL Image
    #             frame_pil = transforms.ToPILImage()(frame)
    #             frame_resized = resize(frame_pil)
    #             # 转换为 Tensor
    #             frame_tensor = transforms.ToTensor()(frame_resized)
    #             # 归一化处理
    #             frame_normalized = normalize(frame_tensor)
    #             # 添加到帧列表
    #             frames.append(frame_normalized)
    #
    #         # 如果帧数少于 n_frame，重复帧
    #         frames *= (n_frame // len(frames)) + 1  # 计算需要重复多少次
    #         # print(len(frames))
    #         frames = frames[:n_frame]  # 截取前 n_frame 帧
    #     else:
    #         frame_indices = torch.linspace(0, total_frames - 1, n_frame).long()  # 计算均匀采样的索引
    #         for index in frame_indices:
    #             index=int(index)
    #             cap.set(cv2.CAP_PROP_POS_FRAMES, index)  # 设置当前帧的位置
    #             ret, frame = cap.read()  # 读取指定帧
    #
    #             if not ret:
    #                 print(f"Error reading frame at index {index}")
    #                 continue  # 如果读取失败，跳过
    #
    #             # OpenCV 读取的图像是 BGR 格式，转换为 RGB 格式
    #             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #             # 将图像转换为 PIL Image
    #             frame_pil = transforms.ToPILImage()(frame)
    #             # Resize 图像到 (470, 800)
    #             frame_resized = resize(frame_pil)
    #             # 转换为 Tensor
    #             frame_tensor = transforms.ToTensor()(frame_resized)
    #             # 归一化处理
    #             frame_normalized = normalize(frame_tensor)
    #             # 添加到帧列表
    #             frames.append(frame_normalized)
    #
    #             # 释放视频捕获对象
    #     cap.release()
    #     video_tensor = torch.stack(frames)
    #     # # 变换形状为 (3, n_frame, 470, 800)
    #     # video_tensor = video_tensor.permute(1, 0, 2, 3)
    #     return video_tensor

    def __getitem__(self, index):
        data = self.data[index]
        # Get label
        label=data[0]
        label=label.split(' ')[1]
        label=int(label)

        #Get video data
        # video_path=data[1][0].split('_')[0]
        # video_path=self.root+'/video_concate/'+video_path+'.avi'
        # video_value=self.video_transforms(video_path,n_frame=self.n_frame)

        # The second way to get video data
        video_value=self.video_list[index]

        # if self.is_train==True:
        #     video_value=augment_video(video_value)

        #Get image data
        images_path=data[2]
        image_root=self.root+'/image_pad/'
        image_values=[]
        images_path=[images_path]
        for image in images_path:
            cur_image = imageio.imread(image_root+image)
            cur_image = Image.fromarray(cur_image, mode='RGB')
            cur_image=self.train_transforms(cur_image)
            image_values.append(cur_image)
        image_values=torch.stack(image_values,dim=0)
        return index, label,video_value,image_values

    def __len__(self):
        return len(self.data)

# json_file='data_image_selected_video.json'
# data_list,train_index_collection,test_index_collection=data_split(json_file,select=True)
# for i in range(len(train_index_collection)):
#     cur_train_index=train_index_collection[i]
#     cur_test_index=test_index_collection[i]
#     cur_data_train_list=[]
#     cur_data_test_list=[]
#     for j in range(len(data_list)):
#         if j in cur_train_index:
#             cur_data_train_list.append(data_list[j])
#         elif j in cur_test_index:
#             cur_data_test_list.append(data_list[j])
#     DataSet=Dataset_Video_Image('E:/香港 视频/',cur_data_train_list)
#     DataLoader=torch.utils.data.DataLoader(
#         DataSet,
#         batch_size=2,
#         num_workers=0,
#         drop_last=False,
#         shuffle=True,
#     )
#     for step, (batch) in enumerate(DataLoader):
#         print()


