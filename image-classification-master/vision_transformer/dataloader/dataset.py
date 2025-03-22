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

def select_metastasis(data_list,label_list,key_list):
    data_selected=[]
    label_selected=[]
    key_selected=[]
    for i in range(len(data_list)):
        cur_data=data_list[i]
        cur_label=label_list[i]
        cur_key=key_list[i]
        if cur_label!=2:
            data_selected.append(cur_data)
            label_selected.append(cur_label)
            key_selected.append(cur_key)
    return data_selected,label_selected,key_selected

def data_split(json_file,select=False,random_seed=0):
    np.random.seed(random_seed)
    random.seed(random_seed)
    with open(json_file, 'r', encoding='utf-8') as file:  # 使用 'r' 模式读取文件
        data_json = json.load(file)
    data_list=[]
    label_list=[]
    key_list=[]
    for key in data_json:
        data_list.append(data_json[key])
        cur_label=(data_json[key][0]).split(' ')[1]
        label_list.append(int(cur_label))
        key_list.append(key)
    if select==True:    # Remove the samples labeled as benign
        data_list,label_list,key_list=select_metastasis(data_list,label_list,key_list)
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)
    label_numpy=np.array(label_list)

    train_index_collection=[]
    test_index_collection=[]
    for train_index, test_index in kf.split(label_numpy,label_numpy):
        train_index_collection.append(train_index)
        test_index_collection.append(test_index)
    return data_list,key_list,train_index_collection,test_index_collection

class Dataset_Video_Image(Dataset):
    def __init__(self, root, data_list, img_size=(470, 800),n_frame=100,is_train=True):
        self.root = root
        self.data = data_list
        self.img_size = img_size
        self.n_frame=n_frame
        self.is_train=is_train

    def video_read(self):
        videoo_list=[]
        return 0

    def train_transforms(self, img):
        img = transforms.Resize((int(1.05 * self.img_size[0]), int(1.05 * self.img_size[1])), Image.BILINEAR)(img)
        img = transforms.RandomHorizontalFlip()(img)
        img = transforms.CenterCrop(self.img_size)(img)
        # img = transforms.ColorJitter(brightness=0.2, contrast=0.2)(img)
        img = transforms.ToTensor()(img)
        img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
        return img

    def test_transforms(self, img):
        img = transforms.Resize((int(1.05 * self.img_size[0]), int(1.05 * self.img_size[1])), Image.BILINEAR)(img)
        img = transforms.CenterCrop(self.img_size)(img)
        img = transforms.ToTensor()(img)
        img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
        return img

    def __getitem__(self, index):
        data = self.data[index]
        # Get label
        label=data[0]
        label=label.split(' ')[1]
        label=int(label)

        #Get video data
        video_path=data[1][0].split('_')[0]
        video_path=self.root+'/video_concate/'+video_path+'.avi'
        # video_value=self.video_transforms(video_path,n_frame=self.n_frame)
        video_value=0

        #Get image data
        images_path=data[2]
        image_root=self.root+'/image_pad/'
        image_values=[]
        images_path=[images_path]
        for image in images_path:
            cur_image = imageio.imread(image_root+image)
            cur_image = Image.fromarray(cur_image, mode='RGB')
            if self.is_train==True:
                cur_image=self.train_transforms(cur_image)
            else:
                cur_image=self.test_transforms(cur_image)
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


