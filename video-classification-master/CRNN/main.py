import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score
import argparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from dataloader.dataset import *
from functions import *
from dataloader.video_extraction import video_transforms

def get_args_parser():
    parser = argparse.ArgumentParser('Conv3D', add_help=False)
    parser.add_argument('--img_size', default=(128,200))
    parser.add_argument('--batch_size', default=2, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epoch', default=100, type=int)
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--device', default='cpu',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--num_classes', type=int,default=2)
    parser.add_argument('--n_frame',default=50,type=int)
    parser.add_argument('--root',default='E:/香港 视频/',type=str)
    parser.add_argument('--json_path',default='dataloader/data_image_selected_video.json',type=str)
    return parser

def fix_random_seeds(seed=27):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def eval_tfpn(output, target, args):
    output=torch.softmax(output,dim=1)
    output=torch.max(output,dim=1)[1]
    output=output.cpu().numpy()
    target=target.cpu().numpy()

    tp = np.sum(output*target)
    fp = np.sum(output*(1-target))
    fn = np.sum((1-output)*target)
    tn = np.sum((1-output)*(1-target))

    acc = (tp+tn)/(tp+tn+fp+fn)
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    F1 =2*precision*recall/(precision+recall)

    return acc, precision, recall, F1

def evaluation(cnn_encoder,rnn_decoder,data_loader,args):
    device=torch.device(args.device)
    cnn_encoder.eval()
    rnn_decoder.eval()
    num_data=len(data_loader.dataset)
    label_bank = torch.zeros(num_data, args.num_classes).to(device)
    GT_bank = torch.zeros(num_data).to(device).long()
    with torch.no_grad():
        for i, (index, label, video_value, image_values) in enumerate(data_loader):
            video_value = video_value.to(device)
            # video_value = video_value.permute(0, 2, 1, 3, 4)

            label = label.to(device)
            raw_logits = rnn_decoder(cnn_encoder(video_value))

            label_bank[index] = raw_logits
            GT_bank[index] = label
        # print(GT_bank.shape,GT_bank.sum())

    pred_classes = torch.argmax(label_bank, dim=1)

    # 将 PyTorch 张量转换为 NumPy 数组
    pred_classes_np = pred_classes.cpu().numpy()
    GT_np = GT_bank.cpu().numpy()

    # 计算指标
    accuracy = accuracy_score(GT_np, pred_classes_np)
    precision = precision_score(GT_np, pred_classes_np, average='binary')  # 适用于二分类
    recall = recall_score(GT_np, pred_classes_np, average='binary')  # 适用于二分类
    f1 = f1_score(GT_np, pred_classes_np, average='binary')  # 适用于二分类

    # 计算 AUC
    # 需要获取概率值，而不是类别标签
    # 通过 softmax 将得分转换为概率
    softmax_pred = torch.softmax(label_bank, dim=1)[:, 1]  # 获取正类的概率
    auc = roc_auc_score(GT_np, softmax_pred.detach().cpu().numpy())  # 计算 AUC
    return accuracy,precision,recall,f1,auc

def train(train_loader,test_loader,args):
    device=torch.device(args.device)
    # EncoderCNN architecture
    CNN_fc_hidden1, CNN_fc_hidden2 = 1024, 768
    CNN_embed_dim = 512  # latent dim extracted by 2D CNN
    img_x, img_y = args.img_size[0], args.img_size[1]  # resize video 2d frame size
    dropout_p = 0.0  # dropout probability

    # DecoderRNN architecture
    RNN_hidden_layers = 1
    RNN_hidden_nodes = 512
    RNN_FC_dim = 256

    # create model

    cnn_encoder = EncoderCNN(img_x=img_x, img_y=img_y, fc_hidden1=CNN_fc_hidden1, fc_hidden2=CNN_fc_hidden2,
                             drop_p=dropout_p, CNN_embed_dim=CNN_embed_dim).to(device)

    rnn_decoder = DecoderRNN(CNN_embed_dim=CNN_embed_dim, h_RNN_layers=RNN_hidden_layers, h_RNN=RNN_hidden_nodes,
                             h_FC_dim=RNN_FC_dim, drop_p=dropout_p, num_classes=args.num_classes).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    parameters = list(cnn_encoder.parameters()) + list(rnn_decoder.parameters())
    optimizer = torch.optim.Adam(parameters, lr=args.lr, weight_decay=args.weight_decay)
    # optimizer = torch.optim.SGD(parameters, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

    train_losses = []
    train_accuracies = []
    test_auc = []

    best_acc = 0.0
    min_loss=10000.0

    for epoch in range(args.epoch):
        total_loss = 0
        correct = 0
        total = 0
        for i_, (index, label, video_value, image_values) in enumerate(train_loader):
            cnn_encoder.train()
            rnn_decoder.train()
            video_value = video_value.to(device)
            # video_value=video_value.permute(0,2,1,3,4)
            label = label.to(device)

            optimizer.zero_grad()
            output = rnn_decoder(cnn_encoder(video_value))
            loss = criterion(output, label)

            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()

            loss.backward()
            optimizer.step()
        avg_loss = total_loss / len(train_loader)
        train_accuracy = correct / total

        train_losses.append(avg_loss)
        train_accuracies.append(train_accuracy)

        print('[epoch:] ', epoch, '. Avg loss: ', avg_loss, '. Train acc: ', train_accuracy)

        if epoch % 1 == 0:
            accuracy, precision, recall, f1, auc = evaluation(cnn_encoder, rnn_decoder, test_loader, args)
            if avg_loss <= min_loss:
                min_loss = avg_loss
            # if train_accuracy >= best_acc:
                cur_best_results = [accuracy, precision, recall, f1, auc]
                best_acc = train_accuracy

            print(accuracy, precision, recall, f1, auc)
            test_auc.append(auc)
    return cur_best_results


def selected_video_index():
    a = '8  10  11  12  15  25  28  33  44  46  50  58  67  73  74  75  79  92,  96 100 105 110 113 116 123 124 127 130 131 135 140 142 149 151 157 168, 170 171 172 176 188 192 202 204 214 218 226 228 231 233 249 254 0 7  17  21  22  32  35  36  43  54  60  61  62  69  87  94  98 118, 120 121 125 132 133 134 141 144 147 154 155 164 166 173 178 187 193 196, 197 198 200 201 206 209 220 224 225 229 230 237 244 251 253 255 16  19  24  26  31  40  41  42  45  51  55  56  59  64  65  82  86  90,  93  95 101 102 106 108 109 115 126 128 136 145 152 156 160 163 174 179, 182 183 185 191 199 211 212 219 221 223 232 239 242 243 246 17  22  43  56  58  60  65  66  68  71  72  80  86  88  89  93  96 104, 110 121 122 126 127 129 137 153 155 161 162 171 175 177 186 188 191 195, 197 205 206 207 211 215 221 230 231 232 233 246 247 251 253 255 10  16  18  30  34  42  44  45  48  51  53  55  63  70  87  91  97  98, 100 103 106 108 112 113 118 123 125 131 132 138 142 144 165 176 179 183, 185 192 200 202 203 220 227 234 236 237 239 241 242 244 249 0   1   2   4  11  27  41  50  54  56  72  75  81  83  93  97 106 109, 111 115 121 123 124 127 129 133 136 138 143 145 152 160 162 171 175 179, 181 183 186 192 205 207 214 218 219 228 230 233 247 255 256 1  22  24  39  41  42  43  45  47  54  59  62  67  69  75  84  85  86,  87  94  97 101 104 106 107 109 110 111 112 114 124 127 141 161 162 164, 180 187 192 193 198 201 207 212 215 219 227 228 231 236 252 1   5   8  10  12  14  21  24  27  35  36  41  43  46  56  58  61  62,  64  67  70  82  83  86  88  97 100 102 103 104 106 116 118 129 132 135, 140 142 147 157 161 162 176 179 186 194 215 219 227 239 256'
    numbers = a.replace(',', ' ').split()
    # 将字符串转换为整数并使用 set 去重
    unique_numbers = set(int(num) for num in numbers)
    # 将不重复的数字排序
    sorted_unique_numbers = sorted(unique_numbers)
    sorted_unique_numbers = list(sorted_unique_numbers)
    return sorted_unique_numbers


def data_split_twice(data_original,index_selected,random_seed=0):
    np.random.seed(random_seed)
    random.seed(random_seed)
    data_list=[]
    label_list=[]
    key_list=[]
    num_selected=len(index_selected)
    for i in range(num_selected):
        cur_data=data_original[index_selected[i]]
        # cur_key=key_original[index_selected[i]]

        data_list.append(cur_data)
        # key_list.append(cur_key)
        label_list.append(int(cur_data[0].split(' ')[0]))

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)
    label_numpy=np.array(label_list)

    train_index_collection=[]
    test_index_collection=[]
    for train_index, test_index in kf.split(label_numpy,label_numpy):
        train_index_collection.append(train_index)
        test_index_collection.append(test_index)
    return data_list,key_list,train_index_collection,test_index_collection


def main(args):
    fix_random_seeds(args.seed)
    device = torch.device(args.device)  # use CPU or GPU
    selected_index = selected_video_index()
    data_list, train_index_collection, test_index_collection = data_split(args.json_path, select=True,
                                                                                    random_seed=args.seed)
    # data_list, key_list, train_index_collection, test_index_collection = data_split_twice(data_list,
    #                                                                                       selected_index, args.seed)


    results_collection = []
    if args.n_frame == 50:
        if args.img_size[0] == 128:
            video_frame1 = np.load('/nfs/yajie/video_fujian/video_values/video_frames/50_0.npy')
            video_frame2 = np.load('/nfs/yajie/video_fujian/video_values/video_frames/50_1.npy')
            video_frame3 = np.load('/nfs/yajie/video_fujian/video_values/video_frames/50_2.npy')
            # video_frame1 = np.load('C:/Yajie/PycharmProjects/Fujian_Video/video_frames/50_0.npy')
            # video_frame2 = np.load('C:/Yajie/PycharmProjects/Fujian_Video/video_frames/50_1.npy')
            # video_frame3 = np.load('C:/Yajie/PycharmProjects/Fujian_Video/video_frames/50_2.npy')
        else:
            video_frame1 = np.load('/nfs/yajie/video_fujian/video_values/video_frames/50_256_0.npy')
            video_frame2 = np.load('/nfs/yajie/video_fujian/video_values/video_frames/50_256_1.npy')
            video_frame3 = np.load('/nfs/yajie/video_fujian/video_values/video_frames/50_256_2.npy')
    else:
        if args.img_size[0] == 128:
            video_frame1 = np.load('/nfs/yajie/video_fujian/video_values/video_frames/100_0.npy')
            video_frame2 = np.load('/nfs/yajie/video_fujian/video_values/video_frames/100_1.npy')
            video_frame3 = np.load('/nfs/yajie/video_fujian/video_values/video_frames/100_2.npy')
        else:
            video_frame1 = np.load('/nfs/yajie/video_fujian/video_values/video_frames/100_256_0.npy')
            video_frame2 = np.load('/nfs/yajie/video_fujian/video_values/video_frames/100_256_1.npy')
            video_frame3 = np.load('/nfs/yajie/video_fujian/video_values/video_frames/100_256_2.npy')
    video_frames = np.vstack((video_frame1, video_frame2, video_frame3))
    del video_frame1
    del video_frame2
    del video_frame3

    # video_frames = video_frames[selected_index]
    print(video_frames.shape)

    for i in range(len(train_index_collection)):
        print('=============================The ', i, '-th experiment==============================')
        cur_train_index = train_index_collection[i]
        cur_test_index = test_index_collection[i]
        cur_data_train_list = []
        cur_data_test_list = []
        cur_video_train_list = []
        cur_video_test_list = []
        for j in range(len(data_list)):
            if j in cur_train_index:
                cur_data_train_list.append(data_list[j])
                cur_video_train_list.append(video_frames[j])
            elif j in cur_test_index:
                cur_data_test_list.append(data_list[j])
                cur_video_test_list.append(video_frames[j])

        # create data loader
        train_dataset = Dataset_Video_Image(args.root, cur_data_train_list, cur_video_train_list, args.img_size,
                                            args.n_frame, is_train=True)
        test_dataset = Dataset_Video_Image(args.root, cur_data_test_list, cur_video_test_list, args.img_size,
                                           args.n_frame, is_train=False)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            drop_last=False,
            shuffle=True,
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            drop_last=False,
            shuffle=False,
        )
        cur_best_results=train(train_loader,test_loader, args)
        results_collection.append(cur_best_results)
    return results_collection


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    seed = [0, 100, 1024]
    results_all_seed = []
    for cur_seed in seed:
        print("********************The seed is ", cur_seed, '***********************')
        args.seed = cur_seed
        print(args)
        cur_results = main(args)
        results_all_seed.append(cur_results)
    for cur_results in results_all_seed:
        print('===================================================')
        for sub_cur_res in cur_results:
            print(sub_cur_res)
    results_all_seed = np.array(results_all_seed)
    mean_results = np.mean(results_all_seed, axis=1)
    print('=======================The mean results for each seed============================')
    print(mean_results)
    mean_results = np.mean(mean_results, axis=0)
    print('=======================The mean results for all seed============================')
    print(mean_results)
    np.save('/home/yjzhang/code/Fujian_Video/results/video_crnn_'+str(args.n_frame)+'frame_'+str(args.img_size[0])+'.npy', results_all_seed)