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
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from functions import CNN3D
from dataloader.dataset import *
from dataloader.video_extraction import video_transforms

def get_args_parser():
    parser = argparse.ArgumentParser('Conv3D', add_help=False)
    parser.add_argument('--img_size', default=(128,200))
    parser.add_argument('--batch_size', default=2, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epoch', default=95, type=int)
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
    parser.add_argument('--root',default='D:/香港 视频/',type=str)
    parser.add_argument('--json_path',default='dataloader/data_image_selected_video.json',type=str)
    parser.add_argument('--distributed',default=True,type=bool)

    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
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

def evaluation(model,data_loader,args, att):
    device=torch.device(args.device)
    model.eval()
    num_data=len(data_loader.dataset)
    label_bank = torch.zeros(num_data, args.num_classes).to(device)
    GT_bank = torch.zeros(num_data).to(device).long()
    with torch.no_grad():
        for i, (index, label, video_value, image_values) in enumerate(data_loader):
            video_value = video_value.to(device)
            video_value = video_value.permute(0, 2, 1, 3, 4)

            image_values = image_values.to(device)
            batch_size, n_frame, channel, height, width = image_values.shape
            image_values = image_values.view(-1, channel, height, width)

            cur_att=image_values

            label = label.to(device)
            # _, raw_logits = model(video_value,image_values)
            _, raw_logits = model(video_value, cur_att)

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

def test_model(train_loader,test_loader,args, i_cross):
    device = torch.device(args.device)
    # create model
    model = CNN3D(t_dim=args.n_frame, img_x=args.img_size[0], img_y=args.img_size[1],
                  drop_p=0.0, fc_hidden1=256, fc_hidden2=256, num_classes=args.num_classes)
    state_dict = torch.load('save_models/image_model_' + str(args.seed) + '_' + str(i_cross),map_location='cpu')
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    accuracy, precision, recall, f1, auc = evaluation(model, test_loader, args)

    print(accuracy, precision, recall, f1, auc)

def train(train_loader,test_loader,args,i_cross, att_train_list, att_test_list):
    device=torch.device(args.device)
    # create model
    model = CNN3D(t_dim=args.n_frame, img_x=args.img_size[0], img_y=args.img_size[1],
                  drop_p=0.0, fc_hidden1=256, fc_hidden2=256, num_classes=args.num_classes)
    # model = CNN3D2D(t_dim=args.n_frame, img_x=args.img_size[0], img_y=args.img_size[1],
    #               drop_p=0.0, fc_hidden1=256, fc_hidden2=256, num_classes=args.num_classes)
    # model = ImgNet(t_dim=args.n_frame, img_x=args.img_size[0], img_y=args.img_size[1],
    #                 drop_p=0.0, fc_hidden1=256, fc_hidden2=256, num_classes=args.num_classes)
    # state_dict=torch.load('save_models/image_model_'+str(args.seed)+'_'+str(i_cross),map_location='cpu')
    # model.load_state_dict(state_dict,strict=False)
    model.to(device)

    # att_train_list=np.array(att_train_list)
    # att_test_list=np.array(att_test_list)
    #
    # scaler = StandardScaler()
    # att_train_list = scaler.fit_transform(att_train_list)
    # att_test_list = scaler.transform(att_test_list)

    # for name,params in model.named_parameters():
    #     params.requires_grad=True
    #     # print(name, params)

    criterion = torch.nn.CrossEntropyLoss()
    criterion_mse=torch.nn.MSELoss()
    parameters = model.parameters()
    optimizer = torch.optim.Adam(parameters, lr=args.lr, weight_decay=args.weight_decay)
    # optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)

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
            model.train()
            video_value = video_value.to(device)
            video_value = video_value.permute(0, 2, 1, 3, 4)
            label = label.to(device)

            image_values = image_values.to(device)
            batch_size, n_frame, channel, height, width = image_values.shape
            image_values = image_values.view(-1, channel, height, width)

            cur_att=image_values

            optimizer.zero_grad()
            # output_drop, output = model(video_value,image_values)
            output_drop, output = model(video_value, cur_att)
            loss = criterion(output, label)#+criterion(output_drop,label)

            # output_fc=torch.nn.functional.normalize(output_fc,dim=1)
            # label_oh=torch.nn.functional.one_hot(label,num_classes=args.num_classes).float().to(device)
            # output_fc=output_fc.matmul(output_fc.T)
            # label_oh=label_oh.matmul(label_oh.T)
            # loss=loss+criterion_mse(output_fc,label_oh)

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
            accuracy, precision, recall, f1, auc = evaluation(model, test_loader, args,att_test_list)
            if avg_loss<=min_loss:
                min_loss=avg_loss
                cur_best_results = [accuracy, precision, recall, f1, auc]

            # if train_accuracy >= best_acc:
            #     cur_best_results = [accuracy, precision, recall, f1, auc]
            #     best_acc = train_accuracy
            #     torch.save(model.state_dict(),'save_models/video_100f_'+str(args.seed)+'_'+str(i_cross))

            print(accuracy, precision, recall, f1, auc)
            test_auc.append(auc)
    return cur_best_results


def main(args):
    # utils.init_distributed_mode(args)
    fix_random_seeds(args.seed)
    device = torch.device(args.device)  # use CPU or GPU
    # selected_index = selected_video_index()
    data_list, key_list, train_index_collection, test_index_collection = data_split(args.json_path, select=True,
                                                                          random_seed=args.seed)
    # data_list, key_list, train_index_collection, test_index_collection = data_split_twice(data_list, key_list, selected_index, args.seed)


    results_collection = []
    if args.n_frame==50:
        if args.img_size[0]==128:
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
        if args.img_size[0]==128:
            video_frame1 = np.load('/nfs/yajie/video_fujian/video_values/video_frames/100_0.npy')
            video_frame2 = np.load('/nfs/yajie/video_fujian/video_values/video_frames/100_1.npy')
            video_frame3 = np.load('/nfs/yajie/video_fujian/video_values/video_frames/100_2.npy')
        else:
            video_frame1 = np.load('/nfs/yajie/video_fujian/video_values/video_frames/100_256_0.npy')
            video_frame2 = np.load('/nfs/yajie/video_fujian/video_values/video_frames/100_256_1.npy')
            video_frame3 = np.load('/nfs/yajie/video_fujian/video_values/video_frames/100_256_2.npy')

    video_frames=np.vstack((video_frame1,video_frame2,video_frame3))
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
        cur_att_train_list=[]
        cur_att_test_list=[]
        cur_video_train_list=[]
        cur_video_test_list=[]
        for j in range(len(data_list)):
            if j in cur_train_index:
                cur_data_train_list.append(data_list[j])
                cur_video_train_list.append(video_frames[j])
                cur_att_train_list.append(data_list[j])
            elif j in cur_test_index:
                cur_data_test_list.append(data_list[j])
                cur_video_test_list.append(video_frames[j])
                cur_att_test_list.append(data_list[j])

        # create data loader
        train_dataset = Dataset_Video_Image(args.root, cur_data_train_list,cur_video_train_list, args.img_size, args.n_frame,is_train=True)
        test_dataset = Dataset_Video_Image(args.root, cur_data_test_list,cur_video_test_list, args.img_size, args.n_frame,is_train=False)

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
        if i>-1:
            cur_best_results=train(train_loader,test_loader, args,i, cur_att_train_list,cur_att_test_list)
            # cur_best_results = test_model(train_loader, test_loader, args, i)
            results_collection.append(cur_best_results)
    return results_collection


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    seed = [0, 100, 1024]   #0,10, 100, 这几个随机种子还可以
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
    results_all_seed=np.array(results_all_seed)
    mean_results=np.mean(results_all_seed,axis=1)
    print('=======================The mean results for each seed============================')
    print(mean_results)
    mean_results=np.mean(mean_results,axis=0)
    print('=======================The mean results for all seed============================')
    print(mean_results)
    # np.save('/home/yjzhang/code/Fujian_Video/results/video_conv3d_'+str(args.n_frame)+'frame_'+str(args.img_size[0])+'.npy', results_all_seed)

    #  seed 0 可以  epoch=95