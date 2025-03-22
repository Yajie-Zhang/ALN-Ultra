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

from resnet50 import Model
from dataloader.dataset import *

def get_args_parser():
    parser = argparse.ArgumentParser('ResNet', add_help=False)
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
    parser.add_argument('--n_frame',default=100,type=int)
    parser.add_argument('--root',default='D:/香港 视频/',type=str)
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

def evaluation(model,data_loader,args):
    device=torch.device(args.device)
    model.eval()
    num_data=len(data_loader.dataset)
    label_bank = torch.zeros(num_data, args.num_classes).to(device)
    GT_bank = torch.zeros(num_data).to(device).long()
    with torch.no_grad():
        for i, (index, label, video_value, image_values) in enumerate(data_loader):
            image_values = image_values.to(device)
            batch_size, n_frame, channel, height, width = image_values.shape
            image_values = image_values.view(-1, channel, height, width)

            label = label.to(device)
            raw_logits = model(image_values)
            raw_logits=torch.softmax(raw_logits,dim=1)
            raw_logits=raw_logits.view(batch_size,n_frame,args.num_classes)
            raw_logits=raw_logits.mean(1)

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
    # create model
    model =Model()
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    parameters = model.parameters()
    optimizer = torch.optim.Adam(parameters, lr=args.lr, weight_decay=args.weight_decay)
    # optimizer = torch.optim.SGD(parameters, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

    train_losses = []
    train_accuracies = []
    test_auc = []

    best_acc = 0.0
    min_loss = 10000.0

    for epoch in range(args.epoch):
        total_loss = 0
        correct = 0
        total = 0
        for i_, (index, label, video_value, image_values) in enumerate(train_loader):
            model.train()
            image_values = image_values.to(device)
            batch_size, n_frame, channel, height, width = image_values.shape
            image_values = image_values.view(-1, channel, height, width)
            label = label.to(device)

            optimizer.zero_grad()

            output = model(image_values)

            label = label.view(-1, 1).repeat(1, n_frame)
            label = label.view(-1)
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
            accuracy, precision, recall, f1, auc = evaluation(model, test_loader, args)
            if avg_loss <= min_loss:
                min_loss = avg_loss
            # if train_accuracy >= best_acc:
                cur_best_results = [accuracy, precision, recall, f1, auc]
                best_acc = train_accuracy

            print(accuracy, precision, recall, f1, auc)
            test_auc.append(auc)
    return cur_best_results


def main(args):
    fix_random_seeds(args.seed)
    device = torch.device(args.device)  # use CPU or GPU
    data_list, key_list, train_index_collection, test_index_collection = data_split(args.json_path, select=True,
                                                                                    random_seed=args.seed)
    results_collection = []

    for i in range(len(train_index_collection)):
        print('=============================The ', i , '-th experiment==============================')
        cur_train_index = train_index_collection[i]
        cur_test_index = test_index_collection[i]
        cur_data_train_list = []
        cur_data_test_list = []
        for j in range(len(data_list)):
            if j in cur_train_index:
                cur_data_train_list.append(data_list[j])
            elif j in cur_test_index:
                cur_data_test_list.append(data_list[j])

        # create data loader
        train_dataset = Dataset_Video_Image(args.root, cur_data_train_list, args.img_size, args.n_frame,is_train=True)
        test_dataset = Dataset_Video_Image(args.root, cur_data_test_list, args.img_size, args.n_frame,is_train=False)
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


        cur_best_results=train(train_loader,test_loader,args)
        results_collection.append(cur_best_results)
    return results_collection


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    seed=[0, 100, 1024]
    results_all_seed=[]
    for cur_seed in seed:
        print("********************The seed is ", cur_seed, '***********************')
        args.seed=cur_seed
        print(args)
        cur_results=main(args)
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
    np.save('/home/yjzhang/code/Fujian_Video/results/image_resnet_1frame_128.npy', results_all_seed)
