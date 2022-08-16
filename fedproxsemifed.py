import argparse
import os
import matplotlib
import matplotlib.pyplot as plt
import copy
import random
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from urllib import request
import json
import math
import re
import sys
import pandas
from torchvision import datasets, transforms
import torch.nn.functional as F
from sklearn.model_selection import train_test_split



def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--num_users', type=int, default=100)
    parser.add_argument('--frac', type=float, default=0.1)
    parser.add_argument('--local_ep', type=int, default=2,)
    parser.add_argument('--local_bs', type=int, default=128)
    parser.add_argument('--bs', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.5)
    # model arguments
    parser.add_argument('--model', type=str, default='cnn')
    parser.add_argument('--kernel_num', type=int, default=9)
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',)
    parser.add_argument('--norm', type=str, default='batch_norm')
    parser.add_argument('--num_filters', type=int, default=32)
    parser.add_argument('--max_pool', type=str, default='True',)
    # other arguments
    parser.add_argument('--iid', action='store_true')
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--num_channels', type=int, default=3)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--stopping_rounds', type=int, default=10)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--decay_rate', type=float, default=0.99)
    parser.add_argument('--num_clusters', type=int, default=2)
    parser.add_argument('--mu', type=int, default=0.3)
    args = parser.parse_args()
    return args

class CNNMnist(nn.Module):
    def __init__(self, args):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(10) # bn1
        self.bn2 = nn.BatchNorm2d(20) # bn2
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, args.num_classes)
#######################################
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(F.max_pool2d(out, 2))
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.conv2_drop(out)
        out = F.relu(F.max_pool2d(out, 2))
        out = out.view(-1, out.shape[1]*out.shape[2]*out.shape[3])
        out = F.relu(self.fc1(out))
        out = F.dropout(out, training=self.training)
        out = self.fc2(out)
        return F.log_softmax(out, dim=1)
###########################################

def mnist_noniid(dataset, num_users):
    num_shards, num_imgs = 100, 600  # num_shards * num_imgs = 60000
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    labels = np.array(dataset.targets)
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    # divide and assign
    for i in range(num_users):
        rand_set = {idx_shard[0]}
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
    return dict_users

def cluster_iid(dict_user, cluster_num):
    cluster = {i: {} for i in range(cluster_num)}
    user_list = list(range(len(dict_user)))
    for i in range(cluster_num):
        tmp = {}
        idxs = [j for j in range(i, len(dict_user), 10)]
        for idx in idxs:
            tmp[idx] = dict_user[idx]
        cluster[i] = tmp
    return cluster

def SemiFedAvg(w_clusters):
    avgweight = copy.deepcopy(w_clusters[0])
    for k in avgweight.keys():
        for i in range(1, len(w_clusters)):
            avgweight[k] += w_clusters[i][k]
        avgweight[k] = torch.div(avgweight[k], len(w_clusters))
    return avgweight

def test_img(model_g, datatest, args):
    model_g.eval()
    # testing
    test_loss = 0
    correct = 0
    data_loader = DataLoader(datatest, batch_size=args.bs,shuffle=False)
    l = len(data_loader)
    for idx, (data, target) in enumerate(data_loader):
        if args.gpu != -1:
            data, target = data.to(args.device), target.to(args.device)
        log_probs = model_g(data)
        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * int(correct) / len(data_loader.dataset)
    return accuracy, test_loss

class DatasetSplit(CNNMnist):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

    def __len__(self):
        return len(self.idxs)

class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

    def train(self, model, stepsize):
        model.train()
        # train and update
        optimizer = torch.optim.SGD(model.parameters(), lr=stepsize, momentum=0.5, weight_decay=1e-5)

        epoch_loss = []

        for iter in range(self.args.local_ep):
            batch_loss = []

            for batch_idx, (images, labels) in enumerate(self.ldr_train):

                images, labels = images.to(self.args.device), labels.to(self.args.device)
                model.zero_grad() # zero the gradients before running the backward pass
                log_probs = model(images)



                loss = self.loss_func(log_probs, labels)
                #  localw - sharedw
                tensor_1 = list(model_glob.parameters())
                tensor_2 = list(model_tmp.parameters())
                difference = sum([torch.sum((tensor_1[i] - tensor_2[i]) ** 2)
                                  for i in range(len(tensor_1))])
                loss += ((args.mu / 2) * difference)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

if __name__ == '__main__':

    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
    acc_threshold = 0
    # load dataset and split users
    trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
    dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
    print('Non-i.i.d data')
    print(f"Device' = {args.device}")
    dict_users = mnist_noniid(dataset_train, args.num_users)

    # build model
    model_glob = CNNMnist(args=args).to(args.device)
    # divide 100 clients into 10 clusters

    clusters = cluster_iid(dict_users, 10)

    # train mode
    model_tmp = copy.deepcopy(model_glob)
    model_glob.train()
    w_glob = model_glob.state_dict() # copy weights
    acc_test_lst, loss_train_lst = [], []
    step_size = args.lr
    # training
    for iter in range(args.epochs):

        w_clusters, loss_clusters = [], []
        model_glob.train()
        # decaying learning rate
        if iter>=10 and iter%10 == 0:
            step_size *= args.decay_rate
            if step_size <= 1e-8:
                step_size = 1e-8

        for idx_cluster, _users in clusters.items():
            idx_users, loss_local = [], []

            model_tmp = copy.deepcopy(model_glob)

            for user_key, user_val in _users.items():
                idx_users.append(int(user_key))

            # shuffle the in-cluster sequential order and randomly select a CH
            random.shuffle(idx_users)
            # each cluster is performed parallel

            for idx in idx_users:

                local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])

                wa, loss = local.train(model=copy.deepcopy(model_tmp).to(args.device), stepsize=step_size, )

                loss_local.append(copy.deepcopy(loss))

                model_tmp.load_state_dict(wa)

            w_clusters.append(copy.deepcopy(wa))

            loss_clusters.append(sum(loss_local) / len(loss_local))

        loss_avg = sum(loss_clusters)/len(loss_clusters)

        w_glob = SemiFedAvg(w_clusters)

        model_glob.load_state_dict(w_glob)

        # test
        model_glob.eval()
        acc_train, loss_train = test_img(model_glob.to(args.device), dataset_train, args)
        acc_test, loss_test = test_img(model_glob.to(args.device), dataset_test, args)

        print('Round:{} Test Accuracy:{:.2f}'.format(iter, acc_test))

        acc_test_lst.append(acc_test)

    t= np.arange(args.epochs)
    s = acc_test_lst
    fig, ax = plt.subplots()
    ax.plot(t, s)
    ax.set(xlabel='Round number', ylabel='Test Accuracy',
           title='SemiFed NonIID')
    fig.savefig("test.png")
    plt.show()