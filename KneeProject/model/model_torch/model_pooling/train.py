"""
Main training script
(c) Aleksei Tiulpin, University of Oulu, 2017
"""
from __future__ import print_function
import sys
sys.path.append('../../KLModel')
from dataloader import *
from argumentation import *

from dataloader import KneeGradingDataset
from tqdm import tqdm
import numpy as np
import argparse
import os

from train_utils import *
from val_utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
import gc

import torchvision.transforms as transforms
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, mean_squared_error, cohen_kappa_score
from torchvision.models import resnet34
import time
import pickle
import pandas as pd
import time


if __name__ == '__main__':
    USE_CUDA = torch.cuda.is_available()
    HOME_PATH = '/gpfs/data/denizlab/Users/bz1030/data/OAI_processed/mix/'
    summary_path = '/gpfs/data/denizlab/Users/bz1030/data/OAI_processed/'
    log_file_path = '/gpfs/data/denizlab/Users/bz1030/model/model_torch/model_pooling/train_log'
    model_file_path = '/gpfs/data/denizlab/Users/bz1030/model/model_torch/model_pooling/model_weights'

    train = pd.read_csv(summary_path + 'train.csv').reset_index()
    val = pd.read_csv(summary_path + 'val.csv').reset_index() # split train - test set.

    start_val = 0
    tensor_transform_train = transforms.Compose([
                    RandomCrop(896),
                    transforms.ToTensor(),
                    lambda x: x.float(),
                ])
    tensor_transform_val = transforms.Compose([
                    CenterCrop(896),
                    transforms.ToTensor(),
                    lambda x: x.float(),
                ])

    dataset_train = KneeGradingDataset(train,HOME_PATH,tensor_transform_train,stage = 'train')
    dataset_val = KneeGradingDataset(val,HOME_PATH,tensor_transform_val,stage = 'val')

    train_loader = data.DataLoader(dataset_train,batch_size=16)
    val_loader = data.DataLoader(dataset_val,batch_size=10)
    print('Training data: ', len(dataset_train))
    print('Validation data:', len(dataset_val))

    net = resnet34(pretrained=True)
    for param in net.parameters():
        param.requires_grad = False

    net.avgpool = nn.AvgPool2d(7,7)
    net.fc = nn.Sequential(nn.Dropout(0.1), nn.Linear(8192,512),nn.ReLU(),nn.Linear(512,5))
    # Network
    params_to_update =[]
    for name,param in net.named_parameters():
        if param.requires_grad:
            print('Tuning {}'.format(name))
            params_to_update.append(param)

    net = nn.DataParallel(net)
    optimizer = optim.Adam(params_to_update, lr=0.001,
                              weight_decay=0)
    print('############### Model Finished ####################')
    print(net)

    criterion = F.cross_entropy
    train_losses = []
    val_losses = []
    val_mse = []
    val_kappa = []
    val_acc = []

    best_dice = 0
    prev_model = None

    train_started = time.time()

    for epoch in range(10):
        train_loss = train_epoch(epoch,net,optimizer,train_loader,criterion,1000,USE_CUDA)
        print('Train Finished')

        if epoch >= start_val:
            start = time.time()
            val_loss, probs, truth, _ = validate_epoch(net, val_loader, criterion)
            preds = probs.argmax(1)
            # Validation metrics
            cm = confusion_matrix(truth, preds)
            kappa = np.round(cohen_kappa_score(truth, preds, weights="quadratic"), 4)
            acc = np.round(np.mean(cm.diagonal().astype(float) / cm.sum(axis=1)), 4)
            mse = np.round(mean_squared_error(truth, preds), 4)
            val_time = np.round(time.time() - start, 4)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_mse.append(mse)
            val_acc.append(acc)
            val_kappa.append(kappa)

        # Making logs backup
        np.save(os.path.join(log_file_path,'logs.npy'),
                [train_losses, val_losses, val_mse, val_acc, val_kappa])

        if epoch > start_val:
            # We will be saving only the snapshot which has lowest loss value on the validation set
            cur_snapshot_name = os.path.join(model_file_path, 'epoch_{}.pth'.format(epoch + 1))
            if prev_model is None:
                torch.save(net.state_dict(), cur_snapshot_name)
                prev_model = cur_snapshot_name
                best_kappa = kappa
            else:
                if kappa > best_kappa:
                    os.remove(prev_model)
                    best_kappa = kappa
                    print('Saved snapshot:', cur_snapshot_name)
                    torch.save(net.state_dict(), cur_snapshot_name)
                    prev_model = cur_snapshot_name

        gc.collect()
    print('Training took:', time.time() - train_started, 'seconds')

