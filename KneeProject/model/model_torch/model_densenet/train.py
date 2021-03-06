"""
Main training script
(c) Aleksei Tiulpin, University of Oulu, 2017
"""
from __future__ import print_function
import sys
sys.path.append('../../KLModel')
from dataloader import *
from augmentation import *

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
from torchvision.models import densenet121
import time
import pickle
import pandas as pd
import time
import sys

parser = argparse.ArgumentParser(description='Arguments for training model')

parser.add_argument('-model','--model',help='Number indicates different training models')
if __name__ == '__main__':
    args = parser.parse_args()
    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda" if USE_CUDA else "cpu")
    EPOCH = 50
    job_number = int(args.model) # get job number
    HOME_PATH = '/gpfs/data/denizlab/Users/bz1030/data/OAI_processed/mix/'
    summary_path = '/gpfs/data/denizlab/Users/bz1030/data/OAI_processed/'
    log_file_path = '/gpfs/data/denizlab/Users/bz1030/KneeNet/KneeProject/model/model_torch/model_densenet/train_log{}'.format(job_number)
    model_file_path = '/gpfs/data/denizlab/Users/bz1030/KneeNet/KneeProject/model/model_torch/model_densenet/model_weights{}'.format(job_number)
    output_file_path = '/gpfs/data/denizlab/Users/bz1030/KneeNet/KneeProject/model/model_torch/model_densenet/train_log{}/output{}.txt'\
        .format(job_number,job_number)
    if not os.path.exists(log_file_path):
        os.makedirs(log_file_path)
    if not os.path.exists(model_file_path):
        os.makedirs(model_file_path)

    train = pd.read_csv(summary_path + 'train.csv').reset_index()
    val = pd.read_csv(summary_path + 'val.csv').reset_index() # split train - test set.

    start_val = 0
    tensor_transform_train = transforms.Compose([
                    Resize(256),
                    RandomCrop(224),
                    transforms.ToTensor(),
                    lambda x: x.float(),
                ])
    tensor_transform_val = transforms.Compose([
                    Resize(256),
                    CenterCrop(224),
                    transforms.ToTensor(),
                    lambda x: x.float(),
                ])
    dataset_train = KneeGradingDataset(train,HOME_PATH,tensor_transform_train,stage = 'train')
    dataset_val = KneeGradingDataset(val,HOME_PATH,tensor_transform_val,stage = 'val')

    train_loader = data.DataLoader(dataset_train,batch_size=96)
    val_loader = data.DataLoader(dataset_val,batch_size=32)
    print('Training data: ', len(dataset_train))
    print('Validation data:', len(dataset_val))
    # Network
    net = densenet121(pretrained=True)
    net.classifier = nn.Linear(1024,5)
    net = net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.0001, weight_decay=1e-4)
    print('############### Model Finished ####################')
    criterion = F.cross_entropy

    train_losses = []
    train_accs= []
    val_losses = []
    val_mse = []
    val_kappa = []
    val_acc = []

    best_dice = 0
    prev_model = None

    train_started = time.time()
    with open(output_file_path, 'a+') as f:
        f.write('######## Train Start #######\n')
    for epoch in range(EPOCH):
        train_loss,train_acc = train_epoch(epoch,net,optimizer,train_loader,criterion,EPOCH,use_cuda = USE_CUDA,output_file_path = output_file_path)

        with open(output_file_path,'a+') as f:
            f.write('Epoch {}: Train Loss {}\n'.format(epoch + 1,train_loss))
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
            train_accs.append(train_acc)
            val_losses.append(val_loss)
            val_mse.append(mse)
            val_acc.append(acc)
            val_kappa.append(kappa)
            with open(output_file_path, 'a+') as f:
                f.write(str(cm) + '\n')
                f.write('Epoch {}: Val Loss {}; Val Acc {}; Val MSE {}; Val Kappa {};\n'\
                        .format(epoch + 1, val_loss, acc, mse, kappa))

        # Making logs backup
        np.save(os.path.join(log_file_path,'logs.npy'),
                [train_losses,train_accs, val_losses, val_mse, val_acc, val_kappa])

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


