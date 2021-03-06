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
sys.path.append('../../KLModel/DenseNet')
import densenet as dn
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
import gc

import torchvision.transforms as transforms
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import torchvision.models as models
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, mean_squared_error, cohen_kappa_score
import time
import pickle
import pandas as pd
import time
import sys

parser = argparse.ArgumentParser(description='Arguments for training model')

parser.add_argument('-model','--model',type=int,help='Number indicates different training models')

if __name__ == '__main__':
    args = parser.parse_args()
    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda" if USE_CUDA else "cpu")
    EPOCH = 20
    job_number = int(args.model) # get job number
    HOME_PATH = '/gpfs/data/denizlab/Users/bz1030/data/OAI_processed/mix/'
    summary_path = '/gpfs/data/denizlab/Users/bz1030/data/OAI_processed/'
    log_file_path = '/gpfs/data/denizlab/Users/bz1030/KneeNet/KneeProject/model/model_torch/model_densenet/train_large_log{}'.format(job_number)
    model_file_path = '/gpfs/data/denizlab/Users/bz1030/KneeNet/KneeProject/model/model_torch/model_densenet/model_large_weights{}'.format(job_number)
    output_file_path = '/gpfs/data/denizlab/Users/bz1030/KneeNet/KneeProject/model/model_torch/model_densenet/train_large_log{}/output{}.txt'\
        .format(job_number,job_number)
    if not os.path.exists(log_file_path):
        os.makedirs(log_file_path)
    if not os.path.exists(model_file_path):
        os.makedirs(model_file_path)

    train = pd.read_csv(summary_path + 'train.csv')#.sample(n=20).reset_index()
    val = pd.read_csv(summary_path + 'val.csv')#.sample(n=20).reset_index() # split train - test set.

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

    train_loader = data.DataLoader(dataset_train,batch_size=6)
    val_loader = data.DataLoader(dataset_val,batch_size=2)
    print('Training data: ', len(dataset_train))
    print('Validation data:', len(dataset_val))
    # Network
    net = dn.densenet121(pretrained = True)
    net.classifier = nn.Sequential(nn.Dropout(0.4),nn.Linear(1024,5))
    optimizer = optim.Adam(net.parameters(), lr=0.0001, weight_decay=1e-4)
    print(net)
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in net.parameters()])))
    print('############### Model Finished ####################')
    criterion = nn.CrossEntropyLoss()
    if USE_CUDA:
        criterion.cuda()
        net.cuda()
    train_losses = []
    train_accs= []
    val_losses = []
    val_mse = []
    val_kappa = []
    val_acc = []

    best_dice = 0
    prev_model = None
    iteration = 500
    train_started = time.time()
    with open(output_file_path, 'a+') as f:
        f.write('######## Train Start #######\n')
    for epoch in range(EPOCH):
        train_loader = data.DataLoader(dataset_train, batch_size=6, shuffle=True)
        train_loss = train_iterations(epoch, net,
                                      optimizer, train_loader,
                                      val_loader, criterion,
                                      EPOCH, use_cuda=USE_CUDA,
                                      output_file_path=output_file_path,
                                      iteration=iteration,
                                      start_val=start_val,
                                      model_file_path=model_file_path)
        with open(output_file_path, 'a+') as f:
            f.write('Epoch {}: Train Loss {}\n'.format(epoch + 1, train_loss))
        gc.collect()
    print('Training took:', time.time() - train_started, 'seconds')


