"""
Train with maximum entropy loss.

"""
from __future__ import print_function
import sys
sys.path.append('../../KLModel')
from dataloader import *
from augmentation import *
from train_utils import *
from val_utils import *
from dataloader import KneeGradingDataset
from tqdm import tqdm
import numpy as np
import argparse
import os


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
import sys
from loss import EntropyLoss
parser = argparse.ArgumentParser(description='Arguments for training model')

parser.add_argument('-model','--model',help='Number indicates different training models')
parser.add_argument('-beta','--beta',help='Confidence Regularizer')
if __name__ == '__main__':
    args = parser.parse_args()
    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda" if USE_CUDA else "cpu")
    EPOCH = 20
    job_number = int(args.model) # get job number
    beta = float(args.beta)
    HOME_PATH = '/gpfs/data/denizlab/Users/bz1030/data/OAI_processed/mix/'
    summary_path = '/gpfs/data/denizlab/Users/bz1030/data/OAI_processed/'
    log_file_path = '/gpfs/data/denizlab/Users/bz1030/KneeNet/KneeProject/model/model_torch/model_flatten_linear_layer/train_Hloss_log{}'.format(job_number)
    model_file_path = '/gpfs/data/denizlab/Users/bz1030/KneeNet/KneeProject/model/model_torch/model_flatten_linear_layer/model_Hloss_weights{}'.format(job_number)
    output_file_path = '/gpfs/data/denizlab/Users/bz1030/KneeNet/KneeProject/model/model_torch/model_flatten_linear_layer/train_Hloss_log{}/output{}.txt'\
        .format(job_number,job_number)
    if not os.path.exists(log_file_path):
        os.makedirs(log_file_path)
    if not os.path.exists(model_file_path):
        os.makedirs(model_file_path)

    train = pd.read_csv(summary_path + 'train.csv')#.sample(n=50).reset_index()
    val = pd.read_csv(summary_path + 'val.csv')#.reset_index() # split train - test set.

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

    train_loader = data.DataLoader(dataset_train,batch_size=8)
    val_loader = data.DataLoader(dataset_val,batch_size=8)
    print('Training data: ', len(dataset_train))
    print('Validation data:', len(dataset_val))
    net = resnet34(pretrained=True)
    print(net)
    net.avgpool = nn.AvgPool2d(28, 28)
    net.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(512, 5))  # OULU's paper.
    load_file = None
    #'/gpfs/data/denizlab/Users/bz1030/KneeNet/KneeProject/model/model_torch/model_flatten_linear_layer/model_weights3/epoch_3.pth'
    if load_file:
        net.load_state_dict(torch.load(load_file))
        start_epoch = 3
    else:
        start_epoch = 0
    net = net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.0001,weight_decay=1e-4)
    # Network

    print('############### Model Finished ####################')
    criterion = EntropyLoss(beta = beta)

    train_losses = []
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
    for epoch in range(start_epoch, EPOCH):
        train_loader = data.DataLoader(dataset_train, batch_size=8, shuffle=True)
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


