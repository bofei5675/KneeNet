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
    HOME_PATH = '/gpfs/data/denizlab/Users/bz1030/data/OAI_processed/mix/'
    summary_path = '/gpfs/data/denizlab/Users/bz1030/data/OAI_processed/'
    log_file_path = '/gpfs/data/denizlab/Users/bz1030/model/model_torch/model_flatten_linear_layer/train_log'
    model_file_path = '/gpfs/data/denizlab/Users/bz1030/model/model_torch/model_flatten_linear_layer/model_weights'

    test = pd.read_csv(summary_path + 'test.csv').sample(n=40000).reset_index() # split train - test set.

    start_test = 0
    tensor_transform_test = transforms.Compose([
                    CenterCrop(896),
                    transforms.ToTensor(),
                    lambda x: x.float(),
                ])
    dataset_test = KneeGradingDataset(test,HOME_PATH,tensor_transform_test,stage = 'test')

    test_loader = data.DataLoader(dataset_test,batch_size=10)
    print('Test data:', len(dataset_test))

    net = resnet34(pretrained=True)
    net.avgpool = nn.AvgPool2d(7,7)
    net.fc = nn.Sequential(nn.Dropout(0.1), nn.Linear(8192,5))
    print(net)
    # Network
    net = nn.DataParallel(net)
    optimizer = optim.Adam(net.parameters(), lr=0.0001,
                              weight_decay=0)
    net.load_state_dict(torch.load('model_weights/checkpoint_520.pth'))
    net.eval()

    print('############### Model Finished ####################')
    criterion = F.cross_entropy

    test_losses = []
    test_mse = []
    test_kappa = []
    test_acc = []


    test_started = time.time()

    test_loss, probs, truth, _ = validate_epoch(net, test_loader, criterion)
    preds = probs.argmax(1)
    # Validation metrics
    cm = confusion_matrix(truth, preds)
    kappa = np.round(cohen_kappa_score(truth, preds, weights="quadratic"), 4)
    a = np.sum(cm.diagonal().astype(float))
    b = cm.sum()
    print(a,b)
    acc = np.round( a / b , 4)
    mse = np.round(mean_squared_error(truth, preds), 4)
    test_time = np.round(time.time() - test_started, 4)
    test_losses.append(test_loss)
    test_mse.append(mse)
    test_acc.append(acc)
    test_kappa.append(kappa)

    gc.collect()
    print('Test losses {}; Test mse {}; Test acc {}; Test Kappa {};'.format(test_loss,test_mse,test_acc,kappa))
    print('Training took:', time.time() - test_started, 'seconds')

