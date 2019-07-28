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
from torchvision.models import resnet34
import time
import pickle
import pandas as pd
import time
import sys
parser = argparse.ArgumentParser(description='Arguments for training model')

parser.add_argument('-model','--model',help='Number indicates different training models')
if __name__ == '__main__':
    USE_CUDA = torch.cuda.is_available()
    args = parser.parse_args()
    device = torch.device("cuda" if USE_CUDA else "cpu")
    job_number = int(args.model)
    HOME_PATH = '/gpfs/data/denizlab/Users/bz1030/data/OAI_processed_new3/'
    model_file_path = '/gpfs/data/denizlab/Users/bz1030/KneeNet/KneeProject/model/model_torch/model_new_preprocessing/model_weights1/epoch_5.pth'


    test = pd.read_csv(HOME_PATH + 'test.csv')#.reset_index() # split train - test set.

    start_test = 0
    tensor_transform_test = transforms.Compose([
                    CenterCrop(896),
                    transforms.ToTensor(),
                    lambda x: x.float(),
                ])
    dataset_test = KneeGradingDatasetNew(test,HOME_PATH,tensor_transform_test,stage = 'test')

    test_loader = data.DataLoader(dataset_test,batch_size=4)
    print('Test data:', len(dataset_test))
    net = resnet34(pretrained=True)
    net.avgpool = nn.AvgPool2d(28, 28)
    net.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(512, 5))  # OULU's paper.
    net = net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.0001,weight_decay=1e-4)
    print(net)
    '''
        # Network load parameter from given directory 
    '''
    if USE_CUDA:
        net.cuda()
        net.load_state_dict(torch.load(model_file_path))
    else:
        net.load_state_dict((torch.load(model_file_path,map_location='cpu')))
    net.eval()

    print('############### Model Finished ####################')
    criterion = nn.CrossEntropyLoss()

    test_losses = []
    test_mse = []
    test_kappa = []
    test_acc = []


    test_started = time.time()

    test_loss, probs, truth, _ = validate_epoch(net, test_loader, criterion,use_cuda = USE_CUDA)
    preds = probs.argmax(1)

    # Validation metrics
    val_correct = (preds == truth).sum()
    acc = val_correct / probs.shape[0]
    cm = confusion_matrix(truth, preds)
    print('Confusion Matrix:\n',cm)
    kappa = np.round(cohen_kappa_score(truth, preds, weights="quadratic"), 4)
    a = np.sum(cm.diagonal().astype(float))
    b = cm.sum()
    #acc = np.round( a / b , 4)
    acc2 = np.round(np.mean(cm.diagonal().astype(float) / cm.sum(axis=1)), 4)
    print('Vanilla Accuracy:{}; Oulu Acc {}'.format(acc,acc2))
    # mse
    mse = np.round(mean_squared_error(truth, preds), 4)
    test_time = np.round(time.time() - test_started, 4)
    test_losses.append(test_loss)
    test_mse.append(mse)
    test_acc.append(acc2)
    test_kappa.append(kappa)

    gc.collect()
    print('Test losses {}; Test mse {}; Test acc {}; Test Kappa {};'.format(test_loss,test_mse,test_acc,kappa))
    print('Training took:', time.time() - test_started, 'seconds')
