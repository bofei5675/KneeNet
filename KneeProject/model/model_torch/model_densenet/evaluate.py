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
parser.add_argument('-pool','--pool',type=str,help='Pool method')

class Identity(nn.Module):
    def __init__(self):
        super(Identity,self).__init__()

    def forward(self, inp):
        return inp
class DenseNet(nn.Module):
    def __init__(self,depth = 121, pretrain = True, pool = 'max'):
        super(DenseNet, self).__init__()
        if depth == 121:
            self.dn = models.densenet121(pretrained=pretrain)
        elif depth ==161:
            self.dn = models.densenet161(pretrained=pretrain)
        self.dn.classifier = Identity()
        if pool == 'max':
            self.pool = nn.AdaptiveMaxPool1d(1024)
        elif pool == 'avg':
            self.pool = nn.AdaptiveAvgPool1d(1024)
        self.fc = nn.Sequential(nn.Dropout(0.2),nn.Linear(1024,5))
    def forward(self, inp):
        output = self.dn(inp)
        output = output.unsqueeze(0)
        output = self.pool(output)
        output = output.squeeze(0)
        output = self.fc(output)
        return output

if __name__ == '__main__':
    args = parser.parse_args()
    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda" if USE_CUDA else "cpu")
    job_number = int(args.model) # get job number
    pool = args.pool
    HOME_PATH = '/gpfs/data/denizlab/Users/bz1030/data/OAI_processed/mix/'
    summary_path = '/gpfs/data/denizlab/Users/bz1030/data/OAI_processed/'
    model_file_path = '/gpfs/data/denizlab/Users/bz1030/KneeNet/KneeProject/model/model_torch/model_densenet/model_large_weights3/epoch_17.pth'
    val = pd.read_csv(summary_path + 'test.csv').sample(n=5).reset_index() # split train - test set.
    tensor_transform_test = transforms.Compose([
                    CenterCrop(896),
                    transforms.ToTensor(),
                    lambda x: x.float(),
                ])
    dataset_test = KneeGradingDataset(val,HOME_PATH,tensor_transform_test,stage = 'val')

    test_loader = data.DataLoader(dataset_test,batch_size=2)
    print('Validation data:', len(dataset_test))
    # Network
    #net = DenseNet(121,True,pool)
    net = dn.densenet121(pretrained = True)
    net.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(1024, 5))
    print(net)
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in net.parameters()])))
    print('############### Model Finished ####################')
    criterion = nn.CrossEntropyLoss()
    if USE_CUDA:
        state_dict = torch.load(model_file_path)
        net.load_state_dict(state_dict)
        criterion.cuda()
        net.cuda()
    else:
        state_dict = torch.load(model_file_path,map_location='cpu')
        net.load_state_dict(state_dict)

    train_losses = []
    train_accs= []
    test_losses = []
    test_mse = []
    test_kappa = []
    test_acc = []

    best_dice = 0
    prev_model = None

    test_started = time.time()

    test_loss, probs, truth, _ = validate_epoch(net, test_loader, criterion, use_cuda=USE_CUDA)
    preds = probs.argmax(1)

    # Validation metrics
    val_correct = (preds == truth).sum()
    acc = val_correct / probs.shape[0]
    cm = confusion_matrix(truth, preds)
    print('Confusion Matrix:\n', cm)
    print('Class Accuracy:\n', cm.diagonal() / cm.sum(axis = 1))
    kappa = np.round(cohen_kappa_score(truth, preds, weights="quadratic"), 4)
    # acc = np.round( a / b , 4)
    acc2 = np.round(np.mean(cm.diagonal().astype(float) / cm.sum(axis=1)), 4)
    print('Vanilla Accuracy:{}; Oulu Acc {}'.format(acc, acc2))
    # mse
    mse = np.round(mean_squared_error(truth, preds), 4)
    test_time = np.round(time.time() - test_started, 4)
    test_losses.append(test_loss)
    test_mse.append(mse)
    test_acc.append(acc)
    test_kappa.append(kappa)

    gc.collect()
    print('Test losses {}; Test mse {}; Test acc {}; Test Kappa {};'.format(test_loss, test_mse, test_acc, kappa))
    print('Training took:', time.time() - test_started, 'seconds')


