"""
Main training script
(c) Aleksei Tiulpin, University of Oulu, 2017
"""
from __future__ import print_function
import sys
sys.path.append('../../KLModel')
from dataloader import *
from augmentation import *
from dataloader import *
from augmentation import *
from train_utils import *
from val_utils import *

from tqdm import tqdm
import numpy as np
import argparse
import os

sys.path.append('../../KLModel/AttnResNet/')
from MODELS.model_resnet import *

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
    model_file_path ='/gpfs/data/denizlab/Users/bz1030/KneeNet/KneeProject/model/model_torch/model_new_preprocessing/model_CBAM_weights1/epoch_9.pth'

    test = pd.read_csv(HOME_PATH + 'test.csv')#.sample(n=20).reset_index() # split train - test set.

    start_test = 0
    tensor_transform_test = transforms.Compose([
                    CenterCrop(896),
                    transforms.ToTensor(),
                    lambda x: x.float(),
                ])
    dataset_test = KneeGradingDatasetNew(test,HOME_PATH,tensor_transform_test,stage = 'test')

    test_loader = data.DataLoader(dataset_test,batch_size=6)
    print('Test data:', len(dataset_test))
    model = ResidualNet('ImageNet', 34, 1000, 'CBAM')
    model.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(512, 5))
    if USE_CUDA:
        model.cuda()
        state_dict = torch.load(model_file_path)
    else:
        state_dict = torch.load(model_file_path, map_location='cpu')
    own_model =model.state_dict().keys()
    load_weights = state_dict.keys()
    own_model = set(own_model)
    load_weights = set(load_weights)
    output = [len(own_model),len(load_weights),len(own_model.intersection(load_weights)),len(own_model.difference(load_weights))]
    print('Own model layers {}; Load weights layers {}; Intersections {}; Difference {};'.format(*output))
    model.load_state_dict(state_dict)
    criterion = nn.CrossEntropyLoss()
    print("model")
    print(model)
    model.eval()
    print('############### Model Finished ####################')

    test_losses = []
    test_mse = []
    test_kappa = []
    test_acc = []


    test_started = time.time()

    test_loss, probs, truth, _ = validate_epoch(model, test_loader, criterion,use_cuda = USE_CUDA)
    preds = probs.argmax(1)

    # Validation metrics
    cm = confusion_matrix(truth, preds)
    print('Confusion Matrix:\n',cm.diagonal() / cm.sum(axis = 1))
    print('Confusion Matrix:\n',cm)

    kappa = np.round(cohen_kappa_score(truth, preds, weights="quadratic"), 4)
    acc = np.round(np.mean(cm.diagonal().astype(float) / cm.sum(axis=1)), 4)
    print('Oulu Acc {}'.format(acc))
    print(cm.diagonal().astype(float) / cm.sum(axis=1))
    # mse
    mse = np.round(mean_squared_error(truth, preds), 4)
    test_time = np.round(time.time() - test_started, 4)
    test_losses.append(test_loss)
    test_mse.append(mse)
    test_acc.append(acc)
    test_kappa.append(kappa)

    gc.collect()
    print('Test losses {}; Test mse {}; Test acc {}; Test Kappa {};'.format(test_loss,test_mse,test_acc,kappa))
    print('Testing took:', time.time() - test_started, 'seconds')

