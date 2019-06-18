"""
Main training script
(c) Aleksei Tiulpin, University of Oulu, 2017
"""
from __future__ import print_function
import sys
sys.path.append('../../KLModel')
from dataloader import *
from augmentation import *
import matplotlib
import matplotlib.pyplot as plt
from dataloader import KneeGradingDataset
from tqdm import tqdm
import numpy as np
import argparse
import os
import itertools
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
from sklearn.metrics import confusion_matrix, mean_squared_error, cohen_kappa_score, roc_auc_score, roc_curve, log_loss
from sklearn.preprocessing import OneHotEncoder
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
    HOME_PATH = '/gpfs/data/denizlab/Users/bz1030/data/OAI_processed/mix/'
    summary_path = '/gpfs/data/denizlab/Users/bz1030/data/OAI_processed/'
    log_file_path = '/gpfs/data/denizlab/Users/bz1030/KneeNet/KneeProject/model/model_torch/model_flatten_linear_layer/train_log{}'.format(job_number)
    model_file_path = '/gpfs/data/denizlab/Users/bz1030/KneeNet/KneeProject/model/model_torch/model_flatten_linear_layer/Experiment/model_resnet34_dropout0.2_weightdecay/'

    test = pd.read_csv(summary_path + 'test.csv').reset_index() # split train - test set.

    start_test = 0
    tensor_transform_test = transforms.Compose([
                    CenterCrop(896),
                    transforms.ToTensor(),
                    lambda x: x.float(),
                ])
    dataset_test = KneeGradingDataset(test,HOME_PATH,tensor_transform_test,stage = 'test')

    test_loader = data.DataLoader(dataset_test,batch_size=8)
    print('Test data:', len(dataset_test))
    net = resnet34(pretrained=True)
    net.avgpool = nn.AvgPool2d(28, 28)
    net.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(512, 5))  # OULU's paper.
    net = net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    print(net)
    # Network
    if USE_CUDA:
        net.load_state_dict(torch.load(model_file_path + '/epoch_4.pth'))
    else:
        net.load_state_dict((torch.load(model_file_path + '/epoch_4.pth',map_location='cpu')))
    net = nn.DataParallel(net)
    if USE_CUDA:
        net.cuda()
    net.eval()

    print('############### Model Finished ####################')
    criterion = F.cross_entropy

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
    test_acc.append(acc)
    test_kappa.append(kappa)

    gc.collect()
    print('Test losses {}; Test mse {}; Test acc {}; Test Kappa {};'.format(test_loss,test_mse,test_acc,kappa))
    print('Testing took:', time.time() - test_started, 'seconds')
    fpr1, tpr1, _ = roc_curve(truth > 1, probs[:, 2:].sum(1))
    auc = np.round(roc_auc_score(truth > 1, probs[:, 2:].sum(1)), 4)

    loss = np.round(log_loss(truth, probs), 4)
    plt.figure(figsize=(6, 6))
    cm = np.round(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], 2)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Greens, resample=False)
    classes = ['No OA', 'Doubtful OA', 'Early OA', 'Mild OA', 'End-stage']
    # plt.colorbar()
    tick_marks = np.arange(len(classes))

    # plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j] * 100,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('conf_rnet.jpg', bbox_inches='tight', dpi=350, pad_inches=0)
    Image.open('conf_rnet.jpg').convert('RGB').save('conf_rnet.jpg', format='JPEG', subsampling=0, quality=100)
    plt.show()

    matplotlib.rcParams.update({'font.size': 16})
    plt.figure(figsize=(6, 6))
    plt.plot(fpr1, tpr1, label='Own'.format(preds.shape[0]), lw=2, c='b')
    plt.grid()
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig('ROC_AUC_test_rnet.jpg', bbox_inches='tight', dpi=300, pad_inches=0)
    Image.open('ROC_AUC_test_rnet.jpg').convert('RGB').save('ROC_AUC_test_rnet.jpg', format='JPEG', subsampling=0,
                                                            quality=100)
    plt.show()

    print('Kappa:', kappa)
    print('Avg. class accuracy', acc)
    print('MSE', mse)
    print('AUC', auc)
    print('Cross-entropy:', loss)
