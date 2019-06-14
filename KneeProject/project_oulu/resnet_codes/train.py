"""
Main training script
(c) Aleksei Tiulpin, University of Oulu, 2017
"""

from __future__ import print_function


from dataset import KneeGradingDataset, LimitedRandomSampler
from train_utils import train_epoch, adjust_learning_rate
from val_utils import validate_epoch
from tqdm import tqdm
import numpy as np
import argparse
import os

from termcolor import colored

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
from visdom import Visdom
import time
import pickle

import time
cudnn.benchmark = True
from augmentation import CenterCrop, CorrectGamma, Jitter, Rotate, CorrectBrightness, CorrectContrast
import pandas as pd
if __name__ =='__main__':
    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda" if USE_CUDA else "cpu")
    train_dir = '/gpfs/data/denizlab/Users/bz1030/KneeNet/KneeProject/project_oulu/train.csv'
    val_dir = '/gpfs/data/denizlab/Users/bz1030/KneeNet/KneeProject/project_oulu/val.csv'
    dataset = pd.read_csv(train_dir).sample(n = 20).reset_index()
    dataset_val = pd.read_csv(val_dir)
    # Defining the transforms
    # This is the transformation for each patch
    saved = '/gpfs/data/denizlab/Users/bz1030/KneeNet/KneeProject/project_oulu/resnet_codes/saved'
    if not os.path.exists(saved):
        os.mkdir(saved)
        scale_tensor_transform = transforms.Compose([
            CenterCrop(300),
            transforms.Resize(224),
            transforms.ToTensor(),
            lambda x: x.float(),
        ])
        train_ds = KneeGradingDataset(dataset,
                                      transform=scale_tensor_transform,
                                      stage='train')

        train_loader = data.DataLoader(train_ds, batch_size=256)

        mean_vector = np.zeros(3)
        std_vector = np.zeros(3)

        print(colored('==> ', 'green') + 'Estimating the mean')
        pbar = tqdm(total=len(train_loader))
        for entry in train_loader:
            batch = entry[0]
            for j in range(mean_vector.shape[0]):
                mean_vector[j] += batch[:, j, :, :].mean()
                std_vector[j] += batch[:, j, :, :].std()
            pbar.update()
        mean_vector /= len(train_loader)
        std_vector /= len(train_loader)
        pbar.close()
        print(colored('==> ', 'green') + 'Mean: ', mean_vector)
        print(colored('==> ', 'green') + 'Std: ', std_vector)
        np.save(os.path.join(saved, 'mean_std.npy'), [mean_vector, std_vector])
    else:
        tmp = np.load(os.path.join(saved, 'mean_std.npy'))
        mean_vector, std_vector = tmp
    print('Finish mean std vector ...')
    # Defining the transforms
    # This is the transformation for each patch
    normTransform = transforms.Normalize(mean_vector, std_vector)
    scale_tensor_transform = transforms.Compose([
        CenterCrop(300),
        transforms.Resize(224),
        transforms.ToTensor(),
        lambda x: x.float(),
        normTransform
    ])

    augment_transforms = transforms.Compose([
        CorrectBrightness(0.7, 1.3),
        CorrectContrast(0.7, 1.3),
        Rotate(-15, 15),
        CorrectGamma(0.5, 2.5),
        Jitter(300, 6, 20),
        scale_tensor_transform
    ])

    # Validation set
    val_ds = KneeGradingDataset(dataset_val,
                                transform=scale_tensor_transform,
                                stage='val')

    val_loader = data.DataLoader(val_ds,
                                 batch_size=8)

    net = resnet34(pretrained=True)

    net.avgpool = nn.AvgPool2d(7, 7)
    net.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(512, 5))

    # Network
    net = net.to(device)
    # Optimizer
    optimizer = optim.Adam(net.parameters(), lr=0.0001,
                           weight_decay=1e-4)
    # Criterion
    criterion = F.cross_entropy
    # Visualizer-realted variables
    vis = Visdom()
    win = None
    win_metrics = None

    train_losses = []
    val_losses = []
    val_mse = []
    val_kappa = []
    val_acc = []

    best_dice = 0
    prev_model = None

    train_started = time.time()
    batch_size = 8
    n_epochs = 20
    start_val = 0
    for epoch in range(n_epochs):
        train_ds = KneeGradingDataset(dataset,
                                     transform=augment_transforms)
        train_loader = data.DataLoader(train_ds,
                                       batch_size=batch_size,
                                       shuffle=True
                                       )

        start = time.time()
        train_loss = train_epoch(epoch, net, optimizer, train_loader, criterion, n_epochs, USE_CUDA)
        epoch_time = np.round(time.time() - start, 4)
        print(colored('==> ', 'green') + 'Epoch training time: {} s.'.format(epoch_time))
        if epoch >= start_val:
            start = time.time()
            val_loss, probs, truth, _ = validate_epoch(net, val_loader, criterion,USE_CUDA)

            preds = probs.argmax(1)
            # Validation metrics
            cm = confusion_matrix(truth, preds)
            kappa = np.round(cohen_kappa_score(truth, preds, weights="quadratic"),4)
            acc = np.round(np.mean(cm.diagonal().astype(float)/cm.sum(axis=1)),4)
            mse = np.round(mean_squared_error(truth, preds), 4)
            val_time = np.round(time.time() - start, 4)
            #Displaying the results
            print(colored('==> ', 'green')+'Kappa:', kappa)
            print(colored('==> ', 'green')+'Avg. class accuracy', acc)
            print(colored('==> ', 'green')+'MSE', mse)
            print(colored('==> ', 'green')+'Val loss:', val_loss)
            print(colored('==> ', 'green')+'Epoch val time: {} s.'.format(val_time))
            # Storing the logs
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_mse.append(mse)
            val_acc.append(acc)
            val_kappa.append(kappa)
        np.save(os.path.join(saved, 'logs.npy'),
                [train_losses, val_losses, val_mse, val_acc, val_kappa])

        if epoch > args.start_val:
            # We will be saving only the snapshot which has lowest loss value on the validation set
            cur_snapshot_name = os.path.join(saved, 'epoch_{}.pth'.format(epoch + 1))
            if prev_model is None:
                torch.save(net.state_dict(), cur_snapshot_name)
                prev_model = cur_snapshot_name
                best_kappa = kappa
            else:
                if kappa > best_kappa:
                    os.remove(prev_model)
                    best_kappa = kappa
                    torch.save(net.state_dict(), cur_snapshot_name)
                    prev_model = cur_snapshot_name

        gc.collect()

    print('Training took:', time.time() - train_started, 'seconds')








