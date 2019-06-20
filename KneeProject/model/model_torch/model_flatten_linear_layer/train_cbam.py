import sys
sys.path.append('../../KLModel')
from dataloader import *
from augmentation import *
from train_utils import *
from val_utils import *
sys.path.append('../../KLModel/AttnResNet/')
import argparse
import os
import shutil
import time
import random
from sklearn.metrics import confusion_matrix, mean_squared_error, cohen_kappa_score

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torchvision.models import resnet34
from MODELS.model_resnet import *
from PIL import ImageFile
import pandas as pd

ImageFile.LOAD_TRUNCATED_IMAGES = True

if not os.path.exists('./checkpoints'):
    os.mkdir('./checkpoints')

if __name__ == '__main__':

    job_number = 5
    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda" if USE_CUDA else "cpu")
    net = resnet34(pretrained=True)
    model = ResidualNet('ImageNet', 34, 1000, 'CBAM')
    load_my_state_dict(model,net.state_dict())
    del net # remove this net
    model.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(512,5))
    criterion = nn.CrossEntropyLoss()
    print("model")
    print(model)
    # get the number of model parameters
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001,weight_decay=1e-4)

    # define the data
    HOME_PATH = '/gpfs/data/denizlab/Users/bz1030/data/OAI_processed/mix/'
    summary_path = '/gpfs/data/denizlab/Users/bz1030/data/OAI_processed/'
    log_file_path = '/gpfs/data/denizlab/Users/bz1030/KneeNet/KneeProject/model/model_torch/model_flatten_linear_layer/train_cbam_log{}'.format(
        job_number)
    model_file_path = '/gpfs/data/denizlab/Users/bz1030/KneeNet/KneeProject/model/model_torch/model_flatten_linear_layer/model_cbam_weights{}'.format(
        job_number)
    output_file_path = '/gpfs/data/denizlab/Users/bz1030/KneeNet/KneeProject/model/model_torch/model_flatten_linear_layer/train_cbam_log{}/output{}.txt' \
        .format(job_number, job_number)
    if not os.path.exists(log_file_path):
        os.makedirs(log_file_path)
    if not os.path.exists(model_file_path):
        os.makedirs(model_file_path)

    train = pd.read_csv(summary_path + 'train.csv')#.sample(n=8).reset_index()
    val = pd.read_csv(summary_path + 'val.csv')#.sample(n=8).reset_index() # split train - test set.

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
    dataset_train = KneeGradingDataset(train, HOME_PATH, tensor_transform_train, stage='train')
    dataset_val = KneeGradingDataset(val, HOME_PATH, tensor_transform_val, stage='val')
    train_loader = data.DataLoader(dataset_train, batch_size=8)
    val_loader = data.DataLoader(dataset_val, batch_size=8)
    # training parameters
    EPOCH = 2
    start_val = 0
    train_losses = []
    val_losses = []
    val_mse = []
    val_kappa = []
    val_acc = []

    best_dice = 0
    prev_model = None

    train_started = time.time()
    if USE_CUDA:
        model.cuda()
        criterion.cuda()
    load_file = None
    if load_file:
        model.load_state_dict(torch.load(load_file))
    for epoch in range(EPOCH):
        #adjust_learning_rate(optimizer,epoch,lr)

        train_loss = train_epoch(epoch, model, optimizer, train_loader, criterion, EPOCH, use_cuda=USE_CUDA,
                                 output_file_path=output_file_path)
        with open(output_file_path, 'a+') as f:
            f.write('Epoch {}: Train Loss {}\n'.format(epoch + 1, train_loss))
        if epoch >= start_val:
            start = time.time()
            val_loss, probs, truth, _ = validate_epoch(model, val_loader, criterion,USE_CUDA)
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
            with open(output_file_path, 'a+') as f:
                f.write(str(cm) + '\n')
                f.write('Epoch {}: Val Loss {}; Val Acc {}; Val MSE {}; Val Kappa {};\n' \
                        .format(epoch + 1, val_loss, acc, mse, kappa))

        # Making logs backup
        np.save(os.path.join(log_file_path, 'logs.npy'),
                [train_losses, val_losses, val_mse, val_acc, val_kappa])

        if epoch > start_val:
            # We will be saving only the snapshot which has lowest loss value on the validation set
            cur_snapshot_name = os.path.join(model_file_path, 'epoch_{}.pth'.format(epoch + 1))
            if prev_model is None:
                torch.save(model.state_dict(), cur_snapshot_name)
                prev_model = cur_snapshot_name
                best_kappa = kappa
            else:
                if kappa > best_kappa:
                    os.remove(prev_model)
                    best_kappa = kappa
                    print('Saved snapshot:', cur_snapshot_name)
                    torch.save(model.state_dict(), cur_snapshot_name)
                    prev_model = cur_snapshot_name
        gc.collect()
    print('Training took:', time.time() - train_started, 'seconds')