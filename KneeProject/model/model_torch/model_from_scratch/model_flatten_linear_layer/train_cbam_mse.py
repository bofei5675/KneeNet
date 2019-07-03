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
np.set_printoptions(precision=4,suppress = True)
ImageFile.LOAD_TRUNCATED_IMAGES = True

parser = argparse.ArgumentParser(description='Arguments for training model')

parser.add_argument('-model','--model',help='Number indicates different training models')
parser.add_argument('-losstype','--losstype',help='CE OR MSE')
if __name__ == '__main__':
    args = parser.parse_args()
    job_number = int(args.model)
    loss_type = str(args.losstype).upper()
    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda" if USE_CUDA else "cpu")
    net = resnet34(pretrained=True)
    model = ResidualNet('ImageNet', 34, 1000, 'CBAM')
    load_my_state_dict(model,net.state_dict())

    model.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(512,1))
    criterion = nn.MSELoss()
    print(model)
    # get the number of model parameters
    own_model = model.state_dict().keys()
    load_weights = net.state_dict().keys()
    own_model = set(own_model)
    load_weights = set(load_weights)
    output = [len(own_model), len(load_weights), len(own_model.intersection(load_weights)),
              len(own_model.difference(load_weights))]
    print('Own model layers {}; Load weights layers {}; Intersections {}; Difference {};'.format(*output))
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001,weight_decay=1e-4)
    del net  # remove this net
    # define the data
    HOME_PATH = '/gpfs/data/denizlab/Users/bz1030/data/OAI_processed/mix/'
    summary_path = '/gpfs/data/denizlab/Users/bz1030/data/OAI_processed/'
    log_file_path = '/gpfs/data/denizlab/Users/bz1030/KneeNet/KneeProject/model/model_torch/model_flatten_linear_layer/train_{}_log{}'.format(
        loss_type, job_number)
    model_file_path = '/gpfs/data/denizlab/Users/bz1030/KneeNet/KneeProject/model/model_torch/model_flatten_linear_layer/model_{}_weights{}'.format(
        loss_type, job_number)
    output_file_path = '/gpfs/data/denizlab/Users/bz1030/KneeNet/KneeProject/model/model_torch/model_flatten_linear_layer/train_{}_log{}/output{}.txt' \
        .format(loss_type, job_number, job_number)
    if not os.path.exists(log_file_path):
        os.makedirs(log_file_path)
    if not os.path.exists(model_file_path):
        os.makedirs(model_file_path)

    train = pd.read_csv(summary_path + 'train.csv')#.sample(n=16).reset_index()
    val = pd.read_csv(summary_path + 'val.csv')#.sample(n=16).reset_index() # split train - test set.
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

    dataset_val = KneeGradingDataset(val, HOME_PATH, tensor_transform_val, stage='val')
    dataset_train = KneeGradingDataset(train, HOME_PATH, tensor_transform_train, stage='train')
    train_loader = data.DataLoader(dataset_train, batch_size=8,shuffle=True)
    val_loader = data.DataLoader(dataset_val, batch_size=8)
    # training parameters

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
    load_file = None #os.path.join(model_file_path,'epoch_2.pth')
    EPOCH = 20
    start_epoch = 0
    # evaluate per iteration
    iteration = 500
    if load_file:
        model.load_state_dict(torch.load(load_file))
    for epoch in range(start_epoch,EPOCH):
        train_loader = data.DataLoader(dataset_train, batch_size=8, shuffle=True)
        train_loss,train_acc,logs = train_iterations(epoch, model,
                                      optimizer, train_loader,
                                      val_loader,criterion,
                                      EPOCH, use_cuda=USE_CUDA,
                                      output_file_path=output_file_path,
                                      iteration = iteration,
                                      start_val=start_val,
                                      model_file_path=model_file_path,
                                      loss_type=loss_type)
        train_losses += logs[0]
        val_losses += logs[1]
        val_mse += logs[2]
        val_kappa += logs[3]
        val_acc += logs[4]
        np.save(os.path.join(log_file_path, 'logs.npy'),
                [train_losses, val_losses, val_mse, val_acc, val_kappa])
        with open(output_file_path, 'a+') as f:
            f.write('Epoch {}: Train Loss {}\n'.format(epoch + 1, train_loss))