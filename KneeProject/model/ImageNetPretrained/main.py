
from model import *
from KLModel.image_data_generator import *
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

HOME_PATH = '/gpfs/data/denizlab/Users/bz1030/data/OAI_processed/mix/'
summary_path = '/gpfs/data/denizlab/Users/bz1030/data/OAI_processed/'
batch_size = 8
train = pd.read_csv(summary_path + 'train.csv')
test = pd.read_csv(summary_path + 'test.csv') # split train - test set.

print('Training set {}, test set {}'.format(train.shape[0], test.shape[0]))
train_dataset = image_generator(batch_size=batch_size, home_path=HOME_PATH, summary=train)
test_dataset = image_generator(batch_size=batch_size,home_path = HOME_PATH,summary=test)
# transformation


resnet34,input_size = get_model()

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
X,y = next(train_dataset)
print(X.shape,y.shape)
X = torch.from_numpy(X).type(torch.LongTensor)
y = torch.from_numpy(y).type(torch.LongTensor)
X = X.permute(0,3,1,2)
print(X.shape,y.shape)

output = resnet34(X)


#print(resnet34,input_size)

