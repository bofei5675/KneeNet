"""
Dataset classes and samplers
(c) Aleksei Tiulpin, University of Oulu, 2017
"""

import torch.utils.data as data
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2
from PIL import Image, ImageEnhance
import os

import torchvision.transforms as transforms
import h5py

class KneeGradingDataset(data.Dataset):
    def __init__(self, dataset, home_path, transform, stage='train'):
        self.dataset = dataset
        self.transform = transform
        self.stage = stage
        self.home_path = home_path
    def __getitem__(self, index):
        row = self.dataset.loc[index]
        fname = row['File Name']
        month = fname.split('_')[1]
        target = int(row['KLG'])
        path = os.path.join(self.home_path,month,fname)
        f = h5py.File(path)
        img = f['data'].value
        f.close()
        img = np.expand_dims(img,axis=2)
        img = np.repeat(img[:, :], 3, axis=2)
        if self.transform:
            img = self.transform(img)

        return img, target, fname

    def __len__(self):
        return self.dataset.shape[0]
