"""
Dataset classes and samplers for knee localization

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

class KneeDetectionDataset(data.Dataset):
    def __init__(self, dataset, transform, stage='train'):
        self.dataset = dataset
        self.transform = transform
        self.stage = stage
    def __getitem__(self, index):
        row = self.dataset.loc[index].tolist()
        fname = row[0]
        bbox = row[1:]
        target = bbox
        f = h5py.File(fname)
        img = f['data'].value
        f.close()
        img = np.expand_dims(img,axis=2)
        img = np.repeat(img[:, :], 3, axis=2)
        if self.transform:
            img = self.transform(img)
        return img, target, fname

    def __len__(self):
        return self.dataset.shape[0]
