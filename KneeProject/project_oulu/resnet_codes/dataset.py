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

class KneeGradingDataset(data.Dataset):
    def __init__(self, dataset, transform, stage='train'):
        self.dataset = dataset
        self.transform = transform
        self.stage = stage

    def __getitem__(self, index):
        data = self.dataset.loc[index]
        fname = data['directory']
        target = data['label']

        img = Image.open(fname)
        # We will use 8bit
        img = np.array(img, dtype=float)
        img = np.uint8(255 * (img / 65535.))
        img = Image.fromarray(np.repeat(img[:, :, np.newaxis], 3, axis=2))

        img = self.transform(img)

        return img, target, fname

    def __len__(self):
        return self.dataset.shape[0]


class LimitedRandomSampler(data.sampler.Sampler):
    def __init__(self, data_source, nb, bs):
        self.data_source = data_source
        self.n_batches = nb
        self.bs = bs

    def __iter__(self):
        return iter(torch.randperm(len(self.data_source)).long()[:self.n_batches * self.bs])

    def __len__(self):
        return self.n_batches * self.bs