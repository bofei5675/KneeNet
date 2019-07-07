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
import pydicom as dicom

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


class DicomDataset(data.Dataset):
    def __init__(self, dataset, home_dir,transform, stage='train',reshape =898):
        self.dataset = dataset
        self.home_dir = home_dir
        self.transform = transform
        self.stage = stage
        self.reshape = reshape

    def __getitem__(self, index):
        row = self.dataset.loc[index].tolist()
        visit = row[1]
        data_path = row[0]
        data_path = os.path.join(self.home_dir,visit,data_path)
        file_name = os.listdir(data_path)[0]
        data_path = os.path.join(data_path,file_name)
        img, row, col, ratio_x, ratio_y = self._preprocessing(data_path)
        img = np.expand_dims(img, axis=2)
        img = np.repeat(img[:, :], 3, axis=2)
        if self.transform:
            img = self.transform(img)
        img = torch.from_numpy(img).float()
        img = img.permute(2, 0, 1)
        return img,row,col,ratio_x,ratio_y,data_path

    def _preprocessing(self,data_path):
        dicom_img = dicom.dcmread(data_path)
        img = dicom_img.pixel_array.astype(float)
        img = (np.maximum(img, 0) / img.max()) * 255.0
        row, col = img.shape
        img = cv2.resize(img, (self.reshape, self.reshape), interpolation=cv2.INTER_CUBIC)
        ratio_x = self.reshape / col
        ratio_y = self.reshape / row
        return img, row, col, ratio_x,ratio_y

    def __len__(self):
        return self.dataset.shape[0]