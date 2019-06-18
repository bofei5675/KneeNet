import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import gc
import itertools
from tqdm import tqdm_notebook
import torch.utils.data as data
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import Variable

from PIL import Image
from dataset import KneeGradingDataset
from augmentation import CenterCrop
from val_utils import validate_epoch
from copy import deepcopy
from sklearn.metrics import confusion_matrix, mean_squared_error, cohen_kappa_score, roc_auc_score, roc_curve, log_loss
from sklearn.preprocessing import OneHotEncoder
from augmentation import CenterCrop
import cv2

from torchvision.models import resnet34


def load_picture16bit(fname):
    img = Image.open(fname)
    # We will use 8bit
    img = np.array(img, dtype=float)
    img = np.uint8(255 * (img / 65535.))
    img = Image.fromarray(np.repeat(img[:, :, np.newaxis], 3, axis=2))

    return CenterCrop(300)(img)
def load_img(fname):
    f = h5py.File(fname)
    img = f['data']
    img = np.expand_dims(img,axis=2)
    img = np.repeat(img[:, :], 3, axis=2)
    f.close()
    return img

def weigh_maps(weights, maps):
    maps = maps.squeeze()
    weights = weights.squeeze()

    res = Variable(torch.zeros(maps.size()[-2:]).cuda(), requires_grad=False)

    for i, w in enumerate(weights):
        res += w * maps[i]

    return res

# Producing the GradCAM output using the equations provided in the article
def gradcam_resnet(fname,net):
    img = load_img(fname)
    inp = scale_tensor_transform(img).view(1, 3, 224, 224)
    net.train(False);
    net.zero_grad()

    features = nn.Sequential(net.module.conv1,
                             net.module.bn1,
                             net.module.relu,
                             net.module.maxpool,
                             net.module.layer1,
                             net.module.layer2,
                             net.module.layer3,
                             net.module.layer4)

    maps = features(Variable(inp.cuda()))
    maps_avg = F.avg_pool2d(maps, 7).view(1, 512)

    grads = []
    maps_avg.register_hook(lambda x: grads.append(x));

    out = net.module.fc(maps_avg)

    ohe = OneHotEncoder(sparse=False, n_values=5)
    index = np.argmax(out.cpu().data.numpy(), axis=1).reshape(-1, 1)
    out.backward(torch.from_numpy(ohe.fit_transform(index)).float().cuda())

    heatmap = cv2.resize(F.relu(weigh_maps(grads[0], maps)).data.cpu().numpy(), (300, 300), cv2.INTER_CUBIC)

    probs = F.softmax(out).cpu().data[0].numpy()

    return img, heatmap, probs